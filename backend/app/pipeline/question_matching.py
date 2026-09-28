"""
Match-first question lookup for photo submissions.

Embeds the extracted text of a submitted question and compares it against the precomputed corpus
embedding cache (`question_embeddings`, built offline by
`scripts/extraction/precompute_question_embeddings.py`) to find a confident match. The corpus is
never embedded at request time: at about 1.8s per text on CPU, that would take hours per request.

Matching policy:
- A single embedding model constant (`EMBEDDING_MODEL_ID`) is used for both the cache and live
  queries, so the two can never drift apart.
- `MATCH_CONFIDENCE_THRESHOLD` is set at 0.90, towards the upper end of the observed gap between
  the worst known-different pair (0.7152) and the worst known-same pair (0.9793), rather than its
  midpoint. A false positive (serving the wrong question's answer key) is worse than a false
  negative (falling through to the labelled weak-mode path) (see Issue 186).
- Two secondary checks run on top of the threshold and are OR-combined, since near-identical but
  different questions can still score above 0.90 (see Issue 199):
  1. `_distinguishing_terms_conflict()`: place-value words, units and ALL-CAPS geometry labels
     (for example "tenths" vs "hundredths", or angle FDE vs angle ABC).
  2. `_numeric_terms_conflict()`: the numbers embedded in the text (for example "gives away 8"
     vs "gives away 15", which scored 0.9463).

Known limitation: numbers are compared as literal digit strings, not parsed values, so "05" vs
"5" or "585 640" vs "585640" count as different. This is not normalised, because an occasional
extra rejection is the safe direction to err in.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy import or_

from app.models.question import (
    ANSWER_VERIFICATION_PENDING_FLAG,
    NON_QUESTION_EXTRACTION_FLAG,
    Question,
)
from app.models.question_embedding import QuestionEmbedding
from app.db.session import get_session

EMBEDDING_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"  # embedding model selection (see Issue 185)
MATCH_CONFIDENCE_THRESHOLD = 0.90  # calibrated threshold, do not change without recalibrating
                                    # (see Issue 186)

# Place-value words and units that the embedding model under-weights when they are the only
# difference between two near-identical questions. Bare single-letter unit abbreviations ("m",
# "g", "l", "h") are excluded because they collide with algebra variable names in the corpus;
# only unambiguous multi-letter forms and the symbols $, % and ¢ are matched.
_PLACE_VALUE_TERMS = {
    "one": "place:ones", "ones": "place:ones",
    "ten": "place:tens", "tens": "place:tens",
    "hundred": "place:hundreds", "hundreds": "place:hundreds",
    "thousand": "place:thousands", "thousands": "place:thousands",
    "tenth": "place:tenths", "tenths": "place:tenths",
    "hundredth": "place:hundredths", "hundredths": "place:hundredths",
    "thousandth": "place:thousandths", "thousandths": "place:thousandths",
}
_UNIT_TERMS = {
    "cm": "unit:cm", "centimetre": "unit:cm", "centimetres": "unit:cm",
    "centimeter": "unit:cm", "centimeters": "unit:cm",
    "km": "unit:km", "kilometre": "unit:km", "kilometres": "unit:km",
    "kilometer": "unit:km", "kilometers": "unit:km",
    "mm": "unit:mm", "millimetre": "unit:mm", "millimetres": "unit:mm",
    "kg": "unit:kg", "kilogram": "unit:kg", "kilograms": "unit:kg",
    "ml": "unit:ml", "millilitre": "unit:ml", "millilitres": "unit:ml",
    "litre": "unit:l", "litres": "unit:l", "liter": "unit:l", "liters": "unit:l",
    "hr": "unit:hour", "hrs": "unit:hour", "hour": "unit:hour", "hours": "unit:hour",
    "min": "unit:minute", "mins": "unit:minute", "minute": "unit:minute", "minutes": "unit:minute",
    "sec": "unit:second", "secs": "unit:second", "second": "unit:second", "seconds": "unit:second",
    "percent": "unit:percent", "percentage": "unit:percent",
    "cent": "unit:cent", "cents": "unit:cent",
    "dollar": "unit:dollar", "dollars": "unit:dollar",
    "perimeter": "measure:perimeter", "area": "measure:area", "volume": "measure:volume",
}
# Named geometry labels: PSLE questions use short ALL-CAPS labels for points, angles and shapes
# (e.g. "ABCD is a rhombus", "\angle FDE"), and these are often the only content that
# distinguishes two near-identical geometry questions. The pattern is case-sensitive and matches
# only fully uppercase runs, so initial-capital words such as "Which" or "Find" are not labels.
_NAMED_LABEL_RE = re.compile(r"\b[A-Z]{1,6}\b")


def _extract_distinguishing_terms(text: str) -> set[str]:
    """Extract place-value words, units (word and symbol forms) and ALL-CAPS labels from a text.

    Returns an empty set when the text contains none of these.
    """
    terms: set[str] = set()
    for word in re.findall(r"[A-Za-z]+", text):
        lower_word = word.lower()
        if lower_word in _PLACE_VALUE_TERMS:
            terms.add(_PLACE_VALUE_TERMS[lower_word])
        elif lower_word in _UNIT_TERMS:
            terms.add(_UNIT_TERMS[lower_word])
    if "$" in text:
        terms.add("unit:dollar")
    if "%" in text:
        terms.add("unit:percent")
    if "¢" in text:
        terms.add("unit:cent")
    for label in _NAMED_LABEL_RE.findall(text):
        terms.add(f"label:{label}")
    return terms


def _distinguishing_terms_conflict(query_text: str, candidate_text: str) -> bool:
    """Return True when query and candidate disagree on place values, units or named labels.

    Runs only after a candidate clears `MATCH_CONFIDENCE_THRESHOLD`, as an extra check rather than
    a replacement for it. Two empty term sets are not a conflict, so the embedding score decides.
    """
    return _extract_distinguishing_terms(query_text) != _extract_distinguishing_terms(candidate_text)


# Numeric signal: questions that differ only in their numbers extract identical (often empty) term
# sets above, so the vocabulary check gives no protection. For example, "Ali has 24 marbles. He
# gives away 8" vs "... gives away 15" scored 0.9463 despite different answers. A fraction
# (`\frac{a}{b}` or `a/b`) is split into two digit tokens, since the check compares which numbers
# are present, not the notation used.
_NUMBER_RE = re.compile(r"\d+\.\d+|\d+")


def _extract_numbers(text: str) -> set[str]:
    """Return every digit sequence (integers and decimals) in a text as a set of strings.

    Tokens are literal strings, not parsed values, so "05" vs "5" or "585 640" vs "585640" are
    treated as different. This is left unnormalised on purpose: an occasional extra rejection is
    the safe direction to err in.
    """
    return set(_NUMBER_RE.findall(text))


def _numeric_terms_conflict(query_text: str, candidate_text: str) -> bool:
    """Return True when the two texts' numbers differ by two or more tokens.

    VLM transcriptions often carry a leading question number that the corpus text lacks
    ('19 Find the value of...' vs 'Find the value of...'), which adds one token and is tolerated
    (see Issue 188). A changed value, such as 'gives away 8' vs 'gives away 15', removes one
    token and adds another, giving a symmetric difference of two, so `>= 2` separates noise from
    a different question. The caller OR-combines this with `_distinguishing_terms_conflict()`.
    """
    query_numbers = _extract_numbers(query_text)
    candidate_numbers = _extract_numbers(candidate_text)
    return len(query_numbers.symmetric_difference(candidate_numbers)) >= 2


_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    """Return the lazily loaded embedding model, kept resident for the life of the process.

    The model is small and runs on CPU, so it does not compete with the one-GPU-model-at-a-time
    budget and does not need to be reloaded per request.
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_ID, device="cpu")
    return _model


def _cosine_sim(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = query / np.linalg.norm(query)
    matrix_norm = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix_norm @ query_norm


@dataclass
class MatchResult:
    """Outcome of a match attempt, recorded on success or failure.

    The best-scoring candidate's score is kept even when it falls short of the threshold.
    """
    matched_question_id: Optional[int]  # None if best score < MATCH_CONFIDENCE_THRESHOLD, or the
                                          # embedding cache is empty
    match_confidence: Optional[float]   # best score found, even on a non-match
    match_method: str


# Minimum similarity for a Tier 3 "related worked example". Separate from
# MATCH_CONFIDENCE_THRESHOLD, which answers "is this the same question" and is far too strict for
# "is this a different but related question worth showing as a parallel example". This value is an
# uncalibrated judgement call, not a measured threshold: low enough that a close sibling question
# (which the secondary checks would block as a same-question match) still qualifies. Revisit once
# live Tier 3 traffic provides data.
RELATED_EXAMPLE_MIN_SIMILARITY = 0.5


@dataclass
class RelatedWorkedExample:
    """A related worked example for Tier 3 hints, excluding the original grounding question.

    Carries `worked_solution_text` itself, not just an id, so that `gate.py` can stay free of
    database access.
    """
    question_id: Optional[int]
    match_confidence: Optional[float]
    worked_solution_text: Optional[str]
    match_method: str


def find_related_worked_example(
    question_id: int, exclude_question_ids: "Optional[set[int]]" = None,
) -> RelatedWorkedExample:
    """Find the most similar other question that has a worked solution, for Tier 3 hints.

    Reuses the same model, embedding cache and exclusion filters as `find_best_match()`, with
    these differences:
    - The query is the existing cached embedding for `question_id`, not a fresh encode. If none is
      cached (e.g. a new row not yet precomputed), a no-match result is returned.
    - Candidates must have a non-empty `worked_solution_text`, and `question_id` plus any
      `exclude_question_ids` are excluded.
    - The secondary conflict checks are not applied: a close sibling question is exactly what a
      good parallel example should be.
    - No threshold is applied here. The caller compares `match_confidence` against
      `RELATED_EXAMPLE_MIN_SIMILARITY`.
    """
    exclude_ids = set(exclude_question_ids or ())
    exclude_ids.add(question_id)
    with get_session() as session:
        own_embedding_row = session.query(QuestionEmbedding).filter(
            QuestionEmbedding.question_id == question_id,
            QuestionEmbedding.embedding_model == EMBEDDING_MODEL_ID,
        ).first()
        if own_embedding_row is None:
            return RelatedWorkedExample(
                question_id=None, match_confidence=None, worked_solution_text=None,
                match_method=f"no_embedding_cached_for_question_{question_id}",
            )
        query_emb = np.array(json.loads(own_embedding_row.embedding_json), dtype=np.float32)

        rows = session.query(QuestionEmbedding, Question.worked_solution_text).join(
            Question, QuestionEmbedding.question_id == Question.id,
        ).filter(
            QuestionEmbedding.embedding_model == EMBEDDING_MODEL_ID,
            Question.superseded_by.is_(None),
            Question.worked_solution_text.is_not(None),
            Question.worked_solution_text != "",
            or_(
                Question.extraction_flag.is_(None),
                Question.extraction_flag.notlike(f"%{NON_QUESTION_EXTRACTION_FLAG}%")
                & Question.extraction_flag.notlike(f"%{ANSWER_VERIFICATION_PENDING_FLAG}%"),
            ),
        ).all()
        candidates = [
            (qe.question_id, qe.embedding_json, worked_text)
            for qe, worked_text in rows if qe.question_id not in exclude_ids
        ]
        if not candidates:
            return RelatedWorkedExample(
                question_id=None, match_confidence=None, worked_solution_text=None,
                match_method="no_real_candidate_worked_example_available",
            )
        candidate_ids = [c[0] for c in candidates]
        candidate_texts = [c[2] for c in candidates]
        matrix = np.array([json.loads(c[1]) for c in candidates], dtype=np.float32)

    sims = _cosine_sim(query_emb, matrix)
    best_idx = int(np.argmax(sims))
    return RelatedWorkedExample(
        question_id=candidate_ids[best_idx],
        match_confidence=float(sims[best_idx]),
        worked_solution_text=candidate_texts[best_idx],
        match_method=f"related_worked_example_cosine_similarity_{EMBEDDING_MODEL_ID}",
    )


def find_best_match(query_text: str) -> MatchResult:
    """Embed `query_text` and return the best-matching corpus question.

    Only embeddings for the current `EMBEDDING_MODEL_ID` are compared; rows from an older model are
    treated as absent. The best score is always returned, even below the threshold.

    A candidate that clears `MATCH_CONFIDENCE_THRESHOLD` must also pass
    `_distinguishing_terms_conflict()` and `_numeric_terms_conflict()`. If either finds a
    conflict, no match is returned and `match_method` names the reason, so a blocked match is
    distinguishable from an ordinary below-threshold miss.
    """
    with get_session() as session:
        # Candidate exclusions:
        # - Superseded (dedup-loser) rows. The dedup pass never deletes rows or their embeddings,
        #   it only sets `superseded_by`, so they must be filtered here.
        # - Rows flagged as non-question boilerplate, such as exam cover pages (see Issue 209).
        # - Rows flagged with a known-wrong `answer_value` awaiting correction (see Issue 248).
        # The flag filter is NULL-safe: most rows have a NULL extraction_flag, and `flag != x`
        # evaluates to NULL for them, which would drop them all. `NOT LIKE` is used rather than
        # `!=` because flags can be combined in one "; "-separated string.
        rows = session.query(QuestionEmbedding).join(
            Question, QuestionEmbedding.question_id == Question.id,
        ).filter(
            QuestionEmbedding.embedding_model == EMBEDDING_MODEL_ID,
            Question.superseded_by.is_(None),
            or_(
                Question.extraction_flag.is_(None),
                Question.extraction_flag.notlike(f"%{NON_QUESTION_EXTRACTION_FLAG}%")
                & Question.extraction_flag.notlike(f"%{ANSWER_VERIFICATION_PENDING_FLAG}%"),
            ),
        ).all()
        # Empty cache: the offline precompute has not run for this model, so report no match.
        if not rows:
            return MatchResult(
                matched_question_id=None, match_confidence=None,
                match_method=f"no_embeddings_cached_for_{EMBEDDING_MODEL_ID}",
            )
        question_ids = [r.question_id for r in rows]
        matrix = np.array([json.loads(r.embedding_json) for r in rows], dtype=np.float32)

    model = _get_model()
    query_emb = model.encode([query_text], convert_to_numpy=True, show_progress_bar=False)[0]
    sims = _cosine_sim(query_emb, matrix)
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])
    best_question_id = question_ids[best_idx]

    if best_score >= MATCH_CONFIDENCE_THRESHOLD:
        # Read `question_text` inside the session block: after it closes, attribute access raises
        # `DetachedInstanceError` because the session expires the instance's attributes.
        with get_session() as session:
            candidate_question = session.get(Question, best_question_id)
            candidate_question_text = candidate_question.question_text
        # A high embedding score is not trusted alone: either a vocabulary or a numeric conflict
        # blocks the match. `match_method` names the specific reason so each
        # `student_submissions` row shows exactly what blocked it.
        term_conflict = _distinguishing_terms_conflict(query_text, candidate_question_text)
        numeric_conflict = _numeric_terms_conflict(query_text, candidate_question_text)
        if term_conflict or numeric_conflict:
            if term_conflict and numeric_conflict:
                block_reason = "distinguishing_terms_and_numbers"
            elif term_conflict:
                block_reason = "distinguishing_terms"
            else:
                block_reason = "numbers"
            # Keep the (misleadingly high) score for the record even though the match is blocked.
            return MatchResult(
                matched_question_id=None, match_confidence=best_score,
                match_method=f"cosine_similarity_{EMBEDDING_MODEL_ID}_blocked_by_{block_reason}",
            )
        return MatchResult(
            matched_question_id=best_question_id, match_confidence=best_score,
            match_method=f"cosine_similarity_{EMBEDDING_MODEL_ID}",
        )
    # Near miss: no match, but the best score is still recorded as evidence of what was found.
    return MatchResult(
        matched_question_id=None, match_confidence=best_score,
        match_method=f"cosine_similarity_{EMBEDDING_MODEL_ID}_below_threshold",
    )
