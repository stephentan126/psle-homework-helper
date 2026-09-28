"""
Corpus-wide near-duplicate detection for the extracted question bank (design specification,
Section 15, step 6; Issue 231).

Works over the offline-precomputed embedding cache (`question_embeddings`, model
`Qwen/Qwen3-Embedding-0.6B`, Issue 185).

Data safety: this module never deletes a `Question` row and never touches `OCRExtractionRecord`.
Its only write, and only when `run_dedup_pass(apply=True)`, is setting `Question.superseded_by` on
each non-survivor to point at its cluster's survivor (Issues 156-157). A test
(`test_dedup.py::test_apply_never_deletes_or_touches_ocr_records_the_only_mutation_is_superseded_by`)
checks that this source contains no `session.delete` or `.delete(` call, and that row counts and
every `OCRExtractionRecord` field are unchanged by an `apply=True` run.

Cost: a Python loop over every pair would be about 14.7 million calls for n=5,425. Instead:
1. `compute_full_similarity_matrix()` computes every cosine similarity with a single BLAS-backed
   matrix product (5425x1024 by 1024x5425, about 60 billion FLOPs, a few seconds on a CPU).
2. `find_candidate_pairs()` uses a vectorised numpy filter to keep only pairs scoring at least
   `REVIEW_FLOOR`.
The per-pair Python work (secondary checks, tie-break scoring, clustering) runs only on that short
candidate list. An approximate-neighbour index (FAISS, LSH) is not needed at this scale, and would
add complexity and a risk of missed pairs.

Thresholds (see Issue 231 for the validation method and figures; do not change either without
updating that entry):
- `REVIEW_FLOOR = 0.90` reuses `question_matching.MATCH_CONFIDENCE_THRESHOLD` (Issue 186), which
  was validated on the same corpus and model.
- `AUTO_MARK_THRESHOLD` is a separate, higher threshold for automatic marking. Dedup changes
  stored data, which is a higher risk than a single match-first decision, so it was calibrated on
  a separate sample of known duplicates and known different near-siblings.

Secondary checks: a pair above `AUTO_MARK_THRESHOLD` is auto-marked only if none of three checks
fires. Two are reused unchanged from `question_matching` (`_distinguishing_terms_conflict()` and
`_numeric_terms_conflict()`, Issue 199), so their validation and known limits apply here too. The
third, `_numeric_multiset_conflict()`, was added for dedup (see its docstring). A pair that fails
any check goes to the review queue rather than being discarded.

Re-runs: `load_corpus_embedding_matrix()` skips rows whose `superseded_by` is already set, so a
marked row is never reconsidered. A survivor remains a candidate, so a new duplicate added later
can join its cluster. If the new question scores higher than the existing survivor, the old
survivor will be marked as superseded for the first time. This is intended (the best record is
always kept), but a newly superseded former survivor is worth checking.

`question_matching.find_best_match()` excludes superseded rows from its candidate pool.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.models.question import NON_QUESTION_EXTRACTION_FLAG, Question
from app.models.question_embedding import QuestionEmbedding
from app.pipeline.question_matching import (
    EMBEDDING_MODEL_ID,
    _distinguishing_terms_conflict,
    _numeric_terms_conflict,
)

# Thresholds: see the module docstring and Issue 231. Do not change without updating both.
REVIEW_FLOOR = 0.90  # same as question_matching.MATCH_CONFIDENCE_THRESHOLD (Issue 186)
AUTO_MARK_THRESHOLD = 0.9994  # Calibrated on all 5,425 corpus embeddings (Issue 231):
                               # known duplicates (99 exact-duplicate pairs) min 0.999798;
                               # known different (2,241 sibling and easy-negative pairs) max
                               # 0.998185; gap 0.001612. Set towards the upper side of the gap,
                               # as in Issue 186, and above the 0.999314 recommended by
                               # scripts/extraction/calibrate_dedup_threshold.py. Re-run that
                               # script before changing this value.

# Ranking of `OCRExtractionRecord.agreement_status` values (Issues 98-101, 220-229); higher means
# more trustworthy. `both_failed` (neither reader produced usable content) ranks below every
# disagreement, since it has the least usable content behind it.
_AGREEMENT_RANK: dict[str, int] = {
    "agree": 3, "disagree": 2, "single_reader_only": 1, "both_failed": 0,
}
_NO_OCR_RECORD_RANK = -1  # a row with no OCRExtractionRecord; should not occur for a fully
                          # processed row, but ranked lowest in case it does.


# Third secondary check, specific to dedup (Issue 231). Calibration found a recurring MCQ shape: the
# same table template with the same numbers rearranged across cells. For example, question_id 741
# vs 742 (`P6-Maths-2021-SA2-ACS-Junior.pdf`, page 4, a "Muffins sold" MCQ family) scores 0.998185,
# only 0.0016 below the lowest known-duplicate score. Both existing checks miss it: the row labels
# ("A" to "D") are the same on both sides, and the set of numbers (60, 80, 100) is identical, so a
# set-based comparison cannot see the rearrangement.
_NUMBER_RE = re.compile(r"\d+\.\d+|\d+")


def _extract_number_multiset(text: str) -> Counter:
    """Returns the number tokens in `text` as a multiset.

    Uses the same tokenisation as `question_matching._extract_numbers()`, but keeps counts. Two
    texts with the same distinct values used a different number of times look identical as sets
    but differ as multisets.
    """
    return Counter(_NUMBER_RE.findall(text))


def _numeric_multiset_conflict(query_text: str, candidate_text: str) -> bool:
    """Returns True if any number appears a different number of times in the two texts.

    Stricter than `_numeric_terms_conflict()`, which needs a difference of at least two tokens to
    tolerate one stray token of transcription noise in a live query (Issue 188). Dedup compares
    two clean corpus texts, and a false auto-mark changes stored data, so any difference blocks.
    For the Muffins case above, Counter({'100': 2, '80': 1, '60': 1}) and
    Counter({'80': 2, '60': 1, '100': 1}) differ and are flagged; identical texts never are.
    """
    return _extract_number_multiset(query_text) != _extract_number_multiset(candidate_text)


# Page-margin and administrative boilerplate to exclude (Issue 231). `NON_QUESTION_EXTRACTION_FLAG`
# covers a different extraction bug (Issue 209) and misses these. Without this filter, the first
# preview run auto-marked 27 of its 125 clusters (155 of 272 superseded rows, 57%) because they
# shared text such as "(Do on to the next page)", "Do not write in this space.", "Scanned with
# CamScanner" or "Parent's signature:". These are text duplicates but not duplicate questions.
#
# A prefix match is used, as in `evaluation/model_selection/embedding/build_gold_pairs.py`
# (Issue 185) and `scripts/extraction/calibrate_dedup_threshold.py`. A minimum-length cutoff is not
# used, because some short questions ("Find the value of \(68.2 \div 4\)\nAns:") are true
# duplicates across papers and must still be deduplicated.
_BOILERPLATE_PREFIXES = (
    "all diagrams in this paper",
    "use a dark blue or black ballpoint pen",
    "for each question, four options are given",
    "(go on to the next page)",
    "(do on to the next page)",
    "use the information below to answer questions",
    "answer all questions",
    "write your answers",
    # Further phrases found in the corpus-wide preview run.
    "do not write in this space",
    "do not write in this column",
    "do not turn over this page",
    "do not turn over this paper",
    "follow all instructions carefully",
    "please do not write in the margin",
    "do not use correction fluid",
    "do not use highlighters",
    "scanned with camscanner",
    "parent's signature",
)


def _is_boilerplate(text: str) -> bool:
    lowered = text.strip().lower()
    return any(lowered.startswith(p) for p in _BOILERPLATE_PREFIXES)


def load_corpus_embedding_matrix(session: Session) -> "tuple[list[int], np.ndarray]":
    """Loads the float32 embedding matrix for the current model.

    Excludes rows flagged with `NON_QUESTION_EXTRACTION_FLAG` (the same NULL-safe filter as
    `find_best_match()`), rows matching `_is_boilerplate()`, and rows already superseded by an
    earlier run.

    Returns:
        The question IDs and a matrix with one embedding per row, in the same order.
    """
    rows = session.query(QuestionEmbedding, Question.question_text).join(
        Question, QuestionEmbedding.question_id == Question.id,
    ).filter(
        QuestionEmbedding.embedding_model == EMBEDDING_MODEL_ID,
        Question.superseded_by.is_(None),
        or_(
            Question.extraction_flag.is_(None),
            Question.extraction_flag.notlike(f"%{NON_QUESTION_EXTRACTION_FLAG}%"),
        ),
    ).all()
    rows = [(r, text) for r, text in rows if not _is_boilerplate(text)]
    question_ids = [r.question_id for r, _ in rows]
    rows = [r for r, _ in rows]
    if not question_ids:
        return [], np.empty((0, 0), dtype=np.float32)
    matrix = np.array([json.loads(r.embedding_json) for r in rows], dtype=np.float32)
    return question_ids, matrix


def compute_full_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    """Returns the pairwise cosine similarity matrix, computed with one matrix product."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid dividing by zero for an all-zero embedding
    normalized = matrix / norms
    return (normalized @ normalized.T).astype(np.float32)


@dataclass
class CandidatePair:
    question_id_a: int
    question_id_b: int
    similarity: float


def find_candidate_pairs(
    question_ids: "list[int]", sim_matrix: np.ndarray, floor: float,
) -> "list[CandidatePair]":
    """Returns every pair in the upper triangle scoring at least `floor`.

    The diagonal is excluded, since a question is not a duplicate of itself. Filtering is vectorised,
    so there is no Python loop over all pairs.
    """
    n = len(question_ids)
    if n < 2:
        return []
    iu = np.triu_indices(n, k=1)
    scores = sim_matrix[iu]
    mask = scores >= floor
    idx_i = iu[0][mask]
    idx_j = iu[1][mask]
    filtered_scores = scores[mask]
    return [
        CandidatePair(question_ids[int(i)], question_ids[int(j)], float(s))
        for i, j, s in zip(idx_i.tolist(), idx_j.tolist(), filtered_scores.tolist())
    ]


# =================================================================================================
# Tie-break scoring (Issue 231). A lexicographic tuple sorted in descending order, so each tier
# outranks every tier after it. Neither a weighted sum nor insertion order is used.
# =================================================================================================

def _agreement_rank_for(question: Question) -> int:
    """Returns the best reader-agreement rank among the question's `ocr_records`.

    There is normally at most one record per question (Issue 102); `max()` handles more.
    """
    if not question.ocr_records:
        return _NO_OCR_RECORD_RANK
    return max(
        _AGREEMENT_RANK.get(r.agreement_status, _NO_OCR_RECORD_RANK) for r in question.ocr_records
    )


def _has_diagram_matches_cluster_majority(
    question: Question, cluster_questions: "list[Question]",
) -> bool:
    """Returns True if the question's `has_diagram` matches the majority of its cluster.

    There is no ground truth for `has_diagram` at scoring time, so the independent extractions in
    a cluster are compared with each other, as the OCR reconciliation does for the two readers
    (Issues 98-101, 220-229). A record that disagrees with the majority is the more likely
    mis-detection. An even split gives no signal.
    """
    true_count = sum(1 for q in cluster_questions if q.has_diagram)
    false_count = len(cluster_questions) - true_count
    if true_count == false_count:
        return False
    majority_value = true_count > false_count
    return question.has_diagram == majority_value


def score_question_for_survivorship(
    question: Question, cluster_questions: "list[Question]",
) -> tuple:
    """Returns a sort key ranking a question as its cluster's survivor (higher is better).

    Tiers, in order: has a worked solution; reader agreement; `has_diagram` matches the cluster
    majority; no extraction flag; has an answer value; school name verified; verification status
    is "verified". The final `-question.id` is not a quality signal. It only breaks exact ties, so
    that a re-run always picks the same survivor regardless of query order.
    """
    return (
        1 if (question.worked_solution_text and question.worked_solution_text.strip()) else 0,
        _agreement_rank_for(question),
        1 if _has_diagram_matches_cluster_majority(question, cluster_questions) else 0,
        1 if question.extraction_flag is None else 0,
        1 if (question.answer_value and question.answer_value.strip()) else 0,
        1 if question.school_name_verified else 0,
        1 if question.verification_status == "verified" else 0,
        -question.id,
    )


# =================================================================================================
# Clustering: union-find over edges that clear AUTO_MARK_THRESHOLD and every secondary check.
# =================================================================================================

class _UnionFind:
    def __init__(self, ids: "list[int]") -> None:
        self._parent = {i: i for i in ids}

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb

    def components(self) -> "list[set[int]]":
        groups: dict[int, set[int]] = {}
        for node in self._parent:
            groups.setdefault(self.find(node), set()).add(node)
        return list(groups.values())


def build_clusters(edges: "list[CandidatePair]") -> "list[set[int]]":
    """Groups auto-mark edges into connected components of two or more questions.

    If A-B and B-C both clear the threshold, {A, B, C} form one cluster even if A-C does not,
    since B's similarity to both indicates the same content. One survivor is then chosen for the
    whole cluster, rather than by merging pairs.
    """
    if not edges:
        return []
    ids = sorted({qid for pair in edges for qid in (pair.question_id_a, pair.question_id_b)})
    uf = _UnionFind(ids)
    for pair in edges:
        uf.union(pair.question_id_a, pair.question_id_b)
    return [c for c in uf.components() if len(c) >= 2]


# =================================================================================================
# Review queue: individual pairs for a human to check, with no automatic action (same shape as the
# Issue 220 review queue).
# =================================================================================================

@dataclass
class ReviewQueueEntry:
    question_id_a: int
    question_id_b: int
    similarity: float
    reason: str  # "below_auto_threshold", or "blocked_by_" plus the failed checks joined by
                 # "_and_" (for example "blocked_by_distinguishing_terms_and_numbers").


@dataclass
class AutoMarkResult:
    survivor_id: int
    superseded_ids: "list[int]"
    cluster_size: int


@dataclass
class DedupRunResult:
    auto_marks: "list[AutoMarkResult]"
    review_queue: "list[ReviewQueueEntry]"
    total_candidates_considered: int


def run_dedup_pass(session: Session, apply: bool = False) -> DedupRunResult:
    """Runs the full dedup pass.

    Args:
        session: Database session.
        apply: If False (the default), compute and return the result without writing anything.
            If True, also set `Question.superseded_by` on every non-survivor and commit. Neither
            mode deletes rows or changes OCR records.

    Returns:
        The auto-mark decisions, the review queue and the number of candidate pairs considered.
    """
    question_ids, matrix = load_corpus_embedding_matrix(session)
    if len(question_ids) < 2:
        return DedupRunResult(auto_marks=[], review_queue=[], total_candidates_considered=0)

    sim_matrix = compute_full_similarity_matrix(matrix)
    candidates = find_candidate_pairs(question_ids, sim_matrix, REVIEW_FLOOR)

    # Load the whole table once rather than use `.in_(question_ids)`: an IN list of corpus size
    # risks SQLite's bound-parameter limit, and `question_ids` covers most of the corpus anyway.
    questions_by_id: dict[int, Question] = {q.id: q for q in session.query(Question).all()}

    auto_edges: "list[CandidatePair]" = []
    review_queue: "list[ReviewQueueEntry]" = []

    for pair in candidates:
        qa = questions_by_id[pair.question_id_a]
        qb = questions_by_id[pair.question_id_b]
        if pair.similarity >= AUTO_MARK_THRESHOLD:
            term_conflict = _distinguishing_terms_conflict(qa.question_text, qb.question_text)
            numeric_conflict = _numeric_terms_conflict(qa.question_text, qb.question_text)
            multiset_conflict = _numeric_multiset_conflict(qa.question_text, qb.question_text)
            if not term_conflict and not numeric_conflict and not multiset_conflict:
                auto_edges.append(pair)
                continue
            reasons = []
            if term_conflict:
                reasons.append("distinguishing_terms")
            if numeric_conflict:
                reasons.append("numbers")
            if multiset_conflict:
                reasons.append("number_multiset")
            reason = "blocked_by_" + "_and_".join(reasons)
            review_queue.append(
                ReviewQueueEntry(pair.question_id_a, pair.question_id_b, pair.similarity, reason),
            )
        else:
            review_queue.append(
                ReviewQueueEntry(
                    pair.question_id_a, pair.question_id_b, pair.similarity, "below_auto_threshold",
                ),
            )

    clusters = build_clusters(auto_edges)
    auto_marks: "list[AutoMarkResult]" = []
    for cluster in clusters:
        cluster_questions = [questions_by_id[qid] for qid in cluster]
        scored = sorted(
            cluster_questions,
            key=lambda q: score_question_for_survivorship(q, cluster_questions),
            reverse=True,
        )
        survivor = scored[0]
        non_survivors = scored[1:]
        auto_marks.append(AutoMarkResult(
            survivor_id=survivor.id,
            superseded_ids=[q.id for q in non_survivors],
            cluster_size=len(cluster_questions),
        ))
        if apply:
            for q in non_survivors:
                q.superseded_by = survivor.id  # the only write this module makes

    if apply:
        session.commit()

    return DedupRunResult(
        auto_marks=auto_marks, review_queue=review_queue,
        total_candidates_considered=len(candidates),
    )
