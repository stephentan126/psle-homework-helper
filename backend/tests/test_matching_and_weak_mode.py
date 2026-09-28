"""
Unmocked tests for the match-first endpoint and the weak-mode gate pipeline (Issues 180, 185, 186).

Each test uses a corpus `Question` row and an embedding computed with the selected embedding model
(`Qwen/Qwen3-Embedding-0.6B`), stored as a `QuestionEmbedding` row as
`scripts/extraction/precompute_question_embeddings.py` would write it, and a FastAPI `TestClient`
POST. The weak-mode gate test runs a 2-pass Phi-4-mini self-consistency check with no external
answer key, which is what a novel photo-submitted question triggers in production.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.db.session import get_session
from app.main import app
from app.models.question import Question
from app.models.question_embedding import QuestionEmbedding
from app.models.student_submission import StudentSubmission
from app.pipeline.gate import run_weak_mode_gate_pipeline
from app.pipeline.question_matching import EMBEDDING_MODEL_ID, _get_model

_REAL_QUESTION_TEXT = (
    "Which digit in 15.89 is in the tenths place?\n(1) 1\n(2) 5\n(3) 8\n(4) 9"
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _insert_real_question_with_embedding() -> int:
    """Insert one corpus question and its computed embedding, as the precompute script would."""
    with get_session() as session:
        question = Question(
            source_paper="data/School papers/2021/P6-Maths-2021-CA1-Henry-Park.pdf",
            source_page_index=2,
            source_page_label=None,
            answer_source_file="data/School papers/2021/P6-Maths-2021-CA1-Henry-Park.pdf",
            answer_source_page_index=2,
            question_number="1",
            question_text=_REAL_QUESTION_TEXT,
            answer_value="4",
            worked_solution_text=None,
            has_diagram=False,
            ocr_confidence="high",
            school_name="Henry Park Primary School",
        )
        session.add(question)
        session.flush()
        question_id = question.id

        model = _get_model()
        embedding = model.encode([_REAL_QUESTION_TEXT], convert_to_numpy=True)[0]
        session.add(QuestionEmbedding(
            question_id=question_id, embedding_model=EMBEDDING_MODEL_ID,
            embedding_json=json.dumps(embedding.tolist()), computed_at=_now_iso(),
        ))
        return question_id


# A noisy submitted text: the Qwen3-VL-8B-Instruct transcription of this question used in the
# embedding bake-off's noisy-query check (Issue 185, true_score=0.9859 against
# `_REAL_QUESTION_TEXT`). It carries a leading question-number prefix that the stored corpus text
# does not. It is deliberately not byte-identical to `_REAL_QUESTION_TEXT`, so asserting that
# `extracted_text` equals this and `matched_question_text` equals the clean text proves the two
# fields come from different sources (Issue 188). A byte-identical fixture could not catch that.
_REAL_NOISY_SUBMITTED_TEXT = (
    "1 Which digit in 15.89 is in the tenths place?\n(1) 1\n(2) 5\n(3) 8\n(4) 9"
)


def test_submit_photo_question_matches_a_real_confident_duplicate():
    """A noisy VLM-style submission still matches above the 0.90 threshold (Issue 186).

    The response returns the raw submitted text as `extracted_text`, which the student confirms
    against their photo. The matched question's verified text appears separately as
    `matched_question_text`, as context only (Issue 188).
    """
    question_id = _insert_real_question_with_embedding()

    with TestClient(app) as client:
        response = client.post(
            "/api/submissions/photo-question",
            json={"question_text": _REAL_NOISY_SUBMITTED_TEXT, "has_diagram": False},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "needs_confirmation"
    assert body["question_id"] == question_id
    assert body["match_confidence"] is not None and body["match_confidence"] >= 0.90
    # The raw submitted text is always the confirmation target.
    assert body["extracted_text"] == _REAL_NOISY_SUBMITTED_TEXT
    # The matched corpus text differs from the submission, so it came from the matched
    # `questions` row rather than echoing the input, and is kept separate from `extracted_text`.
    assert body["matched_question_text"] == _REAL_QUESTION_TEXT
    assert body["matched_question_text"] != body["extracted_text"]
    assert body["student_submission_id"] is not None

    with get_session() as session:
        row = session.get(StudentSubmission, body["student_submission_id"])
        assert row.gate_path_used == "verified_match"
        assert row.matched_question_id == question_id
        # The stored question_text is the raw submission, not overwritten with the corpus text.
        assert row.question_text == _REAL_NOISY_SUBMITTED_TEXT


def test_submit_photo_question_falls_through_to_weak_mode_on_a_genuinely_novel_question():
    """An unrelated question does not match and falls through to weak mode.

    `question_id` stays null and the row's `gate_path_used` is 'weak_mode'. The below-threshold
    `match_confidence` is still recorded (Issue 180 item 5).
    """
    _insert_real_question_with_embedding()  # a corpus row exists, but is unrelated

    novel_text = (
        "A rectangular garden is 12 metres long and 7 metres wide. A gardener wants to put a "
        "fence around the entire garden. What is the total length of fencing needed, in metres?"
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/submissions/photo-question",
            json={"question_text": novel_text, "has_diagram": False},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "needs_confirmation"
    assert body["question_id"] is None
    assert body["extracted_text"] == novel_text  # always the raw text (Issue 188)
    assert body["matched_question_text"] is None  # no confident match -> no supporting context
    assert body["student_submission_id"] is not None
    # A near-miss score is recorded even below the threshold, never a bare null (Issue 180 item 5).
    assert body["match_confidence"] is not None

    with get_session() as session:
        row = session.get(StudentSubmission, body["student_submission_id"])
        assert row.gate_path_used == "weak_mode"
        assert row.matched_question_id is None


# --- Near-duplicate boundary checks (Issue 199). The tests above cover a clear duplicate and an
# unrelated question, but not a different question that scores close to
# MATCH_CONFIDENCE_THRESHOLD. The next test uses near-duplicates where only the numbers change.
def test_find_best_match_correctly_rejects_real_near_duplicate_questions_with_different_numbers():
    """A same-template question with different numbers scores below the threshold (Issue 186).

    This is the common "same question type, different numbers" homework case. A false match
    here would serve the wrong question's answer key.
    """
    import numpy as np

    from app.pipeline.question_matching import MATCH_CONFIDENCE_THRESHOLD, _cosine_sim, _get_model

    corpus_text = _REAL_QUESTION_TEXT  # "Which digit in 15.89 is in the tenths place?..."
    near_duplicates_different_numbers = [
        "Which digit in 27.46 is in the tenths place?\n(1) 2\n(2) 7\n(3) 4\n(4) 6",
    ]
    model = _get_model()
    corpus_emb = model.encode([corpus_text], convert_to_numpy=True)[0]
    for candidate in near_duplicates_different_numbers:
        candidate_emb = model.encode([candidate], convert_to_numpy=True)[0]
        score = float(_cosine_sim(corpus_emb, np.array([candidate_emb]))[0])
        assert score < MATCH_CONFIDENCE_THRESHOLD, (
            f"Real near-duplicate (different numbers) scored {score:.4f}, at or above "
            f"MATCH_CONFIDENCE_THRESHOLD={MATCH_CONFIDENCE_THRESHOLD} — would be silently "
            f"served as a confident match to the WRONG question. Candidate: {candidate!r}"
        )


def test_find_best_match_KNOWN_GAP_real_near_duplicate_mcq_sharing_a_stem_can_cross_threshold():
    """Records a known raw-score gap: a same-stem MCQ scores above the threshold (Issue 199(a)).

    The candidate shares the number and all four choices with the corpus question and differs
    only in 'tenths place' vs 'hundredths place'. It scored 0.9380, above the 0.90 threshold.
    The raw score is model output and cannot be fixed; the fix is the secondary check in
    `find_best_match()`, tested by
    `test_find_best_match_blocks_a_real_high_scoring_same_stem_mcq_false_positive`. This test
    anchors the raw score so that an embedding-model change is visible by itself.
    """
    from app.pipeline.question_matching import MATCH_CONFIDENCE_THRESHOLD, _cosine_sim, _get_model

    corpus_text = _REAL_QUESTION_TEXT  # tenths place, options (1)1 (2)5 (3)8 (4)9
    same_stem_different_correct_answer = (
        "Which digit in 15.89 is in the hundredths place?\n(1) 1\n(2) 5\n(3) 8\n(4) 9"
    )
    model = _get_model()
    corpus_emb = model.encode([corpus_text], convert_to_numpy=True)[0]
    candidate_emb = model.encode([same_stem_different_correct_answer], convert_to_numpy=True)[0]
    import numpy as np
    score = float(_cosine_sim(corpus_emb, np.array([candidate_emb]))[0])
    assert score >= MATCH_CONFIDENCE_THRESHOLD, (
        f"Expected the real raw embedding score to still be >= threshold but got {score:.4f} -- "
        f"if this now fails, the embedding model or its version likely changed; re-read Issue "
        f"199(a) before assuming this test itself is simply wrong."
    )


def test_find_best_match_blocks_a_real_high_scoring_same_stem_mcq_false_positive():
    """`find_best_match()` blocks the same-stem MCQ pair despite its high score (Issue 199(a)).

    Run end to end against a stored `QuestionEmbedding`, `_distinguishing_terms_conflict()` must
    see that "tenths" and "hundredths" disagree. Otherwise the wrong answer key would be served.
    """
    question_id = _insert_real_question_with_embedding()  # tenths-place corpus question

    from app.pipeline.question_matching import MATCH_CONFIDENCE_THRESHOLD, find_best_match

    same_stem_different_correct_answer = (
        "Which digit in 15.89 is in the hundredths place?\n(1) 1\n(2) 5\n(3) 8\n(4) 9"
    )
    result = find_best_match(same_stem_different_correct_answer)

    assert result.matched_question_id is None, (
        f"Real false positive NOT blocked: matched question_id={question_id} for a genuinely "
        f"different question (hundredths, not tenths) — Issue 199(a)'s fix did not hold."
    )
    assert result.match_confidence is not None and result.match_confidence >= MATCH_CONFIDENCE_THRESHOLD, (
        "The real (misleadingly high) score must still be recorded, not discarded, even though "
        "the match itself is correctly blocked (Issue 180 item 5's own discipline)."
    )
    assert result.match_method == (
        f"cosine_similarity_{EMBEDDING_MODEL_ID}_blocked_by_distinguishing_terms"
    ), f"Expected the real, distinct block reason in match_method, got {result.match_method!r}."


def test_find_best_match_blocks_a_real_corpus_geometry_false_positive_sample():
    """A corpus geometry pair with disjoint angle labels is blocked (Issue 199(a)).

    `\\angle FDE + \\angle BDF = 180°` vs `\\angle ABC + \\angle BDC = 180°` came from a 140-pair
    sample of sibling questions in `psle.db` and scored 0.9295. Blocking it shows the fix
    generalises beyond the MCQ pair above.
    """
    with get_session() as session:
        question = Question(
            source_paper="data/School papers/2021/P6-Maths-2021-SA1-ACS-Junior.pdf",
            source_page_index=1, source_page_label=None,
            answer_source_file="data/School papers/2021/P6-Maths-2021-SA1-ACS-Junior.pdf",
            answer_source_page_index=1, question_number="3",
            question_text=r"\(\angle FDE + \angle BDF = 180^{\circ}\)",
            answer_value="96", worked_solution_text=None, has_diagram=True, ocr_confidence="high",
            school_name="ACS Junior",
        )
        session.add(question)
        session.flush()
        question_id = question.id

        model = _get_model()
        embedding = model.encode(
            [r"\(\angle FDE + \angle BDF = 180^{\circ}\)"], convert_to_numpy=True,
        )[0]
        session.add(QuestionEmbedding(
            question_id=question_id, embedding_model=EMBEDDING_MODEL_ID,
            embedding_json=json.dumps(embedding.tolist()), computed_at=_now_iso(),
        ))

    from app.pipeline.question_matching import find_best_match

    result = find_best_match(r"\(\angle ABC + \angle BDC = 180^{\circ}\)")

    assert result.matched_question_id is None, (
        f"Real false positive NOT blocked: matched question_id={question_id} for a genuinely "
        f"different real geometry statement with completely disjoint angle labels."
    )
    assert "blocked_by_distinguishing_terms" in result.match_method


def test_run_weak_mode_gate_pipeline_always_caps_at_tier_1_and_never_leaks_an_answer():
    """Weak mode always caps at Tier 1 and never leaks the answer (Issue 180).

    A 2-pass Phi-4-mini self-consistency run with no answer key must give `tier_allowed == 1`
    whether or not the passes agree, and the hint must be the restate-only Tier 1 template. The
    no-leak check follows `test_hint_endpoint.py`: a computed answer is never revealed without
    the parent-PIN path.
    """
    novel_text = (
        "A rectangular garden is 12 metres long and 7 metres wide. A gardener wants to put a "
        "fence around the entire garden. What is the total length of fencing needed, in metres?"
    )
    gate_result, hint = run_weak_mode_gate_pipeline(novel_text)

    assert gate_result.tier_allowed == 1
    assert gate_result.detail.get("weak_mode") is True
    assert hint["tier"] == 1
    # The correct answer (38 m) must not appear in the Tier 1 hint. Weak mode has no verified key,
    # but the same rule applies to whatever the model computed internally.
    assert "38" not in hint["hint_text"]


# --- Issue 189: the confirm-triggered hint-generation endpoint ---

_FAST_SUM_QUESTION_TEXT = (
    r"Find the value of \(\frac{3}{4} \div 12\). Leave your answer in its simplest form."
)


def _insert_real_matched_submission() -> int:
    """Insert a matched submission for a fast, SymPy-gated sum question and return its id.

    The question is the one `test_hint_endpoint.py` uses, because the SymPy gate is deterministic
    and needs no LLM generation. The `student_submissions` row is stored as a confident match, as
    `submit_photo_question()` would leave it, so tests exercise only the confirm step.
    """
    with get_session() as session:
        question = Question(
            source_paper="data/School papers/2024/P6_Maths_2024_WA2_nanhua.pdf",
            source_page_index=5, source_page_label=None,
            answer_source_file="data/School papers/2024/P6_Maths_2024_WA2_nanhua.pdf",
            answer_source_page_index=20, question_number="19",
            question_text=_FAST_SUM_QUESTION_TEXT, answer_value="1/16",
            worked_solution_text=None, has_diagram=False, ocr_confidence="high",
            school_name="Nan Hua Primary School",
        )
        session.add(question)
        session.flush()
        question_id = question.id

        model = _get_model()
        embedding = model.encode([_FAST_SUM_QUESTION_TEXT], convert_to_numpy=True)[0]
        session.add(QuestionEmbedding(
            question_id=question_id, embedding_model=EMBEDDING_MODEL_ID,
            embedding_json=json.dumps(embedding.tolist()), computed_at=_now_iso(),
        ))

        student = _get_or_create_demo_student_for_test(session)
        submission = StudentSubmission(
            student_id=student.id,
            question_text="19 Find the value of 3/4 div 12. Simplest form pls",  # noisy, deliberately
            # not the clean corpus text (Issue 188).
            has_diagram=False, submitted_at=_now_iso(),
            matched_question_id=question_id, match_confidence=0.97,
            match_method=f"cosine_similarity_{EMBEDDING_MODEL_ID}",
            gate_path_used="verified_match",
        )
        session.add(submission)
        session.flush()
        return submission.id


def _get_or_create_demo_student_for_test(session):
    from app.api.routes import _get_or_create_demo_student
    return _get_or_create_demo_student(session)


def test_confirm_photo_submission_matched_path_uses_the_real_matched_question_for_reasoning():
    """On the matched path, the gate reasons over the matched question, not the raw text (Issue 189).

    The student confirms the raw text (Issue 188), but the gate uses the matched row's clean text
    and answer. The SymPy gate passing for 1/16 shows `answer_value` was read from that row.
    """
    submission_id = _insert_real_matched_submission()

    with TestClient(app) as client:
        response = client.post(
            f"/api/submissions/{submission_id}/confirm", params={"request_id": "confirm-test-1"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "hint"
    assert body["gate_path_used"] == "verified_match"
    assert body["is_unverified"] is False
    # With the multi-hint escalation ladder, the first hint is always Tier 1 whatever the gate
    # allows. can_escalate_further=True shows the SymPy check passed (tier_allowed >= 2).
    assert body["tier"] == 1
    assert body["can_escalate_further"] is True
    assert "1/16" not in body["hint_text"]  # the answer never leaks

    with get_session() as session:
        from app.models.attempt import Attempt
        attempt = session.get(Attempt, body["attempt_id"])
        # The audit trail is keyed by student_submission_id, never question_id, even when matched.
        assert attempt.student_submission_id == submission_id
        assert attempt.question_id is None


def test_confirm_photo_submission_weak_mode_path_caps_tier_1_and_records_real_audit_fields():
    """An unmatched submission gives `capped_tier1` with `is_unverified=True`.

    The weak-mode gate outcome must be written to the row's audit fields (Issue 180 item 3), not
    left null because the match step already wrote match_confidence and match_method.
    """
    novel_text = (
        "A rectangular garden is 12 metres long and 7 metres wide. A gardener wants to put a "
        "fence around the entire garden. What is the total length of fencing needed, in metres?"
    )
    with get_session() as session:
        student = _get_or_create_demo_student_for_test(session)
        submission = StudentSubmission(
            student_id=student.id, question_text=novel_text, has_diagram=False,
            submitted_at=_now_iso(), matched_question_id=None, match_confidence=0.41,
            match_method=f"cosine_similarity_{EMBEDDING_MODEL_ID}_below_threshold",
            gate_path_used="weak_mode",
        )
        session.add(submission)
        session.flush()
        submission_id = submission.id

    with TestClient(app) as client:
        response = client.post(
            f"/api/submissions/{submission_id}/confirm", params={"request_id": "confirm-test-2"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "capped_tier1"
    assert body["gate_path_used"] == "weak_mode"
    assert body["is_unverified"] is True
    assert body["student_submission_id"] == submission_id
    assert body["question_id"] is None
    assert "38" not in body["hint_text"]  # the correct answer never leaks

    with get_session() as session:
        row = session.get(StudentSubmission, submission_id)
        # The gate outcome is written once the gate has run, beyond the submission-time fields.
        assert row.tier_allowed == 1
        assert row.reason is not None
        assert row.detail is not None


def test_confirm_photo_submission_idempotency_guard_real():
    """A repeated `request_id` on confirm returns the cached result without re-running the gate.

    The endpoint reuses the idempotency guard in routes.py (Issue 57); this checks it here too.
    """
    submission_id = _insert_real_matched_submission()

    with TestClient(app) as client:
        first = client.post(
            f"/api/submissions/{submission_id}/confirm", params={"request_id": "confirm-test-3"},
        )
        second = client.post(
            f"/api/submissions/{submission_id}/confirm", params={"request_id": "confirm-test-3"},
        )

    assert first.status_code == 200 and second.status_code == 200
    assert first.json() == second.json()

    with get_session() as session:
        from sqlalchemy import select as sa_select
        from app.models.hint_event import HintEvent
        count = len(session.scalars(
            sa_select(HintEvent).where(HintEvent.request_id == "confirm-test-3"),
        ).all())
        assert count == 1  # one row despite two requests


def test_confirm_photo_submission_and_request_hint_share_one_real_finalization_function():
    """The typed and photo paths finalise hints through one shared function (Issue 199(b)).

    `routes.request_hint()` and both branches of `submission_routes.confirm_photo_submission()`
    must use the same function object, so the output safety check is applied in one place. If
    the two are ever forked again, this fails at import time.
    """
    from app.api import routes, submission_routes

    assert submission_routes._finalize_and_persist_hint is routes._finalize_and_persist_hint


def test_find_best_match_blocks_a_real_numeric_only_near_duplicate_false_positive():
    """A pair differing only in plain numbers is blocked by the numeric check (Issue 199(a)).

    With no place-value word, unit or capital-letter label, `_distinguishing_terms_conflict()`
    sees two empty term sets and cannot help. This common PSLE shape ("gives away ... how many
    left") scored 0.9463 for questions with answers 16 and 9. `_numeric_terms_conflict()` blocks it.
    """
    question_id = _insert_ali_marbles_question_with_embedding()

    from app.pipeline.question_matching import MATCH_CONFIDENCE_THRESHOLD, find_best_match

    result = find_best_match(
        "Ali has 24 marbles. He gives away 15. How many does he have left?",
    )
    assert result.matched_question_id is None, (
        f"Real false positive NOT blocked: matched question_id={question_id} for a genuinely "
        f"different real question (different real answer) — Issue 199(a)'s numeric fix did "
        f"not hold."
    )
    assert result.match_confidence is not None and result.match_confidence >= MATCH_CONFIDENCE_THRESHOLD, (
        "The real (misleadingly high) score must still be recorded, not discarded, even though "
        "the match itself is correctly blocked (Issue 180 item 5's own discipline)."
    )
    assert "blocked_by_numbers" in result.match_method, (
        f"Expected the real, distinct numeric-block reason in match_method, got "
        f"{result.match_method!r} (vocabulary conflict should NOT have fired here — this pair "
        f"shares no place-value/unit/label vocabulary at all)."
    )


def test_find_best_match_blocks_a_second_real_numeric_near_duplicate_from_a_different_domain():
    """A numeric-only near-duplicate about money is also blocked, so the fix generalises.

    The pair scored 0.9186, above the threshold, with different answers ($12 vs $18).
    """
    with get_session() as session:
        question = Question(
            source_paper="data/School papers/2022/P6_Maths_2022_SA2_redswastika.pdf",
            source_page_index=3, source_page_label=None,
            answer_source_file="data/School papers/2022/P6_Maths_2022_SA2_redswastika.pdf",
            answer_source_page_index=3, question_number="9",
            question_text="A shop sells notebooks at $2 each. Ravi buys 6 notebooks. How much "
                           "does he pay?",
            answer_value="$12", worked_solution_text=None, has_diagram=False, ocr_confidence="high",
            school_name="Red Swastika School",
        )
        session.add(question)
        session.flush()
        question_id = question.id

        model = _get_model()
        embedding = model.encode(
            ["A shop sells notebooks at $2 each. Ravi buys 6 notebooks. How much does he pay?"],
            convert_to_numpy=True,
        )[0]
        session.add(QuestionEmbedding(
            question_id=question_id, embedding_model=EMBEDDING_MODEL_ID,
            embedding_json=json.dumps(embedding.tolist()), computed_at=_now_iso(),
        ))

    from app.pipeline.question_matching import find_best_match

    result = find_best_match(
        "A shop sells notebooks at $2 each. Ravi buys 9 notebooks. How much does he pay?",
    )
    assert result.matched_question_id is None, (
        f"Real false positive NOT blocked (different domain): matched question_id={question_id} "
        f"for a genuinely different real question (different real answer, $12 vs $18)."
    )
    assert "blocked_by_numbers" in result.match_method


def test_find_best_match_still_matches_a_real_duplicate_despite_a_leading_question_number_prefix():
    """A true duplicate with a leading question-number prefix still matches.

    The prefix "19" is not among the corpus text's numbers, unlike the "tenths place" MCQ where the
    prefix digit is also an answer choice, so a naive numeric diff would reject it. The numeric
    check tolerates a symmetric difference of size 1 for this noise pattern (Issue 188); this
    checks that tolerance end to end.
    """
    with get_session() as session:
        question = Question(
            source_paper="data/School papers/2024/P6_Maths_2024_WA2_nanhua.pdf",
            source_page_index=5, source_page_label=None,
            answer_source_file="data/School papers/2024/P6_Maths_2024_WA2_nanhua.pdf",
            answer_source_page_index=20, question_number="19",
            question_text=_FAST_SUM_QUESTION_TEXT, answer_value="1/16",
            worked_solution_text=None, has_diagram=False, ocr_confidence="high",
            school_name="Nan Hua Primary School",
        )
        session.add(question)
        session.flush()
        question_id = question.id

        model = _get_model()
        embedding = model.encode([_FAST_SUM_QUESTION_TEXT], convert_to_numpy=True)[0]
        session.add(QuestionEmbedding(
            question_id=question_id, embedding_model=EMBEDDING_MODEL_ID,
            embedding_json=json.dumps(embedding.tolist()), computed_at=_now_iso(),
        ))

    from app.pipeline.question_matching import find_best_match

    result = find_best_match("19 Find the value of 3/4 div 12. Simplest form pls")
    assert result.matched_question_id == question_id, (
        f"Real true positive WRONGLY blocked by the numeric fix: expected match to "
        f"question_id={question_id}, got matched_question_id={result.matched_question_id!r} "
        f"(match_confidence={result.match_confidence!r}, match_method={result.match_method!r}) "
        f"-- the leading-question-number-prefix tolerance (symmetric-difference-size-1) did not "
        f"hold end to end."
    )


def _insert_ali_marbles_question_with_embedding() -> int:
    """Insert the marbles question and its embedding for the numeric-only test (Issue 199(a)).

    Its answer (16) differs from the compared question's (9), and the two share no place-value,
    unit or label vocabulary.
    """
    with get_session() as session:
        question = Question(
            source_paper="data/School papers/2023/P6_Maths_2023_WA2_AiTong.pdf",
            source_page_index=1, source_page_label=None,
            answer_source_file="data/School papers/2023/P6_Maths_2023_WA2_AiTong.pdf",
            answer_source_page_index=1, question_number="5",
            question_text="Ali has 24 marbles. He gives away 8. How many does he have left?",
            answer_value="16", worked_solution_text=None, has_diagram=False, ocr_confidence="high",
            school_name="Ai Tong School",
        )
        session.add(question)
        session.flush()
        question_id = question.id

        model = _get_model()
        embedding = model.encode(
            ["Ali has 24 marbles. He gives away 8. How many does he have left?"],
            convert_to_numpy=True,
        )[0]
        session.add(QuestionEmbedding(
            question_id=question_id, embedding_model=EMBEDDING_MODEL_ID,
            embedding_json=json.dumps(embedding.tolist()), computed_at=_now_iso(),
        ))
        return question_id
