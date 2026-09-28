"""
Tests for the confirm-and-edit endpoint, `POST /submissions/{id}/edit` (Issue 369).

Backend and API only; the frontend is Phase 8. The tests run end to end through `TestClient`, as
in `test_safety_wiring.py` and `test_matching_and_weak_mode.py`. Where a match or no-match outcome
is asserted, corpus `Question` and `QuestionEmbedding` rows are created with an embedding from the
selected model (`Qwen/Qwen3-Embedding-0.6B`). `check_safety` is monkeypatched only where a specific
output, such as a crash, cannot be produced reliably.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.api import submission_routes as submission_routes_module
from app.db.session import get_session
from app.main import app
from app.models.attempt import Attempt
from app.models.question import Question
from app.models.question_embedding import QuestionEmbedding
from app.models.safety_flag import SafetyFlag
from app.models.student_submission import StudentSubmission
from app.pipeline.question_matching import EMBEDDING_MODEL_ID, MATCH_CONFIDENCE_THRESHOLD, _get_model
from app.pipeline.safety_response import SAMARITANS_OF_SINGAPORE_RESOURCES
from app.services import swap_tracker

_REAL_SELF_HARM_TEXT = (
    "I don't really want to be here anymore, I've been thinking about how to end it all the "
    "last few days. Anyway sorry, can you help me with this fraction question, 3/4 divided by 1/2?"
)

# The corpus question validated in the embedding bake-off (test_matching_and_weak_mode.py),
# reused because it is a known, measured true-positive pair.
_REAL_QUESTION_TEXT = (
    "Which digit in 15.89 is in the tenths place?\n(1) 1\n(2) 5\n(3) 8\n(4) 9"
)
_REAL_NOISY_MATCHING_TEXT = (
    "1 Which digit in 15.89 is in the tenths place?\n(1) 1\n(2) 5\n(3) 8\n(4) 9"
)
_REAL_NOVEL_UNRELATED_TEXT = (
    "A rectangular garden is 12 metres long and 7 metres wide. A gardener wants to put a fence "
    "around the entire garden. What is the total length of fencing needed, in metres?"
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_or_create_demo_student_for_test(session):
    from app.api.routes import _get_or_create_demo_student
    return _get_or_create_demo_student(session)


def _insert_real_question_with_embedding(question_number: str = "1") -> int:
    with get_session() as session:
        question = Question(
            source_paper="data/School papers/2021/P6-Maths-2021-CA1-Henry-Park.pdf",
            source_page_index=2, source_page_label=None,
            answer_source_file="data/School papers/2021/P6-Maths-2021-CA1-Henry-Park.pdf",
            answer_source_page_index=2, question_number=question_number,
            question_text=_REAL_QUESTION_TEXT, answer_value="4", worked_solution_text=None,
            has_diagram=False, ocr_confidence="high", school_name="Henry Park Primary School",
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


def _insert_submission(
    *, question_text: str, ocr_raw_text: str | None = None, matched_question_id=None,
    match_confidence=None, match_method=None, gate_path_used="weak_mode",
) -> int:
    """Insert a submission row directly, as in `test_matching_and_weak_mode.py`.

    This isolates the endpoint from match-first and VLM behaviour covered elsewhere.
    `ocr_raw_text` defaults to `question_text`, which is what the creation path would set.
    """
    with get_session() as session:
        student = _get_or_create_demo_student_for_test(session)
        submission = StudentSubmission(
            student_id=student.id, question_text=question_text,
            ocr_raw_text=ocr_raw_text if ocr_raw_text is not None else question_text,
            has_diagram=False, submitted_at=_now_iso(),
            matched_question_id=matched_question_id, match_confidence=match_confidence,
            match_method=match_method, gate_path_used=gate_path_used,
        )
        session.add(submission)
        session.flush()
        return submission.id


def _insert_locking_attempt(student_submission_id: int) -> int:
    """Insert an `Attempt` keyed to `student_submission_id`, which locks the submission.

    As the `confirm_photo_submission()` docstring states, 'every photo-triggered Attempt is keyed
    by student_submission_id, never question_id, matched or not'.
    """
    with get_session() as session:
        submission = session.get(StudentSubmission, student_submission_id)
        attempt = Attempt(
            student_id=submission.student_id, question_id=None,
            student_submission_id=student_submission_id, current_tier=1, status="active",
            started_at=_now_iso(), last_activity_at=_now_iso(),
        )
        session.add(attempt)
        session.flush()
        return attempt.id


# =================================================================================== TEST 1 ===
def test_edit_before_any_attempt_exists_succeeds_and_returns_updated_confirmation():
    submission_id = _insert_submission(question_text=_REAL_NOVEL_UNRELATED_TEXT)
    corrected = _REAL_NOVEL_UNRELATED_TEXT.replace("12 metres", "15 metres")

    with TestClient(app) as client:
        response = client.post(
            f"/api/submissions/{submission_id}/edit", json={"corrected_text": corrected},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "needs_confirmation"
    assert body["extracted_text"] == corrected
    assert body["student_submission_id"] == submission_id

    with get_session() as session:
        row = session.get(StudentSubmission, submission_id)
        assert row.question_text == corrected


# =================================================================================== TEST 2 ===
def test_edit_after_an_attempt_exists_is_refused_with_a_real_error_and_submission_unchanged():
    submission_id = _insert_submission(question_text=_REAL_NOVEL_UNRELATED_TEXT)
    attempt_id = _insert_locking_attempt(submission_id)

    with TestClient(app) as client:
        response = client.post(
            f"/api/submissions/{submission_id}/edit",
            json={"corrected_text": "a completely different corrected text"},
        )

    # A clear error, not a 200 that silently ignores the edit.
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert str(submission_id) in detail
    assert str(attempt_id) in detail

    with get_session() as session:
        row = session.get(StudentSubmission, submission_id)
        assert row.question_text == _REAL_NOVEL_UNRELATED_TEXT  # unchanged


# =================================================================================== TEST 3 ===
def test_ocr_raw_text_is_set_once_at_creation_and_never_changes_across_an_edit():
    original_text = "Wat is 3/4 + 1/2 ?"  # a deliberately misspelt raw transcription

    with TestClient(app) as client:
        create_response = client.post(
            "/api/submissions/photo-question",
            json={"question_text": original_text, "has_diagram": False},
        )
        assert create_response.status_code == 200
        submission_id = create_response.json()["student_submission_id"]

        with get_session() as session:
            row = session.get(StudentSubmission, submission_id)
            assert row.ocr_raw_text == original_text  # set once, at creation

        corrected_text = "What is 3/4 + 1/2?"
        edit_response = client.post(
            f"/api/submissions/{submission_id}/edit", json={"corrected_text": corrected_text},
        )
        assert edit_response.status_code == 200

    with get_session() as session:
        row = session.get(StudentSubmission, submission_id)
        # The permanent audit-trail column is untouched by the edit.
        assert row.ocr_raw_text == original_text
        # `question_text` is the one that changes: "the current, possibly-edited text".
        assert row.question_text == corrected_text


# =================================================================================== TEST 4 ===
def test_edit_flagged_unsafe_corrected_text_routes_through_the_same_safety_response_path():
    submission_id = _insert_submission(question_text=_REAL_NOVEL_UNRELATED_TEXT)

    with TestClient(app) as client:
        response = client.post(
            f"/api/submissions/{submission_id}/edit",
            json={"corrected_text": _REAL_SELF_HARM_TEXT},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "safety_blocked"
    assert body["is_self_harm_path"] is True
    assert body["resources"] == SAMARITANS_OF_SINGAPORE_RESOURCES

    with get_session() as session:
        # Editing must not bypass the safety check, so the row keeps its text from before the
        # edit attempt.
        row = session.get(StudentSubmission, submission_id)
        assert row.question_text == _REAL_NOVEL_UNRELATED_TEXT

        flag = session.scalars(
            select(SafetyFlag).where(SafetyFlag.flagged_input_or_output == "input"),
        ).all()
        assert len(flag) >= 1


# =================================================================================== TEST 5 ===
def test_edit_classifier_crash_fails_closed_same_as_the_creation_path(monkeypatch, caplog):
    submission_id = _insert_submission(question_text=_REAL_NOVEL_UNRELATED_TEXT)

    def _raise(*args, **kwargs):
        raise RuntimeError("simulated real classifier crash (Issue 369 edit-path test)")

    monkeypatch.setattr(submission_routes_module, "check_safety", _raise)

    with caplog.at_level(logging.ERROR, logger="api.submission_routes"):
        with TestClient(app) as client:
            response = client.post(
                f"/api/submissions/{submission_id}/edit",
                json={"corrected_text": "What is 2 + 2?"},
            )

    assert response.status_code == 200
    body = response.json()
    # Same fail-closed response as the creation path (Issue 210), never a raw 500.
    assert body["state"] == "safety_blocked"
    assert body["is_self_harm_path"] is True

    with get_session() as session:
        row = session.get(StudentSubmission, submission_id)
        assert row.question_text == _REAL_NOVEL_UNRELATED_TEXT  # unchanged, nothing was written

        flag = session.scalars(
            select(SafetyFlag).where(SafetyFlag.category == "classifier_error"),
        ).all()
        assert len(flag) >= 1
        assert flag[-1].flagged_input_or_output == "input"


# =================================================================================== TEST 6a ==
def test_edit_rematch_from_unmatched_to_matched_updates_all_the_real_match_fields():
    question_id = _insert_real_question_with_embedding()
    submission_id = _insert_submission(
        question_text=_REAL_NOVEL_UNRELATED_TEXT, matched_question_id=None,
        match_confidence=0.2, match_method="cosine_similarity_test_below_threshold",
        gate_path_used="weak_mode",
    )

    with TestClient(app) as client:
        response = client.post(
            f"/api/submissions/{submission_id}/edit",
            json={"corrected_text": _REAL_NOISY_MATCHING_TEXT},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["question_id"] == question_id
    assert body["match_confidence"] is not None and body["match_confidence"] >= MATCH_CONFIDENCE_THRESHOLD
    assert body["matched_question_text"] == _REAL_QUESTION_TEXT

    with get_session() as session:
        row = session.get(StudentSubmission, submission_id)
        assert row.matched_question_id == question_id
        assert row.gate_path_used == "verified_match"
        assert row.match_confidence >= MATCH_CONFIDENCE_THRESHOLD


# =================================================================================== TEST 6b ==
def test_edit_rematch_from_matched_to_unmatched_updates_all_the_real_match_fields():
    question_id = _insert_real_question_with_embedding(question_number="2")
    submission_id = _insert_submission(
        question_text=_REAL_NOISY_MATCHING_TEXT, matched_question_id=question_id,
        match_confidence=0.97, match_method=f"cosine_similarity_{EMBEDDING_MODEL_ID}",
        gate_path_used="verified_match",
    )

    with TestClient(app) as client:
        response = client.post(
            f"/api/submissions/{submission_id}/edit",
            json={"corrected_text": _REAL_NOVEL_UNRELATED_TEXT},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["question_id"] is None
    assert body["matched_question_text"] is None

    with get_session() as session:
        row = session.get(StudentSubmission, submission_id)
        assert row.matched_question_id is None
        assert row.gate_path_used == "weak_mode"
        # The near miss is still recorded rather than dropped (Issue 180 item 5).
        assert row.match_confidence is not None


# =================================================================================== TEST 7 ===
def test_edit_endpoint_does_not_depend_on_swap_tracker_guard():
    """The edit endpoint does not call the swap_tracker guard (Issue 369).

    Neither `check_safety()` nor `find_best_match()` calls `get_llm()` or `get_vlm()`, so the
    endpoint never touches the swap-tracked models and needs no guard. `swap_tracker.guard()` is
    monkeypatched to always raise `SwapRestartRequired`, as if a swap were pending. A call to it
    would turn the response into a 503.
    """
    submission_id = _insert_submission(question_text=_REAL_NOVEL_UNRELATED_TEXT)

    def _always_requires_restart(*args, **kwargs):
        raise swap_tracker.SwapRestartRequired("simulated pending swap (Issue 369 test)")

    import app.services.swap_tracker as swap_tracker_module
    original_guard = swap_tracker_module.guard
    swap_tracker_module.guard = _always_requires_restart
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/submissions/{submission_id}/edit",
                json={"corrected_text": "What is 5 + 5?"},
            )
    finally:
        swap_tracker_module.guard = original_guard

    # A 503 would mean the endpoint calls the guard; 200 confirms it does not.
    assert response.status_code == 200
    assert response.json()["state"] == "needs_confirmation"
