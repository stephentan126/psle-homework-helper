"""
Photo and text submission endpoints: extract, safety-check, match, confirm, then gate.

`submit_photo()` decodes, preprocesses and transcribes a photo with the VLM;
`submit_photo_question()` accepts pre-extracted text. Both run the input safety check and
match-first, store a `student_submissions` row and return `NeedsConfirmationResponse`. The student
always confirms the raw text, never the matched corpus text (see Issue 188). After confirmation,
the gate runs on the matched corpus question if there is one, otherwise in weak mode on the raw
text. Photo attempts are always keyed by `student_submission_id`, so their origin is visible in
the audit trail.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, OperationalError

from app.api.routes import (
    _existing_hint_event_response,
    _finalize_and_persist_hint,
    _get_or_create_demo_student,
    _resolve_gate_result_and_hint,
    _stage_ctx,
)
from app.api.schemas import (
    AnyHomeworkResponse,
    ExtractionFailedResponse,
    NeedsConfirmationResponse,
)
from app.db.session import get_session
from app.models.attempt import Attempt
from app.models.hint_event import HintEvent
from app.models.question import Question
from app.models.student_submission import StudentSubmission
from app.pipeline.gate import run_weak_mode_gate_pipeline
from app.pipeline.latency_tracking import RequestLatencyTracker
from app.pipeline.photo_preprocessing import preprocess_live_photo
from app.pipeline.question_matching import MatchResult, find_best_match
from app.pipeline.skew_detection import is_photo_too_tilted
from app.pipeline.safety_response import (
    build_classifier_error_response, build_safety_response, record_safety_flag,
)
from app.pipeline.self_harm_detection import check_safety
from app.services import swap_tracker, vlm_service

logger = logging.getLogger("api.submission_routes")

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PHOTO_STORAGE_DIR = _REPO_ROOT / "data" / "student_submissions"

router = APIRouter()


class PhotoQuestionSubmission(BaseModel):
    """Request body for `submit_photo_question()`: text that has already been extracted."""
    question_text: str
    has_diagram: bool = False


class EditSubmissionRequest(BaseModel):
    """Request body for `edit_submission()`: the student's corrected transcription.

    `has_diagram` is not editable through this endpoint."""
    corrected_text: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safety_check_and_match(
    session, student_id: int, question_text: str,
) -> tuple[Optional[AnyHomeworkResponse], Optional[MatchResult]]:
    """Run the input safety check, then match-first, on submitted or edited text.

    Shared by the creation and edit paths, so editing cannot bypass the safety check. The check
    runs first, before matching or any row is touched, so flagged text never reaches the tutoring
    model.

    Returns:
        `(early_return_response, match)`. If the first item is not `None` (safety flag or
        classifier error), the caller must return it immediately without creating or updating a
        `StudentSubmission` row, and `match` is `None`. Otherwise `match` is the
        `find_best_match()` result."""
    # Narrow fail-closed handler around the classifier call: a crash returns a controlled error
    # response and is never treated as safe (see Issue 210).
    try:
        safety_result = check_safety(question_text, "input")
    except Exception:
        logger.error(
            "Safety classifier crashed on INPUT check (Issue 210) -- failing closed, never "
            "falling through to 'assume safe'. student_id=%s question_text_len=%d",
            student_id, len(question_text),
            exc_info=True,
        )
        record_safety_flag(
            session, student_id=student_id, attempt_id=None, flagged_input_or_output="input",
            category="classifier_error",
        )
        return build_classifier_error_response(), None

    if safety_result.flagged:
        record_safety_flag(
            session, student_id=student_id, attempt_id=None, flagged_input_or_output="input",
            category=safety_result.category,
        )
        return build_safety_response(safety_result.category), None

    return None, find_best_match(question_text)


def _matched_question_text_for(session, match) -> Optional[str]:
    """Return the matched corpus question's text, or `None` if there is no match.

    This is supporting context only; the student always confirms the raw text (see Issue 188)."""
    if match.matched_question_id is None:
        return None
    matched_question = session.get(Question, match.matched_question_id)
    return matched_question.question_text


def _create_submission_and_confirmation_response(
    session, student_id: int, question_text: str, has_diagram: bool, photo_path: Optional[str],
    confidence: Optional[str] = None,
) -> AnyHomeworkResponse:
    """Safety-check and match the text, store a `student_submissions` row, and ask for confirmation.

    Shared by `submit_photo_question()` and `submit_photo()` so the two entry points cannot drift.

    Args:
        confidence: Transcription confidence to report; `None` for pre-extracted text.

    Returns:
        `NeedsConfirmationResponse`, or a safety response if the text was flagged."""
    early, match = _safety_check_and_match(session, student_id, question_text)
    if early is not None:
        return early

    is_matched = match.matched_question_id is not None

    row = StudentSubmission(
        student_id=student_id,
        question_text=question_text,
        # The original transcription, set once at creation and never overwritten.
        ocr_raw_text=question_text,
        has_diagram=has_diagram,
        photo_path=photo_path,
        submitted_at=_now_iso(),
        matched_question_id=match.matched_question_id,
        match_confidence=match.match_confidence,
        match_method=match.match_method,
        # Weak-mode fields (tier_allowed, reason, detail) stay null until
        # `confirm_photo_submission()` runs the weak-mode gate.
        gate_path_used="verified_match" if is_matched else "weak_mode",
    )
    session.add(row)
    session.flush()

    return NeedsConfirmationResponse(
        question_id=match.matched_question_id,
        student_submission_id=row.id,
        match_confidence=match.match_confidence,
        extracted_text=question_text,
        matched_question_text=_matched_question_text_for(session, match),
        confidence=confidence,
    )


@router.post("/submissions/photo-question", response_model=AnyHomeworkResponse)
def submit_photo_question(submission: PhotoQuestionSubmission) -> AnyHomeworkResponse:
    """Submit pre-extracted question text for matching and confirmation.

    Returns `NeedsConfirmationResponse`, or `SafetyBlockedResponse` if the text is flagged, hence
    the wider `AnyHomeworkResponse` type."""
    with get_session() as session:
        student = _get_or_create_demo_student(session)
        return _create_submission_and_confirmation_response(
            session, student.id, submission.question_text, submission.has_diagram, None,
        )


@router.post("/submissions/photo", response_model=AnyHomeworkResponse)
def submit_photo(file: UploadFile = File(...)) -> AnyHomeworkResponse:
    """Decode, preprocess and transcribe a question photo (Qwen3-VL-8B-Instruct), then match it.

    The transcription's mean token log-probability is the reader's confidence. Below
    `vlm_service.MIN_TRUSTED_LOG_PROB` the photo is treated as unreadable rather than shown to the
    student. This is separate from match-first's `match_confidence`, which measures corpus
    similarity, not transcription quality.

    Failures return `ExtractionFailedResponse` with a machine-readable `reason`
    (`unreadable_photo`, `photo_not_straight`, `low_confidence`, `reader_unavailable`) so the
    frontend can choose between "retake the photo" and "type your question instead"."""
    raw_bytes = file.file.read()
    buffer = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        return ExtractionFailedResponse(
            message="We couldn't read that photo. Please try again with a clearer photo.",
            reason="unreadable_photo",
        )

    # Tilt check used as the retake trigger, because page-boundary detection needs the page to
    # cover at least 20% of the frame and fails on tight single-question crops (see Issue 368).
    # Run before the VLM so a tilted photo is rejected cheaply.
    too_tilted, _ = is_photo_too_tilted(image)
    if too_tilted:
        return ExtractionFailedResponse(
            message="That photo looks tilted. Please try again holding the camera more level.",
            reason="photo_not_straight",
        )

    _PHOTO_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    submission_uuid = uuid.uuid4().hex
    raw_path = _PHOTO_STORAGE_DIR / f"{submission_uuid}_raw.jpg"
    cv2.imwrite(str(raw_path), image)

    processed_image, _ = preprocess_live_photo(image)
    processed_path = _PHOTO_STORAGE_DIR / f"{submission_uuid}_processed.jpg"
    cv2.imwrite(str(processed_path), processed_image)

    # Only one GPU model may be loaded per process (Issues 366, 385). Kept outside the
    # try/except below, whose `except Exception` means reader_unavailable and must not swallow
    # this.
    try:
        swap_tracker.guard("vlm")
    except swap_tracker.SwapRestartRequired:
        swap_tracker.trigger_restart()
        raise HTTPException(
            status_code=503,
            detail="The server needs to restart before it can read another photo right now. "
                   "Please retry in about 45 seconds.",
            # 45s covers the observed restart-to-ready time of about 35 to 40 seconds.
            headers={"Retry-After": "45"},
        )

    try:
        transcription, mean_log_prob = vlm_service.transcribe_photo(str(processed_path))
    except Exception:
        # Reader load or inference failure: tell the frontend to offer typing instead of a
        # retake, since the photo itself is not the problem (see Issue 56).
        return ExtractionFailedResponse(
            message="The document reader is currently unavailable. Please type your question "
                    "instead.",
            reason="reader_unavailable",
        )

    # A low-confidence transcription never reaches the confirm step (see Issue 194).
    if mean_log_prob < vlm_service.MIN_TRUSTED_LOG_PROB:
        return ExtractionFailedResponse(
            message="We couldn't read that clearly enough to be confident. Please try again "
                    "with a clearer, better-lit photo.",
            reason="low_confidence",
        )

    has_diagram = "[DIAGRAM PRESENT]" in transcription

    with get_session() as session:
        student = _get_or_create_demo_student(session)
        # Anything that passed the threshold above is reported as "high". The cutoff is
        # calibrated on only one good and one bad example, too few to justify a middle tier.
        response = _create_submission_and_confirmation_response(
            session, student.id, transcription, has_diagram, str(raw_path), confidence="high",
        )
    # The confirm step needs the language model, and switching models in one process crashes
    # transformers on Windows (Issues 366, 385). Restart now, while the student reads the confirm
    # screen, so the confirm lands in a fresh process. The frontend retries until the server is
    # back.
    swap_tracker.trigger_restart(delay_s=1.5)
    return response


@router.post(
    "/submissions/{student_submission_id}/confirm", response_model=AnyHomeworkResponse,
)
def confirm_photo_submission(
    student_submission_id: int, request_id: str,
) -> AnyHomeworkResponse:
    """Run the gate and return the first hint once the student has confirmed the submission text.

    The matched path gates against the verified corpus question rather than the raw OCR text,
    which is more reliable. Weak mode has only the raw text and is capped at Tier 1."""
    with get_session() as session:
        # Idempotency guard, checked first. request_id is globally unique on hint_events, so the
        # same replay logic as routes.py applies.
        existing = session.scalar(select(HintEvent).where(HintEvent.request_id == request_id))
        if existing is not None:
            return _existing_hint_event_response(existing)
        # Created after the replay check so replays do not skew latency data.
        latency_tracker = RequestLatencyTracker(request_id)

        submission = session.get(StudentSubmission, student_submission_id)
        if submission is None:
            raise HTTPException(
                status_code=404,
                detail=f"No student_submission with id {student_submission_id}",
            )

        try:
            student = _get_or_create_demo_student(session)
        except (IntegrityError, OperationalError):
            session.rollback()
            student = _get_or_create_demo_student(session)

        # Both branches call the gate, which needs the language model (Issue 366).
        try:
            swap_tracker.guard("llm")
        except swap_tracker.SwapRestartRequired:
            swap_tracker.trigger_restart()
            raise HTTPException(
                status_code=503,
                detail="The server needs to restart before it can process this confirmation. "
                       "Please retry in about 45 seconds.",
                headers={"Retry-After": "45"},
            )

        is_matched = submission.matched_question_id is not None

        if is_matched:
            question = session.get(Question, submission.matched_question_id)
            # A matched photo resolves to a corpus question, so the precompute cache applies as for
            # a typed request. Each confirm starts a new attempt, so the first hint is Tier 1.
            effective_gate_result, hint, self_consistency_summary = _resolve_gate_result_and_hint(
                session, question, latency_tracker, requested_tier=1,
            )
            question_text = question.question_text
            worked_solution_text = question.worked_solution_text
            has_diagram = question.has_diagram
        else:
            # Weak mode has no corpus question and so no cache; it is timed here under the same
            # "gate" stage, tagged weak_mode so the two paths can be told apart in the logs.
            with _stage_ctx(latency_tracker, "gate", cache_hit=False, weak_mode=True):
                effective_gate_result, hint = run_weak_mode_gate_pipeline(submission.question_text)
            self_consistency_summary = None
            question_text = submission.question_text
            # No corpus answer key or diagram flag exists for an unmatched question; weak mode
            # is capped at Tier 1, so any regeneration stays at Tier 1.
            worked_solution_text = None
            has_diagram = False
            # Record the weak-mode gate outcome on the submission row for the audit trail.
            submission.tier_allowed = effective_gate_result.tier_allowed
            submission.reason = effective_gate_result.reason
            submission.detail = json.dumps(effective_gate_result.detail)
            session.add(submission)

        # Shared finalisation path, so the output safety check covers both the matched and
        # weak-mode branches. Photo attempts are keyed by student_submission_id, never question_id.
        gate_path_used = "verified_match" if is_matched else "weak_mode"
        is_unverified = not is_matched
        return _finalize_and_persist_hint(
            session, student, hint, effective_gate_result, self_consistency_summary, request_id,
            attempt_question_id=None, attempt_student_submission_id=submission.id,
            response_question_id=submission.matched_question_id if is_matched else None,
            response_student_submission_id=None if is_matched else submission.id,
            gate_path_used=gate_path_used, is_unverified=is_unverified,
            question_text=question_text, worked_solution_text=worked_solution_text,
            has_diagram=has_diagram, latency_tracker=latency_tracker,
            # Forces Tier 1 first on the matched path; a no-op in weak mode, which is capped anyway.
            requested_tier=1,
        )


@router.post(
    "/submissions/{student_submission_id}/edit", response_model=AnyHomeworkResponse,
)
def edit_submission(
    student_submission_id: int, request: EditSubmissionRequest,
) -> AnyHomeworkResponse:
    """Let the student correct a submission's text before confirming it.

    The corrected text goes through the same safety check and match-first as at creation.
    No `swap_tracker.guard()` is needed: this path uses only the safety classifier and the
    embedding model, never the LLM or VLM.

    Raises:
        HTTPException: 404 if the submission does not exist, 409 if it is already confirmed."""
    with get_session() as session:
        submission = session.get(StudentSubmission, student_submission_id)
        if submission is None:
            raise HTTPException(
                status_code=404,
                detail=f"No student_submission with id {student_submission_id}",
            )

        # An Attempt keyed to this submission means it has already been confirmed and gated, so
        # editing is locked.
        existing_attempt = session.scalar(
            select(Attempt).where(Attempt.student_submission_id == student_submission_id)
        )
        if existing_attempt is not None:
            raise HTTPException(
                status_code=409,
                detail=f"student_submission {student_submission_id} has already been confirmed "
                       f"(Attempt {existing_attempt.id} exists) -- editing is locked once "
                       f"confirmed, real error, not a silent no-op.",
            )

        early, match = _safety_check_and_match(
            session, submission.student_id, request.corrected_text,
        )
        if early is not None:
            return early

        is_matched = match.matched_question_id is not None

        # `ocr_raw_text` keeps the original; only `question_text` and the match fields change.
        submission.question_text = request.corrected_text
        submission.matched_question_id = match.matched_question_id
        submission.match_confidence = match.match_confidence
        submission.match_method = match.match_method
        submission.gate_path_used = "verified_match" if is_matched else "weak_mode"
        session.add(submission)
        session.flush()

        return NeedsConfirmationResponse(
            question_id=match.matched_question_id,
            student_submission_id=submission.id,
            match_confidence=match.match_confidence,
            extracted_text=request.corrected_text,
            matched_question_text=_matched_question_text_for(session, match),
            # Edited text has no transcription confidence to report.
            confidence=None,
        )
