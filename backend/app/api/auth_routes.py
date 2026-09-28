"""
Parent PIN unlock endpoint (design specification, Section 15; Issue 244).

This is the only path that reveals a final answer (`UnlockedSolutionResponse`, Section 10.3)
outside the gate's Tier 2 and Tier 3 hint escalation. It has its own router file, following the
convention of one router file per feature (as with `submission_routes.py`).

The PIN is checked per question (Issue 244): a correct PIN unlocks only the `question_id` in this
request. `parent_auth.verify_pin_attempt()` runs on every call, and the endpoint keeps no cache,
session flag or database row that would let a later request skip verification. The tests confirm
that unlocking question A does not unlock question B.

`parent_notified` is not changed here. The Section 7 alert to the linked parent on a self-harm flag
raises an open PDPA consent and disclosure question (Issue 203) that the parent account
infrastructure does not resolve. It remains in `safety_response.py`.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, OperationalError

from app.api.routes import _get_or_create_demo_student
from app.api.schemas import AnyHomeworkResponse, UnlockedSolutionResponse
from app.db.session import get_session
from app.models.question import (
    ANSWER_VERIFICATION_PENDING_FLAG,
    NON_QUESTION_EXTRACTION_FLAG,
    Question,
)
from app.pipeline.parent_auth import verify_pin_attempt

logger = logging.getLogger("api.auth_routes")

router = APIRouter()


class PinUnlockRequest(BaseModel):
    """Request body for a PIN unlock.

    The PIN is sent in the body, never as a query parameter, so it cannot appear in server access
    logs or browser history. The 4-8 digit check follows Section 10.2's "short numeric PIN" and is
    input hygiene only, not a security boundary: argon2 verifies any string safely, and a malformed
    PIN would fail verification anyway. The range covers common PIN lengths, since no specific
    length has been specified.
    """
    pin: str

    @field_validator("pin")
    @classmethod
    def _real_pin_shape(cls, value: str) -> str:
        if not (4 <= len(value) <= 8) or not value.isdigit():
            raise ValueError("PIN must be 4-8 digits.")
        return value


@router.post("/questions/{question_id}/unlock", response_model=AnyHomeworkResponse)
def unlock_solution(question_id: int, request: PinUnlockRequest) -> AnyHomeworkResponse:
    with get_session() as session:
        question = session.scalar(select(Question).where(Question.id == question_id))
        if question is None:
            raise HTTPException(status_code=404, detail=f"No question with id {question_id}")
        # Reject a question with nothing to unlock (both `answer_value` and
        # `worked_solution_text` empty, as with some weak-mode rows or rows with no matching answer
        # key). Without this, a correct PIN returned an empty `UnlockedSolutionResponse` instead
        # of a clear error.
        #
        # This differs from the guard in `request_hint()`, which rejects a missing
        # `answer_value` alone because the gate runs SymPy and self-consistency checks against it.
        # This endpoint only reveals what is stored, and `UnlockedSolutionResponse` allows either
        # field to be null. The corpus has 276 rows with a `worked_solution_text` but no
        # `answer_value`; the narrower check would have blocked every one of them.
        if not question.answer_value and not question.worked_solution_text:
            raise HTTPException(
                status_code=422,
                detail=f"Question {question_id} has no real answer_value or worked_solution_text "
                       "to unlock.",
            )
        # Same guard as request_hint() (Issue 209): a row flagged as non-question boilerplate has
        # no solution to unlock.
        if question.extraction_flag and NON_QUESTION_EXTRACTION_FLAG in question.extraction_flag:
            raise HTTPException(
                status_code=422,
                detail=f"Question {question_id} is flagged as non-question extraction "
                       "content, so there is no solution to unlock.",
            )
        # Quarantine guard (Issue 248). Unlike the hint path (Issue 247), this endpoint has no
        # verification step of its own and returns `question.answer_value` as stored, so a row
        # whose stored answer is confirmed wrong must be blocked here with a distinct 422 until
        # the answer is corrected, rather than shown as if verified.
        if question.extraction_flag and ANSWER_VERIFICATION_PENDING_FLAG in question.extraction_flag:
            logger.warning(
                "PIN unlock attempt for question_id=%s REFUSED — answer_key_verification_pending "
                "(Issue 248): this question's stored answer is confirmed wrong and "
                "not yet corrected.",
                question_id,
            )
            raise HTTPException(
                status_code=422,
                detail=f"Question {question_id}'s stored answer is not available right now — it is "
                       "pending verification. A "
                       "corrected answer is not yet available for this question.",
            )

        # Same seeding pattern as request_hint() (Issue 168); see that function for the
        # concurrency race this try/except guards against.
        try:
            student = _get_or_create_demo_student(session)
        except (IntegrityError, OperationalError):
            session.rollback()
            student = _get_or_create_demo_student(session)

        parent = student.parent_account
        result = verify_pin_attempt(parent, request.pin)
        # Always commit, on success or failure, so failed_attempts and locked_until persist across
        # requests; a rate limit held only in request memory would have no effect.
        session.commit()

        if result.outcome == "locked_out":
            logger.warning(
                "PIN unlock attempt for question_id=%s LOCKED OUT until %s (student_id=%s).",
                question_id, result.locked_until, student.id,
            )
            raise HTTPException(
                status_code=429,
                detail=f"Too many failed PIN attempts — locked until {result.locked_until} (UTC).",
            )
        if result.outcome == "wrong_pin":
            logger.info(
                "PIN unlock attempt for question_id=%s: wrong PIN (student_id=%s).",
                question_id, student.id,
            )
            raise HTTPException(status_code=401, detail="Incorrect PIN.")

        logger.info(
            "PIN unlock SUCCEEDED for question_id=%s (student_id=%s) — real solution revealed.",
            question_id, student.id,
        )
        return UnlockedSolutionResponse(
            question_id=question.id,
            answer_value=question.answer_value,
            worked_solution_text=question.worked_solution_text,
        )
