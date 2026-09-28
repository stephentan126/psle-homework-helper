"""
"Still stuck" hint escalation endpoint (design specification, Section 9; Issue 180).

It has its own router file because it is keyed by `attempt_id` and serves both typed-question
(`routes.py`) and photo (`submission_routes.py`) attempts in the same way, which is also why
`Attempt` has exactly one of `question_id` or `student_submission_id` (`app/models/attempt.py`).
It fits neither existing router, and follows the one-router-per-feature convention
(`auth_routes.py` for the PIN unlock, `submission_routes.py` for photos).

Flow: look up the `Attempt`, its most recent `HintEvent` (with the persisted `tier_allowed`,
`grounded_in` and `verified_content`) and its `current_tier`, then compute the ceiling from
`tier_allowed` with `gate.tier_ceiling()`.
  - If `current_tier >= ceiling`, no escalation is possible: reuse
    `_existing_hint_event_response()` (the same reconstruction the idempotency guard uses) with a
    reason and `can_escalate_further=False`.
  - Otherwise, rebuild enough of a `GateResult` from the persisted row to call `generate_hint()`
    for the next tier up (one step at a time, never straight to the ceiling) without re-running
    SymPy or self-consistency. Then pass it through `_finalize_and_persist_hint()`, the same
    safety check and persistence every hint goes through, with `existing_attempt` so the attempt
    row is advanced in place rather than duplicated.

Idempotent through the required `request_id`, using the same `hint_events.request_id` unique
constraint and `_existing_hint_event_response()` short-circuit as the other endpoints.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, OperationalError

from app.api.routes import (
    _existing_hint_event_response,
    _finalize_and_persist_hint,
    _get_or_create_demo_student,
)
from app.api.schemas import AnyHomeworkResponse
from app.db.session import get_session
from app.models.attempt import Attempt
from app.models.hint_event import HintEvent
from app.models.question import Question
from app.pipeline.gate import GateResult, generate_hint, tier_ceiling
from app.pipeline.latency_tracking import RequestLatencyTracker
from app.services import swap_tracker

logger = logging.getLogger("api.escalation_routes")

router = APIRouter()


def _reconstruct_gate_result_for_escalation(most_recent: HintEvent) -> GateResult:
    """Builds a synthetic `GateResult` from a persisted `HintEvent` for escalation.

    This avoids adding a bypass parameter to `generate_hint()`. Its derivation logic in `gate.py`
    already turns `{"expr_str": ...}` or `{"claimed_answers": ...}` into the right `grounded_in`
    and prompt content, so this function only supplies the same shape rather than repeating that
    logic.

    `generate_hint()` always prefers `worked_solution_text` over `gate_result.detail` when one is
    present, and nothing edits a `questions` row's `worked_solution_text` in place (Issue 198). A
    question with a worked solution therefore always produced
    `grounded_in="worked_solution_text"`. The caller can pass the current `worked_solution_text`
    unconditionally; `detail` only matters when it is null, and `grounded_in` says which of the
    two remaining shapes to rebuild.
    """
    detail: dict = {}
    if most_recent.grounded_in == "sympy_confirmed_expression":
        detail = {"expr_str": most_recent.verified_content}
    elif most_recent.grounded_in == "self_consistency_confirmed_value":
        detail = {"claimed_answers": most_recent.verified_content}
    return GateResult(
        passed=True, tier_allowed=most_recent.tier_allowed,
        reason="(reconstructed from persisted hint_events row for escalation, development log, "
               "multi-hint escalation ladder brief 2 of 2 — no real SymPy/self-consistency "
               "re-run)",
        detail=detail,
    )


@router.post("/attempts/{attempt_id}/escalate", response_model=AnyHomeworkResponse)
def escalate_hint(attempt_id: int, request_id: str) -> AnyHomeworkResponse:
    with get_session() as session:
        # Idempotency guard (Section 10.5, Issue 57), checked first as in every endpoint;
        # request_id uniqueness is global.
        existing = session.scalar(select(HintEvent).where(HintEvent.request_id == request_id))
        if existing is not None:
            return _existing_hint_event_response(existing)

        attempt = session.get(Attempt, attempt_id)
        if attempt is None:
            raise HTTPException(status_code=404, detail=f"No attempt with id {attempt_id}")

        most_recent = session.scalar(
            select(HintEvent).where(HintEvent.attempt_id == attempt_id)
            .order_by(HintEvent.id.desc()).limit(1),
        )
        if most_recent is None:
            raise HTTPException(
                status_code=422,
                detail=f"Attempt {attempt_id} has no real hint_events row yet — nothing to "
                       "escalate from.",
            )

        # A hint_events row written before tier_allowed was persisted (tier_allowed IS NULL) maps
        # to ceiling 1, so no escalation is offered for it. This fails closed.
        ceiling = tier_ceiling(most_recent.tier_allowed)
        if attempt.current_tier >= ceiling:
            # No gate or generation work runs here, so no latency tracker is created, as with the
            # idempotency short-circuit. A near-zero entry would distort escalation timing data.
            return _existing_hint_event_response(
                most_recent,
                reason_override=(
                    "(no further real escalation available at this attempt's own real gate "
                    "confidence ceiling — a parent can still unlock the full real solution via "
                    "the PIN path, Section 10.2)"
                ),
            )

        latency_tracker = RequestLatencyTracker(request_id)
        next_tier = attempt.current_tier + 1  # one step up, never straight to the ceiling

        # Same swap guard as every other endpoint that calls generate_hint() (Issue 366): every
        # tier calls the LLM.
        try:
            swap_tracker.guard("llm")
        except swap_tracker.SwapRestartRequired:
            swap_tracker.trigger_restart()
            raise HTTPException(
                status_code=503,
                detail="The server needs to restart before it can process this escalation. "
                       "Please retry in about 45 seconds.",
                headers={"Retry-After": "45"},
            )

        try:
            student = _get_or_create_demo_student(session)
        except (IntegrityError, OperationalError):
            session.rollback()
            student = _get_or_create_demo_student(session)

        reconstructed_gate_result = _reconstruct_gate_result_for_escalation(most_recent)

        # Use `student_submission_id` to decide how to find the matched question, as
        # `_existing_hint_event_response()` does (Issue 190). A photo-confirmed attempt always has
        # `attempt.question_id is None` (Issue 189); its match is on
        # `student_submission.matched_question_id`. Checking `attempt.question_id` classed every
        # photo-matched escalation as weak mode, giving `response_question_id=None` with a
        # matched question's `tier_allowed` (for example 4). That built a
        # `HintResponse(question_id=None, ...)` which failed Pydantic validation with a 500.
        if attempt.student_submission_id is not None:
            real_matched_question_id = attempt.student_submission.matched_question_id
            is_matched = real_matched_question_id is not None
        else:
            real_matched_question_id = attempt.question_id
            is_matched = real_matched_question_id is not None
        if is_matched:
            question = session.get(Question, real_matched_question_id)
            question_text = question.question_text
            worked_solution_text = question.worked_solution_text
            has_diagram = question.has_diagram
            tier3_question_id: Optional[int] = question.id
        else:
            # Weak mode's gate is capped at tier_allowed <= 1, so the ceiling check above returns
            # before this point for every weak-mode attempt. This branch is still handled in case
            # that invariant changes.
            submission = attempt.student_submission
            question_text = submission.question_text
            worked_solution_text = None
            has_diagram = False
            tier3_question_id = None

        want_tier3 = next_tier == 3
        hint = generate_hint(
            question_text, reconstructed_gate_result, worked_solution_text,
            has_diagram=has_diagram, requested_tier=next_tier, want_tier3=want_tier3,
            question_id=tier3_question_id,
        )

        gate_path_used = "verified_match" if is_matched else "weak_mode"
        is_unverified = not is_matched
        # self_consistency_summary is None: no new self-consistency check ran, because the
        # reconstructed GateResult reuses the original outcome. The original hint_events row still
        # holds that check's summary.
        return _finalize_and_persist_hint(
            session, student, hint, reconstructed_gate_result, None, request_id,
            attempt_question_id=attempt.question_id,
            attempt_student_submission_id=attempt.student_submission_id,
            # Not attempt.question_id, which is always None for a photo-confirmed attempt.
            response_question_id=real_matched_question_id if is_matched else None,
            response_student_submission_id=None if is_matched else attempt.student_submission_id,
            gate_path_used=gate_path_used, is_unverified=is_unverified,
            question_text=question_text, worked_solution_text=worked_solution_text,
            has_diagram=has_diagram, latency_tracker=latency_tracker,
            existing_attempt=attempt,
            requested_tier=next_tier, want_tier3=want_tier3, tier3_question_id=tier3_question_id,
        )
