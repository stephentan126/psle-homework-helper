"""
Typed-question hint endpoint and the shared hint finalisation path.

Each request first checks for a version-matching precomputed gate result, otherwise runs the live
gate pipeline (classify, gate-check, method alignment, hint generation). The hint is screened by
the output safety check, persisted as `attempts`/`hint_events` rows, and returned as `hint` or
`capped_tier1`. Requests are de-duplicated by `request_id` (a UNIQUE column on `hint_events`).
Parent/student signup is not yet built, so a clearly marked demo student is used (see Issue 244).
"""
from __future__ import annotations

import json
import logging
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from app.api.schemas import AnyHomeworkResponse, CappedTier1Response, HintResponse
from app.db.session import get_session
from app.models.attempt import Attempt
from app.models.hint_event import HintEvent
from app.models.parent_account import ParentAccount
from app.models.precomputed_gate_result import PrecomputedGateResult
from app.models.question import NON_QUESTION_EXTRACTION_FLAG, Question
from app.models.student import Student
from app.pipeline.gate import (
    GATE_VERSION, GateResult, compute_gate_content_fingerprint, generate_hint,
    run_full_gate_pipeline, tier_ceiling,
)
from app.pipeline.answer_leak import guard_hint, tidy_hint_text
from app.pipeline.latency_tracking import RequestLatencyTracker
from app.pipeline.parent_auth import hash_pin
from app.pipeline.safety_response import (
    build_classifier_error_response, build_safety_response, record_safety_flag,
)
from app.pipeline.self_harm_detection import check_safety
from app.services import swap_tracker
from app.services.llm_service import LLM_ADAPTER_PATH, LLM_MODEL_PATH

logger = logging.getLogger("api.routes")

router = APIRouter()

_DEMO_STUDENT_DISPLAY_NAME = "Demo Student (real per-family signup not yet built)"

# Demo-only PIN, not a production value. It is argon2-hashed below so the demo account goes
# through the same verification path (`parent_auth.verify_pin_attempt()`) as a real parent account.
_DEMO_PARENT_PIN = "0000"

# A harassment flag on a generated hint is an output-quality problem, not the student's fault, so
# the hint is silently regenerated instead of blocked (see Issue 207). Other categories
# (self_harm, dangerous_content, sexually_explicit) are content concerns and are blocked at once,
# since a retry has no principled reason to behave differently.
_MAX_HARASSMENT_RETRY_ATTEMPTS = 2  # + the original attempt = 3 total generation attempts

# Deliberately generic: never exposes classifier internals (categories, scores, policy text) to
# the LLM, only a plain, actionable steer.
_HARASSMENT_RETRY_STEERING = (
    "IMPORTANT: your previous attempt at this hint came across as demeaning, mocking, or "
    "dismissive toward the student. Rewrite it to be warm, encouraging, and respectful. Never "
    "criticize the student's intelligence or ability, never use sarcasm, and never suggest the "
    "student is incapable of understanding the material."
)


def _get_or_create_demo_student(session: Session) -> Student:
    existing = session.scalar(
        select(Student).where(Student.display_name == _DEMO_STUDENT_DISPLAY_NAME),
    )
    if existing is not None:
        return existing
    parent = ParentAccount(pin_hash=hash_pin(_DEMO_PARENT_PIN), failed_attempts=0)
    session.add(parent)
    session.flush()
    student = Student(display_name=_DEMO_STUDENT_DISPLAY_NAME, parent_account_id=parent.id)
    session.add(student)
    session.flush()
    return student


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _can_escalate_further(tier_allowed: Optional[int], current_tier: int) -> bool:
    """Return True if a higher tier than `current_tier` is still available.

    Shared by every response builder and the escalation endpoint so they cannot disagree.
    `tier_ceiling()` in `gate.py` is the single source of the tier_allowed-to-ceiling mapping."""
    return current_tier < tier_ceiling(tier_allowed)


def _stage_ctx(latency_tracker: Optional[RequestLatencyTracker], stage_name: str, **detail):
    """Return the tracker's stage context manager, or a no-op `nullcontext()` without a tracker.

    Lets call sites use `with _stage_ctx(...)` unconditionally; a `None` tracker never changes
    which code runs (see Issue 232)."""
    if latency_tracker is None:
        return nullcontext()
    return latency_tracker.stage(stage_name, **detail)


def _resolve_gate_result_and_hint(
    session: Session, question: Question,
    latency_tracker: Optional[RequestLatencyTracker] = None,
    requested_tier: Optional[int] = None,
) -> tuple[GateResult, dict, Optional[str]]:
    """Resolve the gate result and hint from the precompute cache, or live if there is no valid row.

    Shared by the typed-question endpoint and the photo confirm endpoint (`submission_routes.py`)
    so both use identical logic (see Issue 189).

    Args:
        latency_tracker: Optional, observability only. Records a `"gate"` stage tagged with
            `cache_hit` so cached and live requests can be told apart in the logs.
        requested_tier: `None` serves the cached hint text directly. When set, the cached gate
            result is still reused but the hint text for that tier is generated live, since the
            cache stores only one tier's text.

    Returns:
        A `(gate_result, hint, self_consistency_summary)` tuple."""
    # The content fingerprint is recomputed from the current question row, so a question edited
    # after caching invalidates its cached result (see Issue 178). Any mismatch below means
    # "treat as absent" and recompute live (fail closed).
    current_fingerprint = compute_gate_content_fingerprint(
        question.question_text, question.answer_value,
        question.has_diagram, question.worked_solution_text,
    )
    precomputed = session.scalar(
        select(PrecomputedGateResult).where(
            PrecomputedGateResult.question_id == question.id,
            PrecomputedGateResult.is_active.is_(True),
            PrecomputedGateResult.gate_version == GATE_VERSION,
            PrecomputedGateResult.llm_model_path == LLM_MODEL_PATH,
            # A row computed under a different adapter is treated as absent. The adapter is
            # compared directly because a GATE_VERSION bump alone proved unreliable for this
            # (see Issue 369).
            PrecomputedGateResult.adapter_path == LLM_ADAPTER_PATH,
            PrecomputedGateResult.content_fingerprint == current_fingerprint,
        ),
    )
    if precomputed is not None:
        with _stage_ctx(latency_tracker, "gate", cache_hit=True):
            effective_gate_result = GateResult(
                passed=precomputed.gate_passed, tier_allowed=precomputed.tier_allowed,
                reason=precomputed.gate_reason, detail=json.loads(precomputed.gate_detail_json),
            )
            if requested_tier is None:
                # Serve the precomputed hint text directly, with no live generation call.
                hint = {"tier": precomputed.hint_tier, "hint_text": precomputed.hint_text}
            else:
                hint = generate_hint(
                    question.question_text, effective_gate_result, question.worked_solution_text,
                    has_diagram=question.has_diagram, requested_tier=requested_tier,
                    question_id=question.id,
                )
            self_consistency_summary = (
                f"(served from precomputed_gate_results id={precomputed.id}, "
                f"computed_at={precomputed.computed_at}, question_type="
                f"{precomputed.question_type})"
            )
        return effective_gate_result, hint, self_consistency_summary

    with _stage_ctx(latency_tracker, "gate", cache_hit=False):
        return run_full_gate_pipeline(
            question.question_text, question.answer_value,
            has_diagram=question.has_diagram, worked_solution_text=question.worked_solution_text,
            requested_tier=requested_tier, question_id=question.id,
        )


def _known_answer_values(
    session: Session, question_id: Optional[int], gate_result: GateResult,
) -> list[str]:
    """Every answer the student must not be shown: the stored answer key, plus any answer the
    model itself produced during the gate (weak mode has no key, only these)."""
    values: list[str] = []
    if question_id is not None:
        answer = session.scalar(select(Question.answer_value).where(Question.id == question_id))
        if answer:
            values.append(str(answer))
    detail = gate_result.detail or {}
    for key in ("real_value", "llm_value"):
        if detail.get(key):
            values.append(str(detail[key]))
    for claimed in detail.get("claimed_answers") or []:
        if claimed:
            values.append(str(claimed))
    return values


def _guard_against_answer_leak(
    session: Session, hint: dict, gate_result: GateResult, *, question_text: str,
    question_id: Optional[int], worked_solution_text: Optional[str], has_diagram: bool,
    requested_tier: Optional[int], want_tier3: bool, tier3_question_id: Optional[int],
) -> dict:
    """Never show the student the final answer to their own question.

    The end-to-end dry run found a live Tier 3 hint that solved the student's own question despite
    the prompt (Issue 387). Every hint is therefore checked with `answer_leak.check_leak()`,
    regenerated once with extra steering if it leaks, and replaced with a fixed safe prompt if it
    still does."""
    answer_values = _known_answer_values(session, question_id, gate_result)

    def regenerate(steering: str) -> dict:
        return generate_hint(
            question_text, gate_result, worked_solution_text, extra_steering=steering,
            has_diagram=has_diagram, requested_tier=requested_tier or hint.get("tier"),
            want_tier3=want_tier3, question_id=tier3_question_id,
        )

    guarded = guard_hint(hint, answer_values, question_text, regenerate)
    guarded["hint_text"] = tidy_hint_text(guarded.get("hint_text", ""))
    if guarded["leak_guard"] != "clean":
        logger.warning(
            "answer-leak guard %s a tier %s hint (question_id=%s)",
            guarded["leak_guard"], hint.get("tier"), question_id,
        )
    return guarded


def _finalize_and_persist_hint(
    session: Session,
    student: Student,
    hint: dict,
    effective_gate_result: GateResult,
    self_consistency_summary: Optional[str],
    request_id: str,
    *,
    attempt_question_id: Optional[int],
    attempt_student_submission_id: Optional[int],
    response_question_id: Optional[int],
    response_student_submission_id: Optional[int],
    gate_path_used: str,
    is_unverified: bool,
    question_text: str,
    worked_solution_text: Optional[str] = None,
    has_diagram: bool = False,
    latency_tracker: Optional[RequestLatencyTracker] = None,
    existing_attempt: Optional[Attempt] = None,
    requested_tier: Optional[int] = None,
    want_tier3: bool = False,
    tier3_question_id: Optional[int] = None,
) -> AnyHomeworkResponse:
    """Call `_finalize_and_persist_hint_impl()` and always finish the latency tracker.

    The `finally` guarantees `latency_tracker.finish()` runs exactly once on every exit path
    (classifier error, safety block, race replay or normal hint) without repeating it at each
    return. Arguments, return value and exceptions pass through unchanged; see the impl for the
    parameter contract."""
    hint = _guard_against_answer_leak(
        session, hint, effective_gate_result, question_text=question_text,
        question_id=response_question_id or attempt_question_id,
        worked_solution_text=worked_solution_text, has_diagram=has_diagram,
        requested_tier=requested_tier, want_tier3=want_tier3, tier3_question_id=tier3_question_id,
    )
    try:
        return _finalize_and_persist_hint_impl(
            session, student, hint, effective_gate_result, self_consistency_summary, request_id,
            attempt_question_id=attempt_question_id,
            attempt_student_submission_id=attempt_student_submission_id,
            response_question_id=response_question_id,
            response_student_submission_id=response_student_submission_id,
            gate_path_used=gate_path_used, is_unverified=is_unverified,
            question_text=question_text, worked_solution_text=worked_solution_text,
            has_diagram=has_diagram, latency_tracker=latency_tracker,
            existing_attempt=existing_attempt, requested_tier=requested_tier,
            want_tier3=want_tier3, tier3_question_id=tier3_question_id,
        )
    finally:
        if latency_tracker is not None:
            latency_tracker.finish()


def _finalize_and_persist_hint_impl(
    session: Session,
    student: Student,
    hint: dict,
    effective_gate_result: GateResult,
    self_consistency_summary: Optional[str],
    request_id: str,
    *,
    attempt_question_id: Optional[int],
    attempt_student_submission_id: Optional[int],
    response_question_id: Optional[int],
    response_student_submission_id: Optional[int],
    gate_path_used: str,
    is_unverified: bool,
    question_text: str,
    worked_solution_text: Optional[str] = None,
    has_diagram: bool = False,
    latency_tracker: Optional[RequestLatencyTracker] = None,
    existing_attempt: Optional[Attempt] = None,
    requested_tier: Optional[int] = None,
    want_tier3: bool = False,
    tier3_question_id: Optional[int] = None,
) -> AnyHomeworkResponse:
    """Safety-check, persist and return a generated hint. Every generated hint passes through here.

    This is the single place the output safety check runs, for the typed-question, photo
    (matched and weak-mode) and escalation paths, so no path can skip it (see Issue 199).

    Args:
        attempt_question_id, attempt_student_submission_id: Exactly one must be set, mirroring the
            CHECK constraint on `Attempt`. Photo attempts are always keyed by submission id.
            Ignored when `existing_attempt` is given.
        response_question_id, response_student_submission_id, gate_path_used, is_unverified:
            Caller-supplied values copied into the response.
        question_text, worked_solution_text, has_diagram: Inputs for regenerating the hint on a
            harassment flag. `worked_solution_text` is `None` in weak mode.
        existing_attempt: When set (escalation), that attempt's tier is advanced in place instead
            of creating a new `Attempt`. A new `HintEvent` row is always written.
        requested_tier, want_tier3, tier3_question_id: Carried into any regeneration so a retry
            stays at the tier originally requested and never silently degrades.
        latency_tracker: Optional, observability only. Records `"safety_check"` and
            `"hint_regeneration"` stages per attempt; finished by the wrapper above.

    Returns:
        A `CappedTier1Response`, `HintResponse`, or a safety/classifier-error response."""
    # Output safety check: the final hint text is screened by ShieldGemma-2B and the self-harm
    # lexicon (OR-combined). A flag returns a category-appropriate response and the hint is
    # never persisted or shown. Measured recall is 9/10, so this reduces but does not eliminate
    # the risk of a miss.
    # The classifier call has its own narrow fail-closed handler, so a crash returns a controlled
    # error response rather than being assumed safe (see Issue 210). A harassment flag is
    # regenerated up to _MAX_HARASSMENT_RETRY_ATTEMPTS times; any other category, or harassment
    # still flagged on the final attempt, is blocked immediately.
    total_attempts = _MAX_HARASSMENT_RETRY_ATTEMPTS + 1
    for attempt in range(1, total_attempts + 1):
        try:
            with _stage_ctx(latency_tracker, "safety_check", attempt=attempt):
                safety_result = check_safety(hint["hint_text"], "output")
        except Exception:
            logger.error(
                "Safety classifier crashed on OUTPUT check (Issue 210), attempt %d/%d -- "
                "failing closed, never falling through to 'assume safe'. question_id=%s "
                "student_submission_id=%s student_id=%s request_id=%s",
                attempt, total_attempts, response_question_id, response_student_submission_id,
                student.id, request_id,
                exc_info=True,
            )
            record_safety_flag(
                session, student_id=student.id, attempt_id=None, flagged_input_or_output="output",
                category="classifier_error",
            )
            return build_classifier_error_response()

        if not safety_result.flagged:
            break  # success -- `hint` (original or regenerated) is safe, fall through to persist

        is_harassment = safety_result.category == "harassment"
        is_final_attempt = attempt == total_attempts
        if not is_harassment or is_final_attempt:
            # Immediate block: any non-harassment category, or harassment still flagged after
            # all retries.
            record_safety_flag(
                session, student_id=student.id, attempt_id=None, flagged_input_or_output="output",
                category=safety_result.category,
            )
            return build_safety_response(safety_result.category)

        # A distinct category for intermediate retries, so the audit trail can tell a retried
        # hint apart from one that was blocked outright.
        logger.info(
            "Harassment-flagged generated hint on attempt %d/%d -- regenerating with steering "
            "(Issue 207). question_id=%s student_submission_id=%s student_id=%s "
            "request_id=%s",
            attempt, total_attempts, response_question_id, response_student_submission_id,
            student.id, request_id,
        )
        record_safety_flag(
            session, student_id=student.id, attempt_id=None, flagged_input_or_output="output",
            category=f"harassment_retry_attempt_{attempt}",
        )
        with _stage_ctx(latency_tracker, "hint_regeneration", attempt=attempt):
            # Regenerate at the tier originally requested, not the default auto-highest tier.
            hint = generate_hint(
                question_text, effective_gate_result, worked_solution_text,
                extra_steering=_HARASSMENT_RETRY_STEERING, has_diagram=has_diagram,
                requested_tier=requested_tier, want_tier3=want_tier3,
                question_id=tier3_question_id,
            )
        # A regenerated hint gets the same answer-leak check as the first one before it goes
        # round the loop for a fresh safety check, so no path skips the guard.
        hint = _guard_against_answer_leak(
            session, hint, effective_gate_result, question_text=question_text,
            question_id=response_question_id or attempt_question_id,
            worked_solution_text=worked_solution_text, has_diagram=has_diagram,
            requested_tier=requested_tier, want_tier3=want_tier3, tier3_question_id=tier3_question_id,
        )

    try:
        now = _now_iso()
        if existing_attempt is not None:
            # Escalation: advance the existing attempt in place rather than duplicating it.
            attempt = existing_attempt
            attempt.current_tier = hint["tier"]
            attempt.last_activity_at = now
            session.add(attempt)
            session.flush()
        else:
            attempt = Attempt(
                student_id=student.id, question_id=attempt_question_id,
                student_submission_id=attempt_student_submission_id, current_tier=hint["tier"],
                status="active", started_at=now, last_activity_at=now,
            )
            session.add(attempt)
            session.flush()

        hint_event = HintEvent(
            attempt_id=attempt.id, tier=hint["tier"], hint_text=hint["hint_text"],
            self_consistency_result=self_consistency_summary, request_id=request_id,
            # Audit fields. `generate_hint()` always sets these; `.get()` only tolerates test
            # doubles that return a bare {"tier", "hint_text"} dict.
            tier_allowed=hint.get("tier_allowed"), grounded_in=hint.get("grounded_in"),
            verified_content=hint.get("verified_content"),
            # Tier 3 retrieval score and degrade reason, kept for later threshold calibration.
            tier3_similarity_score=hint.get("tier3_similarity_score"),
            tier3_degrade_reason=hint.get("tier3_degrade_reason"),
            created_at=now,
        )
        session.add(hint_event)
        session.flush()
    except (IntegrityError, OperationalError):
        # A concurrent request with the same request_id won the race: return its row instead
        # (see Issue 168).
        session.rollback()
        existing_after_race = session.scalar(
            select(HintEvent).where(HintEvent.request_id == request_id),
        )
        if existing_after_race is not None:
            return _existing_hint_event_response(existing_after_race)
        raise

    # Branch on the gate's `tier_allowed`, not on `hint["tier"]`: the first hint is always Tier 1,
    # so tier 1 with `tier_allowed >= 2` is an escalatable `HintResponse`, not a capped one.
    can_escalate_further = _can_escalate_further(effective_gate_result.tier_allowed, hint["tier"])
    if effective_gate_result.tier_allowed <= 1:
        return CappedTier1Response(
            question_id=response_question_id, student_submission_id=response_student_submission_id,
            attempt_id=attempt.id, hint_text=hint["hint_text"], reason=effective_gate_result.reason,
            gate_path_used=gate_path_used, is_unverified=is_unverified,
            can_escalate_further=can_escalate_further,
        )
    return HintResponse(
        question_id=response_question_id, attempt_id=attempt.id,
        tier=hint["tier"], hint_text=hint["hint_text"],
        gate_path_used=gate_path_used, is_unverified=is_unverified,
        can_escalate_further=can_escalate_further,
    )


def _existing_hint_event_response(
    hint_event: HintEvent, reason_override: Optional[str] = None,
) -> AnyHomeworkResponse:
    """Rebuild the typed response for an already-written hint event, without re-running the gate.

    Used for idempotent replays of a `request_id`, and by the escalation endpoint when there is no
    higher tier left.

    Weak mode is determined from the submission's `matched_question_id`, not from
    `student_submission_id`, because every photo attempt sets the latter (see Issue 190). The
    response type follows the persisted `tier_allowed`, falling back to `tier` for older rows
    where it is NULL.

    Args:
        reason_override: Replaces the default replay reason in a `CappedTier1Response`."""
    attempt = hint_event.attempt
    if attempt.student_submission_id is not None:
        real_question_id = attempt.student_submission.matched_question_id
        is_weak_mode = real_question_id is None
    else:
        real_question_id = attempt.question_id
        is_weak_mode = False
    gate_path_used = "weak_mode" if is_weak_mode else "verified_match"
    effective_tier_allowed = (
        hint_event.tier_allowed if hint_event.tier_allowed is not None else hint_event.tier
    )
    # A NULL tier_allowed makes tier_ceiling() return 1, so older rows never claim escalation
    # room that cannot be confirmed (fail closed).
    can_escalate_further = _can_escalate_further(hint_event.tier_allowed, hint_event.tier)
    if effective_tier_allowed <= 1:
        return CappedTier1Response(
            question_id=None if is_weak_mode else real_question_id,
            student_submission_id=attempt.student_submission_id if is_weak_mode else None,
            attempt_id=hint_event.attempt_id, hint_text=hint_event.hint_text,
            reason=reason_override or "(cached — idempotent replay of request_id)",
            gate_path_used=gate_path_used, is_unverified=is_weak_mode,
            can_escalate_further=can_escalate_further,
        )
    # A weak-mode hint above Tier 1 is invalid (also enforced by the schema). Do not coerce it
    # here; let HintResponse's validator raise so the upstream bug is visible (see Issue 180).
    return HintResponse(
        question_id=real_question_id, attempt_id=hint_event.attempt_id,
        tier=hint_event.tier, hint_text=hint_event.hint_text,
        gate_path_used=gate_path_used, is_unverified=is_weak_mode,
        can_escalate_further=can_escalate_further,
    )


@router.post("/questions/{question_id}/hint", response_model=AnyHomeworkResponse)
def request_hint(question_id: int, request_id: str) -> AnyHomeworkResponse:
    with get_session() as session:
        # Idempotency guard, checked first so a replayed request_id never reaches the gate. It
        # also runs before the latency tracker is created, so replays do not skew latency data.
        existing = session.scalar(
            select(HintEvent).where(HintEvent.request_id == request_id),
        )
        if existing is not None:
            return _existing_hint_event_response(existing)
        latency_tracker = RequestLatencyTracker(request_id)

        question = session.scalar(select(Question).where(Question.id == question_id))
        if question is None:
            raise HTTPException(status_code=404, detail=f"No question with id {question_id}")
        if not question.answer_value:
            raise HTTPException(
                status_code=422,
                detail=f"Question {question_id} has no real answer_value to gate-check against.",
            )
        # A few corpus rows are exam cover-page boilerplate, some with a stray answer_value, so
        # the check above does not catch them (see Issue 209). A substring test is used
        # because extraction_flag may hold several "; "-separated flags.
        if question.extraction_flag and NON_QUESTION_EXTRACTION_FLAG in question.extraction_flag:
            raise HTTPException(
                status_code=422,
                detail=f"Question {question_id} is flagged as non-question extraction "
                       "content, so it cannot be checked by the gate.",
            )

        # The demo student is seeded at startup (app/main.py lifespan), so this is normally a
        # plain SELECT. The retry covers a concurrent create, e.g. in tests (see Issue 168).
        try:
            student = _get_or_create_demo_student(session)
        except (IntegrityError, OperationalError):
            session.rollback()
            student = _get_or_create_demo_student(session)

        # Only one GPU model may be loaded per process, so a switch is refused and the process
        # restarts (Issues 366, 385). The guard runs even before a possible cache hit, in case a
        # stale cache row forces a live gate run.
        try:
            swap_tracker.guard("llm")
        except swap_tracker.SwapRestartRequired:
            swap_tracker.trigger_restart()
            raise HTTPException(
                status_code=503,
                detail="The server needs to restart before it can process this hint request. "
                       "Please retry in about 45 seconds.",
                headers={"Retry-After": "45"},
            )

        # Every call starts a new attempt, so the first hint is always Tier 1 regardless of
        # tier_allowed. Tiers 2 and 3 are reached only via POST /attempts/{id}/escalate.
        effective_gate_result, hint, self_consistency_summary = _resolve_gate_result_and_hint(
            session, question, latency_tracker, requested_tier=1,
        )

        # Safety check, persistence and response via the shared finalisation path. This endpoint
        # only serves corpus questions, so gate_path_used is set explicitly to "verified_match".
        return _finalize_and_persist_hint(
            session, student, hint, effective_gate_result, self_consistency_summary, request_id,
            attempt_question_id=question.id, attempt_student_submission_id=None,
            response_question_id=question.id, response_student_submission_id=None,
            gate_path_used="verified_match", is_unverified=False,
            question_text=question.question_text,
            worked_solution_text=question.worked_solution_text,
            has_diagram=question.has_diagram,
            latency_tracker=latency_tracker,
            requested_tier=1,
        )
