"""
Tests for `app/pipeline/safety_response.py` (Issues 203, 206 and 207).

Covers category-appropriate response construction and database audit logging, built alongside
the lexicon signal (Issue 202). Database writes go through `get_session()` and are not mocked.

Issue 207 renamed `record_self_harm_flag()` to `record_safety_flag()`, which takes a `category`
argument instead of a hardcoded `"self_harm"`, and added `build_sexually_explicit_response()`,
`build_harassment_response()` and the dispatcher `build_safety_response(category)` used by every
call site.
"""
from __future__ import annotations

from app.api.routes import _get_or_create_demo_student
from app.api.schemas import SafetyBlockedResponse
from app.db.session import get_session
from app.models.safety_flag import SafetyFlag
from app.pipeline.safety_response import (
    SAMARITANS_OF_SINGAPORE_RESOURCES,
    build_harassment_response,
    build_safety_response,
    build_self_harm_crisis_response,
    build_sexually_explicit_response,
    record_safety_flag,
)


def test_build_self_harm_crisis_response_is_a_real_safety_blocked_response_on_the_self_harm_path() -> None:
    response = build_self_harm_crisis_response()
    assert isinstance(response, SafetyBlockedResponse)
    assert response.state == "safety_blocked"
    assert response.is_self_harm_path is True
    # Sourced Samaritans of Singapore resources (Issue 203): never empty, never just the generic
    # block message.
    assert response.resources == SAMARITANS_OF_SINGAPORE_RESOURCES
    assert len(response.resources) >= 1
    assert "1767" in " ".join(response.resources)
    # Deliberate scope boundary: no automated parent notification is built.
    assert response.parent_notified is False
    # The message must be supportive, not a generic error string.
    assert "alone" in response.message.lower() or "you" in response.message.lower()
    assert len(response.message) > 20


def test_build_self_harm_crisis_response_message_never_leaks_a_final_answer() -> None:
    # A final answer is never revealed without the parent-PIN unlock path. The crisis response is
    # separate from any answer or hint content and must never carry one.
    response = build_self_harm_crisis_response()
    assert not hasattr(response, "answer_value")
    assert not hasattr(response, "hint_text")


def test_build_sexually_explicit_response_is_distinct_from_the_self_harm_crisis_response() -> None:
    """A sexually explicit flag shows no self-harm framing or crisis resources (Issues 206, 207)."""
    response = build_sexually_explicit_response()
    assert isinstance(response, SafetyBlockedResponse)
    assert response.state == "safety_blocked"
    assert response.is_self_harm_path is False
    assert response.resources is None
    assert response.parent_notified is False
    assert "samaritans" not in response.message.lower()
    assert "1767" not in response.message


def test_build_harassment_response_is_distinct_and_apologetic_not_a_refusal() -> None:
    """The harassment response reads as an apology or retry, not a refusal (Issue 207).

    Harassment here means a generated hint came out demeaning, which is a model output problem
    and never the student's fault. A refusal could imply the student's question was the problem.
    """
    response = build_harassment_response()
    assert isinstance(response, SafetyBlockedResponse)
    assert response.state == "safety_blocked"
    assert response.is_self_harm_path is False
    assert response.resources is None
    assert response.parent_notified is False
    assert "samaritans" not in response.message.lower()
    assert "sorry" in response.message.lower() or "again" in response.message.lower()


def test_build_safety_response_dispatches_self_harm_and_dangerous_content_to_the_crisis_response() -> None:
    """dangerous_content shares the self-harm crisis response, erring towards safety (Issue 207).

    ShieldGemma's policy text bundles the two, and no available signal separates them cleanly
    (gold-set case IN-U4).
    """
    for category in ("self_harm", "dangerous_content"):
        response = build_safety_response(category)
        assert response.is_self_harm_path is True
        assert response.resources == SAMARITANS_OF_SINGAPORE_RESOURCES


def test_build_safety_response_dispatches_sexually_explicit_and_harassment_correctly() -> None:
    sexual = build_safety_response("sexually_explicit")
    assert sexual.is_self_harm_path is False
    assert sexual.resources is None

    harassment = build_safety_response("harassment")
    assert harassment.is_self_harm_path is False
    assert harassment.resources is None
    # Each category has a distinct message, not one generic string reused.
    assert sexual.message != harassment.message


def test_build_safety_response_rejects_an_unknown_category() -> None:
    try:
        build_safety_response("not_a_real_category")
        assert False, "expected a ValueError for an unknown category"
    except ValueError:
        pass


def test_record_safety_flag_writes_a_real_auditable_row_with_the_real_category() -> None:
    with get_session() as session:
        student = _get_or_create_demo_student(session)
        student_id = student.id

    with get_session() as session:
        flag = record_safety_flag(
            session, student_id=student_id, attempt_id=None, flagged_input_or_output="input",
            category="sexually_explicit",
        )
        flag_id = flag.id

    with get_session() as session:
        row = session.get(SafetyFlag, flag_id)
        assert row is not None
        assert row.student_id == student_id
        assert row.attempt_id is None
        assert row.flagged_input_or_output == "input"
        # Issue 206: category is the caller-supplied value, not a hardcoded "self_harm" whatever
        # fired, so the audit trail is labelled correctly.
        assert row.category == "sexually_explicit"
        # No automated parent notification is built yet; this is a deliberate decision.
        assert row.parent_notified_at is None


def test_record_safety_flag_supports_output_direction_too() -> None:
    with get_session() as session:
        student = _get_or_create_demo_student(session)
        student_id = student.id

    with get_session() as session:
        flag = record_safety_flag(
            session, student_id=student_id, attempt_id=None, flagged_input_or_output="output",
            category="harassment",
        )
        flag_id = flag.id

    with get_session() as session:
        row = session.get(SafetyFlag, flag_id)
        assert row.flagged_input_or_output == "output"
        assert row.category == "harassment"


def test_record_safety_flag_rejects_an_invalid_direction() -> None:
    with get_session() as session:
        student = _get_or_create_demo_student(session)
        student_id = student.id

    with get_session() as session:
        try:
            record_safety_flag(
                session, student_id=student_id, attempt_id=None,
                flagged_input_or_output="sideways", category="self_harm",
            )
            assert False, "expected a ValueError for an invalid flagged_input_or_output"
        except ValueError:
            pass
