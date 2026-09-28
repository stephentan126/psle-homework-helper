"""
Tests for regenerating a hint flagged for harassment (Issue 207).

A harassment flag on a generated hint triggers a silent regenerate-and-retry (up to 2 retries, 3
generation attempts in total) before falling back to the generic safety_blocked message.

`check_safety` and `generate_hint` are monkeypatched in the `routes_module` namespace, as
`test_safety_wiring.py` does for `_resolve_gate_result_and_hint` and
`vlm_service.transcribe_photo`. These tests cover the retry orchestration only: the number of
calls, the steering text and the audit trail. Classifier and generation quality are covered by the
live tests in `test_safety_wiring.py`, which also confirm that self_harm, dangerous_content,
sexually_explicit and harassment without a retry still behave as before.
"""
from __future__ import annotations

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.api import routes as routes_module
from app.db.session import get_session
from app.main import app
from app.models.hint_event import HintEvent
from app.models.question import Question
from app.models.safety_flag import SafetyFlag
from app.pipeline.gate import GateResult
from app.pipeline.self_harm_detection import SafetyCheckResult


def _insert_test_question() -> int:
    with get_session() as session:
        question = Question(
            source_paper="data/School papers/2024/P6_Maths_2024_WA2_nanhua.pdf",
            source_page_index=8, source_page_label=None,
            answer_source_file="data/School papers/2024/P6_Maths_2024_WA2_nanhua.pdf",
            answer_source_page_index=20, question_number="19",
            question_text=r"Find the value of \(\frac{3}{4} \div 12\).",
            answer_value="1/16", has_diagram=False, worked_solution_text=None,
        )
        session.add(question)
        session.flush()
        return question.id


def _fake_gate_result_and_hint(session, question, latency_tracker=None, requested_tier=None):
    # `requested_tier` is accepted and ignored only to match the `_resolve_gate_result_and_hint()`
    # signature, which request_hint() calls with requested_tier=1. Tier behaviour is out of scope
    # here, so the fake always returns a fixed tier=2 gate result and hint.
    gate_result = GateResult(passed=True, tier_allowed=2, reason="(test)", detail={})
    hint = {"tier": 2, "hint_text": "ATTEMPT 1 -- HARASSING TEXT THAT MUST NEVER BE SERVED."}
    return gate_result, hint, "(test -- monkeypatched)"


def test_harassment_attempt_1_flagged_attempt_2_safe_serves_the_retry_not_the_block(monkeypatch):
    question_id = _insert_test_question()
    monkeypatch.setattr(routes_module, "_resolve_gate_result_and_hint", _fake_gate_result_and_hint)

    check_safety_calls: list[str] = []

    def _fake_check_safety(text, direction):
        check_safety_calls.append(text)
        if len(check_safety_calls) == 1:
            return SafetyCheckResult(flagged=True, category="harassment")
        return SafetyCheckResult(flagged=False, category=None)

    monkeypatch.setattr(routes_module, "check_safety", _fake_check_safety)

    generate_hint_calls: list[dict] = []

    def _fake_generate_hint(
        question_text, gate_result, worked_solution_text=None, extra_steering=None,
        has_diagram=False, **kwargs,
    ):
        # **kwargs absorbs requested_tier, want_tier3 and question_id, which the retry loop
        # forwards but which do not matter to this fake.
        generate_hint_calls.append({"extra_steering": extra_steering})
        return {"tier": 2, "hint_text": "ATTEMPT 2 -- SAFE REGENERATED HINT.", "grounded_in": "test"}

    monkeypatch.setattr(routes_module, "generate_hint", _fake_generate_hint)

    with TestClient(app) as client:
        response = client.post(
            f"/api/questions/{question_id}/hint",
            params={"request_id": "harassment-retry-then-safe"},
        )

    assert response.status_code == 200
    body = response.json()
    # The student receives the attempt 2 hint, not the generic blocked message or attempt 1's
    # flagged text.
    assert body["state"] == "hint"
    assert body["hint_text"] == "ATTEMPT 2 -- SAFE REGENERATED HINT."

    # check_safety is called twice and the retry generate_hint once, with non-empty steering text
    # rather than a blind re-roll.
    assert len(check_safety_calls) == 2
    assert check_safety_calls[0] == "ATTEMPT 1 -- HARASSING TEXT THAT MUST NEVER BE SERVED."
    assert len(generate_hint_calls) == 1
    assert generate_hint_calls[0]["extra_steering"]
    assert len(generate_hint_calls[0]["extra_steering"]) > 20

    with get_session() as session:
        # Attempt 1's flagged content is not persisted as the hint_events row.
        hint_event = session.scalar(
            select(HintEvent).where(HintEvent.request_id == "harassment-retry-then-safe"),
        )
        assert hint_event is not None
        assert hint_event.hint_text == "ATTEMPT 2 -- SAFE REGENERATED HINT."
        assert "ATTEMPT 1" not in hint_event.hint_text
        assert "HARASSING" not in hint_event.hint_text

        # The retried attempt gets its own audit marker, not the plain "harassment" category
        # used for a final block.
        retry_flags = session.scalars(
            select(SafetyFlag).where(SafetyFlag.category == "harassment_retry_attempt_1"),
        ).all()
        assert len(retry_flags) >= 1
        assert retry_flags[-1].flagged_input_or_output == "output"

        # No final "harassment" block is recorded, because the retry succeeded.
        final_flags = session.scalars(
            select(SafetyFlag).where(SafetyFlag.category == "harassment"),
        ).all()
        assert len(final_flags) == 0


def test_harassment_flagged_all_3_attempts_falls_back_to_the_generic_block(monkeypatch):
    question_id = _insert_test_question()
    monkeypatch.setattr(routes_module, "_resolve_gate_result_and_hint", _fake_gate_result_and_hint)

    check_safety_calls: list[str] = []

    def _fake_check_safety(text, direction):
        check_safety_calls.append(text)
        return SafetyCheckResult(flagged=True, category="harassment")  # every attempt is flagged

    monkeypatch.setattr(routes_module, "check_safety", _fake_check_safety)

    generate_hint_calls: list[dict] = []

    def _fake_generate_hint(
        question_text, gate_result, worked_solution_text=None, extra_steering=None,
        has_diagram=False, **kwargs,
    ):
        generate_hint_calls.append({"extra_steering": extra_steering})
        return {
            "tier": 2, "hint_text": f"ATTEMPT {len(generate_hint_calls) + 1} -- STILL HARASSING.",
            "grounded_in": "test",
        }

    monkeypatch.setattr(routes_module, "generate_hint", _fake_generate_hint)

    with TestClient(app) as client:
        response = client.post(
            f"/api/questions/{question_id}/hint",
            params={"request_id": "harassment-retry-exhausted"},
        )

    assert response.status_code == 200
    body = response.json()
    # The existing generic block, unchanged.
    assert body["state"] == "safety_blocked"
    assert body["is_self_harm_path"] is False  # harassment never gets the self-harm crisis framing

    # check_safety is called 3 times (the original and 2 retries) and there are 2 regeneration
    # calls. Attempt 3's flagged result causes the block, so there is no fourth generation.
    assert len(check_safety_calls) == 3
    assert len(generate_hint_calls) == 2

    with get_session() as session:
        # The audit trail shows the full retry history: 2 retry markers (attempts 1 and 2) and a
        # final "harassment" block (attempt 3), rather than one aggregate row.
        retry_1 = session.scalars(
            select(SafetyFlag).where(SafetyFlag.category == "harassment_retry_attempt_1"),
        ).all()
        retry_2 = session.scalars(
            select(SafetyFlag).where(SafetyFlag.category == "harassment_retry_attempt_2"),
        ).all()
        final_block = session.scalars(
            select(SafetyFlag).where(SafetyFlag.category == "harassment"),
        ).all()
        assert len(retry_1) >= 1
        assert len(retry_2) >= 1
        assert len(final_block) >= 1

        # No hint_events row: a blocked response is never stored as a hint, as with the
        # immediate block for every other category.
        hint_event = session.scalar(
            select(HintEvent).where(HintEvent.request_id == "harassment-retry-exhausted"),
        )
        assert hint_event is None


def test_self_harm_flag_on_generated_hint_gets_no_retry_immediate_block_unchanged(monkeypatch):
    """A self_harm flag on a generated hint is blocked immediately with no retry.

    The retry loop applies only to harassment. self_harm, dangerous_content and sexually_explicit
    get one check_safety call, no regeneration and the usual immediate response.
    """
    question_id = _insert_test_question()
    monkeypatch.setattr(routes_module, "_resolve_gate_result_and_hint", _fake_gate_result_and_hint)

    check_safety_calls: list[str] = []

    def _fake_check_safety(text, direction):
        check_safety_calls.append(text)
        return SafetyCheckResult(flagged=True, category="self_harm")

    monkeypatch.setattr(routes_module, "check_safety", _fake_check_safety)

    generate_hint_calls: list[dict] = []

    def _fake_generate_hint(
        question_text, gate_result, worked_solution_text=None, extra_steering=None,
        has_diagram=False, **kwargs,
    ):
        generate_hint_calls.append({})  # must never be called for this category
        return {"tier": 2, "hint_text": "SHOULD NEVER BE CALLED.", "grounded_in": "test"}

    monkeypatch.setattr(routes_module, "generate_hint", _fake_generate_hint)

    with TestClient(app) as client:
        response = client.post(
            f"/api/questions/{question_id}/hint",
            params={"request_id": "harassment-self-harm-no-retry"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "safety_blocked"
    assert body["is_self_harm_path"] is True  # unchanged crisis-response framing

    # No retry: one check_safety call and no regeneration calls.
    assert len(check_safety_calls) == 1
    assert len(generate_hint_calls) == 0

    with get_session() as session:
        flags = session.scalars(
            select(SafetyFlag).where(SafetyFlag.category == "self_harm"),
        ).all()
        assert len(flags) >= 1
        hint_event = session.scalar(
            select(HintEvent).where(HintEvent.request_id == "harassment-self-harm-no-retry"),
        )
        assert hint_event is None
