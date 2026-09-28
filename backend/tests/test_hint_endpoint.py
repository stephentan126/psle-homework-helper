"""
Integration tests for the Socratic-gate hint endpoint (Phase 3 step 6, design specification
Section 10.6).

These replace the Phase 2 test of the hardcoded response; the Phase 3 step 6 docstring in
`routes.py` explains why that endpoint changed. Nothing is mocked: the tests use a database row, a
FastAPI `TestClient` POST, a Phi-4-mini generation through the gate (`sympy_exact_check()`) and the
`attempts` and `hint_events` writes. The question is a fast sum-type one (`3/4 ÷ 12`) rather than a
reasoning-type one. Self-consistency passes were measured at 10-20s or more each, which would make
the suite slow enough to be skipped, while the SymPy path takes a few seconds and covers the same
end-to-end wiring.

The file also holds the test of the step 7 idempotency guard (Section 10.5, Issue 57): the same
`request_id` posted twice returns the cached result without re-running the gate, checked by
counting `hint_events` rows rather than by mocking.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.db.session import get_session
from app.main import app
from app.models.hint_event import HintEvent
from app.models.question import Question


def _insert_test_question() -> int:
    with get_session() as session:
        question = Question(
            source_paper="data/School papers/2024/P6_Maths_2024_WA2_nanhua.pdf",
            source_page_index=5,
            source_page_label=None,
            answer_source_file="data/School papers/2024/P6_Maths_2024_WA2_nanhua.pdf",
            answer_source_page_index=20,
            question_number="19",
            question_text=(
                r"Find the value of \(\frac{3}{4} \div 12\). Leave your answer in its "
                r"simplest form."
            ),
            answer_value="1/16",
            worked_solution_text=None,
            has_diagram=False,
            ocr_confidence="high",
            school_name="Nan Hua Primary School",
        )
        session.add(question)
        session.flush()
        return question.id


def test_hint_endpoint_returns_real_gate_checked_hint():
    question_id = _insert_test_question()

    with TestClient(app) as client:
        response = client.post(
            f"/api/questions/{question_id}/hint", params={"request_id": "test-request-1"},
        )

    assert response.status_code == 200
    body = response.json()
    # The SymPy exact check passes on this question, giving a "hint" state rather than a
    # capped_tier1 fallback. The tier shown is still 1 because every first hint starts at Tier 1
    # however confident the gate is (multi-hint escalation ladder, development log). This
    # "confident but not yet escalated" state differs from a CappedTier1Response, which
    # can_escalate_further=True confirms: the gate passed at tier_allowed >= 2.
    assert body["state"] == "hint"
    assert body["question_id"] == question_id
    assert body["tier"] == 1
    assert body["can_escalate_further"] is True
    # The answer (1/16) must not appear in the hint in any written form. This is the project's
    # core safety rule, checked here against a gate-generated hint rather than a template.
    assert "1/16" not in body["hint_text"]
    assert "0.0625" not in body["hint_text"]


def test_hint_endpoint_idempotency_guard_real():
    """The same request_id posted twice runs the gate once (step 7, Section 10.5, Issue 57).

    Only one hint_events row exists after both calls.
    """
    question_id = _insert_test_question()

    with TestClient(app) as client:
        first = client.post(
            f"/api/questions/{question_id}/hint", params={"request_id": "test-idempotent-1"},
        )
        second = client.post(
            f"/api/questions/{question_id}/hint", params={"request_id": "test-idempotent-1"},
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["hint_text"] == second.json()["hint_text"]
    assert first.json()["attempt_id"] == second.json()["attempt_id"]

    with get_session() as session:
        from sqlalchemy import select
        rows = session.scalars(
            select(HintEvent).where(HintEvent.request_id == "test-idempotent-1"),
        ).all()
        assert len(rows) == 1, (
            f"Expected exactly one hint_events row for a replayed request_id, found {len(rows)} "
            f"-- the idempotency guard did not prevent a real duplicate gate run."
        )


def test_hint_endpoint_404_for_unknown_question():
    with TestClient(app) as client:
        response = client.post(
            "/api/questions/999999/hint", params={"request_id": "test-404"},
        )

    assert response.status_code == 404
