"""
Tests that a safety classifier crash fails closed through a dedicated, logged path (Issue 210).

Without this, a crash still failed closed, but only as a raw HTTP 500: there was no dedicated
response, no logging, and a stack trace could leak to the client.

The tests run end to end via `TestClient` at both insertion points:
`_create_submission_and_confirmation_response()` on the input side and
`_finalize_and_persist_hint()` on the output side. `test_safety_wiring.py` covers the normal
flagged and safe paths at the same call sites. `check_safety` is monkeypatched to raise, because
a classifier crash cannot be produced reliably otherwise; `test_safety_wiring.py` monkeypatches
in the same way.
"""
from __future__ import annotations

import logging

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.api import routes as routes_module
from app.api import submission_routes as submission_routes_module
from app.db.session import get_session
from app.main import app
from app.models.question import Question
from app.models.safety_flag import SafetyFlag


def _raise(*args, **kwargs):
    raise RuntimeError("simulated real classifier crash (Issue 210 test)")


def _assert_is_real_fail_closed_response(body: dict) -> None:
    """Assert the body is the typed fail-closed response, not a raw 500.

    The classifier-error response reuses the crisis response shape produced for a self_harm or
    dangerous_content flag (see `build_classifier_error_response()` for why).
    """
    assert body["state"] == "safety_blocked"
    assert body["is_self_harm_path"] is True
    assert body["parent_notified"] is False
    assert len(body["message"]) > 20


def test_input_side_classifier_crash_fails_closed_not_a_raw_500(monkeypatch, caplog):
    monkeypatch.setattr(submission_routes_module, "check_safety", _raise)

    with caplog.at_level(logging.ERROR, logger="api.submission_routes"):
        with TestClient(app) as client:
            response = client.post(
                "/api/submissions/photo-question",
                json={"question_text": "What is 2 + 2?", "has_diagram": False},
            )

    # A typed response, never a raw 500 that leaks a stack trace to the client.
    assert response.status_code == 200
    _assert_is_real_fail_closed_response(response.json())

    # Logged at ERROR level with exception info, checked via caplog.
    error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(error_records) >= 1
    record = error_records[0]
    assert "Issue 210" in record.message
    assert "INPUT" in record.message
    assert record.exc_info is not None
    assert record.exc_info[0] is RuntimeError

    # An auditable database record with a distinct category, not labelled as a detected flag.
    with get_session() as session:
        flag = session.scalars(
            select(SafetyFlag).where(SafetyFlag.category == "classifier_error"),
        ).all()
        assert len(flag) >= 1
        assert flag[-1].flagged_input_or_output == "input"


def test_output_side_classifier_crash_fails_closed_not_a_raw_500(monkeypatch, caplog):
    with get_session() as session:
        question = Question(
            source_paper="data/School papers/2024/P6_Maths_2024_WA2_nanhua.pdf",
            source_page_index=7, source_page_label=None,
            answer_source_file="data/School papers/2024/P6_Maths_2024_WA2_nanhua.pdf",
            answer_source_page_index=20, question_number="19",
            question_text=r"Find the value of \(\frac{3}{4} \div 12\).",
            answer_value="1/16", has_diagram=False, worked_solution_text=None,
        )
        session.add(question)
        session.flush()
        question_id = question.id

    monkeypatch.setattr(routes_module, "check_safety", _raise)

    with caplog.at_level(logging.ERROR, logger="api.routes"):
        with TestClient(app) as client:
            response = client.post(
                f"/api/questions/{question_id}/hint",
                params={"request_id": "classifier-crash-output-test"},
            )

    assert response.status_code == 200
    _assert_is_real_fail_closed_response(response.json())

    error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(error_records) >= 1
    record = error_records[0]
    assert "Issue 210" in record.message
    assert "OUTPUT" in record.message
    assert str(question_id) in record.message
    assert record.exc_info is not None
    assert record.exc_info[0] is RuntimeError

    with get_session() as session:
        flag = session.scalars(
            select(SafetyFlag).where(SafetyFlag.category == "classifier_error"),
        ).all()
        assert len(flag) >= 1
        assert flag[-1].flagged_input_or_output == "output"

        # No hint_events row is written for a crashed classification, because the hint text
        # never passed the safety check.
        from app.models.hint_event import HintEvent
        hint_event = session.scalar(
            select(HintEvent).where(HintEvent.request_id == "classifier-crash-output-test"),
        )
        assert hint_event is None
