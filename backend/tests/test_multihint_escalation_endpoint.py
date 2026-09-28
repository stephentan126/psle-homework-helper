"""
Tests for the `POST /attempts/{id}/escalate` endpoint of the multi-hint escalation ladder.

Also covers the rule that the first hint is always Tier 1. The two core ladder cases
(Tier 1 -> Tier 2 -> Tier 3, and "already at ceiling") run end to end with `TestClient`, the test
database and the live model. Cases a live model cannot be forced to produce on demand, such as
the `tier_allowed == 2` method mismatch, are built from database rows directly, as in
`test_safety_wiring.py` and `test_multihint_tier3_and_persistence.py`.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.api.routes import _get_or_create_demo_student
from app.db.session import get_session
from app.main import app
from app.models.attempt import Attempt
from app.models.hint_event import HintEvent
from app.models.question import Question
from app.models.question_embedding import QuestionEmbedding
from app.pipeline import gate as gate_module
from app.pipeline.question_matching import EMBEDDING_MODEL_ID, _get_model

_QUESTION_TEXT = (
    r"Find the value of \(\frac{3}{4} \div 12\). Leave your answer in its simplest form."
)
_ANSWER_VALUE = "1/16"  # must never appear in any hint text


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _insert_test_question(
    source_page_index: int, question_text: str = _QUESTION_TEXT, answer_value: str = _ANSWER_VALUE,
) -> int:
    with get_session() as session:
        question = Question(
            source_paper="data/School papers/2024/P6_Maths_2024_WA2_nanhua.pdf",
            source_page_index=source_page_index, source_page_label=None,
            answer_source_file="data/School papers/2024/P6_Maths_2024_WA2_nanhua.pdf",
            answer_source_page_index=source_page_index, question_number=str(source_page_index),
            question_text=question_text, answer_value=answer_value,
            worked_solution_text=None, has_diagram=False,
            ocr_confidence="high", school_name="Nan Hua Primary School",
        )
        session.add(question)
        session.flush()
        return question.id


# =================================================================================================
# End to end: the full ladder, Tier 1 -> Tier 2 -> Tier 3.
# =================================================================================================


def test_real_escalation_from_tier1_to_tier2_end_to_end():
    question_id = _insert_test_question(201)

    with TestClient(app) as client:
        first = client.post(
            f"/api/questions/{question_id}/hint", params={"request_id": "esc-e2e-1-first"},
        )
        assert first.status_code == 200
        first_body = first.json()
        # The first hint is always Tier 1, even though this question's SymPy check passes
        # (tier_allowed=4, ceiling=3).
        assert first_body["state"] == "hint"
        assert first_body["tier"] == 1
        assert first_body["can_escalate_further"] is True
        attempt_id = first_body["attempt_id"]

        second = client.post(
            f"/api/attempts/{attempt_id}/escalate", params={"request_id": "esc-e2e-1-second"},
        )
        assert second.status_code == 200
        second_body = second.json()

    assert second_body["state"] == "hint"
    assert second_body["tier"] == 2
    assert second_body["attempt_id"] == attempt_id
    assert second_body["question_id"] == question_id
    # The ceiling is 3 (tier_allowed=4); current_tier is now 2, still below it.
    assert second_body["can_escalate_further"] is True
    assert _ANSWER_VALUE not in second_body["hint_text"]
    assert first_body["hint_text"] != second_body["hint_text"]

    with get_session() as session:
        attempt = session.get(Attempt, attempt_id)
        # The same attempt was advanced; no second attempt was created.
        assert attempt.current_tier == 2
        rows = session.scalars(
            select(HintEvent).where(HintEvent.attempt_id == attempt_id),
        ).all()
        assert len(rows) == 2  # one row per hint shown
        rows_by_tier = {r.tier: r for r in rows}
        assert rows_by_tier[1].tier_allowed == 4
        assert rows_by_tier[2].tier_allowed == 4
        assert rows_by_tier[2].grounded_in == "sympy_confirmed_expression"


def test_real_escalation_all_the_way_to_tier3_with_a_real_related_example():
    """The ladder reaches Tier 3 end to end, grounded in a related worked example.

    Embeddings are computed with `_get_model().encode()` and stored as `QuestionEmbedding` rows,
    as in `test_multihint_tier3_and_persistence.py`, so the Tier 3 retrieval finds a qualifying
    candidate instead of degrading.
    """
    current_qid = _insert_test_question(202)
    # A close sibling of _QUESTION_TEXT (same "Find the value of A ÷ B" shape, different numbers),
    # so it scores well above RELATED_EXAMPLE_MIN_SIMILARITY (0.5). A topically different word
    # problem (a stickers or marbles sharing scenario) scored below that floor against this bare
    # division question. That is the model correctly judging them unrelated, which is what the
    # floor is for.
    related_text = r"Find the value of \(\frac{2}{5} \div 6\). Leave your answer in its simplest form."
    with get_session() as session:
        related = Question(
            source_paper="data/School papers/2024/P6_Maths_2024_WA2_nanhua.pdf",
            source_page_index=203, source_page_label=None,
            answer_source_file="data/School papers/2024/P6_Maths_2024_WA2_nanhua.pdf",
            answer_source_page_index=203, question_number="203",
            question_text=related_text, answer_value="1/15",
            worked_solution_text=r"\(\frac{2}{5} \div 6 = \frac{2}{5} \times \frac{1}{6} = "
                                  r"\frac{2}{30} = \frac{1}{15}\)",
            has_diagram=False, ocr_confidence="high", school_name="Nan Hua Primary School",
        )
        session.add(related)
        session.flush()
        model = _get_model()
        embedding = model.encode([related_text], convert_to_numpy=True)[0]
        session.add(QuestionEmbedding(
            question_id=related.id, embedding_model=EMBEDDING_MODEL_ID,
            embedding_json=json.dumps(embedding.tolist()), computed_at=_now_iso(),
        ))
        current_embedding = model.encode([_QUESTION_TEXT], convert_to_numpy=True)[0]
        session.add(QuestionEmbedding(
            question_id=current_qid, embedding_model=EMBEDDING_MODEL_ID,
            embedding_json=json.dumps(current_embedding.tolist()), computed_at=_now_iso(),
        ))

    with TestClient(app) as client:
        first = client.post(
            f"/api/questions/{current_qid}/hint", params={"request_id": "esc-e2e-2-first"},
        )
        attempt_id = first.json()["attempt_id"]

        second = client.post(
            f"/api/attempts/{attempt_id}/escalate", params={"request_id": "esc-e2e-2-second"},
        )
        assert second.json()["tier"] == 2
        assert second.json()["can_escalate_further"] is True

        third = client.post(
            f"/api/attempts/{attempt_id}/escalate", params={"request_id": "esc-e2e-2-third"},
        )
        assert third.status_code == 200
        third_body = third.json()

    assert third_body["state"] == "hint"
    assert third_body["tier"] == 3
    # The ceiling is 3 (tier_allowed=4); current_tier is now 3, so there is nothing further.
    assert third_body["can_escalate_further"] is False
    assert _ANSWER_VALUE not in third_body["hint_text"]

    with get_session() as session:
        attempt = session.get(Attempt, attempt_id)
        assert attempt.current_tier == 3
        third_hint_event = session.scalar(
            select(HintEvent).where(HintEvent.request_id == "esc-e2e-2-third"),
        )
        assert third_hint_event.grounded_in == "related_worked_example"


# =================================================================================================
# Already at the ceiling: no escalation possible.
# =================================================================================================


def test_escalate_when_genuinely_capped_returns_cannot_escalate_and_points_to_pin_unlock():
    """A gate-capped attempt (tier_allowed <= 1, ceiling 1) cannot escalate.

    It must report can_escalate_further=False at once, without attempting any generation.
    """
    from app.models.student_submission import StudentSubmission

    with get_session() as session:
        student = _get_or_create_demo_student(session)
        # Exactly one of question_id and student_submission_id may be set. A weak-mode attempt
        # (student_submission_id set) avoids needing a Question row.
        submission = StudentSubmission(
            student_id=student.id, question_text="A genuinely novel question.", has_diagram=False,
            submitted_at=_now_iso(), matched_question_id=None, match_confidence=0.2,
            match_method="cosine_similarity_test_below_threshold", gate_path_used="weak_mode",
        )
        session.add(submission)
        session.flush()
        attempt = Attempt(
            student_id=student.id, question_id=None, student_submission_id=submission.id,
            current_tier=1, status="active", started_at=_now_iso(), last_activity_at=_now_iso(),
        )
        session.add(attempt)
        session.flush()
        attempt_id = attempt.id

        hint_event = HintEvent(
            attempt_id=attempt_id, tier=1, hint_text="A real, already-shown Tier 1 hint.",
            self_consistency_result=None, request_id="esc-capped-original",
            tier_allowed=1, grounded_in=None, verified_content=None, created_at=_now_iso(),
        )
        session.add(hint_event)

    with TestClient(app) as client:
        response = client.post(
            f"/api/attempts/{attempt_id}/escalate", params={"request_id": "esc-capped-attempt"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "capped_tier1"
    assert body["can_escalate_further"] is False
    assert "parent" in body["reason"].lower() or "pin" in body["reason"].lower()

    with get_session() as session:
        # Nothing new was written: no generation, no new HintEvent, no current_tier change.
        rows = session.scalars(
            select(HintEvent).where(HintEvent.attempt_id == attempt_id),
        ).all()
        assert len(rows) == 1
        attempt = session.get(Attempt, attempt_id)
        assert attempt.current_tier == 1


def test_escalate_idempotency_guard_real():
    question_id = _insert_test_question(204)
    with TestClient(app) as client:
        first = client.post(
            f"/api/questions/{question_id}/hint", params={"request_id": "esc-idem-first"},
        )
        attempt_id = first.json()["attempt_id"]

        second_a = client.post(
            f"/api/attempts/{attempt_id}/escalate", params={"request_id": "esc-idem-escalate"},
        )
        second_b = client.post(
            f"/api/attempts/{attempt_id}/escalate", params={"request_id": "esc-idem-escalate"},
        )

    assert second_a.status_code == 200
    assert second_b.status_code == 200
    assert second_a.json()["hint_text"] == second_b.json()["hint_text"]

    with get_session() as session:
        rows = session.scalars(
            select(HintEvent).where(HintEvent.request_id == "esc-idem-escalate"),
        ).all()
        assert len(rows) == 1, (
            f"Expected exactly one real hint_events row for a replayed escalation request_id, "
            f"found {len(rows)}."
        )


def test_escalate_unknown_attempt_returns_404():
    with TestClient(app) as client:
        response = client.post(
            "/api/attempts/999999999/escalate", params={"request_id": "esc-404-test"},
        )
    assert response.status_code == 404


def test_escalate_attempt_with_no_hint_events_yet_returns_422():
    real_question_id = _insert_test_question(205)
    with get_session() as session:
        student = _get_or_create_demo_student(session)
        attempt = Attempt(
            student_id=student.id, question_id=real_question_id, student_submission_id=None,
            current_tier=1, status="active", started_at=_now_iso(), last_activity_at=_now_iso(),
        )
        session.add(attempt)
        session.flush()
        attempt_id = attempt.id

    with TestClient(app) as client:
        response = client.post(
            f"/api/attempts/{attempt_id}/escalate", params={"request_id": "esc-no-hints-test"},
        )
    assert response.status_code == 422


# =================================================================================================
# Regression: tier_allowed == 2 (method mismatch) must never reach Tier 3 through this endpoint,
# checked on the endpoint's call path rather than relying on the guard inside generate_hint().
# =================================================================================================


def test_escalate_tier_allowed_2_stops_at_tier2_and_never_attempts_a_tier3_retrieval(monkeypatch):
    """A `tier_allowed == 2` attempt escalates to Tier 2 and stops, with no Tier 3 retrieval.

    The attempt is built from database rows because a live model cannot be forced into a method
    mismatch on demand. After one escalation, can_escalate_further must be False, and
    `find_related_worked_example` is patched to raise if it is called.
    """
    def _must_not_be_called(*args, **kwargs):
        raise AssertionError(
            "find_related_worked_example() must never be called for a tier_allowed==2 escalation",
        )

    # Patch the name where it is called from: gate.py imports find_related_worked_example into its
    # namespace, and generate_hint() uses that binding. Patching escalation_routes would miss it.
    # test_diagram_degrade_path.py uses the same technique.
    monkeypatch.setattr(gate_module, "find_related_worked_example", _must_not_be_called)

    question_id = _insert_test_question(206)
    with get_session() as session:
        student = _get_or_create_demo_student(session)
        attempt = Attempt(
            student_id=student.id, question_id=question_id, student_submission_id=None,
            current_tier=1, status="active", started_at=_now_iso(), last_activity_at=_now_iso(),
        )
        session.add(attempt)
        session.flush()
        attempt_id = attempt.id
        session.add(HintEvent(
            attempt_id=attempt_id, tier=1, hint_text="A real, already-shown Tier 1 hint.",
            self_consistency_result=None, request_id="esc-mismatch-original",
            tier_allowed=2, grounded_in="sympy_confirmed_expression", verified_content="3/4 / 12",
            created_at=_now_iso(),
        ))

    with TestClient(app) as client:
        response = client.post(
            f"/api/attempts/{attempt_id}/escalate", params={"request_id": "esc-mismatch-escalate"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "hint"
    assert body["tier"] == 2
    # The ceiling is 2 (tier_allowed=2) and current_tier is now 2, so there is nothing further.
    assert body["can_escalate_further"] is False
    assert _ANSWER_VALUE not in body["hint_text"]
