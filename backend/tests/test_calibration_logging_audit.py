"""
Tests for logging the Tier 3 similarity score used in calibration (calibration logging audit).

The audit found one gap. `generate_hint()` computed the cosine-similarity score of the Tier 3
related-example retrieval on every attempt but never stored it. On a degrade the score existed
only inside a free-text reason string, which the `HintEvent` construction in `routes.py` never
reads. The new `hint_events.tier3_similarity_score` and `tier3_degrade_reason` columns close this.

Dedup matching (`find_best_match()`) needed no change: `student_submissions.match_confidence` and
`match_method` already store the score for every live matching event, as `submission_routes.py`
shows. It has no tests here.

The tests fall into two groups, following the mix used elsewhere in the suite:

1. Fast, GPU-free unit tests of the new `generate_hint()` dict keys across all four branches (not
   attempted, no-question-id degrade, below-floor degrade, Tier 3 success). They monkeypatch
   `gate.find_related_worked_example` and `gate.generate` as in
   `test_multihint_tier3_and_persistence.py`.
2. One end-to-end integration test (TestClient, database, escalation endpoint and `HintEvent`
   persistence) showing the new columns reach the database row, not just the in-memory dict.
   `gate.generate` and `gate.find_related_worked_example` are monkeypatched for speed, since the
   test is about persistence wiring, not model output. The model-backed leak check is already
   covered by the end-to-end test in `test_multihint_tier3_and_persistence.py`.
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.api.routes import _get_or_create_demo_student
from app.db.session import get_session
from app.main import app
from app.models.attempt import Attempt
from app.models.hint_event import HintEvent
from app.models.question import Question
from app.pipeline import gate
from app.pipeline.gate import GateResult, generate_hint
from app.pipeline.question_matching import RelatedWorkedExample


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# =================================================================================================
# Fast, GPU-free unit tests on generate_hint()'s new dict keys.
# =================================================================================================


def _capture_generate(monkeypatch):
    calls: list[dict] = []

    def _fake_generate(system_prompt, user_prompt, max_new_tokens=200, do_sample=False, **kw):
        calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        return "FAKE HINT TEXT -- NOT A REAL GENERATION"

    monkeypatch.setattr(gate, "generate", _fake_generate)
    return calls


def test_tier1_hint_has_null_similarity_score_and_null_degrade_reason(monkeypatch):
    """When Tier 3 is not attempted (target_tier <= 1), both new fields are None."""
    _capture_generate(monkeypatch)
    gate_result = GateResult(passed=False, tier_allowed=1, reason="(test)", detail={})
    hint = generate_hint("Plain question.", gate_result, requested_tier=1)

    assert hint["tier"] == 1
    assert hint["tier3_similarity_score"] is None
    assert hint["tier3_degrade_reason"] is None


def test_no_question_id_degrade_has_null_similarity_score_real_degrade_reason(monkeypatch):
    """With no question_id, the score is None but a degrade reason is still recorded.

    find_related_worked_example is never called in this case, so there is no score to capture.
    """
    def _must_not_be_called(*a, **kw):
        raise AssertionError("find_related_worked_example must not be called with no question_id")

    monkeypatch.setattr(gate, "find_related_worked_example", _must_not_be_called)
    _capture_generate(monkeypatch)

    gate_result = GateResult(passed=True, tier_allowed=4, reason="(test)", detail={"expr_str": "2+2"})
    hint = generate_hint(
        "Plain question.", gate_result, worked_solution_text="2+2=4",
        want_tier3=True, question_id=None,
    )

    assert hint["tier"] == 2
    assert hint["tier3_similarity_score"] is None
    assert hint["tier3_degrade_reason"] == "no_question_id_supplied"


def test_below_floor_degrade_still_captures_the_real_score(monkeypatch):
    """A candidate scored below RELATED_EXAMPLE_MIN_SIMILARITY still has its score captured.

    This is the gap the audit found: the score used to be discarded when the attempt degraded.
    """
    monkeypatch.setattr(
        gate, "find_related_worked_example",
        lambda question_id, exclude_question_ids=None: RelatedWorkedExample(
            question_id=7, match_confidence=0.1, worked_solution_text="Unrelated solution.",
            match_method="related_worked_example_cosine_similarity_test",
        ),
    )
    _capture_generate(monkeypatch)

    gate_result = GateResult(passed=True, tier_allowed=4, reason="(test)", detail={"expr_str": "2+2"})
    hint = generate_hint(
        "Plain question.", gate_result, worked_solution_text="2+2=4",
        want_tier3=True, question_id=99,
    )

    assert hint["tier"] == 2
    assert hint["tier3_similarity_score"] == 0.1
    assert "no_qualifying_related_example" in hint["tier3_degrade_reason"]


def test_no_candidate_degrade_has_null_similarity_score(monkeypatch):
    """With an empty candidate pool, the stored score is None rather than 0.0.

    The returned RelatedWorkedExample has match_confidence=None, so no score exists.
    """
    monkeypatch.setattr(
        gate, "find_related_worked_example",
        lambda question_id, exclude_question_ids=None: RelatedWorkedExample(
            question_id=None, match_confidence=None, worked_solution_text=None,
            match_method="no_real_candidate_worked_example_available",
        ),
    )
    _capture_generate(monkeypatch)

    gate_result = GateResult(passed=True, tier_allowed=4, reason="(test)", detail={"expr_str": "2+2"})
    hint = generate_hint(
        "Plain question.", gate_result, worked_solution_text="2+2=4",
        want_tier3=True, question_id=99,
    )

    assert hint["tier"] == 2
    assert hint["tier3_similarity_score"] is None
    assert "no_qualifying_related_example" in hint["tier3_degrade_reason"]


def test_real_tier3_success_captures_the_real_score_and_null_degrade_reason(monkeypatch):
    """On Tier 3 success, the score is stored and the degrade reason is None."""
    monkeypatch.setattr(
        gate, "find_related_worked_example",
        lambda question_id, exclude_question_ids=None: RelatedWorkedExample(
            question_id=7, match_confidence=0.95,
            worked_solution_text="Step 1: ... Step 2: ... Final answer: 18.",
            match_method="related_worked_example_cosine_similarity_test",
        ),
    )
    _capture_generate(monkeypatch)

    gate_result = GateResult(passed=True, tier_allowed=4, reason="(test)", detail={"expr_str": "24-24*8/15"})
    hint = generate_hint(
        "Ali has 24 marbles. He gives away 8/15 of them. How many does he have left?", gate_result,
        want_tier3=True, question_id=99,
    )

    assert hint["tier"] == 3
    assert hint["tier3_similarity_score"] == 0.95
    assert hint["tier3_degrade_reason"] is None


# =================================================================================================
# End-to-end integration test: the new columns reach the database row.
# =================================================================================================


def _insert_test_question(source_page_index: int) -> int:
    with get_session() as session:
        question = Question(
            source_paper="data/School papers/2024/P6_Maths_2024_WA2_nanhua.pdf",
            source_page_index=source_page_index, source_page_label=None,
            answer_source_file="data/School papers/2024/P6_Maths_2024_WA2_nanhua.pdf",
            answer_source_page_index=source_page_index, question_number=str(source_page_index),
            question_text=(
                r"Find the value of \(\frac{3}{4} \div 12\). Leave your answer in its "
                r"simplest form."
            ),
            answer_value="1/16", worked_solution_text=None, has_diagram=False,
            ocr_confidence="high", school_name="Nan Hua Primary School",
        )
        session.add(question)
        session.flush()
        return question.id


def test_real_escalation_to_tier3_persists_the_real_similarity_score(monkeypatch):
    """Escalating to Tier 3 stores the similarity score in the HintEvent row.

    The call goes through the escalation endpoint with a TestClient and a database, so it checks
    the persistence wiring and not only the in-memory dict. gate.generate and
    find_related_worked_example are monkeypatched for speed.

    The starting Tier 1 HintEvent is written directly to the database, as in the
    `tier_allowed == 2` case of `test_multihint_escalation_endpoint.py`. Escalation rebuilds its
    GateResult from the stored row (`_reconstruct_gate_result_for_escalation()` in
    escalation_routes.py) rather than re-running SymPy. Going through `/hint` first would make the
    SymPy self-consistency check parse the fake `generate()` output as arithmetic, which it
    correctly refuses to do. That caps tier_allowed at 1 and makes Tier 3 unreachable.
    """
    monkeypatch.setattr(
        gate, "find_related_worked_example",
        lambda question_id, exclude_question_ids=None: RelatedWorkedExample(
            question_id=12345, match_confidence=0.87,
            worked_solution_text="A real related worked solution.",
            match_method="related_worked_example_cosine_similarity_test",
        ),
    )
    _capture_generate(monkeypatch)

    question_id = _insert_test_question(301)
    now = _now_iso()
    with get_session() as session:
        student = _get_or_create_demo_student(session)
        attempt = Attempt(
            student_id=student.id, question_id=question_id, current_tier=1,
            status="active", started_at=now, last_activity_at=now,
        )
        session.add(attempt)
        session.flush()
        attempt_id = attempt.id
        session.add(HintEvent(
            attempt_id=attempt_id, tier=1, hint_text="Tier 1 hint (direct DB setup).",
            request_id="calib-log-1-first", tier_allowed=4,
            grounded_in="sympy_confirmed_expression", verified_content="1/16",
            created_at=now,
        ))

    with TestClient(app) as client:
        second = client.post(
            f"/api/attempts/{attempt_id}/escalate", params={"request_id": "calib-log-1-second"},
        )
        assert second.status_code == 200
        assert second.json()["tier"] == 2

        third = client.post(
            f"/api/attempts/{attempt_id}/escalate", params={"request_id": "calib-log-1-third"},
        )
        assert third.status_code == 200
        third_body = third.json()

    assert third_body["tier"] == 3

    with get_session() as session:
        rows = session.scalars(
            select(HintEvent).where(HintEvent.attempt_id == attempt_id),
        ).all()
        rows_by_tier = {r.tier: r for r in rows}
        # The new columns reached the database row.
        assert rows_by_tier[3].tier3_similarity_score == 0.87
        assert rows_by_tier[3].tier3_degrade_reason is None
        # Tier 1 and 2 rows never attempted Tier 3, so both new columns stay NULL rather than
        # picking up a value from a later row.
        assert rows_by_tier[1].tier3_similarity_score is None
        assert rows_by_tier[1].tier3_degrade_reason is None
        assert rows_by_tier[2].tier3_similarity_score is None
        assert rows_by_tier[2].tier3_degrade_reason is None
