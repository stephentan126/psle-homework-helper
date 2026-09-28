"""
Integration tests for the hint endpoint's `precomputed_gate_results` cache lookup in `routes.py`.

Two behaviours are checked: a cached row with matching versions is used and the live gate is
skipped; and any mismatch is treated as "no cached row" and falls back to the live pipeline.

The cache-hit test inserts a `PrecomputedGateResult` row directly rather than producing it with
the LLM. That is sufficient because it tests the endpoint's lookup logic, not gate correctness,
which `test_hint_endpoint.py` covers with a live LLM call. The fallback tests use the fast SymPy
path on the same question as `test_hint_endpoint.py`, so they stay quick without mocking.
"""
from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.db.session import get_session
from app.main import app
from app.models.hint_event import HintEvent
from app.models.precomputed_gate_result import PrecomputedGateResult
from app.models.question import Question
from app.pipeline.gate import GATE_VERSION, compute_gate_content_fingerprint
from app.services.llm_service import LLM_ADAPTER_PATH, LLM_MODEL_PATH

_TEST_QUESTION_TEXT = (
    r"Find the value of \(\frac{3}{4} \div 12\). Leave your answer in its simplest form."
)
_TEST_ANSWER_VALUE = "1/16"


def _insert_test_question() -> int:
    with get_session() as session:
        question = Question(
            source_paper="data/School papers/2024/P6_Maths_2024_WA2_nanhua.pdf",
            source_page_index=5,
            source_page_label=None,
            answer_source_file="data/School papers/2024/P6_Maths_2024_WA2_nanhua.pdf",
            answer_source_page_index=20,
            question_number="19",
            question_text=_TEST_QUESTION_TEXT,
            answer_value=_TEST_ANSWER_VALUE,
            worked_solution_text=None,
            has_diagram=False,
            ocr_confidence="high",
            school_name="Nan Hua Primary School",
        )
        session.add(question)
        session.flush()
        return question.id


def test_hint_endpoint_uses_real_version_matching_precomputed_row():
    """An active cached row with matching versions supplies the gate result without a live rerun.

    The gate result (SymPy or self-consistency) is the expensive part. The cache stores only one
    tier of hint text (tier 2 or above), which cannot satisfy the rule that the first hint is
    always Tier 1. A cheap live generate() call therefore produces the Tier 1 text even on a
    cache hit (see `_resolve_gate_result_and_hint()`). Reuse is shown by the
    `self_consistency_result` marker and the cached `tier_allowed` on the new hint_events row.
    """
    question_id = _insert_test_question()
    with get_session() as session:
        session.add(PrecomputedGateResult(
            question_id=question_id,
            gate_version=GATE_VERSION,
            llm_model_path=LLM_MODEL_PATH,
            # Issue 369: must also match the configured adapter, like gate_version and
            # llm_model_path. A precompute row always stamps this.
            adapter_path=LLM_ADAPTER_PATH,
            # Issue 178: a cache row carries a content fingerprint, computed here from the
            # content _insert_test_question() wrote, as a precompute run would.
            content_fingerprint=compute_gate_content_fingerprint(
                _TEST_QUESTION_TEXT, _TEST_ANSWER_VALUE, False, None,
            ),
            question_type="sum",
            tier_allowed=2,
            gate_passed=True,
            gate_reason="SymPy-confirmed (fake, precomputed test row)",
            gate_detail_json=json.dumps({"expr_str": "3/4 / 12"}),
            hint_tier=2,
            hint_text="THIS EXACT TEXT PROVES THE CACHE WAS USED, NOT A FRESH LLM CALL.",
            computed_at="2026-09-02T00:00:00+00:00",
            is_active=True,
        ))
        session.flush()

    with TestClient(app) as client:
        response = client.post(
            f"/api/questions/{question_id}/hint",
            params={"request_id": "precompute-cache-hit-test"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "hint"
    # The first hint is always Tier 1, even for this confident (tier_allowed=2) cached result.
    assert body["tier"] == 1
    # The cached gate_detail_json (expr_str="3/4 / 12") grounded this Tier 1 generation. A live
    # gate rerun would have extracted "3/4/12" or similar, not this distinctive string.
    # can_escalate_further is True because the ceiling (tier_allowed=2 -> ceiling 2) is above
    # the Tier 1 just shown.
    assert body["can_escalate_further"] is True

    with get_session() as session:
        from sqlalchemy import select
        hint_event = session.scalar(
            select(HintEvent).where(HintEvent.request_id == "precompute-cache-hit-test"),
        )
        assert hint_event is not None
        assert "precomputed_gate_results" in hint_event.self_consistency_result
        # The cached tier_allowed was reused. The wrong-adapter and stale-version rows elsewhere in
        # this file use different values so that this check can tell the right row from a
        # coincidentally similar live result.
        assert hint_event.tier_allowed == 2
        assert hint_event.grounded_in == "sympy_confirmed_expression"
        assert hint_event.verified_content == "3/4 / 12"


def test_hint_endpoint_falls_back_to_live_on_gate_version_mismatch():
    """A cached row from a different gate_version is treated as absent and the live gate runs."""
    question_id = _insert_test_question()
    with get_session() as session:
        session.add(PrecomputedGateResult(
            question_id=question_id,
            gate_version="gate_v0_OLD_STALE_VERSION",  # deliberately does NOT match GATE_VERSION
            llm_model_path=LLM_MODEL_PATH,
            adapter_path=LLM_ADAPTER_PATH,  # correct, isolates the miss to gate_version alone
            content_fingerprint=compute_gate_content_fingerprint(
                _TEST_QUESTION_TEXT, _TEST_ANSWER_VALUE, False, None,
            ),
            question_type="sum",
            tier_allowed=1,
            gate_passed=False,
            gate_reason="(fake stale row -- must never be served)",
            gate_detail_json=json.dumps({}),
            hint_tier=1,
            hint_text="STALE TEXT THAT MUST NEVER BE RETURNED.",
            computed_at="2020-01-01T00:00:00+00:00",
            is_active=True,
        ))
        session.flush()

    with TestClient(app) as client:
        response = client.post(
            f"/api/questions/{question_id}/hint",
            params={"request_id": "precompute-version-mismatch-test"},
        )

    assert response.status_code == 200
    body = response.json()
    # The live SymPy path passes for this question (validated separately as id=4069), giving a
    # "hint" state at tier >= 2 and not the stale cached text.
    assert body["state"] == "hint"
    assert body["hint_text"] != "STALE TEXT THAT MUST NEVER BE RETURNED."

    with get_session() as session:
        from sqlalchemy import select
        hint_event = session.scalar(
            select(HintEvent).where(HintEvent.request_id == "precompute-version-mismatch-test"),
        )
        assert hint_event is not None
        assert "precomputed_gate_results" not in (hint_event.self_consistency_result or "")


def test_hint_endpoint_treats_content_edited_after_caching_as_a_cache_miss():
    """A cached row whose fingerprint no longer matches the question content is a miss (Issue 178).

    The live gate must run instead of serving a stale result for edited content. Nothing edits
    questions in place at present, but the safeguard must work if something does.
    """
    question_id = _insert_test_question()
    with get_session() as session:
        session.add(PrecomputedGateResult(
            question_id=question_id,
            gate_version=GATE_VERSION,
            llm_model_path=LLM_MODEL_PATH,
            adapter_path=LLM_ADAPTER_PATH,  # correct, isolates the miss to content_fingerprint alone
            # A correct fingerprint for the original content. gate_version and llm_model_path
            # both match, so only the content_fingerprint check can reject this row.
            content_fingerprint=compute_gate_content_fingerprint(
                _TEST_QUESTION_TEXT, _TEST_ANSWER_VALUE, False, None,
            ),
            question_type="sum",
            tier_allowed=2,
            gate_passed=True,
            gate_reason="SymPy-confirmed (fake, precomputed test row)",
            gate_detail_json=json.dumps({"expr_str": "3/4 / 12"}),
            hint_tier=2,
            hint_text="STALE TEXT FROM BEFORE THE QUESTION WAS EDITED -- MUST NEVER BE RETURNED.",
            computed_at="2026-09-02T00:00:00+00:00",
            is_active=True,
        ))
        session.flush()

    # Simulate an in-place edit by appending text to question_text in the database. The
    # answer_value stays correct, so the live gate still passes. This separates "was the cache
    # bypassed" from "did the live gate work".
    edited_question_text = _TEST_QUESTION_TEXT + " (edited after caching, Issue 178 test)"
    with get_session() as session:
        question = session.get(Question, question_id)
        question.question_text = edited_question_text
        session.flush()

    # Check the fingerprint changed, so the HTTP result below is explained by this mechanism.
    assert compute_gate_content_fingerprint(
        edited_question_text, _TEST_ANSWER_VALUE, False, None,
    ) != compute_gate_content_fingerprint(
        _TEST_QUESTION_TEXT, _TEST_ANSWER_VALUE, False, None,
    )

    with TestClient(app) as client:
        response = client.post(
            f"/api/questions/{question_id}/hint",
            params={"request_id": "precompute-content-staleness-test"},
        )

    assert response.status_code == 200
    body = response.json()
    # The live gate on the edited text still passes SymPy verification, since the appended text
    # does not change the maths. The result is a "hint" state, not the stale cached text.
    assert body["state"] == "hint"
    assert body["hint_text"] != (
        "STALE TEXT FROM BEFORE THE QUESTION WAS EDITED -- MUST NEVER BE RETURNED."
    )

    with get_session() as session:
        from sqlalchemy import select
        hint_event = session.scalar(
            select(HintEvent).where(HintEvent.request_id == "precompute-content-staleness-test"),
        )
        assert hint_event is not None
        # The live path ran, not the cache (same signal as the gate_version test).
        assert "precomputed_gate_results" not in (hint_event.self_consistency_result or "")


def test_hint_endpoint_treats_a_cached_row_from_a_different_adapter_setting_as_a_cache_miss():
    """A cached row from a different adapter setting is a cache miss (Issue 369).

    This closes caution 3 from Issue 352 and matches the rule for gate_version, llm_model_path
    and content_fingerprint. Those three match here, so adapter_path is the only difference;
    without the check, the wrong-adapter row would be served.
    """
    question_id = _insert_test_question()
    fingerprint = compute_gate_content_fingerprint(
        _TEST_QUESTION_TEXT, _TEST_ANSWER_VALUE, False, None,
    )
    # A deliberately wrong adapter identity, guaranteed to differ from LLM_ADAPTER_PATH whether
    # that is empty or a path.
    wrong_adapter_path = f"{LLM_ADAPTER_PATH}__DEFINITELY_NOT_THE_REAL_ADAPTER_PATH"
    assert wrong_adapter_path != LLM_ADAPTER_PATH

    with get_session() as session:
        session.add(PrecomputedGateResult(
            question_id=question_id,
            gate_version=GATE_VERSION,
            llm_model_path=LLM_MODEL_PATH,
            adapter_path=wrong_adapter_path,  # the one thing that's wrong
            content_fingerprint=fingerprint,
            question_type="sum",
            tier_allowed=2,
            gate_passed=True,
            gate_reason="SymPy-confirmed (fake, precomputed test row, WRONG adapter)",
            gate_detail_json=json.dumps({"expr_str": "3/4 / 12"}),
            hint_tier=2,
            hint_text="STALE TEXT FROM A DIFFERENT ADAPTER SETTING -- MUST NEVER BE RETURNED.",
            computed_at="2026-09-21T00:00:00+00:00",
            is_active=True,
        ))
        session.flush()

    with TestClient(app) as client:
        response = client.post(
            f"/api/questions/{question_id}/hint",
            params={"request_id": "precompute-adapter-mismatch-test"},
        )

    assert response.status_code == 200
    body = response.json()
    # The live SymPy path passes for this question, giving a "hint" state and not the row cached
    # under the wrong adapter.
    assert body["state"] == "hint"
    assert body["hint_text"] != (
        "STALE TEXT FROM A DIFFERENT ADAPTER SETTING -- MUST NEVER BE RETURNED."
    )

    with get_session() as session:
        from sqlalchemy import select
        hint_event = session.scalar(
            select(HintEvent).where(HintEvent.request_id == "precompute-adapter-mismatch-test"),
        )
        assert hint_event is not None
        # The live path ran, not the cache (same signal as the other miss tests).
        assert "precomputed_gate_results" not in (hint_event.self_consistency_result or "")


def test_hint_endpoint_uses_the_correct_row_when_two_rows_differ_only_by_adapter():
    """With two rows differing only by adapter_path, the matching row is selected.

    The partial unique index from Issue 180 (history, not overwrite) forbids two active rows for
    a question, so the wrong-adapter row is inserted with `is_active=False` as superseded
    history. This checks that the right row is selected, not only that the wrong one is rejected.
    Since the first hint is always Tier 1, selection is shown by tier_allowed, grounded_in and
    verified_content on the hint_events row rather than by the cached hint_text.
    """
    question_id = _insert_test_question()
    fingerprint = compute_gate_content_fingerprint(
        _TEST_QUESTION_TEXT, _TEST_ANSWER_VALUE, False, None,
    )
    wrong_adapter_path = f"{LLM_ADAPTER_PATH}__DEFINITELY_NOT_THE_REAL_ADAPTER_PATH"

    with get_session() as session:
        session.add(PrecomputedGateResult(
            question_id=question_id,
            gate_version=GATE_VERSION,
            llm_model_path=LLM_MODEL_PATH,
            adapter_path=wrong_adapter_path,
            content_fingerprint=fingerprint,
            question_type="sum",
            tier_allowed=1,
            gate_passed=False,
            gate_reason="(superseded row, wrong adapter -- must never be served)",
            gate_detail_json=json.dumps({}),
            hint_tier=1,
            hint_text="WRONG-ADAPTER ROW -- MUST NEVER BE RETURNED.",
            computed_at="2026-09-21T00:00:00+00:00",
            is_active=False,  # history, not the active row for this question
        ))
        session.add(PrecomputedGateResult(
            question_id=question_id,
            gate_version=GATE_VERSION,
            llm_model_path=LLM_MODEL_PATH,
            adapter_path=LLM_ADAPTER_PATH,  # the configured one
            content_fingerprint=fingerprint,
            question_type="sum",
            tier_allowed=2,
            gate_passed=True,
            gate_reason="SymPy-confirmed (fake, precomputed test row, CORRECT adapter)",
            gate_detail_json=json.dumps({"expr_str": "3/4 / 12"}),
            hint_tier=2,
            hint_text="THIS EXACT TEXT PROVES THE CORRECT-ADAPTER ROW WAS SERVED.",
            computed_at="2026-09-22T00:00:00+00:00",
            is_active=True,
        ))
        session.flush()

    with TestClient(app) as client:
        response = client.post(
            f"/api/questions/{question_id}/hint",
            params={"request_id": "precompute-adapter-correct-row-test"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "hint"
    # The first hint is always Tier 1.
    assert body["tier"] == 1

    with get_session() as session:
        from sqlalchemy import select
        hint_event = session.scalar(
            select(HintEvent).where(HintEvent.request_id == "precompute-adapter-correct-row-test"),
        )
        assert hint_event is not None
        assert "precomputed_gate_results" in hint_event.self_consistency_result
        # The correct-adapter row's tier_allowed and gate_detail_json were used. The wrong-adapter
        # row has tier_allowed=1 and gate_passed=False, so these values tell the two apart.
        assert hint_event.tier_allowed == 2
        assert hint_event.grounded_in == "sympy_confirmed_expression"
        assert hint_event.verified_content == "3/4 / 12"
