"""
Tests for the Tier 3 path in `generate_hint()` and the `hint_events` persistence columns.

The columns are `tier_allowed`, `grounded_in` and `verified_content`. Related issues: 36, 37
and the Issue 180 addendum. There are two kinds of test:

1. Fast, GPU-free tests with `gate.generate` and `gate.find_related_worked_example`
   monkeypatched, as in `test_diagram_degrade_path.py`. They cover the `tier_allowed == 4`
   gating rule, the `want_tier3=False` no-op contract and each degrade case (`question_id=None`,
   no qualifying candidate).
2. One unmocked end-to-end test, using computed embeddings as in `test_matching_and_weak_mode.py`
   and a Phi-4-mini generation. It checks that the current question's answer never appears in a
   Tier 3 hint, like the `"1/16" not in body["hint_text"]` check in `test_hint_endpoint.py`. It is
   not faked, because a fake `generate()` would pass a leak check without exercising the model
   on the Tier 3 prompt.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from app.db.session import get_session
from app.models.question import Question
from app.models.question_embedding import QuestionEmbedding
from app.pipeline import gate
from app.pipeline.gate import GateResult, generate_hint
from app.pipeline.question_matching import EMBEDDING_MODEL_ID, RelatedWorkedExample, _get_model


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# =================================================================================================
# Fast, GPU-free branching/gating tests.
# =================================================================================================


def _capture_generate(monkeypatch):
    calls: list[dict] = []

    def _fake_generate(system_prompt, user_prompt, max_new_tokens=200, do_sample=False, **kw):
        calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        return "FAKE HINT TEXT -- NOT A REAL GENERATION"

    monkeypatch.setattr(gate, "generate", _fake_generate)
    return calls


def test_want_tier3_false_is_byte_identical_to_before_this_brief(monkeypatch):
    """Callers that do not pass want_tier3 see no change in behaviour.

    As for Issue 56, find_related_worked_example is patched to raise if called, because the
    retrieval must not be attempted when want_tier3=False.
    """

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("find_related_worked_example() must never be called when want_tier3=False")

    monkeypatch.setattr(gate, "find_related_worked_example", _must_not_be_called)
    _capture_generate(monkeypatch)

    gate_result = GateResult(passed=True, tier_allowed=4, reason="(test)", detail={"expr_str": "2+2"})
    hint = generate_hint("Plain question.", gate_result, worked_solution_text="2+2=4")

    assert hint["tier"] == 2
    assert hint["tier_allowed"] == 4
    assert hint["tier3_degrade_reason"] is None


def test_tier_allowed_2_never_reaches_tier3_even_when_want_tier3_true(monkeypatch):
    """Tier 3 is reachable only when tier_allowed == 4.

    A tier_allowed == 2 case (method mismatch, Issue 37) stays at Tier 2 even with
    want_tier3=True. find_related_worked_example is patched to raise, which shows the retrieval
    is never attempted.
    """

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("find_related_worked_example() must never run for tier_allowed == 2")

    monkeypatch.setattr(gate, "find_related_worked_example", _must_not_be_called)
    _capture_generate(monkeypatch)

    gate_result = GateResult(
        passed=True, tier_allowed=2, reason="(test, method mismatch)", detail={"expr_str": "2+2"},
    )
    hint = generate_hint(
        "Plain question.", gate_result, worked_solution_text="2+2=4",
        want_tier3=True, question_id=42,
    )

    assert hint["tier"] == 2
    assert hint["tier_allowed"] == 2


def test_tier1_never_reaches_tier3_even_when_want_tier3_true(monkeypatch):
    """tier_allowed == 1 (SymPy or self-consistency failed) stays at Tier 1, even with want_tier3."""

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("find_related_worked_example() must never run for tier_allowed == 1")

    monkeypatch.setattr(gate, "find_related_worked_example", _must_not_be_called)
    _capture_generate(monkeypatch)

    gate_result = GateResult(passed=False, tier_allowed=1, reason="(test)", detail={})
    hint = generate_hint(
        "Plain question.", gate_result, want_tier3=True, question_id=42,
    )

    assert hint["tier"] == 1
    assert hint["tier_allowed"] == 1


def test_no_question_id_degrades_honestly_to_tier2_with_a_real_logged_reason(monkeypatch):
    """Without a question_id, Tier 3 degrades to the computed Tier 2 result.

    There is no question to exclude from the grounding search, so it must neither invent a
    Tier 3 hint nor crash.
    """

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("find_related_worked_example() must never run with question_id=None")

    monkeypatch.setattr(gate, "find_related_worked_example", _must_not_be_called)
    calls = _capture_generate(monkeypatch)

    gate_result = GateResult(passed=True, tier_allowed=4, reason="(test)", detail={"expr_str": "2+2"})
    hint = generate_hint(
        "Plain question.", gate_result, worked_solution_text="2+2=4",
        want_tier3=True, question_id=None,
    )

    assert hint["tier"] == 2
    assert hint["hint_text"] == "FAKE HINT TEXT -- NOT A REAL GENERATION"
    assert hint["tier3_degrade_reason"] == "no_question_id_supplied"
    assert len(calls) == 1  # only the Tier 2 generation call, no wasted second call


def test_no_qualifying_related_example_degrades_honestly_to_tier2_with_a_real_logged_reason(monkeypatch):
    """If retrieval finds no candidate, Tier 3 degrades to Tier 2.

    This happens for a novel or sparse topic. No 'no example available' filler is produced.
    """
    monkeypatch.setattr(
        gate, "find_related_worked_example",
        lambda question_id, exclude_question_ids=None: RelatedWorkedExample(
            question_id=None, match_confidence=None, worked_solution_text=None,
            match_method="no_real_candidate_worked_example_available",
        ),
    )
    calls = _capture_generate(monkeypatch)

    gate_result = GateResult(passed=True, tier_allowed=4, reason="(test)", detail={"expr_str": "2+2"})
    hint = generate_hint(
        "Plain question.", gate_result, worked_solution_text="2+2=4",
        want_tier3=True, question_id=99,
    )

    assert hint["tier"] == 2
    assert "no_qualifying_related_example" in hint["tier3_degrade_reason"]
    assert len(calls) == 1


def test_below_min_similarity_degrades_even_though_a_candidate_was_found(monkeypatch):
    """A candidate below RELATED_EXAMPLE_MIN_SIMILARITY still degrades to Tier 2.

    Argmax always returns something, so the floor stops a poorly related example being served.
    """
    monkeypatch.setattr(
        gate, "find_related_worked_example",
        lambda question_id, exclude_question_ids=None: RelatedWorkedExample(
            question_id=7, match_confidence=0.1, worked_solution_text="Unrelated solution.",
            match_method="related_worked_example_cosine_similarity_test",
        ),
    )
    calls = _capture_generate(monkeypatch)

    gate_result = GateResult(passed=True, tier_allowed=4, reason="(test)", detail={"expr_str": "2+2"})
    hint = generate_hint(
        "Plain question.", gate_result, worked_solution_text="2+2=4",
        want_tier3=True, question_id=99,
    )

    assert hint["tier"] == 2
    assert "no_qualifying_related_example" in hint["tier3_degrade_reason"]
    assert len(calls) == 1


def test_qualifying_related_example_produces_a_real_tier3_hint(monkeypatch):
    """A qualifying candidate yields a Tier 3 hint with grounded_in, tier_allowed and
    verified_content set."""
    monkeypatch.setattr(
        gate, "find_related_worked_example",
        lambda question_id, exclude_question_ids=None: RelatedWorkedExample(
            question_id=7, match_confidence=0.95,
            worked_solution_text="Step 1: ... Step 2: ... Final answer: 18.",
            match_method="related_worked_example_cosine_similarity_test",
        ),
    )
    calls = _capture_generate(monkeypatch)

    gate_result = GateResult(passed=True, tier_allowed=4, reason="(test)", detail={"expr_str": "24-24*8/15"})
    hint = generate_hint(
        "Ali has 24 marbles. He gives away 8/15 of them. How many does he have left?", gate_result,
        want_tier3=True, question_id=99,
    )

    assert hint["tier"] == 3
    assert hint["grounded_in"] == "related_worked_example"
    assert hint["tier_allowed"] == 4
    assert hint["tier3_degrade_reason"] is None
    assert hint["verified_content"] == "related_question_id=7"
    assert len(calls) == 1
    assert "Step 1: ... Step 2: ... Final answer: 18." in calls[0]["user_prompt"] or (
        "Step 1: ... Step 2: ... Final answer: 18." in calls[0]["system_prompt"]
    )


# =================================================================================================
# Unmocked end-to-end leak check, mirroring the Tier 2 "'1/16' not in body['hint_text']" check.
# =================================================================================================

_CURRENT_QUESTION_TEXT = (
    "Ali has 24 marbles. He gives away 8/15 of them. How many marbles does he have left?"
)
_CURRENT_QUESTION_ANSWER = "16"  # must never appear in the Tier 3 hint text

_RELATED_QUESTION_TEXT = (
    "Mary has 30 stickers. She gives away 2/5 of them. How many stickers does she have left?"
)
_RELATED_QUESTION_WORKED_SOLUTION = (
    "2/5 of 30 stickers = 12 stickers given away. 30 - 12 = 18 stickers left."
)


def _insert_real_question_with_real_embedding(
    question_text: str, answer_value: str, worked_solution_text, source_page_index: int,
) -> int:
    """Insert a question and its computed embedding, as the precompute script would.

    Follows `test_matching_and_weak_mode.py`: `_get_model().encode()` plus a `QuestionEmbedding`
    row, as written by `scripts/extraction/precompute_question_embeddings.py`.
    """
    with get_session() as session:
        question = Question(
            source_paper="data/School papers/2024/P6_Maths_2024_WA2_nanhua.pdf",
            source_page_index=source_page_index, source_page_label=None,
            answer_source_file="data/School papers/2024/P6_Maths_2024_WA2_nanhua.pdf",
            answer_source_page_index=source_page_index, question_number=str(source_page_index),
            question_text=question_text, answer_value=answer_value,
            worked_solution_text=worked_solution_text, has_diagram=False,
            ocr_confidence="high", school_name="Nan Hua Primary School",
        )
        session.add(question)
        session.flush()
        question_id = question.id

        model = _get_model()
        embedding = model.encode([question_text], convert_to_numpy=True)[0]
        session.add(QuestionEmbedding(
            question_id=question_id, embedding_model=EMBEDDING_MODEL_ID,
            embedding_json=json.dumps(embedding.tolist()), computed_at=_now_iso(),
        ))
        return question_id


def test_real_tier3_hint_never_leaks_the_current_questions_own_real_answer():
    """A generated Tier 3 hint never contains the current question's answer.

    Uses a SymPy-confirmed current question (tier_allowed == 4), a different related question
    with a worked solution and computed embedding, and a Phi-4-mini generation. The core safety
    rule is that no final answer is revealed without the parent-PIN unlock path.
    """
    current_qid = _insert_real_question_with_real_embedding(
        _CURRENT_QUESTION_TEXT, _CURRENT_QUESTION_ANSWER, None, source_page_index=101,
    )
    _insert_real_question_with_real_embedding(
        _RELATED_QUESTION_TEXT, "18", _RELATED_QUESTION_WORKED_SOLUTION, source_page_index=102,
    )

    gate_result = GateResult(
        passed=True, tier_allowed=4, reason="(real SymPy pass)",
        detail={"expr_str": "24 - 24*8/15"},
    )
    hint = generate_hint(
        _CURRENT_QUESTION_TEXT, gate_result, want_tier3=True, question_id=current_qid,
    )

    # The retrieval found and used the related question instead of degrading.
    assert hint["tier"] == 3, (
        f"Expected a real Tier 3 hint (a real, related candidate exists and should clear "
        f"RELATED_EXAMPLE_MIN_SIMILARITY) -- got tier={hint['tier']}, "
        f"degrade_reason={hint.get('tier3_degrade_reason')!r}"
    )
    assert hint["grounded_in"] == "related_worked_example"

    # The safety assertion: the current question's answer must not appear, as in the
    # "1/16" not in body["hint_text"] check in test_hint_endpoint.py.
    assert _CURRENT_QUESTION_ANSWER not in hint["hint_text"]
