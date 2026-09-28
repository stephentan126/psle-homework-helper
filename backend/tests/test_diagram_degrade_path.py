"""
Tests for the diagram-interpreter degrade path (Issue 56; design specification Sections 6 and 10.5,
and Section 15 Phase 7 step 5).

The fallback is built before any Section 6 diagram model exists, since Phase 7 has not started.
When a diagram model fails to load, times out or returns low confidence, the pipeline falls back
to the text-only path (the Section 6 rule "never silently guess") and never produces a hint that
treats an unread diagram as decorative (Issue 14). Three layers are tested, most isolated first:

1. `diagram_fallback.resolve_diagram_context()`, the wrapper itself. Each failure mode (load
   failure, timeout, low confidence, unexpected crash) is forced directly, and the current
   unmocked behaviour, with no diagram model available, is also checked.
2. `gate.generate_hint()`, to show the wrapper is wired in. A has_diagram=True hint carries the
   acknowledgment text. For has_diagram=False, `resolve_diagram_context` is monkeypatched to raise
   if called, so an accidental call fails the test instead of relying on a value comparison that
   could pass by coincidence.
3. `gate.run_full_gate_pipeline()`, to show `has_diagram` is passed from the entry point through to
   `generate_hint()`.

All tests are fast and GPU-free. `llm_service.generate` and `generate_multiple` are never called;
the names bound in the `gate` module namespace are monkeypatched, as in
`test_harassment_retry_regeneration.py`.
"""
from __future__ import annotations


from app.pipeline import diagram_fallback, gate
from app.pipeline.diagram_fallback import (
    DIAGRAM_UNAVAILABLE_ACKNOWLEDGMENT,
    DiagramContext,
    resolve_diagram_context,
)
from app.pipeline.gate import GateResult, generate_hint, run_full_gate_pipeline
from app.services.diagram_service import (
    MIN_TRUSTED_DIAGRAM_CONFIDENCE,
    DiagramInterpretationResult,
    DiagramInterpretationTimeoutError,
    DiagramInterpreterUnavailableError,
)

_FAKE_IMAGE = b"\x89PNG-fake-test-bytes-not-a-real-image"


# =================================================================================================
# Layer 1: resolve_diagram_context() itself, with each degrade mode forced directly.
# =================================================================================================


def test_has_diagram_false_never_touches_diagram_service(monkeypatch):
    """With has_diagram=False, the diagram interpreter is never called.

    The interpreter is replaced with a function that raises, so any call fails immediately.
    """

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("interpret_diagram() must never be called when has_diagram=False")

    monkeypatch.setattr(diagram_fallback, "interpret_diagram", _must_not_be_called)

    result = resolve_diagram_context(has_diagram=False)
    assert result == DiagramContext(available=False)
    assert result.acknowledgment is None


def test_has_diagram_true_no_image_supplied_degrades_honestly():
    """With no image supplied, resolve_diagram_context() degrades cleanly instead of failing.

    No caller extracts a per-question diagram crop yet, because Phase 7 has not started.
    """
    result = resolve_diagram_context(has_diagram=True, diagram_image=None)
    assert result.available is False
    assert result.acknowledgment == DIAGRAM_UNAVAILABLE_ACKNOWLEDGMENT


def test_real_unmocked_interpret_diagram_degrades_today():
    """The unmocked `interpret_diagram()` raises and resolve_diagram_context() catches it.

    No diagram model is built yet (see docs/DEVELOPMENT_LOG.md), so `interpret_diagram()` always
    raises DiagramInterpreterUnavailableError.
    """
    result = resolve_diagram_context(has_diagram=True, diagram_image=_FAKE_IMAGE)
    assert result.available is False
    assert result.acknowledgment == DIAGRAM_UNAVAILABLE_ACKNOWLEDGMENT
    assert result.structured_data is None


def test_forced_load_failure_falls_back_not_errors(monkeypatch):
    def _fails_to_load(diagram_image, timeout_s=30.0):
        raise DiagramInterpreterUnavailableError("forced test failure: model failed to load")

    monkeypatch.setattr(diagram_fallback, "interpret_diagram", _fails_to_load)

    result = resolve_diagram_context(has_diagram=True, diagram_image=_FAKE_IMAGE)
    assert result.available is False
    assert result.acknowledgment == DIAGRAM_UNAVAILABLE_ACKNOWLEDGMENT


def test_forced_timeout_falls_back_not_errors(monkeypatch):
    def _times_out(diagram_image, timeout_s=30.0):
        raise DiagramInterpretationTimeoutError("forced test failure: timed out")

    monkeypatch.setattr(diagram_fallback, "interpret_diagram", _times_out)

    result = resolve_diagram_context(has_diagram=True, diagram_image=_FAKE_IMAGE)
    assert result.available is False
    assert result.acknowledgment == DIAGRAM_UNAVAILABLE_ACKNOWLEDGMENT


def test_forced_low_confidence_falls_back_not_served_as_a_guess(monkeypatch):
    def _low_confidence(diagram_image, timeout_s=30.0):
        return DiagramInterpretationResult(
            structured_data={"boxes": ["A", "B"]},
            confidence=MIN_TRUSTED_DIAGRAM_CONFIDENCE - 0.01,
        )

    monkeypatch.setattr(diagram_fallback, "interpret_diagram", _low_confidence)

    result = resolve_diagram_context(has_diagram=True, diagram_image=_FAKE_IMAGE)
    assert result.available is False
    assert result.acknowledgment == DIAGRAM_UNAVAILABLE_ACKNOWLEDGMENT
    # A low-confidence guess must not be passed on as usable data.
    assert result.structured_data is None


def test_forced_unexpected_crash_falls_back_never_propagates(monkeypatch):
    """An unexpected exception from the diagram model falls back instead of propagating.

    This is a fail-closed catch-all, following the same approach as the safety classifier
    (Issue 210).
    """

    def _crashes(diagram_image, timeout_s=30.0):
        raise RuntimeError("forced test failure: unexpected crash, not a named failure mode")

    monkeypatch.setattr(diagram_fallback, "interpret_diagram", _crashes)

    result = resolve_diagram_context(has_diagram=True, diagram_image=_FAKE_IMAGE)
    assert result.available is False
    assert result.acknowledgment == DIAGRAM_UNAVAILABLE_ACKNOWLEDGMENT


def test_high_confidence_result_is_actually_used(monkeypatch):
    """A high-confidence result is returned as available rather than degraded.

    This is the expected path once a diagram model exists.
    """

    def _confident(diagram_image, timeout_s=30.0):
        return DiagramInterpretationResult(
            structured_data={"box_a": 12, "box_b": "unknown", "total": 20},
            confidence=0.99,
        )

    monkeypatch.setattr(diagram_fallback, "interpret_diagram", _confident)

    result = resolve_diagram_context(has_diagram=True, diagram_image=_FAKE_IMAGE)
    assert result.available is True
    assert result.acknowledgment is None
    assert result.structured_data == {"box_a": 12, "box_b": "unknown", "total": 20}


# =================================================================================================
# Layer 2: generate_hint() wiring. The acknowledgment reaches the prompt, and has_diagram=False is
# a no-op.
# =================================================================================================


def _capture_generate(monkeypatch):
    calls: list[dict] = []

    def _fake_generate(system_prompt, user_prompt, max_new_tokens=200, do_sample=False, **kw):
        calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        return "FAKE HINT TEXT -- NOT A REAL GENERATION"

    monkeypatch.setattr(gate, "generate", _fake_generate)
    return calls


def test_generate_hint_tier1_with_diagram_failure_includes_acknowledgment_not_a_crash(monkeypatch):
    calls = _capture_generate(monkeypatch)
    gate_result = GateResult(passed=False, tier_allowed=1, reason="(test)", detail={})

    hint = generate_hint(
        "A figure shows two boxes. What is the total?", gate_result, has_diagram=True,
    )

    assert hint["tier"] == 1
    assert hint["hint_text"] == "FAKE HINT TEXT -- NOT A REAL GENERATION"
    assert len(calls) == 1
    assert DIAGRAM_UNAVAILABLE_ACKNOWLEDGMENT in calls[0]["system_prompt"]


def test_generate_hint_tier2_with_diagram_failure_includes_acknowledgment_not_a_crash(monkeypatch):
    calls = _capture_generate(monkeypatch)
    gate_result = GateResult(
        passed=True, tier_allowed=2, reason="(test)", detail={"expr_str": "12 + 8"},
    )

    hint = generate_hint(
        "A figure shows two boxes. What is the total?", gate_result,
        worked_solution_text="Box A = 12, Box B = 8, total = 20", has_diagram=True,
    )

    assert hint["tier"] == 2
    assert hint["hint_text"] == "FAKE HINT TEXT -- NOT A REAL GENERATION"
    assert len(calls) == 1
    assert DIAGRAM_UNAVAILABLE_ACKNOWLEDGMENT in calls[0]["system_prompt"]


def test_generate_hint_has_diagram_true_never_raises_via_real_gate_binding(monkeypatch):
    """generate_hint() handles a degraded diagram context through `gate.resolve_diagram_context`.

    The patch targets `gate.resolve_diagram_context` because gate.py binds the name at import
    time; patching `diagram_fallback.resolve_diagram_context` would miss it. This covers the same
    case as `test_forced_unexpected_crash_falls_back_never_propagates`, but end to end through
    `generate_hint()`.
    """
    _capture_generate(monkeypatch)

    def _always_degrades(has_diagram, diagram_image=None):
        assert has_diagram is True
        return DiagramContext(available=False, acknowledgment=DIAGRAM_UNAVAILABLE_ACKNOWLEDGMENT)

    monkeypatch.setattr(gate, "resolve_diagram_context", _always_degrades)

    gate_result = GateResult(passed=False, tier_allowed=1, reason="(test)", detail={})
    hint = generate_hint("A question with a figure.", gate_result, has_diagram=True)

    assert hint["hint_text"] == "FAKE HINT TEXT -- NOT A REAL GENERATION"


def test_generate_hint_has_diagram_false_is_byte_identical_and_never_calls_resolve(monkeypatch):
    """Text-only questions are unaffected by the diagram fallback.

    This is checked two ways: (1) resolve_diagram_context is monkeypatched to raise if called, so
    has_diagram=False must never reach it; (2) the prompt must equal the unaugmented template
    exactly.
    """

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError(
            "resolve_diagram_context() must never be called when has_diagram=False",
        )

    monkeypatch.setattr(gate, "resolve_diagram_context", _must_not_be_called)
    calls = _capture_generate(monkeypatch)

    # Tier 1 branch.
    tier1_result = GateResult(passed=False, tier_allowed=1, reason="(test)", detail={})
    hint1 = generate_hint("Plain text-only question, no diagram.", tier1_result)
    # generate_hint() also returns "tier_allowed", "verified_content", "tier3_degrade_reason" and
    # "tier3_similarity_score" (persistence and escalation fields; see its docstring). The check
    # lists them explicitly rather than using a subset check, so any other change to the dict for
    # has_diagram=False still fails the test.
    assert hint1 == {
        "tier": 1, "hint_text": "FAKE HINT TEXT -- NOT A REAL GENERATION", "grounded_in": None,
        "tier_allowed": 1, "verified_content": None, "tier3_degrade_reason": None,
        "tier3_similarity_score": None,
    }
    assert calls[0]["system_prompt"] == gate._TIER1_SYSTEM_PROMPT

    # Tier 2+ branch.
    calls.clear()
    tier2_result = GateResult(
        passed=True, tier_allowed=2, reason="(test)", detail={"expr_str": "3 + 4"},
    )
    hint2 = generate_hint(
        "Plain text-only question, no diagram.", tier2_result,
        worked_solution_text="3 + 4 = 7",
    )
    assert hint2["tier"] == 2
    assert hint2["hint_text"] == "FAKE HINT TEXT -- NOT A REAL GENERATION"
    assert calls[0]["system_prompt"] == gate._GROUNDED_HINT_PROMPT_TEMPLATE.format(
        verified_content="3 + 4 = 7",
    )


# =================================================================================================
# Layer 3: run_full_gate_pipeline() wiring. has_diagram reaches generate_hint().
# =================================================================================================


def test_run_full_gate_pipeline_threads_has_diagram_true_to_generate_hint(monkeypatch):
    fake_gate_result = GateResult(
        passed=True, tier_allowed=1, reason="(test, deliberately capped)", detail={},
    )
    monkeypatch.setattr(gate, "self_consistency_check", lambda *a, **kw: fake_gate_result)

    received: dict = {}

    def _fake_generate_hint(question_text, gate_result, worked_solution_text=None, **kwargs):
        received.update(kwargs)
        received["worked_solution_text"] = worked_solution_text
        return {"tier": 1, "hint_text": "(test hint)", "grounded_in": None}

    monkeypatch.setattr(gate, "generate_hint", _fake_generate_hint)

    run_full_gate_pipeline(
        "A figure shows a rectangle.", "20", has_diagram=True, worked_solution_text=None,
    )

    assert received.get("has_diagram") is True


def test_run_full_gate_pipeline_threads_has_diagram_false_to_generate_hint(monkeypatch):
    fake_gate_result = GateResult(passed=True, tier_allowed=1, reason="(test)", detail={})
    monkeypatch.setattr(gate, "sympy_exact_check", lambda *a, **kw: fake_gate_result)

    received: dict = {}

    def _fake_generate_hint(question_text, gate_result, worked_solution_text=None, **kwargs):
        received.update(kwargs)
        return {"tier": 1, "hint_text": "(test hint)", "grounded_in": None}

    monkeypatch.setattr(gate, "generate_hint", _fake_generate_hint)

    run_full_gate_pipeline(
        "What is 3 + 4?", "7", has_diagram=False, worked_solution_text=None,
    )

    assert received.get("has_diagram") is False
