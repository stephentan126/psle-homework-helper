"""
Schema tests for `gate_path_used` and `is_unverified` on `HintResponse` and `CappedTier1Response`.

Also covers the rule that `CappedTier1Response` has exactly one of `question_id` and
`student_submission_id` (Issue 180 and its addendum).

These are pure Pydantic model tests with no gate, LLM or database. The validators enforce safety
invariants, such as the Tier 1 ceiling for weak mode, at construction time. A pipeline bug that
builds an impossible response (for example a weak-mode hint at tier 2) should fail at
construction, not silently downstream.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.api.schemas import CappedTier1Response, HintResponse


def test_hint_response_defaults_to_verified_match_and_not_unverified():
    """Callers that omit gate_path_used and is_unverified get valid defaults.

    Existing construction sites for corpus questions must keep working unchanged.
    """
    r = HintResponse(question_id=1, attempt_id=1, tier=2, hint_text="x")
    assert r.gate_path_used == "verified_match"
    assert r.is_unverified is False


def test_hint_response_can_never_be_weak_mode():
    """A HintResponse with gate_path_used='weak_mode' raises (Issue 180 addendum).

    Weak mode is capped at Tier 1 and a HintResponse is tier 2 or above, so this state is a bug.
    """
    with pytest.raises(ValidationError, match="weak_mode"):
        HintResponse(
            question_id=1, attempt_id=1, tier=2, hint_text="x",
            gate_path_used="weak_mode", is_unverified=True,
        )


def test_hint_response_is_unverified_must_match_gate_path_used():
    with pytest.raises(ValidationError, match="must match"):
        HintResponse(
            question_id=1, attempt_id=1, tier=2, hint_text="x",
            gate_path_used="verified_match", is_unverified=True,  # contradictory values
        )


def test_capped_tier1_verified_match_shape_unchanged_for_existing_corpus_questions():
    """The corpus-question shape (a questions.id, no student_submission_id) still works."""
    r = CappedTier1Response(question_id=42, attempt_id=1, hint_text="x", reason="disagreed")
    assert r.gate_path_used == "verified_match"
    assert r.is_unverified is False
    assert r.question_id == 42
    assert r.student_submission_id is None


def test_capped_tier1_weak_mode_real_shape():
    """A weak-mode result with only a student_submission_id (no questions.id) constructs."""
    r = CappedTier1Response(
        student_submission_id=7, attempt_id=1, hint_text="x", reason="weak mode, agreed",
        gate_path_used="weak_mode", is_unverified=True,
    )
    assert r.question_id is None
    assert r.student_submission_id == 7
    assert r.is_unverified is True


def test_capped_tier1_rejects_both_ids_set():
    with pytest.raises(ValidationError, match="exactly one"):
        CappedTier1Response(
            question_id=1, student_submission_id=2, attempt_id=1, hint_text="x",
        )


def test_capped_tier1_rejects_neither_id_set():
    with pytest.raises(ValidationError, match="exactly one"):
        CappedTier1Response(attempt_id=1, hint_text="x")


def test_capped_tier1_rejects_weak_mode_with_question_id_instead_of_submission_id():
    """A weak-mode response that sets question_id instead of student_submission_id raises.

    This guards against a subtle wiring mistake that would otherwise be served silently.
    """
    with pytest.raises(ValidationError, match="requires student_submission_id"):
        CappedTier1Response(
            question_id=1, attempt_id=1, hint_text="x",
            gate_path_used="weak_mode", is_unverified=True,
        )


def test_capped_tier1_rejects_verified_match_with_submission_id_instead_of_question_id():
    with pytest.raises(ValidationError, match="requires question_id"):
        CappedTier1Response(
            student_submission_id=1, attempt_id=1, hint_text="x",
            gate_path_used="verified_match", is_unverified=False,
        )


def test_capped_tier1_is_unverified_must_match_gate_path_used():
    with pytest.raises(ValidationError, match="must match"):
        CappedTier1Response(
            question_id=1, attempt_id=1, hint_text="x",
            gate_path_used="verified_match", is_unverified=True,
        )
