"""Tests for the live answer-leak guard.

The guard exists because a Tier 3 hint was seen stating the answer during a user-test dry run.
"""
from app.pipeline.answer_leak import LEAK_RETRY_STEERING, guard_hint

Q = "A bottle holds 1.25 litres of juice. How many litres of juice are there in 8 such bottles?"
LEAKING = {"tier": 3, "hint_text": "1.25 litres/bottle * 8 bottles = 10 litres. So there are 10 litres."}


def test_clean_hint_passes_unchanged_and_is_not_regenerated():
    calls = []
    hint = {"tier": 2, "hint_text": "Multiply the amount in one bottle by the number of bottles."}
    out = guard_hint(hint, ["10"], Q, lambda steering: calls.append(steering))
    assert out["hint_text"] == hint["hint_text"]
    assert out["leak_guard"] == "clean"
    assert calls == []


def test_leaking_hint_is_regenerated_once_with_steering():
    calls = []

    def regenerate(steering):
        calls.append(steering)
        return {"tier": 3, "hint_text": "Here is a similar question: 2 boxes hold 1.5 kg each. 2 x 1.5 = 3 kg."}

    out = guard_hint(LEAKING, ["10"], Q, regenerate)
    assert calls == [LEAK_RETRY_STEERING]
    assert out["leak_guard"] == "regenerated"
    assert "10" not in out["hint_text"]


def test_hint_that_still_leaks_is_replaced_with_a_safe_prompt():
    out = guard_hint(LEAKING, ["10"], Q, lambda steering: dict(LEAKING))
    assert out["leak_guard"] == "replaced"
    assert out["tier"] == 3
    assert "10" not in out["hint_text"]


def test_regeneration_failure_falls_back_to_the_safe_prompt():
    def broken(steering):
        raise RuntimeError("model unavailable")

    out = guard_hint(LEAKING, ["10"], Q, broken)
    assert out["leak_guard"] == "replaced"


def test_number_that_is_also_in_the_question_is_not_treated_as_a_leak():
    hint = {"tier": 1, "hint_text": "The question gives you 8 bottles."}
    out = guard_hint(hint, ["8"], "There are 8 bottles. How many bottles are there?", lambda s: None)
    assert out["leak_guard"] == "clean"


def test_no_known_answer_means_nothing_to_check():
    out = guard_hint(dict(LEAKING), [], Q, lambda s: None)
    assert out["leak_guard"] == "clean"


def test_tidy_drops_an_unfinished_last_sentence_and_splits_steps():
    from app.pipeline.answer_leak import tidy_hint_text

    text = "Here is an example. Step 1: Find the packs. Step 2: Multiply the number of"
    assert tidy_hint_text(text) == "Here is an example.\nStep 1: Find the packs."


def test_tidy_keeps_decimals_and_text_without_a_full_stop():
    from app.pipeline.answer_leak import tidy_hint_text

    assert tidy_hint_text("Each bottle holds 1.25 litres.") == "Each bottle holds 1.25 litres."
    assert tidy_hint_text("Try multiplying") == "Try multiplying"
