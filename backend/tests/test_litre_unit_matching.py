"""
Tests for litre-unit handling in `gate.strip_units()` (Issue 344).

The single-letter litre units ('ℓ', 'l', 'L') were not recognised, so a correct bare answer ('6')
did not match a gold answer carrying the unit ('6ℓ') and fell through to the literal-string
fallback. This was seen in a Qwen2.5-VL-7B evaluation run (gold id=1745). In the corpus, 30 of
3,927 `answer_value` rows end in a digit followed by 'L', 'l' or 'ℓ'.

The tests check that: (1) the litre shapes now match; (2) the shapes that already worked still
match; (3) different values still reject, so no new false positives; (4) the unit must follow a
digit, so a trailing 'l' on a non-numeric answer is left alone.
"""
from __future__ import annotations

from app.pipeline.gate import _values_equal, strip_units


def test_litre_symbol_now_matches_a_correct_bare_numeric_claim() -> None:
    assert _values_equal("6", "6ℓ") is True
    assert _values_equal("6", "6 ℓ") is True


def test_capital_and_lowercase_L_now_match() -> None:
    assert _values_equal("4", "4L") is True
    assert _values_equal("1.25", "1.25l") is True
    assert _values_equal("0.864", "0.864 L") is True


def test_strip_units_reports_the_litre_symbol_as_the_stripped_unit() -> None:
    assert strip_units("6ℓ") == ("6", "ℓ")
    assert strip_units("21.6L") == ("21.6", "L")
    assert strip_units("1.25 l") == ("1.25", "l")


def test_litre_gold_still_matches_when_the_claim_also_carries_the_unit() -> None:
    assert _values_equal("6ℓ", "6ℓ") is True
    assert _values_equal("6 L", "6ℓ") is True


def test_genuinely_different_litre_values_still_reject() -> None:
    assert _values_equal("5", "6ℓ") is False
    assert _values_equal("60", "6L") is False


def test_spelled_out_and_ml_units_are_unchanged() -> None:
    assert strip_units("6 litre") == ("6", "litre")
    assert strip_units("500ml") == ("500", "ml")
    assert strip_units("500 mL") == ("500", "mL")


def test_trailing_letter_l_on_a_non_numeric_answer_is_left_alone() -> None:
    for word in ("Apple", "ball", "Nail", "total"):
        assert strip_units(word) == (word, None), word


def test_a_letter_l_after_a_non_digit_is_not_treated_as_a_unit() -> None:
    assert strip_units("x = l") == ("x = l", None)


def test_previously_working_units_still_behave_as_before() -> None:
    assert strip_units("753.6 cm") == ("753.6", "cm")
    assert strip_units("330g") == ("330", "g")
    assert strip_units("64°") == ("64", "°")
    assert strip_units("$36") == ("36", "$")
    assert strip_units("48") == ("48", None)
