"""
Tests for unit handling in `gate._values_equal()` and `strip_units()` (Issue 293).

The fix is in `gate.py` itself, since the bug affected live serving. `gate._UNIT_SUFFIX_RE` is
anchored at the end of the string by design, because corpus units are trailing suffixes
("753.6 cm"). A currency sign before the number ("$36") was therefore not covered, and '°' was
not in the suffix list. `_values_equal('36', '$36')` and `_values_equal('64', '64°')` both returned False, so
a correct answer to a dollar or angle question was rejected and the student was capped at a lower
hint tier.

The tests check that: (1) the three previously broken shapes now match (leading '$', trailing
'°', Unicode '²'/'³' superscript units); (2) shapes that already worked still match; (3) values
that differ still compare as different; (4) the narrow guard for multiple '$' signs and compound
answers is not bypassed.
"""
from __future__ import annotations

from app.pipeline.gate import _values_equal, strip_units


def test_leading_dollar_now_matches_a_correct_bare_numeric_claim() -> None:
    assert _values_equal("36", "$36") is True


def test_trailing_degree_now_matches_a_correct_bare_numeric_claim() -> None:
    assert _values_equal("64", "64°") is True


def test_unicode_superscript_unit_now_matches_the_ascii_equivalent() -> None:
    assert _values_equal("54", "54cm²") is True
    assert _values_equal("16", "16m²") is True
    assert _values_equal("40", "40m³") is True


def test_strip_units_reports_dollar_as_the_stripped_unit() -> None:
    numeric, unit = strip_units("$36")
    assert numeric == "36"
    assert unit == "$"


def test_strip_units_reports_degree_as_the_stripped_unit() -> None:
    numeric, unit = strip_units("64°")
    assert numeric == "64"
    assert unit == "°"


# --- Shapes that already worked still match ---------------------------------------------------

def test_trailing_unit_suffix_still_matches_as_before() -> None:
    assert _values_equal("72", "72cm") is True


def test_parenthesized_mcq_index_still_matches_as_before() -> None:
    assert _values_equal("3", "(3)") is True


def test_decimal_equivalence_still_holds_as_before() -> None:
    assert _values_equal("48", "48.0") is True


# --- No new false positives: different values still compare as different for each of the
# --- three newly covered shapes ---------------------------------------------------------------

def test_leading_dollar_genuinely_different_values_still_reject() -> None:
    assert _values_equal("36", "$40") is False


def test_trailing_degree_genuinely_different_values_still_reject() -> None:
    assert _values_equal("64", "65°") is False


def test_parenthesized_mcq_index_genuinely_different_still_rejects() -> None:
    assert _values_equal("3", "(4)") is False


# --- The leading-'$' strip applies only when the string has exactly one '$'. A multi-'$' or
# --- compound answer must be left untouched rather than partly stripped ---------------------

def test_leading_dollar_strip_does_not_fire_on_a_multi_dollar_compound_answer() -> None:
    numeric, unit = strip_units("$3\nb) 4/7 of money = $21")
    assert numeric == "$3\nb) 4/7 of money = $21"
    assert unit is None


def test_leading_dollar_strip_does_not_fire_when_the_string_does_not_start_with_dollar() -> None:
    numeric, unit = strip_units("(a) $52 (b) 2/9")
    assert numeric == "(a) $52 (b) 2/9"
    assert unit is None
