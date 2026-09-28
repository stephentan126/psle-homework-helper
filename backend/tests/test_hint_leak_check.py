"""Tests for the deterministic hint answer-leak checker (Issue 357).

The checker is validated on hand-built leaking and clean cases before it is used to score
generated hints.
"""
from app.pipeline.answer_leak import check_leak, numeric_values

Q = "Ali had 160 stickers and gave away 10%. How many stickers did he have left?"


def st(ans, hint, q=Q):
    return check_leak(ans, q, hint)["status"]


def test_plain_integer_leak():
    assert st("144", "So Ali has 144 stickers left.") == "LEAK"


def test_clean_hint_with_method_only():
    assert st("144", "Try finding 10% of 160 first, then subtract it.") == "CLEAN"


def test_currency_and_commas_normalised():
    assert st("$1,250.50", "The total is 1250.5 dollars.", q="Find the total cost.") == "LEAK"
    assert st("$36", "It costs $36 in the end.", q="Find the cost.") == "LEAK"


def test_fraction_decimal_equivalence():
    assert st("1/2", "The answer is 0.5", q="Find the fraction.") == "LEAK"
    assert st("0.75", "You get 3/4 of it", q="Find the value.") == "LEAK"


def test_mixed_number_is_one_value_not_three():
    assert st("3 1/2", "Answer: 3 1/2", q="Find x.") == "LEAK"
    assert st("3 1/2", "Try adding 3 and 1 first.", q="Find x.") == "CLEAN"  # 3 and 1 alone are not 3 1/2


def test_number_words():
    assert st("36", "There are thirty-six of them.", q="Find the number.") == "LEAK"
    assert st("36", "There are thirty six of them.", q="Find the number.") == "LEAK"


def test_no_partial_digit_match():
    assert st("14", "Consider 144 and 1.4", q="Find n.") == "CLEAN"  # 14 is not inside 144 or 1.4


def test_units_do_not_hide_a_leak():
    assert st("22.5cm", "The length is 22.5 cm.", q="Find the length.") == "LEAK"
    assert st("8 h 50 min", "That is 8 h 50 min.", q="How long?") == "LEAK"  # both parts present


def test_echoed_given_is_ambiguous_not_a_leak():
    q = "There are 20 pencils. Ali buys 5 more and gives some away, leaving 20. How many did he give away?"
    r = check_leak("20", q, "Start from the 20 pencils you have.")
    assert r["status"] == "AMBIGUOUS" and r["leaked"] == []


def test_multi_part_answer_leaks_if_any_part_leaks():
    assert st("(a) 15 cm\n(b) 18", "For part (b) you should get 18.", q="Find a and b.") == "LEAK"


def test_non_numeric_answer_excluded():
    assert st("See figure", "Look at the figure.") == "NO_NUMERIC_ANSWER"


def test_numeric_values_basic():
    from fractions import Fraction
    assert numeric_values("1,000 and 2/4 and 3 1/2 and 0.25") == {Fraction(1000), Fraction(1, 2), Fraction(7, 2), Fraction(1, 4)}


def test_unit_exponent_is_not_a_spurious_value():
    """A squared or cubed unit without a superscript ("2592cm2") yields no value for the exponent.

    Without this, an unrelated list marker such as "2." in a hint was flagged as leaking the
    answer "2592cm2" (Issue 358).
    """
    from fractions import Fraction
    assert numeric_values("2592cm2") == {Fraction(2592)}
    assert numeric_values("18m3 of sand") == {Fraction(18)}
    assert st("2592cm2", "1. What information does the question give you? 2. What is being asked?", q="Find the volume.") == "CLEAN"
    # The rule only applies to a digit directly followed by cm or m and then 2 or 3, the shape
    # used in the corpus. A bare "m2" with no preceding digit is out of scope.
    # A standalone "2" must still count as a leak when the answer is "2".
    assert st("2", "The answer is 2.", q="Find x.") == "LEAK"
