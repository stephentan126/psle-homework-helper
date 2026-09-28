"""Tests for the corpus consistency checker (Issue 361).

Each case is a pattern taken from the corpus. A spot check of the checker's first output showed
most of these wrongly flagged as MISMATCH, so they are pinned here before its aggregate count is
trusted.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "data_quality" / "corpus_verification"))
from check_consistency import check_row  # noqa: E402


def test_plain_match_last_equals():
    assert check_row("890", "200÷20=10\n10x5=50\n10x9=90\n90+50=140\n1500÷10=150\n150x5=750\n750+140=890")["status"] == "MATCH"


def test_ans_marker_preferred_over_an_earlier_intermediate_equals():
    # id=294: the last '=' is an intermediate result (65R12); the final answer follows ANS:
    assert check_row("12", "1962 ÷ 30 = 65R12\nANS: 12")["status"] == "MATCH"


def test_ans_marker_with_no_equals_sign_at_all():
    # id=4717: 'Ans: U' has no '=' before it, so the earlier '=' value must not be used
    assert check_row("U", "99÷6=16R3\nAns: U")["status"] == "MATCH"


def test_glued_trailing_ans_after_equals_does_not_break_the_parse():
    # id=4293: in "...= $149. ANS : $149" the capture must not include the glued 'ANS :' text
    assert check_row("$149", "8 x $18 + 2 x $2.50 = $144 + $5 = $149. ANS : $149")["status"] == "MATCH"


def test_trailing_parenthetical_remark_is_stripped_not_left_to_break_the_parse():
    # id=4957-style: a reason clause glued on with no newline before the next part
    assert check_row("245", "∠x = reflex ∠TQU = 360 - 115 = 245. (∠sum at a point)")["status"] == "MATCH"


def test_real_numeric_mismatch_is_still_caught():
    assert check_row("999", "10 x 5 = 50\n50 + 40 = 90")["status"] == "MISMATCH"


def test_compound_answer_matched_via_body_segments():
    assert check_row(
        "(a) 245°\n(b) 50°",
        "(a) ∠x = 245°. (∠sum at a point)\n(b) ∠y = 180 - 65 - 65 = 50°. (∠sum of a straight line)",
    )["status"] == "MATCH"


def test_compound_answer_matched_via_trailing_consolidated_ans_block():
    # id=4957: per-part reasoning with trailing remarks, plus a consolidated ANS: block at the end
    assert check_row(
        "(a) 245°\n(b) 50°",
        "(a) ∠x = reflex ∠TQU = 360° - 115° = 245°. (∠sum at a point)\n"
        "(b) ∠QUV = 180° - 115° = 65°. (interior ∠s, QT//UV)\n    ∠y = 180° - 65° - 65° = 50°. (∠sum of a straight line)\n"
        "ANS: (a) 245°\n(b) 50°",
    )["status"] == "MATCH"


def test_compound_answer_real_mismatch_on_one_part_still_flags():
    r = check_row("(a) 245°\n(b) 999°", "(a) x = 360° - 115° = 245°.\n(b) y = 180° - 65° - 65° = 50°.")
    assert r["status"] == "MISMATCH" and r["per_part"]["b"] == "MISMATCH" and r["per_part"]["a"] == "MATCH"


def test_unparseable_when_no_signal_found():
    assert check_row("42", "Draw a bar model and think carefully about the relationships shown.")["status"] == "UNPARSEABLE"


def test_single_part_answer_with_a_label_but_no_second_label_in_answer():
    # id=4479: answer_value is "(a) 12.5 cm" alone, with no "(b)" in the answer or the solution
    assert check_row("(a) 12.5 cm", "(a) Height = 48000 ÷ 4800 = 10 cm.\n∴ Height of the container = 10 × 5/4 = 12.5 cm.")["status"] == "MATCH"


def test_trailing_period_on_the_stored_answer_value_itself_is_cleaned():
    # id=5016: the stored answer_value itself ends in a period ("45°.")
    assert check_row("45°.", "∠EAD = 90° + 60° = 150°,\n∠AED = ∠ADE = (180° - 150°) ÷ 2 = 15°,\n∴∠BED = 60° - 15° = 45°.")["status"] == "MATCH"


def test_trailing_ans_annotation_on_the_stored_answer_value_itself_is_cleaned():
    # id=3413: the stored answer_value ends in an "(Ans)" annotation
    assert check_row("59km/h (Ans)", "(a) 209 ÷ 76 = 2.75 hours (Ans)\n(b) Ave Speed = 153.4 ÷ 2.6 = 59km/h (Ans)")["status"] == "MATCH"


def test_boolean_literal_exact_match_not_broken_by_sympy_tuple_quirk():
    # id=1950: gate._values_equal parses "True" to a SymPy boolean, which fails nsimplify()
    assert check_row("True", "a) angle = 116 degrees\nABD is an equilateral triangle = True")["status"] == "MATCH"


def test_comma_separated_list_exact_match_not_broken_by_sympy_tuple_quirk():
    # id=4521/2546: "24, 48" parses to a SymPy Tuple, which fails nsimplify() even against an
    # identical Tuple
    assert check_row("24, 48", "6 12 18 24 30 36 42 48\n8 16 24 32 40 48\nAns: 24, 48")["status"] == "MATCH"


def test_value_bearing_trailing_parenthesis_is_not_eaten():
    # id=1544/1422: "$(4d - 60)" is the answer, not an annotation, so it must not become "$"
    assert check_row("$(4d - 60)", "14d - 5d - (5d + 60)\n= 4d - 60\n= $(4d - 60)")["status"] == "MATCH"


def test_trailing_period_after_a_digit_is_still_a_sentence_period_not_a_decimal():
    # id=2558/3132: "$52." and "(5f-24)/3." end in a digit and a full stop with nothing after,
    # which is always a sentence ending, not a decimal point
    r = check_row("(a) $52 (b) 2/9", "(a) least spent = $52. (b) Alice's fraction = 52/234 = 2/9.")
    assert r["status"] == "MATCH" and r["per_part"] == {"a": "MATCH", "b": "MATCH"}
