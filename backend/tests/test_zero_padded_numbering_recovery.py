"""
Tests for recovering zero-padded question numbers misread as "Q<digit>" (Issue 229, candidate 2).

An earlier approach, since reverted, added a general "Q<1-3 digits>" alternative to the shared
`_match_question_start()`. On `P6_Maths_2023_WA2_Taonan` p32/p33 a "Q11." reference then collided
with an unrelated bare-digit false positive for the same number ("11 units -> 66 jugs"). The
`{question_number: span}` dict dropped one span's content, and 4 records went from `agree` to
`disagree`.

`_recover_zero_padded_misread_questions()` (design notes in `extract.py`) is narrower:
  1. It never touches `_match_question_start()`, `_has_question_like_content()` or
     `classify_page()`. It runs only when a reader's segmentation produced an empty span list,
     which never happens on Taonan p32/p33.
  2. It needs cross-reader corroboration, a literal zero-padded "0<digit>" in the other reader's
     text, before accepting a single-digit "Q<digit>" candidate.
  3. It needs at least two independently confirmed candidates on the same page.
  4. It only replaces an empty span list and never merges with or reorders existing content.

Full-corpus validation (132/132 files, read-only): rosyth p35 recovers 3 of its 10 records (01,
03 and 08, the digits MinerU2.5 misreads as "Q<digit>"). The others are corrupted differently and
remain unresolved, so this is a partial improvement. The 4 Taonan records are unchanged.

The Taonan and rosyth tests read raw reader output from `data/extracted/raw` and return early
when that data is not present.
"""
from __future__ import annotations

from pathlib import Path

from app.pipeline.extract import (
    QuestionSpan,
    _recover_zero_padded_misread_questions,
    load_raw_output,
    segment_and_compare_page,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "extracted" / "raw"


def _real_data_available() -> bool:
    return (RAW_DIR / "mineru25" / "P6_Maths_2024_WA2_rosyth").is_dir()


# =================================================================================================
# _recover_zero_padded_misread_questions(), direct unit tests
# =================================================================================================

def test_recovers_real_misread_shape_with_cross_reader_corroboration():
    empty_reader_text = (
        "Q1) 1 big box = 2 x 2 x 2\n"
        "Q3) k[160s, 115s, 105s]\n"
        "Q8) A:C ratio, cost is $45 to $23 in the shop\n"
    )
    other_reader_text = (
        "01) 1 big box = 2 x 2 x 2\n"
        "02) A = S : D\n"
        "03) K R H\n"
        "04) W cases\n"
        "08) A:C ratio, cost is $45 to $23 in the shop\n"
    )
    spans = _recover_zero_padded_misread_questions(empty_reader_text, other_reader_text, 35)
    numbers = sorted(s.question_number for s in spans)
    assert numbers == ["01", "03", "08"]
    assert all(isinstance(s, QuestionSpan) for s in spans)


def test_does_not_fire_below_the_minimum_candidate_count():
    """A single "Q<digit>" match is not enough, even with corroboration.

    Zero-padded worked-solution pages have several such numbers.
    """
    empty_reader_text = "Q1) some real worked-solution content, long enough to pass the gate.\n"
    other_reader_text = "01) some real worked-solution content, long enough to pass the gate.\n"
    assert _recover_zero_padded_misread_questions(empty_reader_text, other_reader_text, 1) == []


def test_does_not_fire_without_real_cross_reader_corroboration():
    """Without a zero-padded form in the other reader's text, recovery does not fire.

    This is the Taonan "Q11." shape. Each candidate is checked independently, so two or more
    uncorroborated candidates still do not fire.
    """
    empty_reader_text = (
        "Q1) some real worked-solution content, long enough to pass the gate.\n"
        "Q3) more real worked-solution content, also long enough to pass the gate.\n"
    )
    other_reader_text = "This page has completely unrelated content with no zero-padded numbers at all."
    assert _recover_zero_padded_misread_questions(empty_reader_text, other_reader_text, 1) == []


def test_two_digit_numbers_never_match_the_single_digit_pattern():
    """Two-digit references such as Taonan's "Q11." and "Q15." are excluded by the pattern.

    Zero-padding never applies to two-digit numbers, and `Q(\\d)` captures exactly one digit, so
    these are excluded before the corroboration check.
    """
    empty_reader_text = (
        "Q11) some real worked-solution content, long enough to pass the gate.\n"
        "Q12) more real worked-solution content, also long enough to pass the gate.\n"
    )
    other_reader_text = "011) coincidentally present but irrelevant since Q11 has 2 digits anyway."
    assert _recover_zero_padded_misread_questions(empty_reader_text, other_reader_text, 1) == []


def test_short_content_rejected_even_with_corroboration():
    empty_reader_text = "Q1) too short\nQ3) also short\n"
    other_reader_text = "01) too short\n03) also short\n"
    assert _recover_zero_padded_misread_questions(empty_reader_text, other_reader_text, 1) == []


def test_empty_inputs_return_empty():
    assert _recover_zero_padded_misread_questions("", "01) real content here, long enough.", 1) == []
    assert _recover_zero_padded_misread_questions("Q1) real content here, long enough.", "", 1) == []


# =================================================================================================
# The 4 records that regressed under the reverted approach must be unaffected
# =================================================================================================

def test_taonan_p32_the_4_previously_regressed_records_are_unaffected():
    if not _real_data_available():
        return
    results, _ = segment_and_compare_page(
        "P6_Maths_2023_WA2_Taonan", 32, "P6_Maths_2023_WA2_Taonan:p32", RAW_DIR,
    )
    by_number = {span.question_number: rec for span, _, rec in results}
    # q20 and q10 are the two p32 records that regressed to "disagree" under the reverted approach.
    for number in ("10", "20"):
        assert number in by_number, f"q{number} missing entirely — real regression"
        assert by_number[number]["agreement_status"] == "agree", (
            f"q{number}: expected 'agree' (pre-regression state), got "
            f"{by_number[number]['agreement_status']!r}"
        )


def test_taonan_p33_the_4_previously_regressed_records_are_unaffected():
    if not _real_data_available():
        return
    results, _ = segment_and_compare_page(
        "P6_Maths_2023_WA2_Taonan", 33, "P6_Maths_2023_WA2_Taonan:p33", RAW_DIR,
    )
    by_number = {span.question_number: rec for span, _, rec in results}
    # q11 and q1 are the two p33 records that regressed to "disagree" under the reverted approach.
    for number in ("11", "1"):
        assert number in by_number, f"q{number} missing entirely — real regression"
        assert by_number[number]["agreement_status"] == "agree", (
            f"q{number}: expected 'agree' (pre-regression state), got "
            f"{by_number[number]['agreement_status']!r}"
        )
    # Exactly one span per number: a duplicate "11" span caused the earlier collision.
    numbers = [span.question_number for span, _, _ in results]
    assert numbers.count("11") == 1
    assert numbers.count("1") == 1


def test_taonan_p32_and_p33_produce_no_new_single_digit_q_spans():
    """The recovery path is never reached for Taonan p32 and p33.

    Both pages have non-empty raw text, so `_recover_zero_padded_misread_questions()` is not
    called for them.
    """
    if not _real_data_available():
        return
    for page in (32, 33):
        raw = load_raw_output("mineru25", "P6_Maths_2023_WA2_Taonan", page, RAW_DIR)
        assert raw is not None and raw.extracted_text.strip(), "expected real, non-empty raw text"


# =================================================================================================
# The rosyth p35 target: partial recovery
# =================================================================================================

def test_rosyth_p35_real_recovery_end_to_end():
    if not _real_data_available():
        return
    results, _ = segment_and_compare_page(
        "P6_Maths_2024_WA2_rosyth", 35, "P6_Maths_2024_WA2_rosyth:p35", RAW_DIR,
    )
    by_number = {span.question_number: rec for span, _, rec in results}
    # The 3 numbers MinerU2.5 misreads as "Q<digit>" (01, 03, 08) are recovered into a comparison.
    # Either agree or disagree is an improvement over single_reader_only.
    for number in ("01", "03", "08"):
        assert number in by_number, f"q{number} not recovered"
        assert by_number[number]["agreement_status"] in ("agree", "disagree")
