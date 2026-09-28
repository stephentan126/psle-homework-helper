"""
Tests for the `answer_key_grid` misclassification fix (Issue 229, candidate 1).

`_has_answer_grid_table_shape()` now uses the length of the value cell to tell a short answer-key
grid entry from a worked solution wrapped in a table. A page that is no longer excluded as a grid
still needs per-question segmentation, which `_has_question_like_content()` and
`_segment_from_regions()` provide.

Candidate 2, a "Q<1-2 digits>" question-start pattern for a zero-padded numbering misread on
`P6_Maths_2024_WA2_rosyth` p35, was reverted. It caused duplicate question numbers on
`P6_Maths_2023_WA2_Taonan` p32/p33 that dropped content and turned 4 records from `agree` to
`disagree` (see the comment on `_match_question_start()`). It has no tests here.

Full-corpus validation of candidate 1 (132/132 files, read-only) moved the totals from 5,679
records (403 single_reader_only) to 5,687 (411 single_reader_only). A per-question diff showed 0
removed and 8 added, all newly segmented single_reader_only records. The only other change was
`P6-Maths-2021-CA1-Rosyth:p32:q17` moving from disagree to agree, checked by hand as a correct
match; its exact cause was not traced, as the page has no `<table>` region and no answer-key file.
Issue 3's case and the existing regression suite (Issues 224/226/227/228, 32 tests) were
unaffected.
"""
from __future__ import annotations

from app.pipeline.extract import (
    PageImage,
    RawReaderOutput,
    _GRID_LABEL_VALUE_MAX_LEN,
    _has_answer_grid_table_shape,
    _has_worked_solution_table_shape,
    _parse_worked_solution_rows,
    classify_page,
    segment_questions,
)


# =================================================================================================
# _has_answer_grid_table_shape(): value-length gate
# =================================================================================================

def test_genuine_short_grid_value_still_recognized_as_grid():
    """A chij-style grid row (bare label, short value) is still recognised as a grid."""
    html = "<table><tr><td>Q17</td><td>6/7÷15=6/7×1/15=2/35</td></tr></table>"
    assert _has_answer_grid_table_shape(None, html) is True


def test_genuine_longest_real_grid_value_still_recognized():
    """The longest grid value in the corpus (223 chars) is still recognised.

    The value comes from `P6-Maths-2021-SA1-Catholic-High` p34 Q23. The threshold of 260 leaves a
    margin above it.
    """
    value = "x" * 223
    html = f"<table><tr><td>Q23</td><td>{value}</td></tr></table>"
    assert _has_answer_grid_table_shape(None, html) is True


def test_worked_solution_table_row_no_longer_misclassified_as_grid():
    """A worked solution in a table row is not classified as a grid.

    On `P6_Maths_2024_SA2_acsprimary` p40/41/56/60 the label cell ("Q27") looks like a grid label,
    but the value cell is a narrative paragraph of at least 302 chars.
    """
    value = (
        "1. Determine the Total Number of Students: Group A (0 devices) represents 40% of the "
        "students. Group C (2 devices) represents 10% of the students. Group D (more than 2 "
        "devices) represents 5% of the students. The remaining percentage for Group B (1 device) "
        "is: 100% - (40% + 10% + 5%) = 45% end of a genuinely long real worked solution paragraph."
    )
    assert len(value) > _GRID_LABEL_VALUE_MAX_LEN
    html = f"<table><tr><td>Q27</td><td>{value}</td></tr></table>"
    assert _has_answer_grid_table_shape(None, html) is False


# =================================================================================================
# _parse_worked_solution_rows() / _has_worked_solution_table_shape()
# =================================================================================================

def test_parse_worked_solution_rows_extracts_real_shape():
    long_value = "1. Step one. " * 30  # well over the length gate
    html = f"<table><tr><td>Q27</td><td>{long_value}</td></tr></table>"
    rows = _parse_worked_solution_rows(html)
    assert rows == [("27", long_value.strip())]


def test_parse_worked_solution_rows_keeps_letter_suffix_distinct():
    """Rows "14a)" and "14b)" keep distinct question numbers (P6_Maths_2024_SA2_acsprimary p56).

    If they collapsed to one number, one would overwrite the other in a {number: span} dict
    downstream.
    """
    long_value = "Solution steps go here at real length. " * 15
    html = (
        f"<table><tr><td>14a)</td><td>{long_value}</td></tr>"
        f"<tr><td>14b)</td><td>{long_value} different ending here</td></tr></table>"
    )
    rows = _parse_worked_solution_rows(html)
    numbers = [n for n, _ in rows]
    assert numbers == ["14a", "14b"]
    assert len(set(numbers)) == 2


def test_parse_worked_solution_rows_empty_for_genuine_short_grid():
    html = "<table><tr><td>Q1</td><td>3</td></tr></table>"
    assert _parse_worked_solution_rows(html) == []


def test_has_worked_solution_table_shape_true_only_for_the_long_shape():
    long_value = "A real worked solution paragraph of real length. " * 10
    assert _has_worked_solution_table_shape(None, f"<table><tr><td>Q5</td><td>{long_value}</td></tr></table>") is True
    assert _has_worked_solution_table_shape(None, "<table><tr><td>Q5</td><td>42</td></tr></table>") is False


# =================================================================================================
# End to end: classify_page() and segment_questions() on the acsprimary shape
# =================================================================================================

def _acsprimary_p40_shaped_long_value() -> str:
    return (
        "1. Determine the Total Number of Students: Group A (0 devices) represents 40% of the "
        "students. Group C (2 devices) represents 10% of the students. Group D (more than 2 "
        "devices) represents 5% of the students. The remaining percentage for Group B (1 device) "
        "is: 100% - (40% + 10% + 5%) = 45% and this sentence is padded out to be a genuinely long "
        "real worked-solution paragraph, well past the real 260-character threshold on purpose."
    )


def _acsprimary_p40_shaped_bounding_boxes() -> list:
    long_value = _acsprimary_p40_shaped_long_value()
    table_html = (
        f"<table><tr><td></td><td>a real preceding cell</td></tr>"
        f"<tr><td>Q27</td><td>{long_value}</td></tr></table>"
    )
    return [
        {"type": "header", "content": "BP-385"},
        {"type": "table", "content": table_html},
        {"type": "page_number", "content": "Pg3"},
    ]


def test_acsprimary_p40_shaped_page_now_classifies_as_question_page():
    bounding_boxes = _acsprimary_p40_shaped_bounding_boxes()
    text = bounding_boxes[1]["content"]
    assert classify_page(bounding_boxes=bounding_boxes, extracted_text=text) == "question_page"


def test_acsprimary_p40_shaped_page_now_segments_the_real_question():
    bounding_boxes = _acsprimary_p40_shaped_bounding_boxes()
    text = bounding_boxes[1]["content"]
    page = PageImage(source_pdf=__import__("pathlib").Path("x"), page_index=40, image_bytes=b"")
    norm = RawReaderOutput(extracted_text=text, structure=None, bounding_boxes=bounding_boxes, field_confidence=None)
    spans = segment_questions(page, norm, None, None, False)
    numbers = [s.question_number for s in spans]
    assert "27" in numbers
    span = next(s for s in spans if s.question_number == "27")
    assert "Determine the Total Number of Students" in span.raw_text


def test_acsprimary_p43_shaped_title_region_grid_still_untouched():
    """The title-region answer-key case still classifies as `answer_key_grid`.

    `P6_Maths_2024_SA2_acsprimary` p43 is classified by the separate title-region signal from
    Issue 144, not the table-cell signal changed here.
    """
    text = "Q5)\n1. A real, long worked-solution paragraph with no <table> markup anywhere on this page at all."
    bounding_boxes = [{"type": "title", "content": "Q5)"}]
    assert classify_page(bounding_boxes=bounding_boxes, extracted_text=text) == "answer_key_grid"
