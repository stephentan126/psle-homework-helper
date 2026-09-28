"""
Tests for the region-sequence signal that marks diagram tables as noise (Issue 145).

The signal has three conditions: the table is the only one on the page, it is immediately preceded
by a bare single-word title region, and it is immediately followed by an equation region. Together
they separate the bar-model diagram table on Nan-Hua p38 (the Issue 145 case) from every answer-key
grid and worked-solution table in the corpus, with no false positives across 132 files (4,423
pages, 1,344 `table` regions). The evidence is in the docstring of
`_is_solo_diagram_noise_table_region()` in `backend/app/pipeline/extract.py`.

The region lists are trimmed excerpts of the MinerU2.5 output for the pages named in each test.
"""
from __future__ import annotations

from pathlib import Path

from app.pipeline.extract import (
    RawReaderOutput,
    _is_solo_diagram_noise_table_region,
    locate_embedded_answer_section,
)


# =================================================================================================
# _is_solo_diagram_noise_table_region(): unit level
# =================================================================================================

def test_real_nan_hua_p38_diagram_table_is_recognized_as_noise():
    """The Issue 145 case (Nan-Hua p38) meets all three conditions and is marked as noise.

    It is a solo table, preceded by the bare title 'After' and followed by an equation region.
    """
    bounding_boxes = [
        {"type": "header", "content": "BP~341"},
        {"type": "title", "content": "After"},
        {"type": "table", "content": "<table><tr><td>9 u</td><td>60</td><td>38</td></tr></table>"},
        {"type": "equation", "content": "7u = 60 + 38 = 98"},
        {"type": "text", "content": "OR"},
    ]
    assert _is_solo_diagram_noise_table_region(bounding_boxes) is True


def test_second_table_on_page_disqualifies_noise_signal():
    """A page with a second table region is never excluded (condition 1).

    This holds even when the first table meets the other two conditions.
    """
    bounding_boxes = [
        {"type": "title", "content": "After"},
        {"type": "table", "content": "<table><tr><td>9 u</td><td>60</td></tr></table>"},
        {"type": "equation", "content": "7u = 98"},
        {"type": "table", "content": "<table><tr><td>Q1</td><td>real answer</td></tr></table>"},
    ]
    assert _is_solo_diagram_noise_table_region(bounding_boxes) is False


def test_real_nan_hua_p32_near_miss_correctly_excluded_multi_word_title_absent():
    """The open-answer grid on Nan-Hua SA2 p32 is not marked as noise.

    It is a solo table with no following equation, and the preceding region ('Section B (20
    marks)') is `text`, not `title`, so condition 2 fails.
    """
    bounding_boxes = [
        {"type": "text", "content": "Section B (20 marks)"},
        {"type": "table", "content": "<table><tr><td>16)</td><td>120</td></tr></table>"},
        {"type": "text", "content": "some other real content"},
    ]
    assert _is_solo_diagram_noise_table_region(bounding_boxes) is False


def test_real_nanyang_p50_near_miss_correctly_excluded_text_not_title_precedes():
    """The True/False answer table on P6_Maths_2022_SA2_nanyang p50 is not marked as noise.

    An equation follows it, but the preceding region is `text`, not `title`, so condition 2 fails.
    This case would be a false positive if condition 3 were used alone.
    """
    bounding_boxes = [
        {"type": "text", "content": "Each of the statements below is either true, false or not "
                                     "possible by using the given information."},
        {"type": "table", "content": "<table><tr><td>Statement</td><td>True</td></tr></table>"},
        {"type": "equation", "content": "fix S = 230"},
    ]
    assert _is_solo_diagram_noise_table_region(bounding_boxes) is False


def test_real_psle2018_p2_near_miss_correctly_excluded_text_not_title_precedes():
    """The Q22 worked-solution table in `PSLE Math 2018 Answer+Solution` p2 is not noise.

    An equation follows this vertical-multiplication table, but the preceding region
    ('22. 27 42 6') is `text`, not `title`, so condition 2 fails.
    """
    bounding_boxes = [
        {"type": "text", "content": "22. 27 42 6"},
        {"type": "table", "content": "<table><tr><td>×</td><td>7.26</td></tr></table>"},
        {"type": "equation", "content": "7.26 * 8 = 58.08"},
    ]
    assert _is_solo_diagram_noise_table_region(bounding_boxes) is False


def test_title_precedes_alone_is_not_sufficient_real_grid_with_section_header_not_excluded():
    """An answer-key grid under a section-header title is not marked as noise.

    A title such as 'BOOKLET A (PAPER 1)' satisfies condition 2, but no equation follows, so
    condition 3 fails.
    """
    bounding_boxes = [
        {"type": "title", "content": "BOOKLET"},
        {"type": "table", "content": "<table><tr><td>Q1</td><td>42</td></tr></table>"},
        {"type": "text", "content": "next real content, not an equation"},
    ]
    assert _is_solo_diagram_noise_table_region(bounding_boxes) is False


def test_multi_word_title_does_not_trigger_noise_signal():
    """A multi-word title does not satisfy condition 2, even when the other two conditions hold."""
    bounding_boxes = [
        {"type": "title", "content": "Before After"},
        {"type": "table", "content": "<table><tr><td>9 u</td><td>60</td></tr></table>"},
        {"type": "equation", "content": "7u = 98"},
    ]
    assert _is_solo_diagram_noise_table_region(bounding_boxes) is False


def test_no_bounding_boxes_returns_false():
    """Empty `bounding_boxes` returns False without error.

    DeepSeek-OCR always returns `[]` here, so there is nothing to check.
    """
    assert _is_solo_diagram_noise_table_region(None) is False
    assert _is_solo_diagram_noise_table_region([]) is False


def test_table_at_start_or_end_of_region_list_has_no_neighbor_and_is_not_noise():
    """A table at either end of the region list has no neighbour and is not noise.

    Conditions 2 and 3 cannot hold, and the check must not fail on the missing neighbour.
    """
    bounding_boxes_first = [
        {"type": "table", "content": "<table><tr><td>9 u</td><td>60</td></tr></table>"},
        {"type": "equation", "content": "7u = 98"},
    ]
    assert _is_solo_diagram_noise_table_region(bounding_boxes_first) is False

    bounding_boxes_last = [
        {"type": "title", "content": "After"},
        {"type": "table", "content": "<table><tr><td>9 u</td><td>60</td></tr></table>"},
    ]
    assert _is_solo_diagram_noise_table_region(bounding_boxes_last) is False


# =================================================================================================
# locate_embedded_answer_section(): integration. The chij-p47 carry-forward no longer sweeps the
# diagram-noise table in as a grid continuation.
# =================================================================================================

def test_diagram_noise_table_no_longer_swept_into_grid_by_carry_forward():
    """The diagram table on p38 is not carried forward as a continuation of the p37 grid.

    This reproduces Issue 145. classify_page() calls p38 'other', but it still has a table. Since
    that table matches the noise signal, the chij-p47 carry-forward must skip it, so neither
    spurious row ('60'->'38', '9u'->'7 u') appears in the answer_map.
    """
    grid_page_boxes = [
        {"type": "table", "content": "<table><tr><td>17.</td><td>real Q17 content</td></tr></table>"},
    ]
    noise_page_boxes = [
        {"type": "header", "content": "BP~341"},
        {"type": "title", "content": "After"},
        {"type": "table", "content": "<table><tr><td>9 u</td><td>60</td><td>38</td></tr>"
                                      "<tr><td>9u</td><td colspan=\"2\">7 u</td></tr></table>"},
        {"type": "equation", "content": "7u = 60 + 38 = 98"},
        {"type": "text", "content": "OR"},
    ]
    reader_output_by_page = {
        37: RawReaderOutput(extracted_text="17. real Q17 content", bounding_boxes=grid_page_boxes),
        38: RawReaderOutput(extracted_text="After\n<table>...</table>\n7u = 60 + 38 = 98",
                             bounding_boxes=noise_page_boxes),
    }
    answer_map, _section_answer_map = locate_embedded_answer_section(
        Path("P6-Maths-2021-SA1-Nan-Hua.pdf"), reader_output_by_page,
    )
    assert "60" not in answer_map
    assert "9u" not in answer_map


def test_genuine_chij_p47_style_continuation_still_carried_forward():
    """An orphan continuation fragment in the chij-p47 shape is still carried forward.

    The fix only excludes the noise shape and must not break the carry-forward itself.
    """
    grid_page_boxes = [
        {"type": "table", "content": "<table><tr><td>30.</td><td>real Q30 content</td></tr></table>"},
    ]
    continuation_page_boxes = [
        {"type": "table", "content": "<table><tr><td></td><td>orphan continuation fragment, no "
                                      "label cell at all</td></tr></table>"},
    ]
    reader_output_by_page = {
        45: RawReaderOutput(extracted_text="30. real Q30 content", bounding_boxes=grid_page_boxes),
        46: RawReaderOutput(extracted_text="orphan continuation fragment",
                             bounding_boxes=continuation_page_boxes),
    }
    answer_map, _section_answer_map = locate_embedded_answer_section(
        Path("chij.pdf"), reader_output_by_page,
    )
    assert "30" in answer_map
    # The fragment's content is appended to Q30's chain rather than lost.
    assert "orphan continuation fragment" in answer_map["30"][1]
