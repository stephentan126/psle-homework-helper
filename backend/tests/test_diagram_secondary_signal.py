"""
Tests for the DeepSeek-OCR secondary signal for `has_diagram` (Issue 233).

`has_diagram` used to come only from MinerU2.5's structured "image" regions. DeepSeek-OCR always
returns empty `bounding_boxes`, but it marks diagrams in its flat text as a
`![](images/N.jpg)` markdown image reference.

The design was checked against corpus data first (see the comments in extract.py above
`_deepseek_ocr_page_has_diagram_marker` and `_question_text_references_a_diagram`). On 48 pages
MinerU2.5 set has_diagram=False for every question while DeepSeek-OCR showed the marker. Only
single-question pages (32 of the 48) are in scope, because a page-level marker on a shared page
(the other 16) cannot safely be attributed to one question. A manual review of the 32 found a
recurring false-positive shape (5 of 32): plain word problems with a trailing non-diagram image
marker. An extra gate therefore requires the question's text to contain a diagram keyword or a
named geometric label. With that gate, 20 of the 32 pages are upgraded, each confirmed by
inspection to depend on a diagram, and the other 12 are left unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.pipeline.extract import (
    QuestionSpan,
    _apply_deepseek_ocr_diagram_secondary_signal,
    _deepseek_ocr_page_has_diagram_marker,
    _question_text_references_a_diagram,
)


def _write_deepseek_raw(raw_dir: Path, pdf_stem: str, page_index: int, extracted_text: str) -> None:
    page_dir = raw_dir / "deepseek_ocr" / pdf_stem
    page_dir.mkdir(parents=True, exist_ok=True)
    (page_dir / f"{page_index}.json").write_text(
        json.dumps({
            "page_id": page_index, "model": "deepseek_ocr", "extracted_text": extracted_text,
            "structure": None, "bounding_boxes": [], "field_confidence": None, "error": None,
        }),
        encoding="utf-8",
    )


# =================================================================================================
# _deepseek_ocr_page_has_diagram_marker(): unit tests
# =================================================================================================

def test_detects_a_real_image_marker(tmp_path: Path):
    _write_deepseek_raw(tmp_path, "fake_paper", 5, "some text\n![](images/0.jpg)\nmore text")
    assert _deepseek_ocr_page_has_diagram_marker("fake_paper", 5, tmp_path) is True


def test_no_marker_present_returns_false(tmp_path: Path):
    _write_deepseek_raw(tmp_path, "fake_paper", 5, "just plain text, no images here at all")
    assert _deepseek_ocr_page_has_diagram_marker("fake_paper", 5, tmp_path) is False


def test_no_real_deepseek_output_for_this_page_returns_false_not_a_crash(tmp_path: Path):
    """A missing output file means no signal, matching the load_raw_output() contract."""
    assert _deepseek_ocr_page_has_diagram_marker("fake_paper", 99, tmp_path) is False


def test_deepseek_error_page_returns_false(tmp_path: Path):
    page_dir = tmp_path / "deepseek_ocr" / "fake_paper"
    page_dir.mkdir(parents=True)
    (page_dir / "5.json").write_text(
        json.dumps({"extracted_text": "![](images/0.jpg)", "error": "real inference error"}),
        encoding="utf-8",
    )
    assert _deepseek_ocr_page_has_diagram_marker("fake_paper", 5, tmp_path) is False


def test_multiple_real_markers_on_one_page_still_detected(tmp_path: Path):
    _write_deepseek_raw(
        tmp_path, "fake_paper", 5,
        "solid A ![](images/0.jpg) and solid B ![](images/1.jpg)",
    )
    assert _deepseek_ocr_page_has_diagram_marker("fake_paper", 5, tmp_path) is True


# =================================================================================================
# _question_text_references_a_diagram(): the false-positive gate, unit tests
# =================================================================================================

def test_real_keyword_alone_is_sufficient():
    assert _question_text_references_a_diagram("Study the pattern below.") is True


def test_real_named_label_alone_is_sufficient_no_keyword_needed():
    """A named label alone passes the gate.

    'Find the area of triangle XYZ' has no diagram, figure or pattern keyword, only the label.
    """
    assert _question_text_references_a_diagram("Find the area of triangle XYZ.") is True


def test_real_fruit_stall_shaped_word_problem_has_neither_signal():
    """A plain arithmetic word problem, the false-positive shape, fails the gate.

    It has no keyword and no upper-case label.
    """
    text = (
        "At a fruit stall, mangoes were sold at 3 for $10 and avocados were sold at 4 for $9. "
        "Mrs Sammy bought an equal number of mangoes and avocados."
    )
    assert _question_text_references_a_diagram(text) is False


def test_single_letter_is_not_mistaken_for_a_named_label():
    """A single-letter algebra variable ('y') is not treated as a named label.

    This follows the same reasoning as _NAMED_LABEL_RE in question_matching.py (Issue 199(a)).
    """
    assert _question_text_references_a_diagram("Find the value of y.") is False


# =================================================================================================
# _apply_deepseek_ocr_diagram_secondary_signal(): wiring, unit tests
# =================================================================================================

def _span(number: str = "1", has_diagram: bool = False, raw_text: str = "In the figure below, ABCD is shown.") -> QuestionSpan:
    return QuestionSpan(question_number=number, raw_text=raw_text, page_index=5,
                         has_diagram=has_diagram)


def test_upgrades_a_real_single_question_page_from_false_to_true(tmp_path: Path):
    _write_deepseek_raw(tmp_path, "fake_paper", 5, "In the figure below ![](images/0.jpg)")
    span = _span(has_diagram=False)
    span_results = [(span, "high", {})]
    _apply_deepseek_ocr_diagram_secondary_signal(span_results, "fake_paper", 5, tmp_path)
    assert span.has_diagram is True


def test_does_not_touch_a_single_question_page_with_no_real_marker(tmp_path: Path):
    _write_deepseek_raw(tmp_path, "fake_paper", 5, "no diagram reference here at all")
    span = _span(has_diagram=False)
    span_results = [(span, "high", {})]
    _apply_deepseek_ocr_diagram_secondary_signal(span_results, "fake_paper", 5, tmp_path)
    assert span.has_diagram is False


def test_real_marker_present_but_question_text_has_no_keyword_or_label_stays_false(tmp_path: Path):
    """A marker on a plain word problem does not upgrade has_diagram.

    This is the false-positive shape found in the corpus, where the marker was a trailing footer
    or margin graphic.
    """
    _write_deepseek_raw(
        tmp_path, "fake_paper", 5,
        "At a fruit stall, mangoes were sold at 3 for $10... Ans: [4]  \n\n![](images/0.jpg)\n",
    )
    span = _span(
        has_diagram=False,
        raw_text="At a fruit stall, mangoes were sold at 3 for $10 and avocados were sold at 4 for $9.",
    )
    span_results = [(span, "high", {})]
    _apply_deepseek_ocr_diagram_secondary_signal(span_results, "fake_paper", 5, tmp_path)
    assert span.has_diagram is False


def test_never_touches_an_already_true_mineru_signal(tmp_path: Path):
    """A span MinerU2.5 already marked True is left as it is.

    The signal only upgrades False to True. For a True span, the DeepSeek-OCR file should not be
    read at all. The page and stem used here have no raw file; load_raw_output would return None
    anyway, but the test shows the short-circuit is deliberate.
    """
    span = _span(has_diagram=True)
    span_results = [(span, "high", {})]
    _apply_deepseek_ocr_diagram_secondary_signal(span_results, "no_such_paper", 999, tmp_path)
    assert span.has_diagram is True


def test_never_applies_to_a_real_multi_question_page_even_with_a_real_marker(tmp_path: Path):
    """A page with two or more questions is never changed by this signal.

    This holds even with a marker present and both questions passing the keyword or label gate,
    because the page-level marker cannot safely be attributed to either question.
    """
    _write_deepseek_raw(tmp_path, "fake_paper", 5, "In the figure below ![](images/0.jpg)")
    span_a = _span(number="1", has_diagram=False)
    span_b = _span(number="2", has_diagram=False)
    span_results = [(span_a, "high", {}), (span_b, "high", {})]
    _apply_deepseek_ocr_diagram_secondary_signal(span_results, "fake_paper", 5, tmp_path)
    assert span_a.has_diagram is False
    assert span_b.has_diagram is False


def test_empty_span_results_is_a_real_noop_not_a_crash(tmp_path: Path):
    _apply_deepseek_ocr_diagram_secondary_signal([], "fake_paper", 5, tmp_path)  # must not raise
