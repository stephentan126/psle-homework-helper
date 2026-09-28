"""
Tests for the `disagreement_type` classification returned by `compare_readers()` (Issue 212).

The function itself, not a downstream string matcher, must return `value_mismatch` or
`structural_mismatch` as appropriate, and `None` for `agree`, `both_failed` and
`single_reader_only`, where there is no disagreement to classify.

The extraction pipeline was otherwise validated with corpus validation scripts (Issues 89-173).
These fast, deterministic unit tests add to that for this one classification concern.
"""
from __future__ import annotations

import json

from app.pipeline.extract import (
    RawReaderOutput,
    compare_readers,
    segment_and_compare_page,
)


def _write_raw_page(
    raw_dir, reader_name, pdf_stem, page_index, extracted_text, bounding_boxes=None, structure=None,
):
    page_dir = raw_dir / reader_name / pdf_stem
    page_dir.mkdir(parents=True, exist_ok=True)
    (page_dir / f"{page_index}.json").write_text(
        json.dumps({
            "extracted_text": extracted_text,
            "structure": structure,
            "bounding_boxes": bounding_boxes,
            "field_confidence": None,
            "error": None,
        }),
        encoding="utf-8",
    )


def test_compare_readers_classifies_a_real_value_mismatch():
    primary = RawReaderOutput(extracted_text="The answer is 42.")
    secondary = RawReaderOutput(extracted_text="The answer is 17.")
    agreement_status, disagreement_detail, disagreement_type = compare_readers(primary, secondary)
    assert agreement_status == "disagree"
    assert disagreement_type == "value_mismatch"
    assert disagreement_detail is not None
    assert "not equivalent" in disagreement_detail


def test_compare_readers_classifies_a_real_structural_mismatch():
    # Identical extracted_text (values_equivalent() -> True), but only one reader reports table
    # structure, so _structure_looks_fabricated() returns True under its current placeholder
    # heuristic.
    same_text = "Find the value of 3/4 + 1/2."
    primary = RawReaderOutput(
        extracted_text=same_text, structure={"tables": [{"rows": []}]},
    )
    secondary = RawReaderOutput(extracted_text=same_text, structure=None)
    agreement_status, disagreement_detail, disagreement_type = compare_readers(primary, secondary)
    assert agreement_status == "disagree"
    assert disagreement_type == "structural_mismatch"
    assert disagreement_detail is not None
    assert "structure mismatch" in disagreement_detail


def test_compare_readers_agree_case_has_no_disagreement_type():
    same_text = "Find the value of 3/4 + 1/2."
    primary = RawReaderOutput(extracted_text=same_text)
    secondary = RawReaderOutput(extracted_text=same_text)
    agreement_status, disagreement_detail, disagreement_type = compare_readers(primary, secondary)
    assert agreement_status == "agree"
    assert disagreement_type is None
    assert disagreement_detail is None


def test_compare_readers_both_failed_has_no_disagreement_type():
    primary = RawReaderOutput(extracted_text="")
    secondary = RawReaderOutput(extracted_text="   ")
    agreement_status, disagreement_detail, disagreement_type = compare_readers(primary, secondary)
    assert agreement_status == "both_failed"
    assert disagreement_type is None


def test_compare_readers_single_reader_only_has_no_disagreement_type():
    primary = RawReaderOutput(extracted_text="")
    secondary = RawReaderOutput(extracted_text="The answer is 42.")
    agreement_status, disagreement_detail, disagreement_type = compare_readers(primary, secondary)
    assert agreement_status == "single_reader_only"
    assert disagreement_type is None


# Issue 217: segment_and_compare_page() is a separate per-question comparison path with inline
# text-similarity logic; it never calls compare_readers(). Its ocr_record dicts had no
# disagreement_type key, so the unconditional `ocr_record["disagreement_type"]` read added for
# Issue 213 (the grid cross-check merge in extract_topology_a/b) raised KeyError on every
# question. All 132 files failed a full corpus Stage C re-run. These tests call
# segment_and_compare_page() directly, since that is where the bug was.

def test_segment_and_compare_page_single_reader_only_has_no_disagreement_type(tmp_path):
    raw_dir = tmp_path / "raw"
    _write_raw_page(raw_dir, "mineru25", "testfile", 0, "1. What is the value of x in this equation?")
    results, representative = segment_and_compare_page("testfile", 0, "testfile:p0", raw_dir)
    assert representative is not None
    assert len(results) == 1
    _span, ocr_confidence, ocr_record = results[0]
    assert ocr_record["agreement_status"] == "single_reader_only"
    assert ocr_record["disagreement_type"] is None


def test_segment_and_compare_page_agree_has_no_disagreement_type(tmp_path):
    raw_dir = tmp_path / "raw"
    text = "1. What is the value of x in this equation? Show your working clearly."
    _write_raw_page(raw_dir, "mineru25", "testfile", 0, text)
    _write_raw_page(raw_dir, "deepseek_ocr", "testfile", 0, text)
    results, _representative = segment_and_compare_page("testfile", 0, "testfile:p0", raw_dir)
    assert len(results) == 1
    _span, _ocr_confidence, ocr_record = results[0]
    assert ocr_record["agreement_status"] == "agree"
    assert ocr_record["disagreement_type"] is None


def test_segment_and_compare_page_value_mismatch_sets_disagreement_type(tmp_path):
    raw_dir = tmp_path / "raw"
    _write_raw_page(
        raw_dir, "mineru25", "testfile", 0,
        "1. What is the value of x in this equation? Show your working clearly.",
    )
    _write_raw_page(
        raw_dir, "deepseek_ocr", "testfile", 0,
        "1. Completely different real content that shares nothing with the primary reader at all.",
    )
    results, _representative = segment_and_compare_page("testfile", 0, "testfile:p0", raw_dir)
    assert len(results) == 1
    _span, _ocr_confidence, ocr_record = results[0]
    assert ocr_record["agreement_status"] == "disagree"
    assert ocr_record["disagreement_type"] == "value_mismatch"
    assert "text similarity" in ocr_record["disagreement_detail"]
    assert "<" in ocr_record["disagreement_detail"]


def test_segment_and_compare_page_structural_mismatch_sets_disagreement_type(tmp_path):
    raw_dir = tmp_path / "raw"
    same_text = "1. What is the value of x in this equation? Show your working clearly."
    _write_raw_page(
        raw_dir, "mineru25", "testfile", 0, same_text, structure={"tables": [{"rows": []}]},
    )
    _write_raw_page(raw_dir, "deepseek_ocr", "testfile", 0, same_text, structure=None)
    results, _representative = segment_and_compare_page("testfile", 0, "testfile:p0", raw_dir)
    assert len(results) == 1
    _span, _ocr_confidence, ocr_record = results[0]
    assert ocr_record["agreement_status"] == "disagree"
    assert ocr_record["disagreement_type"] == "structural_mismatch"
    assert "structure mismatch" in ocr_record["disagreement_detail"]
