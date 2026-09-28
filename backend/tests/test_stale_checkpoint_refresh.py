"""
Tests for refreshing stale Stage C checkpoints (Issue 257).

The "already exists" resumability skip for Stage C checkpoints (the Issue 62 pattern) ignored
changes to the Stage B raw data after the checkpoint was written. In psle.db, `agreement_status`
was stale for 4,224 rows while DeepSeek-OCR kept producing new page output.

Two fixes are tested. `_checkpoint_is_stale()` is the extraction-side check that makes
`_extract_one_file()` recompute a checkpoint whose raw data has changed.
`_find_stale_checkpoint_desyncs()` is the database-side guard: `write_extracted_data_to_db()`
refuses to run, rather than re-inserting, if a checkpoint was refreshed after its marker was
written. It fails loudly, like `_find_marker_db_desyncs()` (Issues 158 and 230).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from app.pipeline.extract import (
    MarkerDatabaseDesyncError,
    _checkpoint_is_stale,
    _find_stale_checkpoint_desyncs,
    write_extracted_data_to_db,
)


def _touch(path: Path, mtime: float, content: str = "{}") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    os.utime(path, (mtime, mtime))


# =================================================================================================
# _checkpoint_is_stale(), the extraction-side check
# =================================================================================================

def test_checkpoint_fresher_than_all_raw_files_is_not_stale(tmp_path: Path):
    """A checkpoint newer than all its Stage B raw files is not stale, so the skip still applies."""
    raw_dir = tmp_path / "raw"
    checkpoint = tmp_path / "extracted" / "fake_paper.json"
    now = time.time()

    _touch(raw_dir / "mineru25" / "fake_paper" / "0.json", now - 100)
    _touch(raw_dir / "deepseek_ocr" / "fake_paper" / "0.json", now - 100)
    _touch(checkpoint, now)  # checkpoint written after both raw files

    assert _checkpoint_is_stale(checkpoint, raw_dir, "fake_paper") is False


def test_checkpoint_older_than_a_later_raw_file_is_stale(tmp_path: Path):
    """A checkpoint is stale if a reader writes a page after it (the Issue 257 shape).

    Here deepseek_ocr produces a page later, as when a resumed OCR run finishes days afterwards.
    """
    raw_dir = tmp_path / "raw"
    checkpoint = tmp_path / "extracted" / "fake_paper.json"
    now = time.time()

    _touch(raw_dir / "mineru25" / "fake_paper" / "0.json", now - 100)
    _touch(checkpoint, now - 50)  # checkpoint written before the deepseek_ocr page arrives
    _touch(raw_dir / "deepseek_ocr" / "fake_paper" / "0.json", now)  # arrives after the checkpoint

    assert _checkpoint_is_stale(checkpoint, raw_dir, "fake_paper") is True


def test_checkpoint_stale_check_ignores_unrelated_pdfs(tmp_path: Path):
    """A newer raw file for a different pdf_stem does not mark this checkpoint stale."""
    raw_dir = tmp_path / "raw"
    checkpoint = tmp_path / "extracted" / "fake_paper.json"
    now = time.time()

    _touch(raw_dir / "mineru25" / "fake_paper" / "0.json", now - 100)
    _touch(checkpoint, now - 50)
    _touch(raw_dir / "mineru25" / "a_different_fake_paper" / "0.json", now)  # unrelated, newer

    assert _checkpoint_is_stale(checkpoint, raw_dir, "fake_paper") is False


def test_checkpoint_stale_check_handles_missing_reader_directory(tmp_path: Path):
    """A reader with no output directory for this pdf_stem does not crash the stale check.

    The pipeline already handles an absent reader elsewhere (the known-bad pages in Issue 221).
    """
    raw_dir = tmp_path / "raw"
    checkpoint = tmp_path / "extracted" / "fake_paper.json"
    now = time.time()

    _touch(raw_dir / "mineru25" / "fake_paper" / "0.json", now - 100)
    _touch(checkpoint, now)
    # deepseek_ocr/fake_paper/ directory deliberately never created.

    assert _checkpoint_is_stale(checkpoint, raw_dir, "fake_paper") is False


# =================================================================================================
# _find_stale_checkpoint_desyncs() / write_extracted_data_to_db(), the database-side guard
# =================================================================================================

def test_checkpoint_refreshed_after_its_own_marker_is_refused_not_reinserted(tmp_path: Path):
    """A checkpoint refreshed after its rows were committed makes the write refuse (Issue 257).

    The extraction-side check recomputes the checkpoint after Stage B changes, but the database
    holds rows from the old version. Leaving them stale was the bug, and re-inserting would risk
    duplicate rows (Issue 158), so the write must refuse.
    """
    extracted_dir = tmp_path / "extracted"
    db_write_log_dir = tmp_path / "db_write_log"
    pdf_stem = "P6_Maths_2099_Test_stale"
    source_paper = f"{pdf_stem}.pdf"

    pair = {
        "question": {
            "source_paper": source_paper, "source_page_index": 1, "source_page_label": "p1",
            "answer_source_file": source_paper, "answer_source_page_index": 1,
            "ocr_confidence": "low", "school_name": "faketestschool", "school_name_verified": True,
            "verification_status": "unverified", "verified_at": None, "superseded_by": None,
            "extraction_flag": None, "question_section": None, "question_number": "1",
            "question_text": "A real, sufficiently long fake question text for a test fixture.",
            "answer_value": "42", "worked_solution_text": None, "has_diagram": False,
        },
        "ocr_record": None,
    }
    checkpoint_path = extracted_dir / f"{pdf_stem}.json"
    extracted_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(json.dumps([pair], indent=2), encoding="utf-8")

    write_extracted_data_to_db(extracted_dir, db_write_log_dir)
    marker_path = db_write_log_dir / f"{pdf_stem}.done"
    assert marker_path.exists()

    # Trigger: the checkpoint is rewritten, with its mtime set strictly after the committed marker.
    marker_mtime = marker_path.stat().st_mtime
    time.sleep(0.02)
    checkpoint_path.write_text(json.dumps([pair], indent=2), encoding="utf-8")
    os.utime(checkpoint_path, (marker_mtime + 5, marker_mtime + 5))

    assert _find_stale_checkpoint_desyncs([checkpoint_path], db_write_log_dir) == [
        {"pdf_stem": pdf_stem, "checkpoint_path": str(checkpoint_path), "marker_path": str(marker_path)}
    ]

    with pytest.raises(MarkerDatabaseDesyncError) as exc_info:
        write_extracted_data_to_db(extracted_dir, db_write_log_dir)
    assert pdf_stem in str(exc_info.value)


def test_checkpoint_older_than_its_own_marker_is_not_a_desync(tmp_path: Path):
    """A checkpoint older than its marker (extract, then write) is not flagged."""
    extracted_dir = tmp_path / "extracted"
    db_write_log_dir = tmp_path / "db_write_log"
    pdf_stem = "P6_Maths_2099_Test_normal"
    checkpoint_path = extracted_dir / f"{pdf_stem}.json"
    marker_path = db_write_log_dir / f"{pdf_stem}.done"
    now = time.time()

    _touch(checkpoint_path, now - 10, content="[]")
    _touch(marker_path, now)

    assert _find_stale_checkpoint_desyncs([checkpoint_path], db_write_log_dir) == []


def test_checkpoint_with_no_marker_at_all_is_not_this_desync_shape(tmp_path: Path):
    """A checkpoint with no marker is left to _find_marker_db_desyncs and not flagged here."""
    extracted_dir = tmp_path / "extracted"
    db_write_log_dir = tmp_path / "db_write_log"
    db_write_log_dir.mkdir(parents=True)
    pdf_stem = "P6_Maths_2099_Test_nomarker"
    checkpoint_path = extracted_dir / f"{pdf_stem}.json"
    _touch(checkpoint_path, time.time(), content="[]")

    assert _find_stale_checkpoint_desyncs([checkpoint_path], db_write_log_dir) == []
