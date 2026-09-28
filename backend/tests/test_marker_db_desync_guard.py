"""
Tests for the marker/database desync guard in `write_extracted_data_to_db()` (Issue 158).

A stale or cleared `db_write_log` marker directory once let `write_extracted_data_to_db()` re-run
against already-populated `questions` and `ocr_extraction_records` tables 15 times over (about
81,479 duplicate rows). The marker check was a file-existence check only, with no check against
database content.

These tests reproduce the trigger (a checkpoint's marker is cleared without truncating the rows
it guards) and check that the pre-flight guard (`_find_marker_db_desyncs()` and
`MarkerDatabaseDesyncError` in extract.py) refuses to write instead of re-inserting. They also
check that normal runs (a fresh run, a resumed run, an incrementally added file) are unaffected.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.db.session import get_session
from app.models import Question
from app.pipeline.extract import (
    MarkerDatabaseDesyncError,
    _find_marker_db_desyncs,
    write_extracted_data_to_db,
)


def _make_pair(
    question_number: str = "1",
    source_paper: str = "fake_test_paper.pdf",
    question_text: str = "A real, sufficiently long fake question text for a test fixture.",
) -> dict:
    """Build a minimal {"question": {...}, "ocr_record": None} pair.

    Every field that write_extracted_data_to_db() reads via `q[...]` (not `.get()`) is present.
    """
    return {
        "question": {
            "source_paper": source_paper,
            "source_page_index": 1,
            "source_page_label": "p1",
            "answer_source_file": source_paper,
            "answer_source_page_index": 1,
            "ocr_confidence": "high",
            "school_name": "faketestschool",
            "school_name_verified": True,
            "verification_status": "unverified",
            "verified_at": None,
            "superseded_by": None,
            "extraction_flag": None,
            "question_section": None,
            "question_number": question_number,
            "question_text": question_text,
            "answer_value": "42",
            "worked_solution_text": None,
            "has_diagram": False,
        },
        "ocr_record": None,
    }


def _write_checkpoint(extracted_dir: Path, pdf_stem: str, pairs: list) -> None:
    extracted_dir.mkdir(parents=True, exist_ok=True)
    (extracted_dir / f"{pdf_stem}.json").write_text(json.dumps(pairs, indent=2), encoding="utf-8")


def _real_question_count(source_paper: str) -> int:
    with get_session() as session:  # type: ignore
        return session.query(Question).filter(Question.source_paper == source_paper).count()


# =================================================================================================
# Regression scenario: the original trigger condition is now blocked
# =================================================================================================

def test_cleared_marker_without_truncate_is_refused_not_reinserted(tmp_path: Path):
    """A missing marker with existing rows for its source_paper must be refused, not re-inserted.

    This is the incident's shape: a prior run committed the rows, then the marker was lost
    without the tables being truncated.
    """
    extracted_dir = tmp_path / "extracted"
    db_write_log_dir = tmp_path / "db_write_log"
    source_paper = "P6_Maths_2099_Test_fake.pdf"
    pdf_stem = "P6_Maths_2099_Test_fake"

    _write_checkpoint(extracted_dir, pdf_stem, [_make_pair(source_paper=source_paper)])

    # First run: marker absent, table empty, so it writes normally.
    write_extracted_data_to_db(extracted_dir, db_write_log_dir)
    assert _real_question_count(source_paper) == 1
    marker_path = db_write_log_dir / f"{pdf_stem}.done"
    assert marker_path.exists()

    # Simulate the trigger: the marker is lost but the database still holds its one row for
    # this source_paper.
    marker_path.unlink()

    with pytest.raises(MarkerDatabaseDesyncError) as exc_info:
        write_extracted_data_to_db(extracted_dir, db_write_log_dir)

    assert pdf_stem in str(exc_info.value)
    assert source_paper in str(exc_info.value)

    # Still exactly one row: nothing was re-inserted.
    assert _real_question_count(source_paper) == 1
    # No marker was written back, because the run refused before reaching that file.
    assert not marker_path.exists()


def test_find_marker_db_desyncs_direct(tmp_path: Path):
    """Unit test of the pre-flight detector, isolated from the full write path."""
    extracted_dir = tmp_path / "extracted"
    db_write_log_dir = tmp_path / "db_write_log"
    db_write_log_dir.mkdir(parents=True)
    source_paper = "P6_Maths_2099_Test_fake2.pdf"
    pdf_stem = "P6_Maths_2099_Test_fake2"
    _write_checkpoint(extracted_dir, pdf_stem, [_make_pair(source_paper=source_paper)])

    with get_session() as session:  # type: ignore
        # No marker, no existing rows -> not a desync.
        assert _find_marker_db_desyncs([extracted_dir / f"{pdf_stem}.json"], db_write_log_dir, session) == []

        # Insert a row directly, as if committed by a prior run, with no marker present.
        session.add(Question(**_make_pair(source_paper=source_paper)["question"]))
        session.flush()

    with get_session() as session:  # type: ignore
        desyncs = _find_marker_db_desyncs(
            [extracted_dir / f"{pdf_stem}.json"], db_write_log_dir, session,
        )
    assert len(desyncs) == 1
    assert desyncs[0]["pdf_stem"] == pdf_stem
    assert desyncs[0]["source_paper"] == source_paper
    assert desyncs[0]["existing_row_count"] == 1


# =================================================================================================
# No regression: runs without a desync are unaffected
# =================================================================================================

def test_fresh_extraction_run_still_works_exactly_as_before(tmp_path: Path):
    """With the marker absent and the tables empty, the run writes normally."""
    extracted_dir = tmp_path / "extracted"
    db_write_log_dir = tmp_path / "db_write_log"
    source_paper = "P6_Maths_2099_Test_fresh.pdf"
    pdf_stem = "P6_Maths_2099_Test_fresh"
    _write_checkpoint(extracted_dir, pdf_stem, [
        _make_pair(question_number="1", source_paper=source_paper),
        _make_pair(question_number="2", source_paper=source_paper),
    ])

    write_extracted_data_to_db(extracted_dir, db_write_log_dir)

    assert _real_question_count(source_paper) == 2
    assert (db_write_log_dir / f"{pdf_stem}.done").exists()


def test_normal_resumability_marker_and_rows_both_present_is_not_a_desync(tmp_path: Path):
    """A marker with matching rows (the state after any successful run) is skipped, not flagged."""
    extracted_dir = tmp_path / "extracted"
    db_write_log_dir = tmp_path / "db_write_log"
    source_paper = "P6_Maths_2099_Test_resume.pdf"
    pdf_stem = "P6_Maths_2099_Test_resume"
    _write_checkpoint(extracted_dir, pdf_stem, [_make_pair(source_paper=source_paper)])

    write_extracted_data_to_db(extracted_dir, db_write_log_dir)
    assert _real_question_count(source_paper) == 1

    # Re-run with everything untouched, as a resumed batch would. It must not raise or duplicate.
    write_extracted_data_to_db(extracted_dir, db_write_log_dir)
    assert _real_question_count(source_paper) == 1


def test_incrementally_added_new_file_alongside_an_already_committed_one(tmp_path: Path):
    """A new file B is written normally while committed file A is left alone.

    A has a marker and rows; B has neither. A's committed state must not be mistaken for a desync.
    """
    extracted_dir = tmp_path / "extracted"
    db_write_log_dir = tmp_path / "db_write_log"
    source_paper_a = "P6_Maths_2099_Test_fileA.pdf"
    source_paper_b = "P6_Maths_2099_Test_fileB.pdf"

    _write_checkpoint(extracted_dir, "P6_Maths_2099_Test_fileA", [_make_pair(source_paper=source_paper_a)])
    write_extracted_data_to_db(extracted_dir, db_write_log_dir)
    assert _real_question_count(source_paper_a) == 1

    # File B arrives in a later Job A batch; file A's checkpoint and marker are untouched.
    _write_checkpoint(extracted_dir, "P6_Maths_2099_Test_fileB", [_make_pair(source_paper=source_paper_b)])
    write_extracted_data_to_db(extracted_dir, db_write_log_dir)

    assert _real_question_count(source_paper_a) == 1  # unchanged, not re-inserted
    assert _real_question_count(source_paper_b) == 1  # newly written


# =================================================================================================
# Rows sharing a natural key: the guard never merges or drops them
# =================================================================================================

def test_desync_guard_never_inspects_or_touches_rows_sharing_a_natural_key(tmp_path: Path):
    """The guard queries by source_paper only and never merges, drops or overwrites rows.

    As documented in _find_marker_db_desyncs(), it decides whether to refuse a whole run and
    ignores question_number, question_section and answer_source_page_index. Two different
    questions sharing a full natural key (one of five such groups in the corpus) are seeded;
    a later write for a different file must leave both rows intact.
    """
    extracted_dir = tmp_path / "extracted"
    db_write_log_dir = tmp_path / "db_write_log"
    db_write_log_dir.mkdir(parents=True)
    shared_source_paper = "P6_Maths_2022_SA2_nanyang_fake.pdf"

    # Two rows with the same natural key but different content (the nanyang Q4 shape), inserted
    # as if committed by a prior run, with the matching marker present. This is not a desync.
    (db_write_log_dir / f"{shared_source_paper.replace('.pdf', '')}.done").write_text("{}", encoding="utf-8")
    with get_session() as session:  # type: ignore
        row1 = _make_pair(
            question_number="4", source_paper=shared_source_paper,
            question_text="A garbled OMR instruction-block read, real fake residual row 1.",
        )["question"]
        row2 = _make_pair(
            question_number="4", source_paper=shared_source_paper,
            question_text="An unrelated rounding-calculation fragment, real fake residual row 2.",
        )["question"]
        session.add(Question(**row1))
        session.add(Question(**row2))
        session.flush()

    assert _real_question_count(shared_source_paper) == 2

    # An unrelated new file writes normally alongside the pair above.
    other_source_paper = "P6_Maths_2099_Test_unrelated.pdf"
    _write_checkpoint(extracted_dir, "P6_Maths_2099_Test_unrelated", [_make_pair(source_paper=other_source_paper)])
    write_extracted_data_to_db(extracted_dir, db_write_log_dir)

    # Both rows survive, neither merged nor dropped.
    assert _real_question_count(shared_source_paper) == 2
    assert _real_question_count(other_source_paper) == 1
