"""
`ocr_extraction_records` table (design specification, Section 10.1): one row per question,
recording whether the two OCR readers (MinerU2.5 and DeepSeek-OCR) agreed.

Rows are produced by `segment_and_compare_page()` and `compare_readers()` in
`app/pipeline/extract.py` (Issues 98-101), with a per-question grid cross-check from
`_apply_grid_cross_check()` (Issue 101).

`agreement_status` takes a fourth value, `"both_failed"`, which Section 10.1 does not list. The
column is a plain String, so it accepts the value without change.

`disagreement_type` (Issue 212) makes disagreements countable by query instead of by reading the
free-text `disagreement_detail`. Current values, taken from the branches of `compare_readers()`:
`"value_mismatch"`, `"structural_mismatch"`, or NULL. NULL covers rows with no disagreement
(`agree`, `both_failed`, `single_reader_only`) and older rows whose detail text could not be
mapped to a known category (see
`scripts/migrations/migrate_ocr_extraction_records_disagreement_type.py`). A value of `"both"`
cannot occur yet: `compare_readers()` checks values first, and a value mismatch skips the
structural check.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.question import Question


class OCRExtractionRecord(Base):
    __tablename__ = "ocr_extraction_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    question_id: Mapped[int] = mapped_column(Integer, ForeignKey("questions.id"), nullable=False)

    primary_reader_output: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    secondary_reader_output: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # "agree" / "disagree" / "single_reader_only" / "both_failed". The `AgreementStatus` Literal
    # in extract.py is the single source of truth. A plain String rather than an Enum: SQLite has
    # no native enum type, so SQLAlchemy would only enforce it in Python, duplicating the Literal
    # and risking the two drifting apart.
    agreement_status: Mapped[str] = mapped_column(String, nullable=False)
    disagreement_detail: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    disagreement_type: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    # String, not DateTime, for the same reason as Question.verified_at (app/models/question.py).
    # extract.py always writes `datetime.now(timezone.utc).isoformat()`.
    extracted_at: Mapped[str] = mapped_column(String, nullable=False)

    question: Mapped["Question"] = relationship(back_populates="ocr_records")
