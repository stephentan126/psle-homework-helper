"""
`student_submissions` table: one row per photographed homework question (Issue 180).

This is a separate table rather than new rows in `questions`. The NOT NULL provenance columns on
`questions` (`source_paper`, `answer_source_file`, `answer_source_page_index`, `question_number`)
record where in an exam paper a question came from. A student's photo has none of that, and
placeholder values would undo the guarantee those constraints give. `ocr_extraction_records` and
`precomputed_gate_results` are kept separate from `questions` for similar reasons.

Unlike `precomputed_gate_results`, this is not a cache. Each row is a one-off audit record of one
submission and is never reused to skip a live run, so it has no `gate_version` or
`llm_model_path` columns.

The two paths of Issue 180 are reflected in the columns:
- Match-first (`matched_question_id` set): the photo matched a verified `questions` row with
  enough confidence. That row's `answer_value` and `worked_solution_text` are used, and the gate
  pipeline and precompute cache run exactly as for any corpus question.
- Weak mode (`matched_question_id` NULL): no confident match. Self-consistency runs without an
  answer key, relying only on agreement between passes. This is a weaker signal (a model that is
  consistently wrong cannot be ruled out, Section 5), so the result is capped at Tier 1 (Issue 180
  addendum). The cap should not be raised without revisiting that decision.

`gate_path_used` records which path produced a result. Downstream consumers (the response
builder, the frontend, analysis) must read it directly rather than infer the path from other
fields (Issue 180, item 4).

`submitted_at` is a String, like every other timestamp in the schema (see
`Question.verified_at`), because SQLite's `DateTime` storage format does not accept the
timezone-aware ISO-8601 strings the app writes (Issue 102).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sqlalchemy import Boolean, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.question import Question
    from app.models.student import Student


class StudentSubmission(Base):
    __tablename__ = "student_submissions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    student_id: Mapped[int] = mapped_column(Integer, ForeignKey("students.id"), nullable=False)

    # The current text, which the student may have edited (Issue 369). The pipeline (safety
    # check, match-first, the gate) acts on this column. Its only readers, the weak-mode gate call
    # in `confirm_photo_submission()` and its audit-response field, both need the current text.
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    # The original text as first submitted (Issue 369): the VLM transcription from
    # `submit_photo()`, or the pre-extracted text passed to `submit_photo_question()`. Set once at
    # creation and never overwritten, so an edit never loses the original record.
    ocr_raw_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    has_diagram: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    photo_path: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    # String, not DateTime (see module docstring).
    submitted_at: Mapped[str] = mapped_column(String, nullable=False)

    # Match-first path (Issue 180, item 1). NULL means weak mode was used.
    matched_question_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("questions.id"), nullable=True,
    )
    # Set whenever a match attempt runs, whether it succeeds or not (Issue 180, item 5). NULL
    # only if the match step never ran.
    match_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    match_method: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    # Weak-mode fields, set only when matched_question_id is NULL. Same shape as the equivalent
    # PrecomputedGateResult fields.
    tier_allowed: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    reason: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    detail: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON, same convention
                                                                          # as PrecomputedGateResult

    # Which path produced the result (Issue 180, item 4); see module docstring.
    gate_path_used: Mapped[str] = mapped_column(String, nullable=False)

    student: Mapped["Student"] = relationship()
    matched_question: Mapped[Optional["Question"]] = relationship()
