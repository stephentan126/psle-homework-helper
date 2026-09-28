"""
`safety_flags` table (design specification, Section 10.1). Kept separate from `hint_events`
because it has stricter access rules (the child safety layer, Section 7).

`attempt_id` is nullable. Section 7 screens every student message before it reaches the tutoring
model, and a message can be typed before any attempt exists (for example, before a question is
loaded).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sqlalchemy import ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.attempt import Attempt
    from app.models.student import Student


class SafetyFlag(Base):
    __tablename__ = "safety_flags"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    student_id: Mapped[int] = mapped_column(Integer, ForeignKey("students.id"), nullable=False)
    attempt_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("attempts.id"), nullable=True,
    )

    # "input" or "output", the two screening points in Section 7. A plain String rather than an
    # enum, for the same reason as OCRExtractionRecord.agreement_status.
    flagged_input_or_output: Mapped[str] = mapped_column(String, nullable=False)
    category: Mapped[str] = mapped_column(String, nullable=False)

    # String, not DateTime, for the same reason as Question.verified_at (app/models/question.py).
    # NULL until the mandatory parent alert (Section 7) is sent.
    parent_notified_at: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    student: Mapped["Student"] = relationship(back_populates="safety_flags")
    attempt: Mapped[Optional["Attempt"]] = relationship(back_populates="safety_flags")
