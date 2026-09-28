"""
`attempts` table (design specification, Section 10.1): lets progress survive a browser refresh.

`current_tier` is a plain Integer (1, 2 or 3, the escalation ladder in Section 5) with no CHECK
constraint. The valid range is enforced by the gate logic, for the same reason
`OCRExtractionRecord.agreement_status` is a plain String rather than an enum. `status` is also a
plain String with no default, as the specification does not fix its value set.

Each attempt refers to exactly one of two sources (Issue 180):
- `question_id`: a verified-match attempt, either a corpus question or a photo that matched one.
- `student_submission_id`: a weak-mode attempt, where a photo had no confident match.

Both columns are therefore nullable, and a CHECK constraint requires exactly one to be set. The
columns were changed by `scripts/migrations/migrate_attempts_nullable_question_id.py`, which rebuilds and
copies the table because SQLite cannot alter a NOT NULL constraint or add a CHECK constraint to
an existing table. The constraint was tested with both-set and neither-set inserts.

Two alternatives were rejected. A polymorphic `(subject_type, subject_id)` pair would have
changed every existing query that reads `attempts.question_id`. Inserting a placeholder
`questions` row for each weak-mode submission would undo the reason `student_submissions` is kept
separate from `questions` (see that model's docstring).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import CheckConstraint, ForeignKey, Integer, String, event
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.hint_event import HintEvent
    from app.models.question import Question
    from app.models.safety_flag import SafetyFlag
    from app.models.student import Student
    from app.models.student_submission import StudentSubmission


class Attempt(Base):
    __tablename__ = "attempts"
    __table_args__ = (
        # Database-level check, paired with the application-level check below, so the
        # invariant holds even for writes that bypass the ORM.
        CheckConstraint(
            "(question_id IS NOT NULL) != (student_submission_id IS NOT NULL)",
            name="ck_attempts_exactly_one_of_question_or_submission",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    student_id: Mapped[int] = mapped_column(Integer, ForeignKey("students.id"), nullable=False)
    question_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("questions.id"), nullable=True,
    )
    student_submission_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("student_submissions.id"), nullable=True,
    )

    current_tier: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    status: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    # String, not DateTime, for the same reason as Question.verified_at: timestamps are written as
    # datetime.now(timezone.utc).isoformat(), which SQLite's DateTime storage format rejects.
    started_at: Mapped[str] = mapped_column(String, nullable=False)
    last_activity_at: Mapped[str] = mapped_column(String, nullable=False)

    student: Mapped["Student"] = relationship(back_populates="attempts")
    question: Mapped[Optional["Question"]] = relationship()
    student_submission: Mapped[Optional["StudentSubmission"]] = relationship()
    hint_events: Mapped[List["HintEvent"]] = relationship(
        back_populates="attempt", cascade="all, delete-orphan",
    )
    safety_flags: Mapped[List["SafetyFlag"]] = relationship(back_populates="attempt")


@event.listens_for(Attempt, "before_insert")
@event.listens_for(Attempt, "before_update")
def _validate_exactly_one_question_source(mapper, connection, target: Attempt) -> None:  # noqa: ANN001
    """Checks that exactly one of `question_id` and `student_submission_id` is set.

    Complements the CHECK constraint above rather than replacing it. Runs as a pre-flush event
    rather than `@validates`, because `@validates` fires on each attribute assignment and would
    reject a valid object built in two steps before both fields are set.

    Raises:
        ValueError: If both or neither of the two fields are set.
    """
    has_question = target.question_id is not None
    has_submission = target.student_submission_id is not None
    if has_question == has_submission:  # both set, or neither set
        raise ValueError(
            f"Attempt must have exactly one of question_id/student_submission_id set, not "
            f"{'both' if has_question else 'neither'} (question_id={target.question_id!r}, "
            f"student_submission_id={target.student_submission_id!r})."
        )
