"""
`hint_events` table (design specification, Section 10.1): an audit trail of every hint shown to
the child, used for debugging and as evaluation evidence.

`self_consistency_result` is nullable Text rather than a fixed enum. The multi-sample
self-consistency check for reasoning questions (Section 5) produces a multi-line outcome (which
passes agreed, and whether they matched the answer key). It is NULL for sum-type questions, which
are checked with SymPy and never run self-consistency.

`request_id` is the client-supplied ID the row was created for (Section 10.5, Issue 57). It is
nullable, since not every write path supplies one, but UNIQUE when present. The gate endpoint
looks for an existing row with the same `request_id` before running the gate and, if one exists,
returns its outcome unchanged. The UNIQUE constraint closes the race window between that check
and the insert, which a check in code alone would leave open.

`tier_allowed`, `grounded_in` and `verified_content` record how each hint was grounded:
- `tier_allowed` is the ceiling set by the gate, which `generate_hint()` gated on. It differs
  from `tier`, which is the tier shown.
- `grounded_in` is the grounding branch `generate_hint()` took: `None`,
  `"worked_solution_text"`, `"sympy_confirmed_expression"` or
  `"self_consistency_confirmed_value"`.
- `verified_content` is the verified content used for the last two branches. It is left NULL for
  `"worked_solution_text"`, since that text is already reachable through
  `attempt -> questions.worked_solution_text`.
All three are NULL for rows written before `scripts/migrations/migrate_hint_events_tier3_fields.py` (no
backfill value can be computed, see that script's docstring) and for tier-1-capped or weak-mode
hints, where no grounding took place.

`tier3_similarity_score` and `tier3_degrade_reason` record the Tier 3 related-example retrieval
decision in `gate.generate_hint()`: the cosine similarity it computed and, if it fell back to
Tier 2, why (for example, a score below `RELATED_EXAMPLE_MIN_SIMILARITY`). Both are NULL when
Tier 3 was never attempted (weak mode, `tier_allowed < 2`, or an explicit `requested_tier` other
than 3) and for rows written before the migration. Read them together with `tier`:
- never attempted: both NULL, `tier != 3`;
- attempted and served: score set, reason NULL, `tier == 3`;
- attempted and degraded: reason set (score set or not, depending on the failure), `tier == 2`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sqlalchemy import Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.attempt import Attempt


class HintEvent(Base):
    __tablename__ = "hint_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    attempt_id: Mapped[int] = mapped_column(Integer, ForeignKey("attempts.id"), nullable=False)

    tier: Mapped[int] = mapped_column(Integer, nullable=False)
    hint_text: Mapped[str] = mapped_column(Text, nullable=False)
    self_consistency_result: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    request_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, unique=True)

    tier_allowed: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    grounded_in: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    verified_content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    tier3_similarity_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tier3_degrade_reason: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    # String, not DateTime, for the same reason as Question.verified_at (app/models/question.py).
    created_at: Mapped[str] = mapped_column(String, nullable=False)

    attempt: Mapped["Attempt"] = relationship(back_populates="hint_events")
