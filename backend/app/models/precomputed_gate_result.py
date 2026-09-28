"""
`precomputed_gate_results` table: gate results and hints computed offline, so the live endpoint
can serve them without running the gate.

The results live in a separate table rather than as columns on `questions`. `questions` holds the
source corpus written by the extraction pipeline (Issue 102), while a precomputed result is a
derived, versioned cache of running `gate.run_full_gate_pipeline()` once. Keeping them apart
avoids an unrelated write process touching cached state (the problem seen in Issue 158), and
follows the precedent of `ocr_extraction_records`.

Versioning: a cached row can only be trusted if the code and model that produced it are
unchanged. The live endpoint compares its current `gate.GATE_VERSION`, `llm_service.LLM_MODEL_PATH`
and `llm_service.LLM_ADAPTER_PATH`, and the question's current content fingerprint, against the
stored values. Any mismatch is treated as "no cached row", and the endpoint falls back to live
computation rather than serving a result produced under different rules.

History, not overwrite: a superseded row is marked `is_active=False` and never deleted, so past
results stay inspectable. A partial unique index allows only one active row per question,
enforced in the database rather than by convention (as with `hint_events.request_id`, Issue 57).

The stored fields are enough to rebuild what `generate_hint()` needs without recomputation: the
question type, the gate's pass/fail, tier, reason and detail (as JSON, since `GateResult.detail`
has no fixed shape), and the final hint tier and text.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sqlalchemy import Boolean, ForeignKey, Index, Integer, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.question import Question


class PrecomputedGateResult(Base):
    __tablename__ = "precomputed_gate_results"
    __table_args__ = (
        # Partial unique index: at most one active row per question ("history, not overwrite"
        # above). SQLite supports WHERE-qualified unique indexes directly.
        Index(
            "ix_precomputed_gate_results_active_question",
            "question_id",
            unique=True,
            sqlite_where=text("is_active = 1"),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    question_id: Mapped[int] = mapped_column(Integer, ForeignKey("questions.id"), nullable=False)

    # Versioning (see module docstring). Both are compared at read time, and a mismatch on either
    # means the row is treated as absent and the result is computed live.
    gate_version: Mapped[str] = mapped_column(String, nullable=False)
    llm_model_path: Mapped[str] = mapped_column(String, nullable=False)

    # `llm_service.LLM_ADAPTER_PATH` (empty string means base weights, no adapter), compared at
    # read time like gate_version and llm_model_path (Issue 369). Needed because the adapter
    # default was changed (Issue 363) without bumping GATE_VERSION, so cached rows could not
    # otherwise tell which adapter produced them. Nullable: the 3,557 rows that existed before
    # adapters were introduced have no adapter identity to backfill, and NULL never equals any
    # LLM_ADAPTER_PATH string (including ""), so those rows can never match.
    # Migration: scripts/migrations/migrate_precomputed_gate_results_adapter_path.py.
    adapter_path: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    # SHA-256 fingerprint of the question content the gate result was computed from (Issue 178;
    # see gate.compute_gate_content_fingerprint() for the fields covered). At read time it is
    # compared with a fingerprint of the question's current row, and a mismatch is handled like a
    # gate_version mismatch (recompute live). Nullable so that a row without a fingerprint never
    # matches by coincidence. The 3,580 rows that existed when
    # scripts/migrations/migrate_precomputed_gate_results_content_fingerprint.py ran were backfilled from
    # their linked question's current content. This is safe because `questions` rows are never
    # edited in place, so current content equals the content at cache time, and it avoided
    # invalidating the whole cache. New rows always get a fingerprint at write time.
    content_fingerprint: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    question_type: Mapped[str] = mapped_column(String, nullable=False)  # "sum" or "reasoning"

    # GateResult fields (gate.py), flattened for storage.
    tier_allowed: Mapped[int] = mapped_column(Integer, nullable=False)
    gate_passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    gate_reason: Mapped[str] = mapped_column(Text, nullable=False)
    gate_detail_json: Mapped[str] = mapped_column(Text, nullable=False)  # json.dumps(detail)

    # Output of generate_hint() for the gate result above. Generation is deterministic
    # (do_sample=False) for a fixed gate result, so caching the hint is reproducible.
    hint_tier: Mapped[int] = mapped_column(Integer, nullable=False)
    hint_text: Mapped[str] = mapped_column(Text, nullable=False)

    # String, not DateTime, as for Question.verified_at and HintEvent.created_at.
    computed_at: Mapped[str] = mapped_column(String, nullable=False)

    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    question: Mapped["Question"] = relationship()
