"""
`question_embeddings` table: an offline-precomputed embedding cache for the match-first step
(Issues 185-186).

It follows the design of `PrecomputedGateResult`. Embeddings are a derived, versioned cache, not
source corpus data, so they sit in a separate table rather than as columns on `questions`. Rows
are versioned by `embedding_model`: a row for a different model is treated as absent, never
served. A unique constraint allows one row per `(question_id, embedding_model)` pair.

Why precompute: the selected model (`Qwen/Qwen3-Embedding-0.6B`, Issue 185) takes about 1.8s per
text on the development machine's CPU. Embedding all 5,425 corpus questions on each photo
submission would add about 2.7 hours per request. Instead, only the new query is embedded at
request time (a couple of seconds) and compared against the precomputed corpus matrix.

Storage: each vector is a JSON list of floats in a `Text` column, the same convention as
`PrecomputedGateResult.gate_detail_json` and `StudentSubmission.detail`. At this scale a BLOB or
vector-extension dependency is not needed (Issue 52 decided against a vector database for RAG).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.question import Question


class QuestionEmbedding(Base):
    __tablename__ = "question_embeddings"
    __table_args__ = (
        # At most one row per (question, embedding_model) pair. A new model gets new rows rather
        # than overwriting the old ones, so a partly completed re-embed run never leaves a
        # question with no row. No is_active flag is needed: rows for an old model are simply not
        # selected by the current model's WHERE clause.
        UniqueConstraint("question_id", "embedding_model", name="uq_question_embeddings_q_model"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    question_id: Mapped[int] = mapped_column(Integer, ForeignKey("questions.id"), nullable=False)

    # The model that produced this vector. Vectors from different models live in different
    # embedding spaces and are not comparable, so this is part of the lookup key.
    embedding_model: Mapped[str] = mapped_column(String, nullable=False)

    # json.dumps(list[float]); see the module docstring for why not a BLOB or vector column.
    embedding_json: Mapped[str] = mapped_column(Text, nullable=False)

    # String, not DateTime, as for every other timestamp in this schema (see
    # Question.verified_at).
    computed_at: Mapped[str] = mapped_column(String, nullable=False)

    question: Mapped["Question"] = relationship()
