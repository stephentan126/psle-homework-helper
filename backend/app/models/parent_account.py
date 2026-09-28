"""
`parent_accounts` table (design specification, Section 10.1): parent identity and PIN security
state.

The PIN is hashed with `argon2-cffi`, never stored in plain text (Section 10.2). `pin_hash` holds
argon2's self-contained encoded hash, which already embeds a random salt, so `pin_salt` is left
unused (NULL) by `app/pipeline/parent_auth.py`. The column is kept because Section 10.1 lists it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.student import Student


class ParentAccount(Base):
    __tablename__ = "parent_accounts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    pin_hash: Mapped[str] = mapped_column(String, nullable=False)
    pin_salt: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    failed_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # String, not DateTime, for the same reason as Question.verified_at (app/models/question.py):
    # timestamps are written with datetime.now(timezone.utc).isoformat(), which does not match
    # SQLite's DateTime storage format.
    locked_until: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    students: Mapped[list["Student"]] = relationship(back_populates="parent_account")
