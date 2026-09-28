"""
`students` table (design specification, Section 10.1): one row per child using the app.

Students have no password (Section 10.2). A lightweight session is tied to this row, which is
created when a parent sets the child up.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List

from sqlalchemy import ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.attempt import Attempt
    from app.models.parent_account import ParentAccount
    from app.models.safety_flag import SafetyFlag


class Student(Base):
    __tablename__ = "students"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    display_name: Mapped[str] = mapped_column(String, nullable=False)
    parent_account_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("parent_accounts.id"), nullable=False,
    )

    parent_account: Mapped["ParentAccount"] = relationship(back_populates="students")
    attempts: Mapped[List["Attempt"]] = relationship(back_populates="student")
    safety_flags: Mapped[List["SafetyFlag"]] = relationship(back_populates="student")
