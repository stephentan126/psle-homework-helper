"""
Shared SQLAlchemy declarative base for every ORM model in this package.

All models register on this one `Base`, so `Base.metadata` (used by `init_db()` in
`app/db/session.py`) sees every table. `app/models/__init__.py` holds the full list of models.
"""
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
