"""
SQLAlchemy engine and session setup (design specification, Section 10.1).

Import path: this module and `app/models/` import each other as `app.X`, not `backend.app.X`
(Issue 102). The editable install maps the top-level package `app` to `backend/app/`, so `app.X`
resolves from any working directory. `backend.app.X` only resolves when the process starts in the
repository root, because it relies on the current directory being on `sys.path`.

Database location (Issue 160): the default `DATABASE_URL` is anchored to the repository root, not
the current working directory. Components are started from different directories (`extract.py`
from the repository root, `uvicorn app.main:app` from `backend/`). With the old CWD-relative
default (`sqlite:///./psle.db`), running from `backend/` silently opened a separate
`backend/psle.db`, and a question ID returned the wrong question with no error. An explicit
`DATABASE_URL` environment variable (deployment, tests) still takes precedence.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.models.base import Base

# backend/app/db/session.py -> parents[0]=db, [1]=app, [2]=backend, [3]=repository root.
_REPO_ROOT = Path(__file__).resolve().parents[3]

# An explicit DATABASE_URL always wins (see backend/tests/conftest.py for why tests must never
# use the default file). Otherwise use an absolute path under the repository root (Issue 160).
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{(_REPO_ROOT / 'psle.db').as_posix()}")

engine: Engine = create_engine(DATABASE_URL, future=True)


@event.listens_for(engine, "connect")
def _enable_wal_mode(dbapi_connection, connection_record) -> None:  # noqa: ANN001
    """Enables WAL mode and a busy timeout on each new SQLite connection.

    WAL mode is more resistant to corruption on an unclean shutdown than the default rollback
    journal (design specification, "SQLite reliability"; Issue 65). PRAGMA settings are
    connection-scoped in SQLite, so they are applied on every new connection rather than once at
    startup. The arguments follow SQLAlchemy's event-hook signature, so they are not annotated.
    """
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    # SQLite's default busy_timeout is 0ms, so a second writer fails at once instead of waiting.
    # This caused `database is locked` errors under two concurrent requests (the idempotency-guard
    # race test in routes.py; Issue 168). 5000ms covers the app's short single-row transactions
    # without hiding a longer deadlock.
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    """Creates every table registered on `Base.metadata` that does not already exist.

    Idempotent (`create_all` uses `checkfirst=True`), so it is safe to call on every run. Called at
    application startup and at the start of `write_extracted_data_to_db()` in extract.py, so
    neither depends on a separate manual initialisation step.
    """
    Base.metadata.create_all(bind=engine)


@contextmanager
def get_session() -> Iterator[Session]:
    """Yields a session for one unit of work, committing on success and rolling back on error.

    Callers such as `write_extracted_data_to_db()` still commit explicitly where they need to. The
    commit on exit is a safety net and does nothing if no changes are pending.
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
