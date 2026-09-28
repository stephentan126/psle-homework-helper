"""
One-time schema migration that adds `ocr_raw_text` to `student_submissions` (Issue 369).

The new nullable column keeps an audit copy of the originally submitted text, which is never
overwritten by a later edit. `question_text` then holds the current, possibly edited text. Run once
against an existing psle.db; a fresh init_db() creates the new schema directly.

Unlike `migrate_attempts_nullable_question_id.py`, this does not rebuild the table. That migration
had to add a CHECK constraint and relax a NOT NULL constraint, which SQLite's `ALTER TABLE` cannot
do, so it created a new table, copied the rows and renamed it into place. Adding one nullable
column with no constraint is supported directly by `ALTER TABLE ... ADD COLUMN`, so the simpler
path is used. The same safety steps are kept: a backup of psle.db is taken first, and the row count
is checked before and after.

Usage:
    backend/.venv/Scripts/python.exe scripts/migrations/migrate_student_submissions_ocr_raw_text.py
"""
from __future__ import annotations

import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = _REPO_ROOT / "psle.db"


def main() -> None:
    if not DB_PATH.exists():
        print(f"No real DB found at {DB_PATH} -- nothing to migrate (a fresh init_db() run will "
              f"create the new schema directly).")
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='student_submissions'")
    if cur.fetchone() is None:
        print("ERROR: student_submissions table does not exist yet -- run init_db() with "
              "StudentSubmission imported/registered first, then re-run this migration.")
        sys.exit(1)

    cur.execute("PRAGMA table_info(student_submissions)")
    existing_cols = {row[1] for row in cur.fetchall()}
    if "ocr_raw_text" in existing_cols:
        print("ocr_raw_text already exists on student_submissions -- migration already applied, nothing to do.")
        conn.close()
        return

    cur.execute("SELECT COUNT(*) FROM student_submissions")
    before_count = cur.fetchone()[0]
    print(f"Real student_submissions row count BEFORE migration: {before_count}")

    backup_path = DB_PATH.with_name(f"psle.db.bak_ocr_raw_text_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}")
    conn.close()  # release the file handle before copying it
    shutil.copy2(DB_PATH, backup_path)
    print(f"Backed up {DB_PATH.name} -> {backup_path.name} before writing")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("ALTER TABLE student_submissions ADD COLUMN ocr_raw_text TEXT")
    conn.commit()

    cur.execute("PRAGMA table_info(student_submissions)")
    after_cols = {row[1] for row in cur.fetchall()}
    cur.execute("SELECT COUNT(*) FROM student_submissions")
    after_count = cur.fetchone()[0]
    print(f"Real student_submissions row count AFTER migration: {after_count}")

    assert "ocr_raw_text" in after_cols, "ALTER TABLE reported success but the column is not really there"
    assert after_count == before_count, f"row count changed ({before_count} -> {after_count}) -- a real ADD COLUMN must never do this"
    print("Migration verified: ocr_raw_text column added, row count unchanged, real backup on disk.")
    conn.close()


if __name__ == "__main__":
    main()
