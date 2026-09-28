"""
One-time schema migration for `attempts` (Issue 180).

`question_id` becomes nullable, a nullable `student_submission_id` foreign key is added, and a
database-level CHECK constraint requires exactly one of the two to be set. The CHECK is defence in
depth alongside the application-layer check.

SQLite's ALTER TABLE cannot drop a NOT NULL constraint or add a CHECK constraint to an existing
table. The standard approach is used instead: create a new table with the desired schema, copy
the existing rows across, drop the old table and rename the new one into place. Row counts are
checked before and after, even though `attempts` held 0 rows when this was written. The script
then tries two invalid inserts to confirm the CHECK constraint is enforced.

A hand-written script is used rather than Alembic, which is not a project dependency (see
`backend/pyproject.toml`). Adding a migration framework for one narrow schema change would be a
larger dependency than the change warrants.

Usage: backend/.venv/Scripts/python.exe scripts/migrations/migrate_attempts_nullable_question_id.py
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = _REPO_ROOT / "psle.db"

NEW_ATTEMPTS_SCHEMA = """
CREATE TABLE attempts_new (
    id INTEGER NOT NULL,
    student_id INTEGER NOT NULL,
    question_id INTEGER,
    student_submission_id INTEGER,
    current_tier INTEGER NOT NULL,
    status VARCHAR,
    started_at VARCHAR NOT NULL,
    last_activity_at VARCHAR NOT NULL,
    PRIMARY KEY (id),
    FOREIGN KEY(student_id) REFERENCES students (id),
    FOREIGN KEY(question_id) REFERENCES questions (id),
    FOREIGN KEY(student_submission_id) REFERENCES student_submissions (id),
    CHECK ((question_id IS NOT NULL) != (student_submission_id IS NOT NULL))
)
"""


def main() -> None:
    if not DB_PATH.exists():
        print(f"No real DB found at {DB_PATH} -- nothing to migrate (a fresh init_db() run will "
              f"create the new schema directly).")
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # The foreign key target must exist for attempts_new's FOREIGN KEY clause to be meaningful.
    # SQLite does not enforce this at CREATE TABLE time, so check it here.
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='student_submissions'")
    if cur.fetchone() is None:
        print("ERROR: student_submissions table does not exist yet -- run init_db() with "
              "StudentSubmission imported/registered FIRST, then re-run this migration.")
        sys.exit(1)

    cur.execute("SELECT COUNT(*) FROM attempts")
    before_count = cur.fetchone()[0]
    print(f"Real attempts row count BEFORE migration: {before_count}")

    cur.execute("DROP TABLE IF EXISTS attempts_new")  # a failed earlier run must not block
                                                        # a clean retry
    cur.execute(NEW_ATTEMPTS_SCHEMA)

    # Copy every existing row across. question_id was NOT NULL, so every existing row already
    # satisfies the new CHECK (question_id set, student_submission_id NULL).
    cur.execute("""
        INSERT INTO attempts_new
            (id, student_id, question_id, student_submission_id, current_tier, status,
             started_at, last_activity_at)
        SELECT id, student_id, question_id, NULL, current_tier, status,
               started_at, last_activity_at
        FROM attempts
    """)
    copied = cur.rowcount
    print(f"Real rows copied into the new schema: {copied}")

    if copied != before_count:
        conn.rollback()
        print(f"ABORTING, not committing: copied {copied} rows but real attempts table had "
              f"{before_count} -- a real data-loss risk, not proceeding.")
        sys.exit(1)

    cur.execute("DROP TABLE attempts")
    cur.execute("ALTER TABLE attempts_new RENAME TO attempts")
    conn.commit()

    cur.execute("SELECT COUNT(*) FROM attempts")
    after_count = cur.fetchone()[0]
    print(f"Real attempts row count AFTER migration: {after_count}")

    if after_count != before_count:
        print(f"WARNING: row count mismatch after migration ({before_count} -> {after_count}) "
              f"-- investigate before trusting this migration, even though it already committed.")
        sys.exit(1)

    print("Real row count preserved exactly. Migration committed.")

    # Confirm the CHECK constraint is enforced by attempting two invalid inserts, rather than
    # relying on the DDL alone.
    print("\nVerifying the CHECK constraint actually enforces (a real bad insert, expected to fail)...")
    try:
        cur.execute("""
            INSERT INTO attempts (student_id, question_id, student_submission_id, current_tier,
                                   started_at, last_activity_at)
            VALUES (1, 1, 1, 1, 'x', 'x')
        """)
        conn.commit()
        print("REAL PROBLEM: the CHECK constraint did NOT reject a row with BOTH question_id and "
              "student_submission_id set -- constraint is not actually enforcing.")
        sys.exit(1)
    except sqlite3.IntegrityError as e:
        conn.rollback()
        print(f"Confirmed: CHECK constraint correctly rejected a both-set row ({e}).")

    try:
        cur.execute("""
            INSERT INTO attempts (student_id, question_id, student_submission_id, current_tier,
                                   started_at, last_activity_at)
            VALUES (1, NULL, NULL, 1, 'x', 'x')
        """)
        conn.commit()
        print("REAL PROBLEM: the CHECK constraint did NOT reject a row with NEITHER set -- "
              "constraint is not actually enforcing.")
        sys.exit(1)
    except sqlite3.IntegrityError as e:
        conn.rollback()
        print(f"Confirmed: CHECK constraint correctly rejected a neither-set row ({e}).")

    # The two failed test inserts were rolled back; confirm no test row was left behind.
    cur.execute("SELECT COUNT(*) FROM attempts")
    final_count = cur.fetchone()[0]
    if final_count != after_count:
        print(f"WARNING: real row count changed during CHECK-constraint testing "
              f"({after_count} -> {final_count}) -- investigate.")
        sys.exit(1)
    print(f"\nReal final attempts row count: {final_count} (unchanged by constraint testing). "
          f"Migration fully verified.")

    conn.close()


if __name__ == "__main__":
    main()
