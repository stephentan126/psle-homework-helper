"""
One-time schema migration that adds `tier_allowed`, `grounded_in` and `verified_content` to
`hint_events`, for the multi-hint escalation ladder.

Before this, nothing durably recorded the gate's per-hint `tier_allowed` ceiling, which of
`generate_hint()`'s three grounding branches was taken, or the verified content used to build the
hint (for the two branches where it cannot be reached through the existing
`attempts -> questions.worked_solution_text` join).

A direct ALTER is enough; the table rebuild used by `migrate_attempts_nullable_question_id.py` is
not needed. All three are plain nullable columns with no CHECK, foreign key or NOT NULL constraint
(the same convention as `precomputed_gate_results.adapter_path` and `content_fingerprint`), which
SQLite's `ALTER TABLE ... ADD COLUMN` supports directly.

No backfill, for the same reason as the `adapter_path` migration. These values were never stored
for past hints. `precomputed_gate_results` does not cover `grounded_in`, and it only covers
`tier_allowed` for offline-precomputed, question-keyed rows, not every past hint event. Any value
for old rows would be a guess, so NULL is their correct permanent value.

A backup of psle.db is taken before the write, and the row count is checked before and after.

Usage:
    backend/.venv/Scripts/python.exe scripts/migrations/migrate_hint_events_tier3_fields.py
"""
from __future__ import annotations

import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = _REPO_ROOT / "psle.db"

_NEW_COLUMNS = {
    "tier_allowed": "INTEGER",
    "grounded_in": "TEXT",
    "verified_content": "TEXT",
}


def main() -> None:
    if not DB_PATH.exists():
        print(f"No real DB found at {DB_PATH} -- nothing to migrate (a fresh init_db() run will "
              f"create the new schema directly).")
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='hint_events'")
    if cur.fetchone() is None:
        print("ERROR: hint_events table does not exist yet -- run init_db() first, then re-run "
              "this migration.")
        sys.exit(1)

    cur.execute("PRAGMA table_info(hint_events)")
    existing_cols = {row[1] for row in cur.fetchall()}
    missing = [c for c in _NEW_COLUMNS if c not in existing_cols]
    if not missing:
        print("tier_allowed/grounded_in/verified_content already exist on hint_events -- "
              "migration already applied, nothing to do.")
        conn.close()
        return

    cur.execute("SELECT COUNT(*) FROM hint_events")
    before_count = cur.fetchone()[0]
    print(f"Real hint_events row count BEFORE migration: {before_count}")

    backup_path = DB_PATH.with_name(
        f"psle.db.bak_multihint_tier3_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
    )
    conn.close()  # release the file handle before copying it
    shutil.copy2(DB_PATH, backup_path)
    print(f"Backed up {DB_PATH.name} -> {backup_path.name} before writing")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    for col in missing:
        cur.execute(f"ALTER TABLE hint_events ADD COLUMN {col} {_NEW_COLUMNS[col]}")
        print(f"Added column: {col} {_NEW_COLUMNS[col]}")
    conn.commit()

    cur.execute("PRAGMA table_info(hint_events)")
    after_cols = {row[1] for row in cur.fetchall()}
    cur.execute("SELECT COUNT(*) FROM hint_events")
    after_count = cur.fetchone()[0]
    print(f"Real hint_events row count AFTER migration: {after_count}")

    for col in _NEW_COLUMNS:
        assert col in after_cols, f"ALTER TABLE reported success but {col} is not really there"
    assert after_count == before_count, (
        f"row count changed ({before_count} -> {after_count}) -- a real ADD COLUMN must never do this"
    )

    cur.execute("SELECT COUNT(*) FROM hint_events WHERE tier_allowed IS NULL")
    null_count = cur.fetchone()[0]
    print(f"Rows with tier_allowed IS NULL (expected: all {after_count}, deliberately not "
          f"backfilled -- see this script's own docstring): {null_count}")
    if null_count != after_count:
        print("Unexpected: some rows already had a non-NULL value for a column that was just "
              "added -- investigate before trusting this migration.")
        conn.close()
        sys.exit(1)

    print("Migration verified: tier_allowed/grounded_in/verified_content columns added, row "
          "count unchanged, all rows NULL as expected, real backup on disk.")
    conn.close()


if __name__ == "__main__":
    main()
