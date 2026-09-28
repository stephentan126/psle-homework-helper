"""
One-time schema migration that adds `tier3_similarity_score` and `tier3_degrade_reason` to
`hint_events`, so the Tier 3 path logs enough data to calibrate `RELATED_EXAMPLE_MIN_SIMILARITY`.

The gap: `generate_hint()` in `gate.py` computes a cosine-similarity score for every Tier 3
attempt (the `match_confidence` from `find_related_worked_example()`), and on a degrade it builds a
readable reason string that includes the score. Neither reached the `HintEvent(...)` built in
`routes.py`. The score was discarded after the floor check, and the degrade reason was never read
by the caller, so a calibration pass would have had only the coarse `tier` and `grounded_in`
columns to work from.

A direct ALTER is enough; no table rebuild is needed. Both are plain nullable columns with no
CHECK, foreign key or NOT NULL constraint (as in this table's `tier_allowed`, `grounded_in` and
`verified_content` migration), which SQLite's `ALTER TABLE ... ADD COLUMN` supports directly.

No backfill, as with the earlier additive migrations on this table. The similarity score of a past
Tier 3 attempt was never stored, and the degrade reason string was never saved to a column either.
NULL is the correct permanent value for those rows.

A backup of psle.db is taken before the write, and the row count is checked before and after.

Usage:
    backend/.venv/Scripts/python.exe scripts/migrations/migrate_hint_events_tier3_calibration_logging.py
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
    "tier3_similarity_score": "REAL",
    "tier3_degrade_reason": "TEXT",
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
        print("tier3_similarity_score/tier3_degrade_reason already exist on hint_events -- "
              "migration already applied, nothing to do.")
        conn.close()
        return

    cur.execute("SELECT COUNT(*) FROM hint_events")
    before_count = cur.fetchone()[0]
    print(f"Real hint_events row count BEFORE migration: {before_count}")

    backup_path = DB_PATH.with_name(
        f"psle.db.bak_tier3calibrationlogging_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
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

    cur.execute("SELECT COUNT(*) FROM hint_events WHERE tier3_similarity_score IS NULL")
    null_count = cur.fetchone()[0]
    print(f"Rows with tier3_similarity_score IS NULL (expected: all {after_count}, deliberately "
          f"not backfilled -- see this script's own docstring): {null_count}")
    if null_count != after_count:
        print("Unexpected: some rows already had a non-NULL value for a column that was just "
              "added -- investigate before trusting this migration.")
        conn.close()
        sys.exit(1)

    print("Migration verified: tier3_similarity_score/tier3_degrade_reason columns added, row "
          "count unchanged, all rows NULL as expected, real backup on disk.")
    conn.close()


if __name__ == "__main__":
    main()
