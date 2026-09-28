"""
One-time schema migration that adds `adapter_path` to `precomputed_gate_results` (Issue 369).

The precompute cache key did not include the adapter, a caution raised in Issue 352. The new
nullable column records which adapter each cached gate result was computed with.

A direct ALTER is enough; the table rebuild used by `migrate_attempts_nullable_question_id.py` is
not needed. `adapter_path` is a plain nullable column with no CHECK, foreign key or NOT NULL
constraint (the same convention as `content_fingerprint`, added by
`migrate_precomputed_gate_results_content_fingerprint.py`), which SQLite's
`ALTER TABLE ... ADD COLUMN` supports directly.

No backfill, unlike the content_fingerprint migration. That one could recompute a value for every
row from the current question content. Here there is nothing to recover: all 3,557 existing rows
were written before the adapter wiring of Issue 352 existed, so which adapter produced them is
unknown, and assuming base-only would be a guess. NULL is the correct permanent value for these
rows. NULL never equals any `LLM_ADAPTER_PATH` string at compare time (including the empty
string), so these rows can never match an adapter by accident, the same property
`content_fingerprint` has for its NULL rows.

A backup of psle.db is taken before the write, and the row count is checked before and after.

Usage:
    backend/.venv/Scripts/python.exe scripts/migrations/migrate_precomputed_gate_results_adapter_path.py
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

    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='precomputed_gate_results'",
    )
    if cur.fetchone() is None:
        print("ERROR: precomputed_gate_results table does not exist yet -- run init_db() with "
              "PrecomputedGateResult imported/registered first, then re-run this migration.")
        sys.exit(1)

    cur.execute("PRAGMA table_info(precomputed_gate_results)")
    existing_cols = {row[1] for row in cur.fetchall()}
    if "adapter_path" in existing_cols:
        print("adapter_path already exists on precomputed_gate_results -- migration already "
              "applied, nothing to do.")
        conn.close()
        return

    cur.execute("SELECT COUNT(*) FROM precomputed_gate_results")
    before_count = cur.fetchone()[0]
    print(f"Real precomputed_gate_results row count BEFORE migration: {before_count}")

    backup_path = DB_PATH.with_name(
        f"psle.db.bak_adapter_path_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
    )
    conn.close()  # release the file handle before copying it
    shutil.copy2(DB_PATH, backup_path)
    print(f"Backed up {DB_PATH.name} -> {backup_path.name} before writing")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("ALTER TABLE precomputed_gate_results ADD COLUMN adapter_path TEXT")
    conn.commit()

    cur.execute("PRAGMA table_info(precomputed_gate_results)")
    after_cols = {row[1] for row in cur.fetchall()}
    cur.execute("SELECT COUNT(*) FROM precomputed_gate_results")
    after_count = cur.fetchone()[0]
    print(f"Real precomputed_gate_results row count AFTER migration: {after_count}")

    assert "adapter_path" in after_cols, "ALTER TABLE reported success but the column is not really there"
    assert after_count == before_count, f"row count changed ({before_count} -> {after_count}) -- a real ADD COLUMN must never do this"

    cur.execute("SELECT COUNT(*) FROM precomputed_gate_results WHERE adapter_path IS NULL")
    null_count = cur.fetchone()[0]
    print(f"Rows with adapter_path IS NULL (expected: all {after_count}, deliberately not "
          f"backfilled -- see this script's own docstring): {null_count}")
    if null_count != after_count:
        print("Unexpected: some rows already had a non-NULL value for a column that was just "
              "added -- investigate before trusting this migration.")
        conn.close()
        sys.exit(1)

    print("Migration verified: adapter_path column added, row count unchanged, all rows NULL as "
          "expected, real backup on disk.")
    conn.close()


if __name__ == "__main__":
    main()
