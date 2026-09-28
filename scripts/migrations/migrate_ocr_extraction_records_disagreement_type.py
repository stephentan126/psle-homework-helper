"""
One-time schema migration that adds a structured `disagreement_type` column to
`ocr_extraction_records`, next to the free-text `disagreement_detail`, and backfills existing rows
(Issue 212).

A direct ALTER is enough. The `migrate_attempts_nullable_question_id.py` migration needed a table
rebuild only because SQLite cannot add a CHECK constraint through `ALTER TABLE`.
`disagreement_type` is a plain nullable column with no CHECK, foreign key or NOT NULL constraint
(the project stores enums as strings, as explained in the comment on `agreement_status`), which
`ALTER TABLE ... ADD COLUMN` supports directly.

Backfill: the live `psle.db` had 130 `disagree` rows, in exactly two `disagreement_detail`
shapes. None used the current message templates of `compare_readers()` ("answer values not
equivalent..." or "...table/grid structure mismatch..."), because the data predates a change to
that function: it used to compare by "text similarity X < threshold Y", since replaced by
`values_equivalent()`. Both historical shapes are value-level comparisons (a whole-page text
similarity, and an independently parsed grid-answer conflict), so both map to "value_mismatch":
  - "text similarity ... < threshold ..." -> "value_mismatch" (72 rows at migration time)
  - "real grid-answer conflict for question ..." -> "value_mismatch" (58 rows)
No row in the corpus had a structural-mismatch shape. "structural_mismatch" is still a tested code
path in `compare_readers()`; it simply had no examples in the corpus at migration time.

A `disagree` row whose detail matches neither shape is left NULL and reported by id rather than
guessed.

Usage:
    backend/.venv/Scripts/python.exe scripts/migrations/migrate_ocr_extraction_records_disagreement_type.py
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = _REPO_ROOT / "psle.db"


def _classify(detail: str | None) -> str | None:
    """Classify the two known historical `disagreement_detail` shapes.

    Both are value-level comparisons; see the module docstring. Returns None for any other shape.
    """
    if detail is None:
        return None
    if detail.startswith("text similarity"):
        return "value_mismatch"
    if "real grid-answer conflict" in detail:
        return "value_mismatch"
    return None  # unrecognised shape: left NULL and reported below


def main() -> None:
    if not DB_PATH.exists():
        print(f"No real DB found at {DB_PATH} -- nothing to migrate (a fresh init_db() run will "
              f"create the new schema directly).")
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("PRAGMA table_info(ocr_extraction_records)")
    existing_columns = {row[1] for row in cur.fetchall()}
    if "disagreement_type" in existing_columns:
        print("disagreement_type already exists on ocr_extraction_records -- migration already "
              "applied, nothing to do.")
        conn.close()
        return

    cur.execute("SELECT COUNT(*) FROM ocr_extraction_records")
    before_count = cur.fetchone()[0]
    print(f"Real ocr_extraction_records row count BEFORE migration: {before_count}")

    cur.execute("ALTER TABLE ocr_extraction_records ADD COLUMN disagreement_type TEXT")
    conn.commit()
    print("Column added.")

    if before_count == 0:
        print("Table is empty -- no real rows to backfill. Migration complete.")
        conn.close()
        return

    cur.execute(
        "SELECT id, agreement_status, disagreement_detail FROM ocr_extraction_records "
        "WHERE agreement_status = 'disagree'",
    )
    disagree_rows = cur.fetchall()
    print(f"Real 'disagree' rows to classify: {len(disagree_rows)}")

    counts = {"value_mismatch": 0, "structural_mismatch": 0, "unrecognized": 0}
    unrecognized_ids: list[int] = []
    for row_id, _status, detail in disagree_rows:
        classification = _classify(detail)
        if classification is None:
            counts["unrecognized"] += 1
            unrecognized_ids.append(row_id)
            continue
        counts[classification] += 1
        cur.execute(
            "UPDATE ocr_extraction_records SET disagreement_type = ? WHERE id = ?",
            (classification, row_id),
        )
    conn.commit()

    print(f"\nReal backfill result: value_mismatch={counts['value_mismatch']}, "
          f"structural_mismatch={counts['structural_mismatch']}, "
          f"unrecognized (left NULL)={counts['unrecognized']}")
    if unrecognized_ids:
        print(f"WARNING: {len(unrecognized_ids)} real 'disagree' row(s) matched neither known "
              f"historical shape -- left NULL, ids: {unrecognized_ids[:20]}"
              f"{' (truncated)' if len(unrecognized_ids) > 20 else ''}")

    cur.execute("SELECT COUNT(*) FROM ocr_extraction_records")
    after_count = cur.fetchone()[0]
    print(f"\nReal ocr_extraction_records row count AFTER migration: {after_count}")
    if after_count != before_count:
        print(f"WARNING: row count changed during migration ({before_count} -> {after_count}) -- "
              f"investigate before trusting this migration.")
        sys.exit(1)
    print("Real row count preserved exactly. Migration committed.")

    # Check that no row outside 'disagree' was given a disagreement_type.
    cur.execute(
        "SELECT COUNT(*) FROM ocr_extraction_records WHERE agreement_status != 'disagree' "
        "AND disagreement_type IS NOT NULL",
    )
    stray = cur.fetchone()[0]
    if stray:
        print(f"REAL PROBLEM: {stray} non-'disagree' row(s) somehow have a non-NULL "
              f"disagreement_type -- investigate.")
        sys.exit(1)
    print("Confirmed: disagreement_type is NULL on every non-'disagree' row, as expected.")

    conn.close()


if __name__ == "__main__":
    main()
