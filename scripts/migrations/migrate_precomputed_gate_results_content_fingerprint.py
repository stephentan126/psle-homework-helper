"""
One-time schema migration that adds `content_fingerprint` to `precomputed_gate_results` and
backfills existing rows (Issue 178).

A direct ALTER is enough here; the table rebuild used by `migrate_attempts_nullable_question_id.py`
is not needed. `content_fingerprint` is a plain nullable column with no CHECK, foreign key or NOT
NULL constraint (the same convention as `ocr_extraction_records.disagreement_type`), which SQLite's
`ALTER TABLE ... ADD COLUMN` supports directly.

Backfill: the live `psle.db` had 3,580 existing `precomputed_gate_results` rows, none with a
fingerprint. Each is backfilled from the current content of its linked `questions` row, using
`gate.compute_gate_content_fingerprint()`, the same function the live endpoint and the precompute
script use. This is correct because nothing in the project edits an existing `questions` row in
place, so the current content is the content at the time the result was cached. A row whose
`question_id` no longer resolves to a question (not expected, given the foreign key) is left NULL
and reported by id, as in the `disagreement_type` migration.

Usage:
    backend/.venv/Scripts/python.exe scripts/migrations/migrate_precomputed_gate_results_content_fingerprint.py
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "backend"))
DB_PATH = _REPO_ROOT / "psle.db"


def main() -> None:
    if not DB_PATH.exists():
        print(f"No real DB found at {DB_PATH} -- nothing to migrate (a fresh init_db() run will "
              f"create the new schema directly).")
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("PRAGMA table_info(precomputed_gate_results)")
    existing_columns = {row[1] for row in cur.fetchall()}
    if "content_fingerprint" in existing_columns:
        print("content_fingerprint already exists on precomputed_gate_results -- migration "
              "already applied, nothing to do.")
        conn.close()
        return

    cur.execute("SELECT COUNT(*) FROM precomputed_gate_results")
    before_count = cur.fetchone()[0]
    print(f"Real precomputed_gate_results row count BEFORE migration: {before_count}")

    cur.execute("ALTER TABLE precomputed_gate_results ADD COLUMN content_fingerprint TEXT")
    conn.commit()
    print("Column added.")
    conn.close()

    if before_count == 0:
        print("Table is empty -- no real rows to backfill. Migration complete.")
        return

    # Backfill through the ORM rather than raw sqlite3, so the fingerprint comes from the same
    # Python function the app uses and each row is joined to its linked Question.
    from app.db.session import get_session, init_db
    from app.models.precomputed_gate_result import PrecomputedGateResult
    from app.models.question import Question
    from app.pipeline.gate import compute_gate_content_fingerprint
    from sqlalchemy import select

    init_db()
    backfilled = 0
    missing_question_ids: list[int] = []
    with get_session() as session:
        rows = session.scalars(select(PrecomputedGateResult)).all()
        print(f"Real rows to backfill: {len(rows)}")
        for row in rows:
            question = session.scalar(select(Question).where(Question.id == row.question_id))
            if question is None:
                missing_question_ids.append(row.question_id)
                continue
            row.content_fingerprint = compute_gate_content_fingerprint(
                question.question_text, question.answer_value,
                question.has_diagram, question.worked_solution_text,
            )
            backfilled += 1
        session.commit()

    print(f"\nReal backfill result: {backfilled} row(s) fingerprinted.")
    if missing_question_ids:
        print(f"WARNING: {len(missing_question_ids)} row(s) reference a question_id with no real "
              f"matching Question row -- left NULL, ids: {missing_question_ids[:20]}"
              f"{' (truncated)' if len(missing_question_ids) > 20 else ''}")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM precomputed_gate_results")
    after_count = cur.fetchone()[0]
    print(f"\nReal precomputed_gate_results row count AFTER migration: {after_count}")
    if after_count != before_count:
        print(f"WARNING: row count changed during migration ({before_count} -> {after_count}) -- "
              f"investigate before trusting this migration.")
        conn.close()
        sys.exit(1)
    print("Real row count preserved exactly.")

    cur.execute("SELECT COUNT(*) FROM precomputed_gate_results WHERE content_fingerprint IS NULL")
    null_count = cur.fetchone()[0]
    print(f"Rows still NULL after backfill: {null_count} "
          f"(expected to equal the {len(missing_question_ids)} missing-question warning above).")
    if null_count != len(missing_question_ids):
        print("REAL PROBLEM: NULL count doesn't match the missing-question count -- investigate.")
        conn.close()
        sys.exit(1)
    print("Confirmed consistent. Migration committed.")
    conn.close()


if __name__ == "__main__":
    main()
