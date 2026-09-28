r"""
Freeze the held-out Stage C database as a separate, timestamped, read-only copy (Issue 27).

Run after stage_c_held_out_extraction.py. data/extracted/held_out.db holds both `questions` and
`ocr_extraction_records`, so one file copy also preserves the extraction records (the gap found in
Issue 68). After making the copy read-only, the script attempts a write to prove that it fails.

The script refuses to overwrite an existing freeze with the same name. To re-freeze, delete the
old copy by hand first.

Output: data/extracted/held_out_frozen/held_out_frozen_<date>.db.

Usage:
    backend/.venv/Scripts/python.exe scripts/held_out/freeze_held_out_stage_c.py
"""
import os
import shutil
import sqlite3
import stat
import sys
from datetime import date
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = _REPO_ROOT / "data" / "extracted" / "held_out.db"
FROZEN_DIR = _REPO_ROOT / "data" / "extracted" / "held_out_frozen"


def main() -> None:
    if not SRC.exists():
        print(f"REFUSING: {SRC} does not exist — run stage_c_held_out_extraction.py first.")
        sys.exit(1)

    FROZEN_DIR.mkdir(parents=True, exist_ok=True)
    dest = FROZEN_DIR / f"held_out_frozen_{date.today().isoformat()}.db"

    if dest.exists():
        print(f"REFUSING: {dest} already exists. Delete it deliberately first if a "
              f"re-freeze is genuinely intended.")
        sys.exit(1)

    shutil.copy2(SRC, dest)
    print(f"Copied {SRC} -> {dest} ({dest.stat().st_size} bytes)")

    # The source database was created through the app's SQLAlchemy engine, which sets
    # journal_mode=WAL on every connection (`_enable_wal_mode` in session.py). WAL mode is stored
    # in the file header, so the copy inherits it. Any connection to a WAL-mode database, even a
    # read, creates "-shm" and "-wal" sidecar files that chmod on the .db file does not cover.
    # Switching the copy (never SRC) back to rollback-journal mode before chmod leaves the frozen
    # artefact as a single read-only file.
    con = sqlite3.connect(dest)
    cur = con.cursor()
    cur.execute("PRAGMA journal_mode=DELETE")
    mode = cur.fetchone()[0]
    print(f"Journal mode on the frozen copy switched to: {mode}")
    cur.execute("SELECT COUNT(*) FROM questions")
    n_q = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM ocr_extraction_records")
    n_r = cur.fetchone()[0]
    con.close()
    print(f"Frozen copy contains: {n_q} questions rows, {n_r} ocr_extraction_records rows.")

    leftover_sidecars = list(dest.parent.glob(f"{dest.name}-*"))
    if leftover_sidecars:
        print(f"Removing leftover sidecar file(s) so the frozen artifact is one single file: "
              f"{[p.name for p in leftover_sidecars]}")
        for p in leftover_sidecars:
            p.unlink()

    os.chmod(dest, stat.S_IREAD)
    print(f"chmod applied (read-only). File mode now: {oct(dest.stat().st_mode)}")

    try:
        con2 = sqlite3.connect(dest)
        cur2 = con2.cursor()
        cur2.execute(
            "INSERT INTO questions (source_paper, source_page_index, question_number, question_text) "
            "VALUES ('WRITE_TEST_SHOULD_FAIL', 0, 'test', 'test')"
        )
        con2.commit()
        con2.close()
        print("!!! WRITE SUCCEEDED — THIS IS A FAILURE, the freeze is NOT actually read-only !!!")
        sys.exit(1)
    except Exception as e:  # noqa: BLE001, broad on purpose to report whatever error occurs
        print(f"Write attempt FAILED as expected — real read-only confirmed.")
        print(f"Real exception type: {type(e).__name__}")
        print(f"Real exception message: {e}")

    print(f"\nFROZEN, REAL, READ-ONLY: {dest}")


if __name__ == "__main__":
    main()
