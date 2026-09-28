r"""
Run Stage C (merge) of the extraction pipeline for the held-out set, writing to its own database.

The held-out set lives in data/extracted/held_out.db, never psle.db, following the freeze
principle of Issue 27. Run this after Stage A (page rendering) and Stage B (OCR) have processed
the held-out pages, and before freeze_held_out_stage_c.py.

This is a thin wrapper around `backend/app/pipeline/extract.py`. It adds no pipeline logic; it
passes the held-out input and output paths and sets DATABASE_URL explicitly. A bare
`python extract.py` would write into the live `psle.db`, so the wrapper makes that mistake hard to
make. extract.py runs as a subprocess so that DATABASE_URL is set before the app's database module
is imported.

Usage:
    backend/.venv/Scripts/python.exe scripts/held_out/stage_c_held_out_extraction.py
"""
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
HELD_OUT_DB_PATH = _REPO_ROOT / "data" / "extracted" / "held_out.db"


def main() -> None:
    database_url = f"sqlite:///{HELD_OUT_DB_PATH.as_posix()}"
    print(f"Real, isolated DATABASE_URL for this run: {database_url}")
    print(f"(psle.db itself is never touched by this script — separate file, separate engine.)")

    cmd = [
        sys.executable, str(_REPO_ROOT / "backend" / "app" / "pipeline" / "extract.py"),
        "--data-dir", str(_REPO_ROOT / "data"),
        "--raw-dir", str(_REPO_ROOT / "data" / "extracted" / "held_out_raw"),
        "--extracted-dir", str(_REPO_ROOT / "data" / "extracted" / "held_out_questions"),
        "--failures-log", str(_REPO_ROOT / "data" / "extracted" / "held_out_failures.log"),
        "--db-write-log-dir", str(_REPO_ROOT / "data" / "extracted" / "held_out_db_write_log"),
        "--discrepancy-log", str(_REPO_ROOT / "data" / "extracted" / "held_out_ocr_discrepancies.log"),
        "--source-subdirs", "Hold out/200", "Hold out/small eval",
        "--verbose",
    ]
    env = {"DATABASE_URL": database_url}
    import os
    full_env = os.environ.copy()
    full_env.update(env)
    result = subprocess.run(cmd, env=full_env, cwd=str(_REPO_ROOT))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
