"""
Check that extracted question checkpoints still match the canonical school-name CSV (Issue 159).

`normalized_school_name()` in backend/app/pipeline/extract.py writes `school_name` and
`school_name_verified` into each question's JSON checkpoint, and from there into its `questions`
row, at extraction time, using whatever `tools/school_name_canonical.csv` says at that moment. The
CSV is meant to be edited over time (see `tools/normalize_filenames.py`, Issue 72), including after
a file has been extracted. Stage C skips files that already have a checkpoint (the
`output_path.exists()` check in `_extract_one_file`), so a file extracted before a CSV correction
keeps its old values, and nothing else would detect the drift. When this tool was written the
corpus had 0 mismatches, but only because every CSV edit had happened before extraction; the
pipeline does not enforce it. This is the same kind of gap as Issue 158: a downstream artefact that
must stay in sync with an upstream source, with nothing checking it.

Following Issue 158, the fix is a small standalone check rather than new resumability logic in the
pipeline. Run it whenever `tools/school_name_canonical.csv` is edited after
`data/extracted/questions/` has content, and before relying on that content or on `psle.db`, which
is loaded from it. It exits non-zero and lists every mismatched file if the CSV and the checkpoint
values have diverged.

Usage (from the repository root, with the backend virtual environment so `app.pipeline.extract`
imports):
    backend/.venv/Scripts/python.exe tools/verify_school_name_consistency.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    sys.path.insert(0, str(REPO_ROOT / "backend"))
    sys.path.insert(0, str(REPO_ROOT))
    from app.pipeline.extract import normalized_school_name  # noqa: E402  (needs sys.path above)

    mismatches = []
    checkpoint_paths = sorted(glob.glob(str(REPO_ROOT / "data" / "extracted" / "questions" / "*.json")))
    for fpath in checkpoint_paths:
        data = json.loads(Path(fpath).read_text(encoding="utf-8"))
        if not data:
            continue
        source_paper = data[0]["question"].get("source_paper")
        if not source_paper:
            continue

        baked_names = {item["question"].get("school_name") for item in data}
        baked_verified = {item["question"].get("school_name_verified") for item in data}
        live_name, live_verified = normalized_school_name(Path(source_paper))

        if baked_names != {live_name} or baked_verified != {live_verified}:
            mismatches.append({
                "file": fpath,
                "source_paper": source_paper,
                "baked_school_name": sorted(str(n) for n in baked_names),
                "live_school_name": live_name,
                "baked_school_name_verified": sorted(str(v) for v in baked_verified),
                "live_school_name_verified": live_verified,
            })

    print(f"Checked {len(checkpoint_paths)} checkpoint file(s) against the live canonical CSV.")
    if not mismatches:
        print("OK — 0 mismatches. Every checkpoint's baked-in school_name/school_name_verified "
              "matches what tools/school_name_canonical.csv says right now.")
        return 0

    print(f"MISMATCH — {len(mismatches)} file(s) have stale school_name/school_name_verified "
          f"baked in from a since-changed canonical CSV entry. Re-run extraction for these files "
          f"(after clearing their existing checkpoint — Stage C's own resumability will otherwise "
          f"silently skip them, Issue 159) before trusting this corpus or re-loading it into "
          f"psle.db:")
    for m in mismatches:
        print(f"  {m['file']}")
        print(f"    baked: school_name={m['baked_school_name']} "
              f"verified={m['baked_school_name_verified']}")
        print(f"    live:  school_name={m['live_school_name']!r} "
              f"verified={m['live_school_name_verified']}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
