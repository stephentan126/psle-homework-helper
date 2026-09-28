r"""
Draw the 50-row sample for the hint-quality evaluation (Issue 357). No GPU or model is needed.

Source: data/extracted/held_out.db, the 446 frozen held-out rows. The corpus databases could not be
used, because every row in both corpus_run_baseline_prefix_20260830.db (5,447) and psle.db (5,819)
has verification_status='unverified', so a "verified" filter returns nothing. The held-out rows
also never appear in the 1,349-row adapter training export (0 exact-text overlaps, 0 shared source
papers), so the adapter is not graded on its own training questions.
Note: held_out.db ids are a separate id space from psle.db ids; do not join across them.

Eligible rows are not superseded, have a non-empty worked_solution_text, and have an answer_value
that parses to at least one number, so the leak check has something to test. The sample is
stratified by (question_section, has_diagram) with proportional allocation and a fixed seed, so
both sections and both text-only and diagram rows are represented.

Output: data/extracted/hint_quality_eval/sample_rows.json.

Usage: python scripts/evaluation/hint_quality/sample_rows.py
"""
import json
import random
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from leak_check import numeric_values  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_N, _SEED = 50, 20260922
_OUT = _REPO_ROOT / "data" / "extracted" / "hint_quality_eval" / "sample_rows.json"


def main() -> None:
    con = sqlite3.connect(f"file:{_REPO_ROOT / 'data' / 'extracted' / 'held_out.db'}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "select id, source_paper, question_number, question_section, question_text, answer_value, worked_solution_text, has_diagram "
        "from questions where superseded_by is null and worked_solution_text is not null and trim(worked_solution_text)<>'' "
        "and answer_value is not null and trim(answer_value)<>''").fetchall()
    rows = [dict(r) for r in rows if numeric_values(r["answer_value"])]
    strata = defaultdict(list)
    for r in rows:
        strata[(r["question_section"], bool(r["has_diagram"]))].append(r)
    total = len(rows)
    rng = random.Random(_SEED)
    # Largest-remainder proportional allocation, with at least 1 row per non-empty stratum.
    quota = {k: max(1, int(_N * len(v) / total)) for k, v in strata.items()}
    while sum(quota.values()) < _N:
        k = max(strata, key=lambda s: _N * len(strata[s]) / total - quota[s])
        quota[k] += 1
    while sum(quota.values()) > _N:
        k = max(quota, key=lambda s: quota[s] - _N * len(strata[s]) / total)
        quota[k] -= 1
    sample = []
    for k in sorted(strata, key=str):
        sample += rng.sample(strata[k], quota[k])
    sample.sort(key=lambda r: r["id"])
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(sample, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"eligible rows: {total}; sampled {len(sample)} (seed {_SEED})")
    for k in sorted(strata, key=str):
        print(f"  {k}: eligible {len(strata[k])}, sampled {quota[k]}")
    print("ids:", [r["id"] for r in sample])
    print(f"saved -> {_OUT}")


if __name__ == "__main__":
    main()
