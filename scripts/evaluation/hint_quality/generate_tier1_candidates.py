r"""
Generate Tier 1 hints for each candidate prompt in tier1_candidates.py (Issue 358).

Uses the same 50 sample rows as Issue 357, so the comparison is paired against the Tier 1 hints
already recorded in hints.json. That baseline is reused, not regenerated.

`gate.generate_hint()` is called for every row, with `gate._TIER1_SYSTEM_PROMPT` replaced by each
candidate in turn and restored afterwards; gate.py itself is not modified. The deterministic leak
check is applied to each hint straight away, as in Issue 357. Resumes from any partial output.

Input: data/extracted/hint_quality_eval/sample_rows.json.
Output: tier1_candidates.json in the same folder, keyed by candidate, then row id.

Usage: backend/.venv/Scripts/python.exe scripts/evaluation/hint_quality/generate_tier1_candidates.py
"""
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from leak_check import check_leak  # noqa: E402
from tier1_candidates import CANDIDATES  # noqa: E402

_DIR = _REPO_ROOT / "data" / "extracted" / "hint_quality_eval"


def main() -> None:
    from app.pipeline import gate

    rows = json.load(open(_DIR / "sample_rows.json", encoding="utf-8"))
    out_path = _DIR / "tier1_candidates.json"
    results = json.load(open(out_path, encoding="utf-8")) if out_path.exists() else {}
    original_prompt = gate._TIER1_SYSTEM_PROMPT
    gr = gate.GateResult(passed=False, tier_allowed=1, reason="hint-quality eval, Task 1: Tier 1 candidate")
    try:
        for name, prompt in CANDIDATES.items():
            cand = results.setdefault(name, {})
            gate._TIER1_SYSTEM_PROMPT = prompt
            for i, r in enumerate(rows):
                k = str(r["id"])
                if k in cand:
                    continue
                t0 = time.time()
                h = gate.generate_hint(r["question_text"], gr, has_diagram=bool(r["has_diagram"]))
                assert h["tier"] == 1
                cand[k] = {"id": r["id"], "hint_text": h["hint_text"], "seconds": round(time.time() - t0, 1),
                          "leak": check_leak(r["answer_value"], r["question_text"], h["hint_text"])}
                out_path.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
                if (i + 1) % 10 == 0 or i + 1 == len(rows):
                    print(f"[{name}] {i + 1}/{len(rows)}")
    finally:
        gate._TIER1_SYSTEM_PROMPT = original_prompt
    print("DONE ->", out_path)


if __name__ == "__main__":
    main()
