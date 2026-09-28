r"""Generate Tier 1 hints for the three revised prompt candidates (Issue 362).

Uses the candidates in tier1_candidates_v2.py, which fix the echo problem found in Issue 358, and
the same 50 sample rows. All 50 rows are regenerated, not only the 8 that echoed, because the
prompt text changed for every row. Works like generate_tier1_candidates.py: it temporarily
replaces gate._TIER1_SYSTEM_PROMPT and restores it afterwards. Resumes from any partial output.

Input: data/extracted/hint_quality_eval/sample_rows.json.
Output: tier1_candidates_v2.json in the same folder.

Usage:
    backend/.venv/Scripts/python.exe scripts/evaluation/hint_quality/generate_tier1_candidates_v2.py
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
from tier1_candidates_v2 import CANDIDATES_V2  # noqa: E402

_DIR = _REPO_ROOT / "data" / "extracted" / "hint_quality_eval"


def main() -> None:
    from app.pipeline import gate

    rows = json.load(open(_DIR / "sample_rows.json", encoding="utf-8"))
    out_path = _DIR / "tier1_candidates_v2.json"
    results = json.load(open(out_path, encoding="utf-8")) if out_path.exists() else {}
    original_prompt = gate._TIER1_SYSTEM_PROMPT
    gr = gate.GateResult(passed=False, tier_allowed=1, reason="hint-quality eval, Task 1 (echo-fixed v2): Tier 1 candidate")
    try:
        for name, prompt in CANDIDATES_V2.items():
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
