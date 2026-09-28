r"""
Generate Tier 1 and Tier 2 hints for the 50 sampled held-out rows (Issue 357).

Runs after sample_rows.py. Each row goes through the live `gate.generate_hint()`, which calls
`llm_service.generate()`, so the default adapter set by LLM_ADAPTER_PATH (step0_qlora_v2_r16_a32
since Issue 353) writes the hints. Nothing is bypassed and gate.py and llm_service.py are not
changed.

Two hints per row:
  Tier 1: GateResult(passed=False, tier_allowed=1), the restate-and-stop template path.
  Tier 2: GateResult(passed=True, tier_allowed=2) with the row's worked_solution_text, the
          grounded path. The gate itself is not run: this measures hint writing given a
          known-good worked solution, not the gate. 22 of the 50 rows have has_diagram=True, so
          those hints take the diagram degrade path and get the "I can't see the figure" note,
          as they would in the live app.

Each hint then goes through the deterministic leak check (leak_check.py). Saves after every row,
so an interrupted run resumes.

Input: data/extracted/hint_quality_eval/sample_rows.json. Output: hints.json in the same folder.

Usage: backend/.venv/Scripts/python.exe scripts/evaluation/hint_quality/generate_hints.py
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

_DIR = _REPO_ROOT / "data" / "extracted" / "hint_quality_eval"


def main() -> None:
    from app.pipeline import gate
    from app.services import llm_service

    rows = json.load(open(_DIR / "sample_rows.json", encoding="utf-8"))
    out_path = _DIR / "hints.json"
    results = json.load(open(out_path, encoding="utf-8")) if out_path.exists() else {}
    print("adapter status:", llm_service.get_llm_adapter_status())
    for i, r in enumerate(rows):
        k = str(r["id"])
        if k in results:
            continue
        rec = {"id": r["id"], "question_text": r["question_text"], "answer_value": r["answer_value"],
               "worked_solution_text": r["worked_solution_text"], "has_diagram": bool(r["has_diagram"]),
               "question_section": r["question_section"], "hints": {}}
        for tier, gr, ws in (
            (1, gate.GateResult(passed=False, tier_allowed=1, reason="hint-quality eval: Tier 1 path"), None),
            (2, gate.GateResult(passed=True, tier_allowed=2, reason="hint-quality eval: Tier 2 path (gate not run)"), r["worked_solution_text"]),
        ):
            t0 = time.time()
            h = gate.generate_hint(r["question_text"], gr, worked_solution_text=ws, has_diagram=bool(r["has_diagram"]))
            rec["hints"][str(tier)] = {"tier_returned": h["tier"], "hint_text": h["hint_text"], "grounded_in": h["grounded_in"],
                                       "seconds": round(time.time() - t0, 1),
                                       "leak": check_leak(r["answer_value"], r["question_text"], h["hint_text"])}
        results[k] = rec
        out_path.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"[{i + 1}/{len(rows)}] id={r['id']} T1 leak={rec['hints']['1']['leak']['status']} T2 leak={rec['hints']['2']['leak']['status']}")
    print("DONE ->", out_path)


if __name__ == "__main__":
    main()
