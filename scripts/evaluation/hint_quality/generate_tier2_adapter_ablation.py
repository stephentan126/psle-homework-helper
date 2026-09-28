r"""
Measure whether the adapter improves Tier 2 hint quality, the task it was trained for (Issue 359).

Uses the same 50 held-out rows as Issue 357 (`sample_rows.json`, same seed), so the comparison is
paired. Runs before judge_tier2_ablation.py and analyze_tier2_ablation.py.

Step A: check that the Issue 357 adapter-on outputs reproduce byte for byte (as in Issue 352).
Tier 2 is regenerated with the adapter on, through the unmodified default path, and compared with
the recorded hint_text. If every row matches, the recorded outputs and their judge scores are
reused for the adapter-on arm, at no extra judge cost. If any row differs, the regenerated set is
used and the "reproducible" flag tells judge_tier2_ablation.py to judge it.

Step B: generate the same rows with the adapter off, using the `use_adapter=False` option from
Issue 353. gate.py's `generate` (imported from llm_service) is wrapped to pass that argument; the
underlying PEFT `disable_adapter()` mechanism is unchanged. One model load serves both arms, so only
one large model is resident.

Every hint goes through the deterministic leak check, as in Issue 357.

Inputs (data/extracted/hint_quality_eval/): sample_rows.json, hints.json.
Output: tier2_ablation.json in the same folder.

Usage:
    backend/.venv/Scripts/python.exe scripts/evaluation/hint_quality/generate_tier2_adapter_ablation.py
"""
import functools
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
    baseline = json.load(open(_DIR / "hints.json", encoding="utf-8"))
    out_path = _DIR / "tier2_ablation.json"
    out = json.load(open(out_path, encoding="utf-8")) if out_path.exists() else {"adapter_on_repro": {}, "adapter_on": {}, "adapter_off": {}}
    print("adapter status:", llm_service.get_llm_adapter_status())
    gr = gate.GateResult(passed=True, tier_allowed=2, reason="hint-quality eval, Task 2: Tier 2 adapter ablation")

    # Step A: reproducibility check, adapter on, through the unmodified default path.
    mismatches = []
    for i, r in enumerate(rows):
        k = str(r["id"])
        if k in out["adapter_on_repro"]:
            continue
        h = gate.generate_hint(r["question_text"], gr, worked_solution_text=r["worked_solution_text"], has_diagram=bool(r["has_diagram"]))
        assert h["tier"] == 2
        recorded = baseline[k]["hints"]["2"]["hint_text"]
        matches = h["hint_text"] == recorded
        if not matches:
            mismatches.append(k)
        out["adapter_on_repro"][k] = {"id": r["id"], "hint_text": h["hint_text"], "matches_entry_357": matches}
        out_path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
        if (i + 1) % 10 == 0 or i + 1 == len(rows):
            print(f"[repro check] {i + 1}/{len(rows)}, mismatches so far: {len(mismatches)}")
    reproducible = len(mismatches) == 0
    out["reproducible"] = reproducible
    out["mismatched_ids"] = mismatches
    print(f"REPRODUCIBILITY: {'PASSED, byte-for-byte, all 50 rows' if reproducible else f'FAILED on {len(mismatches)}/50 rows: {mismatches}'}")
    if reproducible:
        for k in out["adapter_on_repro"]:
            out["adapter_on"][k] = {"id": baseline[k]["id"], "hint_text": baseline[k]["hints"]["2"]["hint_text"],
                                    "leak": baseline[k]["hints"]["2"]["leak"], "source": "entry_357_reused"}
    else:
        for k, v in out["adapter_on_repro"].items():
            r = next(x for x in rows if str(x["id"]) == k)
            out["adapter_on"][k] = {"id": v["id"], "hint_text": v["hint_text"],
                                    "leak": check_leak(r["answer_value"], r["question_text"], v["hint_text"]), "source": "regenerated_here"}
    out_path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    # Step B: adapter off, by wrapping gate.generate with use_adapter=False (Issue 353).
    original_generate = gate.generate
    gate.generate = functools.partial(original_generate, use_adapter=False)
    try:
        for i, r in enumerate(rows):
            k = str(r["id"])
            if k in out["adapter_off"]:
                continue
            t0 = time.time()
            h = gate.generate_hint(r["question_text"], gr, worked_solution_text=r["worked_solution_text"], has_diagram=bool(r["has_diagram"]))
            assert h["tier"] == 2
            out["adapter_off"][k] = {"id": r["id"], "hint_text": h["hint_text"], "seconds": round(time.time() - t0, 1),
                                     "leak": check_leak(r["answer_value"], r["question_text"], h["hint_text"])}
            out_path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
            if (i + 1) % 10 == 0 or i + 1 == len(rows):
                print(f"[adapter OFF] {i + 1}/{len(rows)}")
    finally:
        gate.generate = original_generate
    print("DONE ->", out_path)


if __name__ == "__main__":
    main()
