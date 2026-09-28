r"""
Measure whether running the gate's solving call on the adapter, rather than base weights, changes
the gate's outcomes (Issue 353).

Runs the live `gate.self_consistency_check()` (2 sampled passes, 450 tokens, T=0.8, the live
`_SOLVE_PROMPT`) on a seeded sample of held-out reasoning questions (never in the training export),
twice per question on the same resident model:
  * arm "base": the current behaviour (`use_adapter=False` inside gate.self_consistency_check);
  * arm "adapter": the same call with `generate_multiple` forced to `use_adapter=True`, the
    behaviour before Issue 353.
The RNG is seeded with the question id before each arm, so differences come from the weights and
not from sampling.

Re-running the grounding experiment harness without the adapter would not answer this: it would
reproduce condition B by construction (same base weights, prompts and greedy decoding). The gate's
solving call uses its own prompt, sampling and 450 tokens with no RAG few-shot, so it is measured
directly.

Outcomes per question and arm: `no_answer` (at least one pass has no `FINAL ANSWER:` line, so the
gate fails closed) and `passed` (both passes agree and match the answer key). The arms are
compared with an exact McNemar test. Results are saved after every question, so a run resumes.

Output: data/extracted/llm_adapter_wiring/gate_solving_arms.json and
gate_solving_arms_summary.json in the same folder.

Usage:
    backend/.venv/Scripts/python.exe scripts/evaluation/measure_gate_solving_adapter_vs_base.py
        [--n 80] [--summary-only]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from math import comb
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_OUT = _REPO_ROOT / "data" / "extracted" / "llm_adapter_wiring" / "gate_solving_arms.json"


def _mcnemar(b: int, c: int) -> float:
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    results = json.load(open(_OUT, encoding="utf-8")) if _OUT.exists() else {}

    if not args.summary_only:
        import torch
        from app.pipeline import gate
        from grounding_retrieval import load_held_out

        pool = [h for h in load_held_out() if h["answer_value"] and not h["has_diagram"]
                and gate.classify_question_type(h["question_text"], h["answer_value"], has_diagram=False) == "reasoning"]
        sample = random.Random(20260921).sample(pool, args.n)
        print(f"eligible held-out reasoning/text-only rows: {len(pool)}; sampled {len(sample)}")

        real_multiple = gate.generate_multiple

        def forced_adapter(*a, **k):
            return real_multiple(*a, **{**k, "use_adapter": True})

        for i, hq in enumerate(sample):
            qid = str(hq["question_id"])
            for arm in ("base", "adapter"):
                key = f"{arm}:{qid}"
                if key in results:
                    continue
                gate.generate_multiple = forced_adapter if arm == "adapter" else real_multiple
                torch.manual_seed(int(qid))
                r = gate.self_consistency_check(hq["question_text"], hq["answer_value"])
                claimed = r.detail.get("claimed_answers") or []
                results[key] = {"arm": arm, "question_id": int(qid), "passed": bool(r.passed),
                                "no_answer": any(a is None for a in claimed), "n_missing": sum(a is None for a in claimed),
                                "all_agree": r.detail.get("all_agree"), "matches_key": r.detail.get("matches_key")}
            gate.generate_multiple = real_multiple
            _OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
            print(f"  {i + 1}/{len(sample)} q{qid}: base passed={results['base:' + qid]['passed']} no_answer={results['base:' + qid]['no_answer']} | "
                  f"adapter passed={results['adapter:' + qid]['passed']} no_answer={results['adapter:' + qid]['no_answer']}", flush=True)

    ids = sorted({v["question_id"] for v in results.values() if f"base:{v['question_id']}" in results and f"adapter:{v['question_id']}" in results})
    B = {i: results[f"base:{i}"] for i in ids}
    A = {i: results[f"adapter:{i}"] for i in ids}
    n = len(ids)
    summary = {"n_questions": n,
               "no_answer_base": sum(B[i]["no_answer"] for i in ids), "no_answer_adapter": sum(A[i]["no_answer"] for i in ids),
               "passes_missing_a_final_answer_base": sum(B[i]["n_missing"] for i in ids), "passes_missing_a_final_answer_adapter": sum(A[i]["n_missing"] for i in ids),
               "passed_base": sum(B[i]["passed"] for i in ids), "passed_adapter": sum(A[i]["passed"] for i in ids)}
    for metric in ("no_answer", "passed"):
        b = sum(1 for i in ids if B[i][metric] and not A[i][metric])
        c = sum(1 for i in ids if A[i][metric] and not B[i][metric])
        summary[f"{metric}_base_only"], summary[f"{metric}_adapter_only"] = b, c
        summary[f"{metric}_mcnemar_p"] = round(_mcnemar(b, c), 4)
    print(json.dumps(summary, indent=2))
    (_OUT.parent / "gate_solving_arms_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
