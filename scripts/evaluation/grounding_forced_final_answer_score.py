r"""Score the forced "Final Answer:" continuations against the original generations (Issue 355).

Runs after grounding_forced_final_answer.py. "Before" is the recorded, merged 600-token generation.
"After" is the same text plus "\nFinal Answer:" and the continuation of up to 20 tokens, for rows
that had no parseable answer line. Both are graded with the same scorer as the other grounding
results, and compared with an exact McNemar test. CPU only.

Input: forced_final_answer_continuations.json in the grounding evaluation folder.
Output: forced_final_answer_score.json in the same folder.

Usage: backend/.venv/Scripts/python.exe scripts/evaluation/grounding_forced_final_answer_score.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import grounding_generate as eg  # noqa: E402
from grounding_score_c2 import grade, mcnemar_exact, merged_generations  # noqa: E402

_CUTOFF_TOKENS = 590  # within 10 tokens of the 600 cap: cut off, not finished
_CONDS = ["A_base", "B_rag_fewshot", "C_qlora_rag_fewshot", "C2_qlora_r16a32_rag_fewshot", "C3_qlora_masked_rag_fewshot"]


def main() -> None:
    cont = json.load(open(eg._EVAL_DIR / "forced_final_answer_continuations.json", encoding="utf-8"))
    summary = {}
    print(f"{'condition':30s} {'before':>10s} {'after':>10s} {'no-FA before':>13s} {'no-FA after':>12s} | newly parsed: right/total (cutoff rows | finished rows)")
    for c in _CONDS:
        rows = merged_generations(c)
        before = grade(rows)
        forced = {}
        for qid, r in rows.items():
            r2 = dict(r)
            k = str(qid)
            if k in cont.get(c, {}):
                r2["generated_text"] = r["generated_text"] + "\nFinal Answer:" + cont[c][k]["continuation"]
            forced[qid] = r2
        after = grade(forced)
        nb, na = sum(v["correct"] for v in before.values()), sum(v["correct"] for v in after.values())
        nofa_b, nofa_a = sum(not v["has_final"] for v in before.values()), sum(not v["has_final"] for v in after.values())
        newly = [q for q in before if not before[q]["has_final"] and after[q]["has_final"]]
        cut = [q for q in newly if cont[c][str(q)]["recorded_text_tokens"] >= _CUTOFF_TOKENS]
        fin = [q for q in newly if q not in cut]
        b = sum(1 for q in before if before[q]["correct"] and not after[q]["correct"])
        a = sum(1 for q in before if after[q]["correct"] and not before[q]["correct"])
        summary[c] = {"n": len(before), "correct_before": nb, "correct_after": na, "no_final_before": nofa_b, "no_final_after": nofa_a,
                      "newly_parsed": len(newly), "newly_parsed_correct": sum(after[q]["correct"] for q in newly),
                      "newly_parsed_cutoff_rows": [len(cut), sum(after[q]["correct"] for q in cut)],
                      "newly_parsed_finished_rows": [len(fin), sum(after[q]["correct"] for q in fin)],
                      "rows_lacking_line_that_were_cutoff": sum(1 for q in before if not before[q]["has_final"] and cont[c][str(q)]["recorded_text_tokens"] >= _CUTOFF_TOKENS),
                      "mcnemar_before_vs_after": {"after_only_right": a, "before_only_right": b, "p": round(mcnemar_exact(a, b), 4)}}
        print(f"{c:30s} {nb:>5d}/{len(before)} {na:>5d}/{len(before)} {nofa_b:>13d} {nofa_a:>12d} | {sum(after[q]['correct'] for q in newly)}/{len(newly)} "
              f"(cutoff {sum(after[q]['correct'] for q in cut)}/{len(cut)} | finished {sum(after[q]['correct'] for q in fin)}/{len(fin)}); "
              f"McNemar after-only {a} vs before-only {b}, p={mcnemar_exact(a, b):.3f}")
    # Print a few continuations so the behaviour can be inspected.
    print("\nSample continuations (A_base):")
    for k, v in list(cont["A_base"].items())[:6]:
        print(f"  qid {k}: text_tokens={v['recorded_text_tokens']} -> {v['continuation']!r}"[:200])
    out = eg._EVAL_DIR / "forced_final_answer_score.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Saved ->", out)


if __name__ == "__main__":
    main()
