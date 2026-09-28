r"""
Score condition C3_qlora_masked_rag_fewshot (the v3 masked adapter) against the recorded
baselines (Issue 349, option C).

Compares it with conditions A (base), B (base with RAG and few-shot) and C (v1 r=8 adapter with
RAG and few-shot) on the same 388 exact-match-eligible held-out rows, using paired exact McNemar
tests. The earlier McNemar tests were run by hand and not saved; this script makes the whole
comparison reproducible. Run after grounding_generate_c3.py.

Merging follows the corrected protocol (Issue 316): for A, B and C, the original 300-token
generation of each row is replaced by the 600-token re-run where one exists. C3 was generated in a
single 600-token pass, which is equivalent under greedy decoding (see grounding_generate_c3.py).

All conditions are scored in one pass with extract_final_answer() and answers_match() from
grounding_score.py (which rely on gate.strip_units()), so the comparison is like for like.
gate.strip_units() was extended after the original grounding scores were recorded (Issue 344,
litre unit), so re-scoring can differ from the recorded 54/50/50. The script prints both and flags
any difference. `--baselines-only` validates the merge and the tests on A, B and C alone, without
the C3 generations.

McNemar exact test: paired design, where b counts rows the first condition gets right and the
second gets wrong, and c the reverse. The p-value is the two-sided exact binomial p over the b+c
discordant pairs. The script also reports the most even split of that many discordant pairs that
would reach p < 0.05, so a null result can be read against the test's sensitivity.

Output: score_summary_c3_vs_baselines.json (or score_summary_baselines_revalidation.json with
--baselines-only) in data/extracted/slice9_eval/.

Usage: python scripts/evaluation/grounding_score_c3.py [--baselines-only]
"""
from __future__ import annotations

import argparse
import json
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

from grounding_score import answers_match, extract_final_answer  # noqa: E402

_EVAL_DIR = _REPO_ROOT / "data" / "extracted" / "slice9_eval"
_BASELINES = {"A_base": "A base alone", "B_rag_fewshot": "B base+RAG+few-shot", "C_qlora_rag_fewshot": "C Phase-6 adapter (r=8)+RAG+few-shot"}
_NEW = "C3_qlora_masked_rag_fewshot"


def _load(path: Path) -> dict:
    return {int(k): v for k, v in json.load(open(path, encoding="utf-8")).items()}


def merged_generations(condition: str) -> dict:
    rows = _load(_EVAL_DIR / f"generations_{condition}.json")
    rerun = _EVAL_DIR / f"generations_{condition}_rerun_extended_tokens.json"
    if rerun.exists():
        for qid, r in _load(rerun).items():
            rows[qid] = r  # 600-token re-run replaces the cut-off 300-token row (Issue 316)
    return rows


def grade(rows: dict) -> dict:
    """Grade the exact-match-eligible rows.

    Returns {question_id: {'correct': bool, 'has_final': bool, 'has_diagram': bool}}.
    """
    out = {}
    for qid, r in rows.items():
        if not r["real_answer_value"]:
            continue
        pred = extract_final_answer(r["generated_text"])
        out[qid] = {"correct": answers_match(pred, r["real_answer_value"]), "has_final": pred is not None,
                    "has_diagram": bool(r["has_diagram"])}
    return out


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def smallest_significant_split(n: int, alpha: float = 0.05):
    """Return the most even split (b, c) of n discordant pairs with exact two-sided p < alpha.

    Returns None if no split reaches alpha.
    """
    for k in range(n // 2, -1, -1):
        if mcnemar_exact(k, n - k) < alpha:
            return (n - k, k)
    return None


def compare(name1: str, g1: dict, name2: str, g2: dict, subset=None) -> dict:
    ids = sorted(set(g1) & set(g2))
    if subset is not None:
        ids = [i for i in ids if subset(g1[i])]
    b = sum(1 for i in ids if g1[i]["correct"] and not g2[i]["correct"])
    c = sum(1 for i in ids if g2[i]["correct"] and not g1[i]["correct"])
    d = b + c
    return {"first": name1, "second": name2, "n": len(ids), "first_only_right": b, "second_only_right": c, "discordant": d,
            "p_exact_two_sided": round(mcnemar_exact(b, c), 4), "smallest_significant_split_of_this_many_discordant": smallest_significant_split(d)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines-only", action="store_true", help="validate the merge/tests on A/B/C only (no C2 file needed)")
    args = ap.parse_args()

    conds = dict(_BASELINES)
    if not args.baselines_only:
        conds[_NEW] = "C3 NEW adapter (r=8, completion-only loss, dq)+RAG+few-shot"
    graded = {c: grade(merged_generations(c)) for c in conds}

    recorded = json.load(open(_EVAL_DIR / "score_summary_corrected_after_rerun.json", encoding="utf-8"))
    summary = {"conditions": {}, "recorded_vs_rescored": {}, "mcnemar": []}
    print("=== Accuracy on the 388 exact-match-eligible held-out rows (merged corrected protocol, one scorer for all) ===")
    for c, label in conds.items():
        g = graded[c]
        n, k = len(g), sum(v["correct"] for v in g.values())
        dt = [v for v in g.values() if v["has_diagram"]]
        df = [v for v in g.values() if not v["has_diagram"]]
        no_final = sum(1 for v in g.values() if not v["has_final"])
        summary["conditions"][c] = {"n": n, "correct": k, "accuracy": round(k / n, 4), "diagram_true": [sum(v["correct"] for v in dt), len(dt)],
                                    "diagram_false": [sum(v["correct"] for v in df), len(df)], "no_final_answer": no_final}
        print(f"  {label:44s} {k}/{n} = {k / n:.1%} | has_diagram=T {sum(v['correct'] for v in dt)}/{len(dt)} | "
              f"has_diagram=F {sum(v['correct'] for v in df)}/{len(df)} | no Final Answer line: {no_final}")
        if c in recorded:
            rec = recorded[c]["n_correct"]
            summary["recorded_vs_rescored"][c] = {"recorded": rec, "rescored": k, "match": rec == k}
            print(f"      recorded (Issue 316 corrected) = {rec}  ->  re-scored now = {k}  {'(matches)' if rec == k else '*** DIFFERS ***'}")

    print("\n=== Paired McNemar exact tests ===")
    pairs = [("A_base", "B_rag_fewshot"), ("A_base", "C_qlora_rag_fewshot"), ("B_rag_fewshot", "C_qlora_rag_fewshot")]
    if not args.baselines_only:
        pairs = [(_NEW, "A_base"), (_NEW, "B_rag_fewshot"), (_NEW, "C_qlora_rag_fewshot")] + pairs
    for x, y in pairs:
        for tag, sub in (("overall", None), ("has_diagram=False", lambda v: not v["has_diagram"]), ("has_diagram=True", lambda v: v["has_diagram"])):
            r = compare(x, graded[x], y, graded[y], sub)
            r["subset"] = tag
            summary["mcnemar"].append(r)
            print(f"  {x} vs {y} [{tag}] n={r['n']}: {x}-only {r['first_only_right']}, {y}-only {r['second_only_right']} "
                  f"(discordant {r['discordant']}), p={r['p_exact_two_sided']}, smallest significant split at this discordance: "
                  f"{r['smallest_significant_split_of_this_many_discordant']}")

    out = _EVAL_DIR / ("score_summary_c3_vs_baselines.json" if not args.baselines_only else "score_summary_baselines_revalidation.json")
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
