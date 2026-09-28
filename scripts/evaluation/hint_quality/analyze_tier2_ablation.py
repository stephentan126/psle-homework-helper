r"""Compare Tier 2 hints with the adapter on and off over the same 50 rows (Issue 359).

Runs after judge_tier2_ablation.py. The adapter-on arm reuses the Issue 357 judgments. Each rubric
dimension is tested with a Wilcoxon signed-rank test, because the scores are ordinal (1 to 5);
McNemar's test applies only to binary pass/fail outcomes. Score distributions and p-values are
reported, not just means.

Inputs (data/extracted/hint_quality_eval/): hints.json, judgments.json, tier2_ablation.json,
tier2_ablation_judgments.json. Output: tier2_ablation_analysis.json in the same folder.

Usage: python scripts/evaluation/hint_quality/analyze_tier2_ablation.py
"""
import collections
import json
import statistics
import sys
from pathlib import Path

from scipy.stats import wilcoxon

_DIR = Path(__file__).resolve().parents[3] / "data" / "extracted" / "hint_quality_eval"
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DIMS = ["scaffolding", "age_appropriate", "consistency", "length_clarity"]


def main() -> None:
    hints = json.load(open(_DIR / "hints.json", encoding="utf-8"))
    base_jud = json.load(open(_DIR / "judgments.json", encoding="utf-8"))
    abl = json.load(open(_DIR / "tier2_ablation.json", encoding="utf-8"))
    off_jud = json.load(open(_DIR / "tier2_ablation_judgments.json", encoding="utf-8"))
    ids = sorted(abl["adapter_off"], key=int)
    print(f"n={len(ids)} paired rows, reproducibility of entry 357's adapter-on outputs: {abl['reproducible']} (mismatched: {abl['mismatched_ids']})")

    on_leak = collections.Counter(abl["adapter_on"][k]["leak"]["status"] for k in ids)
    off_leak = collections.Counter(abl["adapter_off"][k]["leak"]["status"] for k in ids)
    print(f"\nLeak check -- adapter ON: {dict(on_leak)} | adapter OFF: {dict(off_leak)}")
    on_leak_ids = [k for k in ids if abl["adapter_on"][k]["leak"]["status"] == "LEAK"]
    off_leak_ids = [k for k in ids if abl["adapter_off"][k]["leak"]["status"] == "LEAK"]
    print(f"  ON leak rows: {on_leak_ids} | OFF leak rows: {off_leak_ids}")

    def on_score(k, d):
        return base_jud[f"{k}:2"]["scores"][d]["score"]

    def off_score(k, d):
        return off_jud[f"adapter_off:{k}"]["scores"][d]["score"]

    print(f"\n{'dimension':16s} {'ON mean':>8s} {'OFF mean':>9s} {'ON-only better':>15s} {'OFF-only better':>16s} {'tied':>5s} {'Wilcoxon p':>11s}")
    summary = {"n": len(ids), "leak": {"on": dict(on_leak), "off": dict(off_leak)}, "dims": {}}
    for d in DIMS:
        on_scores = [on_score(k, d) for k in ids]
        off_scores = [off_score(k, d) for k in ids]
        diffs = [a - b for a, b in zip(on_scores, off_scores)]
        on_better = sum(1 for x in diffs if x > 0)
        off_better = sum(1 for x in diffs if x < 0)
        tied = sum(1 for x in diffs if x == 0)
        if on_better == 0 and off_better == 0:
            p = 1.0
        else:
            _, p = wilcoxon(on_scores, off_scores, zero_method="wilcox")
        summary["dims"][d] = {"on_mean": round(statistics.mean(on_scores), 3), "off_mean": round(statistics.mean(off_scores), 3),
                              "on_only_better": on_better, "off_only_better": off_better, "tied": tied, "wilcoxon_p": round(float(p), 4),
                              "on_dist": {s: on_scores.count(s) for s in range(1, 6)}, "off_dist": {s: off_scores.count(s) for s in range(1, 6)}}
        print(f"{d:16s} {statistics.mean(on_scores):8.2f} {statistics.mean(off_scores):9.2f} {on_better:15d} {off_better:16d} {tied:5d} {p:11.4f}")
        print(f"  ON dist  {list(summary['dims'][d]['on_dist'].values())}   OFF dist {list(summary['dims'][d]['off_dist'].values())}")

    total_cost = sum(v.get("cost_usd", 0) for v in off_jud.values())
    print(f"\nJudge cost this task: ${total_cost:.4f} over {sum(1 for v in off_jud.values() if v.get('status') == 'ok')} NEW calls "
          f"(adapter-on arm reused entry 357's already-paid {len(ids)} calls, $0 additional)")
    (_DIR / "tier2_ablation_analysis.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("Saved -> tier2_ablation_analysis.json")

    print("\n##### Rows with the largest |ON - OFF| gap (any dimension)")
    gaps = sorted(ids, key=lambda k: -max(abs(on_score(k, d) - off_score(k, d)) for d in DIMS))[:4]
    for k in gaps:
        print(f"\n-- id={k} -- ON scores {[on_score(k, d) for d in DIMS]} OFF scores {[off_score(k, d) for d in DIMS]}")
        print("   ON :", abl["adapter_on"][k]["hint_text"][:220].replace("\n", " "))
        print("   OFF:", abl["adapter_off"][k]["hint_text"][:220].replace("\n", " "))


if __name__ == "__main__":
    main()
