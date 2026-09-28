r"""Summarise leak checks and judge scores per hint tier and rubric dimension (Issue 357).

Runs after generate_hints.py and judge_hints.py. Reports score distributions as well as means,
splits Tier 2 by diagram and text-only questions, and prints example hints. CPU only.

Inputs (data/extracted/hint_quality_eval/): hints.json, judgments.json.
Output: analysis_summary.json in the same folder.

Usage: python scripts/evaluation/hint_quality/analyze.py
"""
import collections
import json
import statistics
import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parents[3] / "data" / "extracted" / "hint_quality_eval"
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DIMS = ["scaffolding", "age_appropriate", "consistency", "length_clarity"]


def main() -> None:
    hints = json.load(open(_DIR / "hints.json", encoding="utf-8"))
    jud = json.load(open(_DIR / "judgments.json", encoding="utf-8"))
    summary = {"leak": {}, "scores": {}}
    for t in ("1", "2"):
        leak = collections.Counter(r["hints"][t]["leak"]["status"] for r in hints.values())
        summary["leak"][t] = dict(leak)
        print(f"\n=== Tier {t} ({len(hints)} hints) === leak check: {dict(leak)}")
        summary["scores"][t] = {}
        for d in DIMS:
            sc = [jud[f"{k}:{t}"]["scores"][d]["score"] for k in hints]
            dist = {s: sc.count(s) for s in range(1, 6)}
            summary["scores"][t][d] = {"dist": dist, "mean": round(statistics.mean(sc), 2), "median": statistics.median(sc)}
            print(f"  {d:16s} mean {statistics.mean(sc):.2f} median {statistics.median(sc)}  dist(1..5) {list(dist.values())}")
    # diagram vs text-only, Tier 2
    for flag in (False, True):
        ids = [k for k, r in hints.items() if r["has_diagram"] == flag]
        print(f"\nTier 2, has_diagram={flag} (n={len(ids)}): " + ", ".join(
            f"{d} {statistics.mean(jud[f'{k}:2']['scores'][d]['score'] for k in ids):.2f}" for d in DIMS))
    tot = sum(v.get("cost_usd", 0) for v in jud.values())
    print(f"\njudge calls: {len(jud)}, cost of the FINAL entries ${tot:.4f} (the 4 first-attempt truncated calls, ~$0.0838 + their re-runs, are described in the development log)")
    costs = sorted(((v["cost_usd"], k) for k, v in jud.items()), reverse=True)
    print("most expensive calls:", [(k, c) for c, k in costs[:3]], "| mean per call $%.4f" % (tot / len(jud)))
    (_DIR / "analysis_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    def show(k, t):
        h, j = hints[k], jud[f"{k}:{t}"]["scores"]
        print(f"\n--- id={k} Tier {t} leak={h['hints'][t]['leak']['status']} scores " + str({d: j[d]['score'] for d in DIMS}))
        print("  HINT:", h["hints"][t]["hint_text"][:420].replace("\n", " "))
        print("  judge (scaffolding):", j["scaffolding"]["why"][:200])
    # Lowest and highest Tier 2 totals, then a median Tier 1 hint.
    tot2 = sorted(hints, key=lambda k: sum(jud[f"{k}:2"]["scores"][d]["score"] for d in DIMS))
    print("\n##### Lowest-scoring Tier 2 hints")
    for k in tot2[:2]:
        show(k, "2")
    print("\n##### Highest-scoring Tier 2 hints")
    for k in tot2[-2:]:
        show(k, "2")
    print("\n##### A typical Tier 1 hint")
    show(sorted(hints, key=lambda k: sum(jud[f"{k}:1"]["scores"][d]["score"] for d in DIMS))[len(hints) // 2], "1")
    print("\n##### The deterministic LEAK")
    for k, r in hints.items():
        for t in ("1", "2"):
            if r["hints"][t]["leak"]["status"] == "LEAK":
                show(k, t)
                print("  answer_value:", r["answer_value"].replace("\n", " | "))


if __name__ == "__main__":
    main()
