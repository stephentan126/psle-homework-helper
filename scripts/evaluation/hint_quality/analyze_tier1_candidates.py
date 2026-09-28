r"""Compare each Tier 1 prompt candidate against the current template (Issue 358).

The baseline is the Tier 1 result recorded for Issue 357 in hints.json and judgments.json; it is
not regenerated. The ship rule is checked and printed for each candidate: the leak count must not
increase and the mean scaffolding score must rise by at least one point.

Inputs (data/extracted/hint_quality_eval/): hints.json, judgments.json, tier1_candidates.json,
tier1_candidate_judgments.json. Output: tier1_candidate_analysis.json in the same folder.

Usage: python scripts/evaluation/hint_quality/analyze_tier1_candidates.py
"""
import collections
import json
import statistics
import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parents[3] / "data" / "extracted" / "hint_quality_eval"
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from tier1_candidates import CANDIDATES  # noqa: E402

DIMS = ["scaffolding", "age_appropriate", "consistency", "length_clarity"]


def stats(scores: list[int]) -> dict:
    dist = {s: scores.count(s) for s in range(1, 6)}
    return {"mean": round(statistics.mean(scores), 2), "median": statistics.median(scores), "dist": dist}


def main() -> None:
    base_hints = json.load(open(_DIR / "hints.json", encoding="utf-8"))
    base_jud = json.load(open(_DIR / "judgments.json", encoding="utf-8"))
    cands = json.load(open(_DIR / "tier1_candidates.json", encoding="utf-8"))
    cand_jud = json.load(open(_DIR / "tier1_candidate_judgments.json", encoding="utf-8"))

    baseline_leak = collections.Counter(r["hints"]["1"]["leak"]["status"] for r in base_hints.values())
    baseline_scaff = [base_jud[f"{k}:1"]["scores"]["scaffolding"]["score"] for k in base_hints]
    print(f"BASELINE (current template, entry 357, n={len(base_hints)}): leak {dict(baseline_leak)}")
    for d in DIMS:
        sc = [base_jud[f"{k}:1"]["scores"][d]["score"] for k in base_hints]
        print(f"  {d:16s} " + str(stats(sc)))

    verdicts = {}
    for name in CANDIDATES:
        leak = collections.Counter(v["leak"]["status"] for v in cands[name].values())
        n = len(cands[name])
        print(f"\n{name} (n={n}): leak {dict(leak)}")
        row = {"leak": dict(leak), "n": n}
        for d in DIMS:
            sc = [cand_jud[f"{name}:{k}"]["scores"][d]["score"] for k in cands[name] if f"{name}:{k}" in cand_jud and cand_jud[f"{name}:{k}"].get("status") == "ok"]
            row[d] = stats(sc)
            print(f"  {d:16s} " + str(row[d]))
        leaks_increased = leak.get("LEAK", 0) > baseline_leak.get("LEAK", 0)
        scaff_gain = row["scaffolding"]["mean"] - stats(baseline_scaff)["mean"]
        ships = (not leaks_increased) and scaff_gain >= 1.0  # a full point on the 1-5 scale
        row["leak_count_increased_vs_baseline"] = leaks_increased
        row["scaffolding_gain_vs_baseline"] = round(scaff_gain, 2)
        row["ships_per_hard_rule"] = ships
        verdicts[name] = row
        print(f"  -> leak count increased vs baseline: {leaks_increased} | scaffolding gain: {scaff_gain:+.2f} | SHIPS: {ships}")

    (_DIR / "tier1_candidate_analysis.json").write_text(json.dumps({"baseline_leak": dict(baseline_leak), "candidates": verdicts}, indent=1), encoding="utf-8")
    print("\nSaved -> tier1_candidate_analysis.json")

    print("\n##### Sample hints per candidate (first 3 rows)")
    for name in CANDIDATES:
        print(f"\n-- {name} --")
        for k in list(cands[name])[:3]:
            print(f"  id={k}: {cands[name][k]['hint_text'][:260]!r}")


if __name__ == "__main__":
    main()
