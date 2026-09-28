"""
Real regression check (Issue 205): confirms batching ShieldGemma-2B's 3 policy
checks into one forward pass produces IDENTICAL detection behavior to the pre-batching sequential
version, not just similar aggregate stats, but the same per-case flag on every real case in all
three validation sets, compared directly against the saved `combined_signal_scores.json` (the
pre-batching, sequential-call run).

Usage:
    backend/.venv/Scripts/python.exe evaluation/model_selection/safety_classifier/verify_batching_unchanged.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "backend"))

from app.pipeline.self_harm_detection import (  # noqa: E402
    SHIELDGEMMA_SELF_HARM_THRESHOLD,
    score_shieldgemma,
)
from app.pipeline.self_harm_lexicon import check_self_harm_lexicon  # noqa: E402


def main() -> None:
    prior = json.loads((_HERE / "combined_signal_scores.json").read_text(encoding="utf-8"))

    mismatches = []
    total = 0
    for set_name, filename in [
        ("gold_set", "gold_set.json"),
        ("held_out_set", "held_out_set.json"),
        ("fresh_validation_set", "fresh_validation_set.json"),
    ]:
        cases = json.loads((_HERE / filename).read_text(encoding="utf-8"))
        prior_by_id = {c["id"]: c for c in prior[set_name]["per_case"]}
        for case in cases:
            total += 1
            new_sg_prob = score_shieldgemma(case["text"], case["direction"])
            new_sg_flag = new_sg_prob > SHIELDGEMMA_SELF_HARM_THRESHOLD
            new_lex_flag = check_self_harm_lexicon(case["text"]) is not None
            new_combined = new_sg_flag or new_lex_flag

            old = prior_by_id[case["id"]]
            old_combined = old["combined_flag"]

            status = "OK" if new_combined == old_combined else "MISMATCH"
            if status == "MISMATCH":
                mismatches.append({
                    "id": case["id"], "old_combined": old_combined, "new_combined": new_combined,
                    "old_sg_prob": old["shieldgemma_prob"], "new_sg_prob": round(new_sg_prob, 4),
                })
            print(f"  [{set_name}] {case['id']}: old_combined={old_combined} "
                  f"new_combined={new_combined} (old_sg={old['shieldgemma_prob']:.4f}, "
                  f"new_sg={new_sg_prob:.4f}) -> {status}")

    print(f"\n{total - len(mismatches)}/{total} cases match exactly.")
    if mismatches:
        print(f"MISMATCHES ({len(mismatches)}):")
        for m in mismatches:
            print(f"  {m}")
    else:
        print("Zero mismatches — batching confirmed behavior-identical to the sequential version.")


if __name__ == "__main__":
    main()
