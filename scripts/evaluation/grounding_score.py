r"""
Score the grounding experiment by exact answer match, and select the human spot-check sample.

Runs after grounding_generate.py, over conditions A, B and C.

Primary metric: automated exact-answer match on the 388 of 446 held-out rows that have a non-empty
`answer_value`. Answers are normalised with `strip_units()` from `gate.py`, imported rather than
reimplemented, so grading uses the same unit and formatting tolerance as the live app. Results
are reported overall and split by has_diagram=True (212 of 446) and has_diagram=False, as
Issue 47 requires.

Secondary result: a human-graded spot check, not an LLM judge over everything. It covers two
groups: (1) all 58 rows without a single usable `answer_value`, which exact match cannot grade,
and (2) a random sample of the exact-match-eligible rows, to check that the automated comparison
itself can be trusted, since a normalisation bug would otherwise shift the primary number
unnoticed. This script selects and reports the sample ids; the grading is done by hand.

Sample size uses the finite-population correction already applied to the has_diagram sample
(Issue 309). The population is the 388 eligible rows:
n = N*X/(X+N-1), with X = Z^2*p*(1-p)/E^2, 95% confidence, a +/-10% margin and a conservative
p=0.5 (no prior agreement-rate estimate exists). This gives X = 1.96^2*0.25/0.10^2 ~= 96 and
n = 388*96/(96+388-1) ~= 77. The 58 non-eligible rows are all checked rather than sampled: the
group is small, and automated grading gives no signal on them.

Outputs (data/extracted/slice9_eval/): score_summary.json, human_spot_check_sample.json.

Usage: backend/.venv/Scripts/python.exe scripts/evaluation/grounding_score.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "backend"))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_EVAL_DIR = _REPO_ROOT / "data" / "extracted" / "slice9_eval"
_CONDITIONS = ["A_base", "B_rag_fewshot", "C_qlora_rag_fewshot"]
_SPOT_CHECK_SEED = 20260915  # fixed seed, as for the has_diagram sample


def extract_final_answer(generated_text: str) -> str | None:
    """Return the value after the last "Final Answer:" marker (the format the prompt asks for).

    Returns None if the model did not follow the format (most likely for the ungrounded base
    condition), rather than guessing a value from free text.
    """
    marker = "Final Answer:"
    idx = generated_text.rfind(marker)
    if idx == -1:
        return None
    value = generated_text[idx + len(marker):].strip()
    value = value.splitlines()[0].strip() if value else ""
    return value or None


def answers_match(predicted: str | None, real_answer_value: str) -> bool:
    """Compare answers after normalising both with gate.strip_units(), as the live app does."""
    if predicted is None:
        return False
    from app.pipeline.gate import strip_units

    pred_stripped, pred_unit = strip_units(predicted)
    real_stripped, real_unit = strip_units(real_answer_value)
    return pred_stripped.strip().lower() == real_stripped.strip().lower()


def score_condition(condition: str) -> dict:
    path = _EVAL_DIR / f"generations_{condition}.json"
    with open(path, encoding="utf-8") as f:
        rows = json.load(f).values()

    graded, no_answer_value, no_extracted_answer = [], [], []
    for r in rows:
        real_av = r["real_answer_value"]
        if not real_av:
            no_answer_value.append(r)
            continue
        predicted = extract_final_answer(r["generated_text"])
        if predicted is None:
            no_extracted_answer.append(r)
        correct = answers_match(predicted, real_av)
        graded.append({**r, "predicted_answer": predicted, "correct": correct})

    def _rate(subset):
        return (sum(1 for g in subset if g["correct"]) / len(subset)) if subset else None

    overall = graded
    diagram_true = [g for g in graded if g["has_diagram"]]
    diagram_false = [g for g in graded if not g["has_diagram"]]

    return {
        "condition": condition,
        "n_graded": len(overall), "n_correct": sum(1 for g in overall if g["correct"]),
        "accuracy_overall": _rate(overall),
        "n_diagram_true": len(diagram_true), "accuracy_diagram_true": _rate(diagram_true),
        "n_diagram_false": len(diagram_false), "accuracy_diagram_false": _rate(diagram_false),
        "n_no_answer_value_in_held_out": len(no_answer_value),
        "n_model_did_not_follow_format": len(no_extracted_answer),
        "graded_rows": overall,
        "no_answer_value_ids": [r["question_id"] for r in no_answer_value],
    }


def select_spot_check_sample() -> dict:
    """Select the spot-check sample; see the module docstring for the sample size.

    Returns ids only; nothing is graded here.
    """
    ref_path = _EVAL_DIR / f"generations_{_CONDITIONS[0]}.json"
    with open(ref_path, encoding="utf-8") as f:
        rows = list(json.load(f).values())

    eligible_ids = sorted(r["question_id"] for r in rows if r["real_answer_value"])
    non_eligible_ids = sorted(r["question_id"] for r in rows if not r["real_answer_value"])
    N = len(eligible_ids)
    assert N == 388, f"expected 388 exact-match-eligible held-out rows, got {N}"

    Z, p, E = 1.96, 0.5, 0.10
    X = (Z ** 2) * p * (1 - p) / (E ** 2)
    n = round(N * X / (X + N - 1))

    rng = random.Random(_SPOT_CHECK_SEED)
    sample_ids = sorted(rng.sample(eligible_ids, n))

    return {
        "population_N": N, "formula_X": round(X, 2), "sample_size_n": n,
        "seed": _SPOT_CHECK_SEED,
        "eligible_sample_ids": sample_ids,
        "all_non_eligible_ids_full_check": non_eligible_ids,
        "n_non_eligible": len(non_eligible_ids),
    }


def main() -> None:
    summary = {}
    for condition in _CONDITIONS:
        path = _EVAL_DIR / f"generations_{condition}.json"
        if not path.exists():
            print(f"[{condition}] no generations file yet at {path} -- skipping")
            continue
        result = score_condition(condition)
        summary[condition] = {k: v for k, v in result.items() if k not in ("graded_rows",)}
        print(f"\n=== {condition} ===")
        print(f"  overall:       {result['n_correct']}/{result['n_graded']} "
              f"({result['accuracy_overall']:.1%})" if result["accuracy_overall"] is not None else "  no graded rows")
        print(f"  has_diagram=T: accuracy={result['accuracy_diagram_true']}, n={result['n_diagram_true']}")
        print(f"  has_diagram=F: accuracy={result['accuracy_diagram_false']}, n={result['n_diagram_false']}")
        print(f"  no answer_value in held-out (ungraded): {result['n_no_answer_value_in_held_out']}")
        print(f"  model did not follow output format (counted wrong, not skipped): "
              f"{result['n_model_did_not_follow_format']}")

    if summary:
        _EVAL_DIR.mkdir(parents=True, exist_ok=True)
        with open(_EVAL_DIR / "score_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nSaved score summary -> {_EVAL_DIR / 'score_summary.json'}")

        spot_check = select_spot_check_sample()
        with open(_EVAL_DIR / "human_spot_check_sample.json", "w", encoding="utf-8") as f:
            json.dump(spot_check, f, indent=2, ensure_ascii=False)
        print(f"\n=== Human spot-check sample (selected, NOT graded here) ===")
        print(f"  population N={spot_check['population_N']}, X={spot_check['formula_X']}, "
              f"n={spot_check['sample_size_n']} (95% CI, +/-10% margin, p=0.5, seed={spot_check['seed']})")
        print(f"  {spot_check['n_non_eligible']} non-eligible rows checked in full (not sampled)")
        print(f"  Saved -> {_EVAL_DIR / 'human_spot_check_sample.json'}")


if __name__ == "__main__":
    main()
