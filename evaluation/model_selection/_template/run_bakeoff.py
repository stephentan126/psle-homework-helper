"""
Bake-off script template, copy this folder for every model slot that needs the
three-candidates-plus-comparison-table-plus-rejection-reason treatment (the design specification,
Issue 50 / 51 / 52 / 53).

Usage pattern:
1. Copy evaluation/model_selection/_template/ to evaluation/model_selection/<slot_name>/ (e.g. evaluation/model_selection/topic_classifier/)
2. List real candidates in CANDIDATES below
3. Point GOLD_SET_PATH at a real, hand-labeled gold set for this slot
4. Fill in load_candidate() and evaluate_candidate() for this specific slot's job
5. Run it. Results write to results.md AND results.csv in this folder.
6. Fill in the REJECTION REASON column by hand for every candidate not picked, this is required
   evidence for the report, not optional.

ALWAYS use backend/app/services/gpu_utils.py's free_gpu() between candidates (see the loop below), this is
non-negotiable for any bake-off touching more than one model, per the Windows/VRAM hiccup already
documented in the design specification.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "backend"))
from app.services.gpu_utils import free_gpu, measure_load  # noqa: E402

# ---------------------------------------------------------------------------
# FILL THIS IN per slot
# ---------------------------------------------------------------------------

CANDIDATES = [
    # "candidate_model_id_1",
    # "candidate_model_id_2",
    # "candidate_model_id_3",
]

GOLD_SET_PATH = Path(__file__).parent / "gold_set_labels"  # hand-labeled ground truth for this slot

RESULTS_MD = Path(__file__).parent / "results.md"
RESULTS_CSV = Path(__file__).parent / "results.csv"


def load_candidate(candidate_id: str):
    """Load one candidate model. Replace with the real loading code for this slot."""
    raise NotImplementedError("Fill in load_candidate() for this bake-off slot")


def evaluate_candidate(model, gold_set_path: Path) -> dict:
    """Run the candidate against the gold set. Return a dict of metrics.

    Return shape should match whatever this slot's gate metrics are (the design specification/11),
    e.g. for OCR: {"accuracy": ..., "fidelity_gate_passed": ..., ...}
    for the topic classifier: {"accuracy_topics_6_7": ..., "overall_accuracy": ...}
    """
    raise NotImplementedError("Fill in evaluate_candidate() for this bake-off slot")


# ---------------------------------------------------------------------------
# Runner, should not need changes per slot
# ---------------------------------------------------------------------------


def main() -> None:
    if not CANDIDATES:
        print("CANDIDATES is empty. Fill it in before running this bake-off.")
        sys.exit(1)

    all_results = []
    for candidate_id in CANDIDATES:
        print(f"\n=== Testing {candidate_id} ===")
        with measure_load(candidate_id) as timing:
            model = load_candidate(candidate_id)

        metrics = evaluate_candidate(model, GOLD_SET_PATH)
        metrics["candidate"] = candidate_id
        metrics["load_time_s"] = round(timing.load_time_s, 2)
        metrics["vram_delta_gb"] = round(timing.vram_delta_gb, 2)
        metrics["fits_8gb"] = timing.vram_delta_gb <= 8.0
        all_results.append(metrics)

        free_gpu(model)  # MANDATORY, do not remove, see module docstring

    write_results(all_results)
    print(f"\nResults written to {RESULTS_MD} and {RESULTS_CSV}")
    print("Now fill in the rejection reason for every candidate NOT selected, by hand, in")
    print("results.md — this is required report evidence, not optional (Issue 50).")


def write_results(results: list[dict]) -> None:
    if not results:
        return
    fieldnames = list(results[0].keys()) + ["role", "rejection_reason"]
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            row.setdefault("role", "TBD: primary/secondary/rejected")
            row.setdefault("rejection_reason", "TBD: fill in by hand if not selected")
            writer.writerow(row)

    with open(RESULTS_MD, "w") as f:
        f.write("# Bake-off Results\n\n")
        f.write("Fill in `role` and `rejection_reason` for every row before this counts as\n")
        f.write("report-ready evidence (the design specification Issue 50).\n\n")
        f.write("| " + " | ".join(fieldnames) + " |\n")
        f.write("|" + "---|" * len(fieldnames) + "\n")
        for row in results:
            row.setdefault("role", "TBD")
            row.setdefault("rejection_reason", "TBD")
            f.write("| " + " | ".join(str(row.get(k, "")) for k in fieldnames) + " |\n")


if __name__ == "__main__":
    main()
