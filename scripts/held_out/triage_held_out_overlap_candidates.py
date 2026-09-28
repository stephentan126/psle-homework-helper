r"""
Sort the held-out/training overlap candidates into two review priority groups.

Runs after `check_held_out_training_overlap.py`. It only sorts: it renders no images, reaches no
verdicts and changes no data. The groups follow the rule used to pick the first pairs for visual
review:
  - Group A (higher priority): no secondary-conflict signal fired
    (_distinguishing_terms_conflict, _numeric_terms_conflict and _numeric_multiset_conflict are
    all False). Issue 250 showed this class can hide a true overlap behind a generic,
    diagram-dependent multiple-choice stem with nothing in the text to catch it.
  - Group B (lower priority): at least one signal fired, meaning the two texts differ in a
    concrete way (a number or a labelled term). These are more likely false positives, a pattern
    confirmed on Issue 250's pairs 23 and 24 and on held-out ids 3, 56 and 160.

Each group is sorted by similarity, highest first. Pairs already reviewed by eye are marked, so the
CSV shows what is left.

Input: data/extracted/held_out_training_overlap_candidates.json.
Output: data/extracted/held_out_overlap_triage.csv.

Usage:
    backend/.venv/Scripts/python.exe scripts/held_out/triage_held_out_overlap_candidates.py
"""
import csv
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
CANDIDATES_PATH = _REPO_ROOT / "data" / "extracted" / "held_out_training_overlap_candidates.json"
OUT_PATH = _REPO_ROOT / "data" / "extracted" / "held_out_overlap_triage.csv"

# Pairs already reviewed by eye, keyed by (held_out_id, training_pool_id).
ALREADY_REVIEWED = {
    (3, 3651): "FALSE POSITIVE (confirmed, different diagrams)",
    (7, 3336): "REAL OVERLAP (confirmed, same diagrams reordered)",
    (56, 3472): "FALSE POSITIVE (confirmed, different diagrams)",
    (160, 3652): "FALSE POSITIVE (confirmed, different diagrams)",
}


def main() -> None:
    candidates = json.loads(CANDIDATES_PATH.read_text(encoding="utf-8"))
    print(f"Real total candidates loaded: {len(candidates)}")

    group_a = [c for c in candidates if not c["any_secondary_conflict"]]
    group_b = [c for c in candidates if c["any_secondary_conflict"]]
    group_a.sort(key=lambda c: -c["similarity"])
    group_b.sort(key=lambda c: -c["similarity"])

    print(f"GROUP A (no secondary-conflict signal, HIGHER priority): {len(group_a)}")
    print(f"GROUP B (secondary-conflict signal tripped, LOWER priority): {len(group_b)}")

    n_reviewed_a = sum(1 for c in group_a if (c["held_out_id"], c["training_pool_id"]) in ALREADY_REVIEWED)
    n_reviewed_b = sum(1 for c in group_b if (c["held_out_id"], c["training_pool_id"]) in ALREADY_REVIEWED)
    print(f"Already reviewed in Group A: {n_reviewed_a}. Remaining: {len(group_a) - n_reviewed_a}.")
    print(f"Already reviewed in Group B: {n_reviewed_b}. Remaining: {len(group_b) - n_reviewed_b}.")

    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "priority_group", "similarity", "already_reviewed_verdict",
            "held_out_id", "held_out_source_file", "held_out_page", "held_out_question_number",
            "held_out_question_text",
            "training_pool_id", "training_pool_source_file", "training_pool_page",
            "training_pool_question_number", "training_pool_question_text",
            "distinguishing_terms_conflict", "numeric_terms_conflict", "numeric_multiset_conflict",
        ])
        for group_label, group in (("A_no_conflict_HIGHER_priority", group_a),
                                    ("B_has_conflict_LOWER_priority", group_b)):
            for c in group:
                key = (c["held_out_id"], c["training_pool_id"])
                verdict = ALREADY_REVIEWED.get(key, "")
                ho_meta, tp_meta = c["held_out_meta"], c["training_pool_meta"]
                writer.writerow([
                    group_label, f"{c['similarity']:.6f}", verdict,
                    c["held_out_id"], ho_meta[0], ho_meta[1], ho_meta[2], c["held_out_text"],
                    c["training_pool_id"], tp_meta[0], tp_meta[1], tp_meta[2],
                    c["training_pool_text"],
                    c["distinguishing_terms_conflict"], c["numeric_terms_conflict"],
                    c["numeric_multiset_conflict"],
                ])

    print(f"\nWrote real triage CSV to: {OUT_PATH}")


if __name__ == "__main__":
    main()
