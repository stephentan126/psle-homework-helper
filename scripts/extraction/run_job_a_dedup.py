r"""
Run the corpus-wide Job A deduplication pass (Issue 231).

A thin command-line wrapper around `app.pipeline.dedup.run_dedup_pass()`. See that module's
docstring for the design: cost, thresholds, tie-break rule and data safety.

By default the script is a full read-only preview. It computes and prints every auto-mark decision
and every review-queue entry, and writes nothing, so it is safe to run at any time.

`--apply` also performs the only change this pass makes, setting `Question.superseded_by` on each
non-survivor, and commits. Data changes always need an explicit flag in this project (as with
`scan_question_text_operator_mismatch.py --run-live-scan`), never the default.

Usage:
    # Read-only preview, no writes:
    backend/.venv/Scripts/python.exe scripts/extraction/run_job_a_dedup.py

    # Apply: sets superseded_by and commits:
    backend/.venv/Scripts/python.exe scripts/extraction/run_job_a_dedup.py --apply
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "backend"))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from app.db.session import get_session, init_db  # noqa: E402
from app.pipeline.dedup import AUTO_MARK_THRESHOLD, REVIEW_FLOOR, run_dedup_pass  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually set Question.superseded_by for every real auto-marked non-survivor and "
             "commit. Default is a real, read-only preview -- no writes.",
    )
    args = parser.parse_args()

    print(f"Job A dedup pass -- REVIEW_FLOOR={REVIEW_FLOOR}, AUTO_MARK_THRESHOLD="
          f"{AUTO_MARK_THRESHOLD}. Mode: {'APPLY (will write superseded_by + commit)' if args.apply else 'PREVIEW (read-only, no writes)'}.")

    init_db()
    start = time.time()
    with get_session() as session:  # type: ignore
        result = run_dedup_pass(session, apply=args.apply)
    elapsed = time.time() - start

    print(f"\nReal candidate pairs considered (similarity >= {REVIEW_FLOOR}): "
          f"{result.total_candidates_considered}. Elapsed: {elapsed:.2f}s.")

    total_superseded = sum(len(m.superseded_ids) for m in result.auto_marks)
    cluster_size_counts: dict[int, int] = {}
    for m in result.auto_marks:
        cluster_size_counts[m.cluster_size] = cluster_size_counts.get(m.cluster_size, 0) + 1

    print(f"\nReal AUTO-MARK clusters: {len(result.auto_marks)} (real questions superseded: "
          f"{total_superseded}).")
    for size in sorted(cluster_size_counts):
        label = "pair" if size == 2 else f"group of {size}"
        print(f"  {cluster_size_counts[size]} cluster(s) of size {size} ({label}s)")

    print(f"\nReal REVIEW QUEUE entries (no auto-action): {len(result.review_queue)}.")
    by_reason: dict[str, int] = {}
    for entry in result.review_queue:
        by_reason[entry.reason] = by_reason.get(entry.reason, 0) + 1
    for reason, count in sorted(by_reason.items(), key=lambda kv: -kv[1]):
        print(f"  {count} — {reason}")

    print("\nReal auto-mark clusters (survivor_id <- superseded_ids):")
    for m in sorted(result.auto_marks, key=lambda m: -m.cluster_size)[:50]:
        print(f"  survivor={m.survivor_id} <- {m.superseded_ids} (cluster size {m.cluster_size})")
    if len(result.auto_marks) > 50:
        print(f"  ... and {len(result.auto_marks) - 50} more.")

    print("\nReal review-queue entries (question_id_a, question_id_b, similarity, reason):")
    for entry in sorted(result.review_queue, key=lambda e: -e.similarity)[:50]:
        print(f"  {entry.question_id_a} vs {entry.question_id_b}: {entry.similarity:.4f} "
              f"({entry.reason})")
    if len(result.review_queue) > 50:
        print(f"  ... and {len(result.review_queue) - 50} more.")

    if not args.apply:
        print("\nPREVIEW ONLY -- nothing was written. Re-run with --apply to commit these "
              "auto-mark decisions for real.")


if __name__ == "__main__":
    main()
