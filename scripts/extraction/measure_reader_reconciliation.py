"""
Measure how often the two OCR readers agree across the corpus, without writing anything.

Run after the full-corpus DeepSeek-OCR pass (Issue 214; 4,422 of 4,423 pages, with Issue 221's
`nanyang/44` excluded). It calls the same functions as the Stage C driver (`_extract_one_file()`
in `backend/app/pipeline/extract.py`): `discover_pdfs()`, `find_matching_answer_file()`,
`normalized_school_name()`, and `extract_topology_a()` or `extract_topology_b()`. It stops before
the driver's `save(result, output_path)`. Those functions work in memory by design (the
single-writer database design: a Stage C worker's only side effect is the JSON checkpoint written
by `save()`), so no stored data is touched: not `data/extracted/questions/*.json`, and not
`psle.db`. The only file written is a summary JSON (and, with --out, the per-question records).

Why not re-run Stage C: `_extract_one_file()` skips any file whose checkpoint already exists
(`if output_path.exists(): return ("skipped", ...)`). Stage C had already run over the whole
corpus before the DeepSeek-OCR pass finished, so a normal re-run would skip everything. This
script never looks at the checkpoints, so it can measure agreement on the complete two-reader
data without deleting or overwriting any checkpoint.

Usage:
    backend/.venv/Scripts/python.exe scripts/extraction/measure_reader_reconciliation.py
    backend/.venv/Scripts/python.exe scripts/extraction/measure_reader_reconciliation.py --limit 10
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "backend"))

from app.pipeline.extract import (  # noqa: E402
    discover_pdfs,
    extract_topology_a,
    extract_topology_b,
    find_matching_answer_file,
    normalized_school_name,
)

# Monitoring band for the flagged (disagree) rate (Issues 127, 225). It replaces an unsourced
# 20-35% figure: that range does not appear in Issue 127, which rejects a flat percentage target,
# and it was wrongly attributed to Issue 127 in Issue 214. The band is centred on the validated,
# per-question flagged rate (751/5,705 = 13.16%, Issue 224, all 132 files) with a +/-5 percentage
# point buffer for variation between corpora. The buffer is reasoned, not measured: the only two
# measurements are of this same corpus, before and after Issue 224's fix (13.31% and 13.16%, a
# spread of 0.15 points). The band flags unexpected drift in a future re-run. It is not a
# pass/fail test for this corpus, whose acceptance is governed by Issue 127's four-category
# review requirement.
_ACCEPTANCE_BAND_LOW = 8.16
_ACCEPTANCE_BAND_HIGH = 18.16


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=_REPO_ROOT / "data")
    parser.add_argument("--raw-dir", type=Path, default=_REPO_ROOT / "data" / "extracted" / "raw")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N discovered files (smoke test).")
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Optional path to write the full real per-question record list as JSON (for later "
             "review) — still never touches the DB or any Stage C checkpoint.",
    )
    args = parser.parse_args()

    pdfs = discover_pdfs(args.data_dir)
    if args.limit is not None:
        pdfs = pdfs[: args.limit]
    print(f"=== measure_reader_reconciliation: {len(pdfs)} real question-source PDF(s) discovered ===")

    agreement_counts: Counter[str] = Counter()
    disagreement_type_counts: Counter[str] = Counter()
    disagree_records: list[dict] = []
    all_records: list[dict] = []
    files_ok = 0
    files_failed: list[tuple[str, str]] = []

    for i, pdf_path in enumerate(pdfs, start=1):
        try:
            answer_file = find_matching_answer_file(pdf_path)
            school_name, school_name_verified = normalized_school_name(pdf_path)

            if answer_file is not None:
                result = extract_topology_a(
                    pdf_path, answer_file, school_name, args.raw_dir,
                    discrepancy_log=None, school_name_verified=school_name_verified,
                )
            else:
                result = extract_topology_b(
                    pdf_path, school_name, args.raw_dir,
                    discrepancy_log=None, school_name_verified=school_name_verified,
                )

            for pair in result.pairs:
                record = pair.ocr_record
                if record is None:
                    continue  # both readers failed or no Stage B output: nothing to count
                status = record.get("agreement_status")
                agreement_counts[status] += 1
                dtype = record.get("disagreement_type")
                if status == "disagree":
                    disagreement_type_counts[dtype] += 1
                    disagree_records.append(
                        {
                            "question_id": record.get("question_id"),
                            "disagreement_type": dtype,
                            "disagreement_detail": record.get("disagreement_detail"),
                        },
                    )
                all_records.append(
                    {
                        "question_id": record.get("question_id"),
                        "agreement_status": status,
                        "disagreement_type": dtype,
                    },
                )
            files_ok += 1
        except Exception as e:  # noqa: BLE001, one bad file must not stop the run
            files_failed.append((str(pdf_path), f"{type(e).__name__}: {e}"))

        if i % 20 == 0 or i == len(pdfs):
            print(f"  ...{i}/{len(pdfs)} files processed ({files_ok} ok, {len(files_failed)} failed)")

    total = sum(agreement_counts.values())
    agree = agreement_counts.get("agree", 0)
    disagree = agreement_counts.get("disagree", 0)
    single_reader_only = agreement_counts.get("single_reader_only", 0)
    both_failed_equivalent = total - agree - disagree - single_reader_only  # expected 0

    print()
    print("=" * 90)
    print(f"Files: {files_ok} processed OK, {len(files_failed)} failed")
    if files_failed:
        for path, err in files_failed[:20]:
            print(f"  FAILED: {path} -- {err}")
        if len(files_failed) > 20:
            print(f"  ... and {len(files_failed) - 20} more (see full list not printed here)")
    print()
    print(f"Total real per-question OCR records: {total}")
    for status in ("agree", "disagree", "single_reader_only"):
        n = agreement_counts.get(status, 0)
        pct = 100.0 * n / total if total else float("nan")
        print(f"  {status:20s}: {n:6d}  ({pct:5.1f}%)")
    if both_failed_equivalent:
        print(f"  (other/unexpected)  : {both_failed_equivalent:6d}")
    print()
    disagree_pct = 100.0 * disagree / total if total else float("nan")
    print(f"REAL FLAGGED (disagree) RATE: {disagree}/{total} = {disagree_pct:.2f}%")
    print(
        f"Acceptance band (Issue 127 recalibration, Issue 225): "
        f"{_ACCEPTANCE_BAND_LOW:.2f}%-{_ACCEPTANCE_BAND_HIGH:.2f}% "
        f"(13.16% +/- 5pp, a monitoring range for future drift, not a pass/fail gate on this run).",
    )
    if total and _ACCEPTANCE_BAND_LOW <= disagree_pct <= _ACCEPTANCE_BAND_HIGH:
        print(f"  -> WITHIN the {_ACCEPTANCE_BAND_LOW:.2f}%-{_ACCEPTANCE_BAND_HIGH:.2f}% band.")
    elif total:
        print(
            f"  -> OUTSIDE the {_ACCEPTANCE_BAND_LOW:.2f}%-{_ACCEPTANCE_BAND_HIGH:.2f}% band "
            f"(real result, not reinterpreted) -- flag for a real decision.",
        )
    print()
    print("Real disagreement_type breakdown (of the disagree count above):")
    for dtype in ("value_mismatch", "structural_mismatch", None):
        n = disagreement_type_counts.get(dtype, 0)
        pct = 100.0 * n / disagree if disagree else float("nan")
        print(f"  {str(dtype):20s}: {n:6d}  ({pct:5.1f}% of disagree)")
    other_dtypes = set(disagreement_type_counts) - {"value_mismatch", "structural_mismatch", None}
    for dtype in other_dtypes:
        print(f"  UNEXPECTED dtype={dtype!r}: {disagreement_type_counts[dtype]}")

    summary = {
        "files_ok": files_ok,
        "files_failed": len(files_failed),
        "total_records": total,
        "agreement_counts": dict(agreement_counts),
        "disagree_pct_of_total": disagree_pct,
        "acceptance_band": [_ACCEPTANCE_BAND_LOW, _ACCEPTANCE_BAND_HIGH],
        "within_acceptance_band": (
            _ACCEPTANCE_BAND_LOW <= disagree_pct <= _ACCEPTANCE_BAND_HIGH
        ) if total else None,
        "disagreement_type_counts": {str(k): v for k, v in disagreement_type_counts.items()},
    }
    summary_path = _REPO_ROOT / "data" / "extracted" / "_reader_reconciliation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary written to {summary_path} (not a DB write, not a Stage C checkpoint).")

    if args.out:
        args.out.write_text(json.dumps(all_records, indent=2), encoding="utf-8")
        print(f"Full real per-question record list written to {args.out}.")


if __name__ == "__main__":
    main()
