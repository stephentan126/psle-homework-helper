"""
GPU-free, no-download dry run of run_bakeoff.py's real SCORING logic against the real 21-row gold
set (Issue 318). Monkeypatches only the two model-calling functions
(_extract_for_candidate/_answer_for_candidate) so no model is ever loaded, exercises the real MCQ
resolution, chart-type/value scoring, and rate aggregation logic before any real GPU time or
download happens.

Real, historical value, not hypothetical (Issue 320): this exact script caught two
real scoring bugs before the first live run, a LaTeX-encoded gold value (id=5096,
"\\(\\frac{2}{5}\\)") that a plain substring check would have silently, systematically scored
wrong forever, and a related fraction-tearing bug where a model's own LaTeX-formatted answer would
have been split into two unlinked digit tokens instead of read as one fraction. Re-run this after
ANY change to the scoring functions in run_bakeoff.py, before trusting a real GPU run's numbers.

Usage: backend/.venv/Scripts/python.exe evaluation/model_selection/diagram_charts/dry_run_scoring_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(_REPO_ROOT / "backend"))

import run_bakeoff as rb  # noqa: E402

with open(rb.GOLD_SET_PATH, encoding="utf-8") as f:
    gold_data = json.load(f)
rows_by_id = {r["id"]: r for r in gold_data["rows"]}


def main() -> None:
    print("=== Test 1: real MCQ resolution (Issue 318 convention) ===")
    for rid, expect_contains in [(318, None), (2047, "120"), (5096, None), (3158, None), (3601, None)]:
        row = rows_by_id[rid]
        resolved = rb._resolve_gold_answer(row)
        print(f"id={rid} stored_answer_value={row['stored_answer_value']!r} -> resolved={resolved!r}")
        if expect_contains:
            assert expect_contains in resolved, f"id={rid}: expected {expect_contains!r} in {resolved!r}"

    print(
        "\n=== Test 1b: LaTeX-encoded gold value (id=5096, real bug caught and fixed here "
        "BEFORE any real GPU run, Issue 320) ==="
    )
    row_5096 = rows_by_id[5096]
    assert rb._score_final_answer(row_5096, "2/5"), \
        "id=5096: model answer '2/5' must match LaTeX gold value \\(\\frac{2}{5}\\)"
    assert rb._score_final_answer(row_5096, "0.4"), \
        "id=5096: model answer '0.4' (numerically equal to 2/5) must also match"
    assert rb._score_final_answer(row_5096, r"\(\frac{2}{5}\)"), \
        "id=5096: a model echoing LaTeX notation back must ALSO match (the second real bug " \
        "this test caught — fraction-tearing into two unlinked digit tokens)"
    assert not rb._score_final_answer(row_5096, "1/3"), \
        "id=5096: a genuinely wrong answer must NOT be scored correct"
    print("LaTeX-encoded numeric gold value compares correctly via SymPy: plain-fraction, "
          "decimal-equivalent, and LaTeX-echoed model answers all accepted; a genuinely wrong "
          "answer still rejected.")

    print("\n=== Test 1c: natural-language MCQ gold value (id=318) uses the substring fallback ===")
    row_318 = rows_by_id[318]
    resolved_318 = rb._resolve_gold_answer(row_318)
    assert rb._score_final_answer(row_318, resolved_318)
    assert not rb._score_final_answer(row_318, "something totally unrelated")
    print("Prose MCQ option text compared correctly (non-numeric fallback path).")

    print("\n=== Test 2: image_based_mcq_options flag applies to 3158/3601 only ===")
    for rid in (318, 2047, 5096):
        assert rows_by_id[rid]["assumed_chart_type"] not in ("bar_and_pie_mcq", "pie_and_bar_mcq"), rid
    for rid in (3158, 3601):
        print(f"id={rid} assumed_chart_type={rows_by_id[rid]['assumed_chart_type']!r}")

    print("\n=== Test 3: full evaluate_candidate() dry run, QA-capable candidate (fake chat_vlm) ===")
    fake_bundle_qa = {"candidate_id": "fake/qa-model", "kind": "chat_vlm", "can_answer_qa": True}

    def _fake_extract(bundle, image_path):
        return "This is a bar chart showing values 1, 2, 3, 4."

    def _fake_answer_correct(bundle, image_path, question_text):
        # _answer_for_candidate()'s real contract (chat_vlm path) returns the value AFTER
        # extract_claimed_answer() has already stripped the "FINAL ANSWER:" prefix, the mock
        # must match that, not the raw prefixed line.
        row = next(r for r in gold_data["rows"] if r["question_text"] == question_text)
        return rb._resolve_gold_answer(row)

    with mock.patch.object(rb, "_extract_for_candidate", _fake_extract), \
         mock.patch.object(rb, "_answer_for_candidate", _fake_answer_correct):
        metrics = rb.evaluate_candidate(fake_bundle_qa, rb.GOLD_SET_PATH)
    print(metrics)
    assert metrics["n_rows"] == 21, metrics["n_rows"]
    assert metrics["final_answer_exact_match_rate"] == 1.0, metrics["final_answer_exact_match_rate"]
    print("Perfect-answer synthetic run scores 100% final-answer accuracy — the scoring pipeline "
          "itself is not silently miscounting.")

    print("\n=== Test 4: extraction-only candidate (fake docling/deplot) gets N/A on stage 4 ===")
    fake_bundle_extract_only = {
        "candidate_id": "fake/extract-only", "kind": "docling_extractor", "can_answer_qa": False,
    }
    with mock.patch.object(rb, "_extract_for_candidate", _fake_extract):
        metrics2 = rb.evaluate_candidate(fake_bundle_extract_only, rb.GOLD_SET_PATH)
    print(metrics2)
    assert metrics2["final_answer_exact_match_rate"] == "N/A (extraction-only candidate, Issue 320)"
    print("Extraction-only candidate correctly marked N/A, not scored as a miss.")

    print("\n=== Test 5: a candidate crash on one row doesn't kill the whole run ===")

    def _fake_extract_crashes_on_one(bundle, image_path):
        if "P6_Maths_2022_SA2_henrypark" in image_path:
            raise RuntimeError("simulated real model crash")
        return "some chart"

    with mock.patch.object(rb, "_extract_for_candidate", _fake_extract_crashes_on_one):
        metrics3 = rb.evaluate_candidate(fake_bundle_extract_only, rb.GOLD_SET_PATH)
    print(metrics3)
    assert metrics3["n_errors"] == 1, metrics3["n_errors"]
    assert metrics3["n_rows"] == 21, "a per-row crash must not drop the row count"
    print("Per-row crash correctly isolated: 1 error recorded, all 21 rows still present.")

    # Clean up this dry run's own fake per-row detail files, real output only, never fake
    # candidate-id files left sitting next to genuine results.
    import shutil
    per_row_dir = Path(__file__).resolve().parent / "per_row_results"
    for fake_id in ("fake__qa-model.json", "fake__extract-only.json"):
        fake_path = per_row_dir / fake_id
        if fake_path.exists():
            fake_path.unlink()
    if per_row_dir.exists() and not any(per_row_dir.iterdir()):
        shutil.rmtree(per_row_dir)

    print("\nALL DRY-RUN SCORING CHECKS PASSED (no model loaded, no GPU touched, no download).")


if __name__ == "__main__":
    main()
