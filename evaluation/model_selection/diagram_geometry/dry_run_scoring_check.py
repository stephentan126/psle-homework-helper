"""
GPU-free, no-download dry run of run_bakeoff.py's real SCORING logic against the real 22-row gold
set (Issue 331). Monkeypatches only the two model-calling functions
(_extract_for_candidate/_answer_for_candidate) so no model is ever loaded, exercises the real
MCQ resolution, geometry-type/value scoring, multi-part-answer handling (Issue 332), the
LaTeX-measurement-wrapper fix (Issue 333), and rate aggregation logic before any real GPU time
or download happens. Same purpose and structure as Bucket 1's dry_run_scoring_check.py, which
caught two real scoring bugs before its own first live run, this file's own first run caught two
more, of a different shape, before any Bucket 2 GPU time was spent either. Re-run this after ANY
change to the scoring functions in run_bakeoff.py.

Usage: backend/.venv/Scripts/python.exe evaluation/model_selection/diagram_geometry/dry_run_scoring_check.py

Diagram buckets: Bucket 1 is charts and data, Bucket 2 is simple geometry with measurements,
and Bucket 3 is bar models and number lines.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

# REAL FINDING, distinct root cause from Issue 329 (confirmed by direct investigation, not
# assumed from the symptom alone): this script's own printed gold values can contain real,
# cp1252-representable characters (e.g. U+00B0 '°') that Python's default Windows stdout encoding
# (inherited from the console's active codepage, cp1252 on this machine) encodes without error,
# no UnicodeEncodeError, no crash, so Issue 329's own `_safe_print()` (a crash-avoidance
# re-encode) would be a no-op here and does not apply. The real corruption happens downstream: the
# terminal/tool capturing this process's stdout decodes it as UTF-8, and a lone cp1252 byte like
# 0xB0 is not valid UTF-8, so it renders as U+FFFD. Forcing this script's own stdout to real UTF-8
# fixes it at the source regardless of the downstream reader. `errors="backslashreplace"` kept as
# a safety net for any character that genuinely can't round-trip, matching Issue 329's own
# philosophy without reusing its literal fix (which targets a different failure mode).
sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(_REPO_ROOT / "backend"))

import run_bakeoff as rb  # noqa: E402

with open(rb.GOLD_SET_PATH, encoding="utf-8") as f:
    gold_data = json.load(f)
rows_by_id = {r["id"]: r for r in gold_data["rows"]}


def main() -> None:
    print("=== Test 1: real MCQ resolution (Issue 318 convention, reused) ===")
    row_3 = rows_by_id[3]
    resolved_3 = rb._resolve_gold_answer(row_3)
    print(f"id=3 stored_answer_value={row_3['stored_answer_value']!r} -> resolved={resolved_3!r}")
    assert "106" in resolved_3, f"id=3: expected option text containing '106', got {resolved_3!r}"

    print("\n=== Test 2: multi-part gold answer detection (Issue 332) ===")
    multi_part_ids = {4356, 1568, 2657, 3905, 1304}
    single_value_ids = {3, 652, 1337, 4330, 183, 1348, 3165, 5617, 3983, 500, 3422, 1663, 5293, 2503}
    for rid in multi_part_ids:
        gold = rows_by_id[rid].get("corrected_answer_value") or rows_by_id[rid]["stored_answer_value"]
        assert rb._is_multi_part_gold(gold), f"id={rid}: expected multi-part detection on {gold!r}"
    for rid in single_value_ids:
        gold = rows_by_id[rid].get("corrected_answer_value") or rows_by_id[rid]["stored_answer_value"]
        assert not rb._is_multi_part_gold(gold), f"id={rid}: false-positive multi-part on {gold!r}"
    print(
        f"Multi-part detection correct on all {len(multi_part_ids)} real compound rows and all "
        f"{len(single_value_ids)} real single-value rows checked -- no false positives or negatives.",
    )

    print("\n=== Test 3: degree-symbol gold values compare correctly (real, not LaTeX-encoded) ===")
    row_4330 = rows_by_id[4330]
    assert rb._score_final_answer(row_4330, "65"), "id=4330: plain '65' must match gold '65deg'"
    assert rb._score_final_answer(row_4330, "The angle is 65 degrees."), \
        "id=4330: '65' embedded in free text must still match via the numeric-token fallback path"
    assert not rb._score_final_answer(row_4330, "64"), \
        "id=4330: an off-by-one wrong answer must NOT be scored correct"
    print("Degree-symbol gold values compare correctly: exact match, free-text embedded match, "
          "and a genuinely wrong answer all handled correctly.")

    print("\n=== Test 3b: LaTeX-wrapped symbolic (pi-based) gold value, real finding, Issue 333 ===")
    # id=1663's real gold value is a semicircle-perimeter MCQ option:
    # \((6\pi + 12)\mathrm{cm}\), a real case Bucket 1's own chart-only gold set never had
    # (no pi-based answers there). Caught by this dry run BEFORE any GPU time: strip_units() alone
    # does not unwrap a LaTeX \mathrm{} unit wrapper, and plain parse_expr() rejects "6pi" without
    # implicit-multiplication support.
    row_1663 = rows_by_id[1663]
    assert rb._score_final_answer(row_1663, "6pi+12"), \
        "id=1663: a compliant symbolic answer '6pi+12' must match the real pi-based gold value"
    assert rb._score_final_answer(row_1663, "12+6pi"), \
        "id=1663: term order must not matter (SymPy simplify, not string comparison)"
    assert not rb._score_final_answer(row_1663, "30.85"), \
        "id=1663: a DECIMAL APPROXIMATION of 6pi+12 (~30.85) must NOT be scored correct -- real, " \
        "disclosed limitation of exact SymPy equality on irrational/symbolic gold values, not a " \
        "bug to silently work around; a model must answer symbolically to be scored correct here"
    print("Symbolic pi-based gold value compares correctly: exact symbolic match accepted "
          "regardless of term order, decimal approximation correctly rejected (disclosed "
          "limitation, not a bug).")

    print("\n=== Test 3c: LaTeX cubic-unit wrapper with a stray tie, Issue 333 ===")
    # id=5293's real gold value is \(162 \mathrm{~cm}^{3}\), the "~" (LaTeX tie) and the
    # caret-brace exponent on the unit both blocked strip_units() from recognizing "cm" at all
    # before this fix.
    row_5293 = rows_by_id[5293]
    assert rb._score_final_answer(row_5293, "162"), \
        "id=5293: plain '162' must match despite the real ~cm^{3} wrapper on the gold value"
    print("Cubic-unit LaTeX wrapper with a stray tie character strips correctly.")

    print("\n=== Test 4: geometry-type scoring, including the angle_and_area_perimeter combo ===")
    assert rb._score_geometry_type("This shows an angle of 65 degrees.", "angle") is True
    assert rb._score_geometry_type("This is a rectangle with area 20cm2.", "angle") is False
    assert rb._score_geometry_type("A triangle with perimeter 12cm.", "angle_and_area_perimeter") is True
    assert rb._score_geometry_type("Some unrelated description.", "angle_and_area_perimeter") is False
    print("Geometry-type keyword scoring correct on single and combo categories.")

    print("\n=== Test 5: full evaluate_candidate() dry run, all 22 real rows, perfect-answer model ===")
    fake_bundle_qa = {"candidate_id": "fake/qa-model", "kind": "chat_vlm", "can_answer_qa": True}

    def _fake_extract(bundle, image_path):
        return "This figure shows an angle with area and perimeter values: 1, 2, 3, 4."

    def _fake_answer_correct(bundle, image_path, question_text):
        # Simulates a REASONABLY COMPLIANT model, one that actually follows the QA prompt's
        # "plain number... no units" instruction, rather than a model that echoes the raw gold
        # string verbatim (which, for a row like id=1663, is itself LaTeX/units-wrapped MCQ option
        # text, not a plain value; no real compliant model would echo that unchanged). Runs the
        # resolved gold value through the same normalize -> unwrap -> strip_units pipeline
        # _score_final_answer() itself uses on the gold side, so this stays a fair "perfect
        # answer" simulation (Issue 333's own finding, caught by this test).
        row = next(r for r in gold_data["rows"] if r["question_text"] == question_text)
        resolved = rb._strip_latex_math_wrappers(rb._normalize_latex_expression(rb._resolve_gold_answer(row)))
        numeric, _ = rb.strip_units(resolved)
        return numeric.strip()

    with mock.patch.object(rb, "_extract_for_candidate", _fake_extract), \
         mock.patch.object(rb, "_answer_for_candidate", _fake_answer_correct):
        metrics = rb.evaluate_candidate(fake_bundle_qa, rb.GOLD_SET_PATH)
    print(metrics)
    assert metrics["n_rows"] == 22, metrics["n_rows"]
    assert metrics["n_multi_part_rows"] == 5, metrics["n_multi_part_rows"]
    assert metrics["final_answer_exact_match_rate"] == 1.0, metrics["final_answer_exact_match_rate"]
    print("Perfect-answer synthetic run scores 100% on the 17 real scoreable rows; the 5 "
          "multi-part rows are correctly excluded as N/A, not counted as misses.")

    print("\n=== Test 6: a candidate crash on one row doesn't kill the whole run ===")

    def _fake_extract_crashes_on_one(bundle, image_path):
        if "Henry-Park" in image_path or "henrypark" in image_path.lower():
            raise RuntimeError("simulated real model crash")
        return "some figure"

    with mock.patch.object(rb, "_extract_for_candidate", _fake_extract_crashes_on_one), \
         mock.patch.object(rb, "_answer_for_candidate", _fake_answer_correct):
        metrics2 = rb.evaluate_candidate(fake_bundle_qa, rb.GOLD_SET_PATH)
    print(metrics2)
    assert metrics2["n_errors"] >= 1, metrics2["n_errors"]
    assert metrics2["n_rows"] == 22, "a per-row crash must not drop the row count"
    print("Per-row crash correctly isolated: run continued, all 22 rows still present.")

    fake_path = Path(__file__).resolve().parent / "per_row_results" / "fake__qa-model.json"
    if fake_path.exists():
        fake_path.unlink()

    print("\n=== Test 7: MCQ answered as a bare option index scores correct (Issue 341) ===")
    # Real finding: Qwen2.5-VL-7B answered id=3 correctly ("FINAL ANSWER: 3", option 3 = 106 deg) and
    # _score_final_answer() marked it wrong, because gold was resolved to the option TEXT only.
    assert rb._score_final_answer(row_3, rb.extract_claimed_answer("...so (3).\n\nFINAL ANSWER: 3")), (
        "id=3's real 'FINAL ANSWER: 3' must now score correct")
    for ok in ("3", "(3)", " 3 ", "106", "106°"):
        assert rb._score_final_answer(row_3, ok), f"id=3: {ok!r} (correct option/value) must score True"
    for bad in ("1", "2", "4", "(4)", "74", "90", "286"):
        assert not rb._score_final_answer(row_3, bad), f"id=3: {bad!r} (wrong option/value) must score False"
    # every real MCQ-with-index row in the gold set: its own gold index scores True, any other index False
    mcq_ids = []
    for rid, r in rows_by_id.items():
        gold = (r.get("corrected_answer_value") or r["stored_answer_value"]).strip()
        if r.get("status") == "excluded" or not rb.is_mcq_with_unresolved_index(r.get("question_text", ""), gold):
            continue
        mcq_ids.append(rid)
        assert rb._score_final_answer(r, gold), f"id={rid}: its own gold index {gold!r} must score True"
        assert rb._score_final_answer(r, f"({gold})"), f"id={rid}: '({gold})' must score True"
        for other in {"1", "2", "3", "4"} - {gold}:
            assert not rb._score_final_answer(r, other), f"id={rid}: wrong index {other!r} must score False"
    print(f"MCQ-with-index rows checked: {sorted(mcq_ids)}")
    assert len(mcq_ids) == 8, f"expected 8 real MCQ-with-index rows, got {len(mcq_ids)}"
    # non-MCQ rows must be unaffected: a bare small integer is NOT accepted just because it is small
    row_4330 = rows_by_id[4330]
    assert not rb._score_final_answer(row_4330, "3"), "id=4330 (not MCQ): '3' must stay False"
    assert rb._score_final_answer(row_4330, "65"), "id=4330: correct value must still score True"
    assert rb._score_final_answer(row_4330, "65°"), "id=4330: correct value with degree sign must still score True"
    print("Bare option index accepted for real MCQ rows (correct index True, wrong index False, "
          "'(3)' form accepted); resolved-text matching unchanged; non-MCQ rows unaffected.")

    print("\n=== Test 8: litre-symbol unit fix (Issue 344) changes no other real gold row ===")
    from app.pipeline import gate as _gate

    def _strip_units_before_landmine_344(text):
        """gate.strip_units() exactly as it behaved before Issue 344, rebuilt from gate's own regexes."""
        s = text.strip().translate(_gate._SUPERSCRIPT_FOLD)
        dollar = _gate._LEADING_DOLLAR_RE.match(s) if s.count("$") == 1 else None
        if dollar:
            return s[dollar.end():].strip(), "$"
        m = _gate._UNIT_SUFFIX_RE.search(s)
        if m:
            return s[: m.start()].strip(), m.group(1)
        return s, None

    changed, multi_part_changed = [], []
    for rid, r in rows_by_id.items():
        if r.get("status") == "excluded":
            continue
        resolved = rb._strip_latex_math_wrappers(rb._normalize_latex_expression(rb._resolve_gold_answer(r)))
        if _gate.strip_units(resolved) != _strip_units_before_landmine_344(resolved):
            (multi_part_changed if rb._is_multi_part_gold(resolved) else changed).append(rid)
    print(f"scored gold rows whose strip_units() output changed: {changed}; "
          f"multi-part (N/A, never scored) rows changed: {multi_part_changed}")
    assert changed == [1745], f"only id=1745 (gold '6ℓ') should change, got {changed}"
    row_1745 = rows_by_id[1745]
    assert rb._score_final_answer(row_1745, "6"), "id=1745: correct bare value '6' must now score True"
    assert rb._score_final_answer(row_1745, "6ℓ"), "id=1745: '6ℓ' must still score True"
    assert not rb._score_final_answer(row_1745, "5"), "id=1745: a wrong value must still score False"
    print("Only id=1745 changes; its bare correct value now scores True; a wrong value still scores False.")

    print("\nALL DRY-RUN SCORING CHECKS PASSED (no model loaded, no GPU touched, no download).")


if __name__ == "__main__":
    main()
