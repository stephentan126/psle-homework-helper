"""CPU-only analysis (no GPU, nothing regenerated) of the Qwen2.5-VL-7B 1200-token-answer run against the 600-token run.
Writes per_row_results_answer1200/analysis_vs_600.json and prints a summary.
"""
import json
import statistics
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import run_bakeoff as rb  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

D600, D1200 = _HERE / "per_row_results_answer600", _HERE / "per_row_results_answer1200"
L = lambda p: json.load(open(p, encoding="utf-8"))  # noqa: E731


def main():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    gold_rows = [r for r in L(rb.GOLD_SET_PATH)["rows"] if r["status"] != "excluded"]
    raw600, raw1200 = L(D600 / "raw_answers_by_question.json"), L(D1200 / "raw_answers_by_question.json")
    m1200 = L(D1200 / "Qwen__Qwen2.5-VL-7B-Instruct__metrics.json")
    ext_s = [r.get("extraction_seconds") for r in raw1200.values()]

    rows, scored = [], []
    for g in gold_rows:
        rid = str(g["id"])
        gold_ans = g.get("corrected_answer_value") or g["stored_answer_value"]
        if rb._is_multi_part_gold(gold_ans):
            continue
        q = g["question_text"]
        t6, t12 = raw600[q]["raw"], raw1200[q]["raw"]
        c6, c12 = rb.extract_claimed_answer(t6), rb.extract_claimed_answer(t12)
        ok6 = bool(rb._score_final_answer(g, c6 if c6 is not None else t6))
        ok12 = bool(rb._score_final_answer(g, c12 if c12 is not None else t12))
        n6, n12 = len(tok(t6)["input_ids"]), len(tok(t12)["input_ids"])
        row = {
            "id": rid, "tokens_600": n6, "tokens_1200": n12, "hit_1200_cap": n12 >= 1198,
            "answer_seconds_600": raw600[q]["seconds"], "answer_seconds_1200": raw1200[q]["seconds"],
            "extraction_seconds_1200": raw1200[q].get("extraction_seconds"),
            "text_600_is_prefix_of_1200": t12.startswith(t6), "text_identical": t6 == t12,
            "parsed_line_600": c6 is not None, "claimed_600": c6, "correct_600_harness": ok6,
            "parsed_line_1200": c12 is not None, "claimed_1200": c12, "correct_1200_harness": ok12,
            "correct_1200_parsed_line": ok12 and c12 is not None,
        }
        rows.append(row)
        scored.append(row)

    unresolved_600 = [r for r in scored if not r["parsed_line_600"]]
    ans_s_1200 = [r["answer_seconds_1200"] for r in scored]
    ans_s_600 = [r["answer_seconds_600"] for r in scored]
    summary = {
        "harness_metrics_1200": {k: m1200[k] for k in (
            "n_rows", "n_multi_part_rows", "figure_detected_rate", "geometry_type_identified_rate",
            "values_extracted_rate", "final_answer_exact_match_rate", "n_errors", "load_time_s", "vram_delta_gb",
            "fits_8gb", "peak_vram_gb", "evaluate_wall_s")},
        "scored_rows": len(scored),
        "parsed_line_correct_1200": sum(r["correct_1200_parsed_line"] for r in scored),
        "parsed_line_correct_1200_ids": [r["id"] for r in scored if r["correct_1200_parsed_line"]],
        "harness_style_correct_1200": sum(r["correct_1200_harness"] for r in scored),
        "harness_style_correct_1200_ids": [r["id"] for r in scored if r["correct_1200_harness"]],
        "parsed_line_correct_600": sum(r["parsed_line_600"] and r["correct_600_harness"] for r in scored),
        "harness_style_correct_600": sum(r["correct_600_harness"] for r in scored),
        "rows_with_parsed_line_600": sum(r["parsed_line_600"] for r in scored),
        "rows_with_parsed_line_1200": sum(r["parsed_line_1200"] for r in scored),
        "previously_unresolved_rows_at_600": [r["id"] for r in unresolved_600],
        "of_those_now_reach_final_line": [r["id"] for r in unresolved_600 if r["parsed_line_1200"]],
        "of_those_now_reach_final_line_and_match_gold": [r["id"] for r in unresolved_600
                                                         if r["parsed_line_1200"] and r["correct_1200_harness"]],
        "of_those_still_no_final_line": [r["id"] for r in unresolved_600 if not r["parsed_line_1200"]],
        "rows_hitting_1200_cap": [r["id"] for r in scored if r["hit_1200_cap"]],
        "rows_whose_text_differs_from_600_run": [r["id"] for r in scored if not r["text_identical"]],
        "rows_where_600_text_is_not_a_prefix_of_1200": [r["id"] for r in scored if not r["text_600_is_prefix_of_1200"]],
        "rows_whose_parsed_answer_or_score_changed": [
            r["id"] for r in scored if (r["claimed_600"] != r["claimed_1200"]
                                        or r["correct_600_harness"] != r["correct_1200_harness"])],
        "timing_1200": {
            "answer_seconds_total_scored_rows": round(sum(ans_s_1200), 1),
            "answer_seconds_mean": round(statistics.mean(ans_s_1200), 2),
            "answer_seconds_median": round(statistics.median(ans_s_1200), 2),
            "answer_seconds_min": min(ans_s_1200), "answer_seconds_max": max(ans_s_1200),
            "extraction_seconds_total_all_22_rows": round(sum(ext_s), 1),
            "extraction_seconds_mean": round(statistics.mean(ext_s), 2),
            "evaluate_wall_s": m1200["evaluate_wall_s"],
        },
        "timing_600": {"answer_seconds_total_scored_rows": round(sum(ans_s_600), 1),
                       "answer_seconds_mean": round(statistics.mean(ans_s_600), 2), "evaluate_wall_s": 514.5},
    }
    out = {"summary": summary, "rows": rows}
    (D1200 / "analysis_vs_600.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nid | tok600 -> tok1200 | ans_s 600 -> 1200 | parsed 600 -> 1200 | claimed 1200 | correct(harness) 600 -> 1200")
    for r in rows:
        print(r["id"], "|", r["tokens_600"], "->", r["tokens_1200"], "|", r["answer_seconds_600"], "->",
              r["answer_seconds_1200"], "|", r["parsed_line_600"], "->", r["parsed_line_1200"], "|",
              repr(r["claimed_1200"]), "|", r["correct_600_harness"], "->", r["correct_1200_harness"])


if __name__ == "__main__":
    main()
