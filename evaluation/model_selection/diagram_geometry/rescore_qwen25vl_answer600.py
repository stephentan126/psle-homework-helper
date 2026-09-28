"""CPU-only re-score (no GPU, no model loaded) of the ALREADY-GENERATED Qwen2.5-VL-7B 600-token answers
(per_row_results_answer600/raw_answers_by_question.json) after the Issue 344 fix to gate.strip_units() (litre symbol).

Nothing is regenerated. Writes per_row_results_answer600/rescored_after_landmine344.json and leaves the original
evaluate_candidate() outputs in that folder untouched, so the before/after difference stays visible.

Two counts are reported, on purpose:
  * parsed_line_correct: rows where the model wrote a parseable `FINAL ANSWER:` line AND it matches gold. This is the
    defensible headline (Issue 343).
  * harness_style_correct: how evaluate_candidate() itself scores: when no FINAL ANSWER line is found it falls back to
    scanning every number in the raw text, which can credit a cut-off derivation (id=1348, Issue 343).
"""
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import run_bakeoff as rb  # noqa: E402

D = _HERE / "per_row_results_answer600"


def main():
    gold_rows = [r for r in json.load(open(rb.GOLD_SET_PATH, encoding="utf-8"))["rows"] if r["status"] != "excluded"]
    raw = json.load(open(D / "raw_answers_by_question.json", encoding="utf-8"))
    original = {str(r["id"]): r for r in json.load(open(D / "Qwen__Qwen2.5-VL-7B-Instruct.json", encoding="utf-8"))}

    rows, parsed_correct, harness_correct, n_scored = [], [], [], 0
    for g in gold_rows:
        rid = str(g["id"])
        gold_ans = g.get("corrected_answer_value") or g["stored_answer_value"]
        if rb._is_multi_part_gold(gold_ans):
            rows.append({"id": rid, "status": "N/A (multi-part gold answer, Issue 332)"})
            continue
        n_scored += 1
        text = raw[g["question_text"]]["raw"]
        claimed = rb.extract_claimed_answer(text)
        new_score = bool(rb._score_final_answer(g, claimed if claimed is not None else text))
        old_score = original[rid]["final_answer_exact_match"]
        row = {"id": rid, "parsed_final_answer_line": claimed is not None, "claimed_answer": claimed,
               "score_before_landmine344": old_score, "score_after_landmine344": new_score,
               "changed": old_score != new_score}
        rows.append(row)
        if new_score:
            harness_correct.append(rid)
            if claimed is not None:
                parsed_correct.append(rid)

    summary = {
        "n_scored_rows": n_scored,
        "parsed_line_correct": len(parsed_correct), "parsed_line_correct_ids": parsed_correct,
        "harness_style_correct": len(harness_correct), "harness_style_correct_ids": harness_correct,
        "harness_style_rate": round(len(harness_correct) / n_scored, 3),
        "parsed_line_rate": round(len(parsed_correct) / n_scored, 3),
        "rows_whose_score_changed_vs_original_run": [r["id"] for r in rows if r.get("changed")],
        "rows_with_parsed_final_answer_line": sum(1 for r in rows if r.get("parsed_final_answer_line")),
    }
    out = {"note": "CPU-only re-score of saved 600-token answers after Issue 344; nothing regenerated.",
           "summary": summary, "rows": rows}
    (D / "rescored_after_landmine344.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    for r in rows:
        if r.get("changed"):
            print("CHANGED:", r)


if __name__ == "__main__":
    main()
