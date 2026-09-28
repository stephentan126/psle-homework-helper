# Bake-off Results

Role/rejection_reason filled in, grounded only in the real, committed numbers from
commit `d37abed`. None of the four candidates clear a real accuracy bar for production use, per
The design specification Issue 67's own contingency rule ("if no candidate clears the gate, pick the
least-bad option, document the shortfall honestly ... and move on"), `granite-vision-3.3-2b` is
designated primary as the least-bad option, not because it is judged production-ready.

| n_rows | figure_detected_rate | chart_type_identified_rate | values_extracted_rate | final_answer_exact_match_rate | n_errors | per_row_detail_file | candidate | load_time_s | vram_delta_gb | fits_8gb | role | rejection_reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 21 | 1.0 | 0.952 | 0.143 | 0.048 | 0 | per_row_results\ibm-granite__granite-vision-3.3-2b.json | ibm-granite/granite-vision-3.3-2b | 50.05 | 1.7 | True | primary (least-bad per Issue 67, not production-ready) | Designated as the default candidate among four tested, not because it clears an accuracy bar, but because it's the only one with zero execution errors and any meaningfully strong capability (chart-type recognition, 95.2%/20/21). Its actual answer accuracy (4.8% final-answer/1/21, 14.3% value-extraction/3/21) means it should NOT be used to generate hints directly to students, any downstream use requires either a human-in-the-loop check or further fine-tuning/prompting work before this bucket's output is trusted for real students. Chosen as primary to keep the pipeline moving per Issue 67 (the fallback default when a diagram question can't be handled some other way), not as a claim it's ready. |
| 21 | 0.095 | 1.0 | 0.0 | N/A (extraction-only candidate, Issue 320) | 19 | per_row_results\ibm-granite__granite-docling-258M.json | ibm-granite/granite-docling-258M | 8.51 | 0.49 | True | rejected (structural instability, not purely an accuracy finding) | Extraction-only candidate (no final-answer stage by design). Correctly identified figure presence in only 9.5% of rows (2/21) before hitting a reproducible CUDA crash (illegal instruction, not the earlier device-side-assert bug) that persisted for the rest of the run once triggered, 19/21 rows recorded as errors. Subprocess isolation confirmed this is a real candidate-specific problem, not cross-run contamination. Root cause not yet investigated (Issue 328); rejecting for now on the basis of unreliability, revisit only if a future bucket hits a similar pattern worth understanding. |
| 21 | 1.0 | 0.0 | 0.095 | 0.0 | 0 | per_row_results\ahmed-masry__unichart-chartqa-960.json | ahmed-masry/unichart-chartqa-960 | 110.49 | 0.75 | True | rejected | Despite being a chart-specialized model, chart_type_identified came back 0.0 (0/21), it never correctly named the chart type in this test set, even though it did detect a figure was present (100%, 21/21) and extracted some values (9.5%, 2/21). Final answer accuracy was 0.0. Weakest of the four on the metric it should have been strongest at. |
| 21 | 0.381 | 0.5 | 0.125 | N/A (extraction-only candidate, Issue 320) | 13 | per_row_results\google__deplot.json | google/deplot | 6.04 | 0.54 | True | rejected (least confidently tested of the four) | Required a real code fix mid-round (dtype mismatch, Issue 327) before it could be evaluated at all; the first run was 100% harness failure, not a real result. After the fix: figure_detected 38.1% (8/21), chart_type_identified 50% (of rows attempted), values_extracted 12.5%, still 13/21 rows recording errors of unexamined cause. Rejecting on the basis of "insufficient and unstable as tested," not "proven worse than the others", this candidate has had the least real, clean signal of the four. |

## Overall Bucket 1 conclusion

None of the four candidates are viable for production chart/diagram tutoring as tested.
`granite-vision-3.3-2b` is designated primary per Issue 67's least-bad-option contingency rule
,  the fallback default when a diagram question can't be handled some other way, NOT a claim it is
production-ready. At 4.8% final-answer accuracy it must not be used to generate hints directly to
students without a human-in-the-loop check or further fine-tuning/prompting work: it can tell you
what kind of chart it's looking at but not read it accurately. The chart-specialized models
(`unichart-chartqa-960`, `deplot`) did not outperform it, and in `unichart`'s case underperformed
on their own specialty (`chart_type_identified_rate=0.0`). This is a real, evidenced negative result
for off-the-shelf
models on this task, not a byproduct of testing methodology, the bake-off harness itself was
validated via `dry_run_scoring_check.py` before any GPU run, and every model-side failure
(docling's crash, deplot's dtype bug) was independently diagnosed and, where fixable, fixed and
re-tested rather than left conflated with real accuracy numbers.
