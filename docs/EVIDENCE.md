# Evidence map

Where to find the result behind each main claim in the report. All numbers were produced by the code in this
repository. Source exam text and images are not included (see [data/README.md](../data/README.md)).

## Model selection

Method and full discussion: [MODEL_SELECTION.md](MODEL_SELECTION.md).

| Slot | Chosen | Key evidence | Result files |
|---|---|---|---|
| Offline extraction readers | MinerU2.5 + DeepSeek-OCR | 7 candidates on an 18-page gold set; first choice reversed after full-page reads found wrong values | [ocr_extraction/results.md](../evaluation/model_selection/ocr_extraction/results.md), [scoring_summary.csv](../evaluation/model_selection/ocr_extraction/scoring_summary.csv) |
| Live photo transcription | Qwen3-VL-8B-Instruct | 0 unfaithful edits; kept a deliberate student error | [photo_transcription/results.md](../evaluation/model_selection/photo_transcription/results.md) |
| Hint language model | Phi-4-mini-instruct | 2.91 GB vs 6.05 GB for Qwen3 8B; 0/9 answer leaks | [hint_llm/results.md](../evaluation/model_selection/hint_llm/results.md) |
| Embedding | Qwen3-Embedding-0.6B | Largest separation gap, 0.4896 | [embedding/results.md](../evaluation/model_selection/embedding/results.md) |
| Safety | ShieldGemma-2B + lexicon | Held-out 19/22 alone; 9/10 self-harm recall combined, 0 false positives | [safety_classifier/results.md](../evaluation/model_selection/safety_classifier/results.md) |
| Charts | None production-ready | Best 4.8% final-answer accuracy | [diagram_charts/results.md](../evaluation/model_selection/diagram_charts/results.md) |
| Geometry | None production-ready | 0/17 and 6/17 locally; 16/17 hosted-model feasibility check | [diagram_geometry/results.md](../evaluation/model_selection/diagram_geometry/results.md), [geometry_summary.json](../evaluation/results/geometry/geometry_summary.json) |

## Grounding experiment

446 held-out questions frozen before training; 388 gradable. Paired McNemar exact tests.

| Condition | Correct / 388 |
|---|---|
| A. Base Phi-4-mini | 54 (13.9%) |
| B. Base + retrieval + few-shot | 50 (12.9%) |
| C. Base + QLoRA + retrieval + few-shot | 50 (12.9%) |

p = 0.557 (A vs B), 0.557 (A vs C), 1.000 (B vs C).
Files: [score_summary_corrected_after_rerun.json](../evaluation/results/grounding_experiment/score_summary_corrected_after_rerun.json),
adapter variants in [score_summary_c2_vs_baselines.json](../evaluation/results/grounding_experiment/score_summary_c2_vs_baselines.json) and
[score_summary_c3_vs_baselines.json](../evaluation/results/grounding_experiment/score_summary_c3_vs_baselines.json).
The first run at a 300-token answer limit, which cut off 36% to 41% of answers, is kept in
[superseded/](../evaluation/results/grounding_experiment/superseded/) for transparency.
Code: `scripts/evaluation/grounding_*.py`.

## Other evaluation

| Claim | Evidence |
|---|---|
| Hint quality scores (Tier 1 1.06, Tier 2 3.06, 1 partial leak) | [analysis_summary.json](../evaluation/results/hint_quality/analysis_summary.json), rubric in [RUBRIC.md](../scripts/evaluation/hint_quality/RUBRIC.md) |
| Tier 3 hint quality (scaffolding 2.16; 12 of 50 leaks caught by the guard) | [tier3_summary.json](../evaluation/results/hint_quality/tier3_summary.json), script `scripts/evaluation/hint_quality/generate_tier3_hints.py` |
| Gate outcomes and accuracy audit (83.9% held at Tier 1; 0 of 30 false accepts; 21 of 30 rejections were key problems) | [gate_outcomes_summary.json](../evaluation/results/gate/gate_outcomes_summary.json) |
| QLoRA adapter checks (leaks, gate solving with and without adapter) | [evaluation/results/adapter/](../evaluation/results/adapter/) |
| Corpus answer verification (951 verified) | [verification_summary.json](../evaluation/results/corpus_verification/verification_summary.json) |
| Held-out freeze and overlap check | `scripts/held_out/` and [DATA_AND_EVALUATION.md](DATA_AND_EVALUATION.md) |
| Development history and every numbered issue | [DEVELOPMENT_LOG.md](DEVELOPMENT_LOG.md) |
| Software tests (418: 399 pass, 19 skipped by design) | `backend/tests/` |
| Expert review and end-to-end dry run | [ux_review/](ux_review/) |
