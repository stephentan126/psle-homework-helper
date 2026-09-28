# Data and evaluation

This document covers the source data, how it was turned into a question corpus, how the evaluation set was
kept separate, and the evaluation results. The source papers themselves are not included in this repository
(see [data/README.md](../data/README.md)).

## 1. Source material

| Source | Contents | Role |
|---|---|---|
| School practice papers, 2021 to 2025 | Prelim and in-year papers from Singapore primary schools, purchased in print | Main corpus |
| PSLE yearly papers | Past national papers with answer booklets | Corpus |
| Seven 2025 SA2 school papers and the 2024 to 2025 PSLE papers | Whole papers | Held-out evaluation set, never used for training or retrieval |

Use of purchased material relies on Singapore's Copyright Act 2021 section 244 (computational data analysis
of lawfully obtained works). The material is analysed, not redistributed.

## 2. Extraction pipeline

The papers are scans with no usable text layer, so every page is read by vision models.

| Stage | What happens | Code |
|---|---|---|
| A. Render | Every PDF page is rendered to an image | `backend/app/pipeline/render_pages.py` |
| B. Read | Two OCR readers read every page separately, in batched subprocesses to contain GPU memory growth | `evaluation/model_selection/ocr_extraction/runners/`, `scripts/extraction/run_deepseek_ocr_resilient.sh` |
| C. Merge | Pages are classified, questions segmented, answer keys parsed and matched by paper section and number, and the two readers compared per question | `backend/app/pipeline/extract.py` |
| Dedup | Near-identical questions are linked to one surviving row, nothing is deleted | `scripts/extraction/run_job_a_dedup.py` |

Why two readers: when two models from different families agree on a question, the reading is trusted. When
they disagree, the disagreement is recorded with the question (`ocr_confidence` is set to low) for review. It
does not cap help by itself: the gate's check against the answer key decides the tier.

### Corpus results

| Measure | Value |
|---|---|
| Papers processed | 140 PDFs, 4,423 pages |
| Questions extracted | 5,819 |
| Questions with a worked solution to check against | 1,451 (at the time of the operator scan) |
| Duplicates linked (not deleted) | 111 in the first dedup pass |
| Reader disagreement rate | 13.3% of compared questions, inside the expected 8.2% to 18.2% band |
| Answers verified automatically against worked solutions | 951 verified, 311 need review, 4,557 cannot be checked (no worked solution) |
| `has_diagram` flag precision (90-question hand check) | 12.2% false positives, 95% CI 7.2% to 20.4%, all corrected |

The flagged rate on the first 22-file batch fell from 80.5% to 47.6% through successive parser fixes, each
validated by a full field-level comparison across all files. The full history is in
[DEVELOPMENT_LOG.md](DEVELOPMENT_LOG.md).

## 3. OCR gold set

Eighteen pages were labelled by hand to compare OCR readers. Only metadata is listed here, because the
labelled page text is copyrighted exam content.

| Page | Source type | What it tests | Diagram |
|---|---|---|---|
| GS001 | School paper | MCQ page, option-value resolution | Yes |
| GS002 | School paper | Open-ended questions with a table | Yes |
| GS003 | School paper | Bordered answer-key grid | No |
| GS004 | PSLE yearly | Open-ended questions with sub-parts | Yes |
| GS005 | PSLE yearly | Separate answer booklet grid | No |
| GS006 | Topical workbook | Answer-key grid | No |
| GS007 | School paper | Diagram question, answer key embedded in the paper | Yes |
| GS008 | School paper | Question number printed in the margin | No |
| GS009 | Handwritten | Handwritten working, unit method | No |
| GS010 | Handwritten | Handwritten working, multi-part geometry | Yes |
| GS011 | Handwritten | Handwritten working, money problem | No |
| GS012 | Handwritten | Handwritten working, unit method | No |
| GS013 | PSLE yearly | Narrative worked-solution booklet | No |
| GS014 | PSLE yearly | Narrative worked-solution booklet | No |
| GS015 | PSLE yearly | Narrative worked-solution booklet | No |
| GS016 | School paper | Pie chart | Yes |
| GS017 | School paper | Angle diagram | Yes |
| GS018 | School paper | Crossed-lines angle diagram | Yes |

Checking the gold set against the printed keys found two labelling errors in the gold set itself (an answer
copied from a different question, and an intermediate step recorded as the final answer). Both were
corrected before any model was scored.

## 4. Held-out evaluation set

| Step | Detail |
|---|---|
| Selection | Whole papers, not individual questions, so a school's question style cannot leak between training and testing. Chosen for full-syllabus coverage and a mix of school types. |
| Extraction | Run separately into its own database file |
| Freeze | Made read-only before any training (`scripts/held_out/freeze_held_out_stage_c.py`) |
| Size | 446 questions (352 school, 94 PSLE yearly); 388 have a gradable single answer |
| Overlap check | Every held-out question compared with the whole corpus by embedding similarity; 116 candidate pairs reviewed by hand, including page images for diagram questions |
| Findings | About 108 false alarms. 1 real overlap: two schools used the same commercial item bank. Training rows involved were excluded. One held-out question appearing identically in three training papers was excluded from evaluation through a tracked exclusion list, leaving the frozen file unchanged. |

## 5. Evaluation results

### 5.1 Knowledge grounding (held-out, 388 gradable questions)

| Condition | Correct | Diagram questions | Text-only questions |
|---|---|---|---|
| A. Base Phi-4-mini | 54 (13.9%) | 2.8% | 23.6% |
| B. Base + retrieval + few-shot examples | 50 (12.9%) | 2.8% | 21.6% |
| C. Base + QLoRA adapter + retrieval + few-shot | 50 (12.9%) | 1.7% | 22.6% |

Paired McNemar exact tests: A vs B p = 0.557, A vs C p = 0.557, B vs C p = 1.000. No condition is
statistically better. The first run used a 300-token answer limit and cut off 36% to 41% of answers; it was
re-run at 600 tokens, which reversed the apparent ranking. Two further adapter variants (higher rank, and
loss on the answer only) were also not significantly different (closest p = 0.077). Because the adapter also
doubled the rate of answers with no final line in the gate (32/80 vs 17/80, p = 0.0007), the app serves the
base model by default.

Files: `evaluation/results/grounding_experiment/`.

### 5.2 Hint quality (50 held-out questions, blind judge)

Hints were scored 1 to 5 by a separate large model acting as a blind judge, following a written rubric
(`scripts/evaluation/hint_quality/RUBRIC.md`). The judge did not know which version produced each hint.

| Tier | Scaffolding | Age-appropriate | Answer leaks |
|---|---|---|---|
| Tier 1 | 1.06 | 3.30 | 0 |
| Tier 2 | 3.06 | 3.94 | 1 partial leak |

Tier 1 is a restatement by design, so its low scaffolding score is expected. Three alternative Tier 1 prompts
were tested; none met the improvement bar set in advance. File: `evaluation/results/hint_quality/`.

### 5.3 Component tests

| Component | Result |
|---|---|
| Safety classifier | Self-harm recall 9/10, 0 false positives on 36 borderline and safe cases |
| Photo input | Printed MCQ, handwritten working with a deliberate error, and a diagram question all handled end to end; the student's wrong answer was kept |
| Gate | Correct answers pass, a deliberately wrong answer is capped at Tier 1, and a right-answer-wrong-method case is caught |
| Match threshold | Near-duplicate questions with different numbers are blocked (for example 0.946 similarity, blocked) |
| Software | 418 automated backend tests (399 pass, 19 skipped by design, 72% line coverage). 322 need no GPU or models and run in GitHub Actions on every push; 96 load the models. Frontend type check, lint and production build also run in CI |

### 5.4 Performance

| Operation | Time |
|---|---|
| Cached gate result for a corpus question | about 0.01 s |
| Live gate, two self-consistency passes | about 15 to 40 s |
| Typed question to the confirm screen | 14 s |
| First photo after a server start, to the confirm screen (slowest request measured) | 149 s |
| Photo confirm to first hint, including the planned restart | 34 to 45 s |
| Typed question, confirm to first hint (cached gate result) | 8 s |
| Next hint | 3 to 24 s |
| Safety checks (input and output, CPU) | about 10.5 s |

## 6. What can be reproduced from this repository

| Item | Reproducible here | Needs the withheld source papers |
|---|---|---|
| Running the app on the demo questions | Yes | |
| Running the non-GPU test suite | Yes | |
| Frontend build and lint | Yes | |
| Model selection results | Code and results included | Re-running needs the gold-set pages |
| Held-out evaluation | Code and results included | Re-running needs the held-out papers |
| Corpus extraction | Code included | Needs the papers |
