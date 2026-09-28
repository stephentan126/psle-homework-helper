# Model selection

Every model in the system was chosen through a written comparison ("bake-off") against real project data,
not picked from a leaderboard. This document gives the shared method, then one section per model slot.
Raw results for each slot are in `evaluation/model_selection/<slot>/results.md`.

## 1. Shared method

1. **Build a small gold set from real material** before running any candidate. Ground truth is recorded by
   hand from the printed source. Held-out evaluation papers are never used for gold sets.
2. **Apply hard gates first.** A candidate that fails a gate is rejected, whatever its other scores.
   - *Fits the hardware*: runs within the 8GB GPU budget alongside the rest of the system.
   - *Fidelity*: answers and values read correctly, including units and sub-parts.
   - *Faithfulness*: no silent corrections and no invented content.
3. **Rank the survivors** on the slot's deciding metric (listed per slot), then on speed and memory.
4. **Check by hand.** Automated scores are confirmed by reading full outputs. This step reversed one decision
   (Section 2).
5. **If nothing clears the bar, choose the least-bad candidate and document it as a limitation** rather than
   searching indefinitely.
6. **Pin exact model versions** so results stay reproducible.

## 2. Offline extraction readers (OCR)

**Task:** read 140 scanned exam papers, including answer-key grids, fractions, units and diagrams.
**Gold set:** 18 hand-labelled pages covering two-column MCQ pages, multi-part questions, answer-key grids,
diagrams, handwritten working and pages where question numbers sit in the margin.
**Rule:** choose two readers from different model families so their disagreement is a useful confidence signal.

| Candidate | Peak VRAM | Result |
|---|---|---|
| MinerU2.5 | 2.15 GB | **Selected, primary.** No wrong values in full-page reads |
| DeepSeek-OCR | 6.39 GB | **Selected, secondary.** Best character accuracy among survivors |
| PaddleOCR-VL | 2.9 GB | Rejected: recorded "$50" where the page said "50¢" |
| Unlimited-OCR | 6.31 GB | Rejected: read 1/4 as 4/5, invented a table structure |
| Qwen3-VL (small) | 3.96 GB | Rejected: 0/7 diagram detection |
| GOT-OCR 2.0 | 2.53 GB | Rejected at gates: dropped fraction denominators, silently lost content |
| dots.mocr | - | Rejected: could not be installed on the target machine (hard `flash_attn` dependency) |

The first choice (PaddleOCR-VL with Unlimited-OCR) passed a 35-item automated check. A full-page manual read
then found wrong values in both, and the pair was reversed. Character error rate also proved misleading:
MinerU2.5 had the worst score of the finalists because of cosmetic label errors, yet no wrong numbers.

## 3. Live photo transcription

**Task:** transcribe a student's photo of one question faithfully, keeping any mistakes in their working.
**Deciding metric:** faithfulness, then diagram flagging, then speed.

| Candidate | Result |
|---|---|
| Qwen3-VL-8B-Instruct | **Selected.** 0 unfaithful edits, flagged every diagram, kept a deliberately wrong student answer |
| Qwen2.5-VL-7B-Instruct | Rejected: inserted a word into a question, missed one diagram |
| MiniCPM-V 2.6 | Not testable: access-gated repository |
| Gemini 2.5 Flash | Excluded: paid external API conflicts with running locally |

## 4. Hint language model

**Task:** solve questions for the gate and write age-appropriate hints without leaking the answer.
**Test:** 3 verified corpus questions, 3 generations each, programmatic and manual leak check.

| Candidate | Peak VRAM | Result |
|---|---|---|
| Phi-4-mini-instruct (3.8B) | 2.91 GB | **Selected.** No coherence errors, 0 leaks |
| Qwen3 4B | 2.69 GB | Rejected: garbled the easiest question ("3/4 ÷ 12") |
| Qwen3 8B | 6.05 GB | Rejected: twice the memory with no clear quality gain |

Serving moved from llama.cpp to Hugging Face `transformers` with 4-bit NF4 because llama.cpp's Windows CUDA
builds crashed on the target CPU (no AVX-512). This also lets training and serving share one stack.

## 5. Embedding model

**Task:** match a student's question to the corpus without false matches.
**Test:** 85 known-same and 42 known-different real question pairs.
**Deciding metric:** separation gap (lowest same-pair score minus highest different-pair score), because a
false match would give the gate the wrong answer key.

| Candidate | Recall@1 | Separation gap | Result |
|---|---|---|---|
| Qwen3-Embedding-0.6B | 0.859 | **0.4896** | **Selected** |
| BGE-M3 | 0.824 | 0.3916 | Kept as a faster fallback |
| harrier-oss-v1-0.6b | 0.871 | 0.3151 | Rejected: worst separation despite best recall |

The match threshold (0.90) is backed by two extra checks that block near-identical questions with different
numbers or place-value words.

## 6. Safety classifier

**Task:** block unsafe input and output, and route self-harm signals to a caring response with support
resources, without blocking ordinary maths ("this question is killing me").
**Test sets:** 22 hand-written cases (gold), 22 fresh held-out cases, 10 further fresh cases. None contain exam content.

| Stage | Result |
|---|---|
| Qwen3-Guard-Gen-0.6B | Rejected: its categories fired on a vegetable-knife maths problem and missed self-harm |
| ShieldGemma-2B, threshold tuned on gold | 22/22 on gold |
| Same threshold on held-out cases | 19/22: missed passive self-harm phrasing ("everyone would be better off without me") |
| ShieldGemma-2B OR cited self-harm lexicon | **Selected.** Self-harm recall 9/10, 0 false positives across all sets |

The lexicon is built from published screening instruments (C-SSRS, PHQ-9 item 9, and Joiner's perceived
burdensomeness construct). ShieldGemma needs 5.15 GB, so it runs on the CPU (about 10.5 s per request for
input and output checks together).

## 7. Diagram interpretation

No candidate reached a usable accuracy, so no diagram model is connected. The app says it cannot see the
figure instead of guessing.

| Bucket | Best candidate | Result |
|---|---|---|
| Charts (21-question gold set) | granite-vision-3.3-2b | 4.8% final-answer accuracy |
| Geometry (17 scored questions) | Qwen2.5-VL-7B at 600 answer tokens | 6/17 |
| Geometry | Qwen3-VL-8B-Instruct | 0/17 (answers cut off before completion) |
| Geometry, feasibility check | Large hosted model via API (offline only) | 16/17, about USD 1.34 for the set |

The feasibility check shows the questions are solvable and the gap comes from model size under 8GB. It was
never used on student data or in the app.

## 8. Blocked slots

| Slot | Status |
|---|---|
| Voice input (MERaLiON-2-3B planned) | Not run: no labelled audio evaluation set exists |
| Topic classifier (5-way comparison planned) | Not run: no labelled topic set exists yet |
