# Phase 3 Step 1, LLM Bake-off Results (Section 4 slot 6)

Matches the model-rejection protocol requirement (Issue 50): three candidates, one comparison
table, one real rejection reason per candidate not picked.

## Candidates

| Candidate | HF repo id | Load time (4 runs, warm) | Peak VRAM | Gen. time, 64 tok | Answer leaks | Coherence (3 real Qs) |
|---|---|---|---|---|---|---|
| Qwen3 4B | `Qwen/Qwen3-4B-Instruct-2507` | 5.11–5.95s | 2.691 GB | 3.64–4.20s | 0/9 | 2/3 (1 real defect, Q4069) |
| **Phi-4-mini 3.8B (WINNER)** | `microsoft/Phi-4-mini-instruct` | 5.26–5.34s (+1 cold-cache 12.20s) | 2.913 GB | 2.67–3.84s | 0/9 | 3/3 |
| Qwen3 8B | `Qwen/Qwen3-8B` (non-thinking) | 8.17–8.56s (+1 cold-cache 19.82s) | 6.045 GB | 3.66–4.02s | 0/9 | 3/3 (1 precision trade-off, Q4341) |

Load-time/VRAM numbers from Issues 163/164 (`load_test.py`, `test_candidate_swap.py`).
Hint-quality numbers from this bake-off's own run
(`hint_quality_comparison.py`, raw output docs/MODEL_SELECTION.md (section 4)).

## Real test questions (verified corpus, `psle.db`, `extraction_flag IS NULL`)

| ID | Source | Type | Real answer |
|---|---|---|---|
| 4069 | `P6_Maths_2024_WA2_nanhua.pdf` Q19 | sum-type | `1/16` |
| 3078 | `P6_Maths_2023_WA2_Rosyth.pdf` Q10 | reasoning (ratio, before/after) | `48` |
| 4341 | `P6_Maths_2025_SA2_aitong.pdf` Q30 | reasoning (fraction of remainder) | `63` |

Picked by structural filter only (self-contained text, no figure/diagram dependency, real
`answer_value` present, unflagged), not cherry-picked for a particular outcome.

## Prompt (identical across all candidates, quotes docs/ARCHITECTURE.md's own Tier-1 definition)

> You are a patient Socratic maths tutor helping a Singapore Primary 6 student (age 11-12) with a
> PSLE Mathematics question. You must give a TIER 1 hint ONLY. A Tier 1 hint restates the
> question in your own simple words, to check the student understands what is being asked, and
> then STOPS. Rules, all mandatory: do not solve the question; do not show any working,
> calculation, or steps; do not reveal the final numeric answer, even partially or as a range; do
> not ask a leading question about method. Just restate what the question is asking, in plain
> words a Primary 6 student would understand, then stop.

## Real outputs, side by side

### Q4069 (sum-type), "Find the value of 3/4 ÷ 12. Leave your answer in its simplest form." (real answer: 1/16)

| Candidate | Output |
|---|---|
| Qwen3 4B | "What is one-fourth of three divided by 12?" |
| Phi-4-mini | "You are asked to divide three-fourths by twelve and then express the result in its simplest form." |
| Qwen3 8B | "What is the result of dividing three-quarters by twelve?" |

**Real defect:** Qwen3 4B's restatement ("one-fourth of three divided by 12") is a confusing
misreading of "3/4 ÷ 12", not wrong-value, but a genuinely confusing paraphrase of the simplest
question in the sample.

### Q3078 (reasoning, ratio before/after), ratio 5:9 → both +8 → 3:5, find Clark's sweets now (real answer: 48)

| Candidate | Output |
|---|---|
| Qwen3 4B | "Clark and Daniel started with sweets in the ratio 5:9. After each bought 8 more sweets, the ratio became 3:5. We need to find how many sweets Clark has now." |
| Phi-4-mini | "A student needs to find out how many sweets Clark has now, given the initial ratio of sweets between Clark and Daniel, and the change in the ratio after they both buy 8 more sweets." |
| Qwen3 8B | "Clark and Daniel have some sweets. The ratio of Clark's sweets to Daniel's sweets is 5:9. Then, each of them buys 8 more sweets, and the new ratio becomes 3:5. Find how many sweets Clark has now." |

All three accurate. Phi-4-mini notably more third-person/distanced ("A student needs to find
out...") than the other two's direct address.

### Q4341 (reasoning, fraction of remainder), sold 28 + 5/8 of remainder, 1/4 left unsold (real answer: 63)

| Candidate | Output |
|---|---|
| Qwen3 4B | "The shop started with some bags, sold 28 in the morning, then sold five-eighths of the leftover bags in the afternoon, and at the end, only one-quarter of the original bags were not sold. We need to find out how many bags were sold in total." |
| Phi-4-mini | "A shop had some bags for sale. They sold 28 bags in the morning and then sold 5/8 of the remaining bags in the afternoon. After all the sales, 1/4 of the bags were left unsold. How many bags were sold altogether?" |
| Qwen3 8B | "The shop sold some bags in the morning and then sold most of the remaining bags in the afternoon, leaving only a quarter of all the bags unsold. How many bags were sold in total?" |

Qwen3 4B's "of the **original** bags" genuinely resolves a real ambiguity in the source text
correctly (verified against the real answer: the 1/4 refers to the original total, not the
afternoon remainder). Phi-4-mini's restatement is close to verbatim copy of the source text, weak
paraphrase value. Qwen3 8B's "most" drops the exact "5/8" figure.

## Verdict

**Winner: Phi-4-mini 3.8B (`microsoft/Phi-4-mini-instruct`).** Zero coherence failures across all
3 real test questions, the only candidate with a clean fidelity sheet, which is the deciding
factor once leak-avoidance is satisfied (all three tied at 0/9 leaks). Its one real weakness
(near-verbatim restatement on one question) is a milder failure mode than Qwen3 4B's actively
confusing misreading of the simplest question in the sample. VRAM/load-time cost is in the same
modest class as Qwen3 4B, not Qwen3 8B's heavier footprint, which buys no clear quality advantage
here to justify the doubled VRAM and thinner headroom in this project's 8GB budget.

**Qwen3 4B, REJECTED.** Real, concrete coherence defect on Q4069 (the simplest question in the
sample) is disqualifying on its own: Tier 1's entire job is comprehension-checking, and getting
confused on the easiest question type undermines reliability across what will likely be a large
share of real PSLE questions (direct computation, "sum-type" per Section 5's own routing). Its
genuinely better paraphrasing when correct (Q4341's ambiguity resolution) does not offset this.

**Qwen3 8B, REJECTED.** No coherence failures, and better own-words paraphrasing than Phi-4-mini
on average, but costs more than double the VRAM (6.045 GB vs. 2.913 GB) for no clear hint-quality
advantage over Phi-4-mini in this sample, a real, material cost in this project's tight 8GB
budget (Section 2's "one big model + ~1-1.5GB small helpers resident" design), not offset by
better quality here.

**Honest limit, stated plainly, not smoothed over:** this verdict rests on 3 real questions, not
a large evaluation set. Reasonable people could weigh Qwen3 4B's better paraphrasing-when-correct,
or Qwen3 8B's consistent accuracy, differently. Picked decisively anyway per this project's own
contingency discipline (Issue 67: pick the least-bad option, document the shortfall, move on)
rather than leaving Section 4 slot 6 open indefinitely on a small sample.
