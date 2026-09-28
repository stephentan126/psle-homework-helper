# Section 4 slot 5, Embedding Model Bake-off (real run)

**Winner: `Qwen/Qwen3-Embedding-0.6B`** (Apache 2.0).

## Why this bake-off ran now

Phase 4 step 3 (confirm step + match-first wiring) needs a real embedding model to compare a
photo-submitted question against the existing `questions` corpus. Checked real current status
before assuming anything: design specification Section 4 slot 5 and
docs/DEVELOPMENT_LOG.md's own "open bake-offs" list both still showed this genuinely
undecided, so it was run for real now, not deferred or guessed at.

## Candidates

- **`Qwen/Qwen3-Embedding-0.6B`**, named in the original plan.
- **`BAAI/bge-m3`**, named in the original plan.
- **`microsoft/harrier-oss-v1-0.6b`**, found via a live model-landscape check this session
  (WebSearch + a direct HF model-card fetch confirming MIT license, 1024-dim output, no
  `trust_remote_code` requirement), a real, newer (2026-04) candidate NOT in the original plan,
  the same discovery pattern as Job B's own Qwen3-VL-8B find (Issue 183).
- `NV-Embed-v2`, `jina-embeddings-v3`, already rejected on licence (CC-BY-NC) per Section 4 slot
  5's own pre-existing text; not retested here.

## Real evaluation set (no fabricated text)

`build_gold_pairs.py` built a real evaluation set directly from the real 5,425-row corpus
(`psle.db`): **85 real known-same pairs** (genuine, substantial, non-boilerplate question texts
that occur as exact duplicates across multiple real exam papers, the same real MCQ/structured
question reused verbatim, a real, common practice-question-bank phenomenon, confirmed
directly against the DB) and **42 real known-different pairs** (real, distinct questions from
different duplicate groups).

**Limitation:** these known-same pairs are EXACT string duplicates,
not noisy near-duplicates, encoding an identical string twice trivially favours any deterministic
model and does not, by itself, test the harder, more realistic Phase 4 case (a live VLM
transcription of a photographed question, with real OCR-style textual variation, matched against
its corpus counterpart). This is why a second, smaller, noisy check was added (below)
rather than relying on this test alone.

## Primary results (`run_bakeoff.py`, 85 same-pairs + 42 diff-pairs, real retrieval pool of 127)

| Model | recall@1 | recall@3 | same_min | diff_max | **separation_gap** (min_same − max_diff) | load time | encode time (212 texts) |
|---|---|---|---|---|---|---|---|
| **Qwen3-Embedding-0.6B** | 0.859 | 1.000 | 0.99999982 | 0.5104 | **0.4896** | 79.7s | 384.2s |
| BGE-M3 | 0.824 | 1.000 | 0.99999988 | 0.6084 | 0.3916 | 141.1s | 22.0s |
| Harrier-oss-v1-0.6b | 0.871 | 1.000 | 0.99999982 | 0.6849 | 0.3151 | 113.7s | 272.6s |

All three real candidates achieve perfect recall@3 (the true match is always retrievable in the
top 3) and zero real same/diff score overlap on this test set, any of the three could support
*some* threshold here. The deciding real metric is **separation_gap**, since Phase 4's own,
previously-flagged real risk (Issue 180) is a confidently WRONG match serving the wrong
question's answer key, a wide, real margin between known-same and known-different scores is
what makes a deployed similarity threshold safe against that risk, more directly than raw
recall@1 does. Qwen3-Embedding-0.6B wins decisively on this metric.

## Supplementary real check on noisy text (`score_noisy_queries.py`)

Reused REAL VLM transcription output already generated this session (Issue 184's own
`validate_preprocessing.py` AFTER-run, Qwen3-VL-8B-Instruct reading a real photographed page) as
three noisy queries, real OCR-style variation (added question-number prefixes,
spacing differences, degree/angle-symbol differences), not fabricated text, against a real,
fixed, reproducible 31-row distractor pool (3 true matches + 28 real corpus questions sampled
deterministically, not cherry-picked).

| Model | worst-case gap (true_score − best_distractor_score, n=3) | all 3 ranked #1? |
|---|---|---|
| **Qwen3-Embedding-0.6B** | 0.2641 | Yes |
| BGE-M3 | 0.2686 | Yes |
| Harrier-oss-v1-0.6b | 0.1933 | Yes |

All three correctly rank the true match #1 for all 3 real noisy queries. Qwen3-Embedding-0.6B and
BGE-M3 are statistically indistinguishable here (0.2641 vs. 0.2686, a 0.0045 difference on an n=3
sample, not a real, decisive signal); Harrier is clearly, consistently the worst performer on
separation margin on BOTH the primary and supplementary tests, despite its narrow recall@1 lead
on the primary (weaker, exact-duplicate) test alone.

## Decision and tradeoffs

**Qwen3-Embedding-0.6B wins**, on the strength of the primary test's much larger, more
statistically robust sample (127 pairs vs. 3) showing a clearly better separation margin, with the
supplementary noisy-text check confirming it still holds up (ranks the true match #1 with a solid,
real positive gap in every one of the 3 real cases) rather than overturning that finding.

**Tradeoff:** BGE-M3 encodes ~17x faster on CPU (22.0s vs.
384.2s for the same 212 texts) and was marginally (statistically insignificantly) ahead on the
tiny noisy-text sample. This matters for two real, practical follow-up items, neither blocking
today's decision: (1) the one-time offline full-corpus embedding precompute job, naive linear
scaling suggests ~2.7 hours for Qwen3-Embedding-0.6B across all 5,425 real questions vs. ~9
minutes for BGE-M3, not yet directly measured at full scale; (2) live confirm-step per-request
query-embed latency, which sits behind Job B's own multi-minute VLM step regardless, so a few
extra seconds is real but not yet confirmed as user-facing-significant. If the full-corpus
precompute time is measured and found impractical, BGE-M3's very close, real safety margin makes
it a legitimate real fallback, flagged here as follow-up, matching this project's own
established "real follow-up, not a blocker" pattern (e.g. Issue 183's VRAM-oversubscription
flag).

Harrier-oss-v1-0.6b is rejected on real, measured evidence (clearly the worst separation margin on
both tests), not because it's new or unfamiliar, the live model-landscape check that found it is
exactly the kind of real, live check this project's own bake-off discipline (Issue 50/183)
requires, and it was given a full, real, fair evaluation here.

Full detail, reasoning, and the match-confidence-threshold decision that reuses this same real
score data: Issues 185 (this bake-off) and 186 (the threshold).
