# Job A OCR Bake-off, Results

Phase 1, step 2 (normalize + score). Scored with `normalize_and_score.py` (automated: CER/WER,
answer-key fidelity, diagram detection, structure) plus a manual faithfulness check, against the
18-page gold set in the gold-set table in docs/DATA_AND_EVALUATION.md (gitignored, Issue 75). Decision rule and family
groupings per docs/MODEL_SELECTION.md Sections 6 and 8.

7 candidates attempted: 6 ran to completion, dots.mocr was rejected before any inference run
(Issue 77, hard `flash_attn` import, unbuildable on this Windows/no-system-CUDA-toolkit
machine). Mistral OCR was never run, paid, offline-only, reference-quality-only per protocol
Section 2, out of scope for this bake-off's live-inference comparison.

**Read this whole document before trusting the automated PASS column.** The automated 35-item
checklist scored 5 of 6 candidates identically clean. A subsequent full-page manual read of the
3 worked-solution pages (GS013-15), run because that clean sweep looked too uniform to trust
outright, found real, previously-uncaught defects in every one of the 4 candidates checked at
that depth, including the single worst defect of the whole bake-off, on a candidate the automated
check had called perfect. See "Full-page read findings" below. This is now logged as Issue 79.

## Methodology notes (read before the numbers below)

- **CER/WER is not a gate** (protocol Section 5) and is reported here with a real caveat: the
  only available gold reference is `question_text_clean`, a hand-written condensed paraphrase of
  each page's question content (e.g. "Q1: Round off 314678..."), not a verbatim transcript. Every
  model's actual OCR output includes far more literal page content (instructions, all MCQ option
  text, footer codes, "Ans:" labels) than that paraphrase, so CER/WER is inflated well above 1.0
  (100% char "error") for every candidate. Treat these numbers as a same-reference relative
  ranking signal only, never as an absolute transcription-accuracy figure. **Sharper caveat as of
  the full-page read (see below): CER/WER can also be actively misleading between candidates, not
  just inflated for all of them equally.** MinerU2.5 has the worst CER/WER of the four deep-read
  candidates, driven entirely by cosmetic garbling of label/header prose that never touches a
  numeric value. PaddleOCR-VL has better CER/WER while containing an actual wrong number
  (GS014 Q8, $50 recorded where the source says 50¢). CER/WER ranked the cleaner candidate worse.
- GS013/GS014/GS015's `question_text_clean` is itself a literal `"N/A - this file is the
  worked-solutions document only..."` placeholder (the matching question paper wasn't sourced
  this round), not a real transcript at all. Scoring CER/WER against that placeholder would
  produce meaningless numbers identical in kind for every model; `normalize_and_score.py` excludes
  those 3 pages from the CER/WER aggregate (15/18 pages scored, not 18/18). These same 3 pages are
  exactly where the full-page manual read below was run, since they're excluded from the
  automated checklist entirely, not just the CER/WER aggregate.
- **Answer-key fidelity ground truth, scope correction.** The instruction driving this step named
  8 pages (GS003, GS005, GS006, GS007, GS008, GS016, GS017, GS018). Only GS003/GS005/GS006 are
  actual answer-key pages, the printed final answer is literally on the rendered page. GS007,
  GS008, GS016, GS017, GS018 are QUESTION pages (Issue 71 Topology B: the matching key lives on
  a different, unrendered page of the same source PDF), their `final_answer_values` (e.g.
  "146deg" for GS007) never appear in any model's OCR of the tested page, by construction. Checked
  instead: whether the specific INPUT numbers printed on those 5 pages (e.g. GS007 Q2's "68deg"
  for angleBCD, which the real key's own working starts from) are transcribed correctly and tied
  to the right question, not merged with a mark bracket. Two items (GS016 Q20's pie-chart
  fractions, GS018 Q20's two angle labels) are inside the diagram image itself, not printed page
  text, on every candidate, recorded as not automatable from OCR text, not scored either way.

## Gate results

| Model | Fits 8GB | Answer-key fidelity | Faithfulness | Verdict |
|---|---|---|---|---|
| GOT-OCR 2.0 | PASS (2.53 GB) | **FAIL**, GS005 (28a/28b/29 column-flattening), GS008 (all 3 fraction values dropped) | **FAIL**, GS014 near-total silent content loss (84 chars recovered, no error flag) | **REJECTED** |
| PaddleOCR-VL | PASS (2.90 GB) | PASS, 35/35 | PASS (checklist scope; see full-page read below for a real defect outside checklist scope) | survivor |
| MinerU2.5 | PASS (2.15 GB) | PASS, 35/35 | PASS (checklist scope; full-page read found no numeric defects, cosmetic garbling only) | survivor |
| Qwen3-VL small | PASS (3.96 GB) | PASS, 35/35 | PASS | survivor |
| DeepSeek-OCR | PASS (6.39 GB) | PASS, 35/35 | PASS (checklist scope; full-page read found one structural defect, no numeric defects) | survivor |
| Unlimited-OCR | PASS (6.31 GB) | PASS, 35/35 | PASS (checklist scope; see full-page read below for a real defect outside checklist scope) | survivor |
| dots.mocr | n/a, never ran | n/a | n/a | **REJECTED** (Issue 77, setup difficulty) |

**GOT-OCR 2.0 detail.** This model exposes two prompt modes (`ocr`/plain, `format`/markdown-ish).
Neither is safe alone: `format` mode transcribed almost nothing on GS003's 15-row MCQ grid (only
recovered the title "ANSWER KEY"), so this script uses `ocr` mode as the primary schema text
instead, but `ocr` mode then loses almost the entire page on GS014 (84 characters recovered, vs
1931 in `format` mode on that same page), with `error: None` in both failure cases, i.e. no signal
a real pipeline could catch either failure on. A production system running one fixed mode has no
way to know in advance which pages will silently lose most of their content. That, plus the
GS005/GS008 fidelity misses (both genuine: GS005 is a real got_ocr2 column-flattening defect near
identical in kind to the GS002 case below; GS008's fraction denominators are dropped outright, not
merged/misplaced), is why GOT-OCR 2.0 fails both non-VRAM gates and is rejected outright. Logged
as Issue 78: no single prompt mode of this model is safe for all pages.

## Full-page read findings (beyond the automated 35-item checklist)

The automated checklist only covers 35 pre-named items across 8 pages. It never looks at
GS013-15 (the worked-solutions narrative pages) at all, and even on the 8 covered pages it only
checks specific named values, not full-page content. After the automated pass scored 5 of 6
candidates identically clean (35/35, no faithfulness issues), that uniformity looked too clean to
trust without a direct check. The full output of PaddleOCR-VL, Unlimited-OCR, DeepSeek-OCR and
MinerU2.5 on GS013, GS014 and GS015 was read line by line against the real source, not just
named checkpoints.

**Result: every one of the 4 candidates had a real, previously-uncaught defect. None was clean.**

| Model | Fabricated table schema (GS013 Q1) | Wrong numeric value | Dropped result/heading | Cosmetic word garbling |
|---|---|---|---|---|
| MinerU2.5 | No, real structure kept | **None found** | 1 minor (GS015 Q10, non-checkpoint step omitted, final answer intact) | Yes (headers: "enriche", "honduras", "thioucandis", "rectrongle", "choirs") |
| DeepSeek-OCR | Yes | **None found** | None | 1 (text duplication) |
| Unlimited-OCR | Yes | **Yes, GS015 Q7: 4/5 recorded, source says 1/4** | None |, |
| PaddleOCR-VL | No, real structure kept | **Yes, GS014 Q8: $50 recorded, source says 50¢ (worst magnitude of any defect found)** | 2 (GS014 Q4 both angle results; GS015 Q7 heading) | Yes (GS013 headers) |

All four candidates also share one low-severity, non-differentiating defect: the cent symbol (¢)
is misread as a different single glyph (φ, ξ, or similar) by every one of them, with the numeric
value itself unaffected. Recommend a deterministic glyph-normalization step (¢, °, cm², similar
OCR-prone symbols) in Phase 1 step 3's extraction pipeline, since this is now confirmed systematic
across models, not candidate-specific, and is cheap to fix in post-processing rather than by
picking a candidate that avoids it.

**Read plainly:** MinerU2.5 and DeepSeek-OCR never produced a wrong numeric value across the 3
pages read this deeply. PaddleOCR-VL and Unlimited-OCR each did, once. MinerU2.5's worse-looking
automated CER/WER is fully explained by cosmetic label garbling that never touches a number, not
by any real accuracy problem, direct evidence that CER/WER alone is not a reliable ranking signal
here, sharper than the general caveat noted above.

**Final direct verification, both source and output read firsthand, not taken on
report:** the real source PDFs/pages for GS013-15 were obtained and checked line by line against
the claims above, confirmed genuine, including GS015 Q9's "= 162 cm²" typo, which is a real
oddity in the original printed document itself, not introduced anywhere downstream. Separately,
the raw `extracted_text` for the two decisive defects was read directly, not summarized:
- **PaddleOCR-VL, GS014 Q8:** the raw output contains `= $0.50` immediately followed by a second,
  duplicate line, `= $50`. More precisely described as a duplicated erroneous line following the
  correct value, not a clean value substitution, the practical risk is the same either way, an
  extraction step that takes the last line as the answer gets the wrong one.
- **Unlimited-OCR, GS015 Q7:** the raw output contains `= 4/5` followed two lines later by
  `= 5/20`. 5/20 reduces to 1/4, not 4/5, the model's own output is internally self-contradictory,
  not just wrong against the source.
Neither MinerU2.5 nor DeepSeek-OCR's raw output contains a wrong numeric value anywhere in this
comparison.

**Honest limit on this evidence:** this rests on a full-page read of 3 of 18 gold pages, plus the
original 35-item automated coverage on 8 others. It is the deepest evidence available from this
gold set, not exhaustive proof. Treated here as a strong signal to finalize on, not an absolute
guarantee that hasn't been re-tested. Logged as **Issue 79: a narrow automated checklist can
score a candidate clean while missing real defects. This happened on two separate full-page reads
in this bake-off, not once. Generalizes beyond this bake-off, Phase 1 step 3's human verification
pass needs to sample real full pages, not rely on spot-checks or a fixed checklist, or it will
inherit this same blind spot.**

## Comparison table (protocol Section 7 format)

| Model | Family | CER / WER (15 pages, informational, see caveat) | Structure | Answer-key fidelity (gate) | Faithfulness (gate) | Diagram detection | Latency/page | Fits 8GB (gate) | Setup difficulty | Rejection reason | Role |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MinerU2.5 | Paddle | 2.66 / 4.18 (worst of 4 deep-read, see caveat, driven by cosmetic garbling not real errors) | 4/4 table markup (native `type:"table"` bbox entries); row-pairing preserved | PASS | PASS, cleanest full-page defect profile of any candidate: no fabrication, no wrong values | 7/7 | 32.2s | PASS, 2.15 GB (best of all 6) | `uv pip install torch` without `--reinstall` silently kept the pre-existing unpinned torch |, | **primary** |
| DeepSeek-OCR | DeepSeek-OCR | 2.20 / 3.25 (best of the 5 survivors) | 4/4 table markup; row-pairing preserved | PASS | PASS, no wrong values; shares GS013 table-fabrication defect with Unlimited-OCR (same family, likely shared architectural trait, watch this pattern on real grid pages in Phase 1 step 3) | 7/7 | 60.2s (slowest of all 6, ~2x the next-slowest survivor) | PASS, 6.39 GB (closest to the 8GB ceiling of all 6) | Mixed `--index-url`+`--extra-index-url` let a CPU torch wheel win at first; unpinned torchvision silently upgraded pinned torch; `flash-attn` unbuildable (no `CUDA_HOME`), fixed by omitting the flash-attention kwarg; 2 runner-side bugs (Windows console UnicodeEncodeError losing 2 pages, wrong glob extension losing markdown on all 18) both fixed and backfilled |, | **secondary** |
| PaddleOCR-VL | Paddle | 2.38 / 3.51 | 4/4 table pages show real `<table>` markup; GS002's leaves/arrives row-pairing preserved | PASS | PASS (checklist scope), full-page read found a real wrong-value defect (GS014 Q8, $50 vs 50¢, worst magnitude of any candidate) plus 2 dropped-content instances | 7/7 | 33.6s | PASS, 2.90 GB | Run-plan doc had wrong backend name (`transformers` vs real `paddleocr` package); CUDNN version-mismatch warning, verified harmless |, | survivor (not selected, worst full-page defect profile of the deep-read candidates, see full-page findings above) |
| Unlimited-OCR | DeepSeek-OCR | 2.54 / 3.57 | 4/4 table markup; row-pairing preserved | PASS | PASS (checklist scope), full-page read found a real wrong-value defect (GS015 Q7, 4/5 vs 1/4) plus the shared GS013 table-fabrication defect | 7/7 | 22.9s | PASS, 6.31 GB | `HF_HUB_ENABLE_HF_TRANSFER=1` actively errors on this env's `huggingface_hub` (worse than Qwen3-VL's silent no-op); fixed by dropping the var, relying on `hf_xet` |, | survivor (not selected, same family as secondary; worse full-page defect profile than the chosen secondary) |
| Qwen3-VL small | Qwen | 2.44 / 3.49 | 1/4 table markup (renders one real markdown pipe-table on GS002 only; the 3 HTML-table-style grid/key pages come out as plain prose with no table markup at all) | PASS | PASS | **0/7, no diagram signal at all** (plain-text transcription mode, no bounding boxes or image references of any kind) | 20.5s | PASS, 3.96 GB | README's git-source-transformers implication was wrong; stable PyPI 5.15.0 sufficed. `HF_HUB_ENABLE_HF_TRANSFER=1` silently deprecated (no-op) on this env |, | survivor (not selected, no diagram detection or structural markup, a real functional gap for a paper full of diagram questions; not deep-read, disqualified on structure alone regardless) |
| GOT-OCR 2.0 | GOT | 2.41 / 3.51 | 0/4 table markup (both prompt modes are plain text, no table structure at all) | **FAIL**, GS005, GS008 (see gate table above) | **FAIL**, GS014 near-total silent content loss (see gate table above) | 0/7, no diagram signal at all | 16.9s (fastest of all 6) | PASS, 2.53 GB | `torch==2.0.1` pulled a CPU wheel by default (fixed with `--index-url cu118`); `RuntimeError: Numpy is not available` on all 18 pages until pinned `numpy<2` | Fails answer-key fidelity gate (GS005, GS008) and faithfulness gate (GS014); two-mode output is unreliable with no error signal either way (Issue 78) | **rejected** |
| dots.mocr | dots | n/a | n/a | n/a | n/a | n/a | n/a | n/a, never loaded | Repo drift confirmed (Issue 59/73, org renamed rednote-hilab → dots-studio); hard, non-optional `import flash_attn` at the modeling-file level, `transformers`' `check_imports()` refuses to load the class at all; no system CUDA toolkit, no official Windows wheel | Issue 77, setup-difficulty rejection, user-confirmed via AskUserQuestion, before installing a system CUDA toolkit or chasing an unofficial wheel | **rejected** |
| Mistral OCR (ref) | paid |, |, |, |, |, |, | n/a | Not run, paid, offline-only, out of scope for live inference | Reference-quality benchmark only, not a live candidate | reference |

Role column note: the protocol's four canonical values (primary, secondary, rejected, reference)
don't have a term for "passed every gate but wasn't chosen", this table uses "survivor (not
selected)" for that case rather than silently inventing a different vocabulary; flagging that
explicitly so it doesn't drift if this table is reused later.

## Decision (protocol Section 6)

**Revised after the full-page read above.** The original pick (primary PaddleOCR-VL,
secondary Unlimited-OCR) was based on the automated 35-item checklist plus operational metrics
(VRAM, latency, CER/WER), which scored 5 of 6 candidates identically clean and gave no way to
distinguish them on correctness. A full-page manual read of GS013-15, prompted by that uniformity
looking too clean to trust, found that PaddleOCR-VL and Unlimited-OCR each contain a real wrong
numeric value the checklist never saw, while MinerU2.5 and DeepSeek-OCR do not. That finding
reverses both roles.

1. **Gates applied.** GOT-OCR 2.0 and dots.mocr rejected (above). 5 survivors on the formal gates:
   PaddleOCR-VL, MinerU2.5, Qwen3-VL small, DeepSeek-OCR, Unlimited-OCR.
2. **Family groupings** (protocol Section 8, verbatim): Paddle family = {PaddleOCR-VL, MinerU2.5};
   DeepSeek-OCR family = {DeepSeek-OCR, Unlimited-OCR}; Qwen family = {Qwen3-VL small}. The final
   pair must span two different families.
3. **Primary: MinerU2.5.** Cleanest full-page defect profile of any candidate checked, no
   fabricated table schema, no wrong numeric value found anywhere across the 3 deeply-read pages,
   only cosmetic label garbling that never touches a number. Also the best operational number of
   the entire bake-off by a wide margin: 2.15 GB peak VRAM, well under every other survivor.
   Its CER/WER is the worst of the 4 deep-read candidates, but that number is now understood to be
   an artifact of prose-label garbling, not a real accuracy signal, and is outweighed by the
   substance finding. Qwen3-VL small was not competitive for primary despite decent CER/WER , 
   0/7 diagram detection and 1/4 table markup are real functional gaps for a paper set that is
   full of diagram questions (Q3 scale reading, Q29 inscribed-square area, multiple angle
   figures), disqualifying on structure alone regardless of the full-page findings.
4. **Secondary: DeepSeek-OCR.** Must be a different family from MinerU2.5 (Paddle), so either
   Qwen3-VL small or the DeepSeek-OCR family. Ruled out Qwen3-VL small for the same diagram/
   structure gap as above. Within the DeepSeek-OCR family, DeepSeek-OCR beats Unlimited-OCR on
   substance, not just on the automated checklist: DeepSeek-OCR produced no wrong numeric value on
   the 3 deep-read pages, while Unlimited-OCR did (GS015 Q7). DeepSeek-OCR also has the best
   CER/WER of all 5 survivors, a rare case here where CER/WER and full-page substance agree. The
   real cost is operational: worst VRAM of the 4 full-structure candidates (6.39 GB, closest to
   the 8GB ceiling) and worst latency (60.2s/page, ~2.6x Unlimited-OCR's). For an offline batch
   extraction job with no real-time constraint, that cost is accepted in exchange for the cleaner
   correctness profile, accuracy was weighted over throughput deliberately for this pick, not
   defaulted past. PaddleOCR-VL was excluded from both roles: same family as MinerU2.5 (can't pair
   with itself regardless of score), and separately disqualified on substance anyway, it has the
   single worst individual defect found across the whole bake-off (GS014 Q8, $50 recorded where
   the source says 50¢, a 100x magnitude error) plus 2 dropped-content instances.
5. **Complementary failure modes, informal supporting evidence for independence.** MinerU2.5 never
   fabricates table structure; DeepSeek-OCR does, once (shared with its family-mate Unlimited-OCR,
   likely an architectural trait, not a fluke). DeepSeek-OCR never garbles prose labels the way
   MinerU2.5 does. Their two defect types don't overlap. This is not the formal error-correlation
   proof protocol Section 6 step 5 asks for, that check still returns 0-vs-0 on the gates as
   formally scored, since neither candidate has a recorded gate failure on this 18-page set, but
   it is real, specific evidence in the direction the protocol's pairing logic wants: independent
   failure modes, not a shared blind spot.
6. **Residual risk to monitor, not resolved here.** DeepSeek-OCR's GS013 table-fabrication defect
   was found on a place-value table, not one of the 3 formal answer-key grid pages (which scored
   clean for every candidate). This suggests the defect is real and DeepSeek-OCR-family-wide, not
   limited to this one page, Phase 1 step 3's human verification pass should specifically watch
   for schema fabrication from DeepSeek-OCR on grid/table-shaped pages across the real corpus, not
   assume GS003/5/6 passing generalizes to every grid page the full extraction run will encounter.

**No fallback needed.** Two candidates cleared every formal gate and hold up as the cleanest pair
on the deeper full-page evidence too; Issue 67's least-bad-option fallback does not apply.

## Final verification

Both sides of the two decisive defects were checked directly, not taken on report:
- **Source:** the real GS013/14/15 source pages were obtained and read line by line. Every claim
  above about what the source actually says was confirmed genuine, including GS015 Q9's printed
  "= 162 cm²" typo (a real oddity in the original document, not introduced downstream, attach a
  source screenshot of this page in the dissertation write-up as direct evidence).
- **Model output:** the raw `extracted_text` for PaddleOCR-VL's GS014 Q8 and Unlimited-OCR's
  GS015 Q7 was read directly, verbatim, alongside MinerU2.5 and DeepSeek-OCR's output on the same
  two lines. Both defects hold up on direct inspection, with one wording correction: PaddleOCR-VL's
  GS014 Q8 defect is more precisely a duplicated erroneous line (`= $0.50` immediately followed by
  `= $50`) than a clean value substitution, the practical risk is unchanged, but the mechanism is
  worth describing accurately.

## Automated scoring artifacts

- `scoring_summary.csv` (committed): aggregate numbers only, one row per model. Reflects the
  automated 35-item checklist only, does not capture the full-page read findings above, which
  were manual and are recorded only in this document and the development log.
- `normalized/{model}/*.json` (gitignored, Issue 75): per-page shared-schema output + scores.
  Regenerate with `envs/scoring/.venv/Scripts/python.exe normalize_and_score.py`.
