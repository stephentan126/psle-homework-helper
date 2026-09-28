# Development log

This log records how the project progressed from planning to a working system between July and September 2026. Problems, risks and decisions were numbered as they arose. Comments in the code
and the report refer to them as "Issue N" (for example "Issue 91"). Every numbered entry is listed in [Section 4](#4-full-index).

The build was planned in ten phases, and code comments name them by number:

| Phase | Scope |
|---|---|
| 0 | Environment set-up and GPU checks |
| 1 | Extraction pipeline and question bank |
| 2 | API skeleton, database and the six response states (2b: parent PIN) |
| 3 | Language model, verification gate and hints |
| 4 | Photo input, question matching and weak mode |
| 5 | Safety layer |
| 6 | Fine-tuning experiment |
| 7 | Diagram handling (voice input deferred) |
| 8 | Frontend |
| 9 | Evaluation |

Comments that cite "the design specification" and a section number refer to the planning document written
before development. It is not part of this repository; the decisions it contained are recorded in this log
and in [ARCHITECTURE.md](ARCHITECTURE.md).

Entries 1 to 76 are risks identified during planning, before code was written, each turned into a design
rule. Entries 77 onward were found while building and testing. Fixes were only kept when a before-and-after
comparison on real data showed they helped; several were reverted because they broke something elsewhere.

## 1. Progress at a glance

| Measure | Start | End |
|---|---|---|
| Questions extracted from scanned papers | 0 | 5,819 from 140 papers (4,423 pages) |
| Questions flagged by the extraction checks (first 22-file batch) | 80.5% | 47.6% |
| Two-reader agreement on answer-key grids | 0% (every question marked "disagree") | 94.4% of grid pairs agree |
| Reader disagreement across the full corpus | not measured | 13.3%, inside an evidence-based band |
| Model slots decided by a written bake-off | 0 of 7 | 5 of 7, the other 2 blocked for a stated reason |
| Candidate models compared | 0 | about 25 |
| Self-harm recall of the safety layer | 1 of 3 (threshold only) | 9 of 10, 0 false positives |
| Held-out questions frozen and checked for leakage | 0 | 446 frozen, 116 overlap candidates reviewed by hand |
| Automated backend tests | 0 | 418 (399 pass, 19 skipped by design) |
| Gate decisions precomputed for fast responses | 0 | 3,558 questions (23.3 hours of offline computation) |
| Response states with a designed frontend screen | 0 of 6 | 6 of 6 |

## 2. Timeline

| Dates (2026) | Phase | What was built | What it unlocked |
|---|---|---|---|
| Before 1 Jul | Planning and environment | Design specification, 76 planning risks, bake-off protocol, GPU environment check | A build order where every stage is testable on its own |
| 1 to 29 Jul | Extraction and data quality | Three-stage extraction pipeline, OCR reader bake-off, answer-key parsing, deduplication, reader reconciliation, defect sweeps | A 5,819-question bank with verified answer keys where available |
| 3 to 7 Aug | Confirm-and-edit step | Vision model bake-off, photo preprocessing, match-first retrieval, confirm step, weak mode | Students can photograph a question and correct what was read |
| 3 to 12 Aug | Safety layer and parent PIN | Safety classifier bake-off, held-out validation, lexicon, crisis response, PIN unlock | Safe responses for children, parental control |
| 8 to 18 Aug | Retrieval, few-shot and QLoRA | Held-out freeze and leakage check, 1,414 training hints reviewed by hand, QLoRA training | A controlled test of whether grounding helps |
| 11 to 13 Aug | Hint flow and verification gate | API and six typed response states, LLM bake-off, SymPy check, self-consistency, method alignment, precompute cache | Hints that only go past Tier 1 when verified |
| 16 Aug to 2 Sep | Model selection and evaluation | Three-condition held-out experiment, significance tests, hint-quality scoring, adapter variants | Evidence-based decisions on what to serve |
| 20 to 26 Aug | Diagram handling | Safe fallback, chart and geometry model bake-offs | An honest "cannot see the figure" response |
| 2 to 3 Sep | Frontend and integration | Screen designs, Next.js app wired to the real backend, integration fixes | The complete working application |
| 27 to 28 Sep | Expert review and dry run | Heuristic evaluation, end-to-end dry run, fixes for the problems found | A tested, safer application |

## 3. Key milestones

Each entry follows the same pattern: what was found, how, what was done, and the measured result.

### Extraction and data

**Reversing the OCR choice (79).** An automated 35-item checklist scored five candidate readers as equal.
A full-page manual read found that the chosen primary reader had recorded "$50" where the page said "50¢",
and the secondary had read 1/4 as 4/5. The pair was reversed to MinerU2.5 and DeepSeek-OCR, and every
candidate's misreading of the cent sign led to a glyph normalisation step before comparison.

**GPU memory leak in long OCR runs (82 to 84).** Page time grew from about 13 s to 194 s as memory climbed
from 2.15 GB to 4.27 GB. The first suspected cause (shared GPU memory on a hybrid-graphics laptop) was
measured and ruled out. Running each batch of pages in a fresh subprocess gave 14 of 14 batches with no
timeouts or crashes.

**From regular expressions to layout regions (86, 87).** Text-pattern segmentation found 18 "questions" in
one paper, all of them cover-page instructions. Segmenting by the reader's layout regions found the 35 real
questions and handled shared diagrams and captions.

**Clean counts hiding wrong answers (91).** One paper showed zero flagged and zero unmatched questions, yet
17 of its 47 answers belonged to a different section of the paper, because question numbers restart in each
section. Answer matching was scoped by section, and per-question comparisons replaced summary counts as the
validation method.

**The comparator that never agreed (98 to 101).** The first end-to-end run marked all 82 questions as
readers "disagreeing", which would have capped every hint at Tier 1. The cause was a short-answer comparison
applied to whole pages. Per-question text similarity, with a threshold set inside a measured gap, and a
separate answer-grid check raised agreement to 94.4% of grid pairs and caught 3 real reading errors.

**A validation that validated nothing (116).** A 22-file re-run finished in 1.7 s because the pipeline
skipped files whose output already existed, so earlier "no change" results had compared old output with
itself. The proper re-run showed a small regression. Every validation since uses an emptied output folder.

**One fix, 616 answers (142).** Handling answer cells where the label and value were printed together
recovered 616 answers in one change and cut stray fragments by 90%, with every regression traced.

**Five attempts, then a different signal (145, 380).** Five content-based rules to remove bar-model labels
from answer tables each broke real answers elsewhere and were reverted. A rule based on page position
instead fixed the case with zero false matches across 4,423 pages.

**Setting the bar before the run (127, 156, 225).** A flat "percentage flagged" target was rejected as
arbitrary. Every flagged question was instead sorted into resolved, excluded with a stated reason, or
unreviewed, and the run was accepted when none remained unreviewed. A later band that had crept in without a
source was withdrawn and replaced by one measured from the data.

**Stale records (257, 260).** The database showed 82.4% of questions read by only one reader, while the true
figure was about 8%. File timestamps showed the merge step had run before the second reader finished. 4,224
records were refreshed and a staleness check now refuses to write out-of-date results.

**Wrong answers one PIN away (247 to 249).** A 30-question audit found 36.7% of sampled training rows had
defects, including two confidently wrong answers that the parent-unlock endpoint returned as-is. The rows
were quarantined, and the root cause, retrieval not excluding duplicate rows, was fixed across the corpus.

### Models and the gate

**Serving stack change (161, 162).** llama.cpp crashed with an illegal-instruction error on the laptop CPU,
which lacks AVX-512, and no prebuilt build avoided it. Serving moved to `transformers` with 4-bit NF4, which
also let training and serving share one stack.

**Choosing the language model (165).** Three models were compared on memory, speed and hint quality. Phi-4-mini
was chosen: Qwen3 4B garbled an easy question, and Qwen3 8B needed twice the memory for no clear gain.

**The MCQ index bug (169, 175, 177).** Multiple-choice keys store an option number, while the model answers
with a value, so correct answers failed the gate. Options are now resolved to values before comparison. The
same bug reappeared in a fallback path and was fixed there too.

**Correcting a data-error claim (173, 220).** An audit first reported six wrong answer keys. Checking each
printed page showed only one was a real key error. Four were a reading error that turned "÷" into "+" in the
question text, which became its own tracked issue.

**Latency and the precompute cache (167, 170, 174 to 176).** Two self-consistency passes took 15 to 38 s.
Batching the passes helped short questions. A 23.3-hour offline run cached gate results for 3,558 questions,
answered in about 0.01 s, protected by version and content checks.

**Unit symbols blocking correct answers (292, 293, 344).** "36" did not match "$36", and "64" did not match
"64°". This affected 23.8% of the training pool and capped real students' hints. Unit handling was fixed in
the live gate, and litre symbols were added later.

### Photo input and safety

**Match-first with an honest weak mode (180).** Students photograph questions that may not be in the corpus.
The system matches first, and when there is no confident match it caps help at Tier 1 and labels it
unverified, instead of guessing.

**Near-duplicate false matches (199).** Two questions differing only in "tenths" and "hundredths", or only
in one number, scored above the match threshold. Two extra checks (distinguishing words and differing
numbers) now block them, validated on 140 real sibling pairs.

**Safety threshold that did not hold (200 to 203).** A threshold tuned to 22/22 on hand-written cases fell to
19/22 on fresh cases, missing passive self-harm phrasing. A lexicon built from published screening
instruments was combined with the classifier, reaching 9/10 recall with no false positives. The one remaining
miss was left unpatched to avoid tuning to the test set.

**Category-appropriate responses (206, 207).** Every flagged message first received the self-harm response.
Responses now depend on the category, and harassment in a generated hint triggers a silent retry.

### Evaluation and fine-tuning

**Held-out leakage hunt (252, 282).** All 116 near-duplicate pairs between the held-out set and the training
data were reviewed by hand. One was a real overlap, the same commercial worksheet used by two schools,
recognisable by an identical torn-paper mark. The affected rows were excluded without modifying the frozen set.

**Human-reviewed training data (291).** Training hints were drafted by the live pipeline and then every one was
reviewed by hand: 959 accepted, 388 edited, 67 rejected. The review also surfaced 165 corpus defects that the
automated checks had missed.

**The cut-off that flipped the ranking (315, 316).** The first result suggested retrieval helped (11.3% vs
9.3%), but the difference was not significant, and 36% to 41% of answers were cut off at 300 tokens. Re-running
those at 600 tokens put the base model ahead (13.9% vs 12.9%), still not significant.

**An adapter that did not help (351 to 363).** Four separate tests of the fine-tuned adapter showed no
significant improvement, and it doubled the rate of unanswered gate checks. It was restricted to hint writing
and then switched off by default. Fine-tuning a small model on about 1,350 examples did not improve it.

**Diagram models (317 to 348).** The safe fallback was built before any diagram model. No chart or geometry
model reached a usable accuracy on 8GB, so none was connected. A feasibility check with a large hosted model
scored 16 of 17, showing the task is solvable with more capacity.

### Integration

**Model swap crash (364 to 366, 382, 385).** A second photo in the same server process crashed the
`transformers` library on Windows during model loading. Two suspected causes were tested and ruled out. The
first workaround allowed one model switch per process. The end-to-end dry run (385) then showed that even the
first switch, from the vision model to the language model, crashed the server. The final design never
switches models inside a process: after a successful photo read the server restarts itself, and a supervisor
script starts a fresh process in which the language model loads. The frontend waits and retries during the
restart (386).

**Frontend wired to the real backend (383).** Connecting the frontend exposed five issues: a crash when asking
for a Tier 2 hint on a photo question (the first suspected cause, a stale server process, was ruled out by
reading the code), the second-question restart, a "Done" button that returned to the same screen, an unlock
button shown when there was nothing to unlock, and a lost screen-preview tool. All five were fixed.

**Expert review (384).** A heuristic evaluation of all fourteen screens found eleven usability problems.
Nine were fixed, one was accepted and one needs parent accounts. Details: [ux_review/expert_review.md](ux_review/expert_review.md).

**End-to-end dry run (385 to 389).** Running the four main tasks on the real system found problems that no
unit test had caught: the model-switch crash, a lost photo after a failure, a Tier 3 hint that stated the
student's own answer, hints cut off mid-sentence, and live gate results that differed between runs. Each was
fixed and the tasks re-run. The answer-leak guard (387) was seen replacing a leaking hint in the re-run.
Details: [ux_review/dry_run_notes.md](ux_review/dry_run_notes.md).

## 4. Full index

| No. | Area | Issue | Outcome |
|---|---|---|---|
| 1 | Extraction | Vision models may silently correct a student's error instead of transcribing it. | Design rule adopted (transcribe, then interpret) |
| 2 | Extraction | Sub-part answers such as Q15a, b and c get misattached when keyed on number alone. | Design rule adopted (key on paper, booklet, number, sub-part) |
| 3 | Extraction | Paper 1 and Paper 2 reuse question numbers, so dropping the heading merges them. | Design rule adopted, fixed in 91 |
| 4 | Answer keys | Source types store answers differently, and assuming one layout broke the earlier pipeline. | Design rule adopted (handle each topology) |
| 5 | Extraction | Joining worked solutions by question number attaches working to the wrong question. | Design rule adopted (match by content) |
| 6 | Extraction | Mark allocations such as "[2]" can merge into the answer value. | Design rule adopted (strip mark annotations) |
| 7 | Gate | Answers like "$5", "5" and "500 cents" compare incorrectly without unit normalisation. | Design rule adopted (normalise units) |
| 8 | Gate | Equivalent forms such as 1/2, 0.5 and 50% fail a string comparison. | Design rule adopted (numeric equivalence) |
| 9 | Gate | Rounding differences between the key and the model's answer cause false mismatches. | Design rule adopted (tolerance rule) |
| 10 | Gate | An MCQ option index can be confused with the value it represents. | Design rule adopted (resolve option first) |
| 11 | Extraction | A check line after the answer may be read as the final answer. | Design rule adopted |
| 12 | Extraction | A blank answer line may be misread as a minus sign or fraction bar. | Design rule adopted (bias to null) |
| 13 | Diagrams | Locating a diagram is not the same as understanding it. | Design rule adopted (structured output required) |
| 14 | Diagrams | A diagram needed to solve a question may be ignored, producing a confident wrong hint. | Design rule adopted (explicit "can't see the figure" fallback) |
| 15 | Diagrams | Data tables inside questions may be flattened, losing values needed to solve them. | Design rule adopted (preserve rows and columns) |
| 16 | Diagrams | A misread bar model misleads the whole hint, not just one value. | Design rule adopted (classical vision plus fallback) |
| 17 | Diagrams | Thin number-line strokes are hard to tell apart from stray marks. | Deferred (line-detection plan untested) |
| 18 | Extraction | The reader may invent digits that are not on the page. | Design rule adopted (bias to null, human check) |
| 19 | Photo input | Without a per-field confidence signal the low-confidence cap cannot work. | Design rule adopted, resolved by 44 |
| 20 | Evaluation | Without an OCR gold set the reader bake-off would be subjective. | Design rule adopted (hand-labelled gold set) |
| 21 | Photo input | Phone photos without preprocessing would reduce reading accuracy. | Design rule adopted, built in 184 |
| 22 | Data quality | Records without source provenance break copyright filtering and debugging. | Design rule adopted (paper and page tags) |
| 23 | Data quality | OCR errors can weaken deduplication and leak training twins into the held-out set. | Design rule adopted (manual dedup sample check) |
| 24 | Answer keys | An unverified answer key would undermine the safety gate. | Design rule adopted (mandatory human check) |
| 25 | OCR readers | Two readers from the same model family would share mistakes, weakening agreement. | Design rule adopted (different lineages) |
| 26 | Gate | The gate is unreliable if the language model does its own arithmetic. | Design rule adopted (SymPy for arithmetic) |
| 27 | Evaluation | Changing the held-out set after QLoRA training would invalidate the result. | Design rule adopted (freeze before training) |
| 28 | Diagrams | Secondary-school geometry parsers target the wrong age group and problem type. | Decided: general reader, not geometry parsers |
| 29 | Diagrams | A caption describing a picture cannot be used to solve the question. | Design rule adopted (structured data output) |
| 30 | Diagrams | No off-the-shelf model reads bar models or number lines. | Design rule adopted (classical vision plan) |
| 31 | Diagrams | A reader may invent values that are not on a chart. | Design rule adopted (repeat reads, cross-check) |
| 32 | Hardware | MERaLiON 10B does not fit in 8 GB of VRAM. | Decided: MERaLiON-2-3B |
| 33 | Hardware | Loading several large models at once would exceed the 8 GB GPU. | Design rule adopted (one large model at a time) |
| 34 | Hardware | Long retrieval prompts add KV-cache memory on top of model weights. | Design rule adopted (lean prompts, measure VRAM) |
| 35 | Gate | SymPy cannot check bar-model or reasoning questions. | Design rule adopted (route by question type) |
| 36 | Gate | A verified answer does not guarantee the separately generated hint is sound. | Design rule adopted (hint from verified path) |
| 37 | Gate | The model can reach the right answer by faulty reasoning. | Design rule adopted (method-alignment check) |
| 38 | Matching | Omitting required embedding prefixes lowers retrieval quality. | Design rule adopted (use documented prefixes) |
| 39 | Planning | Some leading embedding models carry non-commercial licences. | Decided: Apache or MIT models only |
| 40 | Planning | Ethics approval for research with minors can take weeks. | Design rule adopted (apply early), not progressed (see 368) |
| 41 | QLoRA | Building QLoRA training pairs is substantial data-preparation work. | Design rule adopted (budgeted as a task) |
| 42 | Matching | A strict topic filter excludes correct examples when the topic label is wrong. | Design rule adopted (soft top-2 filter) |
| 43 | Evaluation | Counterbalancing condition order alone lets answers leak between conditions. | Design rule adopted (Latin square design) |
| 44 | Photo input | The live low-confidence cap had no implementation path with a single reader. | Decided: single-model confidence signal |
| 45 | Hardware | Model load and swap latency was not budgeted. | Closed (measured in 164) |
| 46 | Gate | Self-consistency passes multiply latency. | Design rule adopted (drop to 2 passes if slow) |
| 47 | Evaluation | The headline evaluation would likely exclude the hardest diagram questions. | Design rule adopted (report diagram subset separately) |
| 48 | Safety | No content safety layer existed in the plan. | Design rule adopted (dedicated safety classifier) |
| 49 | Safety | A naive classifier may miss an at-risk child or flag ordinary maths frustration. | Design rule adopted (borderline calibration set) |
| 50 | Evaluation | Only the OCR slot had a full model-rejection protocol. | Design rule adopted (protocol for every slot) |
| 51 | Diagrams | The bar-model line detector was chosen without comparing alternatives. | Deferred (comparison not run) |
| 52 | Matching | The vector store was never chosen, then over-specified for the project's scale. | Decided: plain NumPy cosine similarity |
| 53 | Serving | LLM serving software was never chosen. | Decided: llama.cpp, reversed in 162 |
| 54 | Serving | The quantization format was named too vaguely. | Decided: NF4 (revised in 162) |
| 55 | Planning | Seven live models is too many moving parts for a solo build. | Design rule adopted (four-model core first) |
| 56 | Serving | No degrade path existed if an optional model fails to load. | Design rule adopted, built in 317 |
| 57 | Serving | The costly hint endpoint had no guard against duplicate requests. | Design rule adopted (request ID guard) |
| 58 | Safety | The safety classifier had no defined failure behaviour. | Design rule adopted (fail closed) |
| 59 | Process | Model versions were not pinned, and one model had already been renamed. | Design rule adopted (pin revisions) |
| 60 | Serving | The live pipeline had no observability plan. | Design rule adopted (structured logging) |
| 61 | Data quality | File naming is inconsistent across years, and some school names are ambiguous. | Fixed (normaliser plus human review) |
| 62 | Extraction | Batch extraction over 100+ PDFs will meet at least one bad file. | Design rule adopted (per-file isolation, resumable runs) |
| 63 | Diagrams | The standard OpenCV package lacks a working line segment detector. | Fixed (opencv-contrib-python pinned) |
| 64 | Hardware | Automatic driver updates can invalidate a working environment. | Design rule adopted (disable auto-updates) |
| 65 | Database | SQLite's default journal mode is less robust to unclean shutdown than WAL. | Fixed (WAL mode enabled) |
| 66 | Process | Commits and rollback were not tied to slice completion. | Design rule adopted (commit per slice) |
| 67 | Evaluation | No plan existed for a bake-off with no clear winner. | Design rule adopted (least-bad pick, documented) |
| 68 | Process | No environment template existed for configuration. | Fixed (.env.example added) |
| 69 | Hardware | A crashed script can leave GPU memory allocated. | Design rule adopted (check for orphan processes) |
| 70 | Hardware | The laptop's hybrid graphics can silently route work to the integrated GPU. | Design rule adopted (force high-performance GPU) |
| 71 | Answer keys | Two answer-key topologies exist in the actual files. | Design rule adopted (branch on topology) |
| 72 | Data quality | Human school-name review answers were discarded on every rescan. | Fixed (persistent canonical store) |
| 73 | Process | A corrected model name drifted back in a document not updated with it. | Fixed, rule adopted (check names before use) |
| 74 | Evaluation | Gold-set instructions did not exclude the held-out folders as a page source. | Fixed (source restriction added) |
| 75 | Process | Purchased-paper content in research files was not covered by git ignore rules. | Fixed (ignore rules extended), history purge undecided |
| 76 | Evaluation | Self-computed answers were used in place of the printed answer key. | Fixed (rows reverted), rule adopted |
| 77 | OCR readers | Models with a hard flash_attn import cannot run on this Windows setup. | Decided: dots.mocr rejected |
| 78 | OCR readers | Multi-mode readers need every mode scored on the full gold set. | Decided: GOT-OCR 2.0 rejected |
| 79 | OCR readers | An automated checklist scored candidates clean while missing a $50 versus 50c error. | Decided: reader pair re-chosen from manual reads |
| 80 | Process | Pipeline wiring assumed reader and school-name functions that did not exist. | Fixed (rewired to actual APIs) |
| 81 | OCR readers | The two OCR readers cannot share one process due to conflicting pins and VRAM. | Decided: three-stage pipeline (render, read, merge) |
| 82 | Hardware | MinerU2.5 slowed from about 13s to 194s per page as GPU memory accumulated. | Fixed (per-page memory release) |
| 83 | Hardware | Per-page memory release did not stabilise DeepSeek-OCR, which crashed and stalled. | Mitigated (see 84) |
| 84 | OCR readers | DeepSeek-OCR needed isolation to survive long runs. | Mitigated (batched subprocesses, 0 timeouts in 14 batches) |
| 85 | Extraction | Pipeline stages were slow, so CPU stages were parallelised across pages and files. | Fixed, Stage A 12.36s to 2.44s |
| 86 | Extraction | Regex segmentation read cover-page instructions as questions, 18 fake questions from PSLE 2019. | Reverted, redesigned (region-based, 35 questions) |
| 87 | Extraction | Three regressions were reported after the region-based segmentation redesign. | Fixed (two), one disproven (see 88) |
| 88 | Process | Windows Bash pipes silently dropped debug lines containing the cent symbol. | Design rule adopted (debug output to UTF-8 files) |
| 89 | Answer keys | Separate answer files were never rendered, and no answer-grid parser existed. | Fixed (grid parser built) |
| 90 | Answer keys | Sub-part answers were unmatched or ambiguous between bare and lettered rows. | Fixed, PSLE 2019 unmatched 2 to 0 |
| 91 | Answer keys | Question-number reuse across papers attached answers from the wrong section. | Fixed (section-scoped matching, 17 of 47 chij answers corrected) |
| 92 | Data quality | School-name groups were treated as trusted without evidence from the source papers. | Fixed (five groups verified from school records) |
| 93 | Data quality | The pipeline used unverified school names because no verified field existed. | Fixed (tag, not refuse) |
| 94 | Data quality | The remaining school-name groups needed checking against rendered cover pages. | Closed (verified, one misfiled PDF found) |
| 95 | Data quality | ACS Junior and ACS Primary appeared to be variants of one school name. | Closed (two separate schools) |
| 96 | Data quality | A Tao Nan PDF was a merged compilation containing another school's paper. | Fixed (file excluded, not re-keyed) |
| 97 | Data quality | The answer-file lookup bypassed the exclusion list through its own file search. | Fixed (exclusion enforced at source) |
| 98 | OCR readers | First fresh full run marked all 82 questions as reader disagreement. | Fixed in 99 to 101 |
| 99 | OCR readers | Readers were compared per page instead of per question. | Fixed (per-question segmentation) |
| 100 | OCR readers | Agreement stayed at 0 of 78 because a short-value checker compared whole question text. | Fixed (text similarity, threshold 0.65) |
| 101 | OCR readers | Text similarity alone cannot catch a single-digit difference in final answers. | Fixed (grid answer cross-check, 101 of 107 agree) |
| 102 | Database | Extraction wrote only JSON checkpoints, and diagram and solution fields were placeholders. | Fixed (SQLite wired, fields populated) |
| 103 | Extraction | First 22-file batch of 837 pages exposed three extraction defects. | Closed (fixed in 104, 107, 111) |
| 104 | Answer keys | Answer-key section captions typed as titles were not recognised. | Fixed, flagged 904 to 688 |
| 105 | Answer keys | A Nanyang answer table paired its label and content cells in the wrong order. | Fixed (in 113) |
| 106 | Process | The the design specification issue register stopped being updated after entry 79. | Fixed (entries 80 to 103 added) |
| 107 | Extraction | Equation regions were silently dropped from question text. | Fixed, 13 questions changed |
| 108 | Extraction | A code-type region holding worked-solution content was left out of body text. | Reverted, redesigned (reversed in 109) |
| 109 | Extraction | Code-type regions needed including in body content after review. | Fixed |
| 110 | Answer keys | Narrative answer booklets for 2014 to 2018 left all 240 questions unmatched. | Fixed for MCQ (126), 165 Paper 2 questions deferred |
| 111 | Answer keys | Column-span information was lost before answer-grid pairing. | Fixed, flagged 688 to 661 |
| 112 | Answer keys | Nested sub-tables under row spans vary too much for one parsing rule. | Deferred (mitigated in 114) |
| 113 | Answer keys | A blank cell was prepended even when a row already began with a label. | Fixed, flagged 661 to 541 |
| 114 | Answer keys | Row-span shadow rows contaminated answers with fabricated values. | Mitigated (contamination only, not full fix) |
| 115 | Answer keys | Final answers held in full-width shadow rows were not attached to their question. | Fixed (Rule A, extended in 117) |
| 116 | Process | Validation reused an output directory, so every "no difference" check compared stale output. | Fixed, rule adopted (fresh directories), flagged 541 to 542 |
| 117 | Answer keys | Rule A from 115 was too narrow and missed recoverable answers. | Fixed (Rule B added) |
| 118 | Answer keys | Row-span cells combining a question number and sub-part marker were not recognised. | Fixed (in 121) |
| 119 | Answer keys | One Raffles Girls question received content from two separate contamination sources. | Deferred (root-caused, not fixed) |
| 120 | Answer keys | A sub-part label outside the expected pattern caused its answer to be dropped. | Fixed (in 122) |
| 121 | Answer keys | Recognition needed for row-span cells combining a number and sub-part marker. | Fixed, flagged 542 to 540 |
| 122 | Answer keys | Sub-part rows just past a row-span shadow lost their content. | Fixed, flagged 540 to 539 |
| 123 | Answer keys | An answer continuing into a new table on the next page cannot be reached. | Deferred |
| 124 | Answer keys | The glued number and sub-part marker problem also occurs on ordinary rows. | Fixed (in 125) |
| 125 | Answer keys | Glued-marker labels needed handling in the main row-pairing loop. | Fixed, flagged 539 to 535 |
| 126 | Answer keys | Narrative MCQ answer pages needed a dedicated parsing path. | Fixed, 75 of 75 MCQ answers recovered |
| 127 | Evaluation | The corpus acceptance bar had to be set before the full run. | Decided: categorisation, not a flat percentage |
| 128 | Hardware | The full-corpus reading run needed scoping as a multi-day GPU commitment. | Decided: full run launched after scoping |
| 129 | Hardware | DeepSeek-OCR uses 94 to 96% of VRAM regardless of batch size. | Decided: MinerU2.5-only pass first, DeepSeek-OCR deferred |
| 130 | Evaluation | The first full-corpus merge flagged 2,768 of 5,447 questions (50.8%). | Closed (split into follow-up threads) |
| 131 | Answer keys | The narrative answer-key shape also appears in ordinary school papers. | Deferred (extends 110) |
| 132 | Answer keys | Grids with a label row and a separate value row were garbled during pairing. | Fixed |
| 133 | Answer keys | Grid labels with a closing parenthesis or no decoration were never classified. | Fixed, 19 newly matched |
| 134 | Answer keys | Some papers print answers directly after each question with no answer section. | Deferred (needs new extraction path) |
| 135 | Data quality | One Raffles scan is incomplete and holds Booklet A only. | Accepted limitation |
| 136 | Data quality | A file labelled Raffles has a Rosyth School cover page. | Closed (only mislabel found in corpus scan) |
| 137 | Answer keys | A section caption split into three text regions gave 0 of 15 held-out answers. | Fixed, all 7 held-out files 15 of 15 |
| 138 | Answer keys | The fix in 132 added a spurious worked-solution tail in three files. | Fixed (space-variant fix) |
| 139 | Evaluation | Acceptance-bar recheck after fixes 132 to 138. | Closed, flagged 50.8% to 47.0% |
| 140 | Evaluation | The continuation-fragment rate rose above the batch baseline. | Fixed (in 142) |
| 141 | Answer keys | Two unexplained findings needed checking against the question-number reuse problem. | Closed (both new causes, see 144, 145) |
| 142 | Answer keys | A label and its content glued into one table cell blocked answer pairing. | Fixed, 616 questions newly matched |
| 143 | Evaluation | Acceptance-bar recheck after fix 142. | Closed, flagged 47.0% to 35.7% |
| 144 | Extraction | Worked-solution pages without tables were classified as question pages. | Fixed, 13 fake questions removed |
| 145 | Extraction | Bar-model label tables were parsed as answer grids, creating false question numbers. | Deferred after five reverted attempts, fixed in 380 |
| 146 | Answer keys | Answer-key pages without a section caption could not be looked up by section. | Fixed, 278 newly matched |
| 147 | Answer keys | A "Paper 1" umbrella label did not match Booklet A and B answer sections. | Fixed, 86 newly matched |
| 148 | Answer keys | A Tao Nan paper uses a new answer-page shape and all its questions are flagged. | Deferred |
| 149 | Answer keys | A recurring section-mismatch pattern cannot be resolved safely. | Accepted limitation (303 questions left flagged) |
| 150 | Answer keys | The glued-label fix missed a variant with an explicit column span. | Fixed, 37 newly matched |
| 151 | Answer keys | Parenthesised sub-part labels such as Q21(a) were not recognised. | Fixed, 39 field changes |
| 152 | Answer keys | A row span in one column swallowed another column's rows. | Deferred |
| 153 | Answer keys | A glued label cell next to an empty sibling cell escaped the fix in 150. | Fixed, 15 newly matched |
| 154 | Answer keys | Rows where every cell holds its own label and value were not parsed. | Fixed (later, 24 field changes) |
| 155 | Answer keys | A column span on a label cell shifted label and value alignment. | Reverted (fix broke other files), deferred |
| 156 | Evaluation | The original acceptance bar could not close without forcing unsafe guesses. | Decided: A/B/C/D categories, unreviewed count reached 0 |
| 157 | Evaluation | Sub-counts in 156 summed to 879 instead of 1,002. | Fixed (counts corrected) |
| 158 | Database | The live database held 81,479 question rows against 5,425 questions. | Fixed (truncate and reload, root cause in 230) |
| 159 | Data quality | Other pipeline stages were checked for the same sync gap as 158. | Closed, 0 mismatches, verifier tool added |
| 160 | Database | A relative database path opened a stray database from another directory. | Fixed (path anchored to repo root) |
| 161 | Serving | llama-cpp-python crashed with an illegal instruction because the CPU lacks AVX-512. | Closed (see 162) |
| 162 | Serving | No usable llama.cpp build exists for this machine. | Decided: transformers with bitsandbytes NF4 |
| 163 | Hardware | Model downloads stalled and the GPU cleanup helper never freed memory. | Fixed (curl download, cleanup helper corrected) |
| 164 | Serving | Three candidate LLMs needed load time, VRAM and speed measured. | Closed (Phi-4-mini, Qwen3 4B and 8B measured) |
| 165 | Serving | The LLM pick had to be made on hint quality. | Decided: Phi-4-mini 3.8B |
| 166 | Gate | The Socratic gate steps needed building and testing on corpus data. | Closed (built, MCQ index routing added) |
| 167 | Gate | Self-consistency latency exceeds the 10 to 15 second rule from 46. | Mitigated (2 passes, batching in 170) |
| 168 | Serving | Concurrent identical requests caused a database-locked error in the duplicate guard. | Fixed (busy timeout, rollback) |
| 169 | Gate | Self-consistency still compared against the stored MCQ index. | Fixed (in 175) |
| 170 | Gate | Self-consistency ran repeated sequential generations. | Mitigated (batched, 14.98s to 11.46s on one question) |
| 171 | Gate | A spot-check found misrouted questions in the "sum" bucket. | Closed (full audit in 172) |
| 172 | Gate | All 301 "sum"-classified questions needed auditing. | Fixed (61 diagram routing bugs), rest categorised |
| 173 | Answer keys | Six stored answer-key values were reported as wrong after gate disagreement. | Retracted (incorrect finding), 1 of 6 was a key error |
| 174 | Gate | Live gate latency needed an offline precompute path. | Fixed (built and tested on a bounded slice) |
| 175 | Gate | Self-consistency compared answers before resolving MCQ options. | Fixed |
| 176 | Gate | The full-corpus offline precompute run was outstanding. | Closed, 3,558 questions, 84.0% capped at Tier 1 |
| 177 | Gate | The fallback path in 175 reintroduced the MCQ index bug. | Fixed, 2 rows recomputed |
| 178 | Gate | The precompute cache could not detect content changes to a question. | Fixed (content fingerprint added) |
| 179 | Diagrams | No plan existed to generate a bar model as part of a hint. | Deferred (planned, not built) |
| 180 | Matching | Photographed questions often have no answer key in the corpus. | Decided: match-first with Tier 1-capped weak mode |
| 181 | Serving | The response schema made a weak-mode response impossible to construct. | Fixed |
| 182 | Database | Attempts needed to reference either a corpus question or a student submission. | Fixed (schema migrated and tested) |
| 183 | Photo input | The photo reader needed a bake-off on transcription faithfulness. | Decided: Qwen3-VL-8B-Instruct |
| 184 | Photo input | Live-photo preprocessing was built, and a full-size photo took over 60 minutes. | Fixed (2000px resize cap) |
| 185 | Matching | The embedding model for question matching needed a bake-off. | Decided: Qwen3-Embedding-0.6B |
| 186 | Matching | The match-confidence threshold needed evidence rather than an invented number. | Decided: 0.90 |
| 187 | Matching | The confirm step and match-first wiring needed building. | Closed (built and tested) |
| 188 | Photo input | The confirm step showed matched corpus text instead of the student's photo text. | Fixed |
| 189 | Serving | Photo submissions needed a confirm-triggered hint endpoint. | Closed (built) |
| 190 | Serving | Duplicate-request replay logic broke under the new confirm endpoint. | Fixed |
| 191 | Hardware | The photo reader and the LLM cannot both stay loaded in 8 GB. | Fixed (mutual eviction between services) |
| 192 | Serving | Submission-creation logic was duplicated across typed and photo entry points. | Fixed (shared function) |
| 193 | Photo input | The live photo entry point needed an end-to-end test with a photo. | Closed, whole-page photos unhandled |
| 194 | Photo input | A live confidence signal for photo reading had to be chosen. | Decided: token log-probability, threshold -0.02 |
| 195 | Process | Replacing stdout at module level broke pytest output capture. | Fixed |
| 196 | Serving | Remaining photo steps were mostly built, with a response-contract design tension. | Decided: reason field on extraction-failed response |
| 197 | Photo input | Slice exit criteria required a test photo per layout type. | Closed (MCQ, handwritten and diagram photos passed) |
| 198 | Process | A stale docstring caused a false "not yet wired" claim in a build report. | Fixed (docstring rewritten) |
| 199 | Matching | Near-identical but different questions scored above the 0.90 match threshold. | Fixed (term and numeric conflict checks) |
| 200 | Safety | Both named safety-classifier candidates were gated repositories. | Deferred, then ShieldGemma-2B picked provisionally |
| 201 | Safety | The tuned 0.05 threshold scored 19 of 22 on a held-out set. | Reverted, redesigned (see 202) |
| 202 | Safety | Passive self-harm phrases were missed by the classifier. | Fixed (cited lexicon, self-harm recall 9 of 10) |
| 203 | Safety | Flagged self-harm input needed a crisis-resource response. | Fixed (existing path used), parent alert not built |
| 204 | Safety | Detection and crisis response were built but unreachable by live requests. | Fixed (live, ShieldGemma-2B on CPU) |
| 205 | Safety | ShieldGemma-2B on CPU adds latency, and batching did not help. | Accepted limitation (about 10.5s combined) |
| 206 | Safety | Every safety flag produced the self-harm crisis message. | Fixed (in 207) |
| 207 | Safety | The safety check did not report which category fired. | Fixed (category-specific responses) |
| 208 | Matching | Match-first false positives were elevated while embeddings were incomplete. | Fixed, 3 of 20 to 2 of 20 |
| 209 | Data quality | Four corpus rows were exam cover-page text, two with live answers. | Fixed (flagged and excluded from serving) |
| 210 | Safety | Classifier failure blocked output only through an unlogged HTTP 500. | Fixed (dedicated fail-closed path) |
| 211 | Serving | Worst-case combined request latency had never been quantified. | Closed (about 106s, later about 139s) |
| 212 | Database | Reader disagreement detail was free text with no structured type. | Fixed (structured type added) |
| 213 | OCR readers | Grid-agreement messages had been silently discarded since entry 101. | Fixed |
| 214 | OCR readers | The DeepSeek-OCR full-corpus pass needed to run unattended. | Closed (launched after resume test) |
| 215 | Hardware | A background supervisor killed the DeepSeek-OCR pass three times for low memory. | Mitigated (resilient wrapper), no data lost |
| 216 | Hardware | 347 batches crashed within about 2.5s under memory pressure. | Mitigated (wrapper self-recovered), no data lost |
| 217 | OCR readers | The per-question comparison path lacked the new disagreement type, failing all 132 files. | Fixed, 132 of 132 files |
| 218 | Process | The page-rendering script used a directory-relative import. | Fixed |
| 219 | Extraction | A bare "6)" on one page created a fake sub-question. | Fixed (adjacency signal) |
| 220 | Data quality | A printed division sign was read as a plus sign in question text. | Mitigated (review-queue scan), 73.25% of rows not checkable |
| 221 | OCR readers | The DeepSeek-OCR pass stalled at 99.8%, looping on one ruler diagram. | Fixed (skip guard), 4,422 of 4,423 pages |
| 222 | OCR readers | Corpus-wide reader agreement needed measuring once both passes finished. | Closed, flagged rate 13.31% |
| 223 | Answer keys | A second printed answer-key error was found in a source paper. | Accepted limitation (not corrected) |
| 224 | OCR readers | Most single-reader records came from an alignment bug, not a missing reader. | Fixed in part, single-reader 942 to 621 |
| 225 | Evaluation | The cited 20 to 35% agreement acceptance band had no source. | Retracted, replaced by 8.16 to 18.16% monitoring band |
| 226 | OCR readers | Dropped fraction denominators produced phantom question numbers. | Fixed, 1 record changed |
| 227 | OCR readers | Gaps in the cover and margin boilerplate whitelist left records unreconciled. | Fixed, single-reader 621 to 403 (with 228) |
| 228 | OCR readers | The recovery similarity threshold rejected valid matches between readers. | Fixed, single-reader 621 to 403 (with 227) |
| 229 | OCR readers | Records on worked-solution pages diverged between the two readers. | Fixed 24 of 97, 73 accepted as limitation |
| 230 | Database | Resume markers could fall out of sync with the database, causing the bloat in 158. | Fixed (desync check refuses to write) |
| 231 | Data quality | Deduplication needed a similarity threshold and survivor tie-breaking rules. | Fixed, 111 rows superseded, none deleted |
| 232 | Serving | The worst-case request chain of about 139s had no live tracking. | Fixed (latency monitoring, 90s alert) |
| 233 | Diagrams | The DeepSeek-OCR diagram marker was not used as a diagram signal. | Fixed (secondary signal) |
| 234 | Gate | The extraction-gap bucket from 172 needed re-checking against the current corpus. | Closed (diagnosis only) |
| 235 | Diagrams | The validated diagram-signal fix needed applying to the live corpus. | Fixed, diagram flags 2,649 to 2,669 |
| 236 | Data quality | Boilerplate and margin text carried misattached answers in 52 records. | Fixed (at extraction level) |
| 237 | Hardware | An environment memory guard repeatedly killed long-running jobs. | Mitigated (see 238) |
| 238 | Process | Test-suite runs were killed by the memory guard. | Fixed (self-relaunching test wrapper) |
| 239 | Data quality | MCQ rows with nonstandard option formats needed re-deriving against the live corpus. | Closed (diagnosis only, 12 cases found) |
| 240 | Safety | The safety-classifier pick was still marked provisional. | Decided: ShieldGemma-2B permanent |
| 241 | Data quality | Two answers carried a stray prefix and one answer was misattached. | Fixed, 3 records corrected |
| 242 | Evaluation | A voice-input bake-off was planned but no evaluation set exists. | Deferred (no evaluation set) |
| 243 | Evaluation | A topic-classifier bake-off was planned but no labelled evaluation set exists. | Deferred (no evaluation set) |
| 244 | Auth | Parent PIN unlock needed a defined scope, hashing and lockout. | Decided: per-question unlock, argon2, lockout after 5 failures |
| 245 | Auth | Parent and student signup would collect personal data from families. | Deferred (pending ethics and legal review) |
| 246 | QLoRA | A training-data sample audit before QLoRA showed a high defect rate. | Closed, 9 of 30 defects, pair target retracted |
| 247 | Answer keys | Two wrong stored answers were reachable through the parent PIN unlock. | Fixed (quarantined in 248) |
| 248 | Answer keys | Wrong answers needed blocking from the unlock path. | Fixed, unlock returns 422 for 3 rows |
| 249 | Matching | Retrieval did not exclude superseded duplicate rows. | Fixed (corpus-wide filter) |
| 250 | Matching | Deduplication linked different diagram questions that share generic wording. | Fixed (false pairs unlinked) |
| 251 | OCR readers | Held-out agreement status ignored a missing reader on the answer page. | Fixed (in 253) |
| 252 | Evaluation | The held-out set overlapped with the training pool. | Closed (116 pairs reviewed, see 282) |
| 253 | OCR readers | Grid-merge agreement status needed to reflect single-reader answer pages. | Fixed, held-out set re-frozen |
| 254 | Evaluation | 116 held-out overlap candidates needed triage. | Closed (triaged, reviewed in 282) |
| 255 | Data quality | Three open items needed page images for human review. | Closed (evidence packaged) |
| 256 | Data quality | Human review verdicts needed applying to the database. | Fixed, 16 rows written |
| 257 | Database | Stored agreement status was stale, 82.4% single-reader against 7.2% live. | Fixed (in 260) |
| 258 | Data quality | An unresolved angle answer needed independent re-derivation. | Closed, 13 and 77 degrees kept |
| 259 | Evaluation | The first overlap review batch needed page images. | Closed (images packaged) |
| 260 | Database | Checkpoint-skip logic let stale agreement data persist. | Fixed, 4,224 rows refreshed |
| 261 | Data quality | Three residual gaps from 260 needed sizing. | Closed (audit only) |
| 262 | Data quality | Phantom boilerplate rows were live and unflagged in the database. | Fixed, 67 rows voided |
| 263 | Database | 394 missing rows could not be inserted by the whole-file production writer. | Deferred (stopped, see 265) |
| 264 | Data quality | Six rows sharing a natural key needed review. | Closed, 1 already handled, 5 sent for review |
| 265 | Database | The 394 missing rows needed a safe insert path. | Fixed, rows 5,425 to 5,819 |
| 266 | Data quality | Three open queue items needed page-image review batches. | Closed (images packaged) |
| 267 | Process | The open-issues tracker had drifted from the decision log. | Fixed, rule adopted (update in same commit) |
| 268 | Data quality | A 21-row set of split working fragments needed review images. | Closed (images packaged) |
| 269 | Data quality | Five collision rows needed resolving from page-image review. | Fixed, 3 writes, 3 no-ops |
| 270 | Data quality | Glued-fragment groups carried boilerplate halves. | Fixed (boilerplate halves voided) |
| 271 | Answer keys | Suspected split MCQ groups failed page review, with answers apparently mismatched. | Retracted (incorrect finding), 18 rows reopened |
| 272 | Answer keys | A 40-row set carried misattached answers. | Fixed (voided, not re-attached) |
| 273 | Data quality | Working fragments were the only record for some unanswered questions. | Deferred (held for reconstruction, see 277) |
| 274 | Answer keys | An MCQ option list was scattered across adjacent question numbers. | Fixed (in 278) |
| 275 | Answer keys | Answer-grid cell values bled onto the wrong question. | Fixed for 2 rows, 3 noted |
| 276 | Answer keys | Several defects suggest a systematic MCQ answer and option misattribution mechanism. | Deferred (open investigation) |
| 277 | Answer keys | Seven unanswered questions needed worked solutions reconstructed from source pages. | Fixed, 7 rows written |
| 278 | Data quality | Scattered MCQ option lines needed folding back into their question row. | Fixed |
| 279 | Answer keys | A printed answer key gives 18 beads where the correct answer is 90. | Fixed (90 written in 277) |
| 280 | Data quality | A reconstructed question kept incomplete question text. | Fixed |
| 281 | Matching | The review tracker for superseded-pair disagreements had open pairs. | Closed, 7 pairs resolved |
| 282 | Evaluation | All 116 held-out overlap candidates needed manual review. | Closed, 116 of 116 reviewed, 1 item excluded |
| 283 | Data quality | Short spurious question-text rows were not questions. | Mitigated, 121 rows flagged |
| 284 | Answer keys | Two unresolved sample rows needed reworking. | Fixed 1, 1 left open |
| 285 | Data quality | Further non-question patterns needed verified id lists. | Fixed, 103 rows flagged |
| 286 | Data quality | Thirteen ambiguous rows needed resolving against source pages. | Fixed, 9 flagged, 1 open |
| 287 | Data quality | Three valid questions had truncated extracted text. | Mitigated (quarantined, restored in 288) |
| 288 | Data quality | The three truncated questions needed restoring. | Fixed (restored) |
| 289 | Data quality | One row was suspected of a page-index error. | Closed (hypothesis disproved, row re-characterised) |
| 290 | Database | The insert in 265 created 17 live duplicate rows. | Fixed (deduplicated by superseded link) |
| 291 | QLoRA | The method for producing training hint targets was undecided. | Decided: bootstrap plus mandatory human review |
| 292 | QLoRA | Pool scope was undecided and a gate answer-matching gap appeared. | Decided: full 1,205-row pool including diagrams |
| 293 | Gate | Unit stripping failed on dollar signs, degree signs and superscripts. | Fixed, affected 287 of 1,205 rows |
| 294 | Hardware | Test runs competed for GPU memory with a long bootstrap job. | Mitigated (GPU test files listed, per-row timeout) |
| 295 | Data quality | Unit notation and trailing boilerplate needed normalising. | Fixed, 2,423 rows stripped |
| 296 | Data quality | Prescreening found glued questions and mismatched ground truth at about 4 to 5%. | Closed (sized, see 297) |
| 297 | Data quality | A 580-row sweep needed categorising by hand. | Fixed, 163 defects (2.8%) flagged |
| 298 | Data quality | A truncated-stem defect shape was missed by earlier heuristics. | Fixed, 31 rows quarantined |
| 299 | Data quality | Further candidates of the same defect family were found. | Fixed, 7 new rows quarantined |
| 300 | QLoRA | Unit-notation mismatches recur across generated hint text. | Deferred (generation-source defect logged) |
| 301 | Process | The review tool's sort order needed a secondary key. | Fixed (tooling only) |
| 302 | Data quality | Reject-pile defects needed repair in the database. | Fixed, 41 of 50 ids corrected |
| 303 | Answer keys | A row was judged a source-material defect and not repaired. | Retracted (incorrect finding, see 305) |
| 304 | Data quality | An arithmetic sweep checked three candidate rows. | Fixed 1, 2 false leads |
| 305 | Answer keys | Paper 1 rows in one Nanhua PDF pulled Paper 2's answer key. | Fixed, 7 rows corrected |
| 306 | Process | The decision log working copy was silently truncated by about 7,600 lines. | Fixed (restored from git) |
| 307 | Answer keys | An exposure sweep after 305 found a second cross-paper answer-key defect. | Fixed, exposure 99 to 14 PDFs |
| 308 | Answer keys | One PDF binds two separately keyed booklets, cross-attaching answer keys. | Fixed (in 311) |
| 309 | Diagrams | The reliability of the diagram flag needed a corpus-wide check. | Fixed (2 rows), sample check in 312 |
| 310 | Evaluation | Five points from earlier entries were synthesised for the dissertation. | Closed (no new finding) |
| 311 | Answer keys | All 53 rows of the two-booklet PDF needed checking against both keys. | Fixed, 26 rows corrected |
| 312 | Diagrams | A 90-row stratified sample of diagram flags needed image checks. | Closed, false-positive rate 12.2% |
| 313 | Diagrams | Remaining diagram-flag threads from 309 needed closing. | Fixed, 11 flags corrected |
| 314 | Data quality | One PDF's final pages are scanned graded scripts, not question papers. | Accepted limitation (56 rows excluded) |
| 315 | Evaluation | Three knowledge-grounding conditions were compared on the held-out set. | Closed, superseded by 316 |
| 316 | Evaluation | Results in 315 were confounded by a token cutoff. | Fixed (re-run at 600 tokens), no significant difference |
| 317 | Diagrams | No fallback existed when the diagram interpreter is unavailable. | Fixed (degrade path built) |
| 318 | Diagrams | The chart gold-set shortlist needed review against source pages. | Closed, 21-row gold set |
| 319 | Diagrams | The chart-model candidate list needed checking before downloads. | Decided: candidate list, ChartReLA excluded |
| 320 | Diagrams | A template bug could distort VRAM figures, and the scoring schema was undefined. | Fixed, partial-credit scoring adopted |
| 321 | Diagrams | A bake-off run crashed on a renamed transformers class. | Fixed |
| 322 | Hardware | Windows killed a run for low host memory while loading a model. | Fixed (low-memory loading) |
| 323 | Diagrams | A crash in one candidate stopped the remaining candidates. | Fixed (per-candidate subprocess isolation) |
| 324 | Diagrams | A tokenizer warning was blamed for a CUDA device-side assert. | Retracted (incorrect finding) |
| 325 | Hardware | Memory-guard kills stopped a bake-off run five times. | Mitigated (detached overnight runner) |
| 326 | Process | The runner's exit-code detection was broken, causing needless retries. | Accepted limitation (not fixed) |
| 327 | Diagrams | DePlot failed all 21 rows because of a harness data-type bug. | Fixed, re-run completed |
| 328 | Diagrams | granite-docling-258M crashed on 19 of 21 rows when run alone. | Decided: rejected |
| 329 | Diagrams | The driver crashed printing worker output after a successful run. | Fixed (safe printing) |
| 330 | Diagrams | dots.mocr and its SVG variant still require flash_attn. | Decided: single candidate, Qwen3-VL-8B-Instruct |
| 331 | Diagrams | The geometry gold-set shortlist needed review against source pages. | Closed, 22 usable rows |
| 332 | Evaluation | Five gold rows have two-part answers the scoring format cannot handle. | Decided: final answer scored as not applicable |
| 333 | Gate | Unit stripping missed LaTeX unit wrappers and implicit multiplication. | Fixed (bake-off scorer only) |
| 334 | Evaluation | A dry-run scoring check in the project environment raised two warnings. | Fixed |
| 335 | Hardware | The first geometry run ran out of memory on every row with unclamped scans. | Fixed (768px image cap) |
| 336 | Evaluation | Per-row evidence files truncated model output text. | Fixed |
| 337 | Diagrams | Qwen3-VL-8B-Instruct scored 0 of 17 final answers on the geometry gold set. | Decided: rejected |
| 338 | Hardware | Allocator fragmentation stalled long generations on the 8 GB GPU. | Fixed (periodic cache release) |
| 339 | Diagrams | Qwen3-VL-4B-Thinking was tested on the three hardest geometry rows. | Closed, 1 of 3 correct |
| 340 | Diagrams | The geometry exploration needed a stopping point. | Decided: stop, results recorded |
| 341 | Evaluation | The scorer marked correct MCQ option indices as wrong. | Fixed |
| 342 | Diagrams | Qwen2.5-VL-7B-Instruct was run with a 200-token answer budget. | Closed, 1 of 17 correct |
| 343 | Diagrams | Qwen2.5-VL-7B-Instruct was run with the answer budget raised to 600 tokens. | Closed, 6 of 17 defensible |
| 344 | Gate | Unit stripping did not recognise the litre symbol. | Fixed, 3 cached results flipped to pass |
| 345 | Gate | The precompute cache had been unservable since the version bump in 317. | Deferred (live recompute used) |
| 346 | Diagrams | The answer budget was doubled again to 1,200 tokens. | Closed, no further defensible gain |
| 347 | Diagrams | The geometry investigation needed one consolidated account. | Closed |
| 348 | Diagrams | A commercial API model scored 16 of 17 on the same geometry gold set. | Closed (not wired into the app) |
| 349 | QLoRA | An 8B QLoRA fit test was blocked by missing weights and disk space. | Deferred (see 350) |
| 350 | QLoRA | 8B QLoRA training fits in 8 GB only barely, peaking at 7.181 GB. | Decided: not pursued |
| 351 | QLoRA | A LoRA rank and alpha increase was tested against the first adapter. | Closed, null result (58 vs 50, p=0.077) |
| 352 | QLoRA | The trained adapter had never been applied in the live serving path. | Fixed (adapter wired in, un-merged) |
| 353 | QLoRA | The adapter doubled missing final answers in gate solving calls. | Fixed (adapter scoped to hints) |
| 354 | QLoRA | Completion-only loss and double quantization were tested. | Closed, null result (p=0.54) |
| 355 | Evaluation | Forcing a final-answer line at generation time was tested. | Decided: not adopted, gains not significant |
| 356 | Gate | The premise of a weak-mode pass-count experiment did not hold. | Closed (not run) |
| 357 | Evaluation | Shipped hints had not been evaluated for quality. | Closed, Tier 2 scaffolding 3.06 of 5 |
| 358 | Gate | Three Tier 1 prompt candidates were tested against a ship bar. | Decided: none shipped |
| 359 | QLoRA | Adapter and base weights were compared on Tier 2 hint quality. | Closed, null result |
| 360 | Evaluation | Accuracy targets needed stating separately for each question category. | Decided: three category-specific targets |
| 361 | Data quality | Corpus answers needed automatic verification. | Closed, 951 of 5,819 rows verified |
| 362 | Gate | Tier 1 prompt candidates echoed their own instructions. | Fixed (0 of 50 echo), still not shipped |
| 363 | QLoRA | The adapter remained the live default after four null results. | Decided: base weights by default |
| 364 | Serving | A second photo-reader load after a model swap crashed with an access violation. | Mitigated (see 366) |
| 365 | Serving | The swap crash killed the running server process. | Mitigated (see 366) |
| 366 | Serving | The swap crash needed an interim workaround. | Mitigated (503 response and supervised restart) |
| 367 | Process | Docs, scripts and dead code needed a housekeeping pass. | Closed, 2 files removed |
| 368 | Photo input | A pixel overlay for the confirm screen was assessed, and the child study reviewed. | Decided: side-by-side panel, child study descoped |
| 369 | Photo input | Students had no way to edit extracted question text. | Fixed (confirm-and-edit endpoint built) |
| 370 | Serving | Bake-off records were incomplete and the adapter was missing from the cache key. | Fixed (adapter path added to key) |
| 371 | Answer keys | A merge-safety check found worked-solution fragments stored as question text. | Closed, partially verified (fix in 372) |
| 372 | Answer keys | The answer-key label pattern required a Q prefix and missed text labels. | Fixed, 36 rows flagged |
| 373 | Matching | 34 superseded-versus-surviving answer disagreements needed review. | Closed, no action needed |
| 374 | Matching | The status of the 10-pair unlink list was questioned. | Closed, database correct |
| 375 | Process | The project needed a full audit before frontend work. | Closed, no new removable files |
| 376 | Process | The backend needed a structured code review. | Fixed 3 of 4 findings |
| 377 | Serving | Tier 3 hints were not wired and gate data was not persisted. | Fixed |
| 378 | Serving | No live endpoint existed for escalating to a higher hint tier. | Fixed (escalation endpoint, Tier 1 first) |
| 379 | Data quality | A cleanup round covered several remaining data-quality items. | Fixed, 42 rows flagged, 1 answer corrected |
| 380 | Extraction | A bar-model table was swept into grid parsing, deferred from 145. | Fixed (position signal), 1 record changed |
| 381 | Planning | Topic classifier, voice pilot and calibration logging needed scoping. | Closed (scoping only), disk space freed |
| 382 | Serving | The model-swap restart needed documenting as a known limitation. | Accepted limitation |
| 383 | Frontend | Escalation failed for photo attempts and one button led nowhere. | Fixed |
| 384 | Frontend | An expert review of the 14 screens found 11 usability problems. | Fixed 9, 1 accepted, 1 future work |
| 385 | Serving | The first model switch, from the vision model to the language model, crashed the server three times. | Fixed (no switch inside a process; restart after each photo read) |
| 386 | Frontend | After a failure the app returned to the start screen and lost the photo. | Fixed (waits and retries for up to 150 s while the server restarts) |
| 387 | Safety | A Tier 3 hint stated the student's own answer. | Fixed (answer-leak guard on every hint, 8 tests) |
| 388 | Serving | Long hints stopped mid-sentence and ran steps together. | Fixed (more room for Tier 3, unfinished sentences trimmed, one step per line) |
| 389 | Gate | Live gate results for the demo questions differed between runs. | Fixed (gate results precomputed for the 12 demo questions) |
| 390 | Testing | A clean install on a machine without a GPU or the models found three test files that load a model but ran in the no-GPU group, and a timing test that compared a rounded value with an unrounded one. | Fixed (files marked `gpu`, tolerance matched to the rounding); no-GPU tests added to GitHub Actions |
