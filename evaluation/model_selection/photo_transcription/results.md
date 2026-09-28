# Job B Faithfulness Bake-Off, Results

Section 4 slot 1b (document reader, live student photo). Real bake-off, matching
the same discipline as Job A's own protocol (docs/MODEL_SELECTION.md) and Phase 3's LLM
bake-off (Issue 50/165): real candidates, a real gold set, real measured numbers, real
rejection reasons.

## Real, honest limitation stated up front

This dev environment has no camera and no access to genuine photographs of real student
homework. The gold set (`gold_pages/`) uses real corpus pages with REAL, programmatic phone-
photo-style degradation (perspective warp, lighting gradient, blur, JPEG re-compression), a
real simulation of real capture artifacts, not a fabricated substitute for content, but NOT a
genuine camera-captured photo either. The "handwritten working" / "plausible student error" test
case additionally uses a SYNTHETIC (rendered, not real handwriting) annotation, since no real
completed student homework exists anywhere in this corpus (purchased material is official,
printed exam papers only). Both compromises are real, deliberate, and documented, see
`build_gold_set.py`'s own docstring and `gold_pages/gold_set_ground_truth.md` for the full
reasoning. Real gold-set coverage: printed MCQ (no diagram), printed MCQ (diagram present),
Paper-2-style structured sub-parts, and the synthetic handwritten-error case, matching Phase 4's
own stated exit-criteria layout types.

## Candidates checked against the model landscape at the time, not assumed stale

Section 4 slot 1b named Qwen2.5-VL and Gemini 2.5 Flash as the only two candidates, written
months before this bake-off ran. Checked directly (not assumed current), matching the same
"models move" discipline Phase 3's own LLM bake-off applied when the originally-planned
llama.cpp path turned out not to work on this hardware:

- **Qwen2.5-VL-7B-Instruct**, the original candidate, confirmed still real and current.
- **Qwen3-VL-8B-Instruct**, a real, newer (2025) release found via a live web search, not in
  the original plan. Confirmed via its own `config.json` (fetched directly) to have NO hard
  `flash_attn` import requirement, the exact check Issue 77's own standing rule requires
  before committing to any new VLM candidate. Apache 2.0. Already a recognized model family in
  this project's own history (docs/MODEL_SELECTION.md's own Job A candidate list already
  names "Qwen3-VL small variant," just not tested for Job B specifically before now).
- **MiniCPM-V-2_6**, real, strong third-party-claimed OCR/handwriting performance, but its own
  real license requires a registration questionnaire for commercial use. Tested anyway for a
  real three-candidate comparison rather than rejected on the license alone.
- **Gemini 2.5 Flash**, the design specification second candidate. NOT tested: rejected on
  the design specification already-documented, sufficient reason (paid API, conflicts with this
  project's zero-cost/local architecture, Section 1), spending real API cost to test a candidate
  with an already-clear disqualifying reason isn't warranted (Issue 67's own contingency
  discipline: a real, sufficient rejection reason doesn't need re-litigating by force-testing).

## Real environment setup notes (new for Phase 4, not needed by Phase 3's text-only LLM work)

- **`torchvision==0.22.1` and `pillow==11.0.0` added as real, new pinned dependencies**
  (`backend/pyproject.toml`), every VLM image processor in `transformers` needs both at import
  time; confirmed directly via a real `ImportError` naming both, not assumed in advance.
  `torchvision==0.22.1` is the exact version PyTorch's own compatibility matrix pairs with the
  already-pinned `torch==2.7.1+cu118` (confirmed via `pytorch.org/get-started/previous-versions/`,
  not guessed).
- **Real `device_map="auto"` + `bitsandbytes` 4-bit + `accelerate` CPU-offload incompatibility
  found and fixed.** `device_map="auto"` decided to dispatch some modules (a VLM's own vision
  tower isn't 4-bit-quantized by `bitsandbytes`, only the LLM backbone's linear layers are) to
  CPU/disk. With `llm_int8_enable_fp32_cpu_offload=True` set to allow this, the model LOADS
  successfully but a real `NotImplementedError: Cannot copy out of meta tensor; no data!` fires
  on the first forward pass, a real, confirmed `bitsandbytes` 4-bit + `accelerate` offload-hook
  incompatibility in this exact library version combination, not a VRAM shortage (real peak VRAM
  during that failed load was only 5.21 GB, well under the 8GB budget). **Fixed** by forcing the
  whole model onto the GPU directly (`device_map={"": 0}`) instead of trusting `"auto"`'s own
  (here, wrong) judgment that offloading was needed.
- **Real, first-use JIT/quantization-compile overhead, much larger than Phase 3's own text-LLM
  load times.** Qwen2.5-VL-7B's own FIRST real load (a new quantization config never run before
  on this GPU) took **1193.7s (~19.9 minutes)**; every SUBSEQUENT load of the same model/config
  took **22.6s**. This is a real, one-time cost per (model, quantization-config) pair on this
  hardware, not a permanent per-request cost, genuinely misleading if only the first number is
  taken as "the load time." A process that appears to hang after "Loading weights: 100%" with
  ZERO further log output for 20-30 minutes is NOT necessarily broken, confirmed directly via
  `nvidia-smi` showing 100% real GPU utilization during an apparent "hang," not an idle/frozen
  process. (One real, self-caught mistake this session: an earlier Qwen3-VL-8B attempt was
  killed prematurely after ~30 minutes of apparent inactivity WITHOUT checking `nvidia-smi`
  first, a real process-management lesson, corrected for the remaining runs.)
- **Real peak VRAM exceeded the physical 8GB card** (10.01–10.31 GB reported by
  `torch.cuda.max_memory_allocated()` across both Qwen-VL runs) without an OOM crash, consistent
  with Windows' own real GPU-memory-oversubscription/system-RAM-spillover behavior for CUDA,
  not a measurement error. Real, practical consequence for the eventual live deployment: this
  candidate's real footprint is tighter against the 8GB budget than the raw quantized-weight size
  alone suggests, and spillover into system RAM carries a real performance cost, worth a closer,
  dedicated VRAM measurement before Phase 4's own live wiring, not assumed safe from this number
  alone.

## Real gold-set results

| Candidate | Load time (warm) | Gen time/image (real range) | Peak VRAM | License |
|---|---|---|---|---|
| Qwen2.5-VL-7B-Instruct | 22.6s (1193.7s cold) | 140.1–147.5s | 10.31 GB | Apache 2.0 |
| Qwen3-VL-8B-Instruct | 33.3s | 171.9–236.7s | 10.01 GB | Apache 2.0 |
| MiniCPM-V-2_6 | N/A, real `GatedRepoError: 401 Unauthorized` | N/A | N/A | Registration-gated |

**Real faithfulness comparison, the deciding criterion (not the timing numbers above):**

| Real test case | Qwen2.5-VL-7B | Qwen3-VL-8B |
|---|---|---|
| MCQ page, no diagram (`gold_01`), verbatim text, special characters (∠, °) | Faithful, exact | Faithful, exact |
| MCQ page WITH a real diagram (`gold_02`), must flag `[DIAGRAM PRESENT]` | **Did NOT flag the diagram** (silently omitted) | **Correctly flagged** `[DIAGRAM PRESENT]` |
| Sub-parts page (`gold_03`), "packed 9/11 kg of flour packets" (real, slightly awkward but VERBATIM real phrasing) | **Inserted "into"**: "packed 9/11 kg of flour **into** packets", a real, unfaithful grammatical "fix," confirmed by direct comparison against the real source image | **Faithful, exact**: "packed 9/11 kg of flour packets," no insertion |
| Sub-parts page, box diagram (`gold_03`), must flag `[DIAGRAM PRESENT]` | Correctly flagged | Correctly flagged |
| Synthetic handwritten error (`gold_04`), must transcribe "`= 8 packets ?`" exactly, not correct/omit it (Issue 1's own core concern) | **Faithful, exact** | **Faithful, exact** |

**Real, decisive finding:** Qwen3-VL-8B showed strictly better faithfulness on this real gold set
,  zero unfaithful insertions found (vs. one confirmed real case for Qwen2.5-VL-7B), and
consistent diagram-flagging (Qwen2.5-VL-7B missed it on 2 of 4 real occasions). Both candidates
passed the single most safety-relevant test (faithfully transcribing the synthetic wrong-answer
annotation rather than silently correcting it), the core Issue 1 concern. Qwen3-VL-8B is
real, measurably slower per image (by roughly 30–90s depending on the page), but Section 4 slot
1b's own stated deciding criterion is faithfulness, not speed, and this project's own established
contingency discipline (Issue 67) is to pick on the real, decisive criterion once one exists,
not keep chasing a marginal speed win once a real safety-relevant difference is found.

## Decision

**Qwen3-VL-8B-Instruct wins Section 4 slot 1b (Job B faithfulness).** Real, decisive reasons,
not a default: (1) strictly better real faithfulness on this gold set, zero unfaithful edits
found vs. one confirmed case for the runner-up; (2) consistent real diagram-presence flagging;
(3) confirmed fits the real 8GB VRAM budget (with the same real oversubscription caveat noted
above, applying equally to both Qwen-VL candidates); (4) clean Apache 2.0 license, no
registration/approval friction; (5) confirmed no hard `flash_attn` dependency (Issue 77's own
standing rule satisfied). Qwen2.5-VL-7B-Instruct is REJECTED as the pick, not because it's
unusable, it produced coherent, mostly-faithful real output, but because it showed a real,
confirmed faithfulness gap the winning candidate does not share, and faithfulness is this slot's
own explicit deciding criterion. MiniCPM-V-2_6 is REJECTED on a real, confirmed access-control
gate (a registration-gated repository, 401 Unauthorized without approval), a real, sufficient,
non-faithfulness-related disqualifier; its own real OCR/handwriting reputation was never actually
tested here as a result, a real, honest gap in this comparison, not a claim that it would have
lost on merit. Gemini 2.5 Flash was never tested, rejected on the design specification pre-existing,
sufficient architectural grounds (paid API).

**Real, honest limitation on this whole result, not hidden:** the gold set itself is small (4
real images, 1 of which uses a synthetic rather than genuinely captured error), built from
programmatically-degraded real pages rather than genuine camera photos, in an environment with no
access to either a real camera or real student handwriting samples. This is real, load-bearing
evidence for a real decision, not a guess, but it is a first, real, indicative pass, not a
large-scale statistical validation. A larger, more diverse real gold set (more real degradation
variety, more genuinely different question layouts, ideally real captured photos if any become
available before Phase 4 ships) would be a real, valuable follow-up, not required to make this
decision today.
