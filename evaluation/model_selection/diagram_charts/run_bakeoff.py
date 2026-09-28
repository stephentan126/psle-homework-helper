"""
Phase 7 Bucket 1 (Section 6: charts and data) bake-off, copied from evaluation/model_selection/_template/
(the design specification, Issue 50/51/52/53).

Real, current status: CANDIDATES and GOLD_SET_PATH finalized (Issue
319, TinyChart dropped in a follow-up pass, Issue 320). Real runs so far: #1 crashed on an
import-name mismatch (Issue 321), #2 was OS-killed for low host memory (Issue 322), #3
completed granite-vision-3.3-2b cleanly (21/21 rows, real data in per_row_results/) then crashed
granite-docling-258M with a CUDA device-side assert that turned out to be cross-candidate GPU-state
contamination from sharing one process, not a real bug in docling itself (Issue 323, includes
a retraction of an earlier wrong root-cause claim). Fixed in Issue 323 with per-candidate
subprocess isolation (below), matching evaluation/model_selection/ocr_extraction/runners/run_batched.py's own precedent.

Usage pattern (diverges from the template as of Issue 323, read before copying this file for a
new slot):
1. Copy evaluation/model_selection/_template/ to evaluation/model_selection/<slot_name>/ (e.g. evaluation/model_selection/topic_classifier/)
2. List real candidates in CANDIDATES below
3. Point GOLD_SET_PATH at a real, hand-labeled gold set for this slot
4. Fill in load_candidate() and evaluate_candidate() for this specific slot's job
5. Run `python run_bakeoff.py` with NO arguments, this is driver_main(), which spawns one
   isolated subprocess per candidate (`python run_bakeoff.py --candidate <id>`, worker_main())
   rather than looping over all candidates in-process. A candidate whose
   per_row_results/<id>__metrics.json already exists is skipped and its real prior result reused,
   not re-run, real time-saver given a single candidate can legitimately take hours (Issue
   323's own granite-vision-3.3-2b run: ~4h35m). Results write to results.md AND results.csv in
   this folder once every candidate has either run fresh or been skipped as already-done.
6. Fill in the REJECTION REASON column by hand for every candidate not picked, this is required
   evidence for the report, not optional.

`backend/app/services/gpu_utils.py`'s free_gpu() is still called at the end of each candidate's own isolated
worker process (defense in depth), but the REAL protection against cross-candidate GPU-state
contamination is now the subprocess boundary itself (Issue 323), not free_gpu(), a future slot
copying this template should keep the driver/worker split, not revert to the old single-process
loop, unless every one of its own candidates is verified cheap/fast enough that a full re-run costs
nothing and process-isolation overhead genuinely isn't worth it.
"""

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import torch
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "backend"))
sys.path.insert(0, str(_REPO_ROOT / "backend"))
from app.services.gpu_utils import free_gpu, measure_load  # noqa: E402

# Real gate logic reused directly, never reimplemented (this project's own "one real
# implementation, not two" discipline), the MCQ index/value convention (Issue
# 318), the same "FINAL ANSWER: <value>" extraction convention the live gate's own
# self_consistency_check() uses, and the same unit-stripping normalization.
from app.pipeline.gate import (  # noqa: E402
    _canonicalize_mcq_answer, _normalize_latex_expression, _parse_mcq_options,
    extract_claimed_answer, is_mcq_with_unresolved_index, strip_units,
)
import sympy  # noqa: E402
from sympy.parsing.sympy_parser import parse_expr  # noqa: E402

PER_ROW_RESULTS_DIR = Path(__file__).parent / "per_row_results"

# Real, disclosed approximation (not a certified chart-type classifier): keyword match only,
# against the model's own free-text extraction output. A combo gold-set category (e.g.
# "bar_and_pie_mcq", Issue 318) counts as identified if ANY constituent keyword
# appears, reflects that a real page can genuinely show more than one chart type.
_CHART_TYPE_KEYWORDS = {
    "bar": [r"\bbar\s*(chart|graph)\b"],
    "line": [r"\bline\s*(chart|graph)\b"],
    "pie": [r"\bpie\s*(chart|graph)\b"],
    "table": [r"\btable\b"],
}

# Reuses gate.py's own real, already-validated "solve step by step, end with FINAL ANSWER: <value>"
# convention (Issue 36/166's own established prompt shape) rather than inventing a new
# answer-extraction format for this bake-off alone.
_QA_PROMPT_SUFFIX = (
    " Work through this step by step, then end your response with a line in EXACTLY this format: "
    "FINAL ANSWER: <value>. The <value> must be a plain number, fraction, or option number if this "
    "is a multiple-choice question — no units, no words, no extra text on that final line."
)

# ---------------------------------------------------------------------------
# FILL THIS IN per slot
# ---------------------------------------------------------------------------

# Real HF repo IDs, each confirmed to exist and have real downloadable weights before being listed
# here (Issue 319), none downloaded yet, this is a metadata-only confirmation.
CANDIDATES = [
    "ibm-granite/granite-vision-3.3-2b",  # Section 6 Bucket 1 candidate; chosen over
        # granite-vision-4.1-4b (real, exists, 8.01GB vs this one's 5.96GB) for VRAM safety margin
        # on this specific hybrid-graphics laptop (Issue 70), see Issue 319 for
        # the full reasoning. Revisit 4.1-4b only if 3.3-2b's real bake-off accuracy proves
        # insufficient, not as a default upgrade.
    "ibm-granite/granite-docling-258M",  # extraction-only (chart -> structured data/SVG), not a
        # QA model, scored on the first 3 partial-credit stages only (figure detected/chart type
        # identified/values extracted); "final answer exact-match" is structurally N/A for this
        # candidate, not a miss (Issue 320).
    "ahmed-masry/unichart-chartqa-960",  # the QA-FINETUNED variant, not unichart-base-960 (the
        # raw pretrain checkpoint originally listed here in error), swapped Decision
        # Log Issue 320: the base checkpoint is meant for further finetuning, not zero-shot QA,
        # and running it against a QA gold set would not have been a meaningful comparison.
    "google/deplot",  # extraction-only (chart -> linearized table), same scoring scope as
        # granite-docling-258M above, first 3 stages only, final-answer N/A, not chained into a
        # downstream LLM reasoning step for this bake-off (a deliberate, separate decision if ever
        # revisited, not introduced here, Issue 320).
    # "kppkkp/OneChart",  # OPTIONAL, commented out by default, real model, but UNVERIFIED
        # PROVENANCE (Issue 319): kppkkp is not LingyvKong, the real OneChart
        # paper's own author (github.com/LingyvKong/OneChart). Uncomment only for a deliberately
        # broader comparison, with this caveat carried into results.md's own row for it.
]

# ChartReLA (github.com/nxquang-al/ChartReLA, a real, published, peer-reviewed 326M-param
# chart-reasoning VLM) is DELIBERATELY EXCLUDED, not overlooked, real paper and code exist, but
# every downloadable-asset link (2 checkpoint repos, 2 dataset repos) on the linked chart-rela-ins
# HF org returned 404, confirmed directly against the HF API and Hub search, not
# assumed from the README alone. Using it today would mean training from scratch off code with
# no working data-download path either, out of scope for this bake-off. Revisit if the org
# publishes working checkpoint links later; the finding is dated, not permanent. Full detail:
# Issue 319.

# TinyChart-3B-768 (mPLUG/TinyChart) is likewise DELIBERATELY EXCLUDED, not overlooked, same
# treatment as ChartReLA (Issue 320). Real model, real published work, but its
# `TinyChartPhiForCausalLM` architecture is not a native transformers class and its own config's
# auto_map covers only the Phi-2 text backbone, not the vision projector; real usage requires the
# separate X-PLUG/mPLUG-DocOwl GitHub repo's own loader plus a second, companion vision-tower
# model download (TinyChart-3B-768-siglip) with manual config.json wiring, a real, separate
# integration task, not a load_candidate() branch. Its real checkpoint size (~12.76GB, confirmed
# via HF file metadata) also reintroduces the exact VRAM-risk class granite-vision-4.1-4b was just
# deprioritized for. Dated, not permanent, revisit if the integration cost becomes worth paying
# later. Full detail: Issue 320.

GOLD_SET_PATH = Path(__file__).parent / "shortlist_for_review.json"  # the real, reviewed 21-row
    # gold set (24 shortlisted minus 3 excluded, Issue 318), status="excluded"
    # rows must be filtered out by evaluate_candidate(), not deleted from this file.

RESULTS_MD = Path(__file__).parent / "results.md"
RESULTS_CSV = Path(__file__).parent / "results.csv"


def load_candidate(candidate_id: str) -> dict:
    """Loads one real candidate. Returns a bundle dict, not a bare model, this bake-off spans 4
    genuinely different architectures (Issue 320: confirmed via each real
    config.json, not assumed from the model name), so evaluate_candidate() needs to know HOW to
    call each one. `kind` selects the real, per-architecture inference path below; `can_answer_qa`
    records whether this candidate is structurally capable of a final-answer stage at all.
    """
    if candidate_id == "ibm-granite/granite-vision-3.3-2b":
        # FIXED (Issue 321): `AutoModelForVision2Seq` does not exist in this
        # project's pinned transformers==5.12.1 (confirmed directly against the installed
        # package, not assumed), it was renamed to `AutoModelForImageTextToText` upstream. The
        # real bake-off run crashed on this exact line, on the first candidate, before any
        # download or GPU work occurred.
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        processor = AutoProcessor.from_pretrained(candidate_id)
        # Real size 5.96GB already fits under 8GB unquantized, 4-bit NF4 applied anyway as a
        # deliberate extra safety margin on this specific hybrid-graphics machine (Issue 70),
        # the same real risk class Issue 319 picked this model version to avoid.
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        )
        # FIXED (Issue 322): real run #2 was OS-killed for low host memory while
        # this exact call was loading, without low_cpu_mem_usage=True, from_pretrained() stages
        # the full-precision weights in host RAM before the layer-by-layer 4-bit quantize-onto-GPU
        # step, which on this 32GB machine (already carrying real baseline load from Chrome and
        # VS Code, Issue 70) transiently exceeded available memory. This is the
        # standard, documented HF mitigation for quantized loads on memory-constrained hosts, not
        # a guess, not a certainty it prevents every future OOM if concurrent load is high enough.
        model = AutoModelForImageTextToText.from_pretrained(
            candidate_id, quantization_config=quant_config, device_map={"": 0},
            low_cpu_mem_usage=True,
        )
        return {
            "candidate_id": candidate_id, "kind": "chat_vlm", "can_answer_qa": True,
            "model": model, "processor": processor,
        }

    if candidate_id == "ibm-granite/granite-docling-258M":
        # FIXED (Issue 321): same AutoModelForVision2Seq rename as above.
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(candidate_id)
        model = AutoModelForImageTextToText.from_pretrained(
            candidate_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        ).to("cuda")  # real size 0.53GB, no quantization needed; low_cpu_mem_usage=True added for
        # consistency with the vision-3.3-2b fix above (Issue 322), cheap here given the size
        return {
            "candidate_id": candidate_id, "kind": "docling_extractor", "can_answer_qa": False,
            "model": model, "processor": processor,
        }

    if candidate_id == "ahmed-masry/unichart-chartqa-960":
        from transformers import DonutProcessor, VisionEncoderDecoderModel

        processor = DonutProcessor.from_pretrained(candidate_id)
        model = VisionEncoderDecoderModel.from_pretrained(candidate_id).to("cuda")
        # real size 0.815GB, no quantization needed.
        # REAL, DISCLOSED RISK (Issue 320): this model's own real authors
        # (github.com/vis-nlp/UniChart README) pin transformers==4.28.1 and explicitly warn of a
        # real accuracy drop under other versions. This project's pinned transformers==5.12.1 is
        # many versions past that, not re-pinned or re-verified here. If this candidate's real
        # results look anomalously poor, this version gap is the first thing to check, not
        # assumed innocent.
        return {
            "candidate_id": candidate_id, "kind": "unichart_donut", "can_answer_qa": True,
            "model": model, "processor": processor,
        }

    if candidate_id == "google/deplot":
        from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor

        processor = Pix2StructProcessor.from_pretrained(candidate_id)
        model = Pix2StructForConditionalGeneration.from_pretrained(
            candidate_id, torch_dtype=torch.bfloat16,
        ).to("cuda")  # real size 2.26GB, no quantization needed
        return {
            "candidate_id": candidate_id, "kind": "pix2struct_extractor", "can_answer_qa": False,
            "model": model, "processor": processor,
        }

    raise NotImplementedError(
        f"No real load_candidate() branch for {candidate_id!r} — add one rather than guessing a "
        f"generic loader. Every candidate in this bake-off has a real, individually verified "
        f"architecture (Issue 320); a fifth candidate needs the same treatment, "
        f"not an assumed-compatible fallback.",
    )


# =================================================================================================
# Real, per-architecture inference helpers, each matches that model's own real, documented usage
# (verified against its actual model card / GitHub README before being written, development log
# Issue 320), not a generic call pattern forced onto every candidate.
# =================================================================================================


def _generate_chat_vlm(bundle: dict, image_path: str, prompt_text: str, max_new_tokens: int) -> str:
    model, processor = bundle["model"], bundle["processor"]
    messages = [{
        "role": "user",
        "content": [{"type": "image", "url": image_path}, {"type": "text", "text": prompt_text}],
    }]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.decode(
        output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True,
    ).strip()


def _generate_unichart(bundle: dict, image_path: str, task_prompt: str) -> str:
    """Real, verbatim usage pattern from github.com/vis-nlp/UniChart's own README, task-prompt
    tokens (<chartqa>/<extract_data_table>/etc.) fed as decoder_input_ids, not a chat template."""
    model, processor = bundle["model"], bundle["processor"]
    image = Image.open(image_path).convert("RGB")
    decoder_input_ids = processor.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt",
    ).input_ids
    pixel_values = processor(image, return_tensors="pt").pixel_values
    with torch.no_grad():
        outputs = model.generate(
            pixel_values.to(model.device), decoder_input_ids=decoder_input_ids.to(model.device),
            max_length=model.decoder.config.max_position_embeddings, early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id, use_cache=True, num_beams=4,
            bad_words_ids=[[processor.tokenizer.unk_token_id]], return_dict_in_generate=True,
        )
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "")
    sequence = sequence.replace(processor.tokenizer.pad_token, "")
    if "<s_answer>" in sequence:
        sequence = sequence.split("<s_answer>")[-1]
    return sequence.strip()


def _generate_pix2struct(bundle: dict, image_path: str, prompt_text: str, max_new_tokens: int) -> str:
    model, processor = bundle["model"], bundle["processor"]
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=prompt_text, return_tensors="pt").to(model.device)
    # FIXED (Issue 327): real run #4 failed all 21/21 rows with
    # "RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float !=
    # struct c10::BFloat16". Verified directly against this processor's real output (NOT assumed
    # from a generic "pixel_values" name, which doesn't even exist here) before writing this fix:
    # Pix2StructProcessor returns {"flattened_patches": float32, "attention_mask": float32}, not
    # pixel_values, the model itself is loaded in bfloat16, so flattened_patches (the actual
    # vision-patch input to the first matmul) must be cast to match. attention_mask is
    # deliberately left as-is (confirmed working unchanged in an isolated repro before this fix
    # was applied), the dtype mismatch is specific to the patch embedding matmul, not the mask.
    inputs["flattened_patches"] = inputs["flattened_patches"].to(model.dtype)
    with torch.no_grad():
        predictions = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.decode(predictions[0], skip_special_tokens=True).strip()


def _extract_for_candidate(bundle: dict, image_path: str) -> str:
    """Stage 1-3 call: 'read this chart and tell me what's in it', real, per-architecture prompt,
    never the QA question itself (kept separate from _answer_for_candidate() below so an
    extraction-only model's own real designed task is what gets scored, not a QA prompt it was
    never built for)."""
    kind = bundle["kind"]
    if kind == "chat_vlm":
        return _generate_chat_vlm(
            bundle, image_path,
            "Describe the type of chart or diagram shown in this image, and list every data "
            "value it shows.",
            max_new_tokens=300,
        )
    if kind == "docling_extractor":
        # Real model-card instruction verbatim ("Convert this page to docling."). max_new_tokens
        # deliberately reduced from the model card's own 8192 to 2048 to keep a 21-row run
        # tractable, a real, disclosed choice, not silently different from the documented usage;
        # revisit upward if real output looks truncated.
        return _generate_chat_vlm(bundle, image_path, "Convert this page to docling.", max_new_tokens=2048)
    if kind == "unichart_donut":
        return _generate_unichart(bundle, image_path, "<extract_data_table> <s_answer>")
    if kind == "pix2struct_extractor":
        return _generate_pix2struct(
            bundle, image_path, "Generate underlying data table of the figure below:",
            max_new_tokens=512,
        )
    raise NotImplementedError(f"No extraction path for kind={kind!r}")


def _answer_for_candidate(bundle: dict, image_path: str, question_text: str) -> str:
    """Stage 4 call, QA-capable candidates only (bundle['can_answer_qa'] must be checked by the
    caller first), real answer text, NOT yet resolved against the gold value."""
    kind = bundle["kind"]
    if kind == "chat_vlm":
        raw = _generate_chat_vlm(
            bundle, image_path, question_text + _QA_PROMPT_SUFFIX, max_new_tokens=200,
        )
        claimed = extract_claimed_answer(raw)  # gate.py's own real "FINAL ANSWER:" parser
        return claimed if claimed is not None else raw
    if kind == "unichart_donut":
        return _generate_unichart(bundle, image_path, f"<chartqa> {question_text} <s_answer>")
    raise NotImplementedError(f"{kind!r} is not QA-capable — caller must check can_answer_qa first")


def _score_chart_type(extracted_text: str, assumed_chart_type: str):
    text_lower = extracted_text.lower()
    constituents = [c for c in _CHART_TYPE_KEYWORDS if c in assumed_chart_type]
    if not constituents:
        return None  # a gold-set category this keyword set doesn't cover, not silently scored
    return any(
        re.search(pat, text_lower) for c in constituents for pat in _CHART_TYPE_KEYWORDS[c]
    )


def _score_values_extracted(extracted_text: str, gold_answer: str):
    """Real, disclosed approximation: does the extraction contain the gold answer's own key
    number(s) anywhere? The gold set (Issue 318) has a final answer per row, not a full
    structured per-value ground truth, so this is the honest tool available, not a precise
    value-by-value check, documented as such, not presented as more exact than it is."""
    gold_numbers = re.findall(r"-?\d+(?:\.\d+)?", gold_answer or "")
    if not gold_numbers:
        return None
    text_numbers = set(re.findall(r"-?\d+(?:\.\d+)?", extracted_text))
    return any(n in text_numbers for n in gold_numbers)


def _resolve_gold_answer(row: dict) -> str:
    """Real MCQ index/value resolution, reusing gate.py's own established convention (development log
    Issue 318) verbatim, never reimplemented."""
    gold = row.get("corrected_answer_value") or row["stored_answer_value"]
    question_text = row.get("question_text", "")
    if is_mcq_with_unresolved_index(question_text, gold):
        options = _parse_mcq_options(question_text)
        if options:
            return _canonicalize_mcq_answer(gold, options)
        # Empty options dict = image-based MCQ options (ids 3158/3601, Issue 318)
        #, _parse_mcq_options() finds the real (1)-(4) markers but no text after them. Gold stays
        # a bare index; this row's own result also carries image_based_mcq_options=True so a miss
        # here reads as the known structural gap, not an ordinary wrong answer.
    return gold


_NUMERIC_TOKEN_RE = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?%?")


def _score_final_answer(row: dict, model_answer_text: str) -> bool:
    """Real, two-path comparison, caught and fixed by this file's own dry-run test
    (dry_run_scoring_check.py) BEFORE any real GPU run: a plain substring check against a raw gold
    value silently fails for every LaTeX-encoded answer (e.g. id=5096's real gold value is
    literally '\\(\\frac{2}{5}\\)', which never appears verbatim in a model's own free-text "2/5"
    or "0.4"), a systematic false-negative, not a one-off. Two real paths, not one crude check:
    (1) if the gold value parses as a real math expression (after gate.py's own LaTeX
    normalization, reused not reimplemented), try SymPy numeric equality against every numeric
    token found in the model's answer, the same real "compare VALUES, never raw text" principle
    Issue 8 already established for this project's live gate.
    (2) if the gold value does NOT parse as math (a natural-language MCQ option, e.g. id=318's
    real resolved text "Half the computers were sold in Day 1 and Day 2."), fall back to a
    normalized, case-insensitive substring match, the honest tool for comparing prose, not
    dressed up as more rigorous than it is.
    """
    gold_resolved = _normalize_latex_expression(_resolve_gold_answer(row))
    gold_numeric, _ = strip_units(gold_resolved)
    gold_numeric = gold_numeric.strip()

    try:
        gold_expr = parse_expr(gold_numeric.replace(":", "/"))
    except Exception:
        gold_expr = None

    if gold_expr is not None:
        # Real, disclosed robustness fix, caught by this file's own dry-run test: a model can
        # legitimately echo a fraction back in LaTeX notation too (plausible given what these
        # models are trained on), normalize the MODEL's own answer text the same way the gold
        # value already is.
        normalized_answer_text = _normalize_latex_expression(model_answer_text).strip()

        # Path A (primary): parse the WHOLE normalized answer as one expression. By construction
        # this should usually succeed, gate.py's own reused FINAL ANSWER prompt explicitly asks
        # for "a plain number, fraction... no units, no words, no extra text," and UniChart's own
        # <s_answer> convention is a short direct value too, so a fraction like "(2)/(5)" stays
        # ONE expression instead of being torn into unlinked digit tokens the way a bare-token
        # regex scan would (a real, second bug this file's own dry-run test caught and fixed
        # before Path B below was even reached).
        try:
            whole_expr = parse_expr(normalized_answer_text.replace(":", "/"))
            if sympy.simplify(whole_expr - gold_expr) == 0:
                return True
        except Exception:  # noqa: BLE001, real model non-compliance with the requested format,
            # fall through to Path B rather than fail outright
            pass

        # Path B (fallback, looser): the model didn't fully comply with the plain-value format , 
        # scan individual numeric tokens in its longer free-text answer. Real, disclosed limit:
        # this CAN split an un-normalized fraction into two unlinked tokens; Path A above exists
        # specifically to avoid relying on this path for the common case.
        for token in _NUMERIC_TOKEN_RE.findall(normalized_answer_text):
            token_clean = token.rstrip("%")
            try:
                if sympy.simplify(parse_expr(token_clean) - gold_expr) == 0:
                    return True
            except Exception:  # noqa: BLE001, a real, non-numeric-looking token; try the next one
                continue
        return False

    # Non-numeric gold value (real prose, e.g. an MCQ option's own full sentence, or a day name)
    return gold_numeric.lower() in model_answer_text.strip().lower()


def evaluate_candidate(bundle: dict, gold_set_path: Path) -> dict:
    """Runs one real candidate against the real, reviewed 21-row gold set (Issue
    318), scored on the partial-credit schema adopted in Issue 320: figure
    detected / chart type identified / values extracted / final answer exact-match. The 4th stage
    is structurally N/A, not scored as a miss, for extraction-only candidates
    (bundle['can_answer_qa'] is False), Issue 320's own explicit scoping call.
    """
    with open(gold_set_path, encoding="utf-8") as f:
        gold_data = json.load(f)
    rows = [r for r in gold_data["rows"] if r["status"] != "excluded"]
    if len(rows) != 21:
        raise AssertionError(
            f"Expected 21 non-excluded gold rows (24 shortlisted minus 3 excluded, Issue 318), "
            f"got {len(rows)} — the gold set file may have changed; do not silently proceed.",
        )

    can_answer_qa = bundle["can_answer_qa"]
    per_row = []
    for row in rows:
        image_path = row["image_path_abs"]
        gold_answer = row.get("corrected_answer_value") or row["stored_answer_value"]
        result = {"id": row["id"], "assumed_chart_type": row["assumed_chart_type"]}
        if row["assumed_chart_type"] in ("bar_and_pie_mcq", "pie_and_bar_mcq"):
            # Issue 318/320, real structural gap: this row's own MCQ options are
            # chart IMAGES, not text. Carried on every result for this row regardless of outcome,
            # so a wrong answer here reads as the known gap, never a silent ordinary miss.
            result["image_based_mcq_options"] = True

        try:
            extraction_text = _extract_for_candidate(bundle, image_path)
        except Exception as e:  # noqa: BLE001, a real candidate crash on one row must not kill
            # the whole run or silently skip; recorded and moved on, per this project's own
            # fail-loud-not-silent discipline (Issue 210's philosophy, applied here to a
            # bake-off rather than the live gate).
            result.update({
                "figure_detected": False, "chart_type_identified": None,
                "values_extracted": None, "final_answer_exact_match": None,
                "error": f"{type(e).__name__}: {e}",
            })
            per_row.append(result)
            continue

        result["figure_detected"] = bool(extraction_text.strip())
        result["chart_type_identified"] = _score_chart_type(extraction_text, row["assumed_chart_type"])
        result["values_extracted"] = _score_values_extracted(extraction_text, gold_answer)
        result["raw_extraction_text"] = extraction_text[:500]

        if can_answer_qa:
            try:
                answer_text = _answer_for_candidate(bundle, image_path, row["question_text"])
                result["final_answer_exact_match"] = _score_final_answer(row, answer_text)
                result["raw_answer_text"] = answer_text[:300]
            except Exception as e:  # noqa: BLE001
                result["final_answer_exact_match"] = None
                result["error_final_answer"] = f"{type(e).__name__}: {e}"
        else:
            result["final_answer_exact_match"] = "N/A"  # structural, not a miss, Issue 320

        per_row.append(result)

    def _rate(key):
        scored = [r[key] for r in per_row if r.get(key) is not None and r.get(key) != "N/A"]
        return round(sum(1 for s in scored if s) / len(scored), 3) if scored else None

    qa_rows = [r for r in per_row if r.get("final_answer_exact_match") not in (None, "N/A")]

    metrics = {
        "n_rows": len(per_row),
        "figure_detected_rate": _rate("figure_detected"),
        "chart_type_identified_rate": _rate("chart_type_identified"),
        "values_extracted_rate": _rate("values_extracted"),
        "final_answer_exact_match_rate": (
            round(sum(1 for r in qa_rows if r["final_answer_exact_match"]) / len(qa_rows), 3)
            if qa_rows else "N/A (extraction-only candidate, Issue 320)"
        ),
        "n_errors": sum(1 for r in per_row if "error" in r),
    }

    PER_ROW_RESULTS_DIR.mkdir(exist_ok=True)
    detail_path = PER_ROW_RESULTS_DIR / f"{bundle['candidate_id'].replace('/', '__')}.json"
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(per_row, f, indent=2, ensure_ascii=False)
    metrics["per_row_detail_file"] = str(detail_path.relative_to(Path(__file__).parent))

    return metrics


# ---------------------------------------------------------------------------
# Runner, should not need changes per slot
# ---------------------------------------------------------------------------

SUBPROCESS_LOG_PATH = Path(__file__).parent / "subprocess_run_log.jsonl"


def _metrics_path_for(candidate_id: str) -> Path:
    return PER_ROW_RESULTS_DIR / f"{candidate_id.replace('/', '__')}__metrics.json"


def worker_main(candidate_id: str) -> None:
    """Runs exactly ONE real candidate: load, evaluate, persist. Invoked either directly (for a
    single ad-hoc candidate check) or, the real, intended path, as an isolated subprocess spawned
    by driver_main() below (Issue 323).

    FIXED (Issue 323): real run #3 completed granite-vision-3.3-2b cleanly (21/21
    rows, 0 errors), then granite-docling-258M crashed on every single row with the identical CUDA
    "device-side assert triggered" error. A synchronous, isolated repro (CUDA_LAUNCH_BLOCKING=1,
    fresh process, docling alone, same row) found NO crash at all, real generated output, correct
    tokenization (input_ids max 100348, safely inside the real 100352-row embedding table). This
    confirms the crash was real cross-candidate GPU-state contamination from running multiple
    candidates back-to-back in ONE process/CUDA context (the same class of problem as Issue 163,
    just not fully caught by free_gpu()), NOT a bug in docling's own config, and NOT the
    `pad_token_id`/vocab-size warning this file's own comment previously (and wrongly) blamed , 
    that warning's numbers (32000 vocab, token id 128002) don't match EITHER candidate's real
    on-disk config.json, and it's a separate, harmless, pre-existing upstream quirk (traced to
    transformers/configuration_utils.py's `validate_token_ids()`), confirmed still present and
    confirmed NOT fatal in the same isolated, successful repro. See docs/DEVELOPMENT_LOG.md 323 for
    the full retraction of the earlier (wrong) claim and the real evidence.

    The real fix, matching this project's own Job A OCR bake-off precedent (`evaluation/model_selection/ocr_extraction/
    runners/run_batched.py`, Issue 82/83's "a full process exit reclaims everything an in-process
    fix might miss, regardless of the actual internal cause"): each candidate now runs in its own
    fresh OS process via driver_main() below, so one candidate's GPU-state contamination structurally
    cannot reach the next candidate, whatever its unknown deeper cause. Unlike Job A (separate venvs
    per reader, a genuinely different need), all 4 Bucket 1 candidates share one venv, so this file
    plays both roles (driver + worker) via the `--candidate` flag rather than splitting into several
    near-duplicate scripts, since venv separation was Job A's actual reason for separate files, not
    subprocess isolation itself.
    """
    with measure_load(candidate_id) as timing:
        model = load_candidate(candidate_id)

    metrics = evaluate_candidate(model, GOLD_SET_PATH)
    metrics["candidate"] = candidate_id
    metrics["load_time_s"] = round(timing.load_time_s, 2)
    metrics["vram_delta_gb"] = round(timing.vram_delta_gb, 2)
    metrics["fits_8gb"] = timing.vram_delta_gb <= 8.0

    # Real evaluation is already fully computed and about to be persisted BEFORE the del/free_gpu()
    # cleanup below, so if free_gpu() itself raises (e.g. a poisoned CUDA context from THIS
    # candidate's own crash, the exact shape Issue 320/323 both hit), the real, already-computed
    # results are not lost. Only this candidate's own isolated worker process pays for that crash
    # now (Issue 323), not the driver, not any other candidate.
    metrics_path = _metrics_path_for(candidate_id)
    PER_ROW_RESULTS_DIR.mkdir(exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\nWorker done: {candidate_id} -> {metrics_path}")

    try:
        del model
        free_gpu()
    except Exception as e:  # noqa: BLE001, real, disclosed: a crash here no longer costs us the
        # real results (already persisted above), and process exit reclaims GPU memory regardless
        # (gpu_utils.py's own documented behavior), so this is reported, not re-raised.
        print(
            f"free_gpu() raised after this candidate's real results were already saved "
            f"(non-fatal — process exit reclaims GPU memory regardless): {type(e).__name__}: {e}",
        )


def _safe_print(text: str) -> None:
    """FIXED (Issue 329): printing a worker subprocess's captured stdout crashed
    the DRIVER itself with UnicodeEncodeError when that text contained a character the Windows
    console's own cp1252 encoding can't represent (confirmed real: U+FFFD, the Unicode
    replacement character, surfacing from this same subprocess.run() call's own
    errors="replace" decode of the worker's raw bytes), this happened AFTER a worker had
    already succeeded and written its real results, so the crash cost real aggregation work
    (subprocess_run_log.jsonl append, results.md/csv write) for no real reason. Re-encode
    defensively so a console print can never itself crash the driver post-success."""
    encoding = sys.stdout.encoding or "utf-8"
    print(text.encode(encoding, errors="backslashreplace").decode(encoding, errors="replace"))


def driver_main() -> None:
    if not CANDIDATES:
        print("CANDIDATES is empty. Fill it in before running this bake-off.")
        sys.exit(1)

    PER_ROW_RESULTS_DIR.mkdir(exist_ok=True)
    venv_python = sys.executable  # same shared venv for every candidate (Issue 323, unlike
    # Job A's per-reader venvs, all 4 Bucket 1 candidates already share backend/.venv)
    script_path = Path(__file__).resolve()

    all_results = []
    for candidate_id in CANDIDATES:
        metrics_path = _metrics_path_for(candidate_id)
        if metrics_path.exists():
            # Real resumability, same principle as Job A's run_batched.py get_pending_pages()
            # (skip work whose real output already exists), applied at candidate granularity
            # instead of page granularity. Lets an already-clean candidate (e.g.
            # granite-vision-3.3-2b's real ~4h35m, 21/21, 0-error run) be reused across a re-run
            # without re-executing it, per Issue 323.
            print(f"\n=== Skipping {candidate_id} — already-completed metrics found at {metrics_path} ===")
            with open(metrics_path, encoding="utf-8") as f:
                all_results.append(json.load(f))
            continue

        print(f"\n=== Testing {candidate_id} (isolated subprocess) ===")
        t0 = time.time()
        result = subprocess.run(
            [venv_python, str(script_path), "--candidate", candidate_id],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        wall_s = round(time.time() - t0, 2)
        _safe_print(result.stdout)

        record = {
            "candidate": candidate_id, "returncode": result.returncode, "wall_s": wall_s,
            "metrics_produced": metrics_path.exists(),
        }
        if result.returncode != 0:
            _safe_print(
                f"  Candidate {candidate_id} crashed in its own isolated subprocess "
                f"(exit {result.returncode}) — this can no longer take out any other candidate "
                f"(Issue 323). stderr tail:\n{result.stderr[-2000:]}",
            )
            record["stderr_tail"] = result.stderr[-2000:]
        with open(SUBPROCESS_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if metrics_path.exists():
            with open(metrics_path, encoding="utf-8") as f:
                all_results.append(json.load(f))
        else:
            print(
                f"  WARNING: {candidate_id} produced no metrics file — real, disclosed gap, will "
                f"be missing from results.md/csv below, not silently treated as excluded.",
            )

    write_results(all_results)
    print(f"\nResults written to {RESULTS_MD} and {RESULTS_CSV}")
    print("Now fill in the rejection reason for every candidate NOT selected, by hand, in")
    print("results.md — this is required report evidence, not optional (Issue 50).")


def write_results(results: list[dict]) -> None:
    if not results:
        return
    fieldnames = list(results[0].keys()) + ["role", "rejection_reason"]
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            row.setdefault("role", "TBD: primary/secondary/rejected")
            row.setdefault("rejection_reason", "TBD: fill in by hand if not selected")
            writer.writerow(row)

    with open(RESULTS_MD, "w") as f:
        f.write("# Bake-off Results\n\n")
        f.write("Fill in `role` and `rejection_reason` for every row before this counts as\n")
        f.write("report-ready evidence (the design specification Issue 50).\n\n")
        f.write("| " + " | ".join(fieldnames) + " |\n")
        f.write("|" + "---|" * len(fieldnames) + "\n")
        for row in results:
            row.setdefault("role", "TBD")
            row.setdefault("rejection_reason", "TBD")
            f.write("| " + " | ".join(str(row.get(k, "")) for k in fieldnames) + " |\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--candidate", default=None,
        help="Internal: run exactly one candidate in worker mode (Issue 323) — "
             "this is how driver_main() achieves per-candidate subprocess isolation, matching "
             "evaluation/model_selection/ocr_extraction/runners/run_batched.py's own pattern. Not intended for direct "
             "interactive use, though harmless if passed a valid CANDIDATES entry by hand.",
    )
    args = parser.parse_args()
    if args.candidate:
        worker_main(args.candidate)
    else:
        driver_main()
