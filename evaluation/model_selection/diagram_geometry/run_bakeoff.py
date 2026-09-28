"""
Phase 7 Bucket 2 (Section 6: simple geometry with stated measurements) bake-off.

Real, current status: single-candidate scope decided (Issue 330),
dots.mocr and its SVG-specialized sibling dots.mocr-svg are both blocked on this machine by the
identical flash_attn hard-dependency already rejected for Job A's OCR bake-off (Issue 77),
re-confirmed live against each model's current vision-tower source. Qwen3-VL-8B-Instruct proceeds
as the sole candidate, Section 15 Step 2's own named fallback, already Job B's faithfulness
bake-off winner (evaluation/model_selection/photo_transcription/results.md), already confirmed flash_attn-clean and
loadable on this exact hardware (evaluation/model_selection/photo_transcription/load_test_qwen3_vl_8b.py).

Real, hand-verified 24-row gold-set shortlist built and reviewed (Issue 331):
11 clean, 11 corrected, 2 excluded, 22 rows usable.

Structure and discipline copied deliberately from evaluation/model_selection/diagram_charts/run_bakeoff.py
(driver/worker subprocess isolation per candidate, GPU-free dry-run scoring validation before any
real GPU time, gate.py's own real scoring functions reused rather than reimplemented) so the
write-up stays consistent with the rest of the report, per this project's own established pattern.
Kept as a single-candidate harness rather than trimmed to a flat script, since a second candidate
(a future geometry-capable model, or a revisit if the flash_attn blocker ever lifts) should be able
to slot into CANDIDATES/load_candidate() the same way Bucket 1's four candidates did.

REAL FINDING made while building this harness, before any GPU time was spent (development log
Issue 332): 5 of the 22 usable gold rows (ids 4356, 1568, 2657, 3905, 1304) have a genuinely
compound gold answer, "(a) <value> (b) <value>", because the source question itself asks for
two sub-answers, not one. Bucket 1's gold set never hit this (every chart question resolved to one
final value), so its final-answer scoring logic assumes a single scalar/expression throughout.
Forcing gate.py's own single-value "FINAL ANSWER: <value>" prompt convention onto a two-part
question doesn't make sense, the model has no honest way to comply. Rather than silently scoring
these as ordinary misses (which would understate the candidate, not reflect a real reasoning
failure) or inventing a new, unvalidated compound-answer prompt format on the fly, these rows are
flagged (`multi_part_gold_answer: True`) and `final_answer_exact_match` is recorded as structurally
N/A for them, same disclosed-scoping-decision treatment Issue 320 already established for
extraction-only candidates in Bucket 1, not a new precedent invented here. One of the two sub-parts
on id=4356 ("(b) not captured") was never recoverable during the Issue 331 gold-set review
either, a genuine source-material gap, not guessed at, per Issue 76. figure_detected,
geometry_type_identified, and values_extracted are still scored normally on these 5 rows; only the
final-answer stage is affected.

Usage pattern (unchanged from Bucket 1, Issue 323):
1. Fill in / adjust CANDIDATES and GOLD_SET_PATH if this bake-off's scope ever changes.
2. Run `python run_bakeoff.py` with NO arguments, this is driver_main(), which spawns one
   isolated subprocess per candidate. A candidate whose per_row_results/<id>__metrics.json already
   exists is skipped and its real prior result reused, not re-run.
3. Fill in the REJECTION REASON / ROLE columns by hand for results.md once real numbers exist.

Diagram buckets: Bucket 1 is charts and data, Bucket 2 is simple geometry with measurements,
and Bucket 3 is bar models and number lines.
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

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "backend"))
sys.path.insert(0, str(_REPO_ROOT / "backend"))
from app.services.gpu_utils import free_gpu, measure_load  # noqa: E402

# Real gate logic reused directly, never reimplemented, same discipline as Bucket 1
# (run_bakeoff.py), the same MCQ index/value convention, the same "FINAL ANSWER: <value>"
# extraction convention the live gate's own self_consistency_check() uses, and the same
# unit-stripping normalization.
from app.pipeline.gate import (  # noqa: E402
    _canonicalize_mcq_answer, _normalize_latex_expression, _parse_mcq_options,
    extract_claimed_answer, is_mcq_with_unresolved_index, strip_units,
)
import sympy  # noqa: E402
from sympy.parsing.sympy_parser import (  # noqa: E402
    implicit_multiplication_application, parse_expr, standard_transformations,
)

# Issue 333: id=1663's real gold value normalizes to "6pi + 12", plain parse_expr()
# rejects "6pi" outright (no implicit-multiplication support by default), a real gap
# Bucket 1's own chart gold set never hit since it had no pi-based geometry answers.
# Standard, built-in sympy transformation, not a hand-rolled workaround.
_PARSE_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)


def _parse_math_expr(text: str):
    return parse_expr(text, transformations=_PARSE_TRANSFORMATIONS)

PER_ROW_RESULTS_DIR = Path(__file__).parent / "per_row_results"

# Real, disclosed approximation (not a certified classifier), matching Bucket 1's own
# _CHART_TYPE_KEYWORDS pattern: keyword match against the model's own free-text extraction.
# `assumed_geometry_subtype` values found in the real Issue 331 gold set are "angle",
# "area_perimeter", and the combo "angle_and_area_perimeter", a combo category counts as
# identified if ANY constituent keyword appears (same convention as Bucket 1's combo categories).
_GEOMETRY_TYPE_KEYWORDS = {
    "angle": [r"\bangle(s)?\b", r"∠"],
    "area_perimeter": [r"\barea\b", r"\bperimeter\b"],
}

# Reuses gate.py's own real, already-validated "solve step by step, end with FINAL ANSWER: <value>"
# convention (Issue 36/166's own established prompt shape), same as Bucket 1.
_QA_PROMPT_SUFFIX = (
    " Work through this step by step, then end your response with a line in EXACTLY this format: "
    "FINAL ANSWER: <value>. The <value> must be a plain number, fraction, or option number if this "
    "is a multiple-choice question -- no units, no words, no extra text on that final line."
)

# Real regex for a compound "(a) ... (b) ..." gold answer (Issue 332), matches
# the exact shape Issue 331's own gold-set review produced for genuinely two-part questions.
_MULTI_PART_RE = re.compile(r"\(([a-z])\)\s*([^()]*?)(?=\s*\([a-z]\)|$)", re.IGNORECASE)

# ---------------------------------------------------------------------------
# FILL THIS IN per slot
# ---------------------------------------------------------------------------

CANDIDATES = [
    "Qwen/Qwen3-VL-8B-Instruct",  # Section 15 Step 2's own named fallback for Bucket 2. Already
        # the Job B faithfulness bake-off winner (evaluation/model_selection/photo_transcription/results.md, real
        # decisive faithfulness advantage over Qwen2.5-VL-7B-Instruct), already confirmed no hard
        # flash_attn dependency (Issue 77's standing rule satisfied), already load-tested on
        # this exact hardware (evaluation/model_selection/photo_transcription/load_test_qwen3_vl_8b.py, real peak VRAM
        # 10.01-10.31GB reported via torch.cuda.max_memory_allocated() with real oversubscription
        # accepted, not a crash). dots.mocr / dots.mocr-svg deliberately excluded, not overlooked
        # both re-confirmed blocked by Issue 77's flash_attn issue, Issue 330.
]

GOLD_SET_PATH = Path(__file__).parent / "shortlist_for_review.json"  # the real, reviewed 22-row
    # gold set (24 shortlisted minus 2 excluded, Issue 331), status="excluded"
    # rows must be filtered out by evaluate_candidate(), not deleted from this file.

RESULTS_MD = Path(__file__).parent / "results.md"
RESULTS_CSV = Path(__file__).parent / "results.csv"


def load_candidate(candidate_id: str) -> dict:
    """Loads the real Bucket 2 candidate. Returns a bundle dict, matching Bucket 1's own
    load_candidate() contract, even though this bake-off currently has only one candidate,
    keeps the shape consistent if a second candidate is ever added."""
    if candidate_id == "Qwen/Qwen3-VL-8B-Instruct":
        # Exact real, already-verified loading pattern from
        # evaluation/model_selection/photo_transcription/load_test_qwen3_vl_8b.py, not re-derived from scratch.
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

        # REAL FINDING (Prompt 7 investigation): Job B's own proven-working
        # single-generation-call QA case resolved to ~2.77MP input and 2,728 vision tokens
        # on this same 8GB card. This bake-off's real gold-set page scans run ~8.7MP, a
        # 3.15x pixel gap that maps almost exactly onto vision-token count (8,580 tokens for
        # the 8.7MP case vs Job B's 2,728), confirming the OOM here is a resolution/token-
        # budget problem, not a quantization or model-choice problem. Neither image was ever
        # being clamped, Qwen3-VL's shipped default processor ceiling (~16.78MP) is far
        # more generous than this 8GB card can actually afford, and this harness's own two
        # generation calls per row (extraction stage + answer stage, vs Job B's one) doubles
        # the real per-row vision-token cost on top of that. Fix: cap input resolution
        # explicitly via the processor's own documented, checkpoint-supported max_pixels
        # constructor kwarg (a supported override, not a workaround).
        #
        # REAL, VALIDATED VALUE (same-day follow-up investigation): the first target,
        # 28*28*512 = 401,408 px (~1,024 tokens), was too aggressive, a real, confirmed 2x
        # misread on id=2503's real diagram (radii read as 3.5cm/1.75cm against the real printed
        # 7cm/3.5cm, verified by direct image inspection). A resolution ablation isolating input
        # size as the sole variable (same model, same prompt, only max_pixels changed) showed the
        # misread was resolution-dependent, not model reasoning noise: 512 -> wrong on both radii,
        # 768 -> exactly correct on both, 1024 -> correct on the big radius, off on the small one.
        # 28*28*768 = 602,112 px (~1,200 tokens) is the real, validated setting, confirmed
        # against THREE independently ground-truthed images (id=3, id=2503, id=4330, each read
        # directly off the real source page, not assumed), matching the real printed values in
        # every case. Real measured peak VRAM at this setting: ~6.6-6.7GB (single-row and
        # two-call-per-row checks both), comfortably inside the 8GB card, also confirmed
        # reproducible (byte-for-byte identical output across two runs of identical inputs, ruling
        # out GPU non-determinism as a confound in these comparisons). A separate, real bottleneck
        # (the answer call not converging to a stated final value on harder multi-step rows, e.g.
        # id=2503/id=4330) was found and tested up to max_new_tokens=600 and an alternate
        # brevity-instructing prompt suffix, neither fixed it, so it is NOT resolution-related
        # and is left as a known, disclosed, separate limitation, not folded into this fix.
        processor = AutoProcessor.from_pretrained(
            candidate_id, min_pixels=28 * 28 * 4, max_pixels=28 * 28 * 768,
        )
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            candidate_id, quantization_config=quant_config, device_map={"": 0},
            attn_implementation="sdpa",  # confirmed no hard flash_attn dependency (Issue 77);
                                          # SDPA is built into PyTorch, no extra install needed,
                                          # same real choice Job B's own load test already made.
            # low_cpu_mem_usage=True added here (not present in Job B's original load-test script)
            # per Bucket 1's own real, evidenced Issue 322 fix: without it, from_pretrained()
            # stages full-precision weights in host RAM before the layer-by-layer 4-bit quantize
            # step, which caused a real OS-level low-host-memory kill on this same 32GB machine
            # during Bucket 1's own bake-off. Carried forward as a preventive measure, not because
            # Job B's own run hit this specific failure.
            low_cpu_mem_usage=True,
        )
        return {
            "candidate_id": candidate_id, "kind": "chat_vlm", "can_answer_qa": True,
            "model": model, "processor": processor,
        }

    raise NotImplementedError(
        f"No real load_candidate() branch for {candidate_id!r} -- add one rather than guessing a "
        f"generic loader, matching Bucket 1's own per-candidate verification discipline "
        f"(Issue 320).",
    )


def _generate_chat_vlm(bundle: dict, image_path: str, prompt_text: str, max_new_tokens: int) -> str:
    """Real Qwen3-VL chat-template usage, verbatim from
    evaluation/model_selection/photo_transcription/load_test_qwen3_vl_8b.py, note the message content key is
    "image", not "url" (Bucket 1's granite-vision-3.3-2b used "url", a real, confirmed
    per-architecture difference, not interchangeable)."""
    model, processor = bundle["model"], bundle["processor"]
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt_text},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.decode(
        output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True,
    ).strip()


def _extract_for_candidate(bundle: dict, image_path: str) -> str:
    """Stage 1-3 call: 'read this figure and tell me what's in it', real extraction prompt,
    never the QA question itself, same separation Bucket 1 uses."""
    return _generate_chat_vlm(
        bundle, image_path,
        "Describe the type of geometric figure shown in this image (for example: angle, "
        "triangle, rectangle, parallelogram, trapezium, circle, semicircle, cuboid), and list "
        "every measurement or labelled value it shows (lengths, angles, or areas).",
        max_new_tokens=300,
    )


def _answer_for_candidate(bundle: dict, image_path: str, question_text: str) -> str:
    """Stage 4 call, real answer text, NOT yet resolved against the gold value."""
    raw = _generate_chat_vlm(
        bundle, image_path, question_text + _QA_PROMPT_SUFFIX, max_new_tokens=200,
    )
    claimed = extract_claimed_answer(raw)  # gate.py's own real "FINAL ANSWER:" parser
    return claimed if claimed is not None else raw


def _score_geometry_type(extracted_text: str, assumed_geometry_subtype: str):
    text_lower = extracted_text.lower()
    constituents = [c for c in _GEOMETRY_TYPE_KEYWORDS if c in assumed_geometry_subtype]
    if not constituents:
        return None  # a gold-set category this keyword set doesn't cover, not silently scored
    return any(
        re.search(pat, text_lower) for c in constituents for pat in _GEOMETRY_TYPE_KEYWORDS[c]
    )


def _score_values_extracted(extracted_text: str, gold_answer: str):
    """Same real, disclosed approximation as Bucket 1: does the extraction contain the gold
    answer's own key number(s) anywhere? Not a precise value-by-value check."""
    gold_numbers = re.findall(r"-?\d+(?:\.\d+)?", gold_answer or "")
    if not gold_numbers:
        return None
    text_numbers = set(re.findall(r"-?\d+(?:\.\d+)?", extracted_text))
    return any(n in text_numbers for n in gold_numbers)


def _resolve_gold_answer(row: dict) -> str:
    """Real MCQ index/value resolution, reusing gate.py's own established convention (development log
    Issue 318) verbatim, identical to Bucket 1, never reimplemented."""
    gold = row.get("corrected_answer_value") or row["stored_answer_value"]
    question_text = row.get("question_text", "")
    if is_mcq_with_unresolved_index(question_text, gold):
        options = _parse_mcq_options(question_text)
        if options:
            return _canonicalize_mcq_answer(gold, options)
    return gold


def _is_multi_part_gold(gold_str: str) -> bool:
    """Real, new check (Issue 332): detects a genuinely compound "(a) ... (b) ..."
    gold answer, present on 5 of this gold set's 22 usable rows (4356, 1568, 2657, 3905, 1304),
    a real shape Bucket 1's own chart gold set never had to handle."""
    return len(_MULTI_PART_RE.findall(gold_str or "")) >= 2


_NUMERIC_TOKEN_RE = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?%?")

# REAL, NEW FINDING (Issue 333), caught by this harness's own dry-run check
# before any GPU time was spent, same discovery method as Bucket 1's id=5096 LaTeX-fraction bug
# (Issue 320), on a different real defect. gate.py's own strip_units() (reused verbatim, never
# reimplemented) does not strip a LaTeX `\mathrm{...}` unit wrapper, id=1663's real gold value
# resolves to `\((6\pi + 12)\mathrm{cm}\)`, a semicircle-perimeter MCQ option Bucket 1's own
# chart-only gold set never had a case of. `_normalize_latex_expression()` correctly turns `\pi`
# into `pi`, but `strip_units()` leaves `\mathrm{cm}` attached, so `parse_expr()` fails on the
# whole string. Real, disclosed, LOCAL fix here (bake-off scoring code only) rather than a silent
# edit to gate.py's own strip_units(), a shared production function with a wider blast radius:
# investigate separately whether the live gate ever produces/consumes a `\mathrm{}`-wrapped MCQ
# option in real production traffic before deciding whether this same gap needs fixing there too
# not assumed either way here.
_LATEX_MATH_WRAPPER_RE = re.compile(r"\\(?:mathrm|text)\{([^{}]*)\}")
_LATEX_EXPONENT_UNIT_RE = re.compile(r"(?<=[a-zA-Z])\^\{?-?\d+\}?")


def _strip_latex_math_wrappers(text: str) -> str:
    r"""Real, disclosed local cleanup (Issue 333) for LaTeX measurement artifacts strip_units()
    does not itself handle: a `\mathrm{}`/`\text{}` unit wrapper, a stray `~` (LaTeX tie/
    non-breaking space, e.g. id=5293's real `\mathrm{~cm}`), and a caret-brace unit exponent
    (e.g. `cm^{3}`), the exponent is dropped entirely, not preserved, since only the leading
    numeric/algebraic value is compared here, never the unit's own dimensionality."""
    text = _LATEX_MATH_WRAPPER_RE.sub(r"\1", text)
    text = text.replace("~", "")
    text = _LATEX_EXPONENT_UNIT_RE.sub("", text)
    return text


def _score_final_answer(row: dict, model_answer_text: str) -> bool:
    """Real, two-path comparison, identical logic to Bucket 1's own dry-run-validated
    _score_final_answer(), reused, not reimplemented, since this comparison logic is not
    chart-specific. Caller must check `_is_multi_part_gold()` first (Issue 332) and route
    multi-part rows to the N/A treatment in evaluate_candidate() instead of calling this."""
    # REAL BUG FIXED (Issue 341): this function used to compare ONLY
    # against the gold answer RESOLVED to its option text (e.g. id=3: "3" -> "106°"), so an MCQ
    # answer given as the bare option index, exactly what _QA_PROMPT_SUFFIX asks for ("...or
    # option number if this is a multiple-choice question"), scored False even when correct,
    # while the resolved text ("106"/"106°") scored True. Found when Qwen2.5-VL-7B answered id=3
    # correctly ("FINAL ANSWER: 3") and was marked wrong. Fix: for a row whose stored gold is an
    # unresolved MCQ option index, ALSO accept the model's answer if it is that same index
    # (optionally wrapped in parentheses, e.g. "(3)"); matching the resolved option text still
    # works exactly as before. Disclosed limit: if a model's answer is the option's VALUE and that
    # value happens to equal the correct option's own index, it also scores True, unavoidable
    # while the prompt itself accepts either form. Did NOT affect the Qwen3-VL-8B-Instruct 22-row
    # result (none of its 17 scored answers reached a FINAL ANSWER line).
    gold_raw = (row.get("corrected_answer_value") or row["stored_answer_value"]).strip()
    if is_mcq_with_unresolved_index(row.get("question_text", ""), gold_raw):
        if model_answer_text.strip().strip("()").strip() == gold_raw:
            return True

    # Issue 333: unwrap `\mathrm{...}`/`\text{...}` BEFORE strip_units(), not after, when
    # a unit is still wrapped, strip_units()'s own suffix match never fires at all (confirmed via
    # this harness's own dry run: id=1663's real `\mathrm{cm}` was left fully attached either way
    # until the unwrap happened first), leaving the unit text stuck to the expression regardless
    # of unwrap order tried second.
    gold_resolved = _strip_latex_math_wrappers(_normalize_latex_expression(_resolve_gold_answer(row)))
    gold_numeric, _ = strip_units(gold_resolved)
    gold_numeric = gold_numeric.strip()

    try:
        gold_expr = _parse_math_expr(gold_numeric.replace(":", "/"))
    except Exception:
        gold_expr = None

    if gold_expr is not None:
        # Symmetric normalization (Issue 333, same principle Bucket 1 already established):
        # a model could plausibly echo a \mathrm{}-wrapped unit back too. Unwrap BEFORE
        # strip_units() is even relevant here, model_answer_text is expected to already be a
        # bare value per the QA prompt, so no strip_units() call happens on this side; the unwrap
        # alone (turning e.g. "162\mathrm{cm}" into "162cm") is enough for the numeric-token scan
        # below to still find the real number, same as it already tolerates a bare "162cm".
        normalized_answer_text = _strip_latex_math_wrappers(
            _normalize_latex_expression(model_answer_text),
        ).strip()
        try:
            whole_expr = _parse_math_expr(normalized_answer_text.replace(":", "/"))
            if sympy.simplify(whole_expr - gold_expr) == 0:
                return True
        except Exception:  # noqa: BLE001
            pass
        for token in _NUMERIC_TOKEN_RE.findall(normalized_answer_text):
            token_clean = token.rstrip("%")
            try:
                if sympy.simplify(_parse_math_expr(token_clean) - gold_expr) == 0:
                    return True
            except Exception:  # noqa: BLE001
                continue
        return False

    return gold_numeric.lower() in model_answer_text.strip().lower()


def evaluate_candidate(bundle: dict, gold_set_path: Path) -> dict:
    """Runs one real candidate against the real, reviewed 22-row gold set (Issue
    331), scored on the same partial-credit schema Bucket 1 adopted: figure detected / geometry
    type identified / values extracted / final answer exact-match. Rows with a genuinely compound
    gold answer get final_answer_exact_match = "N/A (multi-part gold answer, Issue 332)" rather
    than being scored as an ordinary miss."""
    with open(gold_set_path, encoding="utf-8") as f:
        gold_data = json.load(f)
    rows = [r for r in gold_data["rows"] if r["status"] != "excluded"]
    if len(rows) != 22:
        raise AssertionError(
            f"Expected 22 non-excluded gold rows (24 shortlisted minus 2 excluded, Issue 331), "
            f"got {len(rows)} -- the gold set file may have changed; do not silently proceed.",
        )

    can_answer_qa = bundle["can_answer_qa"]
    per_row = []
    for row in rows:
        image_path = row["image_path_abs"]
        gold_answer = row.get("corrected_answer_value") or row["stored_answer_value"]
        result = {"id": row["id"], "assumed_geometry_subtype": row["assumed_geometry_subtype"]}
        multi_part = _is_multi_part_gold(gold_answer)
        if multi_part:
            result["multi_part_gold_answer"] = True  # Issue 332, carried regardless of
            # outcome, so a structural N/A here never reads as a silent ordinary miss.

        try:
            extraction_text = _extract_for_candidate(bundle, image_path)
        except Exception as e:  # noqa: BLE001, a real candidate crash on one row must not kill
            # the whole run, matching Bucket 1's own fail-loud-not-silent discipline.
            result.update({
                "figure_detected": False, "geometry_type_identified": None,
                "values_extracted": None, "final_answer_exact_match": None,
                "error": f"{type(e).__name__}: {e}",
            })
            per_row.append(result)
            continue

        result["figure_detected"] = bool(extraction_text.strip())
        result["geometry_type_identified"] = _score_geometry_type(
            extraction_text, row["assumed_geometry_subtype"],
        )
        result["values_extracted"] = _score_values_extracted(extraction_text, gold_answer)
        # REAL BUG FIXED (found auditing the real 22-row run's 0% final_answer_exact_match_rate,
        #): the previous [:500]/[:300] slices here destroyed the ability to review what
        # the model actually said whenever a generation ran past that length (routine at
        # max_new_tokens=300/200), the scoring itself was never affected (_score_final_answer()
        # runs on the full, untruncated answer_text before this save happens, confirmed by reading
        # the real execution order), but the saved JSON became useless for post-hoc review of *why*
        # a row failed, which is exactly the evidence this project's own write-up needs. Full text
        # saved now, no cut, real generations here are bounded by max_new_tokens (300/200), never
        # unbounded, so no runaway-file-size risk from removing the slice.
        result["raw_extraction_text"] = extraction_text

        if multi_part:
            result["final_answer_exact_match"] = "N/A (multi-part gold answer, Issue 332)"
        elif can_answer_qa:
            try:
                answer_text = _answer_for_candidate(bundle, image_path, row["question_text"])
                result["final_answer_exact_match"] = _score_final_answer(row, answer_text)
                result["raw_answer_text"] = answer_text
            except Exception as e:  # noqa: BLE001
                result["final_answer_exact_match"] = None
                result["error_final_answer"] = f"{type(e).__name__}: {e}"
        else:
            result["final_answer_exact_match"] = "N/A"

        per_row.append(result)

    def _rate(key):
        scored = [r[key] for r in per_row if r.get(key) is not None and not str(r.get(key)).startswith("N/A")]
        return round(sum(1 for s in scored if s) / len(scored), 3) if scored else None

    qa_rows = [
        r for r in per_row
        if r.get("final_answer_exact_match") is not None
        and not str(r.get("final_answer_exact_match")).startswith("N/A")
    ]

    metrics = {
        "n_rows": len(per_row),
        "n_multi_part_rows": sum(1 for r in per_row if r.get("multi_part_gold_answer")),
        "figure_detected_rate": _rate("figure_detected"),
        "geometry_type_identified_rate": _rate("geometry_type_identified"),
        "values_extracted_rate": _rate("values_extracted"),
        "final_answer_exact_match_rate": (
            round(sum(1 for r in qa_rows if r["final_answer_exact_match"] is True) / len(qa_rows), 3)
            if qa_rows else "N/A"
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
# Runner, unchanged in shape from Bucket 1, matching Issue 323's precedent
# ---------------------------------------------------------------------------

SUBPROCESS_LOG_PATH = Path(__file__).parent / "subprocess_run_log.jsonl"


def _metrics_path_for(candidate_id: str) -> Path:
    return PER_ROW_RESULTS_DIR / f"{candidate_id.replace('/', '__')}__metrics.json"


def worker_main(candidate_id: str) -> None:
    """Runs exactly ONE real candidate: load, evaluate, persist. Kept as an isolated-subprocess
    worker even for this single-candidate bake-off (Issue 323's precedent), so a second
    candidate can be added later without restructuring this file."""
    with measure_load(candidate_id) as timing:
        model = load_candidate(candidate_id)

    metrics = evaluate_candidate(model, GOLD_SET_PATH)
    metrics["candidate"] = candidate_id
    metrics["load_time_s"] = round(timing.load_time_s, 2)
    metrics["vram_delta_gb"] = round(timing.vram_delta_gb, 2)
    metrics["fits_8gb"] = timing.vram_delta_gb <= 8.0

    metrics_path = _metrics_path_for(candidate_id)
    PER_ROW_RESULTS_DIR.mkdir(exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\nWorker done: {candidate_id} -> {metrics_path}")

    try:
        del model
        free_gpu()
    except Exception as e:  # noqa: BLE001, real results already persisted above; non-fatal.
        print(
            f"free_gpu() raised after this candidate's real results were already saved "
            f"(non-fatal -- process exit reclaims GPU memory regardless): {type(e).__name__}: {e}",
        )


def _safe_print(text: str) -> None:
    """Same real fix as Bucket 1 (Issue 329), defensive re-encode so a worker
    subprocess's captured stdout can never itself crash the driver on a Windows console."""
    encoding = sys.stdout.encoding or "utf-8"
    print(text.encode(encoding, errors="backslashreplace").decode(encoding, errors="replace"))


def driver_main() -> None:
    if not CANDIDATES:
        print("CANDIDATES is empty. Fill it in before running this bake-off.")
        sys.exit(1)

    PER_ROW_RESULTS_DIR.mkdir(exist_ok=True)
    venv_python = sys.executable
    script_path = Path(__file__).resolve()

    all_results = []
    for candidate_id in CANDIDATES:
        metrics_path = _metrics_path_for(candidate_id)
        if metrics_path.exists():
            print(f"\n=== Skipping {candidate_id} -- already-completed metrics found at {metrics_path} ===")
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
                f"(exit {result.returncode}). stderr tail:\n{result.stderr[-2000:]}",
            )
            record["stderr_tail"] = result.stderr[-2000:]
        with open(SUBPROCESS_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if metrics_path.exists():
            with open(metrics_path, encoding="utf-8") as f:
                all_results.append(json.load(f))
        else:
            print(f"  WARNING: {candidate_id} produced no metrics file.")

    write_results(all_results)
    print(f"\nResults written to {RESULTS_MD} and {RESULTS_CSV}")


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
        f.write("# Bucket 2 (Geometry) Bake-off Results\n\n")
        f.write("Fill in `role` and `rejection_reason` once real numbers exist.\n\n")
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
        help="Internal: run exactly one candidate in worker mode (Issue 323's pattern).",
    )
    args = parser.parse_args()
    if args.candidate:
        worker_main(args.candidate)
    else:
        driver_main()
