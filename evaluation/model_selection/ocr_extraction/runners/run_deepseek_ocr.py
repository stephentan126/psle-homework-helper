"""
Job A OCR bake-off runner, DeepSeek-OCR (DeepSeek-OCR family).

Companion: docs/MODEL_SELECTION.md (Section 2, isolated-environment approach, and
scoring/gates). This runner is self-contained: it depends only on
its own venv at ../envs/deepseek_ocr/, never on any other candidate's environment.

Model:  deepseek-ai/DeepSeek-OCR (3B params).
Env:    ../envs/deepseek_ocr/ (Python 3.12, per the README), torch==2.6.0+cu118,
        torchvision==0.21.0+cu118, transformers==4.46.3, tokenizers==0.20.3.
        flash-attn==2.7.3 (README-pinned) is DELIBERATELY NOT installed, fails to build on this
        machine (no system CUDA toolkit, `CUDA_HOME` not set). Verified working fallback: omit
        `_attn_implementation='flash_attention_2'` entirely, model runs on its default attention
        path instead. See requirements.txt for the full gotcha writeup.
Prompt: model.infer()'s own documented grounding prompt (run plan Section 3):
        "<image>\\n<|grounding|>Convert the document to markdown. "
Run order (smallest-to-largest, run plan Section 4): 5 of 7.
"""

import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

# GOTCHA (confirmed): DeepSeek-OCR's own trust_remote_code infer() implementation
# prints extracted content (including real Unicode like checkmarks/arrows from the source pages)
# directly to stdout as part of its own internal logging. On Windows, stdout defaults to the
# cp1252 console codepage when not attached to a real UTF-8-aware terminal, which raises
# UnicodeEncodeError and aborts infer() before it finishes writing output, a real loss (files
# never get saved), not just a cosmetic print issue. Verified: GS010 and GS013 lost this way on
# the first run, no output file existed on disk afterward. Fix: force UTF-8 stdout up front.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "backend"))
from app.services.gpu_utils import current_vram_gb, free_gpu, measure_load  # noqa: E402

MODEL_KEY = "deepseek_ocr"
HF_REPO = "deepseek-ai/DeepSeek-OCR"
GROUNDING_PROMPT = "<image>\n<|grounding|>Convert the document to markdown. "

GOLD_PAGES_DIR = REPO_ROOT / "evaluation" / "model_selection" / "ocr_extraction" / "gold_pages"
RAW_OUT_DIR = REPO_ROOT / "evaluation" / "model_selection" / "ocr_extraction" / "raw" / MODEL_KEY
NORMALIZED_OUT_DIR = REPO_ROOT / "evaluation" / "model_selection" / "ocr_extraction" / "normalized" / MODEL_KEY
# model.infer()'s own save_results=True path, kept separate from RAW_OUT_DIR's per-page JSON so
# neither convention has to change to accommodate the other.
INFER_SCRATCH_DIR = REPO_ROOT / "evaluation" / "model_selection" / "ocr_extraction" / "raw" / MODEL_KEY / "_infer_scratch"


def load_model():
    """Load DeepSeek-OCR per its own README quickstart, MINUS _attn_implementation=
    'flash_attention_2' (not installed on this machine, see requirements.txt gotcha #3).
    Verified working: the model loads and runs fine without it."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(HF_REPO, trust_remote_code=True)
    model = AutoModel.from_pretrained(HF_REPO, trust_remote_code=True, use_safetensors=True)
    model = model.eval().cuda().to(torch.bfloat16)
    return model, tokenizer


def run_on_gold_set(model, tokenizer) -> list[dict]:
    """Run DeepSeek-OCR over all 18 gold-set pages in GOLD_PAGES_DIR, one try/except per page
    (run plan Section 5 / Issue 62 pattern, one bad page must never stop the run on the
    other 17). Saves raw JSON per page, incrementally."""
    page_files = sorted(GOLD_PAGES_DIR.glob("GS*.png"))
    if not page_files:
        raise FileNotFoundError(
            f"No gold pages found in {GOLD_PAGES_DIR} — build it first (18 PNGs, GS001-GS018)."
        )
    if len(page_files) != 18:
        print(f"WARNING: expected 18 gold pages, found {len(page_files)}. Continuing anyway.")

    INFER_SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for page_path in page_files:
        page_id = page_path.stem
        record = {"page_id": page_id, "model": MODEL_KEY, "hf_repo": HF_REPO}
        page_scratch = INFER_SCRATCH_DIR / page_id
        page_scratch.mkdir(exist_ok=True)
        try:
            t0 = time.time()
            result = model.infer(
                tokenizer,
                prompt=GROUNDING_PROMPT,
                image_file=str(page_path),
                output_path=str(page_scratch),
                base_size=1024,
                image_size=640,
                crop_mode=True,
                save_results=True,
            )
            t1 = time.time()

            # infer() writes its result to output_path/result.mmd (confirmed empirically, NOT
            # *.md, that glob silently matched nothing on the first run and every page's
            # "markdown" field came back empty despite error=None; caught by spot-checking real
            # content instead of trusting the None-error status alone) AND may also return text
            # directly depending on release, read the saved file if present, fall back to the
            # return value otherwise.
            md_candidates = list(page_scratch.glob("*.mmd")) or list(page_scratch.glob("*.md"))
            markdown_text = md_candidates[0].read_text(encoding="utf-8") if md_candidates else None
            if markdown_text is None and isinstance(result, str):
                markdown_text = result

            record.update(
                {
                    "markdown": markdown_text,
                    "raw_return": result if isinstance(result, (str, type(None))) else str(result),
                    "latency_s": round(t1 - t0, 3),
                    "error": None,
                }
            )
            print(f"OK   {page_id}: {t1-t0:.2f}s")
        except Exception as e:  # noqa: BLE001, per-page isolation is the whole point here
            record.update(
                {
                    "markdown": None,
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"FAIL {page_id}: {type(e).__name__}: {e}")

        # Save each page's raw result immediately, not just at the end, a crash on page 14
        # should not lose pages 1-13's results (same resumability principle as Issue 62).
        (RAW_OUT_DIR / f"{page_id}.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        results.append(record)

    return results


def run_on_directory(
    model, tokenizer, image_dir: Path, output_dir: Path, pages: "list[tuple[str, str]] | None" = None,
) -> list[dict]:
    """
    Stage B (design specification Section 15 step 3, resolves Issue 81). Generalizes
    run_on_gold_set() above to an arbitrary directory of rendered pages (Stage A's output,
    data/extracted/pages/{pdf_stem}/{page_index}.png) instead of the hardcoded gold-set folder.
    ADDITIVE, run_on_gold_set() is untouched and stays re-runnable against the original 18-page
    gold set; this is a new function alongside it, not a replacement. Reuses the exact same
    per-image inference call (model.infer(...) with the same grounding prompt/kwargs) already
    validated across all 18 gold pages, including the two confirmed gotchas from that run: read
    the saved *.mmd file rather than trusting the raw return value, and the UTF-8 stdout
    reconfigure already done at import time above still applies here.

    Output shape difference from run_on_gold_set: writes the COMMON schema (extracted_text,
    structure, bounding_boxes, field_confidence, docs/MODEL_SELECTION.md Section 4), not this
    model's own raw record shape, per the design specification step 3's explicit requirement, Stage
    C reads this file directly, it should not need to know each reader's raw JSON shape. Reuses
    build_normalized() from ../normalize_and_score.py (the exact, already-validated raw->schema
    conversion used to score the bake-off) rather than re-deriving that conversion here.

    Resumable at page granularity: skips a page if its output JSON already exists.

    VRAM FIX (found running the sibling MinerU2.5 runner against a real 80-page subset,
 logged as a issue in docs/DEVELOPMENT_LOG.md, applied here pre-emptively for the
    same root cause, not yet separately reproduced against DeepSeek-OCR specifically): the
    bake-off's own run_on_gold_set() only ever processed 18 pages in one run, never long enough to
    surface this. Nothing released GPU memory between per-image inference calls inside this loop;
    PyTorch's caching allocator keeps it, and 18 pages never ran long enough for that to matter.
    torch.cuda.empty_cache() + gc.collect() after every page bounds this, call once per page, not
    once per whole run. Confirmed to only partially fix a similar symptom for THIS reader
    specifically (Issue 83), a batched-subprocess driver (run_batched.py) is the real
    mitigation, not this call alone.

    `pages`, if given, restricts processing to exactly this explicit (pdf_stem, page_index) list
    instead of scanning image_dir for every *.png. Added for run_batched.py (Issue 83's
    mitigation): a batch driver needs to hand each subprocess invocation an exact, fixed set of
    pages so a page that timed out in one batch is never silently retried inside the SAME batch
    plan, relying on this function's own directory scan + skip-existing logic alone can't
    guarantee that within one driver run. When `pages` is None, behavior is unchanged from before
    (full directory scan, same as always).
    """
    import torch

    sys.path.insert(0, str(REPO_ROOT / "evaluation" / "model_selection" / "ocr_extraction"))
    from normalize_and_score import build_normalized  # noqa: E402

    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    if pages is not None:
        page_files = [image_dir / pdf_stem / f"{page_index}.png" for pdf_stem, page_index in pages]
        missing = [p for p in page_files if not p.exists()]
        if missing:
            raise FileNotFoundError(f"--pages-file listed page(s) not found in {image_dir}: {missing}")
    else:
        page_files = sorted(image_dir.rglob("*.png"))
    if not page_files:
        raise FileNotFoundError(
            f"No rendered pages found in {image_dir} — run Stage A (render_pages.py) first."
        )

    infer_scratch_dir = output_dir / "_infer_scratch"
    infer_scratch_dir.mkdir(parents=True, exist_ok=True)

    results = []
    skipped = 0
    for page_path in page_files:
        pdf_stem = page_path.parent.name
        page_index = page_path.stem  # filename without extension, e.g. "0", "1", ...
        page_out_dir = output_dir / pdf_stem
        page_out_path = page_out_dir / f"{page_index}.json"
        if page_out_path.exists():
            skipped += 1
            continue  # resumability, already processed this page

        page_id = f"{pdf_stem}:p{page_index}"
        record = {"page_id": page_id, "model": MODEL_KEY, "hf_repo": HF_REPO}
        page_scratch = infer_scratch_dir / pdf_stem / page_index
        page_scratch.mkdir(parents=True, exist_ok=True)
        try:
            t0 = time.time()
            result = model.infer(
                tokenizer,
                prompt=GROUNDING_PROMPT,
                image_file=str(page_path),
                output_path=str(page_scratch),
                base_size=1024,
                image_size=640,
                crop_mode=True,
                save_results=True,
            )
            t1 = time.time()

            # Same gotcha as run_on_gold_set: infer() writes to output_path/result.mmd, not *.md.
            md_candidates = list(page_scratch.glob("*.mmd")) or list(page_scratch.glob("*.md"))
            markdown_text = md_candidates[0].read_text(encoding="utf-8") if md_candidates else None
            if markdown_text is None and isinstance(result, str):
                markdown_text = result

            record.update(
                {
                    "markdown": markdown_text,
                    "raw_return": result if isinstance(result, (str, type(None))) else str(result),
                    "latency_s": round(t1 - t0, 3),
                    "error": None,
                }
            )
            print(f"OK   {pdf_stem}/{page_index}: {t1-t0:.2f}s")
        except Exception as e:  # noqa: BLE001, per-page isolation, same as run_on_gold_set
            record.update(
                {
                    "markdown": None,
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"FAIL {pdf_stem}/{page_index}: {type(e).__name__}: {e}")

        schema, _scores = build_normalized(MODEL_KEY, page_id, record)
        page_out_dir.mkdir(parents=True, exist_ok=True)
        page_out_path.write_text(
            json.dumps(schema, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        results.append(schema)

        # VRAM fix (see docstring), release per-page GPU memory now, not just at the end of the
        # whole run. gc.collect() first so any dropped Python-side tensor references are actually
        # freed before empty_cache() asks PyTorch's allocator to release cached blocks back to CUDA.
        gc.collect()
        torch.cuda.empty_cache()

    print(f"Stage B ({MODEL_KEY}): {len(results)} page(s) processed, {skipped} skipped (already done).")
    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=f"{MODEL_KEY} runner — gold-set mode (default) or Stage B directory mode.")
    parser.add_argument("--image-dir", type=Path, default=None, help="Stage B mode: directory of rendered pages (data/extracted/pages/).")
    parser.add_argument("--output-dir", type=Path, default=None, help="Stage B mode: where to write per-page common-schema JSON.")
    parser.add_argument(
        "--pages-file", type=Path, default=None,
        help="Stage B mode, optional: path to a text file, one 'pdf_stem<TAB>page_index' per "
             "line, restricting this run to exactly those pages instead of scanning --image-dir "
             "for every *.png. Used by run_batched.py (Issue 83) to give each batch subprocess "
             "an exact, fixed page set.",
    )
    args = parser.parse_args()

    stage_b_mode = args.image_dir is not None
    if stage_b_mode and args.output_dir is None:
        parser.error("--output-dir is required when --image-dir is given (Stage B mode).")
    if args.pages_file is not None and not stage_b_mode:
        parser.error("--pages-file only applies in Stage B mode (requires --image-dir).")

    pages = None
    if args.pages_file is not None:
        lines = args.pages_file.read_text(encoding="utf-8").splitlines()
        pages = [tuple(line.split("\t", 1)) for line in lines if line.strip()]

    RAW_OUT_DIR.mkdir(parents=True, exist_ok=True)
    NORMALIZED_OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== {MODEL_KEY} ({HF_REPO}) — {'Stage B directory mode' if stage_b_mode else 'gold-set mode'} ===")
    print(f"VRAM before load: {current_vram_gb():.2f} GB")

    with measure_load(MODEL_KEY) as timing:
        model, tokenizer = load_model()

    peak_vram_gb = current_vram_gb()
    fits_8gb = peak_vram_gb <= 8.0
    print(f"Peak VRAM after load: {peak_vram_gb:.2f} GB (fits 8GB gate: {fits_8gb})")

    if stage_b_mode:
        results = run_on_directory(model, tokenizer, args.image_dir, args.output_dir, pages=pages)
        ok_count = sum(1 for r in results if r.get("error") is None)
    else:
        results = run_on_gold_set(model, tokenizer)
        ok_count = sum(1 for r in results if r["error"] is None)
    fail_count = len(results) - ok_count
    print(f"\nDone: {ok_count} pages OK, {fail_count} pages failed, out of {len(results)}.")

    free_gpu(model)
    gc.collect()
    print(f"VRAM after free_gpu(): {current_vram_gb():.2f} GB (in-process view)")
    print("Run `nvidia-smi` now to confirm no orphaned VRAM before starting the next candidate.")

    summary = {
        "model": MODEL_KEY,
        "hf_repo": HF_REPO,
        "mode": "stage_b_directory" if stage_b_mode else "gold_set",
        "load_time_s": round(timing.load_time_s, 2),
        "load_vram_delta_gb": round(timing.vram_delta_gb, 2),
        "peak_vram_gb": round(peak_vram_gb, 2),
        "fits_8gb_gate": fits_8gb,
        "pages_ok": ok_count,
        "pages_failed": fail_count,
        "pages_total": len(results),
        "flash_attention_2_used": False,
    }
    summary_path = (args.output_dir / "_run_summary.json") if stage_b_mode else (RAW_OUT_DIR / "_run_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nRun summary: {summary}")


if __name__ == "__main__":
    main()
