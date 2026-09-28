"""
Job A OCR bake-off runner, MinerU2.5 (Paddle family, despite the name, this candidate is a
Qwen2-VL-architecture model via the transformers backend, not PaddlePaddle; "Paddle family" here
follows the run plan's own family grouping of opendatalab's OCR line, not the inference backend).

Companion: docs/MODEL_SELECTION.md (Section 2, isolated-environment approach, and
scoring/gates). This runner is self-contained: it depends only on
its own venv at ../envs/mineru25/, never on any other candidate's environment.

Model:  opendatalab/MinerU2.5-2509-1.2B (1.2B params), loaded as
        Qwen2VLForConditionalGeneration via mineru-vl-utils' MinerUClient wrapper.
Env:    ../envs/mineru25/, README gives no pinned versions; installed latest via `uv`, then
        pinned torch==2.7.1+cu118/torchvision==0.22.1+cu118 after hitting the same CPU-only-wheel
        gotcha as got_ocr2/paddleocr_vl (see requirements.txt). transformers==4.57.6.
Prompt: no free-text prompt, MinerUClient.two_step_extract(image) is the model's own documented
        call (run plan Section 3: "whatever each README's own quickstart demonstrates").
Run order (smallest-to-largest, run plan Section 4): 3 of 7.
"""

import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

# Faster HF downloads, per user instruction set before any huggingface_hub import.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "backend"))
from app.services.gpu_utils import current_vram_gb, free_gpu, measure_load  # noqa: E402

MODEL_KEY = "mineru25"
HF_REPO = "opendatalab/MinerU2.5-2509-1.2B"

GOLD_PAGES_DIR = REPO_ROOT / "evaluation" / "model_selection" / "ocr_extraction" / "gold_pages"
RAW_OUT_DIR = REPO_ROOT / "evaluation" / "model_selection" / "ocr_extraction" / "raw" / MODEL_KEY
NORMALIZED_OUT_DIR = REPO_ROOT / "evaluation" / "model_selection" / "ocr_extraction" / "normalized" / MODEL_KEY


def load_model():
    """Load MinerU2.5 per its own README quickstart: Qwen2VLForConditionalGeneration +
    AutoProcessor, wrapped in MinerUClient(backend='transformers')."""
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from mineru_vl_utils import MinerUClient

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        HF_REPO, dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(HF_REPO, use_fast=True)
    client = MinerUClient(backend="transformers", model=model, processor=processor)
    return client, model


def run_on_gold_set(client) -> list[dict]:
    """Run MinerU2.5 over all 18 gold-set pages in GOLD_PAGES_DIR, one try/except per page
    (run plan Section 5 / Issue 62 pattern, one bad page must never stop the run on the
    other 17). Saves raw JSON per page, incrementally."""
    from PIL import Image

    page_files = sorted(GOLD_PAGES_DIR.glob("GS*.png"))
    if not page_files:
        raise FileNotFoundError(
            f"No gold pages found in {GOLD_PAGES_DIR} — build it first (18 PNGs, GS001-GS018)."
        )
    if len(page_files) != 18:
        print(f"WARNING: expected 18 gold pages, found {len(page_files)}. Continuing anyway.")

    results = []
    for page_path in page_files:
        page_id = page_path.stem
        record = {"page_id": page_id, "model": MODEL_KEY, "hf_repo": HF_REPO}
        try:
            t0 = time.time()
            extracted = client.two_step_extract(Image.open(page_path))
            t1 = time.time()

            record.update(
                {
                    "extracted": extracted if isinstance(extracted, (str, list, dict)) else str(extracted),
                    "latency_s": round(t1 - t0, 3),
                    "error": None,
                }
            )
            print(f"OK   {page_id}: {t1-t0:.2f}s")
        except Exception as e:  # noqa: BLE001, per-page isolation is the whole point here
            record.update(
                {
                    "extracted": None,
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"FAIL {page_id}: {type(e).__name__}: {e}")

        # Save each page's raw result immediately, not just at the end, a crash on page 14
        # should not lose pages 1-13's results (same resumability principle as Issue 62).
        (RAW_OUT_DIR / f"{page_id}.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        results.append(record)

    return results


def run_on_directory(
    client, image_dir: Path, output_dir: Path, pages: "list[tuple[str, str]] | None" = None,
) -> list[dict]:
    """
    Stage B (design specification Section 15 step 3, resolves Issue 81). Generalizes
    run_on_gold_set() above to an arbitrary directory of rendered pages (Stage A's output,
    data/extracted/pages/{pdf_stem}/{page_index}.png) instead of the hardcoded gold-set folder.
    ADDITIVE, run_on_gold_set() is untouched and stays re-runnable against the original 18-page
    gold set; this is a new function alongside it, not a replacement. Reuses the exact same
    per-image inference call (client.two_step_extract) run_on_gold_set already uses and already
    validated across all 18 gold pages, only the input source, output layout, and the final
    on-disk shape differ.

    Output shape difference from run_on_gold_set: writes the COMMON schema (extracted_text,
    structure, bounding_boxes, field_confidence, docs/MODEL_SELECTION.md Section 4), not this
    model's own raw record shape, per the design specification step 3's explicit requirement, Stage
    C reads this file directly, it should not need to know each reader's raw JSON shape. Reuses
    build_normalized() from ../normalize_and_score.py (the exact, already-validated raw->schema
    conversion used to score the bake-off) rather than re-deriving that conversion here.

    Resumable at page granularity: skips a page if its output JSON already exists.

 VRAM FIX (found running this against a real 80-page subset, logged as a issue
    in docs/DEVELOPMENT_LOG.md): the bake-off's own run_on_gold_set() only ever processed 18 pages in
    one run, never long enough to surface this. Confirmed on the real subset run: dedicated GPU
    memory for this process alone grew from the bake-off's measured 2.15GB (right after model
    load) to 4.27GB by page 77 (Windows GPU Process Memory perf counters, per-process, not a
    system-wide figure), with per-page latency degrading from ~13s on early pages to as much as
    194s on later ones as free VRAM ran out system-wide (363MB free out of 8188MB total at the
    worst point), a severe, real slowdown, not just a hypothetical. Root cause: nothing released
    GPU memory between per-image inference calls inside this loop; PyTorch's caching allocator
    keeps it, and the bake-off's fixed 18-page run never ran long enough for that to matter.
    torch.cuda.empty_cache() + gc.collect() after every page bounds this, call once per page,
    not once per whole run. This candidate completed a real 80-page subset cleanly with the fix
    active, no confirmed remaining leak, but per the development log (Issue 83's
    follow-up), it's run through run_batched.py defensively for the full corpus anyway, since
    "worked for 80 pages" isn't the same guarantee as "safe for several thousand."

    `pages`, if given, restricts processing to exactly this explicit (pdf_stem, page_index) list
    instead of scanning image_dir for every *.png. Added for run_batched.py: a batch driver needs
    to hand each subprocess invocation an exact, fixed set of pages so a page that timed out in
    one batch is never silently retried inside the SAME batch plan. When `pages` is None, behavior
    is unchanged from before (full directory scan, same as always).
    """
    import torch
    from PIL import Image

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
        try:
            t0 = time.time()
            extracted = client.two_step_extract(Image.open(page_path))
            t1 = time.time()
            record.update(
                {
                    "extracted": extracted if isinstance(extracted, (str, list, dict)) else str(extracted),
                    "latency_s": round(t1 - t0, 3),
                    "error": None,
                }
            )
            print(f"OK   {pdf_stem}/{page_index}: {t1-t0:.2f}s")
        except Exception as e:  # noqa: BLE001, per-page isolation, same as run_on_gold_set
            record.update(
                {
                    "extracted": None,
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
        client, model = load_model()

    peak_vram_gb = current_vram_gb()
    fits_8gb = peak_vram_gb <= 8.0
    print(f"Peak VRAM after load: {peak_vram_gb:.2f} GB (fits 8GB gate: {fits_8gb})")

    if stage_b_mode:
        results = run_on_directory(client, args.image_dir, args.output_dir, pages=pages)
        ok_count = sum(1 for r in results if r.get("error") is None)
    else:
        results = run_on_gold_set(client)
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
    }
    summary_path = (args.output_dir / "_run_summary.json") if stage_b_mode else (RAW_OUT_DIR / "_run_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nRun summary: {summary}")


if __name__ == "__main__":
    main()
