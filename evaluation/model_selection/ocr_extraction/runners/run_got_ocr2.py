"""
Job A OCR bake-off runner, GOT-OCR 2.0 (GOT family).

Companion: docs/MODEL_SELECTION.md (Section 2, isolated-environment approach, and
scoring/gates). This runner is self-contained: it depends only on
its own venv at ../envs/got_ocr2/, never on any other candidate's environment. The one shared
piece it does use is backend/app/services/gpu_utils.py, imported by path (not installed), it only needs
`torch` + stdlib, so it works fine inside this isolated venv without pulling in the main
backend's pinned dependency set.

Model: stepfun-ai/GOT-OCR2_0 (~580M params). Verified this is a real Apache-2.0
        mirror of the original ucaslcl/GOT-OCR2_0 research (same authors, same weights), the
        model's own README quickstart still shows the ucaslcl load path, corrected here to load
        from the run-plan-pinned stepfun-ai repo instead.
Env:    ../envs/got_ocr2/, torch==2.0.1+cu118 (NOT the plain PyPI torch==2.0.1, which installs
        CPU-only), torchvision==0.15.2, transformers==4.37.2,
        tiktoken==0.6.0, verovio==4.3.1, accelerate==0.28.0. Confirmed working: CUDA available,
        RTX 4060 Laptop GPU, compute capability (8, 9).
Prompt: AutoModel + .chat(), ocr_type='ocr' (plain text) AND ocr_type='format' (structured
        pages, MCQ grid, answer key) run on every page, per run plan Section 3, so the two modes
        can be compared directly rather than picking one mode per page by hand.
Run order (smallest-to-largest, run plan Section 4): 1 of 7.
"""

import gc
import json
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "backend"))
from app.services.gpu_utils import current_vram_gb, free_gpu, measure_load  # noqa: E402

MODEL_KEY = "got_ocr2"
HF_REPO = "stepfun-ai/GOT-OCR2_0"

GOLD_PAGES_DIR = REPO_ROOT / "evaluation" / "model_selection" / "ocr_extraction" / "gold_pages"
RAW_OUT_DIR = REPO_ROOT / "evaluation" / "model_selection" / "ocr_extraction" / "raw" / MODEL_KEY
NORMALIZED_OUT_DIR = REPO_ROOT / "evaluation" / "model_selection" / "ocr_extraction" / "normalized" / MODEL_KEY


def load_model():
    """Load GOT-OCR 2.0 per its own README quickstart (device_map='cuda', trust_remote_code=True,
    use_safetensors=True), loading from the run-plan-pinned stepfun-ai/GOT-OCR2_0 repo."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(HF_REPO, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        HF_REPO,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="cuda",
        use_safetensors=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    model = model.eval().cuda()
    return model, tokenizer


def run_on_gold_set(model, tokenizer) -> list[dict]:
    """Run GOT-OCR 2.0 over all 18 gold-set pages in GOLD_PAGES_DIR, one try/except per page
    (run plan Section 5 / Issue 62 pattern, one bad page must never stop the run on the
    other 17). Runs BOTH ocr_type='ocr' and ocr_type='format' per page (run plan Section 3) so
    both modes can be compared. Returns the list of per-page result dicts (also written to disk
    incrementally, in case of a mid-run crash)."""
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
            text_ocr = model.chat(tokenizer, str(page_path), ocr_type="ocr")
            t1 = time.time()
            text_format = model.chat(tokenizer, str(page_path), ocr_type="format")
            t2 = time.time()

            record.update(
                {
                    "ocr_type_ocr": text_ocr,
                    "ocr_type_ocr_latency_s": round(t1 - t0, 3),
                    "ocr_type_format": text_format,
                    "ocr_type_format_latency_s": round(t2 - t1, 3),
                    "error": None,
                }
            )
            print(f"OK   {page_id}: ocr={t1-t0:.2f}s format={t2-t1:.2f}s")
        except Exception as e:  # noqa: BLE001, per-page isolation is the whole point here
            record.update(
                {
                    "ocr_type_ocr": None,
                    "ocr_type_format": None,
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


def main() -> None:
    RAW_OUT_DIR.mkdir(parents=True, exist_ok=True)
    NORMALIZED_OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== {MODEL_KEY} ({HF_REPO}) ===")
    print(f"VRAM before load: {current_vram_gb():.2f} GB")

    with measure_load(MODEL_KEY) as timing:
        model, tokenizer = load_model()

    peak_vram_gb = current_vram_gb()
    fits_8gb = peak_vram_gb <= 8.0
    print(f"Peak VRAM after load: {peak_vram_gb:.2f} GB (fits 8GB gate: {fits_8gb})")

    results = run_on_gold_set(model, tokenizer)

    ok_count = sum(1 for r in results if r["error"] is None)
    fail_count = len(results) - ok_count
    print(f"\nDone: {ok_count} pages OK, {fail_count} pages failed, out of {len(results)}.")

    # MANDATORY cleanup (the design specification Windows/VRAM gotcha), free_gpu() del's + empties
    # cache, but the caller (this script) must still explicitly check nvidia-smi afterward; the
    # in-process torch.cuda.memory_allocated() check below only sees this process's own view.
    free_gpu(model, tokenizer)
    gc.collect()
    print(f"VRAM after free_gpu(): {current_vram_gb():.2f} GB (in-process view)")
    print("Run `nvidia-smi` now to confirm no orphaned VRAM before starting the next candidate.")

    summary = {
        "model": MODEL_KEY,
        "hf_repo": HF_REPO,
        "load_time_s": round(timing.load_time_s, 2),
        "load_vram_delta_gb": round(timing.vram_delta_gb, 2),
        "peak_vram_gb": round(peak_vram_gb, 2),
        "fits_8gb_gate": fits_8gb,
        "pages_ok": ok_count,
        "pages_failed": fail_count,
        "pages_total": len(results),
    }
    (RAW_OUT_DIR / "_run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nRun summary: {summary}")


if __name__ == "__main__":
    main()
