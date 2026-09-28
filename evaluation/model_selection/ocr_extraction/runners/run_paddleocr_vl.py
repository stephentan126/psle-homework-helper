"""
Job A OCR bake-off runner, PaddleOCR-VL (Paddle family).

Companion: docs/MODEL_SELECTION.md (Section 2, isolated-environment approach, and
scoring/gates). This runner is self-contained: it depends only on
its own venv at ../envs/paddleocr_vl/, never on any other candidate's environment. The one shared
piece it does use is backend/app/services/gpu_utils.py, imported by path, but note gpu_utils' `torch`-based
VRAM helpers (current_vram_gb) don't apply to a PaddlePaddle model, so this runner uses
`paddle.device.cuda.*` for its own VRAM measurement instead (see current_vram_gb() override
below) and only reuses gpu_utils' free_gpu()-style del/gc pattern, not its torch calls directly.

Model:  PaddlePaddle/PaddleOCR-VL, via the `paddleocr` package's own `PaddleOCRVL` pipeline
        class (0.9B params). CORRECTION to run plan Section 1: the run plan says this candidate's
        "own quickstart uses: transformers pipeline", the REAL README quickstart uses PaddlePaddle
        (`paddlepaddle-gpu`) via the `paddleocr` package, not raw `transformers` at all. Confirmed
        by reading the actual README.
Env:    ../envs/paddleocr_vl/, paddlepaddle-gpu==3.2.1 (cu126 build, bundles its own CUDA 12.6
        runtime), paddleocr[doc-parser]>=3.4.0 (installed: paddleocr==3.7.0). Confirmed working:
        paddle.device.is_compiled_with_cuda() == True, RTX 4060 Laptop GPU recognized.
Prompt: no free-text prompt, this is a fixed document-parsing pipeline
        (`PaddleOCRVL(pipeline_version="v1")` -> `.predict(image_path)`), per the model's own
        quickstart (run plan Section 3: "whatever each README's own quickstart demonstrates").
        Each page produces one structured result with markdown + JSON output, not separate
        'ocr'/'format' modes like GOT-OCR2.0, there's only one call per page here.
Run order (smallest-to-largest, run plan Section 4): 2 of 7.
"""

import gc
import json
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

MODEL_KEY = "paddleocr_vl"
HF_REPO = "PaddlePaddle/PaddleOCR-VL"

GOLD_PAGES_DIR = REPO_ROOT / "evaluation" / "model_selection" / "ocr_extraction" / "gold_pages"
RAW_OUT_DIR = REPO_ROOT / "evaluation" / "model_selection" / "ocr_extraction" / "raw" / MODEL_KEY
NORMALIZED_OUT_DIR = REPO_ROOT / "evaluation" / "model_selection" / "ocr_extraction" / "normalized" / MODEL_KEY


def current_vram_gb() -> float:
    """PaddlePaddle equivalent of gpu_utils.current_vram_gb(), that helper is torch-specific
    and reports 0 for a process with no torch CUDA context, so it can't be reused here."""
    import paddle

    if paddle.device.cuda.device_count() == 0:
        return 0.0
    return paddle.device.cuda.memory_allocated(0) / (1024**3)


def load_model():
    """Load PaddleOCR-VL per its own README quickstart: PaddleOCRVL(pipeline_version='v1')."""
    from paddleocr import PaddleOCRVL

    pipeline = PaddleOCRVL(pipeline_version="v1")
    return pipeline


def run_on_gold_set(pipeline) -> list[dict]:
    """Run PaddleOCR-VL over all 18 gold-set pages in GOLD_PAGES_DIR, one try/except per page
    (run plan Section 5 / Issue 62 pattern, one bad page must never stop the run on the
    other 17). Saves raw JSON + markdown per page, incrementally."""
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
            output = pipeline.predict(str(page_path))
            page_results = list(output)  # predict() returns an iterable of per-image results
            t1 = time.time()

            # Each result object supports save_to_markdown/save_to_json, pull the markdown text
            # directly for the raw record rather than only writing files, so it's inspectable
            # from the JSON alone.
            markdown_text = None
            if page_results:
                res = page_results[0]
                md = getattr(res, "markdown", None)
                if md is not None:
                    # PaddleX markdown result objects commonly expose a "markdown_texts" field
                    # or behave like a dict; handle both without assuming one shape.
                    markdown_text = md.get("markdown_texts") if isinstance(md, dict) else str(md)
                else:
                    markdown_text = str(res)

            record.update(
                {
                    "markdown": markdown_text,
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
            json.dumps(record, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        results.append(record)

    return results


def main() -> None:
    RAW_OUT_DIR.mkdir(parents=True, exist_ok=True)
    NORMALIZED_OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== {MODEL_KEY} ({HF_REPO}) ===")
    print(f"VRAM before load: {current_vram_gb():.2f} GB")

    t0 = time.time()
    pipeline = load_model()
    load_time_s = time.time() - t0
    peak_vram_gb = current_vram_gb()
    fits_8gb = peak_vram_gb <= 8.0
    print(f"[{MODEL_KEY}] load_time={load_time_s:.2f}s vram={peak_vram_gb:.2f}GB")
    print(f"Peak VRAM after load: {peak_vram_gb:.2f} GB (fits 8GB gate: {fits_8gb})")

    results = run_on_gold_set(pipeline)

    ok_count = sum(1 for r in results if r["error"] is None)
    fail_count = len(results) - ok_count
    print(f"\nDone: {ok_count} pages OK, {fail_count} pages failed, out of {len(results)}.")

    # Cleanup, no torch model object here to free_gpu() the usual way; del + gc is the
    # PaddlePaddle-appropriate equivalent (the design specification Windows/VRAM gotcha still applies).
    del pipeline
    gc.collect()
    print(f"VRAM after cleanup: {current_vram_gb():.2f} GB (in-process view)")
    print("Run `nvidia-smi` now to confirm no orphaned VRAM before starting the next candidate.")

    summary = {
        "model": MODEL_KEY,
        "hf_repo": HF_REPO,
        "load_time_s": round(load_time_s, 2),
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
