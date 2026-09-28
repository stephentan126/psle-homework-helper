"""
Job A OCR bake-off runner, Unlimited-OCR (DeepSeek-OCR family).

Companion: docs/MODEL_SELECTION.md (Section 2, isolated-environment approach, and
scoring/gates). This runner is self-contained: it depends only on
its own venv at ../envs/unlimited_ocr/, never on any other candidate's environment.

Model:  baidu/Unlimited-OCR (3B params, bf16). No flash-attn dependency (unlike deepseek_ocr/
        dots_mocr), one less risk on this machine's no-system-CUDA-toolkit constraint.
Env:    ../envs/unlimited_ocr/ (Python 3.12, per the README), torch==2.10.0+cu128,
        torchvision==0.25.0+cu128, transformers==4.57.1, plus the README's full pin list
        (Pillow, matplotlib, einops, addict, easydict, pymupdf, psutil). CUDA confirmed working
        on the first install attempt (used --index-url alone, lesson learned from every prior
        candidate's gotcha).
Prompt: model.infer()'s own documented prompt: '<image>document parsing.', "gundam"
        configuration (crop_mode=True, image_size=640) per the README's own recommended default.

VRAM RISK (run plan Section 4): 6.78 GB on disk in bf16 against the 8GB RTX 4060 leaves little
room for activations and page-image tensors. Before loading: confirm nvidia-smi shows no
orphaned process from a prior crash (Issue 69), load ALONE, and record actual peak VRAM. If
it doesn't fit, log it as a gate failure with the real measured number, do NOT chase the
sahilchachra/Unlimited-OCR-GGUF fallback (needs an unmerged llama.cpp PR build).

Run order (smallest-to-largest, run plan Section 4): 7 of 7, LAST.
"""

import gc
import json
import sys
import time
import traceback
from pathlib import Path

# NOT setting HF_HUB_ENABLE_HF_TRANSFER here, confirmed on this candidate that some
# huggingface_hub versions actively ERROR if it's set to 1 without the hf_transfer package
# installed (`ValueError: Fast download using 'hf_transfer' is enabled ... but 'hf_transfer'
# package is not available`), rather than just deprecating it silently like qwen3_vl_2b's env
# did. Inconsistent behavior across versions, hf_xet (installed) is the
# real, version-independent acceleration mechanism from here on.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # same gotcha as deepseek_ocr
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "backend"))
from app.services.gpu_utils import current_vram_gb, free_gpu, measure_load  # noqa: E402

MODEL_KEY = "unlimited_ocr"
HF_REPO = "baidu/Unlimited-OCR"
PROMPT = "<image>document parsing."

GOLD_PAGES_DIR = REPO_ROOT / "evaluation" / "model_selection" / "ocr_extraction" / "gold_pages"
RAW_OUT_DIR = REPO_ROOT / "evaluation" / "model_selection" / "ocr_extraction" / "raw" / MODEL_KEY
NORMALIZED_OUT_DIR = REPO_ROOT / "evaluation" / "model_selection" / "ocr_extraction" / "normalized" / MODEL_KEY
INFER_SCRATCH_DIR = REPO_ROOT / "evaluation" / "model_selection" / "ocr_extraction" / "raw" / MODEL_KEY / "_infer_scratch"


def load_model():
    """Load Unlimited-OCR per its own README quickstart."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(HF_REPO, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        HF_REPO, trust_remote_code=True, use_safetensors=True, torch_dtype=torch.bfloat16
    )
    model = model.eval().cuda()
    return model, tokenizer


def run_on_gold_set(model, tokenizer) -> list[dict]:
    """Run Unlimited-OCR over all 18 gold-set pages in GOLD_PAGES_DIR, one try/except per page
    (run plan Section 5 / Issue 62 pattern, one bad page must never stop the run on the
    other 17). Saves raw JSON per page, incrementally. Reads result.mmd first (confirmed the
    real output filename for this family via deepseek_ocr, same underlying infer() pattern),
    falls back to *.md."""
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
                prompt=PROMPT,
                image_file=str(page_path),
                output_path=str(page_scratch),
                base_size=1024,
                image_size=640,
                crop_mode=True,
                max_length=32768,
                no_repeat_ngram_size=35,
                ngram_window=128,
                save_results=True,
            )
            t1 = time.time()

            saved = list(page_scratch.glob("*.mmd")) or list(page_scratch.glob("*.md"))
            markdown_text = saved[0].read_text(encoding="utf-8") if saved else None
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

    if not fits_8gb:
        print(f"GATE FAILURE: {MODEL_KEY} does not fit 8GB (measured {peak_vram_gb:.2f} GB).")
        print("Not running the 18-page set. NOT chasing the GGUF fallback, per run plan Section 2.")
        summary = {
            "model": MODEL_KEY,
            "hf_repo": HF_REPO,
            "load_time_s": round(timing.load_time_s, 2),
            "peak_vram_gb": round(peak_vram_gb, 2),
            "fits_8gb_gate": False,
            "pages_ok": 0,
            "pages_failed": 0,
            "pages_total": 0,
            "gate_failure": "fits_8gb",
        }
        (RAW_OUT_DIR / "_run_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        free_gpu(model)
        gc.collect()
        print(f"VRAM after free_gpu(): {current_vram_gb():.2f} GB (in-process view)")
        return

    results = run_on_gold_set(model, tokenizer)

    ok_count = sum(1 for r in results if r["error"] is None)
    fail_count = len(results) - ok_count
    print(f"\nDone: {ok_count} pages OK, {fail_count} pages failed, out of {len(results)}.")

    free_gpu(model)
    gc.collect()
    print(f"VRAM after free_gpu(): {current_vram_gb():.2f} GB (in-process view)")
    print("Run `nvidia-smi` now to confirm no orphaned VRAM — this is the LAST candidate.")

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
