"""
Job A OCR bake-off runner, Qwen3-VL small (Qwen family).

Companion: docs/MODEL_SELECTION.md (Section 2, isolated-environment approach, and
scoring/gates). This runner is self-contained: it depends only on
its own venv at ../envs/qwen3_vl_2b/, never on any other candidate's environment.

Model:  Qwen/Qwen3-VL-2B-Instruct (2B params).
Env:    ../envs/qwen3_vl_2b/, README FLAG VERIFIED (run plan Section 1's explicit ask): README
        says transformers v4.57.0 wasn't released yet and recommends a git-source install.
        Checked empirically instead of trusting either claim: stable PyPI transformers==5.15.0
        imports Qwen3VLForConditionalGeneration cleanly, no git-source install needed. Same
        CPU-only-wheel gotcha as every other candidate; torch pinned to 2.7.1+cu118.
Prompt: chat-template based, per the model's own quickstart: a user message with an image +
        instruction text, via processor.apply_chat_template(). This runner asks the model to
        transcribe the page (document OCR framing), not the README's generic "Describe this
        image" demo prompt, that demo prompt is for image captioning, not page transcription,
        and using it here would bias results toward whichever model tolerates a mismatched
        prompt best rather than testing real page-reading (run plan Section 3 principle).
Run order (smallest-to-largest, run plan Section 4): 4 of 7.
"""

import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "backend"))
from app.services.gpu_utils import current_vram_gb, free_gpu, measure_load  # noqa: E402

MODEL_KEY = "qwen3_vl_2b"
HF_REPO = "Qwen/Qwen3-VL-2B-Instruct"

TRANSCRIBE_PROMPT = (
    "Transcribe all text on this exam page exactly as printed, preserving question numbers, "
    "answer options, and layout order. Do not correct, complete, or paraphrase anything."
)

GOLD_PAGES_DIR = REPO_ROOT / "evaluation" / "model_selection" / "ocr_extraction" / "gold_pages"
RAW_OUT_DIR = REPO_ROOT / "evaluation" / "model_selection" / "ocr_extraction" / "raw" / MODEL_KEY
NORMALIZED_OUT_DIR = REPO_ROOT / "evaluation" / "model_selection" / "ocr_extraction" / "normalized" / MODEL_KEY


def load_model():
    """Load Qwen3-VL small per its own README quickstart pattern."""
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        HF_REPO, dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(HF_REPO)
    return model, processor


def run_on_gold_set(model, processor) -> list[dict]:
    """Run Qwen3-VL small over all 18 gold-set pages in GOLD_PAGES_DIR, one try/except per page
    (run plan Section 5 / Issue 62 pattern, one bad page must never stop the run on the
    other 17). Saves raw JSON per page, incrementally."""
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
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(page_path)},
                        {"type": "text", "text": TRANSCRIBE_PROMPT},
                    ],
                }
            ]
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)
            generated_ids = model.generate(**inputs, max_new_tokens=2048)
            # Trim the prompt tokens off the front, keep only the newly generated continuation.
            new_tokens = generated_ids[:, inputs["input_ids"].shape[1]:]
            text = processor.batch_decode(
                new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            t1 = time.time()

            record.update({"transcription": text, "latency_s": round(t1 - t0, 3), "error": None})
            print(f"OK   {page_id}: {t1-t0:.2f}s")
        except Exception as e:  # noqa: BLE001, per-page isolation is the whole point here
            record.update(
                {
                    "transcription": None,
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
        model, processor = load_model()

    peak_vram_gb = current_vram_gb()
    fits_8gb = peak_vram_gb <= 8.0
    print(f"Peak VRAM after load: {peak_vram_gb:.2f} GB (fits 8GB gate: {fits_8gb})")

    results = run_on_gold_set(model, processor)

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
