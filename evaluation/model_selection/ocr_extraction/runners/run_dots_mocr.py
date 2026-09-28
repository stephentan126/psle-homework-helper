"""
Job A OCR bake-off runner, dots.mocr (dots family).

Companion: docs/MODEL_SELECTION.md (Section 2, isolated-environment approach, and
scoring/gates). This runner is self-contained: it depends only on
its own venv at ../envs/dots_mocr/, never on any other candidate's environment.

Model: dots-studio/dots.mocr (3B params). REPO DRIFT FOUND (Issue 59/73 pattern,
        second occurrence): the run plan/protocol pin `rednote-hilab/dots.mocr`. That URL still
        resolves (no 404) but now displays "dots-studio" as the publisher org, checked the
        CURRENT README at run time per the standing rule, not from memory. Using
        `dots-studio/dots.mocr` here.
Env:    ../envs/dots_mocr/ (Python 3.12, per the README), torch==2.7.0+cu128,
        torchvision==0.22.0+cu128 (this candidate's own pinned CUDA build, newer than the cu118
        used elsewhere, driver 581.80 handles it fine, confirmed empirically).
        flash-attn==2.8.0.post2 (README-pinned, marked optional) DELIBERATELY NOT installed , 
        same CUDA_HOME/no-system-toolkit blocker as deepseek_ocr. Verified fallback: omit
        attn_implementation='flash_attention_2'.
Prompt: the model's own documented structured layout-JSON prompt (run plan Section 3), verbatim
        from the current README, NOT reconstructed from memory, given this repo's drift history.
Run order (smallest-to-largest, run plan Section 4): 6 of 7.

STATUS: REJECTED before any page was processed. Unlike deepseek_ocr, this model's
trust_remote_code modeling file unconditionally `import flash_attn`, transformers' own import
scanner refuses to load the class at all without it, so the omit-the-kwarg fallback that worked
for deepseek_ocr does not apply here. flash-attn can't be built on this machine (no CUDA_HOME /
system toolkit) and has no official Windows wheel. User-confirmed decision: reject as a genuine
setup-difficulty finding rather than install a full CUDA toolkit or chase an unofficial wheel.
This file is kept as evidence of what was attempted.
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

MODEL_KEY = "dots_mocr"
HF_REPO = "dots-studio/dots.mocr"  # see module docstring, drifted from rednote-hilab/dots.mocr

# Verbatim from the current README's own "Hugginface inference details" example (fetched
#), not reconstructed from memory or from the protocol's older citation.
LAYOUT_PROMPT = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
"""

GOLD_PAGES_DIR = REPO_ROOT / "evaluation" / "model_selection" / "ocr_extraction" / "gold_pages"
RAW_OUT_DIR = REPO_ROOT / "evaluation" / "model_selection" / "ocr_extraction" / "raw" / MODEL_KEY
NORMALIZED_OUT_DIR = REPO_ROOT / "evaluation" / "model_selection" / "ocr_extraction" / "normalized" / MODEL_KEY


def load_model():
    """Load dots.mocr per its own README quickstart, MINUS attn_implementation=
    'flash_attention_2' (not installed on this machine)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    model = AutoModelForCausalLM.from_pretrained(
        HF_REPO,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(HF_REPO, trust_remote_code=True)
    return model, processor


def run_on_gold_set(model, processor) -> list[dict]:
    """Run dots.mocr over all 18 gold-set pages in GOLD_PAGES_DIR, one try/except per page
    (run plan Section 5 / Issue 62 pattern, one bad page must never stop the run on the
    other 17). Saves raw JSON per page, incrementally."""
    from qwen_vl_utils import process_vision_info

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
                        {"type": "text", "text": LAYOUT_PROMPT},
                    ],
                }
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
            generated_ids = model.generate(**inputs, max_new_tokens=24000)
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            t1 = time.time()

            record.update({"layout_json_text": output_text, "latency_s": round(t1 - t0, 3), "error": None})
            print(f"OK   {page_id}: {t1-t0:.2f}s")
        except Exception as e:  # noqa: BLE001, per-page isolation is the whole point here
            record.update(
                {
                    "layout_json_text": None,
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
        "flash_attention_2_used": False,
    }
    (RAW_OUT_DIR / "_run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nRun summary: {summary}")


if __name__ == "__main__":
    main()
