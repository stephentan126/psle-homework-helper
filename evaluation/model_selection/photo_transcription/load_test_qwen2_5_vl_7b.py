"""
Job B faithfulness bake-off, real load test + gold-set faithfulness run for Qwen2.5-VL-7B-Instruct
(the ORIGINAL the design specification slot 1b candidate, tested here for real comparison against the
newer Qwen3-VL-8B-Instruct).
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

# REAL BUG FOUND during this session's own first run: the Windows console's default stdout
# encoding (cp1252) can't encode real characters a VLM's own transcription legitimately produces
# (e.g. '∠', the angle symbol, present in several real gold-set questions), crashed the whole
# script with a UnicodeEncodeError AFTER a real, valid ~20-minute model load, wasting that real
# time on a print-formatting bug, not a model/environment failure. Fixed the same way this
# project's own earlier scratch scripts handle it.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
sys.path.insert(0, str(_REPO_ROOT / "backend"))

import torch  # noqa: E402
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
GOLD_DIR = Path(__file__).resolve().parent / "gold_pages"

GOLD_IMAGES = [
    "gold_01_mcq_no_diagram.jpg",
    "gold_02_mcq_with_diagram.jpg",
    "gold_03_subparts_no_diagram.jpg",
    "gold_04_synthetic_handwritten_error.jpg",
]

FAITHFUL_TRANSCRIPTION_PROMPT = (
    "Transcribe ALL text visible in this photographed exam page EXACTLY as written, including "
    "any handwritten annotations, margin notes, or crossed-out/corrected work. Do NOT solve, "
    "correct, or complete anything -- transcribe faithfully, character for character, preserving "
    "any errors exactly as they appear. If a diagram is present, note '[DIAGRAM PRESENT]' at its "
    "location but do not attempt to interpret it."
)


def main() -> None:
    print(f"Loading {MODEL_ID} ...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
    )
    start = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    # device_map={"": 0} directly, not "auto", real lesson from Qwen3-VL-8B's own run this
    # session: "auto" + CPU-offload for the vision tower hits a real bitsandbytes-4bit +
    # accelerate-offload incompatibility (works to load, hangs/breaks on the first forward pass),
    # even though the model's own real peak VRAM (~5 GB) shows it doesn't need offload at all on
    # this 8GB card.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=quant_config, device_map={"": 0},
        attn_implementation="sdpa",
    )
    load_time = time.time() - start
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"Real load time: {load_time:.1f}s, real peak VRAM during load: {peak_vram_gb:.2f} GB")

    for image_name in GOLD_IMAGES:
        image_path = GOLD_DIR / image_name
        if not image_path.exists():
            print(f"MISSING real gold image: {image_path} -- skipping, not silently continuing.")
            continue
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": FAITHFUL_TRANSCRIPTION_PROMPT},
            ],
        }]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        gen_start = time.time()
        output = model.generate(**inputs, max_new_tokens=600, do_sample=False)
        gen_time = time.time() - gen_start
        result_text = processor.decode(
            output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True,
        ).strip()

        print("=" * 90)
        print(f"IMAGE: {image_name} (real gen time: {gen_time:.1f}s)")
        print(result_text)
        print()

    peak_vram_total = torch.cuda.max_memory_allocated() / 1e9
    print(f"\nReal peak VRAM across the whole run: {peak_vram_total:.2f} GB")


if __name__ == "__main__":
    main()
