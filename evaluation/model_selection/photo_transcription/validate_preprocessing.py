"""
Phase 4 step 2 real validation (Issue 21): run the REAL, winning Job B model
(Qwen3-VL-8B-Instruct, Issue 183) on BOTH the real, harder 'before' phone-photo test image
and the real, preprocessed 'after' result, to show a genuine, measurable faithfulness/legibility
improvement, not asserted, shown, per explicit instruction.
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "backend"))

import torch  # noqa: E402
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration  # noqa: E402

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
TEST_DIR = Path(__file__).resolve().parent / "preprocessing_test"

IMAGES = [
    ("BEFORE (size-capped only, skewed + backgrounded + lit unevenly)", "before_capped.jpg"),
    ("AFTER (full real preprocess_live_photo() output)", "after_full.jpg"),
]

FAITHFUL_TRANSCRIPTION_PROMPT = (
    "Transcribe ALL text visible in this photographed exam page EXACTLY as written. Do NOT "
    "solve, correct, or complete anything -- transcribe faithfully. If a diagram is present, "
    "note '[DIAGRAM PRESENT]' at its location but do not attempt to interpret it. If you cannot "
    "read the page at all, say so plainly rather than guessing."
)


def main() -> None:
    print(f"Loading {MODEL_ID} ...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=quant_config, device_map={"": 0},
        attn_implementation="sdpa",
    )
    print("Real model loaded.\n")

    for label, image_name in IMAGES:
        image_path = TEST_DIR / image_name
        if not image_path.exists():
            print(f"MISSING real test image: {image_path}")
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
        output = model.generate(**inputs, max_new_tokens=500, do_sample=False)
        gen_time = time.time() - gen_start
        result_text = processor.decode(
            output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True,
        ).strip()

        print("=" * 90)
        print(f"{label} (real gen time: {gen_time:.1f}s)")
        print(result_text)
        print()


if __name__ == "__main__":
    main()
