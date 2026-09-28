"""
Job B faithfulness bake-off, real load test + gold-set faithfulness run for MiniCPM-V-2_6.

Real, noted caveat before even running this: MiniCPM-V-2_6's own real license (MiniCPM Model
License.md) requires filling out a registration questionnaire for commercial use, a real
licensing friction point against this project's established Apache-2.0/MIT-only pattern
elsewhere (e.g. Section 4 slot 5's own CC-BY-NC rejections). Tested anyway for a real, complete
three-candidate comparison (Issue 50's own convention), not skipped on the license concern
alone, but the license remains a real, separate reason to prefer a cleaner-licensed alternative
if faithfulness is otherwise comparable.

MiniCPM-V models use their own real `model.chat()` method, not the standard `generate()` +
`apply_chat_template()` pattern the Qwen-VL family uses, a real, distinct usage pattern per
this model's own documentation, not a mistake to unify with the other two scripts.
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
from PIL import Image  # noqa: E402
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig  # noqa: E402

MODEL_ID = "openbmb/MiniCPM-V-2_6"
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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_ID, trust_remote_code=True, quantization_config=quant_config,
        device_map={"": 0}, attn_implementation="sdpa",
    )
    model.eval()
    load_time = time.time() - start
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"Real load time: {load_time:.1f}s, real peak VRAM during load: {peak_vram_gb:.2f} GB")

    for image_name in GOLD_IMAGES:
        image_path = GOLD_DIR / image_name
        if not image_path.exists():
            print(f"MISSING real gold image: {image_path} -- skipping, not silently continuing.")
            continue
        image = Image.open(image_path).convert("RGB")
        msgs = [{"role": "user", "content": [image, FAITHFUL_TRANSCRIPTION_PROMPT]}]

        gen_start = time.time()
        result_text = model.chat(image=None, msgs=msgs, tokenizer=tokenizer, sampling=False)
        gen_time = time.time() - gen_start

        print("=" * 90)
        print(f"IMAGE: {image_name} (real gen time: {gen_time:.1f}s)")
        print(result_text)
        print()

    peak_vram_total = torch.cuda.max_memory_allocated() / 1e9
    print(f"\nReal peak VRAM across the whole run: {peak_vram_total:.2f} GB")


if __name__ == "__main__":
    main()
