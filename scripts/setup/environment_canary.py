"""
Environment canary: check that the CUDA, PyTorch and bitsandbytes chain works on this machine.

Run this once on a new machine before any other setup. Version mismatches between these three
libraries tend to fail deep inside a stack trace rather than with a clear error, so a small
end-to-end test (import, 4-bit load, one generation, memory release) catches them early.

Exit code 0 and "CANARY PASSED" mean the environment is ready. Anything else means the environment
must be fixed before continuing.

MODEL_ID is a small (1-3B), ungated instruct model. The script does not choose a model for the
project; it only proves the chain works.

Usage:
    python scripts/setup/environment_canary.py
"""

import sys
import time
import gc

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"  # small, ungated, well-supported on Windows/bitsandbytes


def main() -> int:
    print("=== Environment Canary ===")
    print("Checking imports...")
    try:
        import torch
        import bitsandbytes as bnb  # noqa: F401  (a successful import is the check)
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as e:
        print(f"FAILED at import stage: {e}")
        print("This means the pinned versions in backend/pyproject.toml are not correctly")
        print("installed for this machine. Do not proceed until this import succeeds.")
        return 1

    if not torch.cuda.is_available():
        print("FAILED: torch.cuda.is_available() is False. No GPU visible to PyTorch.")
        print("Check NVIDIA driver + CUDA toolkit installation before anything else.")
        return 1

    print(f"CUDA available. Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch version: {torch.__version__}")

    if MODEL_ID.startswith("REPLACE_ME"):
        print("FAILED: set MODEL_ID at the top of this script to a real small model first.")
        return 1

    print(f"Loading {MODEL_ID} in 4-bit (NF4)...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    start = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, quantization_config=quant_config, device_map="auto"
        )
    except Exception as e:
        print(f"FAILED during model load: {e}")
        return 1
    load_time = time.time() - start

    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"Load time: {load_time:.2f}s")
    print(f"Peak VRAM during load: {peak_vram_gb:.2f} GB")

    print("Running one generation to confirm inference works...")
    try:
        inputs = tokenizer("2 + 2 =", return_tensors="pt").to(model.device)
        start_gen = time.time()
        output = model.generate(**inputs, max_new_tokens=10)
        gen_time = time.time() - start_gen
        result_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Generation time: {gen_time:.2f}s")
        print(f"Output: {result_text!r}")
    except Exception as e:
        print(f"FAILED during generation: {e}")
        return 1

    print("Freeing GPU memory (this pattern MUST be reused in every bake-off script — see")
    print("backend/app/services/gpu_utils.py — otherwise back-to-back model tests leak VRAM and crash with")
    print("a confusing OOM error unrelated to the real cause).")
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    vram_after_free = torch.cuda.memory_allocated() / (1024**3)
    print(f"VRAM after cleanup: {vram_after_free:.2f} GB (should be near 0)")

    print()
    print("Record these numbers before proceeding.")
    print("CANARY PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
