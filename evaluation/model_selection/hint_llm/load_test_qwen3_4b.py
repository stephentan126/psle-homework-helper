"""
Phase 3 step 1 (design specification Section 15), real load-time/VRAM measurement for the FIRST
bake-off candidate on the new serving stack (Issue 162: `transformers`/`bitsandbytes` NF4,
replacing llama.cpp after Issue 161 found no viable AVX2-compatible CUDA wheel on this real
machine). Same measurement discipline as Phase 0's own `scripts/setup/environment_canary.py`, real
`time.time()` around the actual load call, real `torch.cuda.max_memory_allocated()` for peak
VRAM, one real generation to confirm inference actually works, `app.services.gpu_utils.free_gpu()`
cleanup after (the project rule "use free_gpu() between every candidate in any bake-off loop").

Candidate: `Qwen/Qwen3-4B-Instruct-2507`, Section 4 slot 6's own "Qwen3 4B" default, confirmed as
the real, current, non-thinking instruct release (not the `-Thinking-2507` variant, which would
add reasoning-trace latency this project's own Tier-1-hint use case doesn't want) via a direct
Hugging Face API check before picking a repo id from memory alone.

This is ONE candidate only, the full multi-candidate comparison (Phi-4-mini 3.8B, Qwen3 8B) is
the next real step after this one, not done here.

Run: backend/.venv/Scripts/python.exe evaluation/model_selection/hint_llm/load_test_qwen3_4b.py
"""
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "backend"))

import torch
from app.services.gpu_utils import current_vram_gb, free_gpu
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
# Real, evidence-based override : huggingface_hub's own Python
# downloader (both the implicit fetch inside from_pretrained() and an explicit
# snapshot_download() call, tried separately) repeatedly stalled indefinitely partway through
# this repo's large .safetensors shards on this real connection/environment, confirmed
# reproducible across two different configurations, while a plain `curl` of the same files
# completed reliably at ~15-30MB/s each time. Downloaded all real repo files directly via curl
# into this local directory and loading from there sidesteps the stall, not the model choice.
LOCAL_MODEL_DIR = str(REPO_ROOT / "evaluation" / "model_selection" / "hint_llm" / "models" / "qwen3-4b-instruct-2507")


def main() -> int:
    print("=== Phase 3 step 1: real load-time/VRAM measurement ===")
    print(f"Candidate: {MODEL_ID}")

    if not torch.cuda.is_available():
        print("FAILED: torch.cuda.is_available() is False.")
        return 1

    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    vram_before = current_vram_gb()
    print(f"VRAM before load: {vram_before:.3f} GB")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
        model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_DIR, quantization_config=quant_config, device_map="auto",
        )
    except Exception as e:
        print(f"FAILED during model load: {type(e).__name__}: {e}")
        return 1
    load_time = time.time() - start

    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
    vram_after_load = current_vram_gb()
    print(f"Load time: {load_time:.2f}s")
    print(f"Peak VRAM during load: {peak_vram_gb:.3f} GB")
    print(f"VRAM resident after load: {vram_after_load:.3f} GB")

    print("\nRunning one real generation against a real PSLE-style question...")
    prompt = "Write down the smallest common multiple of 6 and 8."
    messages = [{"role": "user", "content": prompt}]
    try:
        # apply_chat_template(..., return_tensors="pt") returns a BatchEncoding (dict-like),
        # not a raw tensor, on this transformers version, confirmed by a real AttributeError
        # on .shape the first time this ran (`KeyError: 'shape'` inside BatchEncoding.__getattr__).
        # Extract input_ids explicitly rather than assume the tensor comes back bare.
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
        )["input_ids"].to(model.device)
        start_gen = time.time()
        output = model.generate(input_ids, max_new_tokens=64)
        gen_time = time.time() - start_gen
        result_text = tokenizer.decode(
            output[0][input_ids.shape[-1]:], skip_special_tokens=True,
        )
        print(f"Generation time (64 max_new_tokens): {gen_time:.2f}s")
        print(f"Output: {result_text!r}")
    except Exception:
        import traceback
        print("FAILED during generation:")
        traceback.print_exc()
        return 1

    print("\nFreeing GPU memory (Issue 163: del our own references FIRST, in this scope —")
    print("free_gpu() cannot delete a caller's variable for it — then let it collect+empty_cache)...")
    del model
    del tokenizer
    free_gpu()
    vram_after_free = current_vram_gb()
    print(f"VRAM after cleanup: {vram_after_free:.3f} GB (should be near {vram_before:.3f} GB)")

    print("\n=== REAL NUMBERS (record in development log / docs/DEVELOPMENT_LOG.md) ===")
    print(f"model: {MODEL_ID}")
    print(f"load_time_s: {load_time:.2f}")
    print(f"peak_vram_gb: {peak_vram_gb:.3f}")
    print(f"gen_time_s (64 tokens): {gen_time:.2f}")
    print("LOAD TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
