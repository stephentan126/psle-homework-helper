"""
Phase 3 step 1 (design specification Section 15), generalized real load-time/VRAM/generation
measurement for a Section 4 slot 6 candidate on the `transformers`/`bitsandbytes` NF4 serving
stack (Issue 162). Refactored out of `load_test_qwen3_4b.py` (kept as-is, already used and
already logged with its own real numbers, not touched here) once a SECOND candidate needed the
identical measurement, per this project's own "don't duplicate, don't rewrite what's already
shipped and correct" discipline.

Same measurement discipline as `scripts/setup/environment_canary.py` and `load_test_qwen3_4b.py`: real
`time.time()` around the load call, real `torch.cuda.max_memory_allocated()` for peak VRAM, one
real generation against a real PSLE-style question, and the Issue 163-fixed cleanup pattern , 
`del` the model/tokenizer in THIS scope first, then `free_gpu()` (which cannot delete a caller's
variable for it, a function cannot do that for its caller in Python, it only does
`gc.collect()`/`empty_cache()` and warns loudly if VRAM didn't actually drop).

Usage: backend/.venv/Scripts/python.exe evaluation/model_selection/hint_llm/load_test.py <candidate_key>
  candidate_key one of: qwen3-4b, phi4-mini, qwen3-8b
"""
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "backend"))

import torch
from app.services.gpu_utils import current_vram_gb, free_gpu
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODELS_DIR = REPO_ROOT / "evaluation" / "model_selection" / "hint_llm" / "models"

# Real HF repo ids confirmed via direct API check before picking from memory (same discipline as
# load_test_qwen3_4b.py's own Qwen3-4B pick), Section 4 slot 6's three named candidates.
CANDIDATES = {
    "qwen3-4b": {
        "hf_id": "Qwen/Qwen3-4B-Instruct-2507",
        "local_dir": MODELS_DIR / "qwen3-4b-instruct-2507",
        "enable_thinking": None,  # non-thinking by design of this repo, no kwarg needed
    },
    "phi4-mini": {
        "hf_id": "microsoft/Phi-4-mini-instruct",
        "local_dir": MODELS_DIR / "phi-4-mini-instruct",
        "enable_thinking": None,  # Phi-4-mini has no thinking-mode concept
    },
    "qwen3-8b": {
        "hf_id": "Qwen/Qwen3-8B",
        "local_dir": MODELS_DIR / "qwen3-8b",
        # Qwen3-8B (unlike the -Instruct-2507 split release used for the 4B pick) is a hybrid
        # thinking/non-thinking model, explicitly disable thinking mode via the chat template's
        # own real, documented kwarg, matching this project's stated low-latency Tier-1-hint
        # preference (a reasoning trace before every hint is exactly the latency Section 5/
        # Issue 46's own decision rule is trying to protect against).
        "enable_thinking": False,
    },
}


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in CANDIDATES:
        print(f"Usage: load_test.py <{'|'.join(CANDIDATES)}>")
        return 1
    key = sys.argv[1]
    cfg = CANDIDATES[key]
    local_dir = str(cfg["local_dir"])

    print(f"=== Phase 3 step 1: real load-time/VRAM measurement ({key}) ===")
    print(f"Candidate: {cfg['hf_id']}")
    print(f"Local dir: {local_dir}")

    if not torch.cuda.is_available():
        print("FAILED: torch.cuda.is_available() is False.")
        return 1

    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    vram_before = current_vram_gb()
    print(f"VRAM before load: {vram_before:.3f} GB")
    if vram_before > 0.05:
        print(
            f"WARNING: VRAM before load is {vram_before:.3f} GB, not near zero — a previous "
            f"candidate may not have been fully freed. Investigate before trusting this "
            f"candidate's own peak-VRAM number (it would be inflated by whatever's already "
            f"resident)."
        )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(local_dir)
        model = AutoModelForCausalLM.from_pretrained(
            local_dir, quantization_config=quant_config, device_map="auto",
        )
    except Exception:
        import traceback
        print("FAILED during model load:")
        traceback.print_exc()
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
        template_kwargs = {"add_generation_prompt": True, "return_tensors": "pt"}
        if cfg["enable_thinking"] is not None:
            template_kwargs["enable_thinking"] = cfg["enable_thinking"]
        # apply_chat_template(..., return_tensors="pt") returns a BatchEncoding (dict-like), not
        # a raw tensor, on this transformers version (Issue 163), extract input_ids
        # explicitly rather than assume the tensor comes back bare.
        input_ids = tokenizer.apply_chat_template(messages, **template_kwargs)["input_ids"].to(
            model.device,
        )
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
    if vram_after_free > vram_before + 0.05:
        print(
            f"WARNING: VRAM after cleanup ({vram_after_free:.3f} GB) is meaningfully above "
            f"before-load ({vram_before:.3f} GB) — cleanup did not fully work. Do not trust the "
            f"next candidate's numbers without investigating this first."
        )

    print(f"\n=== REAL NUMBERS ({key}) ===")
    print(f"model: {cfg['hf_id']}")
    print(f"load_time_s: {load_time:.2f}")
    print(f"peak_vram_gb: {peak_vram_gb:.3f}")
    print(f"gen_time_s (64 tokens): {gen_time:.2f}")
    print(f"vram_after_cleanup_gb: {vram_after_free:.3f}")
    print("LOAD TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
