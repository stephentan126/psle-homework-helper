"""
Phase 3 step 1, explicit in-process candidate-swap verification (Issue 163's own follow-up). Every load-test run so far (`load_test.py`) has been ONE candidate
per process invocation, a real, but weaker, proof of `free_gpu()`'s fix, since a fresh process
always starts with 0 VRAM regardless of whether cleanup actually worked. This script instead
loads THREE candidates back to back IN ONE PROCESS, the actual real-world shape Phase 3's own
multi-candidate bake-off will eventually run as (all three share one `transformers`/`bitsandbytes`
venv, unlike Job A's per-reader isolated venvs), and asserts, not just prints, that VRAM is
back near zero before each next candidate loads. Explicit instruction: don't assume the
single-candidate test's fix generalizes to a real swap without checking.

Usage: backend/.venv/Scripts/python.exe evaluation/model_selection/hint_llm/test_candidate_swap.py
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
ORDER = ["qwen3-4b", "phi4-mini", "qwen3-8b"]  # deliberately smallest->largest, then re-check
LOCAL_DIRS = {
    "qwen3-4b": MODELS_DIR / "qwen3-4b-instruct-2507",
    "phi4-mini": MODELS_DIR / "phi-4-mini-instruct",
    "qwen3-8b": MODELS_DIR / "qwen3-8b",
}
ENABLE_THINKING = {"qwen3-4b": None, "phi4-mini": None, "qwen3-8b": False}

VRAM_FREE_TOLERANCE_GB = 0.05  # matches load_test.py's own real-checked "near zero" threshold


def load_and_free(key: str) -> dict:
    local_dir = str(LOCAL_DIRS[key])
    vram_before = current_vram_gb()
    print(f"\n--- {key}: VRAM before load = {vram_before:.3f} GB ---")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
    )
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(local_dir)
    model = AutoModelForCausalLM.from_pretrained(
        local_dir, quantization_config=quant_config, device_map="auto",
    )
    load_time = time.time() - start
    peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"{key}: loaded in {load_time:.2f}s, peak VRAM {peak_vram:.3f} GB")

    # One real generation, same as load_test.py, to confirm this isn't just a tensor-loading
    # check, the model must actually run before we trust its VRAM footprint as representative.
    messages = [{"role": "user", "content": "Write down the smallest common multiple of 6 and 8."}]
    template_kwargs = {"add_generation_prompt": True, "return_tensors": "pt"}
    if ENABLE_THINKING[key] is not None:
        template_kwargs["enable_thinking"] = ENABLE_THINKING[key]
    input_ids = tokenizer.apply_chat_template(messages, **template_kwargs)["input_ids"].to(
        model.device,
    )
    output = model.generate(input_ids, max_new_tokens=32)
    _ = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)

    del model
    del tokenizer
    free_gpu()
    vram_after = current_vram_gb()
    print(f"{key}: VRAM after free_gpu() = {vram_after:.3f} GB")

    return {
        "key": key, "vram_before": vram_before, "peak_vram": peak_vram,
        "vram_after_free": vram_after,
    }


def main() -> int:
    if not torch.cuda.is_available():
        print("FAILED: CUDA not available.")
        return 1

    print("=== Candidate-swap verification: real check, not assumed ===")
    print(f"Order: {ORDER}")

    results = []
    failures = []
    for key in ORDER:
        r = load_and_free(key)
        results.append(r)
        # The real assertion this whole script exists for: VRAM before THIS candidate's own
        # load must already be near zero, i.e. the PREVIOUS candidate's cleanup genuinely
        # worked, checked from inside the SAME process, not inferred from a fresh process start.
        if r["vram_before"] > VRAM_FREE_TOLERANCE_GB:
            failures.append(
                f"{key}: VRAM before load was {r['vram_before']:.3f} GB, not near zero -- "
                f"the PREVIOUS candidate's free_gpu() did not actually clear VRAM before this "
                f"one started loading."
            )
        if r["vram_after_free"] > VRAM_FREE_TOLERANCE_GB:
            failures.append(
                f"{key}: VRAM after its own free_gpu() was {r['vram_after_free']:.3f} GB, not "
                f"near zero."
            )

    print("\n=== SUMMARY ===")
    for r in results:
        print(
            f"{r['key']:12s} vram_before={r['vram_before']:.3f}GB "
            f"peak={r['peak_vram']:.3f}GB vram_after_free={r['vram_after_free']:.3f}GB"
        )

    if failures:
        print("\nFAILED -- candidate-swap VRAM reclamation did NOT hold:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nCANDIDATE-SWAP TEST PASSED -- VRAM genuinely reclaimed between every candidate,")
    print("confirmed in-process, not just across separate process invocations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
