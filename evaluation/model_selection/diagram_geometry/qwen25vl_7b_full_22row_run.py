"""One-off full 22-row Bucket 2 evaluation of Qwen/Qwen2.5-VL-7B-Instruct, run as a separate script so that
run_bakeoff.py's candidate list / load_candidate() stay untouched (only its _score_final_answer() was fixed,
Issue 341).

Uses run_bakeoff.evaluate_candidate() itself, the same 22-row loop, prompts (extraction max_new_tokens=300,
answer max_new_tokens=200), scorers and metric aggregation the Qwen3-VL-8B-Instruct baseline run used, with
the same real loading pattern (4-bit NF4, sdpa, low_cpu_mem_usage, max_pixels=28*28*768) and the same
gpu_utils.measure_load() timing. Persists metrics in the same shape run_bakeoff.worker_main() does, under
per_row_results/Qwen__Qwen2.5-VL-7B-Instruct*.json (a different filename from the 8B files, so nothing is overwritten).

Diagram buckets: Bucket 1 is charts and data, Bucket 2 is simple geometry with measurements,
and Bucket 3 is bar models and number lines.
"""
import json
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import run_bakeoff as rb  # noqa: E402

CANDIDATE_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


def load_qwen25vl(candidate_id: str) -> dict:
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(candidate_id, min_pixels=28 * 28 * 4, max_pixels=28 * 28 * 768)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        candidate_id,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16),
        device_map={"": 0}, attn_implementation="sdpa", low_cpu_mem_usage=True)
    return {"candidate_id": candidate_id, "kind": "chat_vlm", "can_answer_qa": True,
            "model": model, "processor": processor}


def main():
    with rb.measure_load(CANDIDATE_ID) as timing:
        bundle = load_qwen25vl(CANDIDATE_ID)

    metrics = rb.evaluate_candidate(bundle, rb.GOLD_SET_PATH)
    metrics["candidate"] = CANDIDATE_ID
    metrics["load_time_s"] = round(timing.load_time_s, 2)
    metrics["vram_delta_gb"] = round(timing.vram_delta_gb, 2)
    metrics["fits_8gb"] = timing.vram_delta_gb <= 8.0

    out = rb._metrics_path_for(CANDIDATE_ID)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print("METRICS", json.dumps(metrics, ensure_ascii=False), flush=True)
    print(f"\nDone -> {out}", flush=True)
    try:
        del bundle
        rb.free_gpu()
    except Exception as e:  # noqa: BLE001, results already persisted
        print(f"free_gpu() raised after results were saved (non-fatal): {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
