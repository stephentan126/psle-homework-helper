"""Final Bucket 2 test: same full 22-row Qwen2.5-VL-7B-Instruct run as qwen25vl_7b_full_22row_run.py, with ONE change,
the answer call's max_new_tokens is 600 instead of 200 (extraction stays 300; scorer fix of Issue 341 in place;
same prompts, greedy decoding, max_pixels=28*28*768, loader). run_bakeoff.py is not modified: its hard-coded
_answer_for_candidate() (200 tokens) is swapped for a 600-token twin at runtime, the same technique the dry-run check
uses to substitute the model-calling functions. Output goes to per_row_results_answer600/ so the 200-token results in
per_row_results/ are not overwritten. Full raw answer text per row is also saved (the harness itself keeps only the
parsed value when a FINAL ANSWER line is found).

Diagram buckets: Bucket 1 is charts and data, Bucket 2 is simple geometry with measurements,
and Bucket 3 is bar models and number lines.
"""
import json
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import run_bakeoff as rb  # noqa: E402
from qwen25vl_7b_full_22row_run import load_qwen25vl  # noqa: E402  (same loader as the 200-token run)

CANDIDATE_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
ANSWER_TOKENS = 600
OUT_DIR = _HERE / "per_row_results_answer600"

_raw_by_question = {}


def _answer_600(bundle, image_path, question_text):
    """Identical to run_bakeoff._answer_for_candidate() except max_new_tokens=600; also records the raw text."""
    t0 = time.time()
    raw = rb._generate_chat_vlm(bundle, image_path, question_text + rb._QA_PROMPT_SUFFIX,
                                max_new_tokens=ANSWER_TOKENS)
    _raw_by_question[question_text] = {"raw": raw, "seconds": round(time.time() - t0, 2)}
    claimed = rb.extract_claimed_answer(raw)
    return claimed if claimed is not None else raw


def main():
    rb.PER_ROW_RESULTS_DIR = OUT_DIR  # evaluate_candidate() writes its per-row JSON here
    rb._answer_for_candidate = _answer_600

    with rb.measure_load(CANDIDATE_ID) as timing:
        bundle = load_qwen25vl(CANDIDATE_ID)
    t0 = time.time()
    metrics = rb.evaluate_candidate(bundle, rb.GOLD_SET_PATH)
    metrics["candidate"] = CANDIDATE_ID
    metrics["answer_max_new_tokens"] = ANSWER_TOKENS
    metrics["load_time_s"] = round(timing.load_time_s, 2)
    metrics["vram_delta_gb"] = round(timing.vram_delta_gb, 2)
    metrics["fits_8gb"] = timing.vram_delta_gb <= 8.0
    metrics["evaluate_wall_s"] = round(time.time() - t0, 1)
    metrics["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "Qwen__Qwen2.5-VL-7B-Instruct__metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "raw_answers_by_question.json").write_text(
        json.dumps(_raw_by_question, indent=2, ensure_ascii=False), encoding="utf-8")
    print("METRICS", json.dumps(metrics, ensure_ascii=False), flush=True)
    try:
        del bundle
        rb.free_gpu()
    except Exception as e:  # noqa: BLE001, results already persisted
        print(f"free_gpu() raised after results were saved (non-fatal): {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
