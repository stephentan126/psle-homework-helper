r"""
Re-run the cut-off rows of condition C only, without loading the base model on its own.

A recovery variant of grounding_rerun_cutoff.py. The full re-run was killed twice by the operating
system for low memory, both times around a model-load transition. Host RAM had only about 9.7 GB
free of 32 GB, from general desktop load rather than one runaway process. The original script
still loaded the base model first for conditions A and B, even though both had no rows left to
generate, then freed it and loaded the adapter for C. That extra load and free added host-RAM
pressure at the worst moment.

A_base (159/159) and B_rag_fewshot (139/139) were already generated and saved, so this variant
skips the base-model-only load and goes straight to base plus adapter for the remaining condition C
rows (148).

Everything else is unchanged: the same deterministic prompts, the same 600-token budget, the same
output file (`generations_C_qlora_rag_fewshot_rerun_extended_tokens.json`) and the same
incremental, resumable saving. psle.db is only read (for the training pool), never written.

Usage: backend/.venv/Scripts/python.exe scripts/evaluation/grounding_rerun_cutoff_c_only.py
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from grounding_generate import build_prompt  # noqa: E402
from grounding_retrieval import (  # noqa: E402
    DEFAULT_K, SIMILARITY_FLOOR, compute_or_load_held_out_embeddings, load_held_out, load_pool,
    retrieve_k_nearest_floored,
)
from grounding_rerun_cutoff import _find_cutoff_ids, run_rerun_subset  # noqa: E402

_MODEL_PATH = str(_REPO_ROOT / "models" / "phi-4-mini-instruct")
_ADAPTER_PATH = str(_REPO_ROOT / "models" / "adapters" / "step0_qlora_v1")
_EVAL_DIR = _REPO_ROOT / "data" / "extracted" / "slice9_eval"


def main() -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    t_start = time.time()

    c_ids = _find_cutoff_ids("C_qlora_rag_fewshot")
    print(f"[C_qlora_rag_fewshot] {len(c_ids)} original cutoff rows to re-run at 600 tokens "
          f"(base-model load skipped -- A/B already done)")

    pool = load_pool()
    held_out = load_held_out()
    held_out_by_id = {hq["question_id"]: hq for hq in held_out}
    held_out_embeds = compute_or_load_held_out_embeddings(held_out)

    prompts_bc: dict[int, tuple[str, str]] = {}
    fallback_ids: set[int] = set()
    for qid in c_ids:
        hq = held_out_by_id[qid]
        neighbors, used_fallback = retrieve_k_nearest_floored(
            held_out_embeds[qid], pool, k=DEFAULT_K, floor=SIMILARITY_FLOOR,
        )
        if used_fallback:
            fallback_ids.add(qid)
            prompts_bc[qid] = build_prompt(hq["question_text"], None)
        else:
            prompts_bc[qid] = build_prompt(hq["question_text"], neighbors)

    tok = AutoTokenizer.from_pretrained(_MODEL_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print("=== Loading base model + ADAPTER (condition C only, base model skipped) ===")
    adapter_base = AutoModelForCausalLM.from_pretrained(_MODEL_PATH, quantization_config=bnb, device_map="cuda:0")
    adapter_model = PeftModel.from_pretrained(adapter_base, _ADAPTER_PATH)
    adapter_model.eval()

    c_rows = [held_out_by_id[qid] for qid in c_ids]
    run_rerun_subset(adapter_model, tok, "C_qlora_rag_fewshot", c_rows, prompts_bc, fallback_ids)

    del adapter_model, adapter_base
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.time() - t_start
    print(f"\nC-only rerun wall-clock: {elapsed:.0f}s ({elapsed / 60:.1f} min)")

    # Write the same timing file the full script would have, merging in the row counts already
    # measured for A and B, so downstream re-scoring has one consistent record.
    with open(_EVAL_DIR / "rerun_extended_tokens_timing.json", "w", encoding="utf-8") as f:
        json.dump({
            "max_new_tokens": 600,
            "n_rows_rerun_per_condition": {"A_base": 159, "B_rag_fewshot": 139, "C_qlora_rag_fewshot": len(c_ids)},
            "n_rows_rerun_total": 159 + 139 + len(c_ids),
            "note": "A_base and B_rag_fewshot were generated in an earlier (separately-timed) "
                    "process attempt that was OS-killed for low host memory before reaching "
                    "condition C; this timing file's wall_clock_seconds covers ONLY this C-only "
                    "recovery run, not the full 446-row wall-clock. See conversation report for "
                    "the full incident timeline.",
            "c_only_wall_clock_seconds": round(elapsed, 1),
        }, f, indent=2)


if __name__ == "__main__":
    main()
