r"""
Re-run only the rows of the grounding experiment that hit the 300-token cutoff, at 600 tokens.

The scope is limited to rows that never produced a usable "Final Answer:" line in the original
446 x 3 run: 159 for A, 139 for B and 148 for C, identified by
grounding_score.extract_final_answer() returning None on the original generated_text. This fills
in the cut-off subset only. Rows that already produced an answer are left untouched in the
original files.

The only change from the original run is max_new_tokens, raised from 300 to 600. Model paths,
quantisation settings, retrieval with the similarity floor, and prompt framing are the same:
build_prompt() is imported from grounding_generate.py, and retrieval is deterministic given the
question, the fixed pool and the fixed floor, so each row gets the same few-shot block as before.
Only one large model is loaded at a time: the base model for A and B, freed, then base plus
adapter for C.

Output: new files `generations_<condition>_rerun_extended_tokens.json` and
rerun_extended_tokens_timing.json in data/extracted/slice9_eval/. The original
`generations_<condition>.json` files and `score_summary.json` are never written. `psle.db` is only
read, for the training pool.

Usage: backend/.venv/Scripts/python.exe scripts/evaluation/grounding_rerun_cutoff.py
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
from grounding_score import extract_final_answer  # noqa: E402

_MODEL_PATH = str(_REPO_ROOT / "models" / "phi-4-mini-instruct")
_ADAPTER_PATH = str(_REPO_ROOT / "models" / "adapters" / "step0_qlora_v1")
_EVAL_DIR = _REPO_ROOT / "data" / "extracted" / "slice9_eval"
_MAX_NEW_TOKENS = 600  # doubled from the original run's 300
_CONDITIONS = ["A_base", "B_rag_fewshot", "C_qlora_rag_fewshot"]


def _find_cutoff_ids(condition: str) -> list[int]:
    """Return the ids whose original generation has no usable Final Answer value.

    Uses the same rule as n_model_did_not_follow_format in score_summary.json: only
    exact-match-eligible rows (real_answer_value set) are considered, and extract_final_answer()
    is used rather than a substring check. The 58 of 446 rows with no real_answer_value are
    excluded, as score_condition() never grades them for format; including them would push the
    counts above the reported 159/139/148.
    """
    orig_path = _EVAL_DIR / f"generations_{condition}.json"
    with open(orig_path, encoding="utf-8") as f:
        rows = json.load(f)
    return sorted(
        int(qid) for qid, r in rows.items()
        if r["real_answer_value"] and extract_final_answer(r["generated_text"]) is None
    )


def _save(path: Path, data: dict[int, dict]) -> None:
    _EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in data.items()}, f, indent=2, ensure_ascii=False)


def _load_incremental(path: Path) -> dict[int, dict]:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return {int(k): v for k, v in json.load(f).items()}
    return {}


def run_rerun_subset(
    model, tok, condition: str, cutoff_rows: list[dict], prompts: dict[int, tuple[str, str]],
    fallback_ids: set[int],
) -> None:
    import torch

    out_path = _EVAL_DIR / f"generations_{condition}_rerun_extended_tokens.json"
    results = _load_incremental(out_path)  # resumable, like the original script

    remaining = [hq for hq in cutoff_rows if hq["question_id"] not in results]
    print(f"[{condition}] rerun: {len(results)} already done, {len(remaining)} remaining "
          f"of {len(cutoff_rows)} cutoff rows")

    for i, hq in enumerate(remaining):
        qid = hq["question_id"]
        system_prompt, user_prompt = prompts[qid]
        msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=_MAX_NEW_TOKENS, do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        results[qid] = {
            "question_id": qid, "condition": condition, "generated_text": text,
            "real_answer_value": hq["answer_value"], "has_diagram": hq["has_diagram"],
            "used_similarity_floor_fallback": qid in fallback_ids,
            "max_new_tokens_used": _MAX_NEW_TOKENS, "rerun_of_original_cutoff": True,
        }
        if (i + 1) % 10 == 0 or (i + 1) == len(remaining):
            _save(out_path, results)
            print(f"[{condition}] rerun {i + 1}/{len(remaining)} generated, saved checkpoint")

    _save(out_path, results)
    print(f"[{condition}] rerun DONE, {len(results)} total rows -> {out_path}")


def main() -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    t_start = time.time()

    pool = load_pool()
    held_out = load_held_out()
    held_out_by_id = {hq["question_id"]: hq for hq in held_out}
    held_out_embeds = compute_or_load_held_out_embeddings(held_out)

    cutoff_ids = {cond: _find_cutoff_ids(cond) for cond in _CONDITIONS}
    for cond in _CONDITIONS:
        print(f"[{cond}] {len(cutoff_ids[cond])} original cutoff rows to re-run at "
              f"{_MAX_NEW_TOKENS} tokens")

    # Recompute the prompts exactly as in the original run (deterministic given held_out, pool and
    # floor), only for the ids that need a re-run.
    needed_ids = sorted(set(cutoff_ids["A_base"]) | set(cutoff_ids["B_rag_fewshot"])
                         | set(cutoff_ids["C_qlora_rag_fewshot"]))
    prompts_a: dict[int, tuple[str, str]] = {}
    prompts_bc: dict[int, tuple[str, str]] = {}
    fallback_ids: set[int] = set()
    for qid in needed_ids:
        hq = held_out_by_id[qid]
        prompts_a[qid] = build_prompt(hq["question_text"], None)
        neighbors, used_fallback = retrieve_k_nearest_floored(
            held_out_embeds[qid], pool, k=DEFAULT_K, floor=SIMILARITY_FLOOR,
        )
        if used_fallback:
            fallback_ids.add(qid)
            prompts_bc[qid] = prompts_a[qid]
        else:
            prompts_bc[qid] = build_prompt(hq["question_text"], neighbors)

    tok = AutoTokenizer.from_pretrained(_MODEL_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print("=== Loading BASE model (conditions A and B share these weights) ===")
    base_model = AutoModelForCausalLM.from_pretrained(_MODEL_PATH, quantization_config=bnb, device_map="cuda:0")
    base_model.eval()

    a_rows = [held_out_by_id[qid] for qid in cutoff_ids["A_base"]]
    b_rows = [held_out_by_id[qid] for qid in cutoff_ids["B_rag_fewshot"]]
    run_rerun_subset(base_model, tok, "A_base", a_rows, prompts_a, set())
    run_rerun_subset(base_model, tok, "B_rag_fewshot", b_rows, prompts_bc, fallback_ids)

    del base_model
    gc.collect()
    torch.cuda.empty_cache()
    print("VRAM after freeing base model:", torch.cuda.memory_allocated() / 1e9, "GB")

    print("\n=== Loading base model + ADAPTER (condition C) ===")
    adapter_base = AutoModelForCausalLM.from_pretrained(_MODEL_PATH, quantization_config=bnb, device_map="cuda:0")
    adapter_model = PeftModel.from_pretrained(adapter_base, _ADAPTER_PATH)
    adapter_model.eval()

    c_rows = [held_out_by_id[qid] for qid in cutoff_ids["C_qlora_rag_fewshot"]]
    run_rerun_subset(adapter_model, tok, "C_qlora_rag_fewshot", c_rows, prompts_bc, fallback_ids)

    del adapter_model, adapter_base
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.time() - t_start
    print(f"\nTOTAL WALL-CLOCK for targeted re-run: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    with open(_EVAL_DIR / "rerun_extended_tokens_timing.json", "w", encoding="utf-8") as f:
        json.dump({
            "max_new_tokens": _MAX_NEW_TOKENS,
            "n_rows_rerun_per_condition": {cond: len(cutoff_ids[cond]) for cond in _CONDITIONS},
            "n_rows_rerun_total": sum(len(cutoff_ids[c]) for c in _CONDITIONS),
            "wall_clock_seconds": round(elapsed, 1),
        }, f, indent=2)


if __name__ == "__main__":
    main()
