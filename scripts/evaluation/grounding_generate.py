r"""
Generate answers for the three-condition grounding experiment over the 446 frozen held-out
questions.

Conditions:
- A (base): the base model with no retrieved context or few-shot examples, asked to solve cold.
- B (base with RAG and few-shot): the base model with a few-shot block of the top-k pool examples
  from `grounding_retrieval.retrieve_k_nearest_floored()` (their question, working and answer;
  k=3 by default), asked to solve the held-out question the same way.
- C (QLoRA with RAG and few-shot): the same prompt as B, generated with the v1 adapter
  (`models/adapters/step0_qlora_v1`) on top of the base weights.

Task shape: this is not the live app's tiered-hint flow, because Tier 1 never states a final
answer and so cannot be graded by exact match. The harness asks the model to solve the question
and state a final answer, which is closer to the live "weak mode" path (Issue 180) for a new
question with no verified match. That is what the held-out set represents, and it gives a gradable
output. `_GROUNDED_HINT_PROMPT_TEMPLATE` from gate.py is not reused, because it embeds the
question's own verified solution; on a held-out row that would leak the answer into the prompt and
defeat the test of whether a grounding strategy can derive it.

A and B share the same base weights and run in one model load (A for all 446, then B). C needs the
adapter on top, so it runs in a second load after the base model is freed. Only one large model
is resident at a time.

Output: `data/extracted/slice9_eval/generations_<condition>.json`, one file per condition, saved
every 10 rows so an interrupted run resumes without losing completed work. The rows that fall back
to no-RAG prompting are logged to similarity_floor_fallback_ids.json.

Usage: backend/.venv/Scripts/python.exe scripts/evaluation/grounding_generate.py
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from grounding_retrieval import (  # noqa: E402
    DEFAULT_K, SIMILARITY_FLOOR, compute_or_load_held_out_embeddings, format_few_shot_block,
    load_held_out, load_pool, retrieve_k_nearest_floored,
)

_MODEL_PATH = str(_REPO_ROOT / "models" / "phi-4-mini-instruct")
_ADAPTER_PATH = str(_REPO_ROOT / "models" / "adapters" / "step0_qlora_v1")
_EVAL_DIR = _REPO_ROOT / "data" / "extracted" / "slice9_eval"
_MAX_NEW_TOKENS = 300  # room for a worked solution and the final-answer line;
                        # the earlier Tier 2 hint spot check used 120 for a much shorter task.

_SOLVE_SYSTEM_PROMPT = (
    "You are a Singapore Primary 6 Mathematics expert. Solve the following PSLE-style question "
    "step by step, showing your working. On its own final line, state your answer in exactly "
    "this format: \"Final Answer: <value>\". Do not add anything after that line."
)


def build_prompt(question_text: str, few_shot_examples: list[dict] | None) -> tuple[str, str]:
    """Return (system_prompt, user_prompt). few_shot_examples=None gives condition A (no RAG)."""
    if not few_shot_examples:
        return _SOLVE_SYSTEM_PROMPT, question_text
    block = format_few_shot_block(few_shot_examples)
    system = (
        _SOLVE_SYSTEM_PROMPT
        + "\n\nHere are similar solved examples for reference (the numbers differ -- use them "
        "only as a pattern for how to work through this type of question):\n\n" + block
    )
    return system, question_text


def _load_incremental(path: Path) -> dict[int, dict]:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return {int(k): v for k, v in data.items()}
    return {}


def _save_incremental(path: Path, data: dict[int, dict]) -> None:
    _EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in data.items()}, f, indent=2, ensure_ascii=False)


def run_condition(
    model, tok, condition: str, held_out: list[dict], prompts: dict[int, tuple[str, str]],
    fallback_ids: set[int] | None = None,
) -> None:
    """Generate for every held-out row not yet in this condition's output file.

    Saves every 10 rows, so a run resumes. `fallback_ids`, when given, marks each output row with
    whether it used the similarity-floor no-RAG fallback, so the fallback is visible per row.
    """
    import torch
    fallback_ids = fallback_ids or set()

    out_path = _EVAL_DIR / f"generations_{condition}.json"
    results = _load_incremental(out_path)

    remaining = [hq for hq in held_out if hq["question_id"] not in results]
    print(f"[{condition}] {len(results)} already done, {len(remaining)} remaining")

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
        }
        if (i + 1) % 10 == 0 or (i + 1) == len(remaining):
            _save_incremental(out_path, results)
            print(f"[{condition}] {i + 1}/{len(remaining)} generated, saved checkpoint")

    _save_incremental(out_path, results)
    print(f"[{condition}] DONE, {len(results)} total rows -> {out_path}")


def main() -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    pool = load_pool()
    held_out = load_held_out()
    held_out_embeds = compute_or_load_held_out_embeddings(held_out)

    # Build all prompts up front (CPU only, no model needed yet). Rows where no pool candidate
    # clears SIMILARITY_FLOOR use condition A's prompt for B and C as well, rather than a weak,
    # irrelevant example. These rows are logged explicitly.
    prompts_a: dict[int, tuple[str, str]] = {}
    prompts_bc: dict[int, tuple[str, str]] = {}
    fallback_ids: list[int] = []
    for hq in held_out:
        qid = hq["question_id"]
        prompts_a[qid] = build_prompt(hq["question_text"], None)
        neighbors, used_fallback = retrieve_k_nearest_floored(
            held_out_embeds[qid], pool, k=DEFAULT_K, floor=SIMILARITY_FLOOR,
        )
        if used_fallback:
            fallback_ids.append(qid)
            prompts_bc[qid] = prompts_a[qid]
        else:
            prompts_bc[qid] = build_prompt(hq["question_text"], neighbors)

    print(f"Similarity floor {SIMILARITY_FLOOR}: {len(fallback_ids)}/{len(held_out)} held-out "
          f"rows had 0 pool candidates clear it -> fell back to no-RAG prompting for conditions "
          f"B/C on those rows.")
    _EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(_EVAL_DIR / "similarity_floor_fallback_ids.json", "w", encoding="utf-8") as f:
        json.dump({
            "similarity_floor": SIMILARITY_FLOOR, "k": DEFAULT_K,
            "n_fallback": len(fallback_ids), "n_total": len(held_out),
            "fallback_ids": fallback_ids,
        }, f, indent=2)
    print(f"Fallback ids logged -> {_EVAL_DIR / 'similarity_floor_fallback_ids.json'}")

    tok = AutoTokenizer.from_pretrained(_MODEL_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print("=== Loading BASE model (conditions A and B share these weights) ===")
    base_model = AutoModelForCausalLM.from_pretrained(_MODEL_PATH, quantization_config=bnb, device_map="cuda:0")
    base_model.eval()

    fallback_id_set = set(fallback_ids)
    run_condition(base_model, tok, "A_base", held_out, prompts_a)
    run_condition(base_model, tok, "B_rag_fewshot", held_out, prompts_bc, fallback_id_set)

    del base_model
    gc.collect()
    torch.cuda.empty_cache()
    print("VRAM after freeing base model:", torch.cuda.memory_allocated() / 1e9, "GB")

    print("\n=== Loading base model + ADAPTER (condition C) ===")
    adapter_base = AutoModelForCausalLM.from_pretrained(_MODEL_PATH, quantization_config=bnb, device_map="cuda:0")
    adapter_model = PeftModel.from_pretrained(adapter_base, _ADAPTER_PATH)
    adapter_model.eval()

    run_condition(adapter_model, tok, "C_qlora_rag_fewshot", held_out, prompts_bc, fallback_id_set)

    del adapter_model, adapter_base
    gc.collect()
    torch.cuda.empty_cache()
    print("\nVRAM after final cleanup:", torch.cuda.memory_allocated() / 1e9, "GB")


if __name__ == "__main__":
    main()
