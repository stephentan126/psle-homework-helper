r"""
Test whether forcing a "Final Answer:" line recovers rows that never produced one (Issue 355).

This is an evaluation-only change; nothing in the live path is affected. For each condition
(A base, B base with RAG few-shot, C v1 adapter, C2 r16/alpha32 adapter, C3 v3 masked adapter),
it takes the recorded, merged 600-token generation of each exact-match-eligible row that has no
parseable "Final Answer:" line. It rebuilds the original prompt (same retrieval and few-shot block,
and refuses to run if the similarity-floor fallback set no longer reproduces), appends the recorded
generation and the literal "\nFinal Answer:", then greedily generates up to 20 more tokens with the
same model and adapter that wrote the recorded text. Greedy decoding makes the result reproducible.

One 4-bit base model is loaded once. The three adapters are attached by name and switched, and the
base conditions run with the adapter disabled, so only one large model is resident at a time.

Output: forced_final_answer_continuations.json in the grounding evaluation folder, scored on CPU
by grounding_forced_final_answer_score.py.

Usage: backend/.venv/Scripts/python.exe scripts/evaluation/grounding_forced_final_answer.py
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

import grounding_generate as eg  # noqa: E402
from grounding_retrieval import (  # noqa: E402
    DEFAULT_K, SIMILARITY_FLOOR, compute_or_load_held_out_embeddings, load_held_out, load_pool, retrieve_k_nearest_floored,
)
from grounding_score import extract_final_answer  # noqa: E402
from grounding_score_c2 import merged_generations  # noqa: E402

_OUT = eg._EVAL_DIR / "forced_final_answer_continuations.json"
_FORCE = "\nFinal Answer:"
_NEW_TOKENS = 20
_ADAPTERS = {
    "C_qlora_rag_fewshot": _REPO_ROOT / "models" / "adapters" / "step0_qlora_v1",
    "C2_qlora_r16a32_rag_fewshot": _REPO_ROOT / "models" / "adapters" / "step0_qlora_v2_r16_a32",
    "C3_qlora_masked_rag_fewshot": _REPO_ROOT / "models" / "adapters" / "step0_qlora_v3_masked",
}
_BASE_CONDS = ["A_base", "B_rag_fewshot"]


def main() -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    pool, held_out = load_pool(), load_held_out()
    embeds = compute_or_load_held_out_embeddings(held_out)
    prompts_a, prompts_bc, fallback_ids = {}, {}, []
    for hq in held_out:
        qid = hq["question_id"]
        prompts_a[qid] = eg.build_prompt(hq["question_text"], None)
        neighbors, used_fallback = retrieve_k_nearest_floored(embeds[qid], pool, k=DEFAULT_K, floor=SIMILARITY_FLOOR)
        if used_fallback:
            fallback_ids.append(qid)
            prompts_bc[qid] = prompts_a[qid]
        else:
            prompts_bc[qid] = eg.build_prompt(hq["question_text"], neighbors)
    recorded = json.load(open(eg._EVAL_DIR / "similarity_floor_fallback_ids.json", encoding="utf-8"))
    if sorted(fallback_ids) != sorted(recorded["fallback_ids"]):
        sys.exit("REFUSING: retrieval no longer reproduces the recorded fallback set -- prompts would differ from the originals.")

    tok = AutoTokenizer.from_pretrained(eg._MODEL_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(eg._MODEL_PATH, quantization_config=bnb, device_map="cuda:0")
    first = True
    model = base
    for cond, path in _ADAPTERS.items():
        if first:
            model = PeftModel.from_pretrained(base, str(path), adapter_name=cond)
            first = False
        else:
            model.load_adapter(str(path), adapter_name=cond)
    model.eval()

    def continue_row(cond: str, qid: int, text: str, prompts: dict) -> dict:
        system_prompt, user_prompt = prompts[qid]
        msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + text + _FORCE
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=_NEW_TOKENS, do_sample=False, pad_token_id=tok.pad_token_id)
        cont = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        n_text_tokens = len(tok(text, add_special_tokens=False)["input_ids"])
        return {"continuation": cont, "recorded_text_tokens": n_text_tokens}

    results = json.load(open(_OUT, encoding="utf-8")) if _OUT.exists() else {}
    for cond in _BASE_CONDS + list(_ADAPTERS):
        rows = merged_generations(cond)
        todo = [qid for qid, r in rows.items() if r["real_answer_value"] and extract_final_answer(r["generated_text"]) is None]
        done = results.setdefault(cond, {})
        print(f"[{cond}] {len(todo)} eligible rows lack a Final Answer line; {len(done)} already continued")
        prompts = prompts_a if cond == "A_base" else prompts_bc
        if cond in _ADAPTERS:
            model.set_adapter(cond)
        for i, qid in enumerate(todo):
            if str(qid) in done:
                continue
            if cond in _ADAPTERS:
                done[str(qid)] = continue_row(cond, qid, rows[qid]["generated_text"], prompts)
            else:
                with model.disable_adapter():
                    done[str(qid)] = continue_row(cond, qid, rows[qid]["generated_text"], prompts)
            if (i + 1) % 10 == 0 or i + 1 == len(todo):
                _OUT.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
                print(f"  [{cond}] {i + 1}/{len(todo)}")
        _OUT.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")

    del model, base
    gc.collect()
    torch.cuda.empty_cache()
    print("DONE ->", _OUT)


if __name__ == "__main__":
    main()
