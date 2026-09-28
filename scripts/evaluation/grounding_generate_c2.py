r"""
Generate answers for one extra grounding condition over the 446 frozen held-out questions
(Issue 349, option C).

The condition is Phi-4-mini with the r=16/alpha=32 adapter
(models/adapters/step0_qlora_v2_r16_a32), RAG and few-shot, as condition
"C2_qlora_r16a32_rag_fewshot".
Conditions A, B and C already have recorded generations and are not re-run.

The prompt builder, retrieval (same pool, k and similarity floor), few-shot formatting and
generation loop are reused unchanged from grounding_generate.py; only the adapter path and the
condition name differ. Decoding is greedy (do_sample=False), as in the original run.

Token budget: the corrected baselines used a 300-token pass for every row, then a 600-token re-run
of rows that never produced a Final Answer (Issue 316). With greedy decoding, a row that finished
within 300 tokens produces identical text under a 600-token cap, and a cut-off row simply
continues. A single pass at max_new_tokens=600 is therefore equivalent, and is what this
script does.

Before the model loads, retrieval must reproduce the recorded similarity-floor fallback list
(data/extracted/slice9_eval/similarity_floor_fallback_ids.json). A mismatch means the prompts
differ from those conditions A, B and C saw, and the script refuses to run. Saves every 10 rows,
so a run resumes.

Usage: backend/.venv/Scripts/python.exe scripts/evaluation/grounding_generate_c2.py
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

import grounding_generate as eg  # noqa: E402  (original harness: prompts, run_condition, paths)
from grounding_retrieval import (  # noqa: E402
    DEFAULT_K, SIMILARITY_FLOOR, compute_or_load_held_out_embeddings, load_held_out, load_pool, retrieve_k_nearest_floored,
)

_ADAPTER_PATH = str(_REPO_ROOT / "models" / "adapters" / "step0_qlora_v2_r16_a32")
_CONDITION = "C2_qlora_r16a32_rag_fewshot"
_TOKENS = 600


def main() -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not Path(_ADAPTER_PATH).exists():
        sys.exit(f"REFUSING: adapter not found at {_ADAPTER_PATH}")
    eg._MAX_NEW_TOKENS = _TOKENS  # run_condition() reads this module global at call time

    pool, held_out = load_pool(), load_held_out()
    embeds = compute_or_load_held_out_embeddings(held_out)
    prompts_bc, fallback_ids = {}, []
    for hq in held_out:
        qid = hq["question_id"]
        neighbors, used_fallback = retrieve_k_nearest_floored(embeds[qid], pool, k=DEFAULT_K, floor=SIMILARITY_FLOOR)
        if used_fallback:
            fallback_ids.append(qid)
            prompts_bc[qid] = eg.build_prompt(hq["question_text"], None)
        else:
            prompts_bc[qid] = eg.build_prompt(hq["question_text"], neighbors)

    recorded = json.load(open(eg._EVAL_DIR / "similarity_floor_fallback_ids.json", encoding="utf-8"))
    if sorted(fallback_ids) != sorted(recorded["fallback_ids"]):
        sys.exit(f"REFUSING: retrieval no longer reproduces the recorded fallback set ({len(fallback_ids)} vs "
                 f"{recorded['n_fallback']} recorded) -- prompts would differ from what conditions A/B/C saw.")
    print(f"retrieval reproduces the recorded similarity-floor fallback set exactly ({len(fallback_ids)}/{len(held_out)} rows); "
          f"max_new_tokens={_TOKENS}, greedy")

    tok = AutoTokenizer.from_pretrained(eg._MODEL_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    print("=== Loading base model + NEW r=16/alpha=32 adapter ===")
    base = AutoModelForCausalLM.from_pretrained(eg._MODEL_PATH, quantization_config=bnb, device_map="cuda:0")
    model = PeftModel.from_pretrained(base, _ADAPTER_PATH)
    model.eval()
    eg.run_condition(model, tok, _CONDITION, held_out, prompts_bc, set(fallback_ids))

    del model, base
    gc.collect()
    torch.cuda.empty_cache()
    print("VRAM after cleanup:", torch.cuda.memory_allocated() / 1e9, "GB")


if __name__ == "__main__":
    main()
