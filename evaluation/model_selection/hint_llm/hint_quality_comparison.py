"""
Phase 3 step 1's real hint-quality comparison (Section 4 slot 6's own stated pick criterion:
"final pick by hint quality on real questions", distinct from the load-time/VRAM measurement
already done, Issues 163/164). This is the FIRST time any candidate is actually asked to
produce the Socratic Tier-1 behaviour the design specification's own core rule defines ("never reveal
the final answer. Tier 1 restates the question and stops."), every generation in the earlier
load-test scripts asked the model to SOLVE the question, which is not what a real Tier 1 hint is.

Three real questions, drawn from the verified, unflagged corpus (`psle.db`, `extraction_flag IS
NULL`), not invented, chosen by a structural filter (self-contained text, no figure/diagram
dependency a text LLM can't see, real `answer_value` present) then picked to span both of Section
5's own question-type categories: one pure sum-type computation, two multi-step reasoning/word
problems of different real structures (a before/after ratio-change problem, a fraction-of-
remainder problem).

For each of the three Section 4 slot 6 candidates, generates a real Tier-1 hint for each real
question, all nine generations, in one process, using the same proven load/free/swap pattern
already verified in `test_candidate_swap.py` (Issue 164). Checks for real answer leakage
programmatically (the actual non-negotiable failure mode) and prints every
real output in full for direct human judgement of Tier-1 compliance, coherence, and style, this
script does NOT itself declare a winner; that judgement is recorded separately in the development log
after reading these real outputs.

Usage: backend/.venv/Scripts/python.exe evaluation/model_selection/hint_llm/hint_quality_comparison.py
"""
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "backend"))

import torch
from app.services.gpu_utils import current_vram_gb, free_gpu
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODELS_DIR = REPO_ROOT / "evaluation" / "model_selection" / "hint_llm" / "models"

CANDIDATES = {
    "qwen3-4b": {
        "hf_id": "Qwen/Qwen3-4B-Instruct-2507",
        "local_dir": MODELS_DIR / "qwen3-4b-instruct-2507",
        "enable_thinking": None,
    },
    "phi4-mini": {
        "hf_id": "microsoft/Phi-4-mini-instruct",
        "local_dir": MODELS_DIR / "phi-4-mini-instruct",
        "enable_thinking": None,
    },
    "qwen3-8b": {
        "hf_id": "Qwen/Qwen3-8B",
        "local_dir": MODELS_DIR / "qwen3-8b",
        "enable_thinking": False,
    },
}

# Real questions, real IDs, real answers, drawn from psle.db, extraction_flag IS NULL,
# structurally filtered (self-contained, no figure dependency), picked to span sum-type and
# reasoning-type per Section 5's own routing distinction. "Ans:" stripped, that's an
# answer-key blank artifact, not part of the question a student/model should see.
QUESTIONS = [
    {
        "id": 4069,
        "source": "P6_Maths_2024_WA2_nanhua.pdf Q19",
        "type": "sum-type",
        "text": r"Find the value of \(\frac{3}{4} \div 12\). Leave your answer in its simplest form.",
        "real_answer": "1/16",
    },
    {
        "id": 3078,
        "source": "P6_Maths_2023_WA2_Rosyth.pdf Q10",
        "type": "reasoning-type (ratio, before/after)",
        "text": (
            r"The ratio of the number of sweets Clark has to the number of sweets Daniel has is "
            r"\(5:9\). After each of them bought 8 more sweets, their ratio becomes \(3:5\). "
            r"Find the number of sweets Clark has now."
        ),
        "real_answer": "48",
    },
    {
        "id": 4341,
        "source": "P6_Maths_2025_SA2_aitong.pdf Q30",
        "type": "reasoning-type (fraction of remainder)",
        "text": (
            r"A shop had some bags for sale. After selling 28 bags in the morning and "
            r"\(\frac{5}{8}\) of the remaining bags in the afternoon, \(\frac{1}{4}\) of the bags "
            r"were left unsold. How many bags were sold altogether?"
        ),
        "real_answer": "63",
    },
]

# The actual Tier-1 rule, quoted directly from the design specification's own core rule, not
# invented here. "restates the question and stops" is the literal, only definition this project
# has given for Tier 1; the system prompt below states it as plainly as the spec itself does.
SYSTEM_PROMPT = (
    "You are a patient Socratic maths tutor helping a Singapore Primary 6 student (age 11-12) "
    "with a PSLE Mathematics question. You must give a TIER 1 hint ONLY. A Tier 1 hint restates "
    "the question in your own simple words, to check the student understands what is being "
    "asked, and then STOPS. Rules, all mandatory: do not solve the question; do not show any "
    "working, calculation, or steps; do not reveal the final numeric answer, even partially or "
    "as a range; do not ask a leading question about method. Just restate what the question is "
    "asking, in plain words a Primary 6 student would understand, then stop."
)


def leak_check(text: str, real_answer: str) -> "list[str]":
    """Real, programmatic check for the one non-negotiable failure mode (the project's own core
    safety rule), does the generated hint contain the real answer, in any of its plausible
    written forms? Checks the exact value and, for fraction answers, both '/'-written and
    space-written forms. Not a substitute for reading the output, but catches the clearest case
    directly rather than relying on human read-through alone."""
    hits = []
    candidates = {real_answer, real_answer.replace("/", " over "), real_answer.replace("/", " / ")}
    for form in candidates:
        if form and re.search(re.escape(form), text, re.IGNORECASE):
            hits.append(form)
    return hits


def generate_hint(model, tokenizer, question: dict, enable_thinking) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question["text"]},
    ]
    template_kwargs = {"add_generation_prompt": True, "return_tensors": "pt"}
    if enable_thinking is not None:
        template_kwargs["enable_thinking"] = enable_thinking
    input_ids = tokenizer.apply_chat_template(messages, **template_kwargs)["input_ids"].to(
        model.device,
    )
    start = time.time()
    output = model.generate(input_ids, max_new_tokens=160, do_sample=False)
    gen_time = time.time() - start
    text = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return {"text": text.strip(), "gen_time_s": round(gen_time, 2)}


def main() -> int:
    if not torch.cuda.is_available():
        print("FAILED: CUDA not available.")
        return 1

    print("=== Real hint-quality comparison (Section 4 slot 6 pick criterion) ===")
    print(f"Questions: {[q['id'] for q in QUESTIONS]}")
    print(f"Candidates: {list(CANDIDATES)}\n")

    all_results = {}
    for key, cfg in CANDIDATES.items():
        vram_before = current_vram_gb()
        print(f"\n{'='*70}\nLOADING {key} ({cfg['hf_id']}) — VRAM before: {vram_before:.3f} GB")
        if vram_before > 0.05:
            print(f"WARNING: VRAM before load not near zero — previous candidate may not be freed.")

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        )
        local_dir = str(cfg["local_dir"])
        tokenizer = AutoTokenizer.from_pretrained(local_dir)
        model = AutoModelForCausalLM.from_pretrained(
            local_dir, quantization_config=quant_config, device_map="auto",
        )
        print(f"{key} loaded.")

        results = []
        for q in QUESTIONS:
            r = generate_hint(model, tokenizer, q, cfg["enable_thinking"])
            leaks = leak_check(r["text"], q["real_answer"])
            print(f"\n--- {key} | Q{q['id']} ({q['type']}) ---")
            print(f"Question: {q['text']}")
            print(f"Real answer (NOT shown to model): {q['real_answer']}")
            print(f"Generated hint ({r['gen_time_s']}s):\n{r['text']}")
            if leaks:
                print(f"*** LEAK CHECK FLAGGED: possible answer leakage — matched {leaks} ***")
            else:
                print("Leak check: no direct match found.")
            results.append({
                "question_id": q["id"], "question_type": q["type"], "hint_text": r["text"],
                "gen_time_s": r["gen_time_s"], "leak_flags": leaks,
            })

        del model
        del tokenizer
        free_gpu()
        vram_after = current_vram_gb()
        print(f"\n{key}: VRAM after free_gpu() = {vram_after:.3f} GB")
        all_results[key] = {"vram_after_free": vram_after, "results": results}

    out_path = REPO_ROOT / "evaluation" / "model_selection" / "hint_llm" / "hint_quality_results.json"
    out_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n\nFull results saved to {out_path} (gitignored, local only).")
    print("HINT QUALITY COMPARISON GENERATION DONE — human judgement of Tier-1 compliance,")
    print("coherence, and style is the next step, not scripted here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
