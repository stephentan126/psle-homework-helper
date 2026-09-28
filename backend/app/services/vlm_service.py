"""
Load and swap orchestration for the vision-language model that transcribes photographed questions.

Serves `Qwen/Qwen3-VL-8B-Instruct` (chosen by the bake-off in
`evaluation/model_selection/photo_transcription/results.md`) through `transformers` with
`bitsandbytes` NF4 and `device_map={"": 0}`. Do not revert to `device_map="auto"`: it triggers a
`bitsandbytes`/`accelerate` CPU-offload incompatibility with this model.

GPU swap rule: only one large model may be resident on the GPU at a time. This model alone peaks at
about 10GB on the 8GB card (via Windows GPU memory oversubscription), so it cannot share the GPU
with Phi-4-mini. `get_vlm()` and `llm_service.get_llm()` each evict the other model first if it is
loaded (see Issue 191). Each imports the other module inside the function body to avoid a
circular top-level import; by the time either function runs, both modules are fully imported.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

_REPO_ROOT = Path(__file__).resolve().parents[3]  # same pattern as llm_service.py

VLM_MODEL_ID = os.environ.get("VLM_MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct")

# Minimum mean token log-probability for a transcription to be trusted. On the same question crop,
# a correct transcription scored -0.0069 and an illegible (heavily blurred, low resolution) crop, on
# which the model hallucinated "[DIAGRAM PRESENT]" over text, scored -0.0657. The threshold sits
# closer to the good value than the midpoint because failing closed into `extraction_failed` (ask
# for a retake or typed input) is safe, while accepting a bad read is not. Calibrated on only one
# good and one bad example, so it should be revisited with more data (see Issue 194).
MIN_TRUSTED_LOG_PROB = -0.02

_model = None
_processor = None
_lock = threading.Lock()  # stops concurrent first requests from both loading the model into VRAM

# The transcription prompt used in the model bake-off, reused verbatim so live behaviour matches
# what was evaluated.
FAITHFUL_TRANSCRIPTION_PROMPT = (
    "Transcribe ALL text visible in this photographed exam page EXACTLY as written. Do NOT "
    "solve, correct, or complete anything -- transcribe faithfully. If a diagram is present, "
    "note '[DIAGRAM PRESENT]' at its location but do not attempt to interpret it. If you cannot "
    "read the page at all, say so plainly rather than guessing."
)


def is_vlm_loaded() -> bool:
    return _model is not None


def get_vlm():
    """Lazy-load Qwen3-VL-8B-Instruct once and keep it resident.

    Follows the same thread-safe singleton pattern as `llm_service.get_llm()`. Evicts the LLM
    first if it is resident, to keep one large model on the GPU at a time.

    Returns:
        (model, processor)

    Raises:
        RuntimeError: CUDA is not available.
    """
    global _model, _processor
    if _model is not None:
        return _model, _processor

    with _lock:
        if _model is not None:
            return _model, _processor

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available — cannot load the VLM.")

        # Local import avoids a circular import (see module docstring).
        from app.services import llm_service
        if llm_service.is_llm_loaded():
            llm_service.unload_llm()

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        )
        _processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
        _model = Qwen3VLForConditionalGeneration.from_pretrained(
            VLM_MODEL_ID, quantization_config=quant_config, device_map={"": 0},
            attn_implementation="sdpa",
        )
    return _model, _processor


def unload_vlm() -> None:
    """Release the resident VLM and reclaim GPU memory.

    Callers must `del` their own references first, because `free_gpu()` cannot delete a caller's
    variables (see Issue 163)."""
    global _model, _processor

    from app.services.gpu_utils import free_gpu  # noqa: E402

    _model = None
    _processor = None
    free_gpu()


def transcribe_photo(image_path: str, max_new_tokens: int = 500) -> tuple[str, float]:
    """Transcribe a photographed question with greedy decoding, as in the bake-off.

    Returns:
        (text, mean_log_prob). `mean_log_prob` is the mean token log-probability of the generated
        sequence, computed from the per-step logits of the same `generate()` call at no extra
        cost. It measures transcription confidence and is kept separate from
        `question_matching.py`'s `match_confidence`, which measures whether the text matches a
        stored question. Compare it against MIN_TRUSTED_LOG_PROB.
    """
    model, processor = get_vlm()
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": FAITHFUL_TRANSCRIPTION_PROMPT},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    output = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        output_scores=True, return_dict_in_generate=True,
    )
    generated_ids = output.sequences[0][inputs["input_ids"].shape[-1]:]
    text = processor.decode(generated_ids, skip_special_tokens=True).strip()

    log_probs = []
    for step_logits, token_id in zip(output.scores, generated_ids):
        step_log_probs = torch.log_softmax(step_logits[0].float(), dim=-1)
        log_probs.append(step_log_probs[token_id].item())
    mean_log_prob = sum(log_probs) / len(log_probs) if log_probs else float("-inf")

    return text, mean_log_prob
