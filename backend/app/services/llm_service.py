"""
Load and swap orchestration for the language model used by the Socratic hint gate.

Serves `microsoft/Phi-4-mini-instruct` (chosen by the bake-off in
`evaluation/model_selection/hint_llm/results.md`) through `transformers` with `bitsandbytes` NF4
quantisation. The model stays resident across requests; `unload_llm()` is for teardown, testing and
GPU swaps with the VLM, not per-request use. `LLM_MODEL_PATH` defaults to an absolute path anchored
at the repository root, so the model resolves the same way whatever the working directory.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from contextlib import contextmanager
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)

# backend/app/services/llm_service.py -> parents[0]=services, [1]=app, [2]=backend,
# [3]=repo root. Same pattern as app/db/session.py (see Issue 160).
_REPO_ROOT = Path(__file__).resolve().parents[3]

LLM_MODEL_PATH = os.environ.get(
    "LLM_MODEL_PATH", str(_REPO_ROOT / "models" / "phi-4-mini-instruct"),
)

# Optional LoRA adapter applied on top of the base weights. The default is "" (base weights only)
# because no trained adapter showed a significant gain over base weights on any measure tested:
#   - Solving accuracy, base+RAG vs adapter+RAG: no significant gain.
#   - r=16/alpha=32 adapter: 58/388 vs 50/388, McNemar p=0.077.
#   - Loss-masking + double-quant retrain (v3): 54/388 vs 50/388, p=0.54.
#   - Tier 2 hint quality via gate.generate_hint(), paired Wilcoxon, n=50: none of 4 rubric
#     dimensions significant (closest: consistency, p=0.075).
# The adapter only affects hint writing; solving, self-consistency and method-alignment calls pass
# use_adapter=False (see Issue 353).
#
# Override with LLM_ADAPTER_PATH=<path> (e.g. models/adapters/step0_qlora_v2_r16_a32). A configured
# path with no adapter is not silently ignored: it logs a warning and get_llm_adapter_status()
# reports applied=False. The adapter is left un-merged because merging into 4-bit weights is lossy
# (peft warns of "different generations due to rounding errors") and cannot be undone in-process.
LLM_ADAPTER_PATH = os.environ.get("LLM_ADAPTER_PATH", "")

_model = None
_tokenizer = None
_adapter_status: dict = {"configured_path": LLM_ADAPTER_PATH, "applied": False, "reason": "LLM not loaded yet"}
# FastAPI can serve concurrent requests, so guard the lazy load to stop two simultaneous first
# requests from both loading the model into VRAM.
_lock = threading.Lock()


def is_llm_loaded() -> bool:
    return _model is not None


def get_llm():
    """Lazy-load Phi-4-mini once and keep it resident.

    The load is thread-safe; all callers share the one model and tokenizer. Evicts the VLM first
    if it is resident, because only one large model may occupy the GPU at a time (see
    `vlm_service.py`).

    Returns:
        (model, tokenizer)

    Raises:
        RuntimeError: CUDA is not available.
        FileNotFoundError: LLM_MODEL_PATH does not exist.
    """
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    with _lock:
        if _model is not None:  # re-check inside the lock (another thread may have won the race)
            return _model, _tokenizer

        # Local import avoids a circular import (see vlm_service.py).
        from app.services import vlm_service
        if vlm_service.is_vlm_loaded():
            vlm_service.unload_vlm()

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available — cannot load the LLM.")
        if not Path(LLM_MODEL_PATH).exists():
            raise FileNotFoundError(
                f"LLM_MODEL_PATH does not exist: {LLM_MODEL_PATH!r}. Real model weights are "
                f"gitignored and must be present on disk — see evaluation/model_selection/hint_llm/results.md "
                f"for how they were obtained (curl, not huggingface_hub — Issue 163)."
            )

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        )
        _tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH)
        _model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_PATH, quantization_config=quant_config, device_map="auto",
        )
        _apply_adapter_if_configured()
    return _model, _tokenizer


def _apply_adapter_if_configured() -> None:
    """Wraps the base `_model` with the LoRA adapter at LLM_ADAPTER_PATH, if one is present.

    Called only while holding get_llm()'s lock. Records the outcome in `_adapter_status` either way,
    so the status never reports an adapter that is not active."""
    global _model, _adapter_status
    path = LLM_ADAPTER_PATH
    if not path.strip():
        _adapter_status = {"configured_path": path, "applied": False,
                           "reason": "adapter explicitly disabled (LLM_ADAPTER_PATH is empty) -- serving BASE weights"}
        logger.info("LLM adapter disabled by configuration; serving base weights only.")
        return
    config_file = Path(path) / "adapter_config.json"
    if not config_file.is_file():
        _adapter_status = {"configured_path": path, "applied": False,
                           "reason": f"adapter NOT FOUND at {path!r} (no adapter_config.json) -- serving BASE weights"}
        logger.warning("LLM_ADAPTER_PATH %r has no adapter_config.json: the adapter is NOT applied and the app is "
                       "serving BASE weights.", path)
        return
    from peft import PeftModel  # local import: peft is only needed when an adapter is applied

    cfg = json.loads(config_file.read_text(encoding="utf-8"))
    # Un-merged, in inference mode (is_trainable defaults to False).
    _model = PeftModel.from_pretrained(_model, path)
    _model.eval()
    _adapter_status = {"configured_path": path, "applied": True, "r": cfg.get("r"), "lora_alpha": cfg.get("lora_alpha"),
                       "merged": False, "reason": "adapter applied (un-merged PEFT wrapper)"}
    logger.info("LLM adapter applied from %s (r=%s, alpha=%s, un-merged).", path, cfg.get("r"), cfg.get("lora_alpha"))


def get_llm_adapter_status() -> dict:
    """Report which weights this process is serving.

    Returns a copy of the status: whether the LoRA adapter is applied, which one (path, r, alpha),
    or why not. Reflects the most recent get_llm() load."""
    return dict(_adapter_status)


def unload_llm() -> None:
    """Release the resident LLM and reclaim GPU memory. Not used per request.

    Callers must `del` their own references to the model and tokenizer first, because this
    function cannot delete a caller's variables (see Issue 163)."""
    global _model, _tokenizer, _adapter_status

    from app.services.gpu_utils import free_gpu  # noqa: E402

    _model = None
    _tokenizer = None
    _adapter_status = {"configured_path": LLM_ADAPTER_PATH, "applied": False, "reason": "LLM not loaded yet"}
    free_gpu()


# Serialises generation while an adapter is loaded, see _generation_context().
_generate_lock = threading.Lock()


@contextmanager
def _generation_context(model, use_adapter: bool):
    """Select which weights a `model.generate()` call runs on.

    `use_adapter=True` runs with whatever `get_llm()` loaded (the LoRA adapter if applied, base
    weights otherwise). `use_adapter=False` switches the adapter off with PEFT's `disable_adapter()`
    on the same resident model, so no second copy is loaded into VRAM; this matched a separate
    base-only load on 7 of 7 test prompts.

    A lock is needed because `disable_adapter()` changes shared state on the one model object, and
    its exit logic depends on the state captured at entry, so two overlapping contexts could
    re-enable the adapter under a call that asked for base weights. Requests run on a thread pool
    (`request_hint` is a sync endpoint), so while an adapter is loaded every generation takes
    `_generate_lock`, including adapter-on calls. Concurrent generations therefore run one at a
    time, which they effectively did anyway on a single 8GB GPU.

    With no adapter loaded the model has no `disable_adapter()`, so this is a no-op."""
    if not hasattr(model, "disable_adapter"):
        yield
        return
    with _generate_lock:
        if use_adapter:
            yield
        else:
            with model.disable_adapter():
                yield


def generate(
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int = 200,
    do_sample: bool = False,
    temperature: float = 1.0,
    use_adapter: bool = True,
) -> str:
    """Run one chat generation against the resident LLM and return the decoded reply.

    Args:
        do_sample: False (greedy) for deterministic calls such as SymPy expression extraction.
            True, with `temperature`, where variation between passes is needed; greedy decoding
            would return the same output every time.
        use_adapter: True runs with the loaded LoRA adapter, if any; False uses base weights only.
            The adapter was trained only on Tier 1/Tier 2 hint writing, so only hint-writing calls
            in gate.py keep the default. See `_generation_context()`.
    """
    model, tokenizer = get_llm()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt",
    )["input_ids"].to(model.device)

    gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.95  # standard pairing with temperature sampling

    with _generation_context(model, use_adapter):
        output = model.generate(input_ids, **gen_kwargs)
    return tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()


def generate_multiple(
    system_prompt: str,
    user_prompt: str,
    num_return_sequences: int,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    use_adapter: bool = True,
) -> list[str]:
    """Sample several replies to one prompt in a single call, for the self-consistency gate.

    The prompt is identical across passes, so `num_return_sequences` runs the prefill once and
    samples N continuations, instead of repeating the prefill per pass (see Issue 170).

    Always sampled: greedy decoding would return N identical sequences. Each sequence stops at its
    own EOS, and `max_new_tokens` applies per sequence. `use_adapter` behaves as in `generate()`.
    """
    model, tokenizer = get_llm()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt",
    )["input_ids"].to(model.device)

    with _generation_context(model, use_adapter):
        output = model.generate(
            input_ids, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=0.95, num_return_sequences=num_return_sequences,
        )
    return [
        tokenizer.decode(seq[input_ids.shape[-1]:], skip_special_tokens=True).strip()
        for seq in output
    ]
