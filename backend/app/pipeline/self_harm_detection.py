"""
Request-path safety detection for student input and generated hints.

Combines two signals, ShieldGemma-2B and the keyword lexicon in `self_harm_lexicon.py`, into
`check_safety()`. It is called on the input side by `_create_submission_and_confirmation_response()`
and on the output side by `_finalize_and_persist_hint()`.

Design decisions:
- CPU placement: ShieldGemma-2B peaks at about 5.15GB of VRAM, which cannot share the 8GB GPU
  budget with the LLM or VLM, and there is no swap coordination for it. Running on CPU keeps it
  at zero VRAM, like the embedding model in `question_matching.py`, at the cost of latency
  (see Issue 204).
- Accepted latency: about 3.7s typical and 7.1s worst case per call, around 10.5s if both input
  and output checks run. Batching the three policy checks gave only about a 3% gain, so the
  simpler sequential version is kept. GPU swap coordination is the long-term fix but is deferred,
  since it would add three-way eviction risk to nearly every request (see Issue 205).
- Categorised results: `check_safety()` returns a `SafetyCheckResult` with one of four categories
  (`self_harm`, `sexually_explicit`, `dangerous_content`, `harassment`), so `safety_response.py`
  can choose a category-appropriate message rather than always showing the crisis response.
- `dangerous_content` is treated as self-harm: ShieldGemma's "Dangerous Content" policy covers
  both harming oneself and harming others, and no available signal separates them (a paracetamol
  overdose question and a threat to hurt a bully both fire only on this policy). Mislabelling a
  self-harm case as a generic refusal is worse than showing crisis resources to a
  violence-to-others case, so both categories route to the crisis response (see Issue 207).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch.nn.functional import softmax
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.pipeline.self_harm_lexicon import check_self_harm_lexicon

# Deliberately low (conservative) threshold, fixed from the classifier validation and not re-tuned
# here. The separation data behind it is in
# `evaluation/model_selection/safety_classifier/results.md` (see Issue 202).
SHIELDGEMMA_SELF_HARM_THRESHOLD = 0.1

_GENERIC_USER_TURN = "Can you help me with my maths homework?"

# Policy text quoted verbatim from the ShieldGemma model card (google/shieldgemma-2b), identical
# to the text used in the validation runs.
_SHIELDGEMMA_POLICIES = {
    "dangerous_content": (
        '"No Dangerous Content": The {actor} shall not contain or seek generation of content '
        "that harming oneself and/or others (e.g., accessing or building firearms and explosive "
        "devices, promotion of terrorism, instructions for suicide)."
    ),
    "harassment": (
        '"No Harassment": The {actor} shall not contain or seek generation of content that is '
        "malicious, intimidating, bullying, or abusive content targeting another individual "
        "(e.g., physical threats, denial of tragic events, disparaging victims of violence)."
    ),
    "sexually_explicit": (
        '"No Sexually Explicit Information": The {actor} shall not contain or seek generation of '
        "content that contains references to sexual acts or other lewd content (e.g., sexually "
        "graphic descriptions, content aimed at causing arousal)."
    ),
}

_model: AutoModelForCausalLM | None = None
_tokenizer: AutoTokenizer | None = None


@dataclass
class SafetyCheckResult:
    """Result of `check_safety()`.

    `category` is one of `self_harm`, `sexually_explicit`, `dangerous_content` or `harassment`, or
    `None` when `flagged` is `False`. The category lets the response layer pick an appropriate
    message instead of routing every flag to the self-harm response (see Issue 206).
    """
    flagged: bool
    category: Optional[str]


def _get_shieldgemma():
    """Return the lazily loaded ShieldGemma model and tokenizer, CPU-resident for the process.

    CPU placement is explained in the module docstring. A batched variant of
    `score_shieldgemma()` was measured at about 3% faster (3.47-7.00s vs 3.68-7.13s) with identical
    results on all 54 gold-set cases, so the simpler sequential version is kept. See
    `evaluation/model_selection/safety_classifier/results.md` for the comparison.
    """
    global _model, _tokenizer
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained("google/shieldgemma-2b")
        # Load without a device map, then move explicitly to CPU. Verified with `nvidia-smi` to
        # use no GPU memory.
        _model = AutoModelForCausalLM.from_pretrained(
            "google/shieldgemma-2b", torch_dtype=torch.float32,
        ).to("cpu")
    return _model, _tokenizer


def _shieldgemma_prompt(text: str, direction: str, category: str) -> str:
    policy = _SHIELDGEMMA_POLICIES[category]
    if direction == "input":
        policy_text = policy.format(actor="prompt")
        return f"""You are a policy expert trying to help determine whether a user prompt is in violation of the defined safety policies.

<start_of_turn>
Human Question: {text.strip()}
<end_of_turn>

Our safety principle is defined in the below:

{policy_text}

Does the human question violate the above principle? Your answer must start with 'Yes' or 'No'."""
    policy_text = policy.format(actor="chatbot response").replace(
        "shall not contain or seek generation of", "shall not generate",
    )
    return f"""You are a policy expert trying to help determine whether an AI response to a prompt is in violation of the defined safety policies.

<start_of_turn>
Human Question: {_GENERIC_USER_TURN}
<end_of_turn>

<start_of_turn>
Chatbot Response: {text.strip()}
<end_of_turn>

Our safety principle is defined in the below:

{policy_text}

Does the chatbot response violate the above principle? Your answer must start with 'Yes' or 'No'."""


def score_shieldgemma(text: str, direction: str) -> tuple[float, Optional[str]]:
    """Score `text` against each ShieldGemma policy and return the highest violation.

    Uses the same method and policy texts as the validation script
    `evaluation/model_selection/safety_classifier/run_combined_signal_validation.py`.

    Returns:
        `(max_violation, fired_category)`: the highest "Yes" probability across the three
        policies and the policy that produced it (`None` if all scored 0).
    """
    model, tokenizer = _get_shieldgemma()
    vocab = tokenizer.get_vocab()
    yes_id, no_id = vocab["Yes"], vocab["No"]
    max_violation = 0.0
    fired_category = None
    for category in _SHIELDGEMMA_POLICIES:
        prompt = _shieldgemma_prompt(text, direction, category)
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        selected = logits[0, -1, [yes_id, no_id]]
        probs = softmax(selected, dim=0)
        violation = probs[0].item()
        if violation > max_violation:
            max_violation = violation
            fired_category = category
    return max_violation, fired_category


def check_safety(text: str, direction: str) -> SafetyCheckResult:
    """Run the combined safety check and return whether, and in which category, it flagged.

    OR-combines `check_self_harm_lexicon()` with `score_shieldgemma()` against
    `SHIELDGEMMA_SELF_HARM_THRESHOLD`. Category resolution, in priority order:
    1. Lexicon match: always `"self_harm"`, since the lexicon is self-harm-specific.
    2. ShieldGemma's fired policy (`"sexually_explicit"`, `"harassment"` or
       `"dangerous_content"`; see the module docstring for how `dangerous_content` is handled).
    3. Neither fires: `flagged=False`, `category=None`.

    Args:
        direction: `"input"` or `"output"`, matching `SafetyFlag.flagged_input_or_output`.

    Raises:
        ValueError: if `direction` is not `"input"` or `"output"`.

    Measured recall across the validation sets is 9/10, so this reduces but does not eliminate
    missed cases (see Issue 202).
    """
    if direction not in ("input", "output"):
        raise ValueError(f"direction must be 'input' or 'output', got {direction!r}")

    lexicon_hit = check_self_harm_lexicon(text) is not None
    if lexicon_hit:
        return SafetyCheckResult(flagged=True, category="self_harm")

    sg_prob, sg_category = score_shieldgemma(text, direction)
    if sg_prob > SHIELDGEMMA_SELF_HARM_THRESHOLD:
        return SafetyCheckResult(flagged=True, category=sg_category)

    return SafetyCheckResult(flagged=False, category=None)
