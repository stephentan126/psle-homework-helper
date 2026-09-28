"""
The Socratic Safety Gate: decides how much help a hint may give by verifying the model's answer
before any hint is generated. Three signals:
1. SymPy exact check for sum-type questions (`sympy_exact_check()`).
2. Multi-sample self-consistency plus answer-key match for reasoning questions (`self_consistency_check()`).
3. Method alignment against the worked solution (`method_alignment_check()`).
All arithmetic is evaluated by SymPy; the LLM's own computed numbers are never trusted.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Literal, Optional

import sympy
from sympy.parsing.sympy_parser import parse_expr

from app.pipeline.diagram_fallback import resolve_diagram_context
from app.pipeline.question_matching import RELATED_EXAMPLE_MIN_SIMILARITY, find_related_worked_example
from app.services.llm_service import generate, generate_multiple

# A cached `PrecomputedGateResult` row is only trusted if BOTH this version string AND the current
# `LLM_MODEL_PATH` match what it was computed under. Bump this whenever the gate logic (prompts,
# thresholds, routing heuristics, answer comparison) changes in a way that could change an outcome,
# so stale cached results fail closed to a live recompute instead of being served under old rules.
# The current value follows the single-letter litre unit fix in `strip_units()` (see Issue 344).
GATE_VERSION = "gate_v4_2026-09-19_landmine344"


def compute_gate_content_fingerprint(
    question_text: str, answer_value: str,
    has_diagram: bool, worked_solution_text: Optional[str],
) -> str:
    """Fingerprint the gate's input fields so cached results are invalidated if content changes.

    A cached `PrecomputedGateResult` row is only trusted if this fingerprint, recomputed from the
    question's current row, matches the one stored at write time. This complements the
    `GATE_VERSION`/`LLM_MODEL_PATH` check by also covering edits to the question itself.

    SHA-256 over the four fields, each length-prefixed so content cannot collide across a field
    boundary ("ab"+"c" never hashes the same as "a"+"bc"). A `None` worked solution uses a
    sentinel so it never collides with an empty string.
    """
    parts = [
        question_text,
        answer_value,
        "1" if has_diagram else "0",
        "\x00NONE\x00" if worked_solution_text is None else worked_solution_text,
    ]
    joined = "\x1f".join(f"{len(p)}:{p}" for p in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()

# =================================================================================================
# Routing heuristic: "sum-type" versus "reasoning" questions.
#
# A deliberately narrow heuristic, not a topic classifier. A question is "sum" if its extracted
# `answer_value` parses as a bare number, fraction or decimal (units handled separately) AND the
# question text has no multi-sentence narrative setup (word problems almost always need at least
# two clauses to state the scenario). Ambiguous or malformed cases fall through to "reasoning",
# the more conservative and more expensive path, never the reverse.
# =================================================================================================

_UNIT_SUFFIX_RE = re.compile(
    r"\s*(cm2|cm3|cm|m2|m3|m/min|m/s|km|kg|g|litre|liter|ml|min|hours?|hrs?|\$|%|°)\s*$",
    re.IGNORECASE,
)
# A leading '$' ("$36") is not covered by the end-anchored suffix regex above, so it would fall
# through to the literal-string fallback in `_values_equal()` and never match a bare '36'. Kept as
# a separate prefix check because '$' also appears as a trailing unit in the corpus. Only applied
# when the string contains exactly one '$', so compound answers containing '$' elsewhere are left
# alone (see Issue 293).
_LEADING_DOLLAR_RE = re.compile(r"^\s*\$\s*")
# The corpus renders area/volume units with superscripts ('54 cm²'), which the ASCII `cm2`/`m2`/`m3`
# branches above would not match. In every corpus row the superscript follows a unit letter, never
# a bare digit, so folding to ASCII cannot reinterpret an exponent. Folded before the unit checks.
_SUPERSCRIPT_FOLD = str.maketrans({"²": "2", "³": "3"})
# Single-letter litre unit ('6ℓ', '4L', '1.25 l'), not covered by the suffix regex above. Anchored
# to a preceding digit so it cannot strip the last letter off a word answer ending in 'l'. Case is
# spelled out ([ℓlL]) rather than relying on IGNORECASE (see Issue 344).
_LITRE_SYMBOL_RE = re.compile(r"(?<=\d)\s*([ℓlL])\s*$")


def strip_units(answer_str: str) -> "tuple[str, Optional[str]]":
    """Split an answer string into its numeric part and an optional unit.

    Corpus answers mix bare values ('48', '1/16') with unit-bearing ones ('753.6 cm', '200m/min',
    '$36', '54 cm²', '1.25 l'). SymPy compares only the numeric part; unit correctness is not
    checked by this gate. Superscripts are folded to ASCII first, then a leading '$', a trailing
    unit suffix and a digit-anchored litre symbol are tried in turn.

    Returns:
        (numeric_part, unit), where unit is None if no unit was found.
    """
    s = answer_str.strip().translate(_SUPERSCRIPT_FOLD)
    dollar_m = _LEADING_DOLLAR_RE.match(s) if s.count("$") == 1 else None
    if dollar_m:
        return s[dollar_m.end():].strip(), "$"
    m = _UNIT_SUFFIX_RE.search(s)
    if m:
        return s[: m.start()].strip(), m.group(1)
    litre_m = _LITRE_SYMBOL_RE.search(s)  # only reached when no suffix above matched
    if litre_m:
        return s[: litre_m.start()].strip(), litre_m.group(1)
    return s, None


_MCQ_OPTION_RE = re.compile(r"(?:\(1\)|\b1\))")
_MCQ_OPTION_4_RE = re.compile(r"(?:\(4\)|\b4\))")


def is_mcq_with_unresolved_index(question_text: str, real_answer: str) -> bool:
    """Return True for an MCQ whose stored answer is a bare option index ('1' to '4').

    About 30% of answerable corpus questions have options (1) to (4) in the text and store the
    option INDEX rather than the option's value. A SymPy check of a computed value against an
    index would be meaningless ('3' versus 302510), so these questions are routed to
    self-consistency instead. The model does not reliably answer with the index either (it
    sometimes states the option's value), so `self_consistency_check()` resolves both forms via
    `_parse_mcq_options()` before comparing (see Issue 175).
    """
    has_option_1 = bool(_MCQ_OPTION_RE.search(question_text))
    has_option_4 = bool(_MCQ_OPTION_4_RE.search(question_text))
    answer_is_bare_index = real_answer.strip() in ("1", "2", "3", "4")
    return has_option_1 and has_option_4 and answer_is_bare_index


_MCQ_OPTION_MARKER_RE = re.compile(r"\(([1-4])\)|(?<![\d.])([1-4])\)")


def _parse_mcq_options(question_text: str) -> dict:
    """Parse MCQ options from question text into an index-to-value dict.

    Splits the text at each `(1)`/`1)`-style marker and takes the first line up to the next marker
    as that option's value, matching the corpus's one-option-per-line layout. For example
    "...\\n(1) 1\\n(2) 5\\n(3) 8\\n(4) 9" gives `{'1': '1', '2': '5', '3': '8', '4': '9'}`.

    Returns:
        A possibly partial or empty dict when options were not captured cleanly. Callers must not
        assume all four keys are present.
    """
    options: dict = {}
    matches = list(_MCQ_OPTION_MARKER_RE.finditer(question_text))
    for i, m in enumerate(matches):
        idx = m.group(1) or m.group(2)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(question_text)
        segment = question_text[start:end]
        value = segment.splitlines()[0].strip() if segment.strip() else ""
        if idx not in options and value:  # keep the first occurrence if a marker repeats
            options[idx] = value
    return options


def _canonicalize_mcq_answer(claimed_or_key: str, options: dict) -> str:
    """Map an option index to its option value, so index and value forms compare as equal.

    If `claimed_or_key` is an index ('1' to '4') present in `options`, return that option's value,
    so '3' and '8' for the same option match downstream. Anything else is returned unchanged.
    """
    stripped = claimed_or_key.strip()
    if stripped in ("1", "2", "3", "4") and stripped in options:
        return options[stripped]
    return claimed_or_key


def classify_question_type(
    question_text: str, real_answer: str, has_diagram: bool = False,
) -> Literal["sum", "reasoning"]:
    # MCQ questions with an index-form answer key are always routed to reasoning, as a hard
    # override (see is_mcq_with_unresolved_index()).
    if is_mcq_with_unresolved_index(question_text, real_answer):
        return "reasoning"

    # Diagram-dependent questions cannot be solved from text alone. On the SymPy path the LLM's
    # guess can match the key by numeric coincidence, a false accept. Self-consistency is likely
    # to disagree across passes and cap safely to Tier 1 instead (see Issue 172).
    if has_diagram:
        return "reasoning"

    numeric_part, _ = strip_units(real_answer)
    try:
        parse_expr(numeric_part.replace(":", "/"))  # tolerate a bare ratio-shaped answer too
        answer_is_bare_numeric = True
    except Exception:
        answer_is_bare_numeric = False

    # A word problem usually states its scenario across at least two sentences before asking the
    # question; a sum-type question that reduces to one calculation is typically one clause.
    #
    # Parenthetical asides such as "(Take pi = 3.14)" are computation instructions, not narrative,
    # so they are stripped before counting. LaTeX inline-math delimiters `\(...\)` are removed
    # first, since their parentheses would otherwise confuse the paren-stripping regex.
    text_no_latex_delims = question_text.strip().replace(r"\(", "").replace(r"\)", "")
    text_without_parens = re.sub(r"\([^)]*\)", "", text_no_latex_delims)
    sentence_count = len([s for s in re.split(r"[.!?]\s+", text_without_parens) if s.strip()])

    if answer_is_bare_numeric and sentence_count <= 2:
        return "sum"
    return "reasoning"


# =================================================================================================
# Signal 1: SymPy exact-check path (sum-type questions).
# =================================================================================================

_EXPRESSION_EXTRACTION_PROMPT = (
    "You will be given a Primary 6 maths question that reduces to a single arithmetic "
    "calculation. Your ONLY job is to write that calculation as a plain arithmetic expression "
    "(numbers, +, -, *, /, parentheses, and Python-style fractions like 3/4 only). "
    "IMPORTANT: the question itself may be written using LaTeX notation (\\frac{}{}, \\div, "
    "\\times, \\(...\\)) — do NOT copy that notation into your answer. Convert it: \\frac{a}{b} "
    "becomes a/b, \\div becomes /, \\times becomes *. "
    "Do NOT compute the result. Do NOT include units. Do NOT include words or explanation. "
    "Output ONLY the plain expression, nothing else."
)

# Despite the prompt, the model sometimes echoes the question's LaTeX (`\frac{3}{4} \div 12`),
# which `parse_expr()` cannot tokenize. Fractions and division in the corpus are always written in
# LaTeX, so common commands are normalized to plain syntax before parsing as a second defence.
_LATEX_FRAC_RE = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")


def _normalize_latex_expression(expr_str: str) -> str:
    s = expr_str
    s = s.replace(r"\(", "").replace(r"\)", "")
    s = s.replace(r"\div", "/").replace(r"\times", "*").replace(r"\cdot", "*")
    s = s.replace(r"\pi", "pi")
    # \frac{a}{b} -> (a)/(b); applied repeatedly to handle nested fractions.
    while _LATEX_FRAC_RE.search(s):
        s = _LATEX_FRAC_RE.sub(r"(\1)/(\2)", s)
    return s.strip()


@dataclass
class GateResult:
    passed: bool
    tier_allowed: int  # 1 = Tier 1 only (capped); 2 = limited help; 4 = full help
    reason: str
    detail: dict = field(default_factory=dict)


def _sympy_equal(a: sympy.Expr, b: sympy.Expr) -> bool:
    """Return True if two SymPy values are mathematically equal.

    Symbolic comparison treats '1/16' and '0.0625' as equal where a string compare would not.
    Falls back to a small numeric tolerance for decimal answers such as '753.6'. The whole body
    fails closed to False: comma-containing LLM output can parse as a `sympy.Tuple`, and
    `nsimplify()` on it raises, which must cap the hint at Tier 1 rather than fail the request.
    """
    try:
        diff = sympy.nsimplify(a) - sympy.nsimplify(b)
        try:
            return sympy.simplify(diff) == 0
        except Exception:
            pass
        try:
            return abs(float(a) - float(b)) < 1e-6
        except Exception:
            return False
    except Exception:
        return False


def sympy_exact_check(question_text: str, real_answer: str) -> GateResult:
    """Gate signal 1: check the model's calculation with SymPy against the answer key.

    The LLM is asked only to write the calculation, never to compute it. SymPy evaluates the
    expression and compares it with the stored `answer_value` from the printed answer key.

    Units are not verified: `strip_units()` discards them, so a right number with a wrong unit
    would still pass.
    """
    # Expression extraction is not hint writing, so it runs on base weights; the LoRA adapter was
    # trained only on hints (see Issue 353).
    raw = generate(_EXPRESSION_EXTRACTION_PROMPT, question_text, max_new_tokens=40, do_sample=False, use_adapter=False)
    # Output is often wrapped in backticks or followed by prose despite the instruction: take the
    # first line and strip common wrapping.
    expr_str = raw.strip().splitlines()[0].strip().strip("`").strip()
    expr_str = re.sub(r"^\$|\$$", "", expr_str).strip()  # occasional LaTeX $ wrapping
    expr_str = _normalize_latex_expression(expr_str)  # model may still emit LaTeX

    try:
        llm_expr = parse_expr(expr_str, evaluate=True)
    except Exception as e:
        return GateResult(
            passed=False, tier_allowed=1,
            reason=f"LLM expression did not parse as valid arithmetic: {expr_str!r} ({e})",
            detail={"raw_generation": raw, "expr_str": expr_str},
        )

    numeric_answer, unit = strip_units(real_answer)
    try:
        real_value = parse_expr(numeric_answer.replace(":", "/"))
    except Exception as e:
        return GateResult(
            passed=False, tier_allowed=1,
            reason=f"Real answer_value did not parse as a comparable number: {real_answer!r} ({e})",
            detail={"raw_generation": raw, "expr_str": expr_str},
        )

    try:
        llm_value = sympy.nsimplify(llm_expr)
    except Exception as e:
        # A validly parsed but non-numeric object (e.g. a Tuple) can make nsimplify() raise.
        # Fail closed.
        return GateResult(
            passed=False, tier_allowed=1,
            reason=f"LLM expression {expr_str!r} parsed but could not be simplified for "
                   f"comparison ({e}) — treated as a real disagreement (Section 5's safe default).",
            detail={"raw_generation": raw, "expr_str": expr_str},
        )
    matches = _sympy_equal(llm_value, real_value)
    return GateResult(
        passed=matches,
        tier_allowed=4 if matches else 1,
        reason=(
            f"SymPy-confirmed: LLM expression {expr_str!r} evaluates to {llm_value}, matches "
            f"real answer {real_value}{(' ' + unit) if unit else ''}."
        ) if matches else (
            f"SymPy check FAILED: LLM expression {expr_str!r} evaluates to {llm_value}, does "
            f"NOT match real answer {real_value}{(' ' + unit) if unit else ''}."
        ),
        detail={
            "raw_generation": raw, "expr_str": expr_str,
            "llm_value": str(llm_value), "real_value": str(real_value), "unit": unit,
        },
    )


# =================================================================================================
# Signal 2: multi-sample self-consistency (reasoning-type and MCQ-shaped questions; see
# is_mcq_with_unresolved_index() for why MCQ is routed here rather than to SymPy).
# =================================================================================================

_SOLVE_PROMPT = (
    "You are solving a Primary 6 PSLE Mathematics question. Work through it step by step, then "
    "end your response with a line in EXACTLY this format: FINAL ANSWER: <value>. The <value> "
    "must be a plain number, fraction (e.g. 3/4), or option number if this is a multiple-choice "
    "question — no units, no words, no extra text on that final line."
)

_FINAL_ANSWER_RE = re.compile(r"FINAL ANSWER:\s*(.+)", re.IGNORECASE)


def extract_claimed_answer(generation: str) -> Optional[str]:
    """Extract the value from the last 'FINAL ANSWER: <value>' line of a generation.

    Returns:
        The claimed answer, or None if the model did not follow the format. Callers treat None
        as a no-answer case, which caps the hint at Tier 1.
    """
    matches = _FINAL_ANSWER_RE.findall(generation)
    if not matches:
        return None
    return matches[-1].strip().rstrip(".").strip()


def _values_equal(claimed: str, real_answer: str) -> bool:
    """Compare two answer strings with SymPy after stripping units and LaTeX.

    '48', '48.0' and '48 sweets' all match '48'. Falls back to a case-insensitive literal match
    if either side does not parse.
    """
    claimed_numeric, _ = strip_units(claimed)
    real_numeric, _ = strip_units(real_answer)
    try:
        claimed_value = parse_expr(_normalize_latex_expression(claimed_numeric).replace(":", "/"))
        real_value = parse_expr(_normalize_latex_expression(real_numeric).replace(":", "/"))
    except Exception:
        return claimed.strip().lower() == real_answer.strip().lower()  # last-resort literal match
    return _sympy_equal(claimed_value, real_value)


def self_consistency_check(
    question_text: str, real_answer: Optional[str] = None, n_passes: int = 2,
    temperature: float = 0.8, max_new_tokens: int = 450,
) -> GateResult:
    """Gate signal 2: solve the question several times and require every pass to agree.

    Passes are temperature-sampled (greedy decoding would repeat the same output). All passes must
    agree with each other AND match the answer key; any disagreement caps at Tier 1 even if one
    pass matched. A single outlier failing the check is the intended safety property.

    `n_passes` defaults to 2 rather than 3 because each multi-step solve takes roughly 10 to 20
    seconds on the target hardware, and three passes pushed total latency to around 50 seconds.
    The passes are generated in one batched `generate_multiple()` call, since the prompt is
    identical, which avoids repeating the prefill (see Issue 170).

    Args:
        real_answer: The answer key, or None for weak mode (no corpus match). In weak mode only
            agreement between passes is checked and the result is always capped at Tier 1.
    """
    import time

    # 450 tokens: at 300, passes were truncated mid-working before the "FINAL ANSWER:" line.
    # The limit applies per sequence within the batch.
    start = time.time()
    # Solving passes run on base weights; the adapter was trained only on hint writing and showed
    # no benefit for solving. Only generate_hint() uses the adapter (see Issue 353).
    raw_generations = generate_multiple(
        _SOLVE_PROMPT, question_text, num_return_sequences=n_passes,
        max_new_tokens=max_new_tokens, temperature=temperature, use_adapter=False,
    )
    total_batch_time = time.time() - start
    pass_times = [total_batch_time] * n_passes  # batched, so per-pass times are not separable;
                                                  # use detail["batched_generation_time_s"].
    claimed_answers = [extract_claimed_answer(raw) for raw in raw_generations]

    if any(a is None for a in claimed_answers):
        return GateResult(
            passed=False, tier_allowed=1,
            reason=f"At least one pass produced no extractable FINAL ANSWER — treated as a "
                   f"real disagreement/no-answer case (Section 5's own safe default).",
            detail={
                "claimed_answers": claimed_answers, "raw_generations": raw_generations,
                "pass_times_s": pass_times, "batched_generation_time_s": total_batch_time,
            },
        )

    # Weak mode: a photo-submitted question with no confident corpus match, so no answer key.
    # Only agreement between passes can be checked. `tier_allowed` is hard-capped at 1 whether or
    # not the passes agree, enforced here rather than left to callers who could forget it.
    # Agreement is still recorded in `reason`/`detail` for the audit trail (see Issue 180).
    if real_answer is None:
        all_agree = all(_values_equal(claimed_answers[0], a) for a in claimed_answers[1:])
        if all_agree:
            reason = (
                f"Weak mode (no real answer key — Issue 180 addendum): all {n_passes} "
                f"passes agreed on {claimed_answers[0]!r}. Still hard-capped at Tier 1 — an "
                f"unverified answer key can never justify Tier 2/3 help, regardless of "
                f"agreement strength."
            )
        else:
            reason = (
                f"Weak mode (no real answer key — Issue 180 addendum): passes disagreed — "
                f"{claimed_answers}. Capped at Tier 1 (same real ceiling either way)."
            )
        return GateResult(
            passed=all_agree, tier_allowed=1, reason=reason,
            detail={
                "claimed_answers": claimed_answers, "raw_generations": raw_generations,
                "pass_times_s": pass_times, "all_agree": all_agree,
                "batched_generation_time_s": total_batch_time, "weak_mode": True,
            },
        )

    # For index-form MCQs, resolve both the claimed answers and the key to option values before
    # comparing, so an index claim ('3') and a value claim ('8') for the same option agree. Any
    # non-empty parsed option dict is used: an option that failed to parse (often a blank or
    # image-only option 1) simply stays unresolved, rather than discarding options that did parse
    # (see Issue 177). Non-MCQ questions keep the plain comparison.
    mcq_options: dict = {}
    if is_mcq_with_unresolved_index(question_text, real_answer):
        parsed = _parse_mcq_options(question_text)
        if parsed:
            mcq_options = parsed

    if mcq_options:
        comparison_answers = [_canonicalize_mcq_answer(a, mcq_options) for a in claimed_answers]
        comparison_real_answer = _canonicalize_mcq_answer(real_answer, mcq_options)
    else:
        comparison_answers = claimed_answers
        comparison_real_answer = real_answer

    # Agreement is checked pairwise with SymPy equality (not string equality), so equivalent but
    # differently written values count as agreeing; the key match is checked separately.
    all_agree = all(
        _values_equal(comparison_answers[0], a) for a in comparison_answers[1:]
    )
    matches_key = all_agree and _values_equal(comparison_answers[0], comparison_real_answer)

    mcq_note = (
        f" (MCQ index-vs-value resolved: raw claims {claimed_answers} -> {comparison_answers}, "
        f"real key {real_answer!r} -> {comparison_real_answer!r}, Issue 175)"
        if mcq_options else ""
    )
    passed = all_agree and matches_key
    if passed:
        reason = (
            f"Self-consistency CONFIRMED: all {n_passes} passes agreed on "
            f"{comparison_answers[0]!r}, matching the real answer {comparison_real_answer!r}."
            f"{mcq_note}"
        )
    elif not all_agree:
        reason = (
            f"Self-consistency FAILED: passes disagreed — {comparison_answers} — capped at Tier "
            f"1 even though at least one may have matched the real answer "
            f"{comparison_real_answer!r} (Section 5's own explicit rule).{mcq_note}"
        )
    else:
        reason = (
            f"Self-consistency passes agreed on {comparison_answers[0]!r} but this does NOT "
            f"match the real answer {comparison_real_answer!r}.{mcq_note}"
        )

    return GateResult(
        passed=passed, tier_allowed=4 if passed else 1, reason=reason,
        detail={
            "claimed_answers": claimed_answers, "raw_generations": raw_generations,
            "pass_times_s": pass_times, "all_agree": all_agree, "matches_key": matches_key,
            "batched_generation_time_s": total_batch_time,
            "mcq_options_resolved": mcq_options if mcq_options else None,
            "comparison_answers": comparison_answers if mcq_options else None,
            "comparison_real_answer": comparison_real_answer if mcq_options else None,
        },
    )


# =================================================================================================
# Signal 3: method alignment. Only possible when a worked solution exists: without a reference
# method, "right answer, wrong method" cannot be detected, which is why questions with no worked
# solution are treated as lower confidence even when the key matches.
# =================================================================================================

_METHOD_ALIGNMENT_PROMPT = (
    "You are comparing two solutions to the same Primary 6 maths problem: a REFERENCE solution "
    "(verified correct) and a MODEL solution. Answer with exactly one word, nothing else: MATCH "
    "if they use fundamentally the same solving METHOD (unit method, ratio method, algebra with "
    "an unknown, etc — even if variable names, wording, or the order of steps differ), or "
    "MISMATCH if they use a genuinely different method, even if both reach the same final answer."
)


def method_alignment_check(model_solution_text: str, worked_solution_text: str) -> GateResult:
    """Gate signal 3: detect 'right answer, wrong method'.

    A passing SymPy or self-consistency check only confirms the final number. The LLM is used as a
    structured comparator, asked for a single word (MATCH or MISMATCH), against the worked solution.
    """
    comparison_input = (
        f"REFERENCE SOLUTION:\n{worked_solution_text}\n\nMODEL SOLUTION:\n{model_solution_text}"
    )
    # A comparison task, not hint writing, so it runs on base weights (see Issue 353).
    raw = generate(_METHOD_ALIGNMENT_PROMPT, comparison_input, max_new_tokens=10, do_sample=False, use_adapter=False)
    raw_upper = raw.strip().upper()
    aligned = "MATCH" in raw_upper and "MISMATCH" not in raw_upper
    return GateResult(
        passed=aligned, tier_allowed=4 if aligned else 2,  # a method mismatch is less severe
        # than a wrong or missing answer, since the final value was independently confirmed, so it
        # caps at "limited help" (tier 2) rather than Tier 1.
        reason=(
            f"Method alignment CONFIRMED: model's solution uses the same method as the "
            f"verified worked solution."
        ) if aligned else (
            f"Method alignment FAILED: model reached the right answer via a different method "
            f"than the verified worked solution (raw judge output: {raw!r})."
        ),
        detail={"raw_judge_output": raw, "model_solution_text": model_solution_text},
    )


# =================================================================================================
# Hint generation from the verified path. The hint call receives the verified solution or worked
# example as context and never builds a hint from a second, unverified answer.
# =================================================================================================

_TIER1_SYSTEM_PROMPT = (
    "You are a patient Socratic maths tutor helping a Singapore Primary 6 student (age 11-12) "
    "with a PSLE Mathematics question. You must give a TIER 1 hint ONLY. A Tier 1 hint restates "
    "the question in your own simple words, to check the student understands what is being "
    "asked, and then STOPS. Rules, all mandatory: do not solve the question; do not show any "
    "working, calculation, or steps; do not reveal the final numeric answer, even partially or "
    "as a range; do not ask a leading question about method. Just restate what the question is "
    "asking, in plain words a Primary 6 student would understand, then stop."
)  # same prompt used in the hint-quality evaluation during model selection (see Issue 165).

_GROUNDED_HINT_PROMPT_TEMPLATE = (
    "You are a patient Socratic maths tutor helping a Singapore Primary 6 student. The question "
    "has ALREADY been verified correct — here is the verified solution path, which you must "
    "build your hint FROM, not ignore or replace with your own fresh calculation:\n\n"
    "VERIFIED SOLUTION:\n{verified_content}\n\n"
    "Give a Tier 2 hint: a conceptual nudge toward the FIRST STEP of the verified solution above, "
    "in plain words a Primary 6 student would understand. Do NOT give the final numeric answer. "
    "Do NOT walk through the full solution — just point at the first real step or concept, then "
    "stop."
)

# Tier 3: present a related, already-solved worked example (retrieved excluding the current
# question) as a parallel case. The student's own final answer must still never be revealed; the
# parallel example's answer belongs to a different published question, so showing it is not a leak.
_TIER3_PARALLEL_EXAMPLE_PROMPT_TEMPLATE = (
    "You are a patient Socratic maths tutor helping a Singapore Primary 6 student who is STILL "
    "STUCK after already receiving a Tier 2 hint on their own question. Their own current question "
    "has ALREADY been verified correct, but you must NOT reveal its final numeric answer, even "
    "partially or as a range, and you must NOT solve or restate their own current question "
    "directly — that would defeat the point of a parallel example.\n\n"
    "Instead, present the following DIFFERENT, ALREADY-SOLVED example question as an illustrative "
    "parallel case. Walk through its own method in plain words a Primary 6 student would "
    "understand, so the student can see how a similar problem is approached and try applying the "
    "same idea back to their own question themselves:\n\n"
    "PARALLEL EXAMPLE (a different, already-solved question, not the student's own):\n"
    "{parallel_example_solution}\n\n"
    "Do NOT restate or solve the student's own current question. Do NOT reveal the student's own "
    "current question's final answer. Present ONLY the parallel example's own method, then stop."
)


# Highest hint tier each tier_allowed value permits. Defined once and shared by `generate_hint()`
# and the escalation endpoint so the two can never disagree on the ceiling.
_TIER_CEILING_BY_TIER_ALLOWED = {1: 1, 2: 2, 4: 3}


def tier_ceiling(tier_allowed: Optional[int]) -> int:
    """Return the highest hint tier allowed for a gate's `tier_allowed` value.

    Used by the routes and the escalation endpoint. `None` (no ceiling information, e.g. an older
    `hint_events` row) fails closed to ceiling 1.
    """
    if tier_allowed is None:
        return 1
    return _TIER_CEILING_BY_TIER_ALLOWED.get(tier_allowed, 1)


def generate_hint(
    question_text: str, gate_result: GateResult, worked_solution_text: Optional[str] = None,
    extra_steering: Optional[str] = None, has_diagram: bool = False,
    want_tier3: bool = False, question_id: Optional[int] = None,
    requested_tier: Optional[int] = None,
) -> dict:
    """Generate a hint grounded in whatever the gate verified.

    Routing by the gate outcome:
    - tier_allowed == 1: Tier 1 only, a restate-the-question-and-stop hint.
    - tier_allowed >= 2: a Tier 2 nudge built from the verified content, either the worked
      solution or, if none exists, the SymPy-confirmed expression or agreed answer.
    - Tier 3: a different, related worked example presented as a parallel case, only reachable
      when tier_allowed == 4.

    Grounding (`grounded_in`, `verified_content`) is computed whenever tier_allowed >= 2, even if
    a lower tier is shown, so a later escalation can rebuild a `GateResult` without re-running the
    gate. `verified_content` is None when grounded in the worked solution, since that is already
    reachable through the `questions` table.

    Args:
        extra_steering: Extra instruction appended to the prompt, used by the moderation retry
            loop. Generation is greedy, so an identical retry would reproduce the flagged output.
        has_diagram: If True, and the diagram cannot be shown to the model, a "can't see the
            figure" acknowledgment is appended to the prompt. If False, no extra work is done.
        want_tier3: Request a Tier 3 hint when `requested_tier` is None. Honoured only when
            tier_allowed == 4; a method mismatch (tier 2) is never reinforced with another example.
        question_id: The current question's id, excluded from Tier 3 retrieval.
        requested_tier: Explicit tier (1, 2 or 3), bounded down to the ceiling for tier_allowed so
            a hint is never served above the gate's confidence. None keeps the default choice.

    Returns:
        Dict with `tier`, `hint_text`, `grounded_in`, `verified_content`, `tier_allowed`,
        `tier3_degrade_reason` and `tier3_similarity_score`. If Tier 3 was attempted but no
        example met `RELATED_EXAMPLE_MIN_SIMILARITY` (or no question_id was given), the Tier 2
        hint is returned with the reason recorded. The similarity score is logged on every Tier 3
        attempt for threshold calibration.
    """
    diagram_note: Optional[str] = None
    if has_diagram:
        diagram_context = resolve_diagram_context(has_diagram)
        if not diagram_context.available:
            diagram_note = diagram_context.acknowledgment

    # Computed whenever the gate is confident, regardless of which tier is rendered below, so a
    # later escalation has grounding to use.
    grounded_in: Optional[str] = None
    verified_content: Optional[str] = None
    if gate_result.tier_allowed >= 2:
        if worked_solution_text:
            verified_content = worked_solution_text
            grounded_in = "worked_solution_text"
        else:
            # SymPy path: the confirmed expression. Self-consistency path: the agreed answers.
            # Both are verified content, never a fresh unverified generation.
            verified_content = gate_result.detail.get("expr_str") or str(
                gate_result.detail.get("claimed_answers"),
            )
            grounded_in = "sympy_confirmed_expression" if "expr_str" in gate_result.detail else (
                "self_consistency_confirmed_value"
            )
    # Persisted value: None when grounded in the worked solution, which is already stored.
    persisted_verified_content = None if grounded_in == "worked_solution_text" else verified_content

    ceiling = _TIER_CEILING_BY_TIER_ALLOWED.get(gate_result.tier_allowed, 1)
    if requested_tier is None:
        # Default: the highest tier the gate allows (Tier 3 only if requested and tier_allowed == 4).
        target_tier = 1 if gate_result.tier_allowed <= 1 else 2
        attempt_tier3 = want_tier3 and gate_result.tier_allowed == 4
    else:
        # An explicit request, bounded down to the ceiling. The escalation endpoint applies the
        # same check; this is a second line of defence, not a substitute.
        target_tier = min(requested_tier, ceiling)
        attempt_tier3 = target_tier == 3

    if target_tier <= 1:
        system_prompt = _TIER1_SYSTEM_PROMPT
        if extra_steering:
            system_prompt = f"{system_prompt}\n\n{extra_steering}"
        if diagram_note:
            system_prompt = f"{system_prompt}\n\n{diagram_note}"
        hint_text = generate(system_prompt, question_text, max_new_tokens=160, do_sample=False)
        return {
            "tier": 1, "hint_text": hint_text,
            "grounded_in": grounded_in, "verified_content": persisted_verified_content,
            "tier_allowed": gate_result.tier_allowed, "tier3_degrade_reason": None,
            "tier3_similarity_score": None,
        }

    tier3_degrade_reason: Optional[str] = None
    tier3_similarity_score: Optional[float] = None
    if attempt_tier3:
        # Retrieval runs before any Tier 2 generation: it is a cheap local embedding lookup, while
        # generate() is the expensive call, which is skipped entirely if Tier 3 succeeds.
        related = None
        if question_id is None:
            tier3_degrade_reason = "no_question_id_supplied"
        else:
            related = find_related_worked_example(question_id)
            # Capture the raw score before `related` may be reset below, so every attempt is
            # logged for threshold calibration, not just successful ones.
            tier3_similarity_score = related.match_confidence
            if (
                related.question_id is None
                or related.match_confidence is None
                or related.match_confidence < RELATED_EXAMPLE_MIN_SIMILARITY
            ):
                tier3_degrade_reason = (
                    f"no_qualifying_related_example (best_match_method={related.match_method!r}, "
                    f"best_match_confidence={related.match_confidence!r})"
                )
                related = None

        if related is not None:
            tier3_prompt = _TIER3_PARALLEL_EXAMPLE_PROMPT_TEMPLATE.format(
                parallel_example_solution=related.worked_solution_text,
            )
            if extra_steering:
                tier3_prompt = f"{tier3_prompt}\n\n{extra_steering}"
            if diagram_note:
                tier3_prompt = f"{tier3_prompt}\n\n{diagram_note}"
            tier3_hint_text = generate(
                tier3_prompt, question_text, max_new_tokens=240, do_sample=False,
            )
            return {
                "tier": 3, "hint_text": tier3_hint_text, "grounded_in": "related_worked_example",
                "tier_allowed": gate_result.tier_allowed,
                "verified_content": f"related_question_id={related.question_id}",
                "tier3_degrade_reason": None,
                "tier3_similarity_score": tier3_similarity_score,
            }

    prompt = _GROUNDED_HINT_PROMPT_TEMPLATE.format(verified_content=verified_content)
    if extra_steering:
        prompt = f"{prompt}\n\n{extra_steering}"
    if diagram_note:
        prompt = f"{prompt}\n\n{diagram_note}"
    hint_text = generate(prompt, question_text, max_new_tokens=120, do_sample=False)
    return {
        "tier": 2, "hint_text": hint_text, "grounded_in": grounded_in,
        "tier_allowed": gate_result.tier_allowed,
        "verified_content": persisted_verified_content,
        "tier3_degrade_reason": tier3_degrade_reason,
        "tier3_similarity_score": tier3_similarity_score,
    }


# =================================================================================================
# The single gate-to-hint pipeline, shared by the live hint endpoint and the offline batch job
# (scripts/extraction/precompute_gate_results.py) so the two cannot drift apart.
# =================================================================================================


def run_full_gate_pipeline(
    question_text: str,
    answer_value: str,
    has_diagram: bool = False,
    worked_solution_text: Optional[str] = None,
    requested_tier: Optional[int] = None,
    question_id: Optional[int] = None,
) -> tuple[GateResult, dict, Optional[str]]:
    """Run the full gate and generate the first hint.

    Classifies the question, runs the SymPy or self-consistency check, runs method alignment when
    the gate passed and a worked solution exists (capping the tier on a mismatch), then generates
    the hint from the verified path.

    Args:
        requested_tier, question_id: Passed through to `generate_hint()`. Live callers pass
            requested_tier=1 for the first hint; later tiers go through the escalation endpoint,
            which calls `generate_hint()` directly. The offline precompute job passes neither.

    Returns:
        (effective_gate_result, hint, self_consistency_summary); the summary is None on the
        SymPy path.
    """
    q_type = classify_question_type(question_text, answer_value, has_diagram=has_diagram)
    if q_type == "sum":
        gate_result: GateResult = sympy_exact_check(question_text, answer_value)
        self_consistency_summary = None
    else:
        gate_result = self_consistency_check(question_text, answer_value)
        self_consistency_summary = (
            f"claimed_answers={gate_result.detail.get('claimed_answers')} "
            f"all_agree={gate_result.detail.get('all_agree')} "
            f"matches_key={gate_result.detail.get('matches_key')}"
        )

    effective_gate_result = gate_result
    if gate_result.tier_allowed >= 2 and worked_solution_text:
        model_solution = None
        raw_generations = gate_result.detail.get("raw_generations")
        if raw_generations:  # self-consistency path: use one of the agreeing passes
            model_solution = raw_generations[0]
        elif "raw_generation" in gate_result.detail:  # SymPy path: the expression extraction output
            model_solution = gate_result.detail["raw_generation"]
        if model_solution:
            method_result = method_alignment_check(model_solution, worked_solution_text)
            if not method_result.passed:
                effective_gate_result = GateResult(
                    passed=gate_result.passed,
                    tier_allowed=min(gate_result.tier_allowed, method_result.tier_allowed),
                    reason=f"{gate_result.reason} | {method_result.reason}",
                    detail={**gate_result.detail, "method_alignment": method_result.detail},
                )

    hint = generate_hint(
        question_text, effective_gate_result, worked_solution_text, has_diagram=has_diagram,
        requested_tier=requested_tier, question_id=question_id,
    )
    return effective_gate_result, hint, self_consistency_summary


def run_weak_mode_gate_pipeline(question_text: str) -> tuple[GateResult, dict]:
    """Run the gate for a submitted question with no confident corpus match (weak mode).

    With no answer key, SymPy and classification are skipped. Self-consistency runs in
    agreement-only mode, hard-capped at Tier 1, and the standard Tier 1 hint is generated. Method
    alignment never runs, as there is no worked solution.

    Returns:
        (gate_result, hint). The caller stores `gate_result.reason`/`detail` on the
        `student_submissions` row.
    """
    gate_result = self_consistency_check(question_text, real_answer=None)
    hint = generate_hint(question_text, gate_result)
    return gate_result, hint
