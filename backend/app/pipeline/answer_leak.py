"""Deterministic answer-leak check for hints.

Parses every number in the answer and the hint to an exact Fraction (commas, currency, units,
fractions, decimals, mixed numbers and number words 0 to 99 are normalised), then:
  - LEAK: an answer value that is not one of the question's own numbers appears in the hint.
  - AMBIGUOUS: every answer value in the hint also occurs in the question, so it may be an echoed
    given rather than the answer. Not treated as a leak.
  - CLEAN: no answer value appears in the hint.
Answers with no parseable number return "NO_NUMERIC_ANSWER".

First built for the offline hint-quality evaluation (Issues 357 and 358). Since the end-to-end
dry run found a live Tier 3 hint that stated the student's own answer (Issue 387), it also guards
every live hint through `guard_hint()` below.
"""
from __future__ import annotations

import re
from fractions import Fraction

_ONES = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
         "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
         "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}

_WORD_RE = re.compile(r"\b(?:(" + "|".join(_TENS) + r")(?:[\s-]+(" + "|".join(k for k, v in _ONES.items() if 1 <= v <= 9) + r"))?|("
                      + "|".join(_ONES) + r"))\b", re.IGNORECASE)
# A number: mixed number, fraction, or decimal/integer with optional thousands commas.
_NUM_RE = re.compile(r"(?<![\d.])(?:(\d+)\s+(\d+)\s*/\s*(\d+)|(\d+)\s*/\s*(\d+)|(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?))(?![\d])")
# Squared and cubed units such as "2592cm2": the exponent belongs to the unit, so it is removed
# before numbers are extracted. Only matched directly after a digit.
_UNIT_EXPONENT_RE = re.compile(r"(?<=\d)(cm|m)([23])(?!\d)", re.IGNORECASE)


def _words_to_digits(text: str) -> str:
    def sub(m: re.Match) -> str:
        if m.group(1):
            return str(_TENS[m.group(1).lower()] + (_ONES[m.group(2).lower()] if m.group(2) else 0))
        return str(_ONES[m.group(3).lower()])
    return _WORD_RE.sub(sub, text)


def numeric_values(text: str) -> set[Fraction]:
    """Every number in `text` as an exact Fraction, including number words 0 to 99. A mixed number counts as one value."""
    text = _UNIT_EXPONENT_RE.sub(r"\1", text or "")
    text = _words_to_digits(text)
    out: set[Fraction] = set()
    for m in _NUM_RE.finditer(text):
        try:
            if m.group(1) is not None:
                out.add(Fraction(int(m.group(1))) + Fraction(int(m.group(2)), int(m.group(3))))
            elif m.group(4) is not None:
                out.add(Fraction(int(m.group(4)), int(m.group(5))))
            else:
                out.add(Fraction(m.group(6).replace(",", "")))
        except ZeroDivisionError:
            continue
    return out


def check_leak(answer_value: str, question_text: str, hint_text: str) -> dict:
    answer_vals = numeric_values(answer_value)
    if not answer_vals:
        return {"status": "NO_NUMERIC_ANSWER", "leaked": [], "ambiguous": []}
    q_vals = numeric_values(question_text)
    h_vals = numeric_values(hint_text)
    leaked = sorted(v for v in answer_vals if v in h_vals and v not in q_vals)
    ambiguous = sorted(v for v in answer_vals if v in h_vals and v in q_vals)
    status = "LEAK" if leaked else ("AMBIGUOUS" if ambiguous else "CLEAN")
    return {"status": status, "leaked": [str(v) for v in leaked], "ambiguous": [str(v) for v in ambiguous]}


LEAK_RETRY_STEERING = (
    "IMPORTANT: Do not solve the student's own question. Do not state its final answer, or any "
    "number equal to it, in any form. Give only a next step for the student to try."
)

_SAFE_FALLBACK_BY_TIER = {
    1: "Read the question again slowly. What is it asking you to find? Write down the numbers "
       "you are given first.",
    2: "Write down the first step using the numbers in the question. Which calculation links them? "
       "Try it and see what you get.",
    3: "Look back at your last hint and try the next step yourself. When you have an answer, a "
       "parent can check it with the PIN.",
}


def guard_hint(hint: dict, answer_values: list[str], question_text: str, regenerate) -> dict:
    """Return a hint that does not state any of `answer_values`.

    If the hint leaks, `regenerate(steering)` is called once with LEAK_RETRY_STEERING. If that
    still leaks (or fails), the hint text is replaced with a fixed, safe prompt for its tier. The
    returned dict records what happened in `leak_guard` ("clean", "regenerated" or "replaced").
    """
    def leaks(text: str) -> bool:
        return any(check_leak(a, question_text, text)["status"] == "LEAK" for a in answer_values if a)

    if not answer_values or not leaks(hint.get("hint_text", "")):
        return {**hint, "leak_guard": "clean"}
    try:
        retried = regenerate(LEAK_RETRY_STEERING)
    except Exception:
        retried = None
    if retried is not None and not leaks(retried.get("hint_text", "")):
        return {**retried, "leak_guard": "regenerated"}
    tier = hint.get("tier", 1)
    return {**hint, "hint_text": _SAFE_FALLBACK_BY_TIER.get(tier, _SAFE_FALLBACK_BY_TIER[1]),
            "leak_guard": "replaced"}


_STEP_RE = re.compile(r"\s*(Step \d+\s*:)")
_SENTENCE_END_RE = re.compile(r"[.!?](?=\s|$)")


def tidy_hint_text(text: str) -> str:
    """Put each "Step N:" on its own line and drop an unfinished last sentence.

    Hints are generated with a token limit, so a long one can stop mid-sentence, for example
    "Step 3: Multiply the number of" (Issue 388). A child should never see a broken sentence.
    If the text contains no complete sentence at all it is returned unchanged.
    """
    text = (text or "").strip()
    ends = [m.end() for m in _SENTENCE_END_RE.finditer(text)]
    if ends and ends[-1] < len(text):
        text = text[: ends[-1]]
    text = _STEP_RE.sub(lambda m: "\n" + m.group(1), text).strip()
    return text
