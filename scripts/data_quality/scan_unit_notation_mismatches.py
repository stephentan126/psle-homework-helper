r"""
Scan a hint review-queue JSONL for unit-notation mismatches between a hint and its question.

This is the defect Issue 295 fixed by hand for one row (id=37): `bootstrapped_hint` spells out a
unit word attached to a number ("1.2 kilograms") while the row writes the same number with the
symbol ("1.2 kg"). A hint that restates a question in notation the question does not use weakens
the training signal, since Tier 1 hints are meant to restate the question in the student's terms
(Issue 291). Report only: it writes a JSON file of suggested edits and does not change the queue.
There is no `--apply` flag.

Why "same number" rather than "unit symbol anywhere in the row": an earlier version checked
whether any symbol of the unit type appeared anywhere in the row, and gave a false positive on
id=771. The hint says "3 liters per minute", and the question also spells it out ("3 litres per
minute"), so there is nothing to fix; the check had been triggered by unrelated numbers (54L,
13.5L) in the worked solution. This scan requires the exact number from the hint's spelled-out
unit.

Fields checked: `question_text`, `answer_value` and `worked_solution_text`, in that order. The
same number-and-symbol pairing in any of them counts as a match. The search was widened from
`question_text` alone after the first pass missed id=37: the hint says "32 degrees" and
"84 degrees", `question_text` has them only as LaTeX (`\angle BAF = 32^\circ`), but
`answer_value` ("32°") and `worked_solution_text` ("...FCD=180°-84°=96°...") use the plain
symbol. A quantity written only in LaTeX in all three fields (for example `3\text{m}`, never a
plain "3m" or "3 m") is a known limitation of this plain-text check and is not covered.

Units covered (spelled word -> symbol searched for):
  degree(s)                                  -> °
  metre(s)/meter(s)                          -> m       (not "square metre(s)/meter(s)")
  centimetre(s)/centimeter(s)                -> cm      (not "square centimetre(s)/centimeter(s)")
  kilometre(s)/kilometer(s)                  -> km
  kilogram(s)                                -> kg
  gram(s)                                    -> g       (the pattern must follow the number
                                                          directly, so "kilograms" never matches)
  litre(s)/liter(s)                          -> l or L
  square centimetre(s)/centimeter(s)         -> cm2 / cm² / cm^2
  square metre(s)/meter(s)                   -> m2 / m² / m^2

The "metre" and "centimetre" patterns use a negative lookbehind for "square ", so "square metres"
is only handled by the square-unit rule and never counted twice. "kilometre" needs no such guard,
because the metre pattern must follow the number directly.

Symbol patterns search for the same number (`NUMBER(?!\d)`, so not the start of a longer number),
then `\s*`, then the symbol. One- and two-letter symbols (m, g, l, cm, km, kg) have a
lookahead, so "5 min" does not count as metres, and "5cm2" or "5cm²" does not count as plain
centimetres.

Output: a JSON array with one entry per (question_id, tier) that has at least one match:
    {"question_id": int, "tier": int, "matched_number": str, "spelled_form": str,
     "symbol_form": str, "suggested_edit": str}
A row with several matches (rare) still gives one entry: `suggested_edit` replaces every match,
and the three descriptive fields become "; "-joined lists in order of appearance.

Usage:
    backend/.venv/Scripts/python.exe scripts/data_quality/scan_unit_notation_mismatches.py \
        --queue data/extracted/slice6_step0c/hint_review_queue_full_pool.jsonl \
        --out data/extracted/slice6_step0c/unit_notation_mismatch_suggestions.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_NUM = r"(?<!\d)(\d+(?:\.\d+)?)(?!\d)"

# (unit_key, spelled_regex, symbol_alt_pattern). symbol_alt_pattern is placed after the captured
# number and \s* when searching the row fields. It must not contain capturing groups, so the
# symbol is always group 1 of the built pattern.
_UNIT_SPECS: list[tuple[str, str, str]] = [
    # Square units come first, before their plain counterparts, so "square metres" is only
    # classified once, under the square rule.
    ("square_centimetre", r"square\s+centimet(?:re|er)s?", r"cm(?:2|²|\^2)"),
    ("square_metre", r"square\s+met(?:re|er)s?", r"m(?:2|²|\^2)"),
    ("kilometre", r"kilomet(?:re|er)s?", r"km(?![A-Za-z])"),
    ("centimetre", r"(?<!square\s)centimet(?:re|er)s?", r"cm(?![A-Za-z0-9²])"),
    ("kilogram", r"kilograms?", r"kg(?![A-Za-z])"),
    ("metre", r"(?<!square\s)met(?:re|er)s?", r"m(?![A-Za-z0-9²])"),
    ("gram", r"grams?", r"g(?![A-Za-z])"),
    ("litre", r"lit(?:re|er)s?", r"[lL](?![A-Za-z])"),
    ("degree", r"degrees?", r"°"),
]
_SPELLED_RES = [
    (key, re.compile(_NUM + r"\s*" + word_pat, re.IGNORECASE), symbol_pat)
    for key, word_pat, symbol_pat in _UNIT_SPECS
]


def _find_matches(hint: str, search_fields: list[str]) -> list[dict]:
    """Return every confirmed (number, spelled_form, symbol_form, span) in `hint`.

    `search_fields` is checked in order (question_text, answer_value, worked_solution_text).
    The same number-and-symbol pairing in any of them counts as a match, and the first field
    that contains it is used.
    """
    found = []
    for key, spelled_re, symbol_pat in _SPELLED_RES:
        for m in spelled_re.finditer(hint):
            number, spelled_form = m.group(1), m.group(0)[len(m.group(1)):].strip()
            spelled_span = m.span()
            sym_re = re.compile(re.escape(number) + r"(?!\d)\s*(" + symbol_pat + r")", re.IGNORECASE)
            sym_m = None
            field_text = None
            for text in search_fields:
                sym_m = sym_re.search(text)
                if sym_m:
                    field_text = text
                    break
            if not sym_m:
                continue
            symbol_form = sym_m.group(1)
            had_space = field_text[sym_m.start(1) - 1: sym_m.start(1)] in (" ", "\t") \
                if sym_m.start(1) > 0 else False
            found.append({
                "unit_key": key, "number": number, "spelled_form": spelled_form,
                "symbol_form": symbol_form, "had_space": had_space,
                "span": spelled_span,
            })
    # If two unit rules matched the same span, keep only the first. The square-first order and
    # lookbehind guards should prevent this, but it guards against double counting.
    seen_spans = set()
    deduped = []
    for f in found:
        if f["span"] in seen_spans:
            continue
        seen_spans.add(f["span"])
        deduped.append(f)
    return deduped


def _apply_edits(hint: str, matches: list[dict]) -> str:
    # Apply right-to-left by span start so earlier replacements don't shift later spans.
    out = hint
    for f in sorted(matches, key=lambda x: x["span"][0], reverse=True):
        start, end = f["span"]
        number = f["number"]
        sep = " " if f["had_space"] else ""
        replacement = f"{number}{sep}{f['symbol_form']}"
        out = out[:start] + replacement + out[end:]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    queue_path = Path(args.queue)
    rows = []
    with queue_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    results = []
    unit_breakdown: dict[str, int] = {}
    for r in rows:
        hint = r.get("bootstrapped_hint") or ""
        qtext = r.get("question_text") or ""
        av = r.get("answer_value")
        av = av if isinstance(av, str) else ("" if av is None else str(av))
        wst = r.get("worked_solution_text") or ""
        if not hint or not (qtext or av or wst):
            continue
        matches = _find_matches(hint, [qtext, av, wst])
        if not matches:
            continue
        for f in matches:
            unit_breakdown[f["unit_key"]] = unit_breakdown.get(f["unit_key"], 0) + 1
        suggested_edit = _apply_edits(hint, matches)
        matches_sorted = sorted(matches, key=lambda x: x["span"][0])
        results.append({
            "question_id": r["question_id"],
            "tier": r["tier"],
            "matched_number": "; ".join(f["number"] for f in matches_sorted),
            "spelled_form": "; ".join(f["spelled_form"] for f in matches_sorted),
            "symbol_form": "; ".join(f["symbol_form"] for f in matches_sorted),
            "suggested_edit": suggested_edit,
        })

    out_path = Path(args.out)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    multi = sum(1 for r in results if ";" in r["matched_number"])
    print(f"Rows scanned: {len(rows)}")
    print(f"Rows with >=1 confirmed unit-notation mismatch: {len(results)} "
          f"({multi} of those have more than one match in the same hint)")
    print(f"Breakdown by unit type (match count, not row count): {unit_breakdown}")
    print(f"Total confirmed matches: {sum(unit_breakdown.values())}")
    print(f"Written -> {out_path}")


if __name__ == "__main__":
    main()
