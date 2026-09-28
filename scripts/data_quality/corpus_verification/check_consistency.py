r"""
Check each question's `worked_solution_text` against its stored `answer_value` (Issue 361).

Runs over the live corpus database, `psle.db`, the one `backend/app/db/session.py` connects to by
default. The other databases on disk, `corpus_run_baseline_prefix_20260830.db` and `held_out.db`,
are evaluation snapshots, not the serving corpus. CPU only: no model is loaded, and only the
pure-Python comparison helpers from `gate.py` are imported, unchanged. Run before
verify_and_prioritize.py.

Row count: 1,262 rows in `psle.db` have both `answer_value` and `worked_solution_text` non-empty
and `superseded_by` null. A figure of 1,001 had been quoted, but it could not be reproduced from
this database with any combination of `extraction_flag` exclusions (1,209 without boilerplate
rows, 1,255 without "no matching answer key" rows, 1,202 without both, 1,143 also without
"verification pending" rows), so it is treated as inaccurate.

Method:
  - Multi-part answers are detected with the same "compound_ab" convention as
    `classify_answer_value()` in `scripts/training/build_hint_review_queue.py`. Both the answer
    and the solution are split on (a)/(b)/(c)/(d) markers, and each part is checked separately.
  - For a single value, or each part of a compound one, the last "= <value>" in the solution
    segment is taken as its final computed value. A manual spot check of 12 rows found every
    stepped solution ends its computation at the rightmost "=". A value after an "ANS:" marker is
    also used when present. Candidates are compared with the stored value using
    `gate._values_equal()`, which is SymPy-aware ('48' == '48.0', '1/2' == '0.5').
  - Outcome per row: MATCH; MISMATCH (a candidate corpus error); or UNPARSEABLE (no confident
    value found, or the (a)/(b) labels could not be lined up). UNPARSEABLE is not a mismatch or
    a corpus error. The automated check cannot judge these rows, so verify_and_prioritize.py
    routes them to human review.

Output: data/extracted/corpus_verification/consistency_results.json (per-row detail) and a printed
summary.

Usage: python scripts/data_quality/corpus_verification/check_consistency.py
"""
import json
import re
import sqlite3
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "backend"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from app.pipeline.gate import _values_equal  # noqa: E402

_OUT_DIR = _REPO_ROOT / "data" / "extracted" / "corpus_verification"
_DB = _REPO_ROOT / "psle.db"

# Same boundary convention as the compound_ab detection in classify_answer_value()
# (build_hint_review_queue.py).
_PART_LABEL_RE = re.compile(r"\(([a-d])\)", re.IGNORECASE)
_LAST_EQUALS_RE = re.compile(r"=\s*([^\n=]+?)\s*(?=\n|$)")
_ANS_MARKER_RE = re.compile(r"\bans\.?\s*:\s*", re.IGNORECASE)
_TRAILING_PAREN_RE = re.compile(r"\s*\([^()]*\)\s*$")  # e.g. "245°. (∠sum at a point)"


def _clean(v: str | None) -> str | None:
    """Remove trailing annotations that would cause false mismatches.

    A spot check of 25 MISMATCH rows over two passes found most were caused by the cases handled
    here and in `_values_match()`:
    1. Both the captured value and the stored `answer_value` often carry a trailing annotation
       with no newline before it ("245°. (∠sum at a point)", "$149. ANS : $149", "59km/h (Ans)").
       This breaks strip_units() and parse_expr(). One trailing parenthetical remark and a
       trailing full stop are removed.
    2. A parenthetical that carries the value must be kept, e.g. the algebraic answer
       "$(4d - 60)". The remark is only removed if a digit remains afterwards; "$(4d - 60)" would
       become a bare "$", so it is left alone.
    """
    if v is None:
        return None
    stripped = _TRAILING_PAREN_RE.sub("", v).strip()
    if stripped and re.search(r"\d", stripped):
        v = stripped
    v = v.strip()
    # A final "." with nothing after it is a full stop, not a decimal point: a decimal always has
    # digits after the dot ("3.14"). It is stripped even after a digit, to handle "$52." and
    # "(5f-24)/3.".
    if v.endswith("."):
        v = v[:-1].strip()
    return v or None


def _values_match(candidate: str, answer: str) -> bool:
    """Compare a candidate with the answer, trying a literal match before `gate._values_equal()`.

    `gate._values_equal()` is shared with the live app, so it is not patched here. It can report
    identical strings as unequal when `parse_expr()` succeeds but `nsimplify()` in
    `_sympy_equal()` then fails. Two cases were confirmed: a boolean literal ("True" parses to
    `sympy.true`, which has no `.evalf()`), and a comma-separated list ("24, 48" parses to a SymPy
    `Tuple`, the interaction described for Issue 172 in gate.py). Checking a case-insensitive
    literal match first stops an exact restatement being counted as a mismatch.
    """
    if candidate.strip().lower() == answer.strip().lower():
        return True
    return _values_equal(candidate, answer)


def _looks_compound(answer_value: str) -> bool:
    low = answer_value.lower()
    return ("(a)" in low and "(b)" in low) or bool(re.match(r"^\(?[ab][).]", low))


def _split_parts(text: str) -> dict[str, str]:
    """Split text on (a)-(d) labels into {'a': '<text up to the next label>', ...}.

    Returns an empty dict if no labels are found.
    """
    marks = list(_PART_LABEL_RE.finditer(text))
    if not marks:
        return {}
    out = {}
    for i, m in enumerate(marks):
        letter = m.group(1).lower()
        start = m.end()
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        out.setdefault(letter, text[start:end])  # first wins: a repeated label is not a new part
    return out


def _final_equals_value(segment: str) -> str | None:
    matches = _LAST_EQUALS_RE.findall(segment)
    return _clean(matches[-1]) if matches else None


def _ans_block(text: str) -> str | None:
    """Return the text after the last 'ANS:' marker, or None if there is none.

    This common corpus convention is often a more reliable final answer than the last '='. For
    example, in a remainder question the last '=' can be an intermediate result, with the answer
    given after 'ANS:' (id=294: '1962/30=65R12  ANS: 12').
    """
    marks = list(_ANS_MARKER_RE.finditer(text))
    return text[marks[-1].end():].strip() if marks else None


def check_row(answer_value: str, worked_solution_text: str) -> dict:
    ans_block = _ans_block(worked_solution_text)
    if _looks_compound(answer_value):
        a_parts = _split_parts(answer_value)
        body_parts = _split_parts(worked_solution_text[: -len(ans_block)] if ans_block else worked_solution_text)
        ans_parts = _split_parts(ans_block) if ans_block else {}
        if not a_parts or (not body_parts and not ans_parts):
            return {"status": "UNPARSEABLE", "reason": "compound answer but could not split into labelled (a)/(b) parts on both sides"}
        results, candidates_used = {}, {}
        for letter, a_val in a_parts.items():
            a_val = _clean(a_val)
            candidates = [c for c in (
                _clean(ans_parts[letter]) if letter in ans_parts else None,
                _final_equals_value(body_parts[letter]) if letter in body_parts else None,
            ) if c is not None]
            candidates_used[letter] = candidates
            if not candidates:
                results[letter] = "UNPARSEABLE"
            elif any(_values_match(c, a_val) for c in candidates):
                results[letter] = "MATCH"
            else:
                results[letter] = "MISMATCH"
        statuses = set(results.values())
        status = "MISMATCH" if "MISMATCH" in statuses else ("MATCH" if statuses <= {"MATCH"} else "UNPARSEABLE")
        return {"status": status, "per_part": results, "candidates": candidates_used}

    candidates = [c for c in (_clean(ans_block), _final_equals_value(worked_solution_text)) if c is not None]
    if not candidates:
        return {"status": "UNPARSEABLE", "reason": "no '= value' or 'ANS:' marker found in worked_solution_text"}
    answer_clean = _clean(answer_value) or answer_value
    matched = any(_values_match(c, answer_clean) for c in candidates)
    return {"status": "MATCH" if matched else "MISMATCH", "candidates": candidates}


def main() -> None:
    con = sqlite3.connect(f"file:{_DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "select id, answer_value, worked_solution_text, has_diagram, question_section, ocr_confidence "
        "from questions where answer_value is not null and trim(answer_value)<>'' "
        "and worked_solution_text is not null and trim(worked_solution_text)<>'' and superseded_by is null").fetchall()
    print(f"eligible rows (both populated, not superseded): {len(rows)}")

    results = {}
    for r in rows:
        d = dict(r)
        res = check_row(d["answer_value"], d["worked_solution_text"])
        results[str(d["id"])] = {**res, "answer_value": d["answer_value"], "worked_solution_text": d["worked_solution_text"],
                                 "has_diagram": bool(d["has_diagram"]), "question_section": d["question_section"],
                                 "ocr_confidence": d["ocr_confidence"]}

    from collections import Counter
    counts = Counter(v["status"] for v in results.values())
    print("status counts:", dict(counts))
    mismatches = [k for k, v in results.items() if v["status"] == "MISMATCH"]
    print(f"MISMATCH rows ({len(mismatches)}): {mismatches[:40]}{'...' if len(mismatches) > 40 else ''}")

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    (_OUT_DIR / "consistency_results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print("Saved ->", _OUT_DIR / "consistency_results.json")


if __name__ == "__main__":
    main()
