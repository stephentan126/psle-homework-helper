r"""
Find questions whose stored arithmetic expression may have a corrupted operator (Issue 220).

The problem: in 4 confirmed cases (question ids 551, 1352, 1924, 4782), `question_text` holds a
different arithmetic problem from the one printed on the source page, because a printed `÷` was
stored as `+`. They were found by chance in a 6-question sample, so the corpus-wide extent is
unknown and should not be assumed small.

The signal: for a question whose `question_text` is a bare "Find the value of \\(EXPR\\)"
expression (the shape of all 4 known cases), compare the numbers and operators in EXPR with those
in the leading segment of `worked_solution_text`. In the two known cases that have a worked
solution (1924, 4782), that segment (up to the first `=` or newline, whichever comes first; see
`_extract_worked_solution_lead_segment()`) restates the original expression with the same numbers
in the same order, except for the one corrupted operator. When the number sequences match exactly
but an operator at the same position differs, the row is a candidate.

Limitation 1: 2 of the 4 known cases (551, 1352) have no `worked_solution_text`, so this signal
cannot see them. Catching a `÷` to `+` corruption on a question with no worked solution would need
a separate signal, which is not designed here. No hits from this scan does not mean there are no
such rows.

Limitation 2: this is a text-pattern signal, not a check against the source. Each candidate is a
hypothesis to check against the source PDF page, as was done for the original 4, and is never
grounds for rewriting `question_text` automatically. The gate currently fails closed on a corrupted
`÷` to `+` row only as a side effect of comparing against a correct stored answer key (see
Issue 220), so this scan's output deserves the same caution.

Expected false-positive shapes, each covered by the dry-run harness below:
  1. A worked solution that starts with an intermediate sub-step rather than restating the
     expression (question `100 - 25 × 2`, working starts `25×2=50`). The number sequences do not
     match, so no candidate is produced: a false negative (lower coverage), never a false
     positive.
  2. A worked solution that uses a different method with different numbers (common in PSLE
     solutions, e.g. bar-model "5u - 2u = 3u" working). Skipped for the same reason.
  3. A word problem with no bare `\(...\)` expression. The expression extractor finds nothing,
     so the row is skipped before any comparison.
  4. An expression containing a fraction (`\frac{a}{b}`). Skipped entirely; this version does not
     compare fractions.
  5. Residual risk, not fully mitigated: a worked solution whose first line happens to use the
     same numbers in the same order for an unrelated reason (for example part (a)'s numbers
     recurring in part (b)'s working). This could give a spurious candidate, which is why every
     candidate goes to a review queue and is never corrected automatically.

Validation before trusting the scan on the full corpus:
  1. Dry run, with no database access: `find_operator_mismatch()` is run on the 4 known cases
     (copied verbatim from the Issue 220 investigation) and on synthetic clean cases that target
     each false-positive shape above. This is what running the script with no arguments does
     (`_run_dry_run_harness()`).
  2. Check the false-positive rate on a larger hand-picked sample of known-clean "Find the value
     of" questions from the live corpus, confirmed against their source pages.
  3. The corpus-wide run (`--run-live-scan`), followed by human verification of the resulting
     review queue. The live scan was deferred until the Issue 214 DeepSeek-OCR pass had
     finished, only to avoid adding load; the scan itself is pure CPU and SQLite, with no model
     or GPU.

Usage:
    # Fast dry run with no database access (the default); asserts the expected outcomes for the
    # 4 known cases and the synthetic clean cases:
    backend/.venv/Scripts/python.exe scripts/data_quality/scan_question_text_operator_mismatch.py

    # Corpus-wide scan. Requires this explicit flag; read-only, never writes question_text or any
    # other field:
    backend/.venv/Scripts/python.exe scripts/data_quality/scan_question_text_operator_mismatch.py \
        --run-live-scan
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "backend"))

try:
    # The Windows console code page does not reliably render ÷ and × otherwise (the first run
    # printed mojibake). Best effort only; never fatal if unsupported.
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# =================================================================================================
# Tokeniser: a left-to-right scan of an expression's printed characters, not a parser. Precedence
# and bracket nesting are not needed, because only the printed operator sequences of two texts
# are compared; neither is evaluated.
# =================================================================================================

# Longest tokens first, so multi-character LaTeX operators (\times, \div) are matched whole rather
# than split into single characters.
_OPERATOR_PATTERN = re.compile(r"\\times|\\div|[+\-−*xX/×÷]")
_NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")

_OPERATOR_NORMALIZE = {
    "+": "+",
    "-": "-",
    "−": "-",  # Unicode minus sign, seen in this corpus's OCR output
    "*": "×", "x": "×", "X": "×", "\\times": "×", "×": "×",
    "/": "÷", "\\div": "÷", "÷": "÷",
}


@dataclass
class _Tokens:
    numbers: list[str]
    operators: list[str]


def _tokenize_expression(text: str) -> Optional[_Tokens]:
    """Split an expression into its numbers and normalised operators, left to right.

    Returns None if the text contains `\\frac`, since fractions are out of scope (false-positive
    shape 4 in the module docstring), or if it contains no numbers.
    """
    if "\\frac" in text:
        return None
    numbers: list[str] = []
    operators: list[str] = []
    pos = 0
    while pos < len(text):
        num_match = _NUMBER_PATTERN.match(text, pos)
        if num_match:
            numbers.append(num_match.group(0))
            pos = num_match.end()
            continue
        op_match = _OPERATOR_PATTERN.match(text, pos)
        if op_match:
            raw = op_match.group(0)
            operators.append(_OPERATOR_NORMALIZE.get(raw, raw))
            pos = op_match.end()
            continue
        pos += 1  # skip anything else (brackets, spaces, "=", LaTeX braces)
    if not numbers:
        return None
    # Defensive trim: an expression whose numbers are all joined by binary operators has
    # len(numbers) - 1 operators. Any extra trailing operators are trimmed so they cannot
    # misalign the position-by-position comparison (a guard against future pattern changes).
    if len(operators) >= len(numbers):
        operators = operators[: len(numbers) - 1]
    return _Tokens(numbers=numbers, operators=operators)


def _extract_question_core_expression(question_text: str) -> Optional[str]:
    """Return the expression inside the first LaTeX `\\( ... \\)` delimiters, or None.

    All 4 known cases use this shape (`"Find the value of \\(EXPR\\)..."`), and the signal applies
    only to it. Word problems with no bare expression return None (false-positive shape 3).
    """
    match = re.search(r"\\\((.*?)\\\)", question_text, re.DOTALL)
    return match.group(1) if match else None


def _extract_worked_solution_lead_segment(worked_solution_text: str) -> Optional[str]:
    """Return the part of the working that restates the original expression, before any step.

    This is not simply the first line. The two known cases with a worked solution use different
    layouts: id=1924 puts each step on its own line (`"30-8+16÷4+2\\n=30-8+4+2\\n..."`), while
    id=4782 puts the whole working on one line separated by `=` (`"144÷6000=144÷6÷1000=..."`).
    Taking the first line would treat id=4782's whole working as the restatement, which then
    fails the number match for the wrong reason (the dry-run harness caught this). The segment
    therefore ends at the first `=` or newline, whichever comes first, or at the end of the text
    if there is no reduction step.
    """
    eq_pos = worked_solution_text.find("=")
    nl_pos = worked_solution_text.find("\n")
    candidates = [p for p in (eq_pos, nl_pos) if p != -1]
    cut = min(candidates) if candidates else len(worked_solution_text)
    segment = worked_solution_text[:cut].strip()
    return segment or None


@dataclass
class OperatorMismatchCandidate:
    """A review-queue entry: a hypothesis to check against the source PDF page.

    Never a confirmed finding, and never grounds for correcting `question_text` automatically.
    """
    question_id: object
    source_paper: str
    source_page_index: int
    question_text: str
    worked_solution_text: str
    question_numbers: list[str]
    question_operators: list[str]
    worked_operators: list[str]
    mismatch_positions: list[tuple[int, str, str]]  # (index, question_op, worked_op)
    confidence: str  # "high" / "medium" / "low"; see find_operator_mismatch()


def find_operator_mismatch(
    question_id: object, source_paper: str, source_page_index: int,
    question_text: str, worked_solution_text: Optional[str],
) -> Optional[OperatorMismatchCandidate]:
    """Return a review-queue candidate if the question and working disagree on an operator.

    Returns None for every case the signal cannot or should not flag; the module docstring gives
    the reason for each bail-out. A candidate is not a confirmed finding.

    Confidence:
      "high"    exactly one operator position differs, and it is the `÷`/`+` pair seen in all 4
                known cases.
      "medium"  exactly one position differs, with a different operator pair: a possible single
                slip, but not a confirmed pattern.
      "low"     more than one position differs, which more likely reflects a different method or
                rearrangement than one transcription slip. Worth a look, at lower priority.
    """
    if not worked_solution_text:
        return None  # known blind spot (limitation 1 in the module docstring)

    core_expr = _extract_question_core_expression(question_text)
    if core_expr is None:
        return None

    lead_segment = _extract_worked_solution_lead_segment(worked_solution_text)
    if lead_segment is None:
        return None

    q_tokens = _tokenize_expression(core_expr)
    w_tokens = _tokenize_expression(lead_segment)
    if q_tokens is None or w_tokens is None:
        return None

    if q_tokens.numbers != w_tokens.numbers:
        return None  # the lead segment does not restate the same expression,
                      # so there is nothing to compare (false-positive shapes 1 and 2).

    if q_tokens.operators == w_tokens.operators:
        return None  # consistent: nothing to flag

    mismatch_positions = [
        (i, qo, wo) for i, (qo, wo) in enumerate(zip(q_tokens.operators, w_tokens.operators))
        if qo != wo
    ]
    if not mismatch_positions:
        return None  # defensive: the equality check above already covers this,
                      # since matching numbers imply equal operator counts.

    if len(mismatch_positions) == 1 and {mismatch_positions[0][1], mismatch_positions[0][2]} == {
        "+", "÷",
    }:
        confidence = "high"
    elif len(mismatch_positions) == 1:
        confidence = "medium"
    else:
        confidence = "low"

    return OperatorMismatchCandidate(
        question_id=question_id, source_paper=source_paper, source_page_index=source_page_index,
        question_text=question_text, worked_solution_text=worked_solution_text,
        question_numbers=q_tokens.numbers, question_operators=q_tokens.operators,
        worked_operators=w_tokens.operators, mismatch_positions=mismatch_positions,
        confidence=confidence,
    )


# =================================================================================================
# Corpus-wide scan. Read-only: it never changes any field. Runs only with the explicit
# --run-live-scan flag (see main()).
# =================================================================================================

def scan_corpus(session) -> list[OperatorMismatchCandidate]:
    """Apply `find_operator_mismatch()` to every question with a `worked_solution_text`.

    Returns the candidates. The function has no write path, in line with the Issue 220 rule:
    flag for human review, never correct automatically.
    """
    from sqlalchemy import select

    from app.models.question import Question

    candidates: list[OperatorMismatchCandidate] = []
    questions = session.scalars(
        select(Question).where(Question.worked_solution_text.is_not(None)),
    ).all()
    for q in questions:
        candidate = find_operator_mismatch(
            q.id, q.source_paper, q.source_page_index, q.question_text, q.worked_solution_text,
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def print_review_queue(candidates: list[OperatorMismatchCandidate]) -> None:
    """Print the review queue grouped by confidence.

    Each entry shows the question_id, the suspected mismatch and a pointer to the source page
    (`source_paper` and `source_page_index`, as in the project's other review outputs), so a
    person can check the page image rather than trust the OCR text again (Issue 220).
    """
    by_confidence: dict[str, list[OperatorMismatchCandidate]] = {"high": [], "medium": [], "low": []}
    for c in candidates:
        by_confidence[c.confidence].append(c)
    print(
        f"Real operator-mismatch review queue: {len(candidates)} candidate(s) "
        f"({len(by_confidence['high'])} high, {len(by_confidence['medium'])} medium, "
        f"{len(by_confidence['low'])} low confidence).",
    )
    print(
        "NOTHING HERE HAS BEEN VISUALLY CONFIRMED. Every candidate below is a hypothesis to check "
        "against the real source PDF page image, not a confirmed finding -- never trust this "
        "signature more than a coincidental, unvalidated text pattern warrants.",
    )
    for level in ("high", "medium", "low"):
        for c in by_confidence[level]:
            print(f"\n[{c.confidence.upper()}] question_id={c.question_id}")
            print(
                f"  source: {c.source_paper} (rendered page index {c.source_page_index}) -- "
                f"view: data/extracted/pages/<stem>/{c.source_page_index}.png",
            )
            print(f"  question_text expression tokens:        {c.question_numbers} / {c.question_operators}")
            print(f"  worked_solution_text (lead segment) tokens: {c.question_numbers} / {c.worked_operators}")
            for idx, qo, wo in c.mismatch_positions:
                print(
                    f"    mismatch at operator position {idx}: question_text says {qo!r}, "
                    f"worked_solution_text says {wo!r}",
                )


# =================================================================================================
# Dry-run harness with hard-coded data and no database access. This is what running the script
# with no arguments does. It checks the signal's logic with pass/fail checks, not just printed
# output, against:
#   (a) the 4 known cases, copied verbatim from the Issue 220 investigation. Two (1924, 4782)
#       must be caught, and the other two (551, 1352) must be reported as "not applicable",
#       since they have no worked_solution_text (the known blind spot).
#   (b) a small set of synthetic clean cases, each targeting one of the false-positive shapes in
#       the module docstring.
# =================================================================================================

@dataclass
class _DryRunCase:
    label: str
    question_id: object
    source_paper: str
    source_page_index: int
    question_text: str
    worked_solution_text: Optional[str]
    expected: str  # "high" / "medium" / "low" / "none" / "not_applicable"


_KNOWN_REAL_CASES = [
    _DryRunCase(
        label="id=551 Raffles-Girls (real, confirmed corruption, but NO worked_solution_text)",
        question_id=551,
        source_paper="data/School papers/2021/P6-Maths-2021-SA1-Raffles-Girls.pdf",
        source_page_index=8,
        question_text=r"Find the value of \(60 + (15 - 8 + 3)\times 9\)\nAns:",
        worked_solution_text=None,
        expected="not_applicable",
    ),
    _DryRunCase(
        label="id=1352 Rosyth SA2 (real, confirmed corruption, but NO worked_solution_text)",
        question_id=1352,
        source_paper="data/School papers/2021/P6-Maths-2021-SA2-Rosyth.pdf",
        source_page_index=11,
        question_text=r"Find the value of \(20 - 8 + 4 \times (2 + 6) + 1\).\nAns:",
        worked_solution_text=None,
        expected="not_applicable",
    ),
    _DryRunCase(
        label="id=1924 Red Swastika SA2 (real, confirmed corruption, HAS worked_solution_text)",
        question_id=1924,
        source_paper="data/School papers/2022/P6_Maths_2022_SA2_redswastika.pdf",
        source_page_index=9,
        question_text=r"Find the value of \(30 - 8 + 16 + 4 + 2\)\nAns:",
        worked_solution_text="30-8+16÷4+2\n=30-8+4+2\n=22+4+2\n=28",
        expected="high",
    ),
    _DryRunCase(
        label="id=4782 Rosyth WA1 (real, confirmed corruption, HAS worked_solution_text)",
        question_id=4782,
        source_paper="data/School papers/2025/P6_Maths_2025_WA1_rosyth.pdf",
        source_page_index=8,
        question_text=r"Find the value of \(144 + 6000\).\nAns:",
        worked_solution_text="144÷6000=144÷6÷1000=24÷1000=0.024\nANS:0.024",
        expected="high",
    ),
]

_SYNTHETIC_CLEAN_CASES = [
    _DryRunCase(
        label="synthetic clean: operators genuinely match, no real signal",
        question_id="synthetic-clean-1",
        source_paper="(synthetic, not a real corpus row)",
        source_page_index=0,
        question_text=r"Find the value of \(45 + 12 - 7\).\nAns:",
        worked_solution_text="45+12-7\n=57-7\n=50",
        expected="none",
    ),
    _DryRunCase(
        label="synthetic clean: real word problem, no bare \\(...\\) expression at all",
        question_id="synthetic-clean-2",
        source_paper="(synthetic, not a real corpus row)",
        source_page_index=0,
        question_text="A shop sold 45 apples and 12 oranges. How many fruits in total?\nAns:",
        worked_solution_text="45+12=57",
        expected="none",
    ),
    _DryRunCase(
        label="synthetic clean: worked solution starts with an intermediate sub-step, not a "
              "full restatement (false-positive shape #1)",
        question_id="synthetic-clean-3",
        source_paper="(synthetic, not a real corpus row)",
        source_page_index=0,
        question_text=r"Find the value of \(100 - 25 \times 2\).\nAns:",
        worked_solution_text="25×2=50\n100-50=50",
        expected="none",
    ),
    _DryRunCase(
        label="synthetic clean: fraction-containing expression, deliberately bailed out of "
              "(false-positive shape #4)",
        question_id="synthetic-clean-4",
        source_paper="(synthetic, not a real corpus row)",
        source_page_index=0,
        question_text=r"Find the value of \(\frac{3}{4} + \frac{1}{2}\).\nAns:",
        worked_solution_text="3/4+1/2=5/4",
        expected="none",
    ),
    _DryRunCase(
        label="synthetic: single operator mismatch that is NOT the confirmed ÷/+ pair "
              "(medium confidence)",
        question_id="synthetic-medium-1",
        source_paper="(synthetic, not a real corpus row)",
        source_page_index=0,
        question_text=r"Find the value of \(50 + 10 \times 2\).\nAns:",
        worked_solution_text="50-10×2\n=50-20\n=30",
        expected="medium",
    ),
    _DryRunCase(
        label="synthetic: two operator positions differ (low confidence)",
        question_id="synthetic-low-1",
        source_paper="(synthetic, not a real corpus row)",
        source_page_index=0,
        question_text=r"Find the value of \(10 + 5 - 3\).\nAns:",
        worked_solution_text="10-5+3\n=5+3\n=8",
        expected="low",
    ),
]


def _run_dry_run_harness() -> None:
    print("=" * 90)
    print("DRY RUN ONLY -- no database access, no live corpus scan. Hard-coded, verbatim/synthetic")
    print("data only. Run with --run-live-scan for the real, corpus-wide scan (deliberately NOT")
    print("done as part of this task).")
    print("=" * 90)

    all_cases = _KNOWN_REAL_CASES + _SYNTHETIC_CLEAN_CASES
    failures = []

    for case in all_cases:
        result = find_operator_mismatch(
            case.question_id, case.source_paper, case.source_page_index,
            case.question_text, case.worked_solution_text,
        )
        if case.expected == "not_applicable":
            actual = "not_applicable" if result is None and case.worked_solution_text is None else (
                "UNEXPECTED_RESULT"
            )
        elif case.expected == "none":
            actual = "none" if result is None else "UNEXPECTED_CANDIDATE"
        else:
            actual = result.confidence if result is not None else "UNEXPECTED_NONE"

        ok = actual == case.expected
        status = "OK" if ok else "MISMATCH"
        print(f"\n[{status}] {case.label}")
        print(f"  question_id={case.question_id!r} expected={case.expected!r} actual={actual!r}")
        if result is not None:
            print(f"  question tokens: {result.question_numbers} / {result.question_operators}")
            print(f"  worked tokens:   {result.question_numbers} / {result.worked_operators}")
            for idx, qo, wo in result.mismatch_positions:
                print(f"    mismatch at position {idx}: question={qo!r} worked={wo!r}")
        if not ok:
            failures.append(case.label)

    print("\n" + "=" * 90)
    if failures:
        print(f"DRY RUN FAILED: {len(failures)} case(s) did not match the expected real outcome:")
        for label in failures:
            print(f"  - {label}")
        sys.exit(1)
    print(f"DRY RUN PASSED: all {len(all_cases)} case(s) (4 known real + {len(_SYNTHETIC_CLEAN_CASES)} "
          f"synthetic) matched their expected real outcome. Signature logic is internally "
          f"consistent -- this is NOT the same as a validated real-corpus false-positive rate; "
          f"see this module's own docstring, validation step 2, before trusting a live run.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run-live-scan", action="store_true",
        help="Actually query the real, live corpus and scan every real question with a "
             "worked_solution_text. Deliberately requires this explicit flag -- NOT the default "
             "action, per explicit instruction (Issue 220): "
             "corpus-wide execution is deliberately deferred until the Issue 214 DeepSeek-OCR "
             "pass finishes. Read-only -- never writes to question_text or any other real field.",
    )
    args = parser.parse_args()

    if args.run_live_scan:
        print("Real corpus-wide scan starting -- this is the FULL, live-DB run.")
        from app.db.session import get_session, init_db

        init_db()
        with get_session() as session:
            candidates = scan_corpus(session)
        print_review_queue(candidates)
        return

    _run_dry_run_harness()


if __name__ == "__main__":
    main()
