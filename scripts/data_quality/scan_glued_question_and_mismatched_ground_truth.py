r"""
Corpus-wide scan for the Issue 296 defect family. Report only: no `psle.db` writes.

The defect was found by a structured prescreen of 465 bootstrapped review-queue rows. It affects
live rows and is separate from the non_question_boilerplate family (Issues 209, 262, 283, 285):
a row's `question_text` glues two unrelated questions together, or its `worked_solution_text` or
`answer_value` belongs to a different question. Checked examples, all live and unflagged when
this scan was written:
  id=192   question_text is a percentage word problem, an `Ans:` marker, then an unrelated
           "figure made up of semicircles" question joined straight on.
  id=838   question_text is a tank-drainage setup with no (a)/(b)/(c) sub-parts, but the
           worked_solution_text and answer_value answer (a)/(b)/(c) sub-questions that only
           exist on id=840.
  id=840   the other side of the same split: its question_text holds the (a)/(b)/(c)
           sub-questions for the tank problem, but its worked_solution_text and answer_value are
           a semicircle radius and area calculation that belongs to neither row.
  id=1127  question_text is exam boilerplate ("show your working clearly... [45 marks]") with no
           question, yet it carries a populated, unrelated answer_value. Issue 209 fixed this
           shape for an earlier batch; this row was missed.

Three heuristics run independently, each with known precision limits. The union is reported, not
one confident count: this sizes the problem and is not a classifier.

  Pattern A: a mid-text "Ans:" marker (not the trailing kind cleaned in Issue 295) followed by at
    least 20 characters. Catches id=192. Known false positive: a legitimate multi-part question
    can have an interior "Ans: (a) [1]" mark-scheme line between two of its own sub-parts (the
    safe id=302 shape from Issue 295). The pattern cannot tell these apart, so hits need human
    triage.

  Pattern B: `worked_solution_text` has at least 2 distinct numbers and none of them appear in
    `question_text`. Meant for cross-row splits (working that solves someone else's numbers). It
    misses id=838 and id=840 themselves: each shares a small number with unrelated text by chance
    (mark-scheme brackets "[1]" and "[2]", or "11:04" against an unrelated "11" in the working).

  Pattern C: `worked_solution_text` uses a distinctive word (4 or more letters, not a bare unit)
    that never appears in `question_text`, while `question_text` is substantial (over 30
    characters, so boilerplate-shaped text like id=1127 is not mixed in). Catches id=840:
    "Radius" and "Area" appear in its working but not in its tank-drainage question.

No single heuristic catches all four examples with full precision. The union is the first
sizing signal, and the sample must be read by a person before anything is treated as confirmed,
as with earlier corpus-wide flags (Issue 283: read every one before acting).

Output: data/extracted/glued_question_sweep_manifest.json.

Usage:
    backend/.venv/Scripts/python.exe scripts/data_quality/scan_glued_question_and_mismatched_ground_truth.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "backend"))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from sqlalchemy import text  # noqa: E402

from app.db.session import get_session, init_db  # noqa: E402
from app.models.question import NON_QUESTION_EXTRACTION_FLAG  # noqa: E402

NQB = NON_QUESTION_EXTRACTION_FLAG
_MID_ANS_RE = re.compile(r"\bAns\b\s*:?", re.I)  # \b after "Ans" as well:
# without it the pattern matches the start of the word "answer" ("...answer correct to 2 decimal
# places"), which inflated the count with ordinary question text (ids 21, 71, 93, 100 and others).
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_WORD_RE = re.compile(r"[A-Za-z]{4,}")
_UNIT_WORDS = {"centimetre", "centimetres", "metre", "metres", "kilogram", "kilograms", "litre",
               "litres", "minute", "minutes", "hour", "hours", "second", "seconds"}


def main() -> None:
    init_db()
    with get_session() as session:  # type: ignore
        live_where = (
            "superseded_by IS NULL AND "
            f"(extraction_flag IS NULL OR extraction_flag NOT LIKE '%{NQB}%')"
        )
        rows = session.execute(text(
            f"SELECT id, question_text, worked_solution_text, answer_value FROM questions "
            f"WHERE {live_where}"
        )).all()

    print(f"Live rows scanned: {len(rows)}")
    qt_by_id = {r.id: (r.question_text or "") for r in rows}

    # --- Pattern A ---
    # Strip one leading label run (a mark-scheme bracket and/or a sub-part letter, e.g.
    # "(a) [2]", "[3]", "a)") before judging the tail. PSLE papers routinely follow
    # "Ans: (a) [2]" with the next sub-part's stem, "(b) Mr Teo has...", which is normal
    # structure, not a glued question; in a 30-row sample most raw hits had this shape. If the
    # remainder starts with another sub-part label, it is treated as the next stem and not
    # flagged. This removes only the dominant false-positive shape; hits still need human review.
    _LABEL_PREFIX_RE = re.compile(r"^\s*(?:\(?[a-e]\)?\s*)?(?:\[\s*\d{1,2}\s*\]\s*)?", re.I)
    _NEXT_LABEL_RE = re.compile(r"^\(?[a-e]\)", re.I)
    pattern_a: dict[int, str] = {}
    for r in rows:
        qt = r.question_text or ""
        for m in _MID_ANS_RE.finditer(qt):
            tail = qt[m.end():].strip()
            if len(tail) < 20:
                continue
            after_one_label = _LABEL_PREFIX_RE.sub("", tail, count=1).strip()
            if _NEXT_LABEL_RE.match(after_one_label):
                continue  # "(a) [2]\n(b) next stem": normal structure
            pattern_a[r.id] = tail[:90]
            break

    # --- Pattern B ---
    pattern_b: dict[int, dict] = {}
    for r in rows:
        ws = r.worked_solution_text or ""
        if not ws.strip():
            continue
        ws_nums = set(_NUM_RE.findall(ws))
        if len(ws_nums) < 2:
            continue
        qt_nums = set(_NUM_RE.findall(r.question_text or ""))
        if not (ws_nums & qt_nums):
            pattern_b[r.id] = {"ws_numbers": sorted(ws_nums), "qt_numbers": sorted(qt_nums)}

    # --- Pattern C ---
    pattern_c: dict[int, dict] = {}
    for r in rows:
        ws = r.worked_solution_text or ""
        qt = r.question_text or ""
        if not ws.strip() or len(qt.strip()) <= 30:
            continue
        ws_words = {w.lower() for w in _WORD_RE.findall(ws)} - _UNIT_WORDS
        if not ws_words:
            continue
        qt_lower = qt.lower()
        missing = {w for w in ws_words if w not in qt_lower}
        if missing and missing == ws_words:  # none of the working's words appear in qt
            pattern_c[r.id] = {"ws_words": sorted(ws_words)[:10]}

    union_ids = set(pattern_a) | set(pattern_b) | set(pattern_c)
    multi_hit = [i for i in union_ids
                 if sum(i in p for p in (pattern_a, pattern_b, pattern_c)) >= 2]

    print(f"\nPATTERN A (mid-text 'Ans:' + >=20 chars following): {len(pattern_a)}")
    print(f"PATTERN B (worked_solution numbers, zero overlap w/ question_text): {len(pattern_b)}")
    print(f"PATTERN C (worked_solution distinctive words, zero overlap w/ question_text): {len(pattern_c)}")
    print(f"\nUNION (flagged by >=1 pattern): {len(union_ids)}")
    print(f"Flagged by >=2 patterns (higher-confidence subset): {len(multi_hit)}")

    print("\n--- Pattern A sample (15) ---")
    for i, tail in list(pattern_a.items())[:15]:
        print(f"  id={i}  ...after 'Ans:': {tail!r}")

    print("\n--- Pattern B sample (15) ---")
    for i, d in list(pattern_b.items())[:15]:
        print(f"  id={i}  ws_numbers={d['ws_numbers'][:8]}  qt_numbers={d['qt_numbers'][:8]}")

    print("\n--- Pattern C sample (15) ---")
    for i, d in list(pattern_c.items())[:15]:
        print(f"  id={i}  ws_words_absent_from_qt={d['ws_words']}")

    print("\n--- multi-hit (>=2 patterns) sample, full detail (up to 15) ---")
    for i in multi_hit[:15]:
        print(f"  id={i}: patterns={[n for n, p in (('A', pattern_a), ('B', pattern_b), ('C', pattern_c)) if i in p]}")
        print(f"    question_text: {qt_by_id[i][:120]!r}")

    manifest = {
        "pattern_a_mid_ans_marker": [{"id": i, "tail": t} for i, t in pattern_a.items()],
        "pattern_b_zero_number_overlap": [{"id": i, **d} for i, d in pattern_b.items()],
        "pattern_c_zero_word_overlap": [{"id": i, **d} for i, d in pattern_c.items()],
        "union_ids": sorted(union_ids),
        "multi_hit_ids": sorted(multi_hit),
    }
    out_path = _REPO_ROOT / "data" / "extracted" / "glued_question_sweep_manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFull manifest -> {out_path}")
    print("\nREPORT ONLY — no psle.db writes. Every flagged row needs individual human review "
          "before any action (flag / restore / re-attribute) — same discipline as every prior "
          "corpus-wide sweep in this project.")


if __name__ == "__main__":
    main()
