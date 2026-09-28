r"""
Find `worked_solution_text` values shared by several distinct, non-superseded rows (Issue 297).

Report only: no `psle.db` writes, and no `--apply` flag, since there is nothing to apply. It is a
sizing scan, like `scan_glued_question_and_mismatched_ground_truth.py`.

Why a separate scan: the earlier three-heuristic sweep (Issue 296) uses fuzzy word-overlap checks.
Issue 297's cross-check found it missed 3 of 15 rows in a duplicated-`worked_solution_text`
cluster on `P6_Maths_2024_SA2_acsprimary.pdf` (ids 3228, 3254, 5649), because their
`question_text` fragments were too short for those heuristics to fire. Exact (or
whitespace-normalised) string equality is a more reliable signal for this failure, where an
answer-key page's text bleeds onto several unrelated rows, and it does not depend on
`question_text` at all. It checks whether that PDF is a one-off or a pattern; it does not replace
the three-heuristic sweep.

Method:
  - Only rows with `superseded_by IS NULL`. A superseded row is expected to share content with
    the row that replaced it, so including those pairs would re-report a known, intended shape.
  - Only rows whose `worked_solution_text` is at least `_MIN_LEN` (40) characters. Short, generic
    text such as a bare "12" or "Ans: 5" recurs across different questions by coincidence, so an
    exact match there is not distinctive. Every example in the Issue 297 cross-check ("Ahmad
    bought more than 10 books...") is over 90 characters.
  - Exact clusters: grouped by the raw string.
  - Near-exact clusters: grouped by the string with whitespace runs collapsed to one space and
    the ends stripped, which catches clusters that differ only in newlines or spacing. No fuzzy
    or edit-distance matching is done; pairwise comparison over about 5,800 rows would be O(n^2)
    and is not needed for this failure. A near-exact cluster is reported only if it is not the
    same set of ids as an exact cluster, so no cluster prints twice.
  - A cluster is 2 or more distinct row ids with the same (normalised) text.

Usage:
    backend/.venv/Scripts/python.exe scripts/data_quality/scan_duplicate_worked_solution_text.py
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "backend"))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from sqlalchemy import select  # noqa: E402

from app.db.session import get_session, init_db  # noqa: E402
from app.models.question import Question  # noqa: E402

_MIN_LEN = 40
_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def main() -> None:
    init_db()
    with get_session() as session:  # type: ignore
        rows = session.execute(
            select(Question.id, Question.worked_solution_text).where(
                Question.superseded_by.is_(None),
                Question.worked_solution_text.is_not(None),
            )
        ).all()

    total_considered = 0
    exact_groups: dict[str, list[int]] = defaultdict(list)
    norm_groups: dict[str, list[int]] = defaultdict(list)
    for row_id, wst in rows:
        wst = wst or ""
        if len(wst.strip()) < _MIN_LEN:
            continue
        total_considered += 1
        exact_groups[wst].append(row_id)
        norm_groups[_normalize(wst)].append(row_id)

    exact_clusters = {k: v for k, v in exact_groups.items() if len(v) >= 2}
    norm_clusters = {k: v for k, v in norm_groups.items() if len(v) >= 2}

    # A norm cluster whose id set exactly matches an exact cluster's id set is redundant to print.
    exact_id_sets = {tuple(sorted(v)) for v in exact_clusters.values()}
    near_only_clusters = {
        k: v for k, v in norm_clusters.items() if tuple(sorted(v)) not in exact_id_sets
    }

    print(f"Rows considered (superseded_by IS NULL, worked_solution_text length >= {_MIN_LEN}): "
          f"{total_considered}")
    print(f"\nEXACT duplicate clusters (byte-identical worked_solution_text, 2+ distinct rows): "
          f"{len(exact_clusters)} cluster(s), {sum(len(v) for v in exact_clusters.values())} row(s) total")

    for text_val, ids in sorted(exact_clusters.items(), key=lambda kv: -len(kv[1])):
        snippet = text_val.replace("\n", " \\n ")[:100]
        print(f"\n  cluster of {len(ids)} rows: {sorted(ids)}")
        print(f"    text: {snippet!r}")

    print(f"\nNEAR-EXACT clusters (identical after whitespace normalization only, NOT already "
          f"covered above): {len(near_only_clusters)} cluster(s), "
          f"{sum(len(v) for v in near_only_clusters.values())} row(s) total")
    for text_val, ids in sorted(near_only_clusters.items(), key=lambda kv: -len(kv[1])):
        snippet = text_val[:100]
        print(f"\n  cluster of {len(ids)} rows: {sorted(ids)}")
        print(f"    normalized text: {snippet!r}")

    print("\nREPORT ONLY — no psle.db writes. This scan takes no --apply argument.")


if __name__ == "__main__":
    main()
