"""
Bucket 2 (Section 6: simple geometry with measurements: shapes, angles, area/perimeter),
keyword shortlist builder, same discipline as Bucket 1's own gold-set process (development log
Issue 318): keyword-shortlist candidate rows from the real corpus first, THEN hand-review each
one field-by-field (and against the real source page image where ambiguous) before any row is
trusted as gold-set ground truth. This script only produces the CANDIDATE list, it does not
write status/corrected_answer_value/reason, those are filled in by the manual review pass that
follows, exactly like Issue 318 did for Bucket 1.

Excludes:
- superseded_by IS NOT NULL rows (a newer corrected row exists, don't gold-set the stale one)
- any id already used in Bucket 1's own chart shortlist (the diagram_charts gold set),
  avoids reusing the same row across two different diagram buckets
- rows from data/Hold out/ (Issue 74's forbidden-source rule), verified empirically below
  that none of the has_diagram=1 rows in this DB are sourced from there at all, but the filter is
  kept explicit rather than relying on that being permanently true.

Stratifies into two rough Bucket 2 subtypes per Section 6's own definition ("shapes, angles,
area/perimeter"): ANGLE_KEYWORDS (angle identification/measurement) and
AREA_PERIMETER_KEYWORDS (area/perimeter/side-length measurement), each capped at 12, for a
24-row candidate shortlist matching Bucket 1's own 24-row starting point (Issue 318).

Diagram buckets: Bucket 1 is charts and data, Bucket 2 is simple geometry with measurements,
and Bucket 3 is bar models and number lines.
"""

import json
import re
import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DB_PATH = REPO_ROOT / "psle.db"
BUCKET1_SHORTLIST_PATH = (
    REPO_ROOT / "evaluation" / "model_selection" / "diagram_charts" / "shortlist_for_review.json"
)
OUT_PATH = Path(__file__).parent / "shortlist_candidates.json"
PAGES_DIR = REPO_ROOT / "data" / "extracted" / "pages"

PER_CATEGORY_CAP = 12

ANGLE_KEYWORDS = [
    r"\bangle\b", r"\bangles\b", r"degrees?\b", r"°", r"protractor",
    r"right angle", r"obtuse", r"\bacute\b", r"\breflex\b",
]

AREA_PERIMETER_KEYWORDS = [
    r"\bperimeter\b", r"\barea\b", r"\bradius\b", r"\bdiameter\b", r"circumference",
    r"\bsquare\b", r"\brectangle\b", r"\btriangle\b", r"\bcircle\b", r"\bsemicircle\b",
    r"parallelogram", r"\brhombus\b", r"trapezium", r"\bpentagon\b", r"\bhexagon\b",
    r"\bcuboid\b", r"\bcube\b", r"\bcylinder\b", r"\bprism\b", r"\bquadrilateral\b",
    r"\bpolygon\b",
]

# Bucket 1 chart keywords, used only to DEPRIORITIZE rows that look like they're actually a
# chart-reading question wearing geometry-flavoured words (e.g. "area" used loosely in a bar-graph
# question about "the area covered"), not to hard-exclude, final call stays with human review.
CHART_KEYWORDS = [
    r"bar graph", r"pie chart", r"line graph", r"pictogram", r"bar chart",
]

CATEGORY_PATTERNS = {
    "angle": re.compile("|".join(ANGLE_KEYWORDS), re.IGNORECASE),
    "area_perimeter": re.compile("|".join(AREA_PERIMETER_KEYWORDS), re.IGNORECASE),
}
CHART_PATTERN = re.compile("|".join(CHART_KEYWORDS), re.IGNORECASE)


def bucket1_ids() -> set[int]:
    data = json.loads(BUCKET1_SHORTLIST_PATH.read_text(encoding="utf-8"))
    return {row["id"] for row in data["rows"]}


def image_path_for(source_paper: str, source_page_index: int) -> Path:
    stem = Path(source_paper.replace("\\", "/")).stem
    return PAGES_DIR / stem / f"{source_page_index}.png"


def main() -> None:
    excluded_ids = bucket1_ids()
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    # REAL FILTER ADDED after a first pass surfaced it, not assumed up front: 6 of the first
    # 24 candidates (ids 3185, 3234, 65, 5174, 3785, 3710) carried the pipeline's own
    # `extraction_flag='no_matching_answer_key_row'`, Job A found no matching answer
    # key for these, a quarter of the shortlist unusable before any human review even started.
    # Excluding at the query stage rather than shortlisting-then-discarding: a row with no
    # captured key can't be gold-set ground truth without a real key-search pass (Issue 76,
    # never self-compute a substitute), which is real, separate work out of scope for a keyword
    # shortlist. Also excludes a blank-string answer_value (id=3358, same "no real answer
    # captured" shape, just not caught by the pipeline's own flag since it's a distinct failure
    # mode, an empty string rather than an absent key row).
    cur.execute(
        """
        SELECT id, source_paper, source_page_index, question_text, answer_value,
               worked_solution_text
        FROM questions
        WHERE has_diagram = 1 AND superseded_by IS NULL
          AND (extraction_flag IS NULL OR extraction_flag != 'no_matching_answer_key_row')
          AND answer_value IS NOT NULL AND TRIM(answer_value) != ''
        ORDER BY id
        """
    )
    rows = cur.fetchall()
    con.close()

    print(f"Real has_diagram=1, non-superseded corpus rows: {len(rows)}")

    by_category: dict[str, list[dict]] = {"angle": [], "area_perimeter": []}
    hold_out_hits = 0

    for row in rows:
        if row["id"] in excluded_ids:
            continue
        if "hold out" in row["source_paper"].lower():
            hold_out_hits += 1
            continue

        text = row["question_text"] or ""
        matched_categories = [
            cat for cat, pattern in CATEGORY_PATTERNS.items() if pattern.search(text)
        ]
        if not matched_categories:
            continue

        is_chart_flavoured = bool(CHART_PATTERN.search(text))
        img_path = image_path_for(row["source_paper"], row["source_page_index"])

        entry = {
            "id": row["id"],
            "source_paper": row["source_paper"],
            "source_page_index": row["source_page_index"],
            "question_text": text,
            "answer_value": row["answer_value"],
            "has_worked_solution": bool(row["worked_solution_text"]),
            "matched_categories": matched_categories,
            "chart_flavoured_deprioritize": is_chart_flavoured,
            "image_exists": img_path.exists(),
            "image_path_abs": str(img_path),
        }
        # a row can match both categories (e.g. "find the area and mark the angle"), file it
        # under whichever it matched first for stratification purposes, human review sees both
        for cat in matched_categories:
            by_category[cat].append(entry)

    print(f"has_diagram=1 rows under data/Hold out/ skipped (Issue 74): {hold_out_hits}")
    for cat, entries in by_category.items():
        print(f"{cat}: {len(entries)} raw candidates before capping/dedup")

    # dedupe (a row matching both categories would otherwise appear twice across the two lists)
    #
    # REAL BUG FOUND AND FIXED before this list was ever treated as a candidate shortlist: an
    # earlier version of this sort sorted primarily by ascending `id` as the final tiebreaker.
    # `id` roughly tracks load/insertion order, not anything semantically meaningful, so that
    # tiebreaker silently concentrated all 24 picks into just 3 source papers, all from the same
    # 2021 CA1/SA1 batch, a real representativeness gap, caught by manually inspecting the
    # output paper-by-paper against Bucket 1's own 22-distinct-papers-across-24-rows precedent
    # (Issue 318's shortlist) before doing any human review of question content. Fixed by
    # capping picks per source_paper and using each row's own id as a DETERMINISTIC PRNG seed
    # (not a random module call, keeps re-runs reproducible without needing a stored seed file)
    # to shuffle within each category, so the paper a row happens to belong to no longer decides
    # priority.
    import random

    MAX_PER_PAPER = 2
    seen_ids: set[int] = set()
    per_paper_count: dict[str, int] = {}
    final_rows = []
    for cat in ("angle", "area_perimeter"):
        entries = by_category[cat]
        rng = random.Random(20260918)
        rng.shuffle(entries)
        entries.sort(key=lambda e: e["chart_flavoured_deprioritize"])  # stable: keeps shuffle order within each group
        picked = 0
        for e in entries:
            if e["id"] in seen_ids:
                continue
            if picked >= PER_CATEGORY_CAP:
                break
            if not e["image_exists"]:
                print(f"  SKIP id={e['id']}: page image not found at {e['image_path_abs']}")
                continue
            if per_paper_count.get(e["source_paper"], 0) >= MAX_PER_PAPER:
                continue
            seen_ids.add(e["id"])
            per_paper_count[e["source_paper"]] = per_paper_count.get(e["source_paper"], 0) + 1
            final_rows.append(e)
            picked += 1
        print(f"{cat}: picked {picked} (capped at {PER_CATEGORY_CAP}, max {MAX_PER_PAPER}/paper)")

    distinct_papers = len({r["source_paper"] for r in final_rows})
    print(f"Distinct source papers in final shortlist: {distinct_papers} (Bucket 1 precedent: 22/24)")

    OUT_PATH.write_text(json.dumps({"rows": final_rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{len(final_rows)} candidate rows written to {OUT_PATH}")
    print("NEXT STEP: manual review of each row against real psle.db content and, where")
    print("ambiguous, the real source page image -- same discipline as Issue 318. Do not")
    print("treat this file as gold-set ground truth yet.")


if __name__ == "__main__":
    main()
