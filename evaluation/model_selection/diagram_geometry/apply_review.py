"""
One-off script applying the manual review pass (Issue 331) to
shortlist_candidates.json, producing shortlist_for_review.json in the same schema Bucket 1 used
(Issue 318), id, assumed_geometry_subtype, status, stored_answer_value, corrected_answer_value,
reason, image_path_abs, question_text, question_text_source.

Review basis for each row: cross-checked against the real psle.db `worked_solution_text` field
where present, and against the real source page image directly for the 3 rows worked_solution_text
alone could not resolve (ids 933, 3422, 5590), same two-tier discipline as Issue 318.
Not run automatically as part of build_shortlist.py, this file only exists to apply this ONE
review pass; re-running build_shortlist.py from scratch would need a fresh review, not a re-run of
this file's hardcoded judgments.

Diagram buckets: Bucket 1 is charts and data, Bucket 2 is simple geometry with measurements,
and Bucket 3 is bar models and number lines.
"""

import json
from pathlib import Path

CANDIDATES_PATH = Path(__file__).parent / "shortlist_candidates.json"
OUT_PATH = Path(__file__).parent / "shortlist_for_review.json"

# id -> (status, corrected_answer_value or None, reason)
REVIEW = {
    3: ("clean", None, "MCQ, real text options present in question_text, bare-digit index matches the established convention (Issue 10/175). No worked_solution_text to cross-check; no contradicting evidence found."),
    652: ("clean", None, "MCQ, real text options present. Index convention. No worked_solution_text; no contradicting evidence."),
    1337: ("clean", None, "MCQ, real text options present. Index convention. No worked_solution_text; no contradicting evidence."),
    1348: ("clean", None, "MCQ, real text options present. Index convention. No worked_solution_text; no contradicting evidence."),
    3165: ("clean", None, "MCQ, real text options present. Index convention. No worked_solution_text; no contradicting evidence."),
    1663: ("clean", None, "MCQ, real text options present. Index convention. No worked_solution_text; no contradicting evidence."),
    5293: ("clean", None, "MCQ, real text options present. Index convention. No worked_solution_text; no contradicting evidence."),
    3983: ("clean", None, "MCQ, real text options present. Index convention. No worked_solution_text; no contradicting evidence."),
    183: ("clean", None, "Verified via worked_solution_text: <ABC=180-69=111, <DEF=180-29-111=40. Matches stored value exactly."),
    5617: ("clean", None, "Verified via worked_solution_text: final line 71+60=131 matches stored 131 exactly."),
    2503: ("clean", None, "Verified via worked_solution_text: final line =693 matches stored '693 cm2' exactly (intermediate OCR noise in the working itself does not affect the final line)."),
    5015: ("corrected", "50°", "Verified via worked_solution_text: <HDB=180-70-30-30=50. Stored value '50°.' carries a stray trailing period, stripped."),
    4330: ("corrected", "65°", "Stored value '96°-31°=65°' is raw arithmetic left in the field, same pattern as Bucket 1's own id=2538 (Issue 318). Arithmetic is complete and self-consistent (96-31=65); no worked_solution_text exists separately, but nothing here is ambiguous or partial the way ids 933/3422/5590 are. Cleaned to the final value."),
    4856: ("corrected", "23°", "Stored value '180°-134°/2=23°' is raw arithmetic left in the field, same pattern as id=4330/2538. Complete, self-consistent arithmetic; cleaned to the final value."),
    500: ("corrected", "54cm", "Stored value '1/2 x 22/7 x 21 + 21 = 33 + 21 =54cm' is raw arithmetic left in the field. Independently re-checked: semicircle perimeter = (1/2 x pi x d) + d = (1/2 x 22/7 x 21) + 21 = 33+21 = 54cm. Complete, self-consistent. Cleaned to the final value."),
    4356: ("corrected", "(a) 21° (b) not captured", "Verified via worked_solution_text: <LOK=180-108-51=21, matches stored '21°' exactly -- this is part (a) only. Part (b) (<LNM) has no working captured anywhere in the DB and is NOT independently derived here (Issue 76 -- do not self-compute a substitute for a missing real key). Only part (a) is gold-set-usable."),
    1568: ("corrected", "(a) 77° (b) 76°", "REAL FIND: worked_solution_text shows TWO complete derivations -- (a) ...=77 (b) ...=76 -- but the stored answer_value ('76°') only ever captured part (b). Part (a)'s real value (77°) was sitting in worked_solution_text the whole time, unused. Both parts now recovered directly from the DB's own real worked_solution_text, not self-computed."),
    2657: ("corrected", "(a) 216cm2 (b) 6", "REAL FIND, same shape as id=1568: worked_solution_text shows (a) 120+36+60=216cm2 (b) 10/3=3r1, 3/3=1, 6/3=2, 3x1x2=6 -- stored answer_value ('6') only ever captured part (b). Part (a)'s real value (216cm2) recovered directly from worked_solution_text."),
    3905: ("corrected", "(a) 30 (b) 7", "REAL FIND, same shape as id=1568/2657: worked_solution_text shows (a) 30 (b) 102-12=90, 90/18=5, 5+2=7 -- stored answer_value ('7') only ever captured part (b). Part (a)'s real value (30) recovered directly from worked_solution_text."),
    1745: ("corrected", "6ℓ", "Verified via worked_solution_text: 2x2x1=4, 10-4=6L -- matches stored bare '6' with the unit (ℓ, matching the question's own 'Container A contains 10 ℓ of water') restored from the same field, not guessed."),
    1304: ("corrected", "(a) 8cm (b) 28°", "Stored value 'a) 8b) 28' is both parts already, just missing unit separators. Units taken directly from the question's own wording ('length ... to the nearest cm' / 'the smallest angle'), not guessed -- reformatted to '(a) 8cm (b) 28°'. This is a measure-from-the-printed-figure question with no computable worked_solution_text possible by design."),
    3422: ("corrected", "96", "REAL, CONFIRMED VIA IMAGE + worked_solution_text: the real source page (data/extracted/pages/P6_Maths_2024_SA2_marisstellahigh/30.png) shows ONLY part (a), 'How many circles are printed on this square tile?' -- a COUNT question. The stored answer_value ('35% (Ans)') is a PERCENTAGE, structurally the wrong answer type for this question -- confirmed via worked_solution_text, which shows TWO full derivations: (a) 2+6=8, 48/8=6, 48/6=8, 6x8x2=96 (b) an area-percentage calculation ending '...=35%=(Ans)'. The stored answer_value captured part (b)'s result and attached it to part (a)'s captured question_text -- a real cross-part mismatch (same failure class as Issue 2/5, wrong working attached to wrong question), not a hallucination or guess on this entry's part. Corrected to 96, part (a)'s real, independently-confirmed value, matching the specific question_text this gold-set row actually carries."),
    933: ("excluded", None, "NEEDS_REAL_KEY. REAL, CONFIRMED VIA IMAGE (data/extracted/pages/P6-Maths-2021-SA2-Henry-Park/29.png): the shaded figure is a leaf/lens shape formed by overlapping quarter-circles inside a 10cm x 12cm bounding box, with the question explicitly stating 'Take pi=3.14'. The stored answer_value ('12x10=120') is exactly the BOUNDING RECTANGLE's area (10x12), using no pi at all despite the problem requiring it for any quarter-circle-based area -- almost certainly a truncated/partial capture of a longer real derivation, matching Bucket 1's id=4210 pattern (Issue 318). No worked_solution_text exists to cross-check. Per Issue 76 (never self-compute a substitute for a missing real key even if independently derivable), NOT corrected here -- excluded pending a real key search against the source paper's own answer section."),
    5590: ("excluded", None, "NEEDS_REAL_KEY. REAL, CONFIRMED VIA IMAGE (data/extracted/pages/P6_Maths_2023_WA1_Nanhua/10.png): the page shows angle A=105°, angle D=32°, target angle x at C, tick marks showing BD=BC (isosceles triangle BDC). The stored answer_value ('LBDC') is not a numeric angle at all -- it reads as a garbled OCR misread of an angle LABEL (most likely '<BDC', the '<' misread as 'L'), and the SAME garbled text appears verbatim inside worked_solution_text ('X = LBDC') after a real, correct partial derivation ('180-105-32=43') -- confirming the real final numeric answer was never actually captured anywhere in this DB row, on either field. Per Issue 76, NOT self-computed here even though the partial working suggests a plausible path -- excluded pending a real key search."),
}

CATEGORY_LABEL = {
    frozenset(["angle"]): "angle",
    frozenset(["area_perimeter"]): "area_perimeter",
    frozenset(["angle", "area_perimeter"]): "angle_and_area_perimeter",
}


def main() -> None:
    data = json.loads(CANDIDATES_PATH.read_text(encoding="utf-8"))
    rows_out = []
    counts = {"clean": 0, "corrected": 0, "excluded": 0}
    for row in data["rows"]:
        rid = row["id"]
        if rid not in REVIEW:
            raise SystemExit(f"id={rid} has no review decision recorded -- fix REVIEW dict first")
        status, corrected, reason = REVIEW[rid]
        counts[status] += 1
        rows_out.append(
            {
                "id": rid,
                "assumed_geometry_subtype": CATEGORY_LABEL[frozenset(row["matched_categories"])],
                "status": status,
                "stored_answer_value": row["answer_value"],
                "corrected_answer_value": corrected,
                "reason": reason,
                "image_path_abs": row["image_path_abs"],
                "question_text": row["question_text"],
                "question_text_source": "psle.db, unmodified",
            }
        )

    out = {"summary": counts, "rows": rows_out}
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT_PATH}: {counts}")


if __name__ == "__main__":
    main()
