"""
Normalise school names in exam-paper filenames, with a human review step (Issue 61).

School papers are named inconsistently across years: 2021 uses hyphens and Title-Case, while 2022
onward uses underscores and lower case. The same school also appears under different spellings.
Run this before extraction, and again whenever new papers are added.

The approach is deliberately not fully automatic:
  Stage 1 (automatic, safe): lower-case the school-name part and remove hyphens, underscores and
  spaces. This merges pure formatting variants (henry-park == henrypark == Henry-Park).
  Stage 2 (human review, required): every group is written to a review CSV rather than merged
  further. Some pairs may or may not be the same school (mgs vs methodistgirls, taonan vs
  tao-nan, nanchiau vs nan-chiau), and a wrong merge would corrupt the held-out/training split
  (Issue 23), so a person resolves them once, by hand, before extraction runs.

Resolutions persist across runs (Issue 72). `school_name_review.csv` is a working sheet that every
--real, --scan or --demo run regenerates from the filenames present. Decisions already made are
kept in `tools/school_name_canonical.csv`, which no scan overwrites. A key already there is filled
in on the review sheet and marked "no (already resolved)" instead of "YES", so it is not decided
again when new papers trigger a re-scan. Only new or unresolved keys are flagged.

Usage:
    Scan data/School papers/ and data/Yearly/ (run from the repository root, or pass --data-root):
        python tools/normalize_filenames.py --real

    Scan specific folders instead:
        python tools/normalize_filenames.py --scan "data/School papers" "data/Yearly"

    Run against the filenames already observed in this project (no PDFs needed):
        python tools/normalize_filenames.py --demo

    After filling in canonical_school_name for the rows marked YES in school_name_review.csv,
    save those answers to the permanent store so later scans do not ask again:
        python tools/normalize_filenames.py --commit

Output: tools/school_name_review.csv, listing every group with its member filenames. Groups
already decided are filled in and marked resolved; the rest are flagged for a decision.
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

# Filenames observed in this project's data/School papers/ and data/Hold out/ folders. Used by
# --demo to show the tool working on observed names without the PDFs present. Extend the list, or
# point --scan at the folder once the files are in place.
REAL_OBSERVED_FILENAMES = [
    # 2021, hyphen, Title-Case
    "P6-Maths-2021-SA1-ACS-Junior.pdf", "P6-Maths-2021-SA1-Ai-Tong.pdf",
    "P6-Maths-2021-SA1-Catholic-High.pdf", "P6-Maths-2021-SA1-CHIJ.pdf",
    "P6-Maths-2021-SA1-Henry-Park.pdf", "P6-Maths-2021-SA1-Methodist-Girls.pdf",
    "P6-Maths-2021-SA1-Nan-Hua.pdf", "P6-Maths-2021-SA1-Nanyang.pdf",
    "P6-Maths-2021-SA1-Raffles-Girls.pdf", "P6-Maths-2021-SA1-Rosyth.pdf",
    "P6-Maths-2021-SA1-Tao-Nan.pdf", "P6-Maths-2021-SA2-ACS-Junior.pdf",
    "P6-Maths-2021-SA2-Catholic-High.pdf", "P6-Maths-2021-SA2-CHIJ.pdf",
    "P6-Maths-2021-SA2-Henry-Park.pdf", "P6-Maths-2021-SA2-Maris-Stella.pdf",
    "P6-Maths-2021-SA2-Methodist-Girls.pdf", "P6-Maths-2021-SA2-Nan-Chiau.pdf",
    "P6-Maths-2021-SA2-Raffles-Girls.pdf", "P6-Maths-2021-SA2-Red-Swastika.pdf",
    "P6-Maths-2021-SA2-Rosyth.pdf", "P6-Maths-2021-SA2-Tao-Nan.pdf",
    "P6-Maths-2021-CA1-Henry-Park.pdf", "P6-Maths-2021-CA1-Rosyth.pdf",
    # 2022, underscore, lowercase
    "P6_Maths_2022_SA2_acsjunior.pdf", "P6_Maths_2022_SA2_chij.pdf",
    "P6_Maths_2022_SA2_henrypark.pdf", "P6_Maths_2022_SA2_methodistgirls.pdf",
    "P6_Maths_2022_SA2_nanchiau.pdf", "P6_Maths_2022_SA2_nanhua.pdf",
    "P6_Maths_2022_SA2_nanyang.pdf", "P6_Maths_2022_SA2_redswastika.pdf",
    "P6_Maths_2022_SA2_taonan.pdf",
    # 2023, underscore, Mixed_Case. MGS appears here as a separate name from methodistgirls.
    "P6_Maths_2023_SA2_ACS_Junior.pdf", "P6_Maths_2023_SA2_ACS_Primary.pdf",
    "P6_Maths_2023_SA2_AiTong.pdf", "P6_Maths_2023_SA2_Catholichigh.pdf",
    "P6_Maths_2023_SA2_CHIJ.pdf", "P6_Maths_2023_SA2_Marisstella.pdf",
    "P6_Maths_2023_SA2_MGS.pdf", "P6_Maths_2023_SA2_Nanhua.pdf",
    "P6_Maths_2023_SA2_Rafflesgirls.pdf", "P6_Maths_2023_SA2_Redswastika.pdf",
    "P6_Maths_2023_SA2_Rosyth.pdf", "P6_Maths_2023_SA2_Taonan.pdf",
    "P6_Maths_2023_WA1_Raffles.pdf", "P6_Maths_2023_WA2_RafflesGirls.pdf",
    # 2024
    "P6_Maths_2024_SA2_acsjunior.pdf", "P6_Maths_2024_SA2_acsprimary.pdf",
    "P6_Maths_2024_SA2_catholichigh.pdf", "P6_Maths_2024_SA2_henrypark.pdf",
    "P6_Maths_2024_SA2_methodistgirls.pdf", "P6_Maths_2024_SA2_nanhua.pdf",
    "P6_Maths_2024_SA2_nanyang.pdf", "P6_Maths_2024_SA2_rafflesgirls.pdf",
    "P6_Maths_2024_SA2_redswastika.pdf", "P6_Maths_2024_SA2_rosyth.pdf",
    "P6_Maths_2024_SA2_taonan.pdf", "P6_Maths_2024_WA1_Methodistgirls.pdf",
    "P6_Maths_2024_WA1_Rafflesgirls.pdf",
    # 2025. Both mgs and mgspayalebar appear; most likely different schools.
    "P6_Maths_2025_SA2_acsjunior.pdf", "P6_Maths_2025_SA2_aitong.pdf",
    "P6_Maths_2025_SA2_catholichigh.pdf", "P6_Maths_2025_SA2_chij.pdf",
    "P6_Maths_2025_SA2_henrypark.pdf", "P6_Maths_2025_SA2_mgs.pdf",
    "P6_Maths_2025_SA2_mgspayalebar.pdf", "P6_Maths_2025_SA2_nanhua.pdf",
    "P6_Maths_2025_SA2_nanyang.pdf", "P6_Maths_2025_SA2_raffles.pdf",
    "P6_Maths_2025_SA2_redswastika.pdf", "P6_Maths_2025_SA2_rosyth.pdf",
    "P6_Maths_2025_SA2_scgs.pdf", "P6_Maths_2025_SA2_taonan.pdf",
    "P6_Maths_2025_WA1_raffles.pdf", "P6_Maths_2025_WA2_raffles.pdf",
]

# Known ambiguous pairs, flagged in advance by manual inspection. They normalise to different keys
# (correctly not merged automatically), but are printed so the reviewer of
# school_name_review.csv knows which pairs need the most thought.
KNOWN_AMBIGUOUS_PAIRS = [
    ("mgs", "methodistgirls", "Could be the same school abbreviated differently, or two "
     "different schools (Methodist Girls' School vs. a different MGS). CONFIRM BY HAND."),
    ("mgs", "mgspayalebar", "Both appear in 2025 SA2 — 'payalebar' suggests these are two "
     "DIFFERENT campuses/schools, not a duplicate. Do not merge."),
    ("taonan", "tao-nan", "Almost certainly the same school (Tao Nan School), just the 2021 "
     "hyphenated form vs. later unhyphenated. Should auto-merge under Stage 1 — verify it did."),
    ("nanchiau", "nan-chiau", "Almost certainly the same school (Nan Chiau), same pattern as "
     "Tao Nan above. Should auto-merge under Stage 1 — verify it did."),
    ("raffles", "rafflesgirls", "Likely both refer to Raffles Girls' Primary School, but "
     "'raffles' alone is ambiguous without seeing the actual paper. CONFIRM BY HAND."),
    # Found by running --real against the data/ folder. They were not in the hand-inspected list
    # above, which shows Stage 2 review is needed beyond the cases anticipated.
    ("marisstella", "marisstellahigh", "2021/2023 'marisstella' vs. 2024 'marisstellahigh' "
     "(from P6_Maths_2024_SA2_marisstellahigh.pdf). Plausibly the same school (Maris Stella High "
     "School) with a naming change/addition in 2024, or two different entities. CONFIRM BY HAND."),
    ("payalebarmethodist", "mgspayalebar", "2022 'payalebarmethodist' vs. 2025 'mgspayalebar' — "
     "plausibly the same campus (Methodist Girls' School, Paya Lebar) named two different ways "
     "across years, which would also implicate 'mgs' and 'methodistgirls' above. CONFIRM BY HAND."),
]


def normalize_stage1(filename: str) -> str:
    """Stage 1: return the normalised school-name key for a filename.

    Drops the extension, isolates the school-name part, lower-cases it and removes hyphens,
    underscores and spaces. It does not guess whether two names mean the same school (that is
    Stage 2, human review); it only merges formatting differences.
    """
    stem = Path(filename).stem
    # Strip the common prefix pattern (P6, Maths, year, paper-type) to isolate the school name.
    # The regex is loose because the prefix format varies by year (Issue 61); it takes whatever
    # follows the paper-type token.
    match = re.search(
        r"(?:SA1|SA2|WA1|WA2|WA3|CA1)[-_](.+)$", stem, flags=re.IGNORECASE
    )
    school_part = match.group(1) if match else stem
    normalized = re.sub(r"[-_\s]", "", school_part).lower()
    return normalized


def build_review_table(filenames: list[str]) -> dict[str, list[str]]:
    """Group filenames by their Stage 1 key. Returns {normalized_key: [filenames]}."""
    groups: dict[str, list[str]] = defaultdict(list)
    for fname in filenames:
        key = normalize_stage1(fname)
        groups[key].append(fname)
    return dict(groups)


CANONICAL_MAP_PATH = Path(__file__).parent / "school_name_canonical.csv"


def load_canonical_map(path: Path) -> dict[str, dict[str, str]]:
    """Load the canonical names already resolved by hand, keyed by Stage 1 key.

    This is the persistent store (Issue 72), separate from school_name_review.csv, which every
    --real, --scan or --demo run regenerates and which would otherwise lose hand-filled answers
    whenever new papers arrive. Returns {} if the file does not exist yet, as on a fresh checkout.

    Each entry also has a `verified` flag (Issue 93). A canonical_school_name means someone
    recorded a decision, not that it was checked against the source PDFs. `verified` is that
    stronger claim, set True only after direct evidence (for example reading the cover pages).
    Missing or unrecognised values default to False, so nothing is assumed verified; this is the
    same fail-closed approach as the safety classifier (Issue 58).
    """
    if not path.exists():
        return {}
    mapping: dict[str, dict[str, str]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row["normalized_key"].strip()
            if key:
                mapping[key] = {
                    "canonical_school_name": row.get("canonical_school_name", "").strip(),
                    "verified": row.get("verified", "").strip().lower() == "true",
                    "notes": row.get("notes", "").strip(),
                }
    return mapping


def write_canonical_map(mapping: dict[str, dict[str, str]], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["normalized_key", "canonical_school_name", "verified", "notes"])
        for key, data in sorted(mapping.items()):
            verified = data.get("verified", False)
            writer.writerow([
                key, data["canonical_school_name"],
                "true" if verified else "false", data["notes"],
            ])


def write_review_csv(
    groups: dict[str, list[str]], output_path: Path, canonical_map: dict[str, dict[str, str]],
) -> None:
    already_resolved = 0
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "normalized_key", "member_filenames", "member_count",
            "needs_human_review", "canonical_school_name (FILL IN)", "notes",
        ])
        for key, members in sorted(groups.items()):
            resolved = canonical_map.get(key)
            if resolved and resolved["canonical_school_name"]:
                # Decided in an earlier run (Issue 72); a re-run, e.g. for next year's papers,
                # must not ask again.
                status = "no (already resolved)"
                canonical_name = resolved["canonical_school_name"]
                notes = resolved["notes"]
                already_resolved += 1
            else:
                needs_review = len(members) > 1  # several raw filenames merged into this
                # key. Worth a human check even for Stage 1 merges, which are safe by design but
                # not proven safe in every case.
                status = "YES" if needs_review else "no (unique)"
                canonical_name = ""  # filled in by the reviewer
                notes = ""
            writer.writerow([key, "; ".join(members), len(members), status, canonical_name, notes])

    print(f"Review table written: {output_path}")
    print(f"{len(groups)} distinct normalized school keys found from {sum(len(m) for m in groups.values())} filenames.")
    if canonical_map:
        print(f"{already_resolved} key(s) auto-filled from previously resolved entries in "
              f"{CANONICAL_MAP_PATH.name} — no re-review needed for those.")
    print()
    print("Known ambiguous pairs to pay special attention to when reviewing:")
    for name_a, name_b, note in KNOWN_AMBIGUOUS_PAIRS:
        print(f"  - {name_a!r} vs {name_b!r}: {note}")


def commit_resolutions(review_path: Path, canonical_path: Path) -> None:
    """Save the names filled in on the review sheet to the permanent canonical store (Issue 72).

    Reads the canonical_school_name values entered in school_name_review.csv and adds them to
    school_name_canonical.csv, so later --real or --scan runs do not ask about those keys again.

    An existing entry that disagrees with a new one is never overwritten: resolving that
    automatically would be the kind of guessed merge or split Issue 61 exists to prevent. The
    conflict is printed for a person to fix, either in school_name_canonical.csv or by correcting
    the review CSV and re-running --commit.

    New entries are always written with verified=False (Issue 93). A name entered during review
    is a decision, not proof that it was checked against the source PDFs. Setting verified=True
    is a separate manual step in school_name_canonical.csv, done after direct evidence such as
    reading the cover pages. --commit never changes an existing entry's verified flag.
    """
    if not review_path.exists():
        print(f"{review_path} does not exist — run --real, --scan, or --demo first.")
        return

    canonical_map = load_canonical_map(canonical_path)
    added, unchanged, conflicts = 0, 0, 0

    with open(review_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row["normalized_key"].strip()
            name = row.get("canonical_school_name (FILL IN)", "").strip()
            notes = row.get("notes", "").strip()
            if not name:
                continue  # not resolved yet; nothing to commit
            existing = canonical_map.get(key)
            if existing is None:
                canonical_map[key] = {
                    "canonical_school_name": name, "verified": False, "notes": notes,
                }
                added += 1
            elif existing["canonical_school_name"] == name:
                unchanged += 1
            else:
                conflicts += 1
                print(
                    f"CONFLICT for {key!r}: canonical store already has "
                    f"{existing['canonical_school_name']!r}, review CSV has {name!r}. "
                    f"NOT overwritten — resolve by hand."
                )

    write_canonical_map(canonical_map, canonical_path)
    print(f"Committed to {canonical_path}: {added} new, {unchanged} unchanged, {conflicts} conflict(s).")
    if conflicts:
        print("Fix conflicts by hand (in the CSV or the review sheet) and re-run --commit.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scan", type=str, nargs="+", default=None,
        help="One or more real folders to scan for PDF filenames, e.g. 'data/School papers' 'data/Yearly'",
    )
    parser.add_argument(
        "--real", action="store_true",
        help="Scan the two canonical folders the design specification Section 15 Phase 0 step 6 requires: "
             "<data-root>/School papers and <data-root>/Yearly",
    )
    parser.add_argument(
        "--data-root", type=str, default="data",
        help="Root of the data folder, used with --real (default: 'data', i.e. run from repo root)",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run against the real filenames already observed in this project (no PDFs needed)",
    )
    parser.add_argument(
        "--commit", action="store_true",
        help="Persist hand-filled canonical_school_name values from school_name_review.csv into "
             "the permanent school_name_canonical.csv store (Issue 72), so future runs don't "
             "ask about the same normalized key again. Run this AFTER filling in the review CSV.",
    )
    args = parser.parse_args()

    if args.commit:
        commit_resolutions(Path(__file__).parent / "school_name_review.csv", CANONICAL_MAP_PATH)
        return

    if args.real:
        data_root = Path(args.data_root)
        folders = [data_root / "School papers", data_root / "Yearly"]
        filenames = []
        for folder in folders:
            found = [p.name for p in folder.rglob("*.pdf")]
            print(f"Scanned {folder}, found {len(found)} PDF files.")
            filenames.extend(found)
    elif args.scan:
        filenames = []
        for scan_arg in args.scan:
            folder = Path(scan_arg)
            found = [p.name for p in folder.rglob("*.pdf")]
            print(f"Scanned {folder}, found {len(found)} PDF files.")
            filenames.extend(found)
    elif args.demo:
        filenames = REAL_OBSERVED_FILENAMES
        print(f"Demo mode: using {len(filenames)} real filenames observed in this project.")
    else:
        parser.error("Pass --real, --scan <folder> [<folder> ...], --demo, or --commit")
        return

    groups = build_review_table(filenames)
    output_path = Path(__file__).parent / "school_name_review.csv"
    canonical_map = load_canonical_map(CANONICAL_MAP_PATH)
    write_review_csv(groups, output_path, canonical_map)
    print()
    print("NEXT STEP (mandatory, per Issue 61): open school_name_review.csv and fill in")
    print("'canonical_school_name' for every row still marked YES, by hand, before Phase 1")
    print("extraction runs. Do NOT skip this and let extraction code guess — that's exactly the")
    print("risk this tool exists to prevent.")
    print("THEN (Issue 72): run 'python tools/normalize_filenames.py --commit' to persist")
    print("those answers permanently, so the next scan doesn't ask about them again.")


if __name__ == "__main__":
    main()
