"""Job A OCR bake-off: normalize raw model output into the shared schema (protocol Section 4)
and score it against the gold-set table in docs/DATA_AND_EVALUATION.md, in one pass per model.

CPU-only, stdlib + jiwer. Run with the shared scoring venv:
    envs/scoring/.venv/Scripts/python.exe normalize_and_score.py
from evaluation/model_selection/ocr_extraction/.

Writes:
  - normalized/{model}/{page_id}.json   (gitignored, per-page schema + automated scores)
  - normalized/{model}/_model_summary.json (gitignored, per-model aggregate)
  - scoring_summary.csv (COMMITTED, aggregate numbers only, no verbatim page content)

Does NOT do the faithfulness gate (deliberately manual, see results.md)
and does NOT apply gates or write results.md (that step runs after this script and after the
manual faithfulness check is recorded).

--- Ground-truth scope note, discovered while building this script (record in results.md) ---
The answer-key fidelity instruction names 8 pages: GS003, GS005, GS006, GS007, GS008, GS016,
GS017, GS018. Only GS003, GS005, GS006 are actual answer-key pages (the printed final answer is
literally on the rendered page). GS007, GS008, GS016, GS017, GS018 are QUESTION pages (their
matching answer key lives on a different, unrendered page of the same source PDF, per Issue 71
Topology B), their `final_answer_values` (e.g. "146deg" for GS007) do not and cannot appear in
any model's OCR of the tested page. Checking those 5 pages against `final_answer_values` verbatim
would silently fail every model on a text string that was never present.

What IS checkable, and what this script actually checks for those 5 pages: whether the specific
INPUT numbers printed on the page (e.g. GS007 Q2's "68deg" for angleBCD, which the real answer key
uses as `180-68=112...`) are transcribed correctly and tied to the right question, not merged with
a mark-bracket. Where a page's needed values are themselves inside the diagram image (GS016's pie
fractions, GS018's two angle labels) rather than printed page text, this script records that
explicitly as "not automatable from OCR text" rather than a silent pass or fail.
"""
import csv
import json
import re
import sys
from pathlib import Path

# jiwer is imported lazily inside main()'s CER/WER block, not at module level: build_normalized()
# (the raw->shared-schema conversion) doesn't need it at all, and is reused directly by each OCR
# reader's own run_on_directory() (backend/app/pipeline extraction, Stage B, Issue 81) from
# inside that reader's own isolated venv, which has no reason to carry a jiwer install just to
# import one unrelated function. Found the hard way: ModuleNotFoundError when
# run_mineru25.py's Stage B tried `from normalize_and_score import build_normalized`.

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "raw"
NORM_DIR = ROOT / "normalized"
TRACKER_CSV = ROOT / "gold_set_tracker.csv"
SUMMARY_CSV = ROOT / "scoring_summary.csv"

MODELS = ["got_ocr2", "paddleocr_vl", "mineru25", "qwen3_vl_2b", "deepseek_ocr", "unlimited_ocr"]

DIAGRAM_PAGES = ["GS001", "GS004", "GS007", "GS010", "GS016", "GS017", "GS018"]
TABLE_PAGES = ["GS002", "GS003", "GS005", "GS006"]  # table_contents_structured != none/no

# ---------------------------------------------------------------------------
# Answer-key fidelity ground truth (see scope note above).
# type "grid": full MCQ/answer grid, pairs are (label, expected_value).
# type "grid_subset": same grid mechanics, but only checking a named subset of a big table.
# type "input_values": question page; checking verifiable printed INPUT numbers, not the
#   (unprintable-on-this-page) final answer. `unit` is checked as a soft substring near the value
#   when given. `not_automatable` items are recorded, not scored, with a reason.
# ---------------------------------------------------------------------------
GS003_BOOKLET_A = [4, 2, 2, 3, 1, 2, 3, 3, 4, 3, 2, 2, 4, 4, 3]
FIDELITY_SPECS = {
    "GS003": {
        "type": "grid",
        "pairs": [(str(i), str(v)) for i, v in enumerate(GS003_BOOKLET_A, start=1)],
    },
    "GS005": {
        "type": "grid_subset",
        "pairs": [("28a", "45", None), ("28b", "6", None), ("29", "72", "cm2")],
    },
    "GS006": {
        "type": "grid_subset",
        "pairs": [("30", "61", None), ("31", "229", None), ("32", "33", None)],
    },
    "GS007": {
        "type": "input_values",
        "items": [
            {"q": "2", "label": "angleBCD", "value": "68", "unit": "deg"},
            {"q": "3", "label": "total shelves", "value": "60", "unit": None},
            {"q": "3", "label": "shelves removed", "value": "6", "unit": None},
            {"q": "3", "label": "shelves remaining", "value": "54", "unit": None},
            {"q": "3", "label": "cans per remaining shelf", "value": "40", "unit": None},
        ],
    },
    "GS008": {
        "type": "input_values",
        "items": [
            {"q": "29", "label": "average price", "value": "12", "unit": None},
            {"q": "29", "label": "added book cost", "value": "42", "unit": None},
            {"q": "29", "label": "new average", "value": "18", "unit": None},
            {"q": "30", "label": "red fraction of total", "value": "3/4", "unit": None},
            {"q": "30", "label": "red fraction used", "value": "3/5", "unit": None},
            {"q": "30", "label": "total fraction used for gift", "value": "1/2", "unit": None},
        ],
    },
    "GS016": {
        "type": "input_values",
        "items": [
            {"q": "20", "label": "members who chose white", "value": "7", "unit": None},
        ],
        "not_automatable": [
            {"q": "20", "reason": "Pie-chart sector fractions (Black=1/2, Blue=3/8) are inside "
                                   "the diagram image region, not printed as separate OCR text."},
        ],
    },
    "GS017": {
        "type": "input_values",
        "items": [
            {"q": "17", "label": "angleEOF", "value": "31", "unit": "deg"},
            {"q": "17", "label": "angleKOH", "value": "96", "unit": "deg"},
        ],
    },
    "GS018": {
        "type": "input_values",
        "items": [],
        "not_automatable": [
            {"q": "20", "reason": "Both labelled angles (104deg, 67deg) are inside the diagram "
                                   "figure image, not printed as separate OCR text on this page. "
                                   "No checkable numeric input exists in the page's own text."},
        ],
    },
}


def load_tracker():
    rows = {}
    with open(TRACKER_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows[row["page_id"]] = row
    return rows


# ---------------------------------------------------------------------------
# Per-model raw -> shared schema
# ---------------------------------------------------------------------------

def latex_to_plain(t):
    """Convert LaTeX-ish escapes (seen in got_ocr2/paddleocr_vl/deepseek_ocr output) to plain
    text so grid/value matching isn't fooled by markup: \\frac{3}{4} -> 3/4, strip \\(, \\), etc."""
    t = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"\1/\2", t)
    t = t.replace("\\(", " ").replace("\\)", " ").replace("\\[", " ").replace("\\]", " ")
    t = t.replace("\\", " ")
    return t


def strip_markup_to_tokens(text):
    """Turn HTML/markdown-ish text into an ordered token stream, one 'cell' per token,
    preserving reading order. Used for grid-style (label, value) adjacency matching."""
    t = latex_to_plain(text)
    # HTML tag/cell boundaries become a hard separator so we don't fuse adjacent cell content.
    t = re.sub(r"</?(td|tr|table|div|p|br)[^>]*>", "\x1f", t, flags=re.I)
    t = re.sub(r"<[^>]+>", " ", t)  # drop any remaining tags
    t = t.replace("|", "\x1f")  # markdown pipe-table cell boundary
    t = t.replace("\x1f", " \x1f ")
    tokens = [tok for tok in re.split(r"[\s\x1f]+", t) if tok and tok != "-"]
    return tokens


def normalize_value(s):
    s = s.strip().strip(".,;:")
    s = s.replace("²", "2").replace("³", "3").replace("°", "deg")
    s = s.replace("$", "").replace(" ", "")
    return s.lower()


def check_grid_pairs(text, pairs):
    """pairs: list of (label, value) or (label, value, unit), given in ascending question order
    (true of every grid spec here). Returns list of per-item results.

    Grid labels (1-15, 28a, 30, ...) and grid VALUES (also small integers, e.g. Q1's answer is
    literally "4") share the same character space, so a naive "first token matching the label"
    scan can lock onto an earlier row's *value* that happens to equal this row's *label* (e.g.
    Q1=4 means the token "4" appears well before the real "Q4" label token). Fixed by advancing
    a search cursor monotonically forward as labels are matched in order, instead of rescanning
    from the start of the text for every pair.
    """
    tokens = strip_markup_to_tokens(text)
    norm_tokens = [normalize_value(t) for t in tokens]
    results = []
    cursor = 0
    for pair in pairs:
        if len(pair) == 3:
            label, value, unit = pair
        else:
            label, value = pair
            unit = None
        expected = normalize_value(value + (unit or ""))
        expected_bare = normalize_value(value)
        label_variants = {label.lower(), f"q{label}".lower(), f"q{label}.".lower(), f"{label}.".lower()}
        found_idx = None
        for i in range(cursor, len(norm_tokens)):
            if norm_tokens[i] in label_variants:
                found_idx = i
                break
        if found_idx is None:
            results.append({"label": label, "expected": value, "status": "NOT_FOUND", "found": None})
            continue
        # look ahead a couple tokens for the value (skip a stray repeated label token)
        candidate = None
        for j in range(found_idx + 1, min(found_idx + 3, len(norm_tokens))):
            candidate = norm_tokens[j]
            break
        if candidate is None:
            results.append({"label": label, "expected": value, "status": "NOT_FOUND", "found": None})
        elif candidate == expected_bare or candidate == expected:
            results.append({"label": label, "expected": value, "status": "MATCH", "found": tokens[found_idx + 1] if found_idx + 1 < len(tokens) else None})
        else:
            results.append({"label": label, "expected": value, "status": "MISMATCH", "found": candidate})
        cursor = found_idx + 1  # never re-match an earlier row's tokens for a later label
    return results


def check_input_values(text, items):
    norm_text = normalize_value(re.sub(r"<[^>]+>", " ", latex_to_plain(text)))
    results = []
    for item in items:
        expected = normalize_value(item["value"])
        found = expected in norm_text
        merged = False
        if found:
            idx = norm_text.find(expected)
            before = norm_text[idx - 1] if idx > 0 else ""
            after = norm_text[idx + len(expected)] if idx + len(expected) < len(norm_text) else ""
            merged = before in "[]" or after in "[]"
        results.append({
            "q": item["q"], "label": item["label"], "expected": item["value"],
            "status": ("MATCH" if found and not merged else
                       "MATCH_BUT_MERGED_WITH_BRACKET" if found and merged else "NOT_FOUND"),
        })
    return results


def score_fidelity(page_id, extracted_text):
    spec = FIDELITY_SPECS.get(page_id)
    if spec is None:
        return None
    out = {"type": spec["type"]}
    if spec["type"] in ("grid", "grid_subset"):
        out["items"] = check_grid_pairs(extracted_text, spec["pairs"])
    elif spec["type"] == "input_values":
        out["items"] = check_input_values(extracted_text, spec["items"])
        if "not_automatable" in spec:
            out["not_automatable"] = spec["not_automatable"]
    return out


def has_table_markup(extracted_text, bounding_boxes):
    if "<table" in extracted_text.lower():
        return True
    if any(b.get("type") == "table" for b in bounding_boxes):
        return True
    lines = extracted_text.splitlines()
    for i in range(len(lines) - 1):
        if lines[i].count("|") >= 2 and re.match(r"^[\s|:-]+$", lines[i + 1] or ""):
            return True
    return False


def detect_diagram(model, extracted_text, bounding_boxes):
    if model in ("got_ocr2", "qwen3_vl_2b"):
        return False, "model has no region/bbox output at all (plain-text OCR mode)"
    if model == "paddleocr_vl":
        found = "<img" in extracted_text
        return found, "checked for <img> tag in markdown"
    if model == "mineru25":
        found = any(b.get("type") in ("image", "figure", "picture") for b in bounding_boxes)
        return found, "checked for type in {image,figure,picture} in extracted regions"
    if model in ("deepseek_ocr", "unlimited_ocr"):
        found = bool(re.search(r"!\[[^\]]*\]\([^)]*\)", extracted_text))
        return found, "checked for markdown image ref ![...](...))"
    return False, "unknown model"


def gs002_row_pairing_check(extracted_text):
    """GS002-specific structural check for the 4-row bus-times table (rows: 8.20/9.05, 8.40/9.25,
    9.15/10.00, 9.45/10.30). Distinguishes a correctly row-paired transcription from a flattened
    one (all 4 'leaves' values emitted as one block, then all 4 'arrives' values as a second
    block) by ORDER, not raw character distance: a naive nearest-neighbour distance check doesn't
    actually work here, because even under full flattening "9.45" still sits textually closer to
    "10.00" than to any earlier arrives value simply by block order, not genuine pairing.
    The real tell: in a correctly paired/interleaved transcription, row 4's leaves value ("9.45")
    comes AFTER row 3's arrives value ("10.00"), since 10.00 immediately follows its own row's
    9.15. Under flattening, the entire leaves block (ending in 9.45) precedes the entire arrives
    block (starting with 9.05), so 9.45 comes BEFORE 10.00."""
    t = extracted_text.lower()
    i_945 = t.find("9.45")
    i_1000 = t.find("10.00")
    if i_945 == -1 or i_1000 == -1:
        return {"checked": False, "reason": "9.45 or 10.00 not found in text"}
    return {"checked": True, "row_pairing_preserved": i_945 > i_1000,
            "idx_9_45_row4_leaves": i_945, "idx_10_00_row3_arrives": i_1000}


def build_normalized(model, page_id, raw):
    """Map one model's raw JSON for one page into the shared schema."""
    error = raw.get("error")
    bounding_boxes = []
    structure = {}
    if model == "got_ocr2":
        extracted_text = raw.get("ocr_type_ocr") or ""
        structure["format_mode_text_len"] = len(raw.get("ocr_type_format") or "")
        structure["ocr_mode_text_len"] = len(extracted_text)
    elif model == "paddleocr_vl":
        extracted_text = raw.get("markdown") or ""
    elif model == "mineru25":
        extracted = raw.get("extracted") or []
        # "content" included below (segment_questions() region-based redesign), the
        # bake-off's own scoring (has_table_markup/detect_diagram, above) only ever reads "type",
        # so this is purely additive there. Without content, a region-based question segmenter
        # downstream can see WHERE a region is but not WHAT text is in it, which defeats the
        # entire point of segmenting on real regions instead of flat-text line matching.
        bounding_boxes = [
            {"type": e.get("type"), "bbox": e.get("bbox"), "content": e.get("content")}
            for e in extracted
        ]
        extracted_text = "\n".join((e.get("content") or "") for e in extracted)
    elif model == "qwen3_vl_2b":
        extracted_text = raw.get("transcription") or ""
    elif model in ("deepseek_ocr", "unlimited_ocr"):
        extracted_text = raw.get("markdown") or ""
    else:
        raise ValueError(f"unknown model {model}")

    diagram_detected, diagram_method = detect_diagram(model, extracted_text, bounding_boxes)
    table_markup = has_table_markup(extracted_text, bounding_boxes)

    field_confidence = None  # none of the 6 candidates expose a per-field confidence score

    schema = {
        "page_id": page_id,
        "model": model,
        "extracted_text": extracted_text,
        "structure": {**structure, "has_table_markup": table_markup},
        "bounding_boxes": bounding_boxes,
        "field_confidence": field_confidence,
        "error": error,
    }

    scores = {}
    if not error and extracted_text:
        scores["diagram_detected"] = diagram_detected
        scores["diagram_detection_method"] = diagram_method
        if page_id == "GS002":
            scores["table_row_pairing"] = gs002_row_pairing_check(extracted_text)
        fidelity = score_fidelity(page_id, extracted_text)
        if fidelity is not None:
            scores["answer_key_fidelity"] = fidelity

    return schema, scores


def normalize_for_cer(text):
    t = re.sub(r"<[^>]+>", " ", text)
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", t)
    t = t.replace("\\(", " ").replace("\\)", " ").replace("\\[", " ").replace("\\]", " ")
    t = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"\1/\2", t)
    t = t.replace("\\", " ")
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t


def main():
    import jiwer  # lazy, only main()'s CER/WER step needs it, see module-level note above

    tracker = load_tracker()
    NORM_DIR.mkdir(exist_ok=True)

    summary_rows = []
    for model in MODELS:
        model_raw_dir = RAW_DIR / model
        if not model_raw_dir.exists():
            print(f"[skip] no raw/ for {model}", file=sys.stderr)
            continue
        run_summary_path = model_raw_dir / "_run_summary.json"
        run_summary = json.loads(run_summary_path.read_text(encoding="utf-8")) if run_summary_path.exists() else {}

        model_norm_dir = NORM_DIR / model
        model_norm_dir.mkdir(parents=True, exist_ok=True)

        cers, wers = [], []
        table_hits, table_total = 0, 0
        diagram_hits, diagram_total = 0, 0
        fidelity_match, fidelity_mismatch, fidelity_not_found, fidelity_merged = 0, 0, 0, 0
        pages_with_errors = []

        for page_id, gold in tracker.items():
            raw_path = model_raw_dir / f"{page_id}.json"
            if not raw_path.exists():
                continue
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            schema, scores = build_normalized(model, page_id, raw)

            if schema["error"]:
                pages_with_errors.append(page_id)

            # CER/WER against question_text_clean. GS013/GS014/GS015's question_text_clean is a
            # literal "N/A - this file is the worked-solutions document only..." placeholder (the
            # matching question paper wasn't sourced this round), NOT a real transcript. Scoring
            # CER/WER against that placeholder produces meaningless, hugely inflated numbers (the
            # actual worked-solution output vs an unrelated one-line disclaimer), identically for
            # every model, so those 3 pages are excluded from CER/WER here.
            gold_text = gold.get("question_text_clean", "")
            if gold_text and gold_text.strip() and not gold_text.startswith("N/A") and not schema["error"]:
                ref = normalize_for_cer(gold_text)
                hyp = normalize_for_cer(schema["extracted_text"])
                if ref and hyp:
                    try:
                        cer = jiwer.cer(ref, hyp)
                        wer = jiwer.wer(ref, hyp)
                        scores["cer"] = cer
                        scores["wer"] = wer
                        cers.append(cer)
                        wers.append(wer)
                    except Exception as e:
                        scores["cer_error"] = str(e)

            if page_id in TABLE_PAGES and not schema["error"]:
                table_total += 1
                if schema["structure"]["has_table_markup"]:
                    table_hits += 1

            if page_id in DIAGRAM_PAGES and not schema["error"]:
                diagram_total += 1
                if scores.get("diagram_detected"):
                    diagram_hits += 1

            if "answer_key_fidelity" in scores:
                for item in scores["answer_key_fidelity"]["items"]:
                    status = item["status"]
                    if status == "MATCH":
                        fidelity_match += 1
                    elif status == "MISMATCH":
                        fidelity_mismatch += 1
                    elif status == "MATCH_BUT_MERGED_WITH_BRACKET":
                        fidelity_merged += 1
                    else:
                        fidelity_not_found += 1

            out = {"schema": schema, "scores": scores}
            (model_norm_dir / f"{page_id}.json").write_text(
                json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        model_summary = {
            "model": model,
            "hf_repo": run_summary.get("hf_repo"),
            "peak_vram_gb": run_summary.get("peak_vram_gb"),
            "fits_8gb_gate": run_summary.get("fits_8gb_gate"),
            "pages_scored": len(cers),
            "pages_with_errors": pages_with_errors,
            "mean_cer": sum(cers) / len(cers) if cers else None,
            "mean_wer": sum(wers) / len(wers) if wers else None,
            "table_markup_pages": f"{table_hits}/{table_total}",
            "diagram_detected_pages": f"{diagram_hits}/{diagram_total}",
            "fidelity_match": fidelity_match,
            "fidelity_mismatch": fidelity_mismatch,
            "fidelity_not_found": fidelity_not_found,
            "fidelity_merged_with_bracket": fidelity_merged,
        }
        (model_norm_dir / "_model_summary.json").write_text(
            json.dumps(model_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary_rows.append(model_summary)
        print(f"[done] {model}: {model_summary}")

    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["model", "hf_repo", "peak_vram_gb", "fits_8gb_gate", "pages_scored",
                      "pages_with_errors", "mean_cer", "mean_wer", "table_markup_pages",
                      "diagram_detected_pages", "fidelity_match", "fidelity_mismatch",
                      "fidelity_not_found", "fidelity_merged_with_bracket"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in summary_rows:
            row = dict(row)
            row["pages_with_errors"] = ";".join(row["pages_with_errors"])
            w.writerow(row)

    print(f"\nWrote {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
