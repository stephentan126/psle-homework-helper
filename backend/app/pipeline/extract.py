"""
Stage C of the extraction pipeline: merge both OCR readers' saved output into question rows.

Converts the purchased papers (`data/School papers/`, `data/Yearly/`) into structured `questions`
rows, using the two OCR readers chosen in the OCR bake-off: MinerU2.5 as primary and DeepSeek-OCR
as secondary (see `evaluation/model_selection/ocr_extraction/results.md`).

The pipeline has three stages rather than one loop, because the two readers cannot share a
process: their `transformers` pins conflict with each other and with the backend, and their
combined peak VRAM (8.54GB) is over the 8GB budget (Issue 81).
  - Stage A (`render_pages.py`): pure CPU. Renders every page of every discovered PDF to
    `data/extracted/pages/{pdf_stem}/{page_index}.png`.
  - Stage B (`evaluation/model_selection/ocr_extraction/runners/run_mineru25.py` and
    `run_deepseek_ocr.py`): each reader runs separately in its own venv. It loads once, reads
    every rendered page and saves one common-schema JSON file per page to
    `data/extracted/raw/{reader_name}/{pdf_stem}/{page_index}.json`.
  - Stage C (this module): pure CPU, no model loaded. Reads Stage B's saved files and:
      - normalises glyphs in both readers' output before any comparison (step 3a);
      - classifies each page, splits question pages into questions and compares the two
        readers question by question (segment_and_compare_page());
      - compares the readers on answer-key pages (merge_page_with_agreement(), step 3b) and
        parses answer-key grids (parse_answer_key_grid());
      - links each answer to its question, whether the answers are in a separate file
        (Topology A) or in the same file (Topology B, locate_embedded_answer_section());
      - records provenance (source_page_index, source_page_label, answer_source_file,
        answer_source_page_index).

values_equivalent() is a SymPy-based numeric and semantic equivalence check, so that "1/4" and
"0.25" count as the same answer.

Files are processed by parallel workers (DEFAULT_WORKERS), and a failure in one file does not
stop the run (Issues 62 and 71). Workers never hold a database session. Each writes one JSON
checkpoint per file, and write_extracted_data_to_db() then writes every checkpoint to the
database in a single, non-parallel pass.

Human verification, deduplication (superseded_by) and the held-out freeze are separate later
steps. This module writes verification_status, verified_at and superseded_by only at their
schema defaults (unverified, None, None).
"""

from __future__ import annotations

import argparse
import difflib
import html
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Literal, Optional, Sequence

import sympy
from sympy.parsing.sympy_parser import parse_expr

# ---------------------------------------------------------------------------------------------
# Real DB session / ORM models (Issue 102), wired for real,
# replacing the placeholder try/except-ImportError block this used to be. Section 10.1 defines
# the schema (partially stale, see backend/app/models/question.py's own docstring for the real,
# confirmed field-list discrepancy); backend/app/models/ and backend/app/db/session.py now hold
# the actual SQLAlchemy classes/session wiring these names refer to.
#
# `app.X`, not `backend.app.X` (Issue 102): confirmed by direct testing that this backend's
# real installed top-level package name is `app` (backend/.venv's editable install maps
# `app` -> `backend/app/`), and that `app.X` imports resolve from any working directory, unlike
# `backend.app.X` (this file's OLD placeholder convention), which only resolved when the process
# happened to be launched with the repo root as its CWD. Kept as a real, hard import (no
# try/except fallback), these modules are a genuine, always-required part of this same
# installed package now, not an optional layer that might not exist yet.
# ---------------------------------------------------------------------------------------------
from app.db.session import get_session, init_db
from app.models import OCRExtractionRecord, Question


logger = logging.getLogger("pipeline.extract")

# 24 physical cores measured on this dev machine via `wmic cpu get NumberOfCores`,
# minus 2 reserved for the OS, same reasoning and same number as render_pages.py's
# DEFAULT_WORKERS (Stage C is pure CPU work too, no GPU, safe to parallelize the same way).
DEFAULT_WORKERS = 22

AgreementStatus = Literal["agree", "disagree", "single_reader_only", "both_failed"]


# =================================================================================================
# Data model, shared shapes for reader output and extracted questions
# =================================================================================================

@dataclass
class PageImage:
    """A single rendered page, ready to hand to a reader. Real rendering is implemented below
    via PyMuPDF (fitz) in render_pdf_page(), this is a standard, deterministic utility, not a
    guessed model signature, so it's implemented rather than stubbed."""
    source_pdf: Path
    page_index: int  # 0-indexed, matches PyMuPDF's own indexing
    image_bytes: bytes
    dpi: int = 300


@dataclass
class RawReaderOutput:
    """Common schema every runner already converts to (docs/MODEL_SELECTION.md Section 4)."""
    extracted_text: str
    structure: Optional[dict] = None  # headings/tables/reading order, if the model gives it
    bounding_boxes: Optional[list] = None
    field_confidence: Optional[dict] = None


@dataclass
class QuestionSpan:
    """One question's location and raw text on a rendered page. Produced by segment_questions()
    (region-based, the development log)."""
    question_number: str  # e.g. "7", "12(a)", paper/booklet/sub-part tagging (Issue 2/3)
    raw_text: str
    page_index: int
    page_label: Optional[str] = None  # printed page code, e.g. "BP-527", if visible
    has_diagram: bool = False  # real, position-based signal (Issue
                                 # 102 part 2), region-path only (_segment_from_regions), a
                                 # MinerU2.5 "image"-type region attributed to this span by its
                                 # position in reading order relative to this span's own content
                                 # regions. Confirmed against real data before building: 77/77
                                 # real "image" regions across the validation subset's question
                                 # pages were attributable to exactly one question this way (73
                                 # falling within an already-open span or a pending bare-number
                                 # label, 4 more as a genuine leading stimulus image resolved
                                 # once the next real question opens), zero ambiguous.
                                 # Stays False on the flat-text fallback path
                                 # (_segment_from_flat_text, DeepSeek-OCR), that reader's raw
                                 # output carries no region-type signal at all (bounding_boxes is
                                 # always empty there), so there is no real per-question
                                 # diagram signal on that path; not guessed.
    flagged: Optional[str] = None  # short reason string when this span needs human review rather
                                     # than being trusted silently, e.g. "low_confidence_flat_
                                     # text_recovery" (Bug 1 fix) or "ambiguous_unnumbered_span_
                                     # number_collision" (Bug 3 fix), both the development log
                                     # follow-up. None for the normal, unflagged case.


@dataclass
class ExtractedQuestion:
    """The final shape written to `questions` (Section 10.1) for one question."""
    source_paper: Path
    source_page_index: int
    source_page_label: Optional[str]
    answer_source_file: Path
    answer_source_page_index: int
    question_number: str
    question_text: str
    answer_value: Optional[str]
    worked_solution_text: Optional[str]
    has_diagram: bool
    ocr_confidence: Optional[Literal["high", "low"]]
    school_name: Optional[str]
    # Schema defaults, steps 5/6 update these later, not this file:
    verification_status: str = "unverified"
    school_name_verified: bool = False  # Issue 93 : True only when
                                          # school_name came from a canonical entry a human has
                                          # confirmed against real source PDFs (tools/
                                          # school_name_canonical.csv's `verified` column), not
                                          # just a filled-in guess. Defaults False so any record
                                          # built without going through normalized_school_name()
                                          # (e.g. a future direct construction) fails closed rather
                                          # than silently claiming verification it never had.
    verified_at: Optional[datetime] = None
    superseded_by: Optional[int] = None
    extraction_flag: Optional[str] = None  # carried over from QuestionSpan.flagged (development log
                                             # follow-up), surfaces a real, specific
                                             # extraction-time uncertainty a human should review,
                                             # distinct from verification_status (step 5's later
                                             # human-verification workflow state). None normally.
    question_section: Optional[str] = None  # real schema addition (the development log
                                               # follow-up, Issue 3's real fix), which real
                                               # section of the paper this question came from
                                               # ("BOOKLET A"/"BOOKLET B"/"PAPER 2"), extracted from
                                               # the question's own page (see
                                               # _extract_question_page_section()). None on a real
                                               # page that carries no detectable section
                                               # signal, not guessed, not defaulted to a paper's
                                               # first/most-common section. This is what makes
                                               # section-scoped answer matching possible at all
                                               # (_lookup_answer_for_span() below); before this
                                               # field existed, a question's own section was
                                               # unknowable downstream of segment_questions().


@dataclass
class QuestionWithOcrRecord:
    """
    One ExtractedQuestion paired with its own ocr_extraction_record (the development log,
    resolving the question_id linking gap flagged after commit 214aa39). Built together, as a
    single unit, rather than two separate lists matched afterward by a provisional string id, the
    real FK relationship (questions.id -> ocr_extraction_records.question_id) can only be
    established once a real Question row is inserted and flushed, which write_extracted_data_to_db()
    does per-pair; keeping question and ocr_record paired from the moment they're built is what
    makes that possible without re-matching them by a fragile string key later.

    Granularity (Issue 98, the development log): ocr_record is per-QUESTION,
    produced by segment_and_compare_page(), which segments each reader's own output independently
    and compares matched question spans directly, not a page-level record reused identically
    across every question on a page. That page-level reuse was this dataclass's original design
    (through Issue 92) and was deliberately kept on the theory that true per-question
    granularity would require re-running the readers per segmented span; a fresh, real Stage
    A->B->C run (Issue 98) confirmed that theory was the actual root cause of a real bug, whole
    -page comparison via values_equivalent() produced agreement_status="disagree" on 82 of 82 real
    questions across both validation files, and that re-running the readers was never actually
    required: segment_questions() is pure CPU parsing of already-saved Stage B output, so it can
    run twice per page (once per reader) for free. Fixed for real, not layered on top of the old
    behavior.

    ocr_record is None only in the both_failed/no-Stage-B-output case, which already means no
    ExtractedQuestion exists for that page either (segment_and_compare_page() returns ([], None)
    there, and callers skip the page entirely), kept Optional to match ocr_record's own Optional
    type rather than assert something this dataclass alone can't guarantee.
    """
    question: ExtractedQuestion
    ocr_record: Optional[dict]


@dataclass
class ExtractionRunResult:
    """Per-file result, used by the main loop to decide what to save/log. One entry per
    (question, ocr_record) pair, see QuestionWithOcrRecord."""
    pairs: list[QuestionWithOcrRecord] = field(default_factory=list)


# =================================================================================================
# Step 3a, glyph normalization (must run BEFORE comparison, not just before saving)
# =================================================================================================

# Known OCR-prone glyphs, confirmed cross-candidate in the Job A bake-off (results.md).
# The cent symbol was misread by every candidate tested, always as some other single glyph, with
# the underlying number unaffected. Extend this map as new systematic misreads are found , 
# real corpus is ~140 PDFs, more of these are likely to surface than the 18-page gold set caught.
_GLYPH_NORMALIZATION_MAP = {
    # cent symbol: models have been observed substituting a variety of single glyphs for ¢.
    # Confirm the actual substitution characters seen against the real gold-set findings in
    # results.md before trusting this list, placeholders below cover the commonly-reported
    # OCR confusions for ¢ (c, C with cedilla, superscript c) pending that confirmation.
    "\u00a2": "\u00a2",  # ¢ itself, identity, in case a reader gets it right
    "\u00e7": "\u00a2",  # ç, a common OCR substitution for ¢, confirm against real findings
    "\u1d9c": "\u00a2",  # superscript c, another plausible substitution, confirm before trusting
    # Degree and squared-cm, flagged in results.md as similar OCR-prone symbols:
    "\u00b0": "\u00b0",  # ° identity
    "\u00b2": "\u00b2",  # ² identity (used in cm²)
}


def normalize_glyphs(text: str) -> str:
    """
    Step 3a. Deterministic normalization pass over known OCR-prone glyphs. Must run on both
    readers' raw output BEFORE compare_readers() is called, comparing unnormalized output would
    register this exact cosmetic noise as a real disagreement .

    NOTE: the specific substitution characters in _GLYPH_NORMALIZATION_MAP above are inferred,
    not confirmed against the real per-candidate raw output in results.md. Before running this
    against real data, check results.md's actual recorded misreads for the cent-symbol defect and
    correct this map to match exactly what each candidate really produced, rather than trusting
    the placeholders here.
    """
    normalized = text
    for bad_glyph, good_glyph in _GLYPH_NORMALIZATION_MAP.items():
        normalized = normalized.replace(bad_glyph, good_glyph)
    return normalized


# =================================================================================================
# values_equivalent, real SymPy-based numeric/semantic equivalence check
# =================================================================================================

def _strip_currency(s: str) -> str:
    """
    Strips a single leading or trailing currency symbol ($/¢), SymPy can't parse "$0.50"
    directly, and the actual currency-unit check (is this a $ answer vs a ¢ answer) is a separate
    structural concern, not this function's job. Promoted to module level (the development log
    follow-up, answer-key-grid parsing) so parse_answer_key_grid()'s cell cleanup can reuse it
    instead of re-deriving the same logic, a parallel copy could drift out of sync with this one.
    """
    return re.sub(r"^[$¢]|[$¢]$", "", s).strip()

def values_equivalent(a: Optional[str], b: Optional[str]) -> bool:
    """
    Numeric/semantic equivalence, not string equality : two correct
    answers can look textually different (1/4 vs 0.25, 50 cents vs $0.50) without either reader
    being wrong. This is the same check Section 8's safety gate is planned to use, when Phase 3
    builds that gate, promote this function out to a shared module instead of duplicating it.

    Returns False (not equivalent) on anything that fails to parse, rather than raising, a
    parse failure here should count as a mismatch for compare_readers() to flag, not crash the
    extraction run.
    """
    if a is None or b is None:
        return a == b  # both None => equivalent; one None => not

    a_clean, b_clean = a.strip(), b.strip()
    if a_clean == b_clean:
        return True

    try:
        expr_a = parse_expr(_strip_currency(a_clean), evaluate=True)
        expr_b = parse_expr(_strip_currency(b_clean), evaluate=True)
        return bool(sympy.simplify(expr_a - expr_b) == 0)
    except Exception as e:  # noqa: BLE001, intentionally broad, matches this function's own
        # documented contract ("Returns False on anything that fails to parse, rather than
        # raising"). The original narrower tuple (SympifyError, TypeError, ValueError,
        # SyntaxError) missed a real case found running Stage C against real corpus pages
        # compare_readers currently passes whole-page extracted_text through this
        # function as a stand-in for "the answer value" (a known limitation, see compare_readers'
        # own docstring), on a real page of prose/instructions text, sympy.parse_expr can raise
        # tokenize.TokenError ("unterminated string literal", "unexpected character after line
        # continuation character") for input that looks nothing like a parseable expression.
        # TokenError isn't a SyntaxError subclass, so it was escaping this except clause entirely,
        # propagating up through compare_readers -> merge_page_with_agreement -> the per-file
        # try/except in run_extraction, and failing the WHOLE FILE instead of just registering
        # this one page as "not equivalent" the way a parse failure is supposed to. Broadened
        # instead of enumerating every exception type sympy's parser can raise for malformed
        # input, the intent was already "anything that fails to parse", the implementation just
        # hadn't caught up to a real example yet.
        logger.debug(
            "values_equivalent: failed to parse %r vs %r as expressions (%s: %s)",
            a, b, type(e).__name__, e,
        )
        return False


# =================================================================================================
# compare_readers, checks BOTH answer-value equivalence AND structural fabrication risk
# =================================================================================================

def _structure_looks_fabricated(primary: RawReaderOutput, secondary: RawReaderOutput) -> bool:
    """
    Flags a structural mismatch even when the final answer value matches (development log
), the DeepSeek-OCR residual risk flagged in results.md was a table-schema
    fabrication tendency found OUTSIDE the formal answer-key-grid pages that scored clean, so a
    matching number alone isn't sufficient evidence of agreement on grid/table-shaped content.

    PLACEHOLDER HEURISTIC, real bounding-box/table structure isn't available in this chat to
    build a confirmed check against. Current version: flag when one reader reports table
    structure and the other reports none at all for the same page, since that's the cheapest
    real signal available from RawReaderOutput.structure without guessing either model's real
    JSON shape further. Replace with a stronger check once the real structure field format from
    both runners is confirmed.
    """
    primary_has_table = bool(primary.structure and primary.structure.get("tables"))
    secondary_has_table = bool(secondary.structure and secondary.structure.get("tables"))
    return primary_has_table != secondary_has_table


def compare_readers(
    primary_norm: RawReaderOutput,
    secondary_norm: RawReaderOutput,
) -> tuple[AgreementStatus, Optional[str], Optional[str]]:
    """Compare the two readers' normalised output for one page (step 3b).

    Callers must run normalize_glyphs() on both inputs first. Comparing raw output would report
    glyph differences as disagreements.

    Two checks run, in order:
      (a) Answer-value equivalence, using values_equivalent() (numeric and semantic, not string
          equality).
      (b) A structural check. If the values match but one reader's table or grid markup looks
          fabricated compared with the other's, the result is "disagree", not "agree".

    A reader has failed only when it returned no text at all. Wrong text is not a failure: it is a
    "disagree", which is the signal the two-reader design exists to produce.

    The disagreement type is returned as a separate value, set at the branch that decides it,
    rather than parsed from the human-readable detail string afterwards (Issue 212). Stored detail
    strings from earlier versions of this function use different wording, so parsing them would be
    fragile.

    Args:
        primary_norm: normalised MinerU2.5 output, or None if there is none.
        secondary_norm: normalised DeepSeek-OCR output, or None if there is none.

    Returns:
        (agreement_status, disagreement_detail, disagreement_type). agreement_status is "agree",
        "disagree", "single_reader_only" or "both_failed". disagreement_type is "value_mismatch",
        "structural_mismatch" or None. It is None when no comparison ran or nothing disagreed.
        "both" is never returned, because a value mismatch returns before structure is checked.
    """
    primary_empty = not primary_norm.extracted_text.strip() if primary_norm else True
    secondary_empty = not secondary_norm.extracted_text.strip() if secondary_norm else True

    if primary_empty and secondary_empty:
        return "both_failed", "both readers returned empty output", None
    if primary_empty:
        return (
            "single_reader_only", "primary (MinerU2.5) returned empty output; using secondary",
            None,
        )
    if secondary_empty:
        return (
            "single_reader_only", "secondary (DeepSeek-OCR) returned empty output; using primary",
            None,
        )

    # Both readers produced something. Check answer-value equivalence first.
    # The whole extracted_text stands in for the answer value here. Question pages are compared
    # question by question in segment_and_compare_page() instead; this whole-page comparison serves
    # answer-key pages through merge_page_with_agreement().
    if values_equivalent(primary_norm.extracted_text, secondary_norm.extracted_text):
        if _structure_looks_fabricated(primary_norm, secondary_norm):
            return "disagree", (
                "answer values equivalent, but table/grid structure mismatch between readers "
                "(one reports a table, the other does not) — flagged per the DeepSeek-OCR "
                "residual fabrication risk, not silently counted as agreement"
            ), "structural_mismatch"
        return "agree", None, None

    return "disagree", (
        f"answer values not equivalent: primary={primary_norm.extracted_text!r} "
        f"secondary={secondary_norm.extracted_text!r}"
    ), "value_mismatch"


# =================================================================================================
# Step 3b, merge_page_with_agreement (Stage C; design specification Section 15 step 3b, matching
# its pseudocode: load_raw_output() replaces a direct reader call, everything downstream , 
# both_failed skip, single_reader_only fallback fix, is unchanged from before Issue 81)
# =================================================================================================

def load_raw_output(
    reader_name: str, pdf_stem: str, page_index: int, raw_dir: Path
) -> Optional[RawReaderOutput]:
    """
    Loads one page's Stage B output (common-schema JSON, already normalized to extracted_text/
    structure/bounding_boxes/field_confidence by build_normalized() inside that reader's own
    run_on_directory()) for the given reader.

    Returns None, meaning "no usable output from this reader for this page", in TWO distinct
    real cases, both treated the same way by merge_page_with_agreement() below: the file doesn't
    exist at all (Stage B hasn't been run yet, or was run with a --limit that didn't cover this
    page), or the file exists but Stage B itself recorded an error for this page (the reader
    threw during inference). Neither case is "the reader ran and returned empty text", that
    third, different case is already handled inside compare_readers() via its own empty-string
    check, not here.
    """
    path = raw_dir / reader_name / pdf_stem / f"{page_index}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("error"):
        return None  # Stage B recorded a real error for this page, no usable output
    return RawReaderOutput(
        extracted_text=data.get("extracted_text") or "",
        structure=data.get("structure"),
        bounding_boxes=data.get("bounding_boxes"),
        field_confidence=data.get("field_confidence"),
    )


# Real, second, DeepSeek-OCR-derived has_diagram secondary signal (the development log,
# Issue 233), real, confirmed gap found in the Issue 222 corpus-wide reconciliation:
# `has_diagram` was set EXCLUSIVELY from MinerU2.5's own structured "image"-type regions (see
# QuestionSpan's own has_diagram field, set inside segment_questions() below); DeepSeek-OCR's own
# `bounding_boxes` field is confirmed always empty (`[]`) for every real corpus page checked, so it
# structurally could never contribute via that same mechanism, but DeepSeek-OCR DOES label real
# diagrams internally, in its own flat-text output, as a real `![](images/N.jpg)` markdown image
# reference (confirmed directly against real corpus JSON, not assumed).
#
# REAL, EVIDENCE-GROUNDED DESIGN, checked against real data BEFORE implementing, not assumed
# correct on the "OR combination" name alone: a direct, real, corpus-wide comparison (5,425 real
# questions, 3,128 real (source_paper, page) pairs) found 48 real pages where MinerU2.5 said
# has_diagram=False for every question on the page but DeepSeek-OCR's own raw text for that same
# page contained a real image marker. A random, seeded sample of 10 of those 48 real pages was
# manually inspected against their own real extracted question text (not just the raw counts): 9
# of 10 are confirmed, genuine real diagrams MinerU2.5 missed entirely (e.g. "In the figure below,
# find \(\angle y\)" immediately followed by a real image marker; "Tom built solid A ... as shown"
# with two real image markers), real, direct evidence DeepSeek-OCR's own marker is trustworthy
# here, not a guess. `mineru_only` (155 real pages, the reverse direction) is NOT investigated
# here: under an OR-combination design a MinerU2.5 True is never revisited, so no real design
# decision depends on that direction, a deliberate scope limit.
#
# Attribution limit: DeepSeek-OCR's
# own image marker is PAGE-level only (its flat-text output has no structured region data to
# attribute a marker to one specific question by position, unlike MinerU2.5's own signal), of the
# 48 real disagreement pages, 32 have exactly one real question (safe, unambiguous attribution) but
# 16 have 2-5 real questions sharing the page, where a page-level marker cannot be safely assigned
# to one specific question without guessing (a real risk of introducing a NEW false-positive class
# this fix does not currently have). This signal is therefore deliberately SCOPED to real,
# single-question pages only, see this function's own real caller in extract_topology_a/b. The 16
# real multi-question pages are a real, documented, NOT-fixed residual (docs/DEVELOPMENT_LOG.md),
# not silently dropped.
#
# REAL, SECOND FALSE-POSITIVE RISK FOUND, checking beyond the initial 10-item sample, not stopping
# at the first reassuring result: a full manual classification of all 32 real single-question
# disagreement pages found a real, concrete, RECURRING false-positive shape, 5 of the 32 (e.g.
# "At a fruit stall, mangoes were sold at 3 for $10...", "Three teachers accompanied a group of 38
# pupils...") are plain arithmetic/algebra word problems with NO real diagram dependency at all;
# their real DeepSeek-OCR image marker sits immediately after the answer space, at the very end of
# the page's own text, in every one of the 5 confirmed cases, real, direct evidence of a trailing
# footer/margin graphic (or a non-diagram visual), not the question's own figure. A
# blind, page-level-only OR would have wrongly flagged all 5. REAL FIX: additionally require the
# real QUESTION'S OWN TEXT to contain either a real diagram-referencing keyword/phrase OR a real
# named geometric point/shape label (`_DIAGRAM_REFERENCE_LABEL_PATTERN` below, the same real
# 2-6-letter ALL-CAPS pattern already validated in `question_matching.py`'s own
# `_NAMED_LABEL_RE`, Issue 199(a), reused as a design precedent, not imported directly, to
# avoid a new dependency from this earlier-Slice extraction module onto a later-Slice matching
# module). Re-checked against all 32 real cases with this combined gate: all 5 confirmed false
# positives are correctly excluded; 20 of 32 are kept (every one independently confirmed by direct
# inspection to have a real, genuine diagram dependency, e.g. "Find the area of triangle XYZ" , 
# `XYZ` alone is enough via the label pattern with no keyword present); the remaining 12 (data
# tables and further word problems with no real keyword/label match) are conservatively left
# UNCHANGED rather than guessed, a real, deliberate, safety-biased trade of some recall for real,
# checked precision, the same "false positive worse than false negative" asymmetry this project
# applies everywhere else (Issues 58/180/186/231).
_DEEPSEEK_IMAGE_MARKER_PATTERN = re.compile(r"!\[\]\(images/\d+\.jpg\)")
_DIAGRAM_REFERENCE_KEYWORD_PATTERN = re.compile(
    r"figure|diagram|graph|grid|pattern|shown below|shown above|as shown|net of", re.IGNORECASE,
)
_DIAGRAM_REFERENCE_LABEL_PATTERN = re.compile(r"\b[A-Z]{2,6}\b")  # e.g. "ABCD", "XYZ", "PQRS"


def _deepseek_ocr_page_has_diagram_marker(pdf_stem: str, page_index: int, raw_dir: Path) -> bool:
    """Real, standalone check for DeepSeek-OCR's own internal diagram marker on one real page , 
    see the real module-level comment immediately above this function for the full real evidence
    and design reasoning. Returns `False`, never raises, when DeepSeek-OCR has no real output for
    this page at all (Stage B never ran for it, or errored), the same honest "no signal" contract
    `load_raw_output()` itself already uses, not a guess either way."""
    secondary_raw = load_raw_output("deepseek_ocr", pdf_stem, page_index, raw_dir)
    if secondary_raw is None:
        return False
    return bool(_DEEPSEEK_IMAGE_MARKER_PATTERN.search(secondary_raw.extracted_text))


def _question_text_references_a_diagram(question_text: str) -> bool:
    """Real, second-stage gate (Issue 233), see the real module-level
    comment above `_DEEPSEEK_IMAGE_MARKER_PATTERN` for the full real false-positive evidence this
    exists to close. True when the real question's OWN text contains a real diagram-referencing
    keyword/phrase OR a real named geometric point/shape label, either is independently
    sufficient (confirmed real cases exist of each without the other, e.g. "Find the area of
    triangle XYZ" has a label but no keyword; "Study the pattern below" has a keyword but no
    label)."""
    return bool(
        _DIAGRAM_REFERENCE_KEYWORD_PATTERN.search(question_text)
        or _DIAGRAM_REFERENCE_LABEL_PATTERN.search(question_text)
    )


def _apply_deepseek_ocr_diagram_secondary_signal(
    span_results: "list[tuple[QuestionSpan, Literal['high', 'low'], dict]]",
    pdf_stem: str, page_index: int, raw_dir: Path,
) -> None:
    """Real, standalone, testable wiring for the secondary has_diagram signal (development log
 Issue 233), shared by extract_topology_a/b's own otherwise-identical per-page
    blocks, rather than duplicated inline in both. Mutates `span_results[0][0].has_diagram` IN
    PLACE (real, deliberate: `QuestionSpan` is a real, mutable dataclass already relied on
    elsewhere in this file for this exact field, e.g. line ~1110's own recovered-span merge) when,
    and only when, ALL THREE real conditions hold: this is a real, unambiguous SINGLE-question
    page; MinerU2.5's own derived value is still `False`; DeepSeek-OCR's own marker is present;
    AND the question's own text passes `_question_text_references_a_diagram()` (the real,
    evidence-grounded false-positive gate, see that function's and `_DEEPSEEK_IMAGE_MARKER_
    PATTERN`'s own module-level comment for the full real investigation). A genuine, silent no-op
    for every other real case (a multi-question page, an already-True MinerU2.5 signal, no
    DeepSeek-OCR marker, or a question with no real diagram-referencing keyword/label), this
    function never touches `span_results` when any condition fails, confirmed directly by its own
    unit tests, not just by reading the guards."""
    if len(span_results) != 1 or span_results[0][0].has_diagram:
        return
    span = span_results[0][0]
    if _deepseek_ocr_page_has_diagram_marker(pdf_stem, page_index, raw_dir) and \
            _question_text_references_a_diagram(span.raw_text):
        span.has_diagram = True


def merge_page_with_agreement(
    pdf_stem: str,
    page_index: int,
    question_id: object,  # the question's identifier for logging/DB association, not yet a
                           # real questions.id since the row hasn't been inserted at this point
                           # in the flow. To do: confirm ordering against the real ORM
                           # save pattern (insert Question first to get an id, or generate one
                           # up front) once Phase 2 step 1 exists.
    raw_dir: Path,
) -> tuple[Optional[RawReaderOutput], Optional[Literal["high", "low"]], Optional[dict]]:
    """
    Stage C's core per-page merge. Reads both readers' Stage B output for this page (does NOT
    call either reader, that already happened in Stage B, in each reader's own isolated venv),
    normalizes both (3a) BEFORE comparing them, compares them (3b), and returns the usable output,
    the derived ocr_confidence, and the ocr_extraction_record dict for this page (or
    (None, None, None) if there's nothing usable, in which case the caller must skip saving this
    question entirely rather than treat None as valid content, same principle as Issue 62's
    per-file skip, one level down, at question granularity).

    Single-writer database design (the development log, "Single-writer database design
    decided"): this function does NOT write the ocr_extraction_record to anywhere itself, it
    used to call a save_ocr_extraction_record() that attempted a real DB write per call, which is
    exactly the per-worker-writer pattern the single-writer decision reversed (Stage C's parallel
    workers, DEFAULT_WORKERS = 22, must never hold their own DB session). The record is returned
    to the caller instead, which bundles it into that file's single JSON checkpoint alongside its
    questions, see save() and write_extracted_data_to_db() further below for where it actually
    gets persisted and, eventually, written to the real database in one single, non-parallel pass.
    """
    primary_raw = load_raw_output("mineru25", pdf_stem, page_index, raw_dir)
    secondary_raw = load_raw_output("deepseek_ocr", pdf_stem, page_index, raw_dir)

    # Missing-output case (Stage B never processed this page for EITHER reader, or both errored)
    # is distinct from a reader returning empty text on a page it did process, handled here,
    # before normalize_glyphs runs, per Section 15 step 3b. normalize_glyphs() takes a str, not
    # None, so this check must come first regardless.
    if primary_raw is None and secondary_raw is None:
        log_question_failure(question_id, "no Stage B output from either reader for this page")
        return None, None, None

    primary_norm = (
        RawReaderOutput(
            extracted_text=normalize_glyphs(primary_raw.extracted_text),  # 3a
            structure=primary_raw.structure,
            bounding_boxes=primary_raw.bounding_boxes,
            field_confidence=primary_raw.field_confidence,
        )
        if primary_raw is not None else None
    )
    secondary_norm = (
        RawReaderOutput(
            extracted_text=normalize_glyphs(secondary_raw.extracted_text),  # 3a
            structure=secondary_raw.structure,
            bounding_boxes=secondary_raw.bounding_boxes,
            field_confidence=secondary_raw.field_confidence,
        )
        if secondary_raw is not None else None
    )

    agreement_status, disagreement_detail, disagreement_type = compare_readers(
        primary_norm, secondary_norm,
    )

    if agreement_status == "both_failed":
        log_question_failure(question_id, "both readers produced no usable output")
        return None, None, None  # caller must skip saving this question, not save an empty row

    # Build the ocr_extraction_record dict here, but do NOT write it anywhere, single-writer
    # design (see docstring). The caller collects this into its file's ExtractionRunResult and
    # save() persists it as part of that file's one JSON checkpoint; the real DB row is written
    # later by write_extracted_data_to_db()'s single, non-parallel pass.
    ocr_record = {
        "question_id": str(question_id),
        "primary_reader_output": primary_norm.extracted_text if primary_norm else None,
        "secondary_reader_output": secondary_norm.extracted_text if secondary_norm else None,
        "agreement_status": agreement_status,
        "disagreement_detail": disagreement_detail,
        # Issue 212, real, structured value/structure classification,
        # computed directly by compare_readers() itself (or, for a real per-question grid-conflict
        # override, by _apply_grid_cross_check() below), not derived from disagreement_detail's
        # own free text after the fact.
        "disagreement_type": disagreement_type,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
    }

    ocr_confidence: Literal["high", "low"] = "high" if agreement_status == "agree" else "low"

    # Fallback logic, do NOT unconditionally return primary_norm (the development log,
    # single-reader fallback bug fix). If single_reader_only, return whichever reader actually
    # produced output.
    if agreement_status == "single_reader_only":
        usable_output = (
            primary_norm if primary_norm is not None and primary_norm.extracted_text.strip()
            else secondary_norm
        )
    else:
        # readers agree or disagree; primary is the recorded value either way, disagreement is
        # what drives ocr_confidence, not which reader's text is kept
        usable_output = primary_norm

    # LOG THE FIRST REAL DISAGREEMENT INSTANCES FOUND : the 18-page gold
    # set never produced a real "disagree", every comparison came back "agree". Stage C running
    # against real corpus pages is the first point that data will actually exist. Log it plainly
    # here so the first real instances are captured, not lost, per the standing "log every real
    # finding" discipline.
    if agreement_status == "disagree":
        logger.warning(
            "FIRST-CLASS FINDING — real OCR disagreement, question_id=%s: %s. "
            "Log this in docs/DEVELOPMENT_LOG.md per the development log's standing instruction.",
            question_id, disagreement_detail,
        )

    return usable_output, ocr_confidence, ocr_record


# =================================================================================================
# _text_similarity, Issue 100's real fix , the two-comparison-jobs
# distinction. This project has two different real comparison questions:
#   (1) "Did the two readers transcribe the same real question", a TEXT-similarity question. Two
#       readers reading the same real page produce close-but-not-identical text (spacing, LaTeX
#       escaping conventions, diagram-vs-image-placeholder handling, boilerplate insertion) and
#       that should still count as agreement when close. This is what
#       segment_and_compare_page()'s per-question comparison actually needs, NOT
#       values_equivalent(), which Issue 99 correctly scoped down to per-span text but which
#       remained the wrong TOOL for this job (a numeric/semantic equivalence checker built for
#       short answer values like "1/4" vs "0.25", not free-text prose comparison, confirmed real
#       by running it: agree stayed 0/78 on matching real content, see Issue 99).
#   (2) "Do these two candidate answer VALUES mean the same thing", a real numeric/semantic
#       equivalence question (1/4 vs 0.25, $0.50 vs 50¢). This is what values_equivalent() was
#       built for and is still correctly used for elsewhere (parse_answer_key_grid()'s collision
#       checks, sub-part answer combination), untouched by this fix, not the problem.
# =================================================================================================

def _text_similarity(a: str, b: str) -> float:
    """
    Real text-similarity ratio for job (1) above, difflib.SequenceMatcher, stdlib, no new
    dependency, already the standard tool for exactly this real comparison shape (near-duplicate
    text detection). Whitespace-normalized first (collapse all runs of whitespace, including
    newlines, to one space, then strip), SequenceMatcher compares raw character sequences, and the
    two readers wrap lines differently for the SAME real content (confirmed real, e.g. MinerU2.5
    vs DeepSeek-OCR's own line-break conventions), which would otherwise register as spurious
    dissimilarity unrelated to real content agreement. Returns a float in [0, 1]; 1.0 is identical.
    """
    a_norm = re.sub(r"\s+", " ", a).strip()
    b_norm = re.sub(r"\s+", " ", b).strip()
    return difflib.SequenceMatcher(None, a_norm, b_norm).ratio()


def _text_similarity_layout_normalized(a: str, b: str) -> float:
    """
    Real, narrow companion to `_text_similarity()` above (Issue 228(a) fix, development log
), used ONLY at `segment_and_compare_page()`'s own comparison when at least one side
    of the pair is a cross-reader-RECOVERED synthetic span (`QuestionSpan.flagged ==
    "cross_reader_recovered"`, Issue 224). A recovered span's own `raw_text` is a real, verbatim
    slice of the OTHER reader's raw page text (see `_find_recovered_span_text()`'s own docstring) , 
    unlike a normally-segmented span, it never passed through that reader's own per-question
    boilerplate-stripping, so it can still carry real cross-reader layout/markup differences (HTML
    table tags, image placeholders) a normal pair's two independently-segmented `raw_text` values
    don't carry nearly as often. Confirmed directly: without this, a GOOD recovery (a
    long, real content match, found via the SAME `_normalize_layout_for_recovery()` used to decide
    whether to recover it in the first place) could still score a WORSE plain `_text_similarity()`
    than a shorter, markup-coincidental one, because that function only normalizes whitespace, not
    markup (the real case this fixes: `P6_Maths_2024_SA2_methodistgirls` p8 q17/q18, an 82-char
    real recovered match scored 0.608, below the 0.65 agreement threshold, purely because of
    leftover `</td><td>` markup literally present in the recovered text; the same real match,
    layout-normalized, is close content).

    Deliberately NOT used for a normal, non-recovered pair, `_text_similarity()`'s own 0.65
    threshold was tuned against 78 real, normally-matched pairs that never
    carried this same recovered-span markup-fragmentation issue; broadening layout normalization to
    every comparison is a materially bigger, untested change this fix does not make.
    """
    a_norm, _ = _normalize_layout_for_recovery(a)
    b_norm, _ = _normalize_layout_for_recovery(b)
    return difflib.SequenceMatcher(None, a_norm, b_norm).ratio()


# Real threshold, picked against real data, not a guessed round number :
# pulled all 78 real matched-pair "disagree" records from Issue 99's own validation run (both
# validation files, every pair where BOTH readers independently segmented a question with the same
# number), computed _text_similarity() on each real pair, and inspected the full sorted list by
# hand before choosing. Real, clean gap found in the actual data between 0.590 and 0.469 (the
# widest gap in the whole distribution below 0.9), every pair scoring >= ~0.69 was confirmed by
# eye to be the same real question with only cosmetic differences (diagram-vs-image-placeholder,
# LaTeX escaping, "Do not write in this space" boilerplate insertion, minor spacing); every pair
# scoring <= ~0.62 showed a real, meaningful difference in captured content (typically one reader
# including shared-stimulus preamble text the other didn't, or severe currency/LaTeX corruption
# on one side), worth a human review flag, not safe to silently trust. 0.65 sits in that
# real gap, closer to its lower edge (favors flagging borderline cases for review over silently
# passing them, matching this project's standing safety-first bias, see Issue 58's fail-closed
# principle, same instinct applied to a data-quality signal instead of a safety one).
_TEXT_SIMILARITY_AGREEMENT_THRESHOLD = 0.65


# Real, deliberate thresholds for cross-reader recovery (Issue 224 fix, development log
# 2026-09-0X), see _find_recovered_span_text()'s own docstring. Both an ABSOLUTE and a
# PROPORTIONAL minimum, not just one: absolute alone would let a long span recover from a tiny
# coincidental overlap (e.g. a shared boilerplate fragment); proportional alone would let a short,
# unrelated span recover from a short coincidental match. Requiring both keeps this
# conservative in the direction that matters (a missed recovery falls back to the existing,
# already-safe single_reader_only default; a false recovery would run a real, potentially wrong
# value comparison).
_RECOVERY_MIN_MATCH_CHARS = 40
_RECOVERY_MIN_MATCH_RATIO = 0.5

# Issue 228(b) fix (the development log, 621-record single_reader_only investigation): the
# 40-char absolute floor above is the right, conservative call for a PARTIAL match (guards against
# a short, generic, coincidentally-recurring fragment, "Ans:", a bare shared number, satisfying
# the 50% ratio purely by chance at small scale). It is the WRONG mechanism for a match that is a
# COMPLETE, verbatim containment of the needle (match.size == len(needle_norm), ratio 1.0):
# unlike a partial overlap, the coincidence risk for a FULL match scales with the needle's own
# length, not a fixed floor, the entire real needle recurring by chance inside unrelated page text
# is already implausible well below 40 characters. Two real, confirmed cases were being wrongly
# blocked by the flat 40-char floor despite a 100%-ratio complete match: `PSLE Math 2019` p9 q16
# ("Find the value of 1056 - 68 Ans:", 33 normalized chars) and `P6_Maths_2022_SA2_nanhua` p45 q23
# ("a) North-West b) C", 18 normalized chars). This separate, LOWER floor for the full-match case
# only is picked below both real confirmed cases with a small margin, while still excluding
# clearly-trivial, coincidence-prone full matches a real page could produce on its own
# ("Ans:" = 4 chars, "(a)" = 3 chars, a bare 1-2 digit number), 15 sits in the real gap between
# those trivial stumps and the shorter of the two confirmed genuine cases (18 chars), not a
# re-guessed round number picked independently of the real data.
_RECOVERY_MIN_MATCH_CHARS_FOR_FULL_CONTAINMENT = 15


# Issue 228(a) fix (the development log, 621-record single_reader_only investigation): real,
# confirmed root cause for why `_find_recovered_span_text()` below was missing real, substantial
# content overlaps, its longest-CONTIGUOUS-match check ran on RAW text, and MinerU2.5's own
# region-based `raw_text` embeds blank-line separators, `<table>`/`<tr>`/`<td>` markup, and
# `![](images/N.jpg)` image placeholders that DeepSeek-OCR's flat text doesn't reproduce in the
# same layout, none of which reflect a real CONTENT difference, only how each reader represents
# structure, but each one breaks a single long real match into several short contiguous pieces.
# Confirmed directly by re-running the real match on 3 real pairs: raw ratio 15.6%/25.0%/32.5% vs.
# layout-normalized ratio 74%/77%/48% for the SAME pairs (`P6-Maths-2021-SA2-Catholic-High` p7,
# `P6-Maths-2021-SA2-Red-Swastika` p31, `P6-Maths-2021-SA1-Methodist-Girls` p22).
_LAYOUT_ARTIFACT_PATTERN = re.compile(r"!\[\]\(images/\d+\.jpg\)|</?[a-zA-Z][^>]*>")


def _normalize_layout_for_recovery(text: str) -> tuple[str, list[int]]:
    """
    Strips the real, confirmed layout-only artifacts above and collapses whitespace runs to a
    single space, WITHOUT losing the ability to map a match found in the normalized text back to
    the exact real substring of the ORIGINAL, unmodified text. Returns (normalized_text,
    index_map) where index_map[i] is normalized_text[i]'s real position in the original `text`.

    Why a position map, not just a normalized string: `_find_recovered_span_text()` must keep
    returning a real, verbatim slice of the reader's own original `haystack`, nothing downstream
    (`_text_similarity()`'s own separate comparison, or anything reading `QuestionSpan.raw_text`
    later) should see anything different from before this fix. This fix's own scope is narrowly
    "does a real match exist", never "change what text represents a recovered span."

    An artifact span is treated as a WORD BOUNDARY (one virtual space), never a plain deletion , 
    a real, confirmed bug found while verifying this fix directly against `PSLE Math 2019` p9 q16:
    the real raw HTML has zero literal whitespace between adjacent cell/row tags
    (`...68</td><td></td></tr><tr><td></td><td>Ans:</td>...`), so deleting the tags outright fuses
    "68" and "Ans:" into "68Ans:" with no separating character, breaking the very match this fix
    exists to find (the needle's own real content has "68 Ans:", WITH a space). Emitting one space
    per artifact span instead keeps "68" and "Ans:" correctly word-separated, matching how a human
    reading two different table cells' content would naturally join them.
    """
    normalized_chars: list[str] = []
    index_map: list[int] = []
    prev_was_space = True  # collapses real leading whitespace too, same as needle/haystack .strip()
    pos = 0
    for m in _LAYOUT_ARTIFACT_PATTERN.finditer(text):
        for i in range(pos, m.start()):
            ch = text[i]
            if ch.isspace():
                if prev_was_space:
                    continue
                normalized_chars.append(" ")
                index_map.append(i)
                prev_was_space = True
            else:
                normalized_chars.append(ch)
                index_map.append(i)
                prev_was_space = False
        if not prev_was_space:
            normalized_chars.append(" ")
            index_map.append(m.start())
            prev_was_space = True
        pos = m.end()
    for i in range(pos, len(text)):
        ch = text[i]
        if ch.isspace():
            if prev_was_space:
                continue
            normalized_chars.append(" ")
            index_map.append(i)
            prev_was_space = True
        else:
            normalized_chars.append(ch)
            index_map.append(i)
            prev_was_space = False
    while normalized_chars and normalized_chars[-1] == " ":
        normalized_chars.pop()
        index_map.pop()
    return "".join(normalized_chars), index_map


def _find_recovered_span_text(needle: str, haystack: str) -> Optional[str]:
    """
    Real, bounded cross-reader content recovery (Issue 224 fix), does `haystack` (the OTHER
    reader's own whole-page `extracted_text`) contain a substantial match for `needle` (an
    unmatched span's own `raw_text`)?

    Real root cause this recovers, confirmed directly investigating Issue 224's
    `single_reader_only` bucket: a completely-dropped leading question-number means
    `segment_questions()` itself, given only ONE reader's own data, cannot tell "a genuine new,
    unnumbered question" from "a continuation of the still-open previous span", there is no
    reliable per-reader signal once the number itself is gone (confirmed real cases: MinerU2.5
    silently merging a dropped-number question's content onto the still-open previous span;
    `classify_page()` finding no recognizable question-start anywhere on a page at all, at the
    page-classification level, before segmentation even runs, even though the page's real prose IS
    a genuine question). The OTHER reader's own independent segmentation remains real, direct
    evidence the content exists, whether or not THIS reader gave it its own correctly-numbered span.

    Issue 228 fix : the match is now DECIDED on layout-normalized text
    (see `_normalize_layout_for_recovery()`'s own docstring for why cross-reader markup/whitespace
    differences were defeating real, substantial matches), with a separate, lower absolute floor
    for a full, complete containment of the needle (see `_RECOVERY_MIN_MATCH_CHARS_FOR_FULL_
    CONTAINMENT`'s own docstring). The RETURNED text is still mapped back to a real, verbatim slice
    of the ORIGINAL `haystack`, this fix changes only WHETHER a match is found, never what text a
    successful recovery hands to the caller.

    Returns the matched substring of `haystack` (a real window of the OTHER reader's own text
    covering this same content, usable for a genuine `_text_similarity()` value comparison exactly
    like any normally-matched pair) if a sufficiently long AND sufficiently proportional match is
    found, else `None`, deliberately conservative on both dimensions so a coincidental short
    overlap (a shared number, a common short phrase) never triggers a false recovery. `None` here
    means the caller falls back to the existing, already-safe `single_reader_only` classification,
    never a worse outcome than before this fix.
    """
    if not needle.strip() or not haystack.strip():
        return None
    needle_norm, _ = _normalize_layout_for_recovery(needle)
    haystack_norm, haystack_index_map = _normalize_layout_for_recovery(haystack)
    if not needle_norm or not haystack_norm:
        return None
    matcher = difflib.SequenceMatcher(None, needle_norm, haystack_norm, autojunk=False)
    match = matcher.find_longest_match(0, len(needle_norm), 0, len(haystack_norm))
    is_full_containment = match.size == len(needle_norm)
    min_chars = (
        _RECOVERY_MIN_MATCH_CHARS_FOR_FULL_CONTAINMENT if is_full_containment
        else _RECOVERY_MIN_MATCH_CHARS
    )
    if match.size < min_chars:
        return None
    if match.size / len(needle_norm) < _RECOVERY_MIN_MATCH_RATIO:
        return None
    orig_start = haystack_index_map[match.b]
    orig_end = haystack_index_map[match.b + match.size - 1] + 1
    return haystack[orig_start: orig_end]


# Issue 229 fix, second attempt, real, narrow recovery for
# `P6_Maths_2024_WA2_rosyth` p35's own confirmed root cause: MinerU2.5 misreads this file's real
# zero-padded worked-solution numbering ("01)", "02)"...) as "Q1)", "Q2)"... (a real, confirmed
# glyph confusion, "0" read as "Q", checked directly against the real bounding-box regions, not
# guessed). The FIRST attempt at this fix (reverted, the development log, pre-610a408) added
# "Q<1-3 digits>" as a general new alternative inside the SHARED, corpus-wide `_match_question_
# start()`, real, confirmed regression: on `P6_Maths_2023_WA2_Taonan` p32/p33, a real, GENUINE
# (not zero-padding-related at all) "Q11." reference, Taonan's own actual printed worked-solution
# numbering convention, unrelated to any OCR misread, collided with a pre-existing, unrelated
# bare-digit false positive for the SAME number ("11 units -> 66 jugs", a real numeric quantity
# mistaken for a question start by the ALREADY-EXISTING digit pattern), and the downstream
# `{question_number: span}` dict silently dropped one span's real content. Real root evidence for
# the fix below, checked directly (not assumed): the first attempt's own real matches (253 across
# 29 files) included MANY genuine "Q<N>." worked-solution conventions used directly, unrelated to
# zero-padding at all (Tao-Nan, Henry-Park, peichun, chij, SCGS, ...), the broader pattern was
# never safely scopeable to "zero-padding" by regex shape alone, since a MISREAD "Q1)" and a
# GENUINE "Q1)" are byte-for-byte identical text with no way to tell them apart from either
# reader's own text in isolation.
#
# This second attempt is narrower on every real axis the first one was not:
#   1. NEVER touches `_match_question_start()`, `_has_question_like_content()`, or
#      `classify_page()`, the shared, corpus-wide functions every file's segmentation runs
#      through, and the exact functions the first attempt's own regression came from touching.
#      Zero blast radius on any file whose classification or segmentation already works, by
#      construction, this only ever runs when a reader's own segmentation already found NOTHING
#      on a page (see the two `if not primary_spans` / `if not secondary_spans` call sites in
#      `segment_and_compare_page()`), which real, direct checking confirms Taonan never does (its
#      own real "Q7."-"Q15." content and real bare-digit workings already produce real, non-empty
#      spans there, both before AND after this fix, this code path is never even reached for
#      that file).
#   2. Requires REAL CROSS-READER CORROBORATION before accepting a single-digit "Q<N>" match as a
#      genuine zero-padding misread: the OTHER reader's own whole-page text must contain the
#      LITERAL zero-padded form "0<N>" at a real question-start position. A coincidental,
#      unrelated "Q<N>." reference (a genuine worked-solution convention, Taonan's own real shape)
#      has no reason to ALSO have a literal "0<N>" anywhere in the other reader's text, DeepSeek-
#      OCR's own real flat text for Taonan p32/p33 has no "011)"/"01)" anywhere, confirmed
#      directly. Bounded to single digits (1-9) only, zero-padding a real question number never
#      applies to a number already 2+ digits, which by itself already excludes Taonan's own real
#      "Q10."/"Q11."-"Q15." references (all 2-digit) from ever being considered at all.
#   3. Requires AT LEAST 2 real, independently-confirmed candidates on the SAME page before
#      accepting any of them, a single isolated coincidence is not enough evidence; a genuine
#      zero-padded worked-solution page naturally has several (rosyth p35 has 3: "01"/"03"/"08"
#      confirmed misread as "Q1"/"Q3"/"Q8").
#   4. Only ever REPLACES an EMPTY span list, never merges with, reorders, or otherwise disturbs
#      any span a reader's own normal segmentation already produced. The recovered spans' own
#      `question_number` is set to the REAL zero-padded string found in the other reader's text
#      ("01", not "1"), so a successful recovery aligns EXACTLY with the other reader's own
#      existing number and flows through the normal per-question comparison unchanged, no new
#      recovery-specific comparison logic, no new collision surface with anything else on the page
#      (there is nothing else on the page, by construction, since this only fires on an otherwise-
#      empty span list).
_ZERO_PADDED_MISREAD_Q_PATTERN = re.compile(r"^\s*Q(\d)[.)]?\s+(.+)$", re.DOTALL)
_ZERO_PADDED_LITERAL_PATTERN_TEMPLATE = r"(?<!\d)0{digit}[.)]"
_ZERO_PADDED_MISREAD_MIN_CANDIDATES = 2


def _recover_zero_padded_misread_questions(
    empty_reader_text: str, other_reader_text: str, page_index: int,
) -> list[QuestionSpan]:
    """
    Real, narrow recovery, see the real evidence and design comment block directly above. Called
    ONLY when a reader's own `segment_questions()` already returned an empty span list for this
    page. Scans `empty_reader_text` (that reader's own whole-page text) for a real "Q<digit>
    <content>" line/region matching `_ZERO_PADDED_MISREAD_Q_PATTERN`; for each candidate, confirms
    the OTHER reader's own text (`other_reader_text`) contains the literal zero-padded form
    ("0<digit>[.)]", not preceded by another digit), real, direct cross-reader evidence this
    specific candidate is a genuine misread, not a coincidental "Q<digit>" reference. Returns real
    `QuestionSpan`s (one per confirmed candidate, `question_number` set to the real zero-padded
    string, e.g. "01") only if at least `_ZERO_PADDED_MISREAD_MIN_CANDIDATES` are confirmed on
    this page, otherwise returns `[]` unchanged (the caller's own already-safe empty-span
    fallback, never a worse outcome than before this fix).
    """
    if not empty_reader_text.strip() or not other_reader_text.strip():
        return []
    candidates: list[tuple[str, str]] = []
    for line in empty_reader_text.splitlines():
        match = _ZERO_PADDED_MISREAD_Q_PATTERN.match(line.strip())
        if not match:
            continue
        digit, rest = match.group(1), match.group(2).strip()
        if len(rest) < _MIN_QUESTION_CONTENT_CHARS:
            continue
        literal_pattern = re.compile(_ZERO_PADDED_LITERAL_PATTERN_TEMPLATE.format(digit=digit))
        if not literal_pattern.search(other_reader_text):
            continue  # no real cross-reader corroboration for THIS specific digit, reject it
        candidates.append((f"0{digit}", rest))
    if len(candidates) < _ZERO_PADDED_MISREAD_MIN_CANDIDATES:
        return []
    return [
        QuestionSpan(
            question_number=number, raw_text=rest, page_index=page_index, page_label=None,
        )
        for number, rest in candidates
    ]


# Real, narrow fraction-numerator-phantom pattern (Issue 224's remaining gap, development log
#/226), see `_fraction_numerator_phantom_evidence()`'s own docstring for the real,
# confirmed root cause and why this is deliberately scoped to fraction structure alone, never a
# general sequence-position heuristic (the earlier, reverted attempt's own real mistake, see
# Issue 224's revert entry). Tolerant of a real, confirmed OCR digit-splitting quirk (MinerU2.5
# rendering "17" as "1 7" inside a fraction's own braces, e.g. `\frac {1 4}{5}`, confirmed directly
# on `P6_Maths_2022_SA2_chij` p13's own real region data) via the caller's own digit-by-digit
# `\s*`-joined pattern construction, matched WITHOUT this tolerance, a real fraction numerator
# using this same OCR quirk would silently fail to match and this signal would just miss it (a
# safe, accepted false-negative, never a false positive).
_FRACTION_NUMERATOR_PATTERN_TEMPLATE = r"\\frac\s*\{{\s*{digits}\s*\}}\{{"

# Real, small, deliberate tolerance (Issue 226), the matched overlap
# block must start within this many characters of the very beginning of the candidate's own
# normalized text, not merely overlap it somewhere in the middle (see
# _fraction_numerator_phantom_evidence()'s own docstring for the real confirmed false-positive
# this specifically guards against). Small enough to reject a genuine question's own unrelated
# internal fraction reference (which matches deep inside its own real sentence), generous enough
# to absorb real leftover LaTeX closing syntax (`}\)`) between the fraction's own end and where
# real prose resumes.
_FRACTION_PHANTOM_MAX_LEAD_OFFSET = 10

# Real, confirmed MCQ-option-marker pattern (Issue 226), see
# _fraction_numerator_phantom_evidence()'s own docstring for the real, confirmed, multi-file
# regression this guards against (a fraction that is itself just the last of a real MCQ question's
# four printed options is never the source of a dropped-denominator phantom). Matches "(1)"/"(2)"/
# "(3)"/"(4)" (any single digit, real PSLE MCQs are always exactly 4 options, but matching any
# single-digit option marker here rather than hardcoding 1-4 costs nothing and stays correct if a
# future real paper ever has a differently-numbered option list), optionally followed by a LaTeX
# math-mode opener (`\(`) and whitespace, anchored at the END of the checked window (immediately
# before the fraction match itself starts).
_MCQ_OPTION_IMMEDIATELY_BEFORE_PATTERN = re.compile(r"\(\s*\d\s*\)\s*\\?\(?\s*$")


def _fraction_numerator_phantom_evidence(
    candidate_number: str, candidate_text: str, other_reader_text: str,
) -> bool:
    """
    Real, narrow signal (Issue 224's remaining gap, fixed here), is `candidate_number` (an
    unmatched span's own question_number) actually a LaTeX fraction's own NUMERATOR in the OTHER
    reader's text, not a genuine new question?

    Real, confirmed root cause: DeepSeek-OCR sometimes renders a fraction (`\\frac{5}{7}`) with the
    denominator DROPPED entirely, leaving a bare numerator ("5") at a line start that
    `_segment_from_flat_text()`'s own regex then reads as a new "question 5", confirmed directly
    on `P6-Maths-2021-CA1-Henry-Park` p13 (the real page's own questions are 24 and 25; DeepSeek-
    OCR's own text has NO trace of the fraction left at all by the time this check would run on
    ITS OWN text, the denominator is already gone, not just malformed, so the only real evidence
    left is in the OTHER reader's own, more complete transcription).

    Deliberately NARROW, fraction-specific, by design and by construction, NOT the reverted
    general "ascending number looks suspicious" heuristic (Issue 224's own revert: that
    mechanism corrupted real content on `P6_Maths_2022_SA2_chij` p13, where the bare number "18"
    legitimately labels two different real questions across sections, and "17" is a genuine,
    out-of-reading-order real question from a two-column layout, neither has anything to do with
    a fraction). This function CANNOT fire on that shape: verified directly, not assumed, chij
    p13's real content contains real fractions (`\frac{9}{10}`, `\frac{14}{5}`, `\frac{9}{8}`, all
    part of question 17's own real content), but neither reused number ("17", "18") is ever used as
    ANY fraction's numerator anywhere on that page. A candidate this function is asked about is
    only ever suspected if the OTHER reader's text contains a `\frac{N}{` structure for that EXACT
    number, a page with genuine number reuse but no fraction involving that number produces zero
    matches, by construction, not by tuning.

    A SECOND, additional real safeguard beyond the bare fraction-numerator match, to guard against
    a different but real risk, CONFIRMED REAL during this fix's own full-corpus
    validation, not hypothetical: a GENUINE, correctly-numbered question can legitimately discuss
    a fraction with the SAME number as its own numerator somewhere INSIDE its own real body text
    (real confirmed case: `P6-Maths-2021-SA1-Singapore-Chinese-Girls` p16's own genuine Q1 , 
    "Eliza ate `\frac{1}{5}` of a pizza...", a real question numbered "1" that also happens to
    discuss a fraction whose OWN numerator is "1", entirely coincidentally, partway through its
    own sentence). Checking for "substantial content overlap anywhere" (this function's own first,
    real, but too permissive draft) fires on this case too, since the OTHER reader's real text
    right after that SAME `\frac{1}{` naturally overlaps heavily with Q1's own real continuation , 
    it's the SAME real sentence, just anchored at a different, unrelated point within it.

    THE REAL, DEFINING DIFFERENCE, checked here, BOTH sides of the match, not just one (a second
    real, confirmed false positive this fix's own full-corpus validation caught after the first:
    `P6_Maths_2024_SA2_nanhua` p2, an UNRELATED real fraction, `\frac{6}{7}`, is itself
    just one MCQ option's own mixed number inside a DIFFERENT question (Q5); a genuine, unrelated
    real Q6 ("Which of the following shows...") simply happens to follow it, later in the same
    page's real text. Q6's own candidate text starts with that exact phrase, so checking only
    "does the match cover the candidate from ITS OWN start" (`match.a`) passes, the real, missing
    half is checking that the match ALSO starts near the BEGINNING OF THE WINDOW right after the
    fraction (`match.b`), not deep inside it after real intervening content (closing LaTeX syntax,
    an image reference, the real "6" label itself)):
      - `match.a` (position within the candidate's own text): a genuine phantom's OWN content
        begins essentially AT the point where the lost fraction's own denominator would have
        been, so the match must cover the candidate from at or very near its own position 0 (a
        real question's own UNRELATED internal fraction reference fails this, Singapore-Chinese-
        Girls p16's genuine Q1 matches deep inside its own sentence, not at its start).
      - `match.b` (position within the window immediately following the fraction match): the
        SAME real content must begin at or very near the START of that window (a small tolerance
        for leftover LaTeX closing syntax like a closing brace-backslash-paren sequence), a
        coincidental, unrelated real question appearing later in the page's text, after other
        real content the fraction has nothing to do with, fails this (the nanhua p2 case above).
    Both conditions must hold; either one alone is not sufficient, confirmed by these two real,
    distinct false-positive shapes each defeating only one of the two checks.

    A THIRD, real, confirmed safeguard, found by this fix's own full-corpus validation AFTER the
    first two were already in place, the most serious of the three, since it recurred across
    MULTIPLE real files, not one: a fraction that is itself just the LAST of a real MCQ question's
    four printed options ("(1) ... (2) ... (3) ... (4) `\frac{N}{...}`") is, structurally, always
    immediately followed by whatever REAL question comes next on the page, and PSLE question
    numbers are small integers, so that next real question's own number coincidentally matching
    the option's own fraction numerator is common, not rare (confirmed real cases:
    `P6-Maths-2021-SA1-Methodist-Girls` p3, `P6-Maths-2021-SA2-Henry-Park` p3,
    `P6_Maths_2023_WA1_Nanyang` p4, `P6_Maths_2025_SA2_redswastika` p3, all four are a real,
    independently-numbered NEXT question, wrongly suppressed and merged into the wrong preceding
    question, a real, confirmed regression the first two safeguards alone did not catch since the
    real next question begins close enough after the option to satisfy both position checks).
    Real, structural fix: an MCQ option's own fraction is never the source of a dropped-denominator
    phantom (the phantom mechanism requires the fraction to BE the sentence, not one item in an
    enumerated list), checked by requiring the text immediately BEFORE the fraction match does
    NOT itself look like an MCQ option marker ("(1)", "(2)", "(3)", or "(4)", optionally followed
    by a LaTeX math-mode opener).
    """
    if not candidate_number or not candidate_number.isdigit():
        return False
    if not candidate_text.strip():
        return False
    digit_pattern = r"\s*".join(re.escape(d) for d in candidate_number)
    frac_pattern = re.compile(_FRACTION_NUMERATOR_PATTERN_TEMPLATE.format(digits=digit_pattern))
    needle = re.sub(r"\s+", " ", candidate_text).strip()
    for m in frac_pattern.finditer(other_reader_text):
        preceding = other_reader_text[max(0, m.start() - 15): m.start()]
        if _MCQ_OPTION_IMMEDIATELY_BEFORE_PATTERN.search(preceding):
            continue  # this fraction is just one of the question's own MCQ options, not a
                       # standalone sentence, never the source of a dropped-denominator phantom
        window = other_reader_text[m.end(): m.end() + max(len(candidate_text) * 2, 150)]
        haystack = re.sub(r"\s+", " ", window).strip()
        if not haystack:
            continue
        matcher = difflib.SequenceMatcher(None, needle, haystack, autojunk=False)
        match = matcher.find_longest_match(0, len(needle), 0, len(haystack))
        if match.size < _RECOVERY_MIN_MATCH_CHARS:
            continue
        if match.size / len(needle) < _RECOVERY_MIN_MATCH_RATIO:
            continue
        # BOTH positions must be near zero, not just one, a real, confirmed second false-positive
        # this fix's own full-corpus validation caught: `match.a` alone (candidate/needle-side
        # position) is NOT sufficient. Real case: `P6_Maths_2024_SA2_nanhua` p2, an UNRELATED
        # real fraction (`\frac{6}{7}`, itself just one MCQ option's own mixed number inside a
        # DIFFERENT question, Q5) happens to be followed, later in the same page's real text, by a
        # genuine, unrelated real Q6 ("Which of the following shows..."). Q6's own candidate_text
        # starts with that exact same phrase, so `match.a == 0`, but the match starts DEEP INSIDE
        # the window (`match.b` large, after real intervening content: closing LaTeX syntax, an
        # image reference, the real "6" label itself), checking `match.b` too catches this: a
        # genuine phantom's real content begins essentially immediately after the lost
        # denominator, so `match.b` must also be small, not merely `match.a`.
        if match.a > _FRACTION_PHANTOM_MAX_LEAD_OFFSET:
            continue
        if match.b > _FRACTION_PHANTOM_MAX_LEAD_OFFSET:
            continue
        return True
    return False


def _suppress_fraction_numerator_phantoms(
    spans: list[QuestionSpan], other_reader_text: str,
) -> list[QuestionSpan]:
    """
    Real, narrow fix applying `_fraction_numerator_phantom_evidence()` across one reader's own
    ordered span list (Issue 224's remaining gap). A confirmed fraction-numerator phantom's own
    real content is merged onto the immediately preceding span in the SAME reader's own list, the
    same "never lose real content" principle the reverted general filter also followed for this
    exact sub-case (the phantom's own prose, e.g. "of all the students who were late were girls...",
    belongs to the preceding question, just OCR'd under the wrong label), never
    discarded, never merged across readers (that is `_find_recovered_span_text()`'s own separate,
    already-shipped job).

    Only ever suppresses a span when there IS a preceding real span in THIS SAME list to merge
    onto, the first span on a page has no real predecessor, so it is never touched here regardless
    of its own number, the identical safety property the reverted general filter also relied on.
    """
    if len(spans) < 2 or not other_reader_text:
        return spans
    kept: list[QuestionSpan] = []
    for span in spans:
        if kept and _fraction_numerator_phantom_evidence(
            span.question_number, span.raw_text, other_reader_text,
        ):
            kept[-1].raw_text = f"{kept[-1].raw_text}\n{span.raw_text}".strip()
            if span.has_diagram:
                kept[-1].has_diagram = True
            continue
        kept.append(span)
    return kept


# =================================================================================================
# _grid_answers_agree, Issue 101's real fix : the grid-answer
# cross-check both_readers_agree() calls need to compare EACH reader's own independently-parsed
# answer-key-grid value for the same real (section, question_number), the actual candidate answer
# VALUE this project's two-reader design always meant to catch a real misread of, sourced correctly
# this time (Issue 98/99/100's own reported residual: pure text-similarity on a
# QUESTION span can't catch a single-digit final-answer difference; the real answer lives in the
# separate answer-key grid, not the question page, confirmed 0/78 extractable there, development log
# "step 1" finding).
#
# NOT a single values_equivalent() call, though, real evidence (pulled from this project's own
# 30 real matched grid-answer pairs, both readers' own independent parses, before finalizing this)
# confirmed the SAME "wrong-scope comparator" pattern resurfaces one level deeper: 84/113 (74%) of
# real grid-answer cells are short bare values (values_equivalent() is exactly right there), but
# 29/113 (26%) are full multi-line worked solutions (chij's Booklet B/Paper 2 print complete
# workings, not just final answers, a real, school-specific answer-key printing convention, not
# universal; PSLE Math 2019's own answer key happens to print bare values only). Directly tested:
# values_equivalent() on two readers' matching worked-solution text for the same real
# question returns False both times (confirmed on real Q18/Q29 pairs), the exact same prose-vs-
# short-value mismatch already fixed once for question text, now one level deeper.
# =================================================================================================

def _is_prose_shaped_grid_answer(a: str, b: str) -> bool:
    """
    Real signal for whether a grid-answer pair needs the text-similarity
    fallback below rather than trusting values_equivalent()'s "not equivalent" verdict at face
    value: a real newline present in EITHER reader's own cell text. Checked directly against this
    project's own 113 real grid-answer cells before using it, not assumed, 99.1% clean (112/113)
    against the length-based classification used to first characterize the real 74%/26% split.
    The one real exception found (a single-line, no-newline chained calculation, "150/60=15/6=
    21/2ANS:3.30 pm") is NOT specially handled here, it falls through to plain values_equivalent(),
    which correctly (if conservatively) calls it "not equivalent" given the chained "=" signs and
    embedded "ANS:" text sympy's parser can't make sense of as one expression, flagging it for
    review rather than silently trusting it. An accepted, documented trade-off (fails closed on one
    known rare shape), not a silent gap.
    """
    return ("\n" in a) or ("\n" in b)


# Real threshold, picked against real data, NOT reused from Issue 100's question-text 0.65 , 
# explicit instruction, confirmed real evidence it may (and does) land somewhere different. Pulled
# all 30 real prose-shaped matched grid-answer pairs from both files (both readers' own independent
# parses, common (section, question_number) keys where either side's text contained a newline),
# computed real _text_similarity() scores, inspected the full sorted list by hand. Real, clean,
# WIDE gap found: 28 genuine matches (confirmed by eye, same real worked solution, differing only
# in "x" vs "×", spacing around "=", degree-sign spacing, "ANS:" vs "ANS: ") cluster from 1.000 down
# to 0.681; exactly 2 real corrupted values sit far below, at 0.400 and 0.333, investigated
# directly, not just numerically: both trace to a genuine, PRE-EXISTING limitation in
# parse_answer_key_grid()'s odd-cell-count handling (built for a different real shape, chij p47's
# continuation row, where the missing cell is first in the row) mishandling a MID-row
# HTML rowspan gap in PSLE Math 2019 Answer's own grid ("22a. 22b." combined into one rowspan cell,
# confirmed via direct _parse_html_table_rows() inspection), not new to this fix, not something
# either fallback comparator is expected to see through, and correctly flagged as disagree either
# way (Log: kept separate from this fix, not chased further here). 0.60 sits in the
# real 0.400-0.681 gap: comfortably above both corrupted values (no risk of misclassifying either as
# agree) and with real margin below the lowest genuine match (0.681), not picked to just barely
# clear it, same conservative-edge philosophy as Issue 100's own threshold choice, landing at a
# different real number exactly as anticipated, not assumed to transfer.
_GRID_ANSWER_TEXT_SIMILARITY_THRESHOLD = 0.60


_ANS_MARKER_PATTERN = re.compile(r"ans\s*[:.]", re.IGNORECASE)


def _extract_grid_final_answer(text: str) -> Optional[str]:
    """
    Real, evidence-checked extractor for a grid answer-key cell's own
    candidate FINAL answer, built after a real, confirmed failure of the first design attempt
    (whole-cell text similarity alone): a deliberately constructed counter-example, two worked
    solutions identical except the final computed value (72 vs 75), was WRONGLY judged "agree" by
    whole-cell text similarity (0.979, comfortably above threshold), the exact class of error this
    whole cross-check exists to catch, just relocated one level deeper. Fixed by comparing each
    side's own extracted FINAL answer instead of the whole cell.

    Checked against this project's real 29 prose-shaped grid cells before trusting it, not assumed:
    9 of 13 hand-checked pairs (69%) yielded a confident, CONSISTENT extraction on both sides; the
    remaining 4 correctly yielded no confident candidate on either side (a lettered/prose answer
    with no "=" or "ANS:" signal at all, e.g. "b) C"); never a false positive on the 2 known-
    corrupted rowspan-artifact cells either (Issue 101's own real finding, a pre-existing
    parse_answer_key_grid() limitation unrelated to this extraction), graceful "don't know" in
    both directions, not a guess dressed up as one.

    Two real signals, in order:
      1. An explicit "ANS:"/"Ans:" marker, take everything after its LAST occurrence (a multi-
         sub-part cell can carry more than one; the last is this cell's own final line).
      2. No marker, the right-hand side of the LAST "=" on the LAST non-blank line, the same real
         convention as plain arithmetic working ("5+4=9\n72÷9=8\n8×4=32" -> "32").
    Returns None (not a guess) when neither signal is present or the matched trailing content is
    itself blank.
    """
    matches = list(_ANS_MARKER_PATTERN.finditer(text))
    if matches:
        trailing = text[matches[-1].end():].strip()
        return trailing or None
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    last_line = lines[-1]
    if "=" in last_line:
        rhs = last_line.rsplit("=", 1)[1].strip()
        return rhs or None
    return None


# Real, confirmed-necessary character equivalences for
# _cosmetically_equal() below, NOT added to normalize_glyphs()'s own shared map (its existing "²"
# entry is a deliberate identity mapping, and that map's own docstring already cautions against
# changing it without confirming every existing real usage first; this project's Stage 3a
# normalization runs on ALL reader output before segmentation too, a much broader blast radius than
# the narrow, LOCAL need confirmed here). Both entries found real and necessary while validating
# THIS fix, not hypothetical: superscript digits (¹²³...) vs their plain-digit form in a squared/
# cubed unit ("120cm2" vs "120cm²"); the multiplication sign "×" vs the plain letter "x" (confirmed
# real: "9.81×1000=9810g" vs "9.81 x 1000 = 9810g", the exact same real working).
_COSMETIC_CHAR_MAP = str.maketrans({
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "×": "x",
})


def _cosmetically_equal(a: str, b: str) -> bool:
    """
    Real, confirmed-necessary "same real content, different rendering" check (development log
), applied UNCONDITIONALLY everywhere this module compares two candidate grid-answer
    strings, not just inside the prose-extraction branch. Found needed there FIRST (a prose cell's
    extracted final answer, "(6π+72)cm" vs "(6 π + 72) cm", sympy can't parse either as an
    expression, so values_equivalent() fails on both sides regardless of true equality), then AGAIN
    at the whole-cell/short-value level once real data was checked broadly (SHORT grid values that
    are lists/codes rather than single numeric expressions, "F;F;N" vs "F; F; N", "62500" vs
    "62 500", "1,3,9" vs "1, 3, 9", and the "×"/superscript-digit cases above), the same class of
    gap kept resurfacing at every level checked, so this collapses whitespace-removal AND the small
    character-equivalence map into ONE shared check, used consistently, rather than duplicated
    ad hoc at each call site.

    Safe in every case: only removes whitespace or maps a small, closed, confirmed-necessary set of
    characters to a canonical form, never changes a DIGIT or drops real content, a genuine value
    conflict (e.g. "72" vs "75") still correctly fails this check, confirmed directly against this
    project's own real data. Does not reopen the blind spot this whole function exists to close.
    """
    def _canon(s: str) -> str:
        return re.sub(r"\s+", "", s).translate(_COSMETIC_CHAR_MAP)
    return _canon(a) == _canon(b)


def _grid_answers_agree(a: Optional[str], b: Optional[str]) -> Optional[bool]:
    """
    Cross-checks two readers' own independently-parsed answer-key-grid values for the same real
    (section, question_number). Returns True (agree), False (a real, confirmed conflict, the
    single most safety-relevant signal this whole two-reader design exists to catch), or None when
    a confident comparison isn't possible at all (one or both sides has no grid answer here), the
    caller must fall back to the existing text-similarity-only result in that case, honestly
    documented as reduced coverage on this specific comparison, not silently treated as agreement.

    For prose-shaped (multi-line/worked-solution) cells, does NOT fall back to whole-cell text
    similarity as its first move (an earlier design attempt did, and was found, during this
    fix's own validation, not in production, to wrongly call two workings "agree" when they were
    identical except the actual final computed value; see _extract_grid_final_answer()'s own
    docstring for the real counter-example that caught it). Extracts each side's own candidate
    final answer FIRST and compares those with values_equivalent() (the real, correct tool for a
    genuine short-value comparison) instead, whole-cell text similarity is used only as the last
    resort, when a confident final answer can't be extracted from one or both sides at all.
    """
    if a is None or b is None:
        return None
    if values_equivalent(a, b) or _cosmetically_equal(a, b):
        return True
    if not _is_prose_shaped_grid_answer(a, b):
        return False  # a short value, not just a cosmetic difference, values_equivalent() already
                       # found NOT equivalent, the right, sufficient tool for this shape

    a_final = _extract_grid_final_answer(a)
    b_final = _extract_grid_final_answer(b)
    if a_final is not None and b_final is not None:
        if values_equivalent(a_final, b_final) or _cosmetically_equal(a_final, b_final):
            return True
        # Both sides confidently extracted a real final answer, and they disagree , 
        # exactly the signal this cross-check exists to produce. Deliberately does NOT fall
        # through to whole-cell text similarity from here, that is exactly the false-agree this
        # fix was built to close (see this function's own docstring).
        return False

    # Could not confidently extract a final answer from one or both sides, reduced
    # coverage (explicit instruction): whole-cell text similarity is the best remaining real
    # signal for "did these two readers describe the same real working," a real but weaker claim
    # than "the final answers agree," which is the case for the cells that reach this line.
    return _text_similarity(a, b) >= _GRID_ANSWER_TEXT_SIMILARITY_THRESHOLD


_COMBINED_SUBPART_LINE_PATTERN = re.compile(r"^\([a-zA-Z0-9]{1,3}\)\s")


def _is_combined_subpart_answer(text: str) -> bool:
    """
    Real, structural detector for _combine_lettered_siblings()'s OWN output shape
    ("(a) 21\n(b) 24/9", GS005/PSLE Math 2019's real separate-lettered-answer-row structure , 
    see that function's own docstring), every real line in the text starts with its own
    "(letter)" sub-part marker. Found necessary (Issue 102 part 3
    follow-up), NOT anticipated: the real end-to-end validation run this fix's own discipline
    requires showed 0/17 real PSLE Math 2019 "prose-shaped" answers yielding a distilled
    answer_value via _extract_grid_final_answer(), a stark, suspicious drop from chij's own
    25/28 on the identical code path. Direct inspection of the real 17 found every one was this
    combined-siblings shape, not a genuine worked solution, _combine_lettered_siblings() joins
    multiple real SHORT per-sub-part answers ("22a."->"21", "22b."->"24/9") with "\n", a
    completely different, legitimate reason for a newline to appear than Issue 101's own
    validated signal (a newline WITHIN one raw grid cell's own text, meaning "this is
    a worked solution, not a short value"), _is_prose_shaped_grid_answer() alone can't tell
    these apart, since it was only ever validated at the single-raw-cell level (Issue 101),
    not against this combined, joined-from-multiple-rows shape.

    Distinguishes this from a genuine multi-line worked solution: chij's own real folded-cell
    shape ALSO uses "a)"/"b)" markers, but only on the FIRST line of each sub-part, real
    working lines in between do NOT start with a marker (confirmed real example: "a) 60 - 48 =
    12\n12 ÷ 5 = 2.4L\nb) 3/8 → 48L\n...", only 2 of 4 real lines start with one). Requiring
    EVERY line to start with a marker is what correctly rejects that real shape while accepting
    the real combined-siblings one.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return bool(lines) and all(_COMBINED_SUBPART_LINE_PATTERN.match(ln.strip()) for ln in lines)


def _split_answer_and_worked_solution(answer_text: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """
    Part 3 (Issue 102): reuses Issue 101's own real, confirmed
    distinction between a short final-answer cell and a full worked-solution cell
    (_is_prose_shaped_grid_answer(), the newline-presence signal, 99.1% clean, 112/113 real
    grid cells) to split the single `answer_text` this file's existing grid lookup already
    produces into the two real fields ExtractedQuestion actually has for this
    (`answer_value`/`worked_solution_text`), rather than dumping a whole multi-line worked
    solution into `answer_value`, a field this project's own schema (Section 10.1) and every
    prior real usage (grid collision checks, sub-part combination, `values_equivalent()`
    comparisons) has always treated as a short value.

    Checks _is_combined_subpart_answer() FIRST, before the prose check, a real, necessary
    ordering found via this fix's own real end-to-end validation (see that function's own
    docstring for the full real story: PSLE Math 2019's own combined-lettered-siblings answers
    were being wrongly classified as prose by the newline signal alone, a real bug found and
    fixed before this was called done, not shipped).

    Returns (answer_value, worked_solution_text):
      - answer_text is None (no matched grid answer at all) -> (None, None), unchanged from
        before this fix.
      - answer_text is a combined-siblings compound short value ("(a) 21\n(b) 24/9") -> stored
        as answer_value UNCHANGED, worked_solution_text None, this is a compound SHORT value,
        not a worked solution, regardless of its embedded newlines.
      - answer_text is a short value (not prose-shaped, not combined-siblings) -> (answer_text,
        None), EXACTLY the existing behavior, byte-for-byte, per explicit instruction ("when
        the matched answer is already short, behavior stays exactly as it is now, nothing
        changes there").
      - answer_text is prose-shaped -> (distilled_or_None, answer_text). The full text
        always goes to worked_solution_text. For answer_value, this reuses
        _extract_grid_final_answer() (Issue 101's own real, validated extractor) to pull a
        distilled final value. Left None (not guessed) when no confident marker/trailing-"="
        value can be found, same fail-closed choice _grid_answers_agree() already makes with
        this same extractor.
    """
    if answer_text is None:
        return None, None
    if _is_combined_subpart_answer(answer_text):
        return answer_text, None
    if not _is_prose_shaped_grid_answer(answer_text, answer_text):
        return answer_text, None
    return _extract_grid_final_answer(answer_text), answer_text


# =================================================================================================
# segment_and_compare_page, Issue 98's real fix , replacing the
# question-page use of merge_page_with_agreement() + segment_questions() above. That combo is kept
# UNCHANGED and still correct for its one remaining real job, Topology A/B's answer-key-grid pages
# (topology_a's own answer_file loop, topology_b's reader_output_by_page dict), where a single
# usable whole-page RawReaderOutput really is the right unit: parse_answer_key_grid() reads a
# page's table regions structurally, it never compares two readers' prose against each other.
# Issue 100 replaced this function's OWN matched-pair comparison
# (originally values_equivalent(), Issue 99) with _text_similarity() above, see that function's
# own comment block for the real two-comparison-jobs distinction this was built around.
# =================================================================================================

def segment_and_compare_page(
    pdf_stem: str, page_index: int, question_id_prefix: str, raw_dir: Path,
    discrepancy_log: Optional[Path] = None,
    previous_page_was_grid: bool = False,
) -> tuple[list[tuple[QuestionSpan, Literal["high", "low"], dict]], Optional[RawReaderOutput]]:
    """
    Issue 98's real fix, the development log (confirmed the bug with a fresh, from-empty
 Stage A->B->C run) / (this fix). The OLD flow (merge_page_with_agreement() then
    segment_questions() once on its single "usable_output") decided agreement at PAGE granularity,
    before any question-level content existed to compare, compare_readers() was comparing the two
    readers' ENTIRE raw page text via values_equivalent(), a comparator built for short answer
    values ("1/4 vs 0.25"), not full free-text pages. Confirmed real consequence, not theoretical:
    a completely fresh run against the real 2-file/81-page subset produced agreement_status=
    "disagree" on 82 of 82 real questions, zero "agree", direct inspection of one such record
    (PSLE Math 2019:p28) showed both readers actually agreed on the real question content; the
    mismatch was page-level formatting/boilerplate noise (branding text, page numbers, whether a
    diagram gets an image placeholder) that has nothing to do with question-level agreement.

    NEW order, this function: segment BOTH readers' own normalized output into their own
    independent QuestionSpan lists FIRST (segment_questions() already supported this, it always
    took a single reader's own RawReaderOutput, confirmed before writing this function; nothing in
    its interface assumed pre-merged input, so it needed no change), THEN match spans across the
    two lists by question_number (both lists are already scoped to this one physical page, which
    the caller's own section-tracking already keys per-page, see _extract_question_page_section)
    and compare each MATCHED PAIR's own raw_text, not the whole page's.

    Real per-question ocr_confidence/ocr_record, not page-level reused across every span (the
    schema's own ocr_extraction_records table already has a question_id FK, true per-question
    granularity was always the intended shape; the page-level reuse this replaces was a documented,
    deliberate stopgap specifically because building this seemed to require re-running the readers
    per span, which it does not, segment_questions() is pure CPU parsing of ALREADY-SAVED Stage B
    output, not a new reader call, so real per-question comparison costs nothing extra to compute).

    Returns:
      - list of (QuestionSpan, ocr_confidence, ocr_record), one tuple per real question found on
        this page (union of both readers' segmentations), ocr_record now scoped to that
        one question's own primary/secondary text and agreement_status.
      - a representative RawReaderOutput (primary if it produced any real text, else secondary,
        else None), for the caller's OWN page-level needs unrelated to per-question comparison:
        _extract_question_page_section() (a real page-level signal, not a per-question one) and,
        in extract_topology_b, reader_output_by_page for locate_embedded_answer_section()'s grid
        lookup on non-question pages. Equivalent to the old usable_output in the common case
        (primary preferred, secondary fallback), computed here instead of via a second
        merge_page_with_agreement() call so the raw JSON isn't loaded and normalized twice per page.
    """
    primary_raw = load_raw_output("mineru25", pdf_stem, page_index, raw_dir)
    secondary_raw = load_raw_output("deepseek_ocr", pdf_stem, page_index, raw_dir)

    if primary_raw is None and secondary_raw is None:
        log_question_failure(question_id_prefix, "no Stage B output from either reader for this page")
        return [], None

    primary_norm = (
        RawReaderOutput(
            extracted_text=normalize_glyphs(primary_raw.extracted_text),  # 3a
            structure=primary_raw.structure,
            bounding_boxes=primary_raw.bounding_boxes,
            field_confidence=primary_raw.field_confidence,
        )
        if primary_raw is not None else None
    )
    secondary_norm = (
        RawReaderOutput(
            extracted_text=normalize_glyphs(secondary_raw.extracted_text),  # 3a
            structure=secondary_raw.structure,
            bounding_boxes=secondary_raw.bounding_boxes,
            field_confidence=secondary_raw.field_confidence,
        )
        if secondary_raw is not None else None
    )

    primary_empty = primary_norm is None or not primary_norm.extracted_text.strip()
    secondary_empty = secondary_norm is None or not secondary_norm.extracted_text.strip()

    if primary_empty and secondary_empty:
        log_question_failure(question_id_prefix, "both readers returned empty output")
        return [], None

    representative = primary_norm if not primary_empty else secondary_norm
    page = PageImage(source_pdf=Path(pdf_stem), page_index=page_index, image_bytes=b"")

    if primary_empty or secondary_empty:
        # Whole-reader page failure (the development log's single-reader fallback bug fix,
        # same real case, now expressed per resulting question instead of per page): only one
        # reader produced any usable text on this page at all, so there is no second segmentation
        # to compare against, every question found is single_reader_only.
        spans = segment_questions(
            page, representative, None, discrepancy_log, previous_page_was_grid,
        )
        using_primary = not primary_empty
        results = [
            (
                span,
                "low",
                {
                    "question_id": f"{question_id_prefix}:q{span.question_number}",
                    "primary_reader_output": span.raw_text if using_primary else None,
                    "secondary_reader_output": None if using_primary else span.raw_text,
                    "agreement_status": "single_reader_only",
                    "disagreement_detail": (
                        f"only {'MinerU2.5' if using_primary else 'DeepSeek-OCR'} produced usable "
                        f"output for this page"
                    ),
                    # Real bug found and fixed (development log, Issue 217): this dict
                    # never carried a disagreement_type key at all, unconditionally read by
                    # extract_topology_a/b's own grid-cross-check merge (Issue 213), a fresh
                    # Stage C run crashed with KeyError on every single_reader_only question,
                    # confirmed via a full corpus re-run, not caught by Issue 212's own unit
                    # tests (which only exercised compare_readers() directly, never this real,
                    # separate per-question comparison path). None is correct here, matching
                    # compare_readers()'s own single_reader_only -> None contract.
                    "disagreement_type": None,
                    "extracted_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            for span in spans
        ]
        return results, representative

    # Both readers produced real, non-empty page text, segment EACH independently. This is the
    # real fix: agreement is now judged per matched QUESTION, not by comparing the two readers'
    # entire page text before any question boundary exists.
    primary_spans = segment_questions(
        page, primary_norm, None, discrepancy_log, previous_page_was_grid,
    )
    secondary_spans = segment_questions(
        page, secondary_norm, None, discrepancy_log, previous_page_was_grid,
    )

    # Issue 229 fix, second attempt, real, narrow recovery for a
    # reader whose OWN segmentation found NOTHING on this page at all (empty spans). See
    # _recover_zero_padded_misread_questions()'s own docstring for the full real evidence and why
    # this is deliberately scoped to fire ONLY here, never touching _match_question_start()/
    # classify_page() (the first attempt's own real mistake, reverted after a confirmed regression
    # on a file that had nothing to do with zero-padded numbering at all).
    if not primary_spans:
        primary_spans = _recover_zero_padded_misread_questions(
            primary_norm.extracted_text, secondary_norm.extracted_text, page.page_index,
        )
    if not secondary_spans:
        secondary_spans = _recover_zero_padded_misread_questions(
            secondary_norm.extracted_text, primary_norm.extracted_text, page.page_index,
        )

    # Real, narrow fraction-numerator-phantom suppression (Issue 224's remaining gap, fixed
    # here, see _suppress_fraction_numerator_phantoms()'s own docstring). Applied to each
    # reader's own span list using the OTHER reader's whole-page text as the only real evidence
    # source (a reader that lost a fraction's denominator has no trace of it left in its OWN text
    # to check), BEFORE building the number dicts below, so a suppressed phantom never gets the
    # chance to become its own spurious single_reader_only record in the first place.
    primary_spans = _suppress_fraction_numerator_phantoms(primary_spans, secondary_norm.extracted_text)
    secondary_spans = _suppress_fraction_numerator_phantoms(secondary_spans, primary_norm.extracted_text)

    primary_by_number = {s.question_number: s for s in primary_spans}
    secondary_by_number = {s.question_number: s for s in secondary_spans}

    # Real cross-reader recovery pass (Issue 224 fix, development log 2026-09-0X), BEFORE the
    # per-number comparison loop below. See _find_recovered_span_text()'s own docstring for the
    # real root cause this targets. For any number present on only one side, check whether the
    # OTHER reader's own whole-page text contains a substantial match for that span's own content;
    # if so, synthesize a real QuestionSpan from the matched window and let it flow through the
    # SAME per-question value comparison below, exactly as if both readers had produced a
    # matching-numbered span, this only ever changes whether a genuine comparison is attempted,
    # never assumes agreement just because matching content was located. Recovered spans are
    # `flagged` (existing QuestionSpan field, same convention as this module's other real
    # human-review flags) so a recovered pair's own real provenance stays visible in its
    # `ocr_extraction_record`, not silently indistinguishable from a normally-matched pair.
    for number, primary_span in list(primary_by_number.items()):
        if not number or number in secondary_by_number:
            continue
        recovered_text = _find_recovered_span_text(primary_span.raw_text, secondary_norm.extracted_text)
        if recovered_text is not None:
            secondary_by_number[number] = QuestionSpan(
                question_number=number, raw_text=recovered_text,
                page_index=primary_span.page_index, page_label=primary_span.page_label,
                flagged="cross_reader_recovered",
            )
    for number, secondary_span in list(secondary_by_number.items()):
        if not number or number in primary_by_number:
            continue
        recovered_text = _find_recovered_span_text(secondary_span.raw_text, primary_norm.extracted_text)
        if recovered_text is not None:
            primary_by_number[number] = QuestionSpan(
                question_number=number, raw_text=recovered_text,
                page_index=secondary_span.page_index, page_label=secondary_span.page_label,
                flagged="cross_reader_recovered",
            )

    # Stable order: primary's own order first, then any secondary-only numbers appended.
    all_numbers = list(primary_by_number) + [
        n for n in secondary_by_number if n not in primary_by_number
    ]

    # Structural-fabrication signal (the development log's own PLACEHOLDER HEURISTIC) stays
    # page-scoped, computed once and applied to every matched pair on this page, NOT narrowed to
    # per-span here. QuestionSpan carries no per-question structure field to check this against
    # more precisely (only RawReaderOutput.structure, a whole-page dict, exists), and
    # _structure_looks_fabricated()'s own docstring already states it is a placeholder pending real
    # per-question structure data that does not exist yet, inventing fake per-span structure data
    # to force a narrower check would be a worse kind of guess than being honest that this one
    # sub-signal is still page-level. The real, confirmed bug this function fixes is the VALUE
    # comparison running on whole-page text; that part is now per-question below.
    page_structure_mismatch = _structure_looks_fabricated(primary_norm, secondary_norm)

    results = []
    for number in all_numbers:
        primary_span = primary_by_number.get(number)
        secondary_span = secondary_by_number.get(number)
        question_id = f"{question_id_prefix}:q{number}"

        if primary_span is not None and secondary_span is not None:
            # The real per-question comparison, both readers independently segmented a question
            # with this number on this page; compare THEIR OWN TEXT, not the whole page's, and with
            # the real tool for THIS job (Issue 100): text similarity, not values_equivalent()
            # (a numeric/semantic comparator built for short answer values, confirmed the wrong
            # tool for free-text prose, see this module's own comment above _text_similarity()).
            # Issue 228(a) fix : a recovered synthetic span's own
            # raw_text can still carry real cross-reader markup (see
            # _text_similarity_layout_normalized()'s own docstring), use the layout-normalized
            # comparison for THIS pair only when one side is such a span, never for a normal pair.
            is_recovered_pair = (
                primary_span.flagged == "cross_reader_recovered"
                or secondary_span.flagged == "cross_reader_recovered"
            )
            similarity = (
                _text_similarity_layout_normalized(primary_span.raw_text, secondary_span.raw_text)
                if is_recovered_pair
                else _text_similarity(primary_span.raw_text, secondary_span.raw_text)
            )
            similar_enough = similarity >= _TEXT_SIMILARITY_AGREEMENT_THRESHOLD
            # disagreement_type (Issue 217, this function's own real,
            # separate branch logic never fed compare_readers() at all, so Issue 212's fix never
            # reached it; found via a full corpus Stage C re-run, not by unit tests alone) mirrors
            # compare_readers()'s own value_mismatch/structural_mismatch/None contract, mapped onto
            # THIS function's own three real outcomes below.
            if similar_enough:
                if page_structure_mismatch:
                    agreement_status, disagreement_detail, disagreement_type = "disagree", (
                        f"text similarity {similarity:.3f} >= threshold "
                        f"{_TEXT_SIMILARITY_AGREEMENT_THRESHOLD}, but page-level table/grid "
                        f"structure mismatch between readers (one reports a table, the other does "
                        f"not) — flagged per the DeepSeek-OCR residual fabrication risk, not "
                        f"silently counted as agreement"
                    ), "structural_mismatch"
                else:
                    agreement_status, disagreement_detail, disagreement_type = "agree", None, None
            else:
                agreement_status, disagreement_detail, disagreement_type = "disagree", (
                    f"text similarity {similarity:.3f} < threshold "
                    f"{_TEXT_SIMILARITY_AGREEMENT_THRESHOLD}: primary={primary_span.raw_text!r} "
                    f"secondary={secondary_span.raw_text!r}"
                ), "value_mismatch"
            # primary's own text is the recorded value either way, same "disagreement is what
            # drives confidence, not which reader's text is kept" precedent as the old page-level
            # logic. Its own segmenter-time flag is kept; secondary's is folded in too if present
            # and different, rather than silently dropped.
            flags = [f for f in (primary_span.flagged, secondary_span.flagged) if f]
            chosen_span = (
                primary_span if not flags or flags == [primary_span.flagged]
                else QuestionSpan(
                    question_number=primary_span.question_number, raw_text=primary_span.raw_text,
                    page_index=primary_span.page_index, page_label=primary_span.page_label,
                    flagged="; ".join(dict.fromkeys(flags)),
                )
            )
        elif primary_span is not None or secondary_span is not None:
            chosen_span = primary_span if primary_span is not None else secondary_span
            reader_name = "MinerU2.5" if primary_span is not None else "DeepSeek-OCR"
            agreement_status, disagreement_detail, disagreement_type = "single_reader_only", (
                f"question {number!r} segmented by {reader_name} only on this page — the other "
                f"reader's own independent segmentation of this page produced no matching "
                f"question_number"
            ), None
        else:
            continue  # unreachable, number came from one of the two dicts above

        if agreement_status == "disagree":
            logger.warning(
                "FIRST-CLASS FINDING — real per-question OCR disagreement, question_id=%s: %s. "
                "Log this in docs/DEVELOPMENT_LOG.md per the development log's standing "
                "instruction (Issue 98's fix narrowed comparison scope to per-question — a "
                "disagreement found here is a real, question-level one, not the whole-page-text "
                "artifact Issue 98 confirmed and fixed).",
                question_id, disagreement_detail,
            )

        ocr_confidence: Literal["high", "low"] = "high" if agreement_status == "agree" else "low"
        ocr_record = {
            "question_id": question_id,
            "primary_reader_output": primary_span.raw_text if primary_span is not None else None,
            "secondary_reader_output": secondary_span.raw_text if secondary_span is not None else None,
            "agreement_status": agreement_status,
            "disagreement_detail": disagreement_detail,
            "disagreement_type": disagreement_type,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
        }
        results.append((chosen_span, ocr_confidence, ocr_record))

    return results, representative


# =================================================================================================
# Provenance and topology helpers (Step 3 / Step 4 / Section 3.4 / Issue 22)
# =================================================================================================

def find_matching_answer_file(pdf_path: Path) -> Optional[Path]:
    """
    Section 3.4 / Issue 71: Topology A detection. Looks for a same-titled sibling file
    containing "Answer" (e.g. "PSLE Math 2014 Answer+Solution.pdf" beside
    "PSLE Math 2014.pdf"). Returns None if no such file exists (=> Topology B).

    Issue 97 : this function does its own independent sibling glob
    (`pdf_path.parent.glob(...)`), NOT routed through discover_pdfs(), EXCLUDED_SOURCE_FILES
    (Issue 96) is checked explicitly here too, not assumed inherited. Without this, a future
    excluded file whose name happens to contain "answer" could get silently picked up as some
    OTHER paper's matched answer_file, completely bypassing the exclusion, the current single
    excluded entry (P6_Maths_2024_SA2_taonan.pdf) doesn't trigger this today (no "answer" in its
    own stem), but that was a property of that one entry, not a property of this function, so it
    is not something a future entry could safely rely on. The check belongs everywhere a file can
    enter the pipeline, not just discover_pdfs()'s own top-level scan.
    """
    if pdf_path.name in EXCLUDED_SOURCE_FILES:
        # pdf_path itself is excluded, should never have reached here if the caller went through
        # discover_pdfs() (which already filters it out), but guard here too in case this function
        # is ever called directly, same defensive reasoning as the "answer" guard just below.
        return None

    if "answer" in pdf_path.stem.lower():
        # pdf_path IS itself an answer file, should not be independently discovered as a
        # question source. discover_pdfs() below filters these out before this is called, but
        # guard here too in case this function is ever called directly.
        return None

    candidates = sorted(pdf_path.parent.glob(f"{pdf_path.stem}*"))
    for candidate in candidates:
        if candidate == pdf_path:
            continue
        if candidate.name in EXCLUDED_SOURCE_FILES:
            logger.warning(
                "find_matching_answer_file: %s matched %s by name, but it's in "
                "EXCLUDED_SOURCE_FILES (Issue 96) — not returning it as a valid answer_file.",
                pdf_path, candidate,
            )
            continue
        if "answer" in candidate.stem.lower():
            return candidate
    return None


_CANONICAL_SCHOOL_MAP: Optional[dict[str, dict[str, str]]] = None  # lazy module-level cache


def normalized_school_name(pdf_path: Path) -> tuple[Optional[str], bool]:
    """
    Section 3.2 / Issue 61: delegate to the reviewed Stage 1 + Stage 2 output of
    tools/normalize_filenames.py (Phase 0, step 6) rather than re-deriving normalization logic
    here, that tool's human-reviewed mapping is the source of truth for ambiguous school-name
    variants (e.g. mgs vs methodistgirls), which must never be auto-merged by a fresh heuristic.

 WIRING NOTE: the original draft imported a `normalized_school_name` function
    from tools.normalize_filenames, that function does not exist there. The tool's real API is
    `normalize_stage1(filename) -> key` + `load_canonical_map(path) -> {key: {canonical_school_name,
    verified, notes}}`, confirmed by reading the real file. Wired to that real API below, not
    guessed. Cache is loaded once per process, not per call, tools/school_name_canonical.csv is a
 small, static, human-reviewed file (39 entries), not something that changes
    mid-run.

    Returns (canonical_name_or_None, verified). Issue 93 : a
    canonical_school_name being present in the store means SOME human wrote it down, either a
    real, evidence-checked confirmation, or a same-day Stage 2 review guess that has not actually
    been checked against the real source PDFs yet (that gap was exactly the problem, the
    `needs_human_review` flag on school_name_review.csv used to be purely documentary, read
    identically to a verified entry by every downstream consumer). `verified` surfaces that
    distinction for real: True only for canonical entries whose `verified` column is `true` in
    tools/school_name_canonical.csv, set by hand after direct evidence (e.g. rendering and reading
    the actual cover pages), same discipline as the raffles/rafflesgirls and 12-group verification
    round this fix shipped alongside. False (not blocked) for the rest, see the call site in
    extract_topology_a/b for why this project tags rather than refuses unverified groups.
    """
    global _CANONICAL_SCHOOL_MAP
    try:
        from tools.normalize_filenames import (  # type: ignore
            CANONICAL_MAP_PATH,
            load_canonical_map,
            normalize_stage1,
        )
    except ImportError:
        logger.warning(
            "tools.normalize_filenames not importable — school_name left as None for %s. "
            "Confirm Phase 0 step 6 has been run and produced the reviewed mapping.",
            pdf_path,
        )
        return None, False

    if _CANONICAL_SCHOOL_MAP is None:
        _CANONICAL_SCHOOL_MAP = load_canonical_map(CANONICAL_MAP_PATH)
        if not _CANONICAL_SCHOOL_MAP:
            logger.warning(
                "tools/school_name_canonical.csv loaded empty or missing (%s) — every "
                "school_name will be None this run. Confirm Phase 0 step 6's --commit has "
                "actually been run on this checkout.",
                CANONICAL_MAP_PATH,
            )

    key = normalize_stage1(pdf_path.name)
    entry = _CANONICAL_SCHOOL_MAP.get(key)
    if entry is None:
        # Issue 61: never guess an unresolved school name, flag it, don't auto-merge.
        logger.warning(
            "No canonical school name resolved for key %r (from %s) — not in "
            "tools/school_name_canonical.csv. Needs human review via "
            "tools/normalize_filenames.py --real, not a guess here.",
            key, pdf_path,
        )
        return None, False

    name = entry["canonical_school_name"] or None
    verified = bool(entry.get("verified")) if name else False
    if name and not verified:
        # Issue 93: this is the real, operational half of the fix, a group can be committed
        # to the canonical store (so extraction proceeds, see extract_topology_a/b) without a
        # human having confirmed it against the real source PDFs yet. Surface that loudly here,
        # not just in the CSV's notes column, so it shows up in real run logs too.
        logger.warning(
            "school_name %r for key %r (from %s) is UNVERIFIED — canonical store has a name but "
            "tools/school_name_canonical.csv's `verified` column is not true for this key. "
            "Every question from this file will be tagged school_name_verified=False.",
            name, key, pdf_path,
        )
    return name, verified


def render_pdf_page(pdf_path: Path, page_index: int, dpi: int = 300) -> PageImage:
    """
    Real implementation via PyMuPDF (fitz), standard, deterministic utility, not a guessed model
    signature. Confirm PyMuPDF is the library actually pinned in backend/pyproject.toml (Section
    15, Phase 0, step 4); swap the two lines below for pdf2image or another library if a
    different one was pinned instead.
    """
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    page = doc[page_index]
    zoom = dpi / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    image_bytes = pix.tobytes("png")
    doc.close()
    return PageImage(source_pdf=pdf_path, page_index=page_index, image_bytes=image_bytes, dpi=dpi)


# =================================================================================================
# segment_questions(), region-based redesign . Replaces the naive
# flat-text regex that was confirmed broken three separate real ways: GS004 (real numbering "28
# Amy bought...", no period, the old regex required one), GS005 (an answer-key grid, table-shaped
# content, no bare "N. " line-start pattern exists at all), and the real corpus run (captured a
# cover page's own numbered instructions, "1. Write your Index No...", as if it were question 1).
#
# Designed against real region data pulled from the bake-off's raw MinerU2.5 output before writing
# any of this (GS001-GS006, plus the real subset's own cover/question pages), not against assumed
# type values. Real "type" vocabulary confirmed present: header, footer, page_number, title,
# table_caption, table, list, image, text. Two things learned from that pull that shaped this
# design directly:
#   - "list"-type regions carry content: null always, in every real example seen, they're a
#     structural grouping annotation over already-captured "text" regions (their bbox literally
#     overlaps the text regions they group), not their own content. Skipped entirely, never
#     treated as a question-boundary signal.
#   - GS004's own leading question number ("28") is ABSENT from every region on that
#     page, not merged into an adjacent region, not hiding in a "list" region, just never OCR'd
#     by MinerU2.5 at all. No design of this segmenter can recover a digit the reader never
#     produced; handled by inferring the number from the immediately-following confirmed question
#     number when unambiguous, logged clearly as inferred rather than silently presented as
#     directly observed (see _segment_from_regions below).
# =================================================================================================

_MIN_QUESTION_CONTENT_CHARS = 15  # filters short MCQ options ("1) 314000") and short
                                   # administrative lines that happen to start with a number,
                                   # without hand-listing every possible instruction phrase , 
                                   # confirmed against real GS001 MCQ options during design.

# Region types confirmed, across every real page inspected (GS001-GS006, the real subset's cover
# and question pages), to never carry real question body content, a whitelist of the types that
# DO, not a blacklist of noise types, so an unrecognized future type defaults to "not body
# content" rather than risking a new noise type being swept into question text.
#
# image_caption/image_footnote added after the full-subset spot-check 
# found chij.pdf p14's Q20 ("20. The graph shows the number of children...") living entirely in an
# image_caption region, a diagram-anchored question whose stem is literally captioned under the
# diagram, and chij.pdf p35's Q(a) similarly inside image_footnote. Both types are confirmed
# MIXED, not purely question content: image_caption also carries repeated cover-page branding
# ("CHIJ ST NICHOLAS GIRLS' SCHOOL (PRIMARY)") and bare diagram axis labels ("X", "Y"), and
# image_footnote otherwise only ever carries "Do not write in this space" (already excluded via
# _MARGIN_NOTE_PATTERNS). Safe to include because _MIN_QUESTION_CONTENT_CHARS now also guards
# provisional-span and bare-number-consumption starts below, not just explicit "N. " matches, a
# short label like "X" can no longer spawn a fake standalone span.
#
# equation added (Issue 107, Issue 103 finding (b)), a real
# MinerU2.5 region type never seen in the original 2-file validated subset (0 occurrences there),
# but confirmed real and common across the wider 22-file/837-page batch (556 real occurrences,
# every one of them a genuine fraction/ratio/algebra fragment MinerU2.5 typeset as its own LaTeX
# region instead of folding into the surrounding "text" region, real examples pulled directly
# from the raw batch output, not invented: Henry-Park p4 `\[\frac{4}{3}, \frac{1}{5}, \frac{5}{4}\]`
# (a fraction-comparison question's own stem), Ai-Tong p3 `\[10:\square]=15:6\]` (a ratio question
# with a real fill-in-the-blank box), Nan-Hua p3 `\[15m+12-9m-8\]` (an algebraic-simplification
# question's own expression) and p9 `\[\frac{8}{12}=\frac{\square}{9}\]` (an equivalent-fraction
# fill-in). Before this fix, every one of these was silently dropped from the segmented question
# text, real, solve-critical content missing from what the student-facing question actually
# said, not a cosmetic gap. Unlike image_caption/image_footnote, "equation" was checked and found
# NOT mixed with unrelated noise in every real occurrence inspected, confirmed via a full re-scan
# of all 556 real occurrences batch-wide before shipping this addition (see Issue 107).
#
# code added (Issue 109, reversing Issue 108's initial exclusion
# after review), one real occurrence across the whole batch (`PSLE Math 2015
# Answer+Solution.pdf` p7), pulled and read directly, not described: a genuine, complete
# worked-solution passage ("...Yan made 8.1l of drink."), not noise, by the identical evidentiary
# standard applied to equation above. Held out initially over a sample-size concern (n=1 vs.
# equation's 556) that does not actually weigh against THIS occurrence being real content, the
# same standard (read it, is it noise or not) applies regardless of how many other real
# occurrences exist to cross-check against. Real caveat, not a reason for exclusion: this
# occurrence's own page currently sits inside Issue 103's still-open narrative-answer-booklet
# gap (consumed only as an answer_file, never independently extracted as its own "questions"), so
# including "code" here has zero observed effect on any of the 22 real extracted files' own
# output today, confirmed empirically, not assumed (see Issue 109). Kept in the set anyway,
# on the same real-content-not-noise basis as equation, so it's already correctly handled the
# moment Issue 103's narrative-booklet gap is fixed, rather than needing a second pass here.
_BODY_CONTENT_REGION_TYPES = {"text", "table", "image_caption", "image_footnote", "equation", "code"}

# Cover/instructions-page signal, keywords confirmed present on real cover pages this session
# (chij's own cover: "Name:", "Class:", "INSTRUCTIONS TO CANDIDATES"; PSLE Math 2019's cover:
# "Index No.", "MINISTRY OF EDUCATION", numbered instructions). Requiring 2+ hits, not 1, so a
# single incidental match (e.g. a real question mentioning "class" in an unrelated sense) doesn't
# misclassify a genuine question page.
#
# Issue 227(a) fix (the development log, 621-record single_reader_only investigation): a
# real, confirmed gap, none of the keywords above ever match `P6_Maths_2022_SA2_nanchiau`'s own
# real cover-page template (its own real MinerU2.5 text checked directly on p0/p8/p20, not
# guessed), so its 3 cover pages fall through to `question_page` and their real instruction lines
# get segmented as spurious "question 0/1/2/3". Two more keywords added below, each a real,
# confirmed substring of that exact template (both appear on a single real text line, so a plain
# lowercase substring check finds them, nanchiau's own "Instructions\nto students" header spans
# two separate MinerU2.5 text regions joined by a newline and was deliberately NOT added as a
# keyword for that reason, it would never match): "name / index" (from the real cell "Name /
# Index #"), "this paper consists of" (from "This paper consists of N pages altogether.", present
# on all 3 real pages, sibling of the existing "this booklet consists of"). Together these already
# give nanchiau's real p0/p8/p20 pages 2+ hits each, confirmed directly against the real text of
# all three.
#
# A THIRD candidate keyword, "duration for booklet" (from "Duration for Booklets A and B"), was
# tried and REVERTED after full-corpus validation, same standard as the Issue 224 revert.
# It correctly gave a second real hit for `P6-Maths-2021-SA1-Nanyang` p0/p11 (genuine, real,
# standalone cover pages this fix would otherwise still miss), but ALSO produced a real, confirmed
# false positive: `P6_Maths_2022_SA2_nanyang` p44 is NOT a cover page, it's a real, if heavily
# OCR-corrupted, question page (real questions 13-18 on it) that happens to carry a mid-page
# booklet-transition administrative block ("Total Duration for Booklets A and B: 1 hour", "Do not
# turn over this page...") BETWEEN two sets of real questions, not on a page of its own. Adding
# this keyword would have wrongly reclassified that whole page `cover_page`, silently discarding
# 11 real (if corrupted) question records. `_COVER_PAGE_KEYWORDS` operates at whole-PAGE
# granularity and cannot distinguish "this whole page is a cover page" from "this page contains a
# cover-page-shaped block mixed with real questions", a real, structural limitation, not a
# threshold to retune. Left unfixed rather than shipped with a confirmed corruption risk; the
# Nanyang SA1 2021 p0/p11 case this would have also fixed remains open (not attempted here).
_COVER_PAGE_KEYWORDS = [
    "index no", "name:", "class:", "instructions to candidates",
    "do not turn over this page", "do not open this booklet", "for examiner's use",
    "total time for", "this booklet consists of", "ministry of education",
    "primary school leaving examination",
    "name / index", "this paper consists of",
]
_COVER_PAGE_MIN_KEYWORD_HITS = 2

# Answer-key-grid signal, confirmed against real GS003 ("ANSWER KEY") and GS006 ("ANSWER KEYS")
# title-typed regions specifically, the most precise real signal found (not just any text
# mentioning "answer", a real question can say "answer" too).
_ANSWER_KEY_TITLE_KEYWORDS = ["answer key", "answer keys"]

# Standard PSLE boilerplate fragments, each confirmed as a real, RECURRING MinerU2.5 'text'
# region during the real 2-file subset validation run, not a guessed
# or hand-built list. Excluded before any of them can be folded into a provisional question span,
# which without this produced either a fabricated question with no real content ("Do not write in
# this space" standing in as a fake "Q20"), or a real question's text polluted with boilerplate
# glued to the front ("Questions 1 to 10 carry..."/"(20 marks)" both preceding the real Q1). Three
# specific, confirmed-recurring patterns, not an attempt to hand-list every possible administrative
# phrase a real paper might contain, the same principle _MIN_QUESTION_CONTENT_CHARS already
# applies to short MCQ options, extended to the small number of boilerplate shapes actually
# observed rather than guessed in advance.
_SECTION_INSTRUCTION_PATTERN = re.compile(r"^(?:for\s+)?questions?\s+\d+\s+to\s+\d+\b", re.IGNORECASE)
# Broadened from an earlier, narrower "...carry \d+ marks" version after finding a second real
# phrasing of the same section-header boilerplate family in this same subset ("For questions 6 to
# 17, show your working clearly...", distinct wording from "Questions 21 to 30 carry 2 marks
# each..." but the same kind of sentence). The common, distinctive, safe signal across both real
# variants: it's ABOUT a question-number range, not itself a question, a real question would not
# start with "Questions N to M" or "For questions N to M" as its own opening content.
_MARGIN_NOTE_PATTERNS = [
    re.compile(r"^do not write in this space$", re.IGNORECASE),  # confirmed on GS004 AND this
                                                                    # subset's own PSLE Math 2019
                                                                    # p10, same exact phrase,
                                                                    # standard PSLE boilerplate
    re.compile(r"^do not write in this column$", re.IGNORECASE),  # Issue 227(b) fix (Decision
                                                                    # Log 621-record
                                                                    # investigation): a real,
                                                                    # confirmed sibling phrase to
                                                                    # the one above, checked
                                                                    # directly against
                                                                    # Singapore-Chinese-Girls p22's
                                                                    # own real MinerU2.5 text , 
                                                                    # "Do not write in this column"
                                                                    # on its own line, standing in
                                                                    # as a fake "question 5" before
                                                                    # this fix, exactly the same
                                                                    # mechanism the "space" pattern
                                                                    # above already covers for the
                                                                    # more common wording
    re.compile(r"^\(\s*\d+\s*marks?\s*\)$", re.IGNORECASE),  # "(20 marks)", a paper-section
                                                                # total, not a question's own text
    re.compile(r"^marks\s*:?\s*$", re.IGNORECASE),  # "MARKS:", confirmed on chij p12 and p20: a
                                                        # marks-tally box in the margin, OCR'd as
                                                        # its own region, immediately preceded by a
                                                        # bare number region (the marks value, e.g.
                                                        # "2" or "10"). Without this, the bare-number
                                                        # region got mistaken for a detached QUESTION
                                                        # number (the same real MinerU2.5 pattern
                                                        # that legitimately precedes GS004-style
                                                        # numbers, see pending_bare_number below),
                                                        # producing a fabricated question whose only
                                                        # "content" was the literal text "MARKS:".
    re.compile(r"^[(\{]?\s*[gd]o\s+on\s+to\s+(?:the\s+next\s+page|book(?:l)?et\s*[ab]\b)", re.IGNORECASE),
    # "(Go on to the next page)" / "Go on to Booklet B", real, confirmed recurring PSLE
    # page-footer navigation text (development log 2026-09-0X, Issue 224 fix). Real, confirmed
    # cause: Rosyth CA1 2021 pp3-7 all end in a real bare page-footer numeral MinerU2.5 emits as
    # its own detached region (identical real mechanism to GS004/chij's genuine bare-number
    # labels above, structurally indistinguishable from content alone), immediately followed by
    # this exact navigation phrase, without this, that phrase becomes the bare number's
    # "content", fabricating a phantom question (real confirmed case: p7's page-footer "7"
    # became a fake "question 7" whose only content was "Go on to Booket B\nScanned with
    # CamScanner..."). `book(?:l)?et` tolerates the real observed OCR typo "Booket" (missing
    # the "l") alongside the correctly-spelled "Booklet", both confirmed real spellings in this
    # corpus, not a guessed tolerance. `[gd]o` similarly tolerates a second real, confirmed OCR
    # typo, "Do on to the next page" (Catholic-High's own real papers, "Go" misread as "Do", a
    # plausible G/D letterform confusion), alongside the correctly-spelled "Go on to"; both
    # confirmed real spellings, not a guessed tolerance. `[(\{]?` similarly tolerates a real,
    # confirmed curly-brace variant ("{Go on to the next page}", also Catholic-High) alongside the
    # more common parenthesized form, real OCR punctuation-substitution noise, not guessed.

    # Issue 236 fix, real, confirmed sibling margin-note/instruction
    # phrases found during a direct, corpus-wide re-verification of Issue 172's own 52-record
    # "boilerplate misattached to a real answer_value" bucket: each of the phrases below is the
    # REAL, ENTIRE `question_text` of a currently-live, wrongly-answerable corpus row (not a
    # guessed addition), the exact same real failure shape the patterns above already fix for
    # their own phrases, just for real wording this session's earlier passes hadn't yet found.
    # PREFIX matches (no trailing `$`) are used where the real corpus shows more than one genuine
    # real ending for the same real opening phrase (confirmed by direct query, not assumed), a
    # full match would incorrectly miss the longer real variant.
    re.compile(r"^do not turn over this page", re.IGNORECASE),  # "...until you are told to do
                                                                    # so.", real, confirmed exact
                                                                    # corpus text (ids 467/488)
    re.compile(r"^follow all instructions carefully\.?$", re.IGNORECASE),  # real corpus text (ids
                                                                           # 468/1654/2116/1673) , 
                                                                           # capitalization varies
                                                                           # ("Instructions" in one
                                                                           # real instance), IGNORECASE
                                                                           # already tolerates that
    re.compile(r"^answer all questions\.?$", re.IGNORECASE),  # real corpus text (ids
                                                                  # 469/1655/1674)
    re.compile(r"^write your answers in", re.IGNORECASE),  # two real, distinct full endings
                                                              # confirmed in this exact corpus:
                                                              # "...in this booklet." (id=491) and
                                                              # "...in the spaces provided. For
                                                              # questions which require units, give
                                                              # your answers in the units stated."
                                                              # (id=4730), a shared real prefix,
                                                              # deliberately not anchored at the end
    re.compile(r"^please do not write in the margin\.?$", re.IGNORECASE),  # real corpus text
                                                                              # (ids 1495/1499/1519)
    re.compile(r"^use a dark blue or black ballpoint pen", re.IGNORECASE),  # real corpus text
                                                                               # (id=2118), the
                                                                               # question-side sibling
                                                                               # of the answer-key-side
                                                                               # instruction this
                                                                               # project's own real
                                                                               # exam papers print
    re.compile(r"^all diagrams in this paper are not drawn to scale", re.IGNORECASE),  # real
        # corpus text (ids 1374/2883/2893/3073/3083/3939/3962/4176/4199/4802/5085), the SAME real
        # phrase `_COVER_PAGE_KEYWORDS` already uses for whole-PAGE cover-page classification
        # (development log 2026-08-2X), reused here for a DIFFERENT real purpose: this
        # phrase can ALSO appear as its own detached per-REGION margin note on an otherwise real
        # question page (not just as one signal among several on a genuine cover page), where it
        # was fabricating its own phantom "question", the two mechanisms operate at different
        # granularity (whole-page vs. single-region) and do not conflict; a page already correctly
        # classified `cover_page` via the existing keyword-hit-count check never reaches
        # `_is_recognized_boilerplate()` at all.
]

# Shared-stimulus signal (the development log follow-up, Bug 3 fix), a real, recurring PSLE
# sentence shape that introduces context shared by TWO upcoming questions, confirmed in three real
# instances across both subset files, wording varying only in "Question"/"Questions" and the
# specific numbers: "Use the information below to answer Question 9 and 10." (chij p6), "...answer
# Questions 5 and 6." (PSLE Math 2019 p2), "...answer Question 21 and 22." (chij p15). Without
# recognizing this specific phrasing, the stimulus paragraph (and any supporting description/table
# between it and the first of its two questions) falls through to the same provisional-span
# machinery GS004's unnumbered content uses, and gets inferred a question_number equal
# to (next_confirmed - 1), which collides with an already-used number when, as here, the stimulus
# precedes TWO questions rather than continuing the ONE question right before it. Captures both
# referenced numbers directly from the sentence itself, rather than guessing from position.
_SHARED_STIMULUS_PATTERN = re.compile(
    r"^use the information below to answer questions?\s+(\d{1,3})\s+and\s+(\d{1,3})\b",
    re.IGNORECASE,
)

# =================================================================================================
# Section-label extraction (the development log follow-up, Issue 3 real fix), the QUESTION
# side's half of a real signal the ANSWER-KEY side (parse_answer_key_grid's section_label) already
# had. Two different real shapes confirmed on real question pages before writing either
# pattern, not assumed:
#   - chij: the section name is printed as its own plain text region ("Booklet A"/"Booklet B"/
#     "Paper 2") ONLY on that section's own cover page (classify_page's own cover_page case),
#     never repeated on the actual question pages that follow it, requires the same
#     "carry forward the last real value seen" pattern parse_answer_key_grid() already uses for
#     section_label across its own caption-less continuation pages, applied here by the caller
#     (extract_topology_a/b's own per-page loop) across cover-less question pages instead.
#   - PSLE Math 2019.pdf: a real printed paper CODE in a `footer` region on EVERY question page,
#     confirmed real: "0008/1(A)/2019" (Booklet A), "0008/1(B)/2019" (Booklet B), "0008/2/2019"
#     (Paper 2), present everywhere, no carry-forward needed for this shape specifically.
# Both are normalized to the SAME small canonical vocabulary ("BOOKLET A"/"BOOKLET B"/"PAPER 2")
# so a question's section and a grid row's section can be compared with plain string equality,
# regardless of which real surface form either side's signal came from.
# =================================================================================================

_SECTION_KEYWORD_PATTERNS = [
    (re.compile(r"\bbooklet\s*a\b", re.IGNORECASE), "BOOKLET A"),
    (re.compile(r"\bbooklet\s*b\b", re.IGNORECASE), "BOOKLET B"),
    (re.compile(r"\bpaper\s*2\b", re.IGNORECASE), "PAPER 2"),
    (re.compile(r"\bpaper\s*1\b", re.IGNORECASE), "PAPER 1"),  # fallback only, no real subset
                                                                  # page confirmed to need this on
                                                                  # its own (chij/PSLE both also
                                                                  # print "Booklet A"/"Booklet B"
                                                                  # for Paper 1's own two parts),
                                                                  # kept in case a future real file
                                                                  # has an undivided Paper 1.
]

# PSLE Math 2019.pdf's real printed paper-code shape, confirmed on every one of its question
# pages' own footer region: "0008/1(A)/2019", "0008/1(B)/2019", "0008/2/2019", the digits before
# and after are a real, stable paper/year code, not part of what's being matched here.
_PAPER_CODE_BOOKLET_PATTERN = re.compile(r"\d+/1\((A|B)\)/\d+", re.IGNORECASE)
_PAPER_CODE_PAPER2_PATTERN = re.compile(r"\d+/2/\d+")


def _normalize_section_label(raw: str) -> Optional[str]:
    """Maps any real observed section-label surface form to the shared canonical vocabulary
    ("BOOKLET A"/"BOOKLET B"/"PAPER 2"/"PAPER 1"), or None if raw doesn't look like a section
    label at all. Shared by both the question side and parse_answer_key_grid()'s grid side so
    they can never independently drift into different canonical spellings."""
    for pattern, canonical in _SECTION_KEYWORD_PATTERNS:
        if pattern.search(raw):
            return canonical
    return None


def _extract_question_page_section(bounding_boxes: Optional[list], extracted_text: str) -> Optional[str]:
    """
    Real, confirmed section signal on a QUESTION page (the development log follow-up), see
    the module comment block above for the two real shapes this checks, in order:
      1. PSLE Math 2019.pdf's real footer paper-code (present on every question page directly).
      2. chij's real plain-text section name (present only on cover pages, the caller is
         responsible for carrying this forward across the question pages that follow, the same
         way parse_answer_key_grid() already does for the grid side).
    Returns the normalized canonical label, or None if this specific page carries neither real
    signal, a real, expected case (most of chij's own question pages), not an error.
    """
    for bb in bounding_boxes or []:
        if bb.get("type") != "footer":
            continue
        content = bb.get("content") or ""
        m = _PAPER_CODE_BOOKLET_PATTERN.search(content)
        if m:
            return f"BOOKLET {m.group(1).upper()}"
        if _PAPER_CODE_PAPER2_PATTERN.search(content):
            return "PAPER 2"

    # Two passes, not one: chij's own real cover page has BOTH "Paper 1" and "Booklet A" as
    # separate text regions, with "Paper 1" appearing FIRST in reading order (confirmed real,
    # chij p0), a single first-match scan would wrongly return the less-specific "PAPER 1"
    # instead of the real, more specific "BOOKLET A" that follows it. Booklet A/B and Paper 2
    # (specific, always correct when present) are checked in full before "Paper 1" (the
    # unconfirmed fallback) is ever considered.
    text_regions = [
        bb.get("content") or "" for bb in (bounding_boxes or []) if bb.get("type") in ("title", "text")
    ]
    for content in text_regions:
        for pattern, canonical in _SECTION_KEYWORD_PATTERNS[:3]:  # Booklet A/B, Paper 2 only
            if pattern.search(content):
                return canonical
    for content in text_regions:
        if _SECTION_KEYWORD_PATTERNS[3][0].search(content):  # Paper 1 fallback
            return _SECTION_KEYWORD_PATTERNS[3][1]

    return None


def _is_recognized_boilerplate(content: str) -> bool:
    if _SECTION_INSTRUCTION_PATTERN.match(content):
        return True
    return any(p.match(content) for p in _MARGIN_NOTE_PATTERNS)


# A detached question-number label, its own region with no other content, real MinerU2.5 pattern
# (module docstring). Optional trailing "." or ")" (the development log, Bug 2 fix): the
# original punctuation-free-only version missed a real, confirmed shape (chij p37: MinerU2.5
# emitted "14.", WITH the period, as its own bare region, distinct from GS004's punctuation-free
# "28"). Shared by _segment_from_regions (which consumes it) AND _has_question_like_content below
# (which must agree with the segmenter on what counts as question-like, per that function's own
# docstring principle, a bare "14." page must not be classified 'other' and skipped before
# segmentation ever gets a chance to use it).
_BARE_NUMBER_LABEL_PATTERN = re.compile(r"(\d{1,3})[.)]?")


# Issue 229 candidate 2 (the development log, deep-dive commit 1fa0f0a), a "Q<1-2 digits>"
# question-start pattern (real, confirmed root cause: MinerU2.5 misreads `P6_Maths_2024_WA2_
# rosyth` p35's real zero-padded numbering, "01)"/"02)"/etc, as "Q1)"/"Q2)"/etc, "0" read as "Q")
# was DESIGNED, IMPLEMENTED, and REVERTED after full-corpus validation found a real, confirmed
# regression, same "stop, don't ship that piece" standard as the Issue 224/227 reverts. The
# real pattern's own true corpus-wide footprint turned out far broader than the single targeted
# page (253 real matches across 29 files, not just rosyth), and on at least one of them
# (`P6_Maths_2023_WA2_Taonan` p32/p33) it collided with a PRE-EXISTING, unrelated bare-digit
# false-positive match for the SAME number ("11 units -> 66 jugs" mistaken for a "question 11"
# start by the existing digit pattern, unrelated to this fix), the dict-based `{question_number:
# span}` matching downstream silently kept only the LAST of the two same-numbered spans, dropping
# the other's real content entirely. Confirmed real, concrete harm, not theoretical: 4 real records
# on this one file flipped from a correct `agree` to `disagree` (exhaustive full-corpus diff, not a
# sample). Not shipped. See the development log's own Issue 229 resolution entry for the full real
# validation trace (the 5 genuine, real improvements this same fix produced elsewhere, and why the
# one confirmed regression was still enough to withhold it, per the explicit standard this task was
# held to).


def _match_question_start(content: str) -> Optional["tuple[str, str]"]:
    """
    Permissive question-number pattern: "N.", "N)", or bare "N " followed by whitespace, not
    just "N.", the old pattern's exact failure on GS004's real "28 Amy bought..." (no period).
    Requires real content after the number (>= _MIN_QUESTION_CONTENT_CHARS), confirmed this
    correctly rejects real MCQ options ("1) 314000", 6 chars after the number) as new questions
    while still accepting real question openers ("1. Round off 314 678 to the nearest thousand").
    Returns (question_number, rest_of_content) or None.
    """
    m = re.match(r"^\s*(\d{1,3})[.)]?\s+(.+)$", content.strip(), re.DOTALL)
    if not m:
        return None
    number, rest = m.group(1), m.group(2).strip()
    if len(rest) < _MIN_QUESTION_CONTENT_CHARS:
        return None
    return number, rest


def _has_question_like_content(bounding_boxes: Optional[list], extracted_text: str) -> bool:
    """Does this page have at least one region/line matching a real question start? Reuses
    _match_question_start AND _BARE_NUMBER_LABEL_PATTERN so "classified as a question page" and
    "segment_questions() actually finds something" can't silently disagree with each other, a
    bare detached number region (e.g. "14.", chij p37, the development log, Bug 2 fix) is a
    real question-start signal to _segment_from_regions, so it must count here too, or the page
    gets classified 'other' and segmentation never runs at all despite being able to recover it.

    Also reuses _has_worked_solution_table_shape() (Issue 229 fix, part 1b, development log
) for the identical real reason, a page whose only real question-start signal is a
    worked-solution table row (see that function's own docstring) must count here too, or
    classify_page() still returns 'other' even after _has_answer_grid_table_shape() correctly
    stops excluding it as a grid.
    """
    if bounding_boxes:
        for bb in bounding_boxes:
            if bb.get("type") not in _BODY_CONTENT_REGION_TYPES:
                continue
            content = bb.get("content") or ""
            if _match_question_start(content) or _BARE_NUMBER_LABEL_PATTERN.fullmatch(content.strip()):
                return True
        return _has_worked_solution_table_shape(bounding_boxes, extracted_text)
    if _has_worked_solution_table_shape(None, extracted_text):
        return True
    for line in extracted_text.splitlines():
        if _match_question_start(line):
            return True
    return False


def _is_question_page_via_bare_number_only(bounding_boxes: Optional[list]) -> bool:
    """
    Issue 144 residual fix (the development log, acsprimary p44). True iff a page's
    classify_page() verdict of 'question_page' traces ENTIRELY to _has_question_like_content()'s
    bare-number-only branch (_BARE_NUMBER_LABEL_PATTERN.fullmatch on some region), with no real
    _match_question_start hit anywhere on the page.

    Used by segment_questions() to distinguish a genuine bare-labeled question (chij p37's real
    "14." case, the development log Bug 2, must stay question_page unconditionally) from a
    worked-solution page whose own fragment number ("6)", acsprimary p44) coincidentally matches
    the same bare-number branch. The two are structurally indistinguishable from page content
    ALONE, confirmed, no safe per-page-only signal exists (Issue 144's
    own residual note), so this signal is deliberately combined with an adjacency check (was the
    immediately preceding page classify_page()=='answer_key_grid'?) at the call site, mirroring
    locate_embedded_answer_section()'s own previous_page_was_grid carry-forward for the identical
    real reason (chij p47).

 Corpus-wide, checked directly before trusting this, not assumed: 194 real pages
    across the full 132-file corpus classify question_page via this branch alone; exactly ONE of
    them (P6_Maths_2024_SA2_acsprimary p44) immediately follows a real answer_key_grid page, zero
    collision risk for the other 193, including every real chij bare-number case checked directly.
    """
    if not bounding_boxes:
        return False  # the flat-text path's _match_question_start already requires real content,
                       # no separate bare-number-only branch exists for it (see
                       # _has_question_like_content's own flat-text loop above)
    real_trigger = False
    bare_only_trigger = False
    for bb in bounding_boxes:
        if bb.get("type") not in _BODY_CONTENT_REGION_TYPES:
            continue
        content = (bb.get("content") or "").strip()
        if _match_question_start(content):
            real_trigger = True
            break
        if _BARE_NUMBER_LABEL_PATTERN.fullmatch(content):
            bare_only_trigger = True
    return bare_only_trigger and not real_trigger


def _has_answer_grid_table_shape(bounding_boxes: Optional[list], extracted_text: str = "") -> bool:
    """
    Real, confirmed signal (the development log follow-up) that a page is part of an
    answer-key grid even without an "ANSWER KEY" title on THIS specific page: a `table`-type
    region whose HTML contains at least one cell matching _GRID_LABEL_CLASSIFICATION_PATTERN
    ("Q1"/"16."/"22a.") in a label position.

    NOT the same (deliberately stricter) pattern parse_answer_key_grid() itself uses to accept a
    label once a page is ALREADY confirmed to be a grid, found and fixed a real false-positive
    here first: the broader _GRID_LABEL_CELL_PATTERN (bare digits, no punctuation required) matches
    a plain data value in a completely unrelated real question's own embedded table by pure
    coincidence, confirmed real, chij page 6's bead-count table ("Box"/"Green"/"Orange"/"Total"
    headers, rows like ['A','11','15','26']) matched FOUR times ("15","21","23","14", the Orange
    column) purely because those are bare 1-3-digit numbers landing on an even cell position, with
    nothing to do with any answer key. Every real grid label actually observed, across BOTH real
    shapes pulled for this design, carries some real decoration a coincidental data value won't:
    chij's own labels are always "Q"-prefixed ("Q1".."Q30"), GS005's are always "."-suffixed
    ("1.".."17c."). Requiring that decoration specifically (not just "looks like a small number")
    is what actually distinguishes a real grid cell from a real data table's plain value.

    `extracted_text` (Issue 100's grid-side follow-up): checks
    flat-text-embedded `<table>` HTML too (DeepSeek-OCR's real shape, confirmed `bounding_boxes`
    is always `[]` on real grid pages, but `extracted_text` embeds the same real table markup
    inline), same SIGNAL, same STRICT pattern, just also sourced from flat text so a continuation
    page with no repeated "ANSWER KEY" title still classifies correctly for a flat-text reader, not
    just a structured-region one. A QUESTION page's own embedded data table (chij p6's bead-count
    table is real flat text too on the DeepSeek-OCR side) is rejected the identical way, the
    strict label-decoration requirement doesn't care which source the table HTML came from.

    Issue 229 fix, part 1 (the development log, 621-record investigation, deep-dive commit
    1fa0f0a): a real, confirmed second false-positive family, distinct from the chij p6 one above , 
    MinerU2.5 wraps a worked-solution answer BLOCK (a full narrative paragraph of solution steps)
    in an HTML `<table>` row whose own first cell is a bare grid-label-shaped string ("Q27", "28)",
    "14a)"), real, confirmed on `P6_Maths_2024_SA2_acsprimary` p40/41/56/60. The label-decoration
    requirement above does not distinguish this from a genuine grid row, since both use the exact
    same real "Q"-prefixed or punctuation-suffixed convention. The real, distinguishing signal
    checked directly against the whole corpus (not guessed): a genuine answer-key grid's own
    label-adjacent VALUE cell is always short (corpus-wide exhaustive check across every real
    "ANSWER KEY"-titled page, 2,625 real label-value pairs: max real length 223 characters, an
    OCR-padded diagram-label outlier, the next-longest real values are 158/148/146), while these
    misclassified worked-solution rows' own adjacent cell is always a long narrative paragraph
    (confirmed directly: 302/388/392/452/472/736 characters, the SAME 4 real pages). A clean,
    real gap exists between 223 and 302 with no overlap, `_GRID_LABEL_VALUE_MAX_LEN` (260) sits in
    that real gap, not a guessed round number. Deliberately does NOT touch
    `P6_Maths_2024_SA2_acsprimary` p43, that page's own `answer_key_grid` classification comes
    from a DIFFERENT, real, deliberate signal (Issue 144's title-region check, a genuine single-
    long-answer-key-entry shape, confirmed the exact same real page in that issue's own
    docstring), this fix is scoped to the table-cell signal only, never the title-region one.
    """
    table_htmls = [
        bb.get("content") or "" for bb in (bounding_boxes or []) if bb.get("type") == "table"
    ]
    if extracted_text:
        table_htmls.extend(m.group(0) for m in _EMBEDDED_HTML_TABLE_PATTERN.finditer(extracted_text))
    for table_html in table_htmls:
        if not table_html.strip():
            continue
        try:
            table_rows = _parse_html_table_rows(table_html).rows
        except Exception:  # noqa: BLE001, a classification check must never itself crash a page
            continue
        for cells in table_rows:
            for i in range(0, len(cells) - 1, 2):
                if not _GRID_LABEL_CLASSIFICATION_PATTERN.match(cells[i].strip()):
                    continue
                value = cells[i + 1].strip() if i + 1 < len(cells) else ""
                if len(value) > _GRID_LABEL_VALUE_MAX_LEN:
                    continue  # Issue 229 fix: a worked-solution paragraph, not a real grid value
                return True
    return False


def _parse_worked_solution_rows(table_html: str) -> list[tuple[str, str]]:
    """
    Issue 229 fix, part 1b, the POSITIVE side of the SAME real
    signal `_has_answer_grid_table_shape()` uses negatively (see that function's own docstring for
    the full real evidence: a grid-label-shaped cell, `_GRID_LABEL_CLASSIFICATION_PATTERN`, paired
    with an adjacent value cell longer than `_GRID_LABEL_VALUE_MAX_LEN`). Fixing `classify_page()`
    alone is not sufficient, a page correctly no longer excluded as `answer_key_grid` still has no
    real per-question content for `segment_questions()` to find, since MinerU2.5's own region for
    a worked-solution table is ONE opaque `table`-typed blob, not a line/region that itself starts
    with the label followed by real content (confirmed directly: `P6_Maths_2024_SA2_acsprimary`
    p40/41/56/60's own real table rows are exactly this shape). This function parses ONE table's
    own rows into real `(question_number, raw_text)` pairs, one per matching row, in row order , 
    so each real worked-solution block becomes its own real question span, the same real outcome a
    genuine per-row question-start match would produce. `question_number` is the label's own bare
    digit portion (any real "Q"/punctuation decoration stripped), matching `_match_question_
    start()`'s own convention so cross-reader number alignment (or, failing that, the existing
    Issue 224/228 content-overlap recovery) behaves identically to any other real span pair.
    Returns `[]` for an ordinary table (a real question's own embedded data table, or a genuine
    short-value answer-key row), never fires on either, confirmed via the same length gate.
    """
    if not table_html.strip():
        return []
    try:
        table_rows = _parse_html_table_rows(table_html).rows
    except Exception:  # noqa: BLE001, a segmentation helper must never itself crash a page
        return []
    rows_found: list[tuple[str, str]] = []
    for cells in table_rows:
        for i in range(0, len(cells) - 1, 2):
            label = cells[i].strip()
            if not _GRID_LABEL_CLASSIFICATION_PATTERN.match(label):
                continue
            value = cells[i + 1].strip() if i + 1 < len(cells) else ""
            if len(value) <= _GRID_LABEL_VALUE_MAX_LEN:
                continue  # a real short grid value, _has_answer_grid_table_shape's own job, not this
            # Real, confirmed bug found and fixed while building this (P6_Maths_2024_SA2_
            # acsprimary p56): keeping only the bare digit here collapses "14a)" and "14b)" , 
            # two REAL, SEPARATE table rows, each its own worked answer, into the same
            # question_number "14", silently losing one row's content to the other in the
            # `{question_number: span}` dict a caller builds. The letter suffix is kept as part
            # of the number, exactly as the label itself prints it.
            number_match = re.match(r"Q?(\d{1,3}[a-zA-Z]?)", label)
            if number_match:
                rows_found.append((number_match.group(1), value))
    return rows_found


def _has_worked_solution_table_shape(bounding_boxes: Optional[list], extracted_text: str = "") -> bool:
    """Page-level real signal for `_has_question_like_content()` (Issue 229 fix, part 1b), does
    ANY table on this page contain at least one real worked-solution row (see
    `_parse_worked_solution_rows()`'s own docstring)? Same table-source scan as
    `_has_answer_grid_table_shape()` (structured `table`-typed regions, plus embedded `<table>`
    HTML in flat text for DeepSeek-OCR's own shape), the two functions are deliberately mutually
    exclusive per row (short value vs. long value), never both true for the same real row."""
    table_htmls = [
        bb.get("content") or "" for bb in (bounding_boxes or []) if bb.get("type") == "table"
    ]
    if extracted_text:
        table_htmls.extend(m.group(0) for m in _EMBEDDED_HTML_TABLE_PATTERN.finditer(extracted_text))
    return any(_parse_worked_solution_rows(html) for html in table_htmls)


def _is_solo_diagram_noise_table_region(bounding_boxes: Optional[list]) -> bool:
    """
    Real, corpus-validated position/region-sequence signal (development log, rowspan/colspan position-
 signal investigation brief) for Issue 145 (`P6-Maths-2021-SA1-Nan-Hua.pdf` p38):
    a bar-model diagram's own unit-label table (`<table><tr><td>9 u</td><td>60</td><td>38</td>
    </tr>...</table>`) has no real content-shape signal that safely separates it from a genuine
    answer-key grid elsewhere (five prior content-shape attempts tried and reverted, see that
    issue's own history) -- but MinerU2.5's own `bounding_boxes` list carries real region TYPE
    and ORDER metadata that was captured all along and never previously consulted for its sequence
    context (every existing call site in this file reads only `.get("type")`/`.get("content")` on
    a SINGLE region at a time, confirmed via a direct grep before writing this function -- this is
    the first place `bounding_boxes`' own real document order is used as a signal).

    A `table`-type region is treated as diagram noise only if ALL THREE hold:
      1. it is the ONLY `table`-type region on the page (a page with a second, real table is never
         excluded by this check -- deliberately false-negative-safe, not false-positive-safe);
      2. the region immediately preceding it in the raw list (MinerU2.5's own real reading order,
         confirmed monotonically non-decreasing in y0) is `title`-type with bare, single-word
         content (real: `"After"` on Nan-Hua p38 -- a bar-model diagram's own short figure label);
      3. the region immediately following it is `equation`-type (real: the diagram's own algebraic
         working, not a real grid's next cell/text row).

 Corpus-wide validated against the FULL real corpus -- 4,423 pages, 1,344 real
    `table`-type regions: this exact 3-condition combination matches EXACTLY ONE region in the
    whole corpus, the known Nan-Hua p38 case itself, with ZERO false positives against all 1,176
    real, confirmed answer-source pages. Four loosened variants (any-word-count title instead of
    single-word; equation-anywhere-later-on-page instead of immediately-following; dropping the
    solo-table requirement) were also tried and found no additional matches -- this exact
    combination is not an arbitrarily narrow fit to the one known case.

    Neither condition 2 nor condition 3 is safe ALONE -- checked directly, not assumed:
      - "title precedes" alone: 79 matches, 72 real false positives (a real answer-key grid very
        commonly has its own real section-header title directly above it too, e.g. "BOOKLET A
        (PAPER 1)").
      - "equation follows" alone: 3 matches -- the real Nan-Hua p38 case, plus TWO real,
        individually verified near-misses that are genuine answer-source content, not noise:
        `P6_Maths_2022_SA2_nanyang` p50 (a real True/False answer table on a messy, scanned-
        worked-solution-style page) and `PSLE Math 2018 Answer+Solution` p2 (a real vertical-
        multiplication worked-solution table for Q22). Both are correctly excluded by condition 2:
        their own immediately-preceding region is real `text` ("Each of the statements below is
        either true, false or not possible..." / "22. 27 42 6"), not `title`.

    Deliberately scoped to the real bug's own mechanism -- `locate_embedded_answer_section()`'s
    chij-p47 carry-forward heuristic is the ONLY real place this diagram table gets swept into grid
    parsing at all (see that function's own docstring); this page's own `classify_page()` result
    ('other') is already correct on its own. NOT folded into `classify_page()` itself, since nothing
    about how this page is individually classified is wrong -- only the cross-page carry-forward's
    decision to still treat it as a grid continuation is the real bug.
    """
    if not bounding_boxes:
        return False
    table_indices = [i for i, bb in enumerate(bounding_boxes) if bb.get("type") == "table"]
    if len(table_indices) != 1:
        return False  # condition 1 -- solo table only
    idx = table_indices[0]
    if idx == 0 or idx == len(bounding_boxes) - 1:
        return False  # no real preceding/following region at all -- can't be this shape
    preceding = bounding_boxes[idx - 1]
    following = bounding_boxes[idx + 1]
    if preceding.get("type") != "title":
        return False  # condition 2
    preceding_content = (preceding.get("content") or "").strip()
    if not preceding_content or len(preceding_content.split()) != 1:
        return False  # condition 2 -- bare, single-word title only
    if following.get("type") != "equation":
        return False  # condition 3
    return True


def classify_page(
    bounding_boxes: Optional[list], extracted_text: str,
) -> Literal["cover_page", "question_page", "answer_key_grid", "other"]:
    """
    Page-level classifier, run BEFORE question segmentation . Real cover
    pages have been confirmed to contain numbered lines too (PSLE Math 2019's own cover: "1. Write
    your Index No...", "2. Do not turn over this page..."), the region-based segmenter alone
    would still wrongly capture those as fake questions without this classifier running first and
    excluding cover_page/answer_key_grid/other pages from segmentation entirely.

    Checked in this order (answer_key_grid and cover_page must be checked before question_page,
    since a cover page can otherwise look like it has "question-like content" too, the ordering
    is deliberate, not incidental):
      1. answer_key_grid, a `title`-typed region literally containing "answer key(s)" (or, for
         readers without structured title regions, the same phrase anywhere in flat text); OR a
         `table`-type region whose HTML contains a real grid-label-shaped cell ("Q1"/"16."/"22a.",
         _GRID_LABEL_CELL_PATTERN), added (the development log follow-up) after finding the
         title-only signal misses a real, confirmed case: a multi-page grid's CONTINUATION pages
         (chij p45/p46) have no repeated "ANSWER KEY" title at all, only more `table` regions, and
         were being classified 'other' and silently skipped by locate_embedded_answer_section()
         entirely, not just mis-segmented, invisible to it. ALSO (Issue 144, Decision
 Log/09-01): a `title`-typed region that is ITSELF Q-prefixed grid-label-shaped
         ("Q5)"), a real, confirmed shape distinct from the table-cell case above: some answer-key
         pages carry NO `<table>` at all when a single answer's own working runs long enough to be
         typeset as ordinary paragraphs (`P6_Maths_2024_SA2_acsprimary.pdf` p43/Q5, p44/Q6, see
         `_ANSWER_KEY_STYLE_TITLE_LABEL_PATTERN`'s own docstring for why this is scoped to the
         "Q"-prefixed shape only, not reusing the broader table-cell pattern here). ALSO (Issue
         372, development log, directly continuing Issue 371's Topic 4 verification finding): the
         SAME real page shape as the Issue 144 case above, but where MinerU2.5 typed the page's
         own leading label as a plain `text` region instead of `title`, a REAL, confirmed, common
         case (a real corpus-wide survey found 10 of 12 real matching pages this way, not just the
         one `id=3265` example that surfaced it), and NOT limited to "Q"-prefixed labels either
         (`"9"`, `"13a)"`, `"19."` all real, confirmed labels on real worked-solution pages this
         way). See `_has_narrative_worked_solution_shape`'s own docstring for the real, gated check
         this needed (a bare label alone is NOT enough, chij p37's own real "14." bare-number
         `text` region is a genuine question start, not an answer page, the exact real
         false-positive risk `_ANSWER_KEY_STYLE_TITLE_LABEL_PATTERN`'s own docstring already
         warned against for this reason).
      2. cover_page, 2+ cover-page keyword hits in the page's text.
      3. question_page, at least one real question-start match (region-based if bounding_boxes
         are available, flat-text line-start otherwise).
      4. other, none of the above; segment_questions() returns no spans for this page too,
         conservatively. NOTE this signal still misses one real, confirmed shape (chij p47): a
         continuation page whose only content is an orphan fragment with NO label cell at all, not
         even a partial one, nothing on that page alone distinguishes it from any other page with
         a table. Not solved here; locate_embedded_answer_section() carries forward a
         "was the previous page part of the grid" signal specifically to cover this narrow,
         page-adjacency-only case, see its own docstring.
    """
    text_lower = extracted_text.lower()

    for bb in bounding_boxes or []:
        if bb.get("type") == "title":
            title_text = (bb.get("content") or "").lower()
            if any(kw in title_text for kw in _ANSWER_KEY_TITLE_KEYWORDS):
                return "answer_key_grid"
            # Issue 144 fix (the development log-01): see this function's own docstring
            # point 1 and _ANSWER_KEY_STYLE_TITLE_LABEL_PATTERN's own docstring for the real case
            # and why this is scoped Q-prefix-only.
            if _ANSWER_KEY_STYLE_TITLE_LABEL_PATTERN.match((bb.get("content") or "").strip()):
                return "answer_key_grid"
    if any(kw in text_lower for kw in _ANSWER_KEY_TITLE_KEYWORDS):
        return "answer_key_grid"
    if _has_answer_grid_table_shape(bounding_boxes, extracted_text):
        return "answer_key_grid"
    # Issue 372 fix (development log, answer-key label regex fix brief, directly continuing
    # Issue 371's Topic 4 verification finding), see _has_narrative_worked_solution_shape's
    # own docstring for the real evidence and the real false-positive (chij p37, both real 2022
    # and 2025 SA2 files) this is checked safe against.
    if _has_narrative_worked_solution_shape(bounding_boxes):
        return "answer_key_grid"

    if sum(1 for kw in _COVER_PAGE_KEYWORDS if kw in text_lower) >= _COVER_PAGE_MIN_KEYWORD_HITS:
        return "cover_page"

    if _has_question_like_content(bounding_boxes, extracted_text):
        return "question_page"

    return "other"


def _segment_from_regions(
    page: PageImage, bounding_boxes: list, discrepancy_log: Optional[Path] = None,
) -> list[QuestionSpan]:
    """
    Region-based segmentation, each body-content region is a candidate
    question start or a continuation of the currently-open one, not a line-start regex match on
    flattened text. A question span runs from one matched question-number region up to (but not
    including) the next one, so administrative fragments between them (an "Ans:" line, a margin
    note like "Do not write in this space") land inside the span they visually belong to rather
    than being separately mis-parsed, the same real imperfection this pipeline already accepts
    elsewhere (Structure preservation scoring, protocol Section 5) in exchange for not needing a
    hand-built exception list for every administrative phrase a real paper might contain.

    Real patterns found and handled during validation against the actual 2-file subset (Decision
 Log), not anticipated from GS001-GS006 alone:
      - MinerU2.5 sometimes emits a question's own number as a BARE separate region ("26" on its
        own, content-wise, followed by a fully separate region "The graph shows..."), rather than
        fused into the same region as GS001's "1. Round off..." pattern. Handled below by
        recognizing a region whose content is JUST digits (optionally with trailing "." or ")" , 
        Bug 2 fix, follow-up, confirmed real chij p37 shape "14.") as a detached number label,
        consumed by the next region rather than folded into a provisional span.
      - Standard PSLE section-header sentences ("Questions 21 to 30 carry 2 marks each...") are a
        real, recurring MinerU2.5 'text' region that would otherwise get swept into a provisional
        span the same way GS004's unnumbered Q28 content does. _SECTION_INSTRUCTION_
        PATTERN excludes this specific, real, recurring sentence shape before it can ever be
        folded into current_parts.
      - Shared-stimulus paragraphs ("Use the information below to answer Question 9 and 10.") name
        the two questions they belong to directly in their own text (Bug 3 fix, follow-up) , 
        recognized via _SHARED_STIMULUS_PATTERN and held in `pending_stimulus` rather than being
        folded into a provisional span at all; prepended to BOTH named questions' raw_text once
        they're found among this page's real spans (see the post-processing pass below), instead
        of being guessed into a standalone, wrongly-numbered row.
    """
    page_index = page.page_index
    spans: list[QuestionSpan] = []
    current_number: Optional[str] = None
    current_parts: list[str] = []
    pending_bare_number: Optional[str] = None
    pending_stimulus: Optional[dict] = None
    stimulus_attachments: list[dict] = []

    # has_diagram tracking (Issue 102 part 2), three real,
    # confirmed-clean attribution cases, mirroring the three real states this loop already
    # tracks (an open span, a pending bare-number label, a pending shared stimulus):
    current_has_diagram = False        # attaches to the span currently being accumulated
    pending_bare_number_has_diagram = False  # attaches to whatever span pending_bare_number becomes
    pending_leading_has_diagram = False      # attaches to the FIRST span opened when nothing at
                                               # all has started yet on this page (real confirmed
                                               # case: a stimulus/diagram printed before its
                                               # question's own number, e.g. chij p18's angle
                                               # figure before "25.")

    def flush() -> None:
        if current_parts:
            spans.append(
                QuestionSpan(
                    question_number=current_number or "",
                    raw_text="\n".join(current_parts).strip(),
                    page_index=page_index,
                    page_label=None,  # TODO: extract the printed page code once a real signal
                                       # for it is confirmed against region data, not yet done
                    has_diagram=current_has_diagram,
                )
            )

    for bb in bounding_boxes:
        if bb.get("type") == "image":
            # Real signal, MinerU2.5 only (Issue 102 part 2) , 
            # attribute by position in reading order, same sequential walk this loop already
            # does for text content. Deliberately evaluated before the _BODY_CONTENT_REGION_TYPES
            # filter below, since "image" itself is NOT in that set (it's the diagram itself, not
            # question-body text) and would otherwise be silently skipped like a page_number/
            # footer region.
            if pending_stimulus is not None:
                pending_stimulus["has_diagram"] = True
            elif current_parts:
                current_has_diagram = True
            elif pending_bare_number is not None:
                pending_bare_number_has_diagram = True
            else:
                pending_leading_has_diagram = True
            continue

        if bb.get("type") not in _BODY_CONTENT_REGION_TYPES:
            continue
        content = (bb.get("content") or "").strip()
        if not content:
            continue

        stimulus_match = _SHARED_STIMULUS_PATTERN.match(content)
        if stimulus_match and pending_stimulus is None:
            flush()
            pending_stimulus = {
                "text": [content],
                "numbers": {stimulus_match.group(1), stimulus_match.group(2)},
                "has_diagram": pending_leading_has_diagram,  # a diagram seen before this
                                                                # stimulus paragraph itself opened
                                                                #, no real example found with
                                                                # this exact ordering, carried in
                                                                # anyway since it's cheap and
                                                                # correct if it ever occurs
            }
            pending_bare_number = None
            pending_bare_number_has_diagram = False
            pending_leading_has_diagram = False
            current_has_diagram = False
            continue

        if pending_stimulus is not None:
            # Still collecting a shared stimulus's supporting content (an intro sentence plus
            # whatever description/table sits between it and the first of its two named
            # questions, both real observed shapes, chij p6's table, PSLE Math 2019 p2's "Figure
            # 1"/"Figure 2" labels). Ends only when a real question-start or a detached
            # bare-number region shows up, anything else (that isn't boilerplate) keeps
            # accumulating into the stimulus text rather than becoming its own provisional span.
            ends_stimulus = (
                _match_question_start(content) is not None
                or _BARE_NUMBER_LABEL_PATTERN.fullmatch(content) is not None
            )
            if not ends_stimulus:
                if not _is_recognized_boilerplate(content):
                    pending_stimulus["text"].append(content)
                continue
            stimulus_attachments.append(pending_stimulus)
            pending_stimulus = None
            # falls through below to handle THIS region under the normal rules, it's the region
            # that ended stimulus collection, not stimulus content itself.

        if _is_recognized_boilerplate(content):
            logger.debug(
                "segment_questions: page %d skipping recognized boilerplate, not treated as "
                "question content: %r", page_index, content[:60],
            )
            # A pending bare-number region (see below) that turns out to be followed by
            # boilerplate, the real chij p12/p20 case: "2"/"10" followed by "MARKS:", was never
            # a detached question number at all, just the marks value in a marks-tally box.
            # Discard it rather than letting it silently attach to whatever real content follows.
            pending_bare_number = None
            pending_bare_number_has_diagram = False
            continue

        if bb.get("type") == "table":
            # Issue 229 fix, part 1b, a worked-solution table (see
            # _parse_worked_solution_rows()'s own docstring) is not a candidate for the normal
            # single-region content-matching logic below at all: it is ONE opaque region whose OWN
            # rows are each a real, separate question's worked answer. Explode it directly into one
            # real QuestionSpan per matching row, in row order, bypassing current_parts entirely , 
            # the same real per-question granularity a genuine sequence of matched regions would
            # produce. An ordinary data table embedded in a real question (chij p6's bead-count
            # table) or a genuine short-value answer-key row never matches (same length gate as
            # _has_answer_grid_table_shape(), mutually exclusive by construction) and falls through
            # to the normal table-content handling below unchanged.
            worked_solution_rows = _parse_worked_solution_rows(content)
            if worked_solution_rows:
                flush()
                current_number, current_parts = None, []
                current_has_diagram = False
                for row_number, row_text in worked_solution_rows:
                    spans.append(
                        QuestionSpan(
                            question_number=row_number, raw_text=row_text,
                            page_index=page_index, page_label=None,
                        ),
                    )
                pending_bare_number = None
                pending_bare_number_has_diagram = False
                pending_leading_has_diagram = False
                continue

        bare_number_match = _BARE_NUMBER_LABEL_PATTERN.fullmatch(content)
        if bare_number_match:
            # Consumed by the next region below, not folded into current_parts itself. Only the
            # digits are kept as pending_bare_number, not any trailing punctuation, since this
            # value becomes the span's question_number directly.
            pending_bare_number = bare_number_match.group(1)
            continue

        match = _match_question_start(content)
        if match and _is_recognized_boilerplate(match[1]):
            # Real, confirmed shape (Issue 224 fix, development log 2026-09-0X,
            # P6-Maths-2021-SA1-Catholic-High p11): a page-footer/marks-tally numeral and real
            # navigation boilerplate sometimes arrive as ONE combined region ("9 (Go on to the
            # next page)"), not two separate ones, `_is_recognized_boilerplate()`'s own check on
            # the WHOLE content can't catch this (it only matches a string that itself STARTS with
            # the boilerplate phrase, and here a digit comes first), but `_match_question_start()`
            # happily matches "9" as a number with real content-length "rest". Checking the
            # extracted REST here catches it: real question content is never itself entirely
            # recognized boilerplate. Treat exactly like a real bare-number-then-boilerplate
            # region (see the boilerplate branch above) rather than fabricating a phantom
            # question whose only content is navigation text.
            pending_bare_number = None
            pending_bare_number_has_diagram = False
            continue
        if match:
            flush()
            current_number, rest = match
            current_parts = [rest]
            current_has_diagram = pending_leading_has_diagram
            pending_leading_has_diagram = False
        elif pending_bare_number is not None:
            if len(content) < _MIN_QUESTION_CONTENT_CHARS:
                # Too short to trust as this bare number's real content (e.g. a bare diagram
                # axis label like "X"/"Y", confirmed real image_caption content, development log
                # now that image_caption/image_footnote are in-scope). Drop it and
                # keep pending_bare_number in case the real content follows in a later region,
                # rather than starting a span with near-empty text.
                logger.debug(
                    "segment_questions: page %d dropping short content after pending bare "
                    "number %r, not enough to start a span: %r",
                    page_index, pending_bare_number, content,
                )
                continue
            flush()
            current_number, current_parts = pending_bare_number, [content]
            current_has_diagram = pending_bare_number_has_diagram or pending_leading_has_diagram
            pending_bare_number_has_diagram = False
            pending_leading_has_diagram = False
        elif current_parts or current_number is not None:
            current_parts.append(content)
        elif len(content) < _MIN_QUESTION_CONTENT_CHARS:
            # First real body content on the page, no number seen yet, but too short to trust as
            # a genuine unnumbered question opener (e.g. a bare diagram axis label, same real
            # image_caption evidence as above). Drop rather than starting a fake provisional span.
            logger.debug(
                "segment_questions: page %d dropping short unnumbered content, not enough to "
                "start a provisional span: %r", page_index, content,
            )
        else:
            # First real body content on the page, no number seen yet (and no pending bare-number
            # label either), provisional span; GS004's real Q28 case (its leading "28" was never
            # OCR'd by MinerU2.5 at all, in any form, see module docstring note above).
            # question_number filled in below if the next real question's number makes it
            # unambiguous.
            current_parts = [content]
            current_has_diagram = pending_leading_has_diagram
            pending_leading_has_diagram = False
        pending_bare_number = None
        pending_bare_number_has_diagram = False
    flush()

    if pending_stimulus is not None:
        # Page ended while still "inside" a shared stimulus block, neither of its named
        # questions ever showed up as a region on THIS page (could be a real page-break case, or
        # a genuine anomaly). Not silently dropped: recorded the same as any other stimulus
        # attachment, so the "referenced number never found among this page's spans" check below
        # logs it durably rather than the text just vanishing.
        stimulus_attachments.append(pending_stimulus)

    # Attach each shared stimulus's collected text to BOTH of its named questions (Bug 3 fix,
    # follow-up), never as its own standalone, guessed-numbered row. A referenced number that
    # doesn't match any real span on this page is a genuine anomaly, logged durably rather than
    # silently discarded.
    span_by_number = {s.question_number: s for s in spans if s.question_number}
    for attachment in stimulus_attachments:
        stimulus_text = "\n".join(attachment["text"]).strip()
        for number in sorted(attachment["numbers"]):
            target = span_by_number.get(number)
            if target is None:
                message = (
                    f"page {page_index}: shared-stimulus paragraph named question {number!r} "
                    f"but no span with that number exists on this page — stimulus text not "
                    f"attached anywhere, needs human review: {stimulus_text[:100]!r}"
                )
                logger.warning("segment_questions: %s", message)
                log_ocr_discrepancy(page.source_pdf.stem, page_index, message, discrepancy_log)
                continue
            target.raw_text = f"{stimulus_text}\n{target.raw_text}".strip()
            if attachment.get("has_diagram"):
                # Same real signal, propagated to BOTH of the stimulus's named questions, a
                # diagram appearing inside a shared-stimulus block is data both named questions
                # depend on, not just the one whose own span happened to be open when it
                # appeared. Real confirmed case: PSLE Math 2019 p2's "Figure 1"/"Figure 2" sit
                # inside the "Use the information below to answer Questions 5 and 6" stimulus , 
                # both Q5 and Q6 need them, not just Q5.
                target.has_diagram = True

    existing_numbers = {s.question_number for s in spans if s.question_number}
    for i, span in enumerate(spans):
        if span.question_number != "":
            continue
        if i + 1 < len(spans) and spans[i + 1].question_number.isdigit():
            inferred = str(int(spans[i + 1].question_number) - 1)
            if inferred in existing_numbers:
                # Bug 3 fix, follow-up: inferring here would collide with a number some OTHER
                # span on this page already uses, exactly the shared-stimulus collision
                # mechanism, but for a phrasing _SHARED_STIMULUS_PATTERN doesn't recognize (the
                # specific pattern above already prevents the two real confirmed cases from ever
                # reaching this point at all). Don't guess either way on a genuine ambiguity , 
                # log it durably and leave question_number empty, same discipline this project
                # already applies elsewhere (e.g. Issue 61's ambiguous school-name matches).
                message = (
                    f"page {page_index}: span {i} has no question number, and the usual "
                    f"inference (next confirmed number {spans[i + 1].question_number!r} minus "
                    f"one = {inferred!r}) would collide with an already-used number on this same "
                    f"page — left unassigned, not guessed, needs human review: "
                    f"{span.raw_text[:100]!r}"
                )
                logger.warning("segment_questions: %s", message)
                log_ocr_discrepancy(page.source_pdf.stem, page_index, message, discrepancy_log)
                span.flagged = "ambiguous_unnumbered_span_number_collision"
                continue
            logger.info(
                "segment_questions: page %d span %d has no question number in the reader's own "
                "output — inferred %r from the following question's number %r, not directly "
                "observed. See GS004's real case in this module's docstring.",
                page_index, i, inferred, spans[i + 1].question_number,
            )
            span.question_number = inferred
        else:
            logger.warning(
                "segment_questions: page %d span %d has no question number and none could be "
                "inferred (no following numbered question on this page) — left as an empty "
                "string, not guessed.",
                page_index, i,
            )

    return spans


def _segment_from_flat_text(
    page_index: int, extracted_text: str, log_as_fallback: bool = True,
) -> list[QuestionSpan]:
    """
    Flat-text fallback, for readers with no structured regions to
    segment from at all (DeepSeek-OCR's markdown output; RawReaderOutput.bounding_boxes is always
    empty for it, confirmed across the whole bake-off). This is the single_reader_only path where
    MinerU2.5 (primary) failed and DeepSeek-OCR (secondary) is the only usable output for a page , 
    a real, expected case, not an error. Uses the same permissive number pattern and minimum
    content length as the region-based path, applied to line starts instead of region boundaries.
    Logged clearly as the lower-confidence path, since a region boundary is real reader-reported
    structure and a line-start regex on flattened markdown is not, not silently indistinguishable
    from the region-based result.

    log_as_fallback=False (Bug 1 fix, follow-up): this function is also reused as a pure
    cross-check utility by _cross_check_low_confidence_page below, where bounding_boxes is NOT
    empty (the region-based path is still primary), the "no structured regions" warning would be
    actively wrong in that case, so the caller suppresses it.

    Trailing recognized-boilerplate lines are stripped from each span's own content (Issue 224
    fix, development log 2026-09-0X), real, confirmed asymmetry found investigating that fix's own
    corpus-wide validation: the region-based path (`_segment_from_regions`) already discards a
    recognized-boilerplate REGION entirely via `_is_recognized_boilerplate()` before it can attach
    to any span, but this flat-text path never applied that same check at all, so a trailing
    boilerplate line (e.g. "(Go on to the next page)") stayed glued onto whichever question's
    content happened to run up to the page's own end. Confirmed real, concrete regression this
    caused: P6-Maths-2021-SA1-Rosyth p14 q23, MinerU2.5's own "(Go on to the next page)" got
    correctly stripped by the region path, DeepSeek-OCR's own copy of the exact same real
    boilerplate did not, and the resulting ASYMMETRY (one side clean, one side not) dropped their
    real text-similarity score from 0.701 (agree) to 0.585 (disagree), a real, spurious
    disagreement neither reader's own real content ever caused. Stripping this here, the same real
    signal the region path already trusts, restores parity between the two paths.
    """
    if log_as_fallback:
        logger.warning(
            "segment_questions: page %d has no structured regions to segment from (bounding_boxes "
            "empty) — using the flat-text fallback. Lower confidence than the region-based path; "
            "expected when only DeepSeek-OCR's output was usable for this page (single_reader_only).",
            page_index,
        )
    spans: list[QuestionSpan] = []
    pattern = re.compile(r"(?m)^\s*(\d{1,3})[.)]?\s+(?=\S)")
    matches = list(pattern.finditer(extracted_text))
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(extracted_text)
        content = _strip_trailing_boilerplate_lines(extracted_text[start:end].strip())
        if len(content) >= _MIN_QUESTION_CONTENT_CHARS:
            spans.append(
                QuestionSpan(
                    question_number=m.group(1), raw_text=content,
                    page_index=page_index, page_label=None,
                )
            )
    return spans


def _strip_trailing_boilerplate_lines(content: str) -> str:
    """Real, targeted fix (Issue 224, development log 2026-09-0X), see
    `_segment_from_flat_text()`'s own docstring for the real, confirmed regression this prevents.
    Strips recognized-boilerplate lines (the same `_is_recognized_boilerplate()` check the
    region-based path already applies per-region) from the END of a flat-text span's own content,
    working backwards line by line so genuine boilerplate sandwiched between blank lines at the
    very end (a common real shape) is still caught, and stops the moment a real, non-boilerplate
    line is reached, never touches content earlier in the span."""
    lines = content.split("\n")
    while lines and (not lines[-1].strip() or _is_recognized_boilerplate(lines[-1].strip())):
        lines.pop()
    return "\n".join(lines).strip()


def segment_questions(
    page: PageImage,
    reader_output: RawReaderOutput,
    ocr_confidence: Optional[Literal["high", "low"]] = None,
    discrepancy_log: Optional[Path] = None,
    previous_page_was_grid: bool = False,
) -> list[QuestionSpan]:
    """Split one page's reader output into per-question spans.

    The page is classified first with classify_page(). Only question pages are split. Cover pages,
    answer-key grids and other pages return no spans, so a cover page's numbered instructions or a
    grid's table cells never become questions.

    Region-based splitting (_segment_from_regions()) is used when the reader supplied bounding
    boxes (MinerU2.5). Otherwise the flat-text fallback (_segment_from_flat_text()) is used and
    logged as lower confidence. This covers DeepSeek-OCR, including pages where it is the only
    reader with output.

    Args:
        page: the rendered page.
        reader_output: one reader's saved output for the page.
        ocr_confidence: accepted but not used. A cross-check of low-confidence pages against the
            flat-text pattern was tried and removed: on pages with region structure it matched
            page numbers and MCQ option numbers as false question starts. The parameter is kept
            for a future, better-guarded use of per-page confidence.
        discrepancy_log: passed to _segment_from_regions() for its collision check.
        previous_page_was_grid: when True, and the page counts as a question page only because of
            a bare-number-only region match (see _is_question_page_via_bare_number_only()), the
            page is treated as a continuation of the preceding answer-key grid and no spans are
            returned (Issue 144). Defaults to False. extract_topology_b() sets it per page;
            extract_topology_a() leaves it False because its grid is in a separate answer file.

    Returns:
        The question spans on the page, or an empty list for any other page type.
    """
    bounding_boxes = reader_output.bounding_boxes or []
    classification = classify_page(bounding_boxes, reader_output.extracted_text)
    logger.info("segment_questions: page %d classified as %r", page.page_index, classification)

    if (
        classification == "question_page"
        and previous_page_was_grid
        and _is_question_page_via_bare_number_only(bounding_boxes)
    ):
        logger.debug(
            "segment_questions: page %d classified 'question_page' via a bare-number-only match "
            "but immediately follows an answer_key_grid page — treating as a grid continuation "
            "instead (Issue 144 residual fix, acsprimary p44's real shape), no spans created.",
            page.page_index,
        )
        return []

    if classification != "question_page":
        return []

    if bounding_boxes:
        return _segment_from_regions(page, bounding_boxes, discrepancy_log)
    return _segment_from_flat_text(page.page_index, reader_output.extracted_text)


@dataclass
class GridRow:
    """One real row's (question_number, answer) pair, parsed from an answer-key-grid page's
    `table`-type region(s) by parse_answer_key_grid() (the development log follow-up).

    question_number is kept RAW, matching whatever label the grid itself printed, confirmed real
    shapes from actual data pulled before designing this: bare "Q1"/"1." (chij, GS005's Booklet A/
    B columns) and letter-suffixed "1a."/"22a." (GS005's Paper 2 column, real confirmed sub-part
    labels). Normalized only by stripping a leading "Q" and any trailing ".", "Q1"/"1."/"1" all
    become "1"; "1a." becomes "1a", NOT further split into a (number, sub-part) pair, since
    segment_questions() itself only ever produces bare top-level question_numbers (confirmed real
    gap, see this module's own docstring note where GridRow is used) and a match against it can
    only ever be attempted at that same bare-number granularity.

    question_number == "" marks an UNRESOLVED cross-page continuation fragment, a real, confirmed
    shape (chij p45->p46, p46->p47: MinerU2.5 splits one grid row's overflow content across two
    separate page-level table regions, with the continuation row's label cell literally blank in
    the source). parse_answer_key_grid() does not guess which prior question a blank label
    continues (it has no cross-page context, by design, see its own docstring); the caller
    (locate_embedded_answer_section) resolves this using real column-position continuity across a
    file's grid pages, in order, the only place that context actually exists.
    """
    question_number: str
    answer_text: str
    raw_cell_content: str  # unmodified, for audit, before _clean_answer_cell_text()
    page_index: int
    column_index: int = 0  # which (label, content) pair-position within its source row this came
                             # from (0, 1, 2, ...), needed by the caller to resolve a "" continuation
                             # fragment against the right column's last-seen real question_number,
                             # not just "whatever was last seen anywhere" (real rows genuinely
                             # interleave multiple independent columns per row, e.g. GS005's three
                             # side-by-side sections, chij's Booklet A's five MCQ pairs per row).
    section_label: Optional[str] = None  # the last `table_caption` region seen before this row's
                                           # table region on THIS page ("BOOKLET A"/"BOOKLET B"/
                                           # "PAPER 2", confirmed real on chij), None on a page with
                                           # no caption at all (chij p45's Q30-only table, p46, p47).
                                           # Real, confirmed reason this matters (Issue 3, paper/
                                           # booklet question-number reuse): GS005's own single page
                                           # has question_number "2" TWICE with different
                                           # answers (Booklet A's Q2 vs Paper 2's Q2), column_index
                                           # alone can't disambiguate that, since they're also in
                                           # different (section, column) combinations, not just
                                           # different rows. Used by the caller
                                           # (locate_embedded_answer_section) to scope both
                                           # continuation-resolution and collision-detection
                                           # correctly, not globally per bare question_number alone.


# Real, confirmed question-label cell shapes, pulled from chij p44-47 and GS005 (development log
# follow-up) before this pattern was written: "Q1".."Q30" (chij), "1.".."30." and
# letter-suffixed "1a.".."17c." (GS005's Paper 2 column, real sub-part answer-key rows, distinct
# from chij's shape where sub-parts are folded as free text "a)...b)..." inside ONE cell under a
# single bare number). Optional leading "Q", optional single trailing letter, optional trailing
# ".": matches all of "Q1", "1", "1.", "1a", "1a.", "22a." in one pattern.
#
# Issue 132/137/138 fix : `\s?` added right after the optional "Q" , 
# real, confirmed shape found investigating the header-row/value-row bug, a genuine OCR spacing
# variant, not a typo in one file: several real files print the header cell as "Q 1"/"Q 11" (a
# literal space between "Q" and the digits) rather than "Q1"/"Q11". Confirmed this was NOT simply
# going to be covered by the header/value-row pairing fix alone, tested the pre-widening pattern
# directly against "Q 1" and it did not match at all, even with correct column pairing. Confirmed
# real and necessary via a live, tracked contamination artifact (Issue 138): 3 real files
# showed a wrong fragment landing in worked_solution_text specifically because a "Q 1"-labeled row
# was still falling through to the old, unrelated buggy pairing path while this pattern couldn't
# recognize it. Re-diffed those exact 3 files after this widening landed, not assumed fixed from
# the theory alone (Issue 138's own explicit condition for closing).
# Issue 142 fix : trailing punctuation widened from `\.?` to `[.)]?` , 
# real, confirmed gap found investigating Issue 127's own 119/123 secondary signal: Issue 133
# widened `_GRID_LABEL_CLASSIFICATION_PATTERN` (page-level detection) to accept a trailing ")" but
# never widened THIS pattern (row-level parsing) the same way, confirmed by direct regex testing
# that `_GRID_LABEL_CELL_PATTERN.match("Q17)")` returned None even after 133 landed. Real, measured
# consequence: 133's own classification widening let MORE pages reach `parse_answer_key_grid()`,
# but paren-suffixed rows on those pages still failed to parse at the ROW level, either silently
# skipped (a 2-cell row, logged at debug only, no discrepancy-log trace at all) or, far more
# visibly, misrouted into an orphaned "continuation fragment, no open chain" event (see Issue
# 142's own second fix below), the dominant real cause of that signal's 112->279-event growth.
#
# Issue 151 fix : the sub-part letter's own decoration widened to also
# accept PARENTHESES around it (`"Q21(a)"`, `"Q28(b)"`), not just a bare trailing letter
# (`"1a"`/`"22a."`), a real, confirmed, third real sub-part-label shape (distinct from both chij's
# free-text-fold and GS005's bare-letter-suffix shapes this pattern's own module docstring already
# named), found on real, clean 2/3-cell rows (`<td>Q21(a)</td><td>3/4 + 1/6 = ...</td>`) that
# otherwise parse correctly in every other respect, only the label cell itself was unrecognized.
# `\(?` before and `\)?` after the existing letter group, both independently optional (not a
# balanced pair), keeps this a pure textual widening: every previously-matched shape still matches
# exactly the same way, `group(1)`/`group(2)` still mean the same thing at every one of this
# pattern's own real call sites (no new capture group), and a bare `[.)]?` trailing label like
# "Q17)" still resolves identically (its own ")" is still free to be consumed by either the new
# optional `\)?` or the pre-existing `[.)]?`, functionally the same match either way). The
# resulting normalized `question_number` ("21a") lands on the exact same sibling-letter convention
# GS005's own bare-suffix shape already produces, so it combines correctly with the existing
# `_combine_lettered_siblings` machinery on the question side with no further change needed there.
_GRID_LABEL_CELL_PATTERN = re.compile(r"^Q?\s?(\d{1,3})\(?([a-zA-Z])?\)?[.)]?$")

# Issue 118 fix : a PREFIX-only sibling of _GRID_LABEL_CELL_PATTERN , 
# real, confirmed shape: a cell combining a real question number with an embedded sub-part marker
# glued directly onto it with no separating whitespace before the marker itself (`"Q17\na)"`,
# `"Q7 a)"`, `"10.\n(a)"`). The trailing `(?=\s|$)` lookahead is the real, load-bearing safety
# boundary, confirmed necessary, not decorative: ACS-Junior's own Q7 CONTENT cell, `"14y=112 cm
# \nY=112 ÷ 14=8cm"`, also starts with digits immediately followed by a letter (`"14y"`) and would
# otherwise match this pattern's own digit+letter prefix, but the character right after that
# prefix is `"="`, not whitespace or end-of-string, so the lookahead correctly rejects it. Checked
# directly against all three real glued-marker cases (all end their own real prefix in a newline
# or space, both accepted) and this real near-miss case before relying on it, not assumed safe by
# construction alone.
#
# Originally scoped to ONLY parse_answer_key_grid()'s own rowspan-origin recovery pass (Issue
# 118), renamed and widened by Issue 125 after a real batch scan
# (done investigating Issue 122's own pre-check, logged as Issue 124) found the identical
# glued-marker shape on ORDINARY rows with no rowspan attribute at all (Nan-Hua p34's own
# `"8.\n(a)\n(b)"`, p36's own `"13.\n(a)"`), the pattern itself was never rowspan-specific, only
# where it was tried was. Now also tried directly in the main per-row pairing loop's own label
# match, gated on the row's own content being non-blank (mirroring the recovery pass's own
# "nothing to create if nothing recoverable" rule), real, confirmed reason for that gate:
# Raffles-Girls p33's own `"Q3\na) b)"` (Issue 119's own subject row) ALSO matches this prefix
# pattern but its own adjacent content is blank; without the gate this would create a
# blank-content GridRow, the exact "resolved but empty" danger already avoided elsewhere in this
# session (ACS-Junior's own pre-115 Q8 shape), confirmed by the gate that this row stays
# completely untouched, leaving Issue 119's own scenario exactly as it already was.
# Issue 142 fix : same real paren-suffix widening as
# _GRID_LABEL_CELL_PATTERN above, for the identical reason, a glued sub-part label ("Q17)\na)")
# using ")" instead of "." was equally missed by Issue 133's own pass.
_GLUED_SUBPART_LABEL_PREFIX_PATTERN = re.compile(r"^Q?(\d{1,3})([a-zA-Z])?[.)]?(?=\s|$)")

# Issue 122 fix : a BARE sub-part marker with no digit at all , 
# `"a)"`, `"(a)"`, `"b)"`, `"(b)"`, used ONLY by parse_answer_key_grid()'s own rowspan-origin
# recovery pass, to recognize a real, confirmed shape: a row sitting immediately past a rowspan
# origin's own nominal shadow (`Raffles-Girls p34's own real "b)"` row, `<td>b)</td><td colspan=
# "4">22 x 23 = 506\nANS:22</td>...`, the rowspan="4" on `"Q7 a)"` covers only the 3 rows before
# this one). Real, confirmed reason this needs its OWN pattern, distinct from
# _GLUED_SUBPART_LABEL_PREFIX_PATTERN above: that pattern requires a digit; this shape has none
# at all, the sub-part letter alone is the whole label, with no way to recover which base
# question number it belongs to except its own position, directly after a real, already-known
# origin.
_BARE_SUBPART_MARKER_PATTERN = re.compile(r"^\(?[a-zA-Z]\)\.?$")

# Stricter sibling of _GRID_LABEL_CELL_PATTERN, used ONLY by _has_answer_grid_table_shape's page
# classification check (never by parse_answer_key_grid()'s own row parsing, once a page is already
# confirmed to be a grid), requires the real decoration every actual grid label observed has (a
# "Q" prefix or a trailing "." or ")"), rejecting a bare, undecorated number a real question's own
# data table can produce by pure coincidence. See _has_answer_grid_table_shape's own docstring for
# the real false-positive this was found fixing.
#
# Issue 133 fix : trailing punctuation widened from `\.` to `[.)]` on
# both branches, real, confirmed shape found investigating the 31-file zero-grid group: several
# real files decorate a grid label with a trailing ")" instead of "." (`"Q17)"`, and a few use a
# bare paren-suffixed number with no "Q" at all (`"11)"`). Confirmed via direct testing that this
# widening does NOT relax the real safety margin the original pattern existed for, a fully
# undecorated bare digit (`"1"`, no "Q", no trailing punctuation at all, `P6_Maths_2023_SA2_
# Peichun.pdf`'s own real header-row shape) still does not match either branch, so an ordinary
# data table's own coincidental small integers still can't false-trigger classification; that
# file's own gap is real but explicitly out of this fix's scope, not silently swept in.
_GRID_LABEL_CLASSIFICATION_PATTERN = re.compile(r"^(?:Q\d{1,3}[a-zA-Z]?[.)]?|\d{1,3}[a-zA-Z]?[.)])$")

# Issue 229 fix, part 1, see _has_answer_grid_table_shape()'s own
# docstring for the full real evidence. A real, exhaustive gap in the actual corpus data, not a
# guessed round number: every genuine "ANSWER KEY"-titled page's own real grid-label-adjacent
# value cell is <= 223 real characters (2,625 real label-value pairs checked); every confirmed
# worked-solution "grid-shaped" row's own adjacent cell is >= 302 real characters. 260 sits in
# that real, clean gap with margin on both sides.
_GRID_LABEL_VALUE_MAX_LEN = 260

# Issue 144 fix (the development log-01): the Q-PREFIXED branch only, deliberately
# narrower than _GRID_LABEL_CLASSIFICATION_PATTERN's own full alternation (which also accepts a
# bare, punctuation-decorated digit like "14.", that bare shape is a REAL, confirmed genuine
# question-page signal too, chij p37's own detached "14." title region, _has_question_like_
# content()'s own docstring, reusing the full pattern here would misclassify that real question
# page as a grid page instead). "Q"-prefixed is different: every real example checked across this
# whole codebase (`_GRID_LABEL_CELL_PATTERN`'s own docstring, every issue touching real grid
# rows) uses a literal "Q" prefix ONLY as an answer-key/marking-scheme convention ("Q1)", "Q16",
# "Q24"), no real QUESTION page's own printed number has EVER been observed using this prefix.
# Used ONLY by classify_page()'s own answer_key_grid check below, for a `title`-type region
# specifically (not a table cell, that's still `_GRID_LABEL_CELL_PATTERN`'s own job).
_ANSWER_KEY_STYLE_TITLE_LABEL_PATTERN = re.compile(r"^Q\d{1,3}[a-zA-Z]?[.)]?$")

# Issue 372 fix (development log, answer-key label regex fix brief), real signal that the
# REMAINDER of a page's own body content narrates a worked solution's own derivation, rather than
# posing a real question. A genuine question stem does not walk through multiple internally
# numbered solution steps and then answer itself; this shape is specific to a worked-solution
# page's own real, confirmed structure (`P6_Maths_2024_SA2_acsprimary.pdf` p47/p54/p50/p55/p57/
# p58/p59, `P6_Maths_2024_SA2_nanyang.pdf` p44, `PSLE Math 2015 Answer+Solution` p2/p3, every one
# of the 10 real "text"-typed-label cases found in the corpus-wide survey this fix ran before
# writing any pattern, per instruction).
_NUMBERED_NARRATIVE_STEP_PATTERN = re.compile(r"^\d{1,2}[.)]\s+\S")


def _has_narrative_worked_solution_shape(bounding_boxes: Optional[list]) -> bool:
    """
    Issue 372 fix (development log, answer-key label regex fix brief, directly continuing Issue
    371's Topic 4 verification finding, `id=3265`). Real gap found: `_ANSWER_KEY_STYLE_TITLE_LABEL_
    PATTERN` above only ever looks at `title`-typed regions, but a real, corpus-wide survey (run
    BEFORE writing this function, not generalized from the one known `id=3265` example) found the
    SAME real worked-solution page shape recurring with its own leading label typed as plain `text`
    instead, 10 of 12 real matching pages found corpus-wide, across 3 distinct real papers, not
    just the one file that first surfaced it. Real label strings confirmed on these pages: "9",
    "11a)", "13a)", "13b)", "15a)", "15b)", "16a)", "10", "19.", "28.", a mix of bare, lettered,
    and punctuation-decorated shapes, NOT limited to "Q"-prefixed labels the way the title-region
    check above deliberately is.

    A bare/lettered leading label ALONE is not a safe signal on its own, `_ANSWER_KEY_STYLE_TITLE_
    LABEL_PATTERN`'s own docstring already documents a real, confirmed false-positive this exact
    shape risks: chij p37's own real bare `"14."` `text` region (both `P6_Maths_2022_SA2_chij.pdf`
    and `P6_Maths_2025_SA2_chij.pdf` carry this shape) is a genuine QUESTION start, not a
    worked-solution label, re-checked directly against both real files' own raw reader output
    while building this fix, not assumed still true. What makes a real worked-solution page
    different, checked against every one of the 12 real matching pages found: its OWN remaining
    body content narrates 2 or more internally-numbered derivation steps ("1. Calculate...", "2.
    Combine...") and ends in a real answer/conclusion statement ("Answer: ..." or a "So, ..." line)
   , chij p37's own real remaining content (an MCQ-style word problem ending "Ans: [4]") has NEITHER
    of those two real signals present, confirmed directly: 0 numbered-step regions, and its own
    "Ans:" line does not satisfy the stricter "answer"/"so," check below (`"ans:"` alone is a normal,
    everyday real question-page marker across this ENTIRE corpus, deliberately NOT treated as this
    function's own answer-line signal, only a real, narrated "Answer:"/"So," CONCLUSION counts).

    Checked against the FIRST real body region (skipping header/footer/page_number furniture, the
    same real "administrative regions never count" convention `_segment_from_regions` already uses)
    for the label shape, and the REST of the page's own body regions for the narrative-step/
    answer-line signal, both must be true, matching the real, evidenced combination every one of
    the 12 corpus-wide matches actually has.
    """
    body_boxes = [
        bb for bb in (bounding_boxes or [])
        if bb.get("type") not in ("header", "footer", "page_number")
    ]
    if not body_boxes:
        return False

    first_content = (body_boxes[0].get("content") or "").strip()
    if not _GRID_LABEL_CELL_PATTERN.match(first_content):
        return False

    remaining = body_boxes[1:]
    numbered_step_hits = sum(
        1 for bb in remaining
        if _NUMBERED_NARRATIVE_STEP_PATTERN.match((bb.get("content") or "").strip())
    )
    # A real, caught-before-shipping false positive (found running this exact check against a
    # broader real sample than the 12 known matches, per this brief's own "verify candidates, don't
    # trust a pattern match alone" instruction applied to this NEW pattern too, not just the old
    # one): an earlier version used `"answer" in content.lower()` (substring, anywhere in the
    # text), real, ordinary PSLE instruction boilerplate ("Write your answers in the spaces
    # provided...", `P6_Maths_2024_SA2_nanyang.pdf` p44's own real text) ALSO contains that
    # substring and would have wrongly satisfied this check on a genuine multi-question page with
    # no worked-solution shape at all. `.startswith()` instead, a real worked-solution page's own
    # conclusion line always LEADS with "Answer:"/"So," (every one of the 12 real corpus matches
    # confirmed this directly), ordinary instruction text never does.
    has_answer_line = any(
        (bb.get("content") or "").strip().lower().startswith("answer")
        or (bb.get("content") or "").strip().lower().startswith("so,")
        for bb in remaining
    )
    return numbered_step_hits >= 2 and has_answer_line


# Issue 142 : a real, distinct shape found investigating Issue
# 127's own 119/123 secondary signal, NOT the same as either Issue 119's or 123's own original
# case (both bare-digit/marker rows with NO label decoration at all, ambiguous by
# design). This one has a real, unambiguous label right there in the cell, just never split into
# its own `<td>`: MinerU2.5 sometimes emits ONE `<td>` per real answer, carrying BOTH the label and
# the content glued together with no cell boundary at all (`P6_Maths_2024_WA1_Rosyth.pdf` p32's
# own real table: `<tr><td>Q16) 17/18</td></tr><tr><td>Q17) 28</td></tr>...`). Every row like this
# is a genuine ODD cell count (1 real `<td>`), so it was reaching the existing leading-blank-
# prepend heuristic (Issue 111), `starts_with_real_label` there uses `_GRID_LABEL_CELL_PATTERN`
# anchored at BOTH ends, so a cell containing a label PLUS real content never matches it, and the
# prepend fires: `["", "Q16) 17/18"]`, a blank label with real content, exactly the shape the main
# loop treats as a continuation of whatever chain is already open at that column. Confirmed via a
# random, unbiased 25-event sample of the real discrepancy log (not cherry-picked) that this shape
#, or its own genuine blank-content sibling, a bare label alone in its own cell (`"Q19)"`, real
# answer on the FOLLOWING row), accounts for the clear majority of real "continuation fragment, no
# open chain" events. The trailing punctuation is REQUIRED here (not optional, unlike
# _GRID_LABEL_CELL_PATTERN's own `[.)]?`), deliberately narrower: a bare digit glued directly to
# real content with NO separator at all (e.g. a genuine "3 apples were bought" sentence starting a
# content cell) must never be mistaken for a label, the same real ambiguity Issue 119's own
# entry already identified as unsafe to guess at. Content is optional (`.*`, DOTALL for real
# multi-line workings), a bare "Q19)" with nothing after it is real and expected (its own answer
# lands on a later row via the ordinary, unmodified continuation path).
#
# Real, caught-before-shipping false positive, found running this exact page's own diagnostic
# re-check, not left for the full corpus diff to surface first: a first version used `[.)]` (plain
# alternation) for the trailing punctuation, same as every other pattern in this family, but
# unlike those (all fully end-anchored, so a "." immediately followed by more digits can never
# satisfy their own `$`), THIS pattern's own trailing `(.*)$` happily consumes anything after the
# punctuation, including more digits, silently mis-splitting a genuine decimal answer like
# "7.1cm" into label "7." / content "1cm". Fixed with `\.(?!\d)`: a "." only counts as a real
# label terminator when NOT immediately followed by another digit; ")" needs no such guard, never
# used as a decimal separator in any real file checked.
_GLUED_LABEL_AND_CONTENT_IN_ONE_CELL_PATTERN = re.compile(
    r"^(Q?\s?\d{1,3}[a-zA-Z]?(?:\)|\.(?!\d)))\s*(.*)$", re.DOTALL
)


class _TableCellParser(HTMLParser):
    """
    Minimal HTML `<table>` cell extractor, stdlib HTMLParser, no new dependency, scoped to
    exactly the real shape confirmed in MinerU2.5's `table`-type region content: `<table><tr>
    <td>...</td>...</tr></table>`, no nested tables, `<br>`/`<br/>` inside a cell (confirmed real,
    the secondary-reader fallback shape) treated as an embedded newline, and a `colspan` attribute
    (confirmed real on GS005's header/spacer rows AND Catholic High's own answer working, Issue
    103 finding (c)) expanded into that many flat cells, the first carrying the real text, the
    rest empty placeholders, so column POSITION stays aligned across every row of the same table.
    Column position is exactly what parse_answer_key_grid() uses below to pair each label cell
    with its content cell, so this alignment is load-bearing, not cosmetic, this part of the
    parser's behavior is UNCHANGED from before Issue 111's fix, deliberately: GS005's own
    real rowspan/`max_pairs` alignment (see parse_answer_key_grid()'s own comments) depends on it
    staying exactly as it was. `rowspan` is NOT specially handled, GS005's one real rowspan use
    (a rowspan="18" spacer over an already-empty cell) still produces an empty cell in every row
    it visually spans when read as ordinary HTML row-by-row, which is exactly the "empty label,
    empty content -> skip" case parse_answer_key_grid() already needs for other reasons (see its
    own docstring). Deliberately unchanged, a full, real rowspan-aware cross-row parser is real,
    separate, NOT-yet-designed follow-up work (Issue 112, Issue 103
    finding (c)'s own Shape 3).

    `is_colspan_tail` (Issue 111, the development log): a new, parallel per-row list of
    bools, same length and position as `rows`' own cells, True at a flat position that exists
    ONLY because an earlier real `<td>` in the SAME row declared `colspan > 1`, False at every
    real `<td>`'s own position (whatever its own text is, including a blank real cell).
    Real, confirmed reason this is needed: Catholic High page 36's own answer rows (Issue 103
    finding (c)) have a `colspan="4"` on a CONTENT cell sitting between two real label/content
    pairs on the same row, flattening that colspan the way this parser already correctly does for
    GS005 is right (needed for cross-row alignment there), but parse_answer_key_grid()'s pairing
    loop previously had no way to tell "this flat position is real, pair-worthy content" apart
    from "this position is just that wide cell's own overflow, skip it, don't start a new pair
    here", `is_colspan_tail` is exactly that missing signal, consumed only by the pairing loop
    below, not by header-row detection or `max_pairs`/`column_offset` (both stay on the flat,
    padded `rows` cells, completely unchanged, so GS005's own real behavior is untouched).
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self.is_colspan_tail: list[list[bool]] = []
        # Issue 114 (Issue 112's own contamination-mitigation
        # follow-up): one entry per real `<td rowspan > 1>` cell found, (row_index, rowspan_value,
        # cell_text), row_index is the index this cell's own row WILL have once its `</tr>`
        # closes (== len(self.rows) at the moment this td closes, since every td in a row closes
        # before its own tr does). Deliberately NOT used to reshape `rows`/`is_colspan_tail`
        # themselves (that would be the full cross-row column-realignment fix, explicitly deferred
        #, see Issue 112), only consumed by _parse_html_table_rows() to compute
        # rowspan_shadow_rows, a much narrower signal.
        self.rowspan_origins: list[tuple[int, int, str]] = []
        self._current_row: Optional[list[str]] = None
        self._current_row_tail: Optional[list[bool]] = None
        self._current_cell: Optional[list[str]] = None
        self._current_colspan = 1
        self._current_rowspan = 1

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag == "tr":
            self._current_row = []
            self._current_row_tail = []
        elif tag in ("td", "th"):
            self._current_cell = []
            attrs_dict = dict(attrs)
            try:
                self._current_colspan = max(1, int(attrs_dict.get("colspan") or 1))
            except (TypeError, ValueError):
                self._current_colspan = 1
            try:
                self._current_rowspan = max(1, int(attrs_dict.get("rowspan") or 1))
            except (TypeError, ValueError):
                self._current_rowspan = 1
        elif tag == "br" and self._current_cell is not None:
            self._current_cell.append("\n")

    def handle_startendtag(self, tag: str, attrs: list) -> None:
        # Self-closing form, "<br/>", confirmed real (secondary-reader table markup); HTMLParser
        # does not route this through handle_starttag on its own.
        if tag == "br" and self._current_cell is not None:
            self._current_cell.append("\n")

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th"):
            if self._current_row is not None and self._current_cell is not None:
                cell_text = "".join(self._current_cell)
                self._current_row.append(cell_text)
                self._current_row_tail.append(False)
                for _ in range(self._current_colspan - 1):
                    self._current_row.append("")
                    self._current_row_tail.append(True)
                if self._current_rowspan > 1:
                    self.rowspan_origins.append((len(self.rows), self._current_rowspan, cell_text))
            self._current_cell = None
            self._current_colspan = 1
            self._current_rowspan = 1
        elif tag == "tr":
            if self._current_row is not None:
                self.rows.append(self._current_row)
                self.is_colspan_tail.append(self._current_row_tail)
            self._current_row = None
            self._current_row_tail = None


@dataclass
class _ParsedTableRows:
    """Real output of _parse_html_table_rows(), the flat, colspan-padded cells (unchanged shape
    from before Issue 111) plus the parallel is_colspan_tail markers (Issue 111) and, now,
    rowspan_shadow_rows (Issue 114). Kept as one small object, not loose parallel lists/sets,
    so a caller can't accidentally zip them against the wrong row after some future edit."""
    rows: list[list[str]]
    is_colspan_tail: list[list[bool]]
    # Issue 114 (Issue 112's own contamination-mitigation
    # follow-up, NOT the full Shape-3 structural fix, that stays explicitly deferred, see
    # Issue 112): the set of row indices that fall AFTER the origin row of some real
    # `rowspan > 1` cell whose own text is non-blank, within that cell's own vertical span
    # (origin_row itself is NEVER included, its own row still needs normal pairing, e.g.
    # ACS-Junior's real Q7 answer sits ON the origin row and must stay reachable).
    #
    # Real, confirmed reason this is scoped to NON-blank origins only, not every rowspan: GS005's
    # own real rowspan usage (Booklet A/B's column running out of rows before Paper 2's own) is a
    # BLANK spacer cell, `<td rowspan="18"></td>`, confirmed real, see _TableCellParser's own
    # docstring, and its shadow rows carry real, unrelated Paper 2 answers in OTHER columns that
    # must stay reachable; a blanket "skip every rowspan's shadow" rule would have silently
    # destroyed those. A NON-blank origin (e.g. ACS-Junior's "Q7"/"Q8", Catholic High's "Q17\na)",
    # Tao-Nan's "Q17") is the real, confirmed signal separating "this rowspan reserves a column
    # for a nested/embedded sub-table" (Shape 3) from "this rowspan is a pure structural spacer"
    # (GS005), checked directly against all three known real Shape 3 occurrences plus GS005's own
    # shape before relying on it, not assumed (Issue 114).
    #
    # Scope limit: this suppresses the ENTIRE shadow row, not just the
    # specific columns the rowspan itself covers, correct for every real row observed so far
    # (each shadow row in all three known cases is entirely consumed by the nested sub-table, with
    # nothing unrelated sharing it), but a future file with real, unrelated content sharing a
    # shadow row with a non-blank rowspan's nested table would also lose that content. Recovering
    # per-column precision needs the real cross-row column-realignment fix Issue 112 already
    # deferred, not attempted here.
    rowspan_shadow_rows: set[int]
    # Issue 115 (Issue 114's own recovery follow-up): the raw,
    # UNREDUCED per-origin records `_TableCellParser.rowspan_origins` produced, `(row_index,
    # rowspan_value, cell_text)` for every real `<td rowspan > 1>` cell, blank origins included.
    # Kept separately from `rowspan_shadow_rows` (which only needs the reduced row-index set for
    # the Issue 114 skip) because recovering real content needs to know WHICH origin a shadow
    # row's content belongs to, not just that it should be skipped, see parse_answer_key_grid()'s
    # own recovery pass for how this is used.
    rowspan_origins: list[tuple[int, int, str]]


def _parse_html_table_rows(table_html: str) -> _ParsedTableRows:
    """Returns one list of raw (unescaped-later, see _clean_answer_cell_text) cell strings per
    real `<tr>` row in table_html, plus the parallel is_colspan_tail markers, the
    rowspan_shadow_rows set, and the raw rowspan_origins list, see _TableCellParser's own
    docstring for the real, confirmed shape this is scoped to, and _ParsedTableRows' own docstring
    for rowspan_shadow_rows/rowspan_origins specifically."""
    parser = _TableCellParser()
    parser.feed(table_html)
    rowspan_shadow_rows: set[int] = set()
    for row_index, rowspan, text in parser.rowspan_origins:
        if text.strip():
            rowspan_shadow_rows.update(range(row_index + 1, row_index + rowspan))
    return _ParsedTableRows(
        rows=parser.rows, is_colspan_tail=parser.is_colspan_tail,
        rowspan_shadow_rows=rowspan_shadow_rows, rowspan_origins=parser.rowspan_origins,
    )


# Marks-annotation bracket, e.g. "(1)"/"[2]"/"(2 marks)", Issue 6 ("marks-annotation bleed:
# '[2]' next to an answer becomes part of it, '5 [2]' -> '52'"). Not directly observed inside a
# grid CELL in the real data pulled for this design (chij/GS005's own grid cells), but confirmed
# real in this exact subset on the QUESTION side ("Ans: (a) [1]", chij's own segmented question
# text), extended here defensively per explicit instruction, not blindly, since the same paper
# family is confirmed to use this annotation shape. Digit-only bracket content, optionally with
# "marks"/"mark", bounded by whitespace/string edges so a real answer that happens to contain a
# parenthesized digit mid-expression (none observed, but not assumed impossible) is less likely to
# be caught by accident.
_MARKS_ANNOTATION_PATTERN = re.compile(
    r"(?:^|(?<=\s))[\(\[]\s*\d{1,3}\s*marks?\s*[\)\]](?=\s|$)"
    r"|(?:^|(?<=\s))[\(\[]\s*\d{1,3}\s*[\)\]](?=\s|$)",
    re.IGNORECASE,
)


def _clean_answer_cell_text(raw: str) -> str:
    """
    Cleans one answer-key-grid cell's raw content into GridRow.answer_text, the same class of
    cleanup values_equivalent()'s _strip_currency helper already does for general answer
    comparison, reused (not re-derived, so the two can't drift apart) and extended here for the
    grid-parsing context specifically (Issue 6/7, explicit instruction): HTML entities (e.g.
    "&#36;" for "$", confirmed real in the secondary reader's raw table markup) unescaped first,
    _TableCellParser's embedded "\n" markers already stand in for "<br>"/"<br/>" by the time this
    runs, then a marks-annotation bracket stripped (_MARKS_ANNOTATION_PATTERN, see its own
    docstring), then a leading/trailing currency symbol stripped via the shared _strip_currency.

    Deliberately does NOT strip other real units (cm, g, %, degree signs, all confirmed present
    in this exact grid data: "9810g", "38°", "50.2%", "366.74 cm"), those are part of the actual
    printed answer, not scanning noise, and stripping them would destroy real answer-key content
    (Issue 24) with no real evidence any of this subset's grid cells need it stripped.
    """
    text = html.unescape(raw)
    text = _MARKS_ANNOTATION_PATTERN.sub("", text)
    text = _strip_currency(text.strip())
    return text.strip()


# Real, confirmed shape (Issue 100's grid-side follow-up): a flat-text
# reader (DeepSeek-OCR) never produces `table`-type bounding_boxes at all, confirmed on all 5 real
# answer-key-grid pages in this project's own validation subset, `bounding_boxes` is always `[]` , 
# but its own `extracted_text` DOES embed real, well-formed HTML `<table>...</table>` markup
# directly inline, with each real section caption ("BOOKLET A", "PAPER2") printed as plain text
# immediately before its own table. `structure.get("has_table_markup")` already flags this case (a
# signal some earlier Stage B work anticipated but parse_answer_key_grid() was never extended to
# use). Same real problem shape MinerU2.5-vs-flat-text already had on the QUESTION side (Issue
# 86's `_segment_from_regions`/`_segment_from_flat_text` split), same fix shape here: a second,
# flat-text path feeding the SAME downstream parsing (_parse_html_table_rows(), the cell-pairing/
# label-matching logic in parse_answer_key_grid() itself), not a parallel reimplementation.
_EMBEDDED_HTML_TABLE_PATTERN = re.compile(r"<table\b.*?</table>", re.IGNORECASE | re.DOTALL)


def _synthesize_grid_regions_from_flat_text(text: str) -> list[dict]:
    """
    Builds a synthetic `page_regions` list (the same `[{"type": ..., "content": ...}, ...]` shape
    parse_answer_key_grid() already expects from real bounding_boxes) out of a flat-text reader's
    own extracted_text, so the EXISTING parsing logic in parse_answer_key_grid() runs completely
    unchanged on either source; this function's only job is sourcing real `table_caption`/`table`
    regions from plain text instead of structured region data.

    For each real `<table>...</table>` block found (in document order), the text in the gap
    immediately BEFORE it (back to the end of the previous table, or the start of the page) is
    searched for a real section-label keyword via the SAME _normalize_section_label() the
    bounding_boxes path already uses, shared, not re-derived, so the two paths can never drift
    into different canonical spellings. Confirmed real and correct against this project's own
    subset: the very first table's gap also contains real header noise ("ANSWER KEY", "YEAR :
    2025", ...) ahead of the real caption, which _normalize_section_label()'s own keyword search
    correctly ignores; a CONTINUATION page (chij p46/p47's real shape) has no caption text in its
    gap at all, no table_caption region is emitted for it, exactly matching MinerU2.5's own real
    behavior on the same pages, so the caller's existing section carry-forward
    (locate_embedded_answer_section()'s last_real_section_label) picks it up unchanged.
    """
    regions: list[dict] = []
    last_end = 0
    for match in _EMBEDDED_HTML_TABLE_PATTERN.finditer(text):
        gap = text[last_end:match.start()]
        caption = _normalize_section_label(gap)
        if caption is not None:
            regions.append({"type": "table_caption", "content": caption})
        regions.append({"type": "table", "content": match.group(0)})
        last_end = match.end()
    return regions


def _is_header_row_for_value_row_shape(cells: list[str]) -> bool:
    """
    Issue 132 fix : real, precise detection for a row that is a
    header row of ALL labels (e.g. "Q1","Q2","Q3",...,"Q10"), immediately followed, as its own
    separate table row, by a value row of the same width holding the real answers, column-
    aligned. Confirmed real and directly diagnosed (not guessed) across 10+ real files spanning
    6 schools and 5 years, plus 6 of 7 held-out evaluation files: `parse_answer_key_grid()`'s own
    existing pairing logic assumes every row independently alternates (label, content, label,
    content, ...), the real shape every prior grid fix (Shapes 1-5) was built against, and
    garbles this different, header-row/value-row shape into nonsense pairs (`(Q1,"Q2")`,
    `(3,"3")`, ...) when it's actually two full rows, not alternating pairs within one.

    Real, deliberate distinguishing test from an ordinary MCQ value row, not just "every cell
    matches _GRID_LABEL_CELL_PATTERN": that pattern also matches a bare undecorated digit (e.g.
    "3"), so a genuine value row ("3,3,4,2,1,2,4,3,2,4") would ALSO match cell-by-cell if checked
    in isolation. The real, reliable signal that tells the two apart: a header row's own numbers
    are always a strictly increasing run (Q1,Q2,Q3,... or a later table's Q11,Q12,...); a real MCQ
    answer row (option values 1-4, real, checked across every file examined) never is. Also
    requires at least one cell to carry the real "Q" prefix, and at least 3 cells, so a short,
    coincidental 2-cell run doesn't false-trigger.
    """
    if len(cells) < 3:
        return False
    numbers: list[int] = []
    has_q_prefix = False
    for cell in cells:
        stripped = cell.strip()
        match = _GRID_LABEL_CELL_PATTERN.match(stripped)
        if not match:
            return False
        if stripped.upper().startswith("Q"):
            has_q_prefix = True
        numbers.append(int(match.group(1)))
    if not has_q_prefix:
        return False
    return all(numbers[i] < numbers[i + 1] for i in range(len(numbers) - 1))


def parse_answer_key_grid(
    page_regions: Optional[list], page_index: int, extracted_text: str = "",
) -> list[GridRow]:
    """Parse one answer-key grid page's table regions into GridRows.

    Built against the two grid shapes seen in the papers: one captioned table per section
    (chij p44-47, with `table_caption` regions for Booklet A, Booklet B and Paper 2), and a single
    table with the sections side by side and `colspan`/`rowspan` structure (GS005).

    Data rows are a flat, alternating sequence of (label, content) cells, and cells are paired by
    position (0/1, 2/3, ...). Position is the only reliable signal, because a content cell such
    as a bare MCQ answer "2" can look exactly like a label.

    Each pair falls into one of three cases:
      1. The label matches _GRID_LABEL_CELL_PATTERN. A GridRow is created with the question number
         normalised ("Q1" -> "1", "22a." -> "22a").
      2. Label and content are both blank. Skipped: these are rowspan spacer cells, or trailing
         blank pairs on rows shorter than the table.
      3. The label is blank but the content is not. This is a cross-page continuation fragment,
         kept as a GridRow with question_number="". This function has no cross-page context, so
         the caller resolves it (see locate_embedded_answer_section()).
    Any other label (for example "Question No.") is header noise. It is logged at debug level and
    skipped.

    Errors are isolated per table region and per row (Issue 62). A malformed row is logged and
    skipped, and the rest of the page is still parsed.

    Args:
        page_regions: the reader's regions for the page. When none has type "table" (DeepSeek-OCR
            never produces structured regions), regions are built from the HTML tables embedded in
            `extracted_text` instead (Issue 100). Parsing then proceeds in the same way.
        page_index: the page's index, recorded on each GridRow.
        extracted_text: the reader's flat text for the page, used only for that fallback.

    Returns:
        The GridRows found on the page.
    """
    has_real_table_region = any((r.get("type") == "table") for r in (page_regions or []))
    if not has_real_table_region and extracted_text:
        page_regions = _synthesize_grid_regions_from_flat_text(extracted_text)

    rows: list[GridRow] = []
    current_section_label: Optional[str] = None
    # Column indices for header-row/value-row pairs (Issues 132 and 133). The offset is scoped to
    # the page, not the table region, because one page can split its answers across separate table
    # regions (for example Q1-10 and Q11-15). A per-region offset would restart at 0 and collide.
    # The base is a large page-scoped constant rather than 0. Low column indices are shared by many
    # unrelated tables in a file, and a stray continuation fragment from a later page could chain
    # onto them (one file's Q1 collected 11 unrelated fragments this way). This uses the same
    # out-of-range isolation as origin-recovery rows (Issue 121).
    header_value_column_offset = 100_000 + page_index * 1_000
    # Per-column section labels. Some grids (GS005) have no caption regions and print the section
    # names as cells in the first row, one per column. This is filled in when such a header row is
    # found. Rows fall back to current_section_label (the caption-based shape) when their column
    # has no entry. The two mechanisms did not overlap in any file checked.
    section_by_column: dict[int, str] = {}
    for region in page_regions or []:
        if region.get("type") == "table_caption":
            current_section_label = _normalize_section_label(region.get("content") or "") or (region.get("content") or "").strip() or None
            continue
        if region.get("type") == "title":
            # Some files caption grid tables with a "title" region instead of "table_caption"
            # (Issue 104). There is no raw-text fallback here: title regions also carry unrelated
            # headings such as "ANSWER KEY", so only a recognised section keyword updates
            # current_section_label. Other titles are skipped.
            normalized = _normalize_section_label(region.get("content") or "")
            if normalized is not None:
                current_section_label = normalized
            continue
        if region.get("type") == "text":
            # Some files print "PAPER 1", "BOOKLET A" and "BOOKLET B" as separate short text
            # regions rather than as a caption or title (Issue 137). Without this, every row on
            # such a page got section_label=None and no answers matched. Text regions are common
            # and can hold worked solutions or question text, so only short ones (30 characters or
            # fewer) are checked for a section keyword.
            content = (region.get("content") or "").strip()
            if len(content) <= 30:
                normalized = _normalize_section_label(content)
                if normalized is not None:
                    current_section_label = normalized
            continue
        if region.get("type") != "table":
            continue
        table_html = region.get("content") or ""
        if not table_html.strip():
            continue

        try:
            parsed = _parse_html_table_rows(table_html)
        except Exception as e:  # noqa: BLE001, one malformed table region must not fail the page
            logger.warning(
                "parse_answer_key_grid: page %d — could not parse a table region's HTML, "
                "skipped: %s: %s", page_index, type(e).__name__, e,
            )
            continue
        table_rows, table_rows_tail = parsed.rows, parsed.is_colspan_tail
        rowspan_shadow_rows = parsed.rowspan_shadow_rows
        rowspan_origins = parsed.rowspan_origins
        # Rows that are themselves rowspan origins (Issue 125). The main pairing loop below skips
        # them. Pairing an origin there would create an incomplete GridRow (the adjacent cell only,
        # without its shadow rows), and the fuller origin-recovery pass (Issue 121) would then skip
        # the question because a row already exists.
        rowspan_origin_row_indices = {r for r, _, _ in rowspan_origins}

        # Fix odd-length rows before the table-wide width is computed. A continuation row can be
        # missing its leading blank label cell (chij p47), so a blank cell is prepended. This is
        # done only when:
        #   - no cell in the row declares colspan > 1 (Issue 111). An odd count caused by a colspan
        #     content cell between two pairs has a different cause, and prepending there would
        #     shift a label into a content slot and drop working;
        #   - the first cell is not already a question label (Issue 113). A row of complete pairs
        #     plus one trailing empty cell (Nanyang p36) would otherwise be shifted the same way.
        # Only a row's first pair is affected.
        fixed_rows: list[list[str]] = []
        fixed_rows_tail: list[list[bool]] = []
        for row_index, (cells, tail) in enumerate(zip(table_rows, table_rows_tail)):
            has_colspan = any(tail)
            # A label and its content glued into one cell (Issue 142, see
            # _GLUED_LABEL_AND_CONTENT_IN_ONE_CELL_PATTERN). Checked before the blank-prepend step
            # and only for rows with exactly one real cell. Splitting here means every later step
            # sees an ordinary two-cell (label, content) row. A real cell is one that is not a
            # colspan tail (Issue 150): MinerU2.5 also emits the glued shape with an explicit
            # colspan, which _TableCellParser expands into blank placeholder cells.
            real_cell_indices = [i for i, is_tail in enumerate(tail) if not is_tail]

            # A row where every real cell carries its own glued "label. value" pair (Issue 154,
            # Tao-Nan p32: `<td>Q12. 2</td><td>Q13. 1</td>...`). Checked first, and only when two
            # or more real cells all match, so it cannot overlap with the single-cell cases below.
            # The row is expanded into ordinary (label, content) pairs, one per cell.
            multi_cell_glued_matches = None
            if len(real_cell_indices) > 1:
                candidate_matches = [
                    _GLUED_LABEL_AND_CONTENT_IN_ONE_CELL_PATTERN.match(cells[i])
                    for i in real_cell_indices
                ]
                # Every cell must also have non-empty content. The pattern's content group is
                # optional, so an already-split label such as "Q2)" matches with empty content, and
                # the label shape also matches a worked solution that starts a numbered list ("1.
                # Assume the side length..."). Requiring content in every cell keeps ordinary
                # two-cell (label, content) rows out of this path (acsprimary p42).
                if all(
                    m is not None and m.group(2).strip() for m in candidate_matches
                ):
                    multi_cell_glued_matches = candidate_matches

            if multi_cell_glued_matches is not None:
                new_cells: list[str] = []
                for m in multi_cell_glued_matches:
                    new_cells.extend([m.group(1), m.group(2)])
                logger.debug(
                    "parse_answer_key_grid: page %d row %d — %d real cells each independently "
                    "glued (Issue 154): labels=%r", page_index, row_index,
                    len(multi_cell_glued_matches), [m.group(1) for m in multi_cell_glued_matches],
                )
                fixed_rows.append(new_cells)
                fixed_rows_tail.append([False] * len(new_cells))
                continue

            glued_match = (
                _GLUED_LABEL_AND_CONTENT_IN_ONE_CELL_PATTERN.match(cells[real_cell_indices[0]])
                if len(real_cell_indices) == 1 else None
            )
            glued_pad = 0
            # The same glued shape next to one or more empty cells (Issue 153, Nan-Chiau p34:
            # `<td>Q16) 967</td><td></td>`). These are empty `<td>` cells, not colspan tails, so
            # the single-cell check above does not apply. The row keeps its original cell count,
            # padded with `glued_pad` blank tail cells after the label/content pair. Shrinking it
            # to two cells would change its column_offset (`max_pairs - len(cells) // 2`), move the
            # question into a neighbouring column and let a stray fragment from a later page
            # corrupt that neighbour's answer (MGS p36). This path changes content recognition
            # only, never column arithmetic. When the check above already narrowed the row to one
            # cell, glued_pad is 0.
            if glued_match is None and len(real_cell_indices) > 1:
                non_blank_real_indices = [i for i in real_cell_indices if cells[i].strip()]
                if len(non_blank_real_indices) == 1:
                    glued_match = _GLUED_LABEL_AND_CONTENT_IN_ONE_CELL_PATTERN.match(
                        cells[non_blank_real_indices[0]]
                    )
                    if glued_match is not None:
                        glued_pad = len(cells) - 2
                        logger.debug(
                            "parse_answer_key_grid: page %d row %d — single-non-blank-real-cell "
                            "%r (plus %d genuinely blank real cell(s)) split into a real label/"
                            "content pair (Issue 153): label=%r content=%r", page_index,
                            row_index, cells[non_blank_real_indices[0]], len(real_cell_indices) - 1,
                            glued_match.group(1), glued_match.group(2)[:60],
                        )
            if glued_match is not None:
                if glued_pad == 0 and len(real_cell_indices) == 1:
                    logger.debug(
                        "parse_answer_key_grid: page %d row %d — single-real-cell %r split into a "
                        "real label/content pair (Issue 142/150): label=%r content=%r",
                        page_index, row_index, cells[real_cell_indices[0]],
                        glued_match.group(1), glued_match.group(2)[:60],
                    )
                # (the Issue 153 case, glued_pad > 0, is logged above)
                cells = [glued_match.group(1), glued_match.group(2)] + [""] * glued_pad
                tail = [False, False] + [True] * glued_pad
            else:
                starts_with_real_label = (
                    bool(cells) and bool(_GRID_LABEL_CELL_PATTERN.match(cells[0].strip()))
                )
                if len(cells) % 2 == 1 and not has_colspan and not starts_with_real_label:
                    logger.debug(
                        "parse_answer_key_grid: page %d row %d has an odd cell count (%d) — "
                        "treating as a continuation row missing its leading blank label cell "
                        "(confirmed real shape, chij p47).", page_index, row_index, len(cells),
                    )
                    cells = [""] + cells
                    tail = [False] + tail
            fixed_rows.append(cells)
            fixed_rows_tail.append(tail)

        # Header-row/value-row pairs (Issue 132): a row of question labels followed by a row of
        # answers (see _is_header_row_for_value_row_shape()). They are detected before max_pairs is
        # computed and excluded from it, because they are not (label, content) rows and would
        # otherwise inflate max_pairs for other rows in the table.
        header_value_pairs_consumed: set[int] = set()
        # header_value_column_offset is not reset here. It is scoped to the page (see above), so
        # several header/value blocks on one page get non-overlapping column ranges.
        for row_index in range(len(fixed_rows) - 1):
            if row_index in header_value_pairs_consumed:
                continue
            header_cells = fixed_rows[row_index]
            value_cells = fixed_rows[row_index + 1]
            # Trim trailing padding first. A table with stacked header/value pairs of different
            # widths (for example Q1-8 then Q9-15) pads the narrower pair with blank cells. A blank
            # cell never matches _GRID_LABEL_CELL_PATTERN, so the pair would not be detected
            # (Nanyang lost Q10, Q12 and Q14 this way). Only cells blank in both rows at the same
            # trailing position are trimmed, so no value is discarded.
            trim = 0
            while (
                trim < len(header_cells) and trim < len(value_cells)
                and not header_cells[-1 - trim].strip() and not value_cells[-1 - trim].strip()
            ):
                trim += 1
            if trim:
                header_cells = header_cells[:len(header_cells) - trim]
                value_cells = value_cells[:len(value_cells) - trim]
            if len(header_cells) != len(value_cells):
                continue
            if not _is_header_row_for_value_row_shape(header_cells):
                continue
            for col, (label_cell, value_cell) in enumerate(zip(header_cells, value_cells)):
                label_match = _GRID_LABEL_CELL_PATTERN.match(label_cell.strip())
                number = label_match.group(1) + (label_match.group(2) or "")
                rows.append(
                    GridRow(
                        question_number=number,
                        answer_text=_clean_answer_cell_text(value_cell),
                        raw_cell_content=value_cell,
                        page_index=page_index,
                        column_index=header_value_column_offset + col,
                        section_label=current_section_label,
                    )
                )
            header_value_pairs_consumed.add(row_index)
            header_value_pairs_consumed.add(row_index + 1)
            logger.debug(
                "parse_answer_key_grid: page %d rows %d-%d recognized as a real header-row/"
                "value-row pair (Issue 132) — %d (label, value) pairs zipped column-by-column "
                "at offset %d, not paired within-row.", page_index, row_index, row_index + 1,
                len(header_cells), header_value_column_offset,
            )
            header_value_column_offset += len(header_cells)

        # Right-align short rows against the widest row. With HTML rowspan, a spanning cell is
        # declared only once, so later rows have fewer cells, missing from the start. Without an
        # offset, their pairs land in the wrong column_index (GS005's Paper 2 "16a."/"16b."
        # collided with Booklet B's "16."). Tables without short rows get offset 0 everywhere.
        max_pairs = max(
            (len(cells) // 2 for i, cells in enumerate(fixed_rows) if i not in header_value_pairs_consumed),
            default=0,
        )

        # Section-header rows (GS005): no label cell matches a question number, but at least one
        # normalises to a section keyword ("Paper 1 Booklet A", "Paper 2"). Sections are recorded
        # per column, because one such row sets several columns at once. The row itself is then
        # skipped from normal processing.
        header_rows: set[int] = set()
        for row_index, cells in enumerate(fixed_rows):
            column_offset = max_pairs - (len(cells) // 2)
            pair_count = len(cells) // 2
            any_real_label = any(
                _GRID_LABEL_CELL_PATTERN.match(cells[2 * j].strip()) for j in range(pair_count)
            )
            if any_real_label:
                continue
            found_any_section = False
            for j in range(pair_count):
                label = _normalize_section_label(cells[2 * j].strip())
                if label is not None:
                    section_by_column[column_offset + j] = label
                    found_any_section = True
            if found_any_section:
                header_rows.add(row_index)
                logger.debug(
                    "parse_answer_key_grid: page %d row %d recognized as a section-header row "
                    "(GS005's real shape) — section_by_column now %s",
                    page_index, row_index, section_by_column,
                )

        rows_before_this_table = len(rows)
        for row_index, cells in enumerate(fixed_rows):
            if row_index in header_rows:
                continue
            if row_index in header_value_pairs_consumed:
                # Already emitted column by column above (Issue 132), so it is not paired again
                # within the row.
                continue
            if row_index in rowspan_shadow_rows:
                # Rows inside a rowspan origin's span form a nested sub-table (Issue 114), not
                # top-level (label, content) rows, so they are skipped like header rows. Small
                # integers in the sub-table's data would otherwise match _GRID_LABEL_CELL_PATTERN
                # and be stored as answers to questions that do not exist (seen in ACS-Junior,
                # Catholic High and Tao-Nan). Useful content in these rows is recovered by the
                # origin-recovery pass below. The origin row itself is never in this set and is
                # paired normally.
                continue
            tail = fixed_rows_tail[row_index]
            column_offset = max_pairs - (len(cells) // 2)
            try:
                # Walk pairs rather than a fixed stride of 2 (Issue 111). After each pair, `i`
                # skips any colspan-tail cells that belong to the content cell just consumed, so a
                # wide content cell (colspan="4", Catholic High p36) does not push the next label
                # into a content slot. For rows without colspan tails, `i` advances by 2 and
                # `this_column` equals `i // 2`, the same as a fixed stride. max_pairs,
                # column_offset and header-row detection still use the flat, colspan-padded cell
                # positions.
                i = 0
                pair_index = 0
                while i < len(cells) - 1:
                    label_raw, content_raw = cells[i].strip(), cells[i + 1]
                    this_column = column_offset + pair_index
                    effective_section = section_by_column.get(this_column, current_section_label)
                    match = _GRID_LABEL_CELL_PATTERN.match(label_raw)
                    if (
                        not match and label_raw and content_raw.strip()
                        and row_index not in rowspan_origin_row_indices
                    ):
                        # A number with a glued sub-part marker on an ordinary row ("8.\n(a)\n(b)",
                        # Nan-Hua p34; Issue 125). None of these rows has a sub-table header beside
                        # it, so the adjacent cell is the answer and the pair is used as it is. Two
                        # guards:
                        #   - rowspan origins are excluded, because the origin-recovery pass below
                        #     (Issue 121) handles them more completely, and a partial row created
                        #     here would make that pass skip them;
                        #   - the content must be non-blank, so a label such as "Q3\na) b)" with
                        #     blank content (Raffles-Girls p33, Issue 119) does not create an empty
                        #     answer.
                        match = _GLUED_SUBPART_LABEL_PREFIX_PATTERN.match(label_raw)
                    if match:
                        number = match.group(1) + (match.group(2) or "")
                        rows.append(
                            GridRow(
                                question_number=number,
                                answer_text=_clean_answer_cell_text(content_raw),
                                raw_cell_content=content_raw,
                                page_index=page_index,
                                column_index=this_column,
                                section_label=effective_section,
                            )
                        )
                    elif not label_raw and content_raw.strip():
                        rows.append(
                            GridRow(
                                question_number="",
                                answer_text=_clean_answer_cell_text(content_raw),
                                raw_cell_content=content_raw,
                                page_index=page_index,
                                section_label=effective_section,
                                column_index=this_column,
                            )
                        )
                    elif not label_raw and not content_raw.strip():
                        pass  # structural spacer (rowspan/short row), nothing here, see docstring
                    else:
                        logger.debug(
                            "parse_answer_key_grid: page %d row %d — label cell %r doesn't match "
                            "a recognized question-number shape, skipped (expected header/caption "
                            "noise): content=%r", page_index, row_index, label_raw, content_raw[:60],
                        )
                    step = 2
                    j = i + 2
                    while j < len(cells) and tail[j]:
                        step += 1
                        j += 1
                    i += step
                    pair_index += 1
            except Exception as e:  # noqa: BLE001, per-row isolation, Issue 62's principle
                logger.warning(
                    "parse_answer_key_grid: page %d row %d — malformed, skipped, rest of the "
                    "page's real rows still processed: %s: %s",
                    page_index, row_index, type(e).__name__, e,
                )
                continue

        # Recover final-answer content from rowspan shadow rows (Issues 115 and 117). A shadow row
        # with exactly one non-blank real cell is attached to the origin's GridRow for the same
        # question number. The search covers only this table's rows (`rows_before_this_table`), so
        # a same-numbered row from another table is never used.
        #
        # candidates_per_row counts, for each row, the rowspan origin cells that match
        # _GRID_LABEL_CELL_PATTERN, that is, how many different question numbers share the row
        # (ACS-Junior's Q7/Q8 row scores 2; Nan-Hua's and Tao-Nan's Q17 rows score 1).
        #   Rule A (blank target, Issue 115): always safe, as there is nothing to overwrite.
        #     Handles ACS-Junior's Q8.
        #   Rule B (non-blank target, Issue 117): the recovered text is appended, never
        #     overwritten, and only when the origin is the row's only matching rowspan cell
        #     (candidates_per_row == 1). With a sibling number on the row (ACS-Junior's Q7), the
        #     shadow content could belong to either, so it is left alone. Handles Nan-Hua's Q17
        #     (plain continuation text in the shadow row) and Tao-Nan's Q17.
        # An origin whose text only matches the glued sub-part prefix (Catholic High's "Q17\na)")
        # has no target GridRow. The Issue 118 path below handles it.
        candidates_per_row: dict[int, int] = {}
        for candidate_row_index, _candidate_rowspan, candidate_text in rowspan_origins:
            if _GRID_LABEL_CELL_PATTERN.match(candidate_text.strip()):
                candidates_per_row[candidate_row_index] = candidates_per_row.get(candidate_row_index, 0) + 1

        # The trailing sub-part scan below stops at any row that is itself a different rowspan
        # origin (Issue 122, using rowspan_origin_row_indices from above). Otherwise back-to-back
        # origins, such as Nan-Hua p35's "10.\n(a)" and "(b)", would have "(b)" absorbed into the
        # first answer.

        for origin_row_index, rowspan, origin_text in rowspan_origins:
            if not origin_text.strip():
                continue
            label_match = _GRID_LABEL_CELL_PATTERN.match(origin_text.strip())
            extracted_via_prefix = False
            if label_match:
                origin_number = label_match.group(1) + (label_match.group(2) or "")
            else:
                # The origin text does not fully match a label, so try the narrower prefix pattern
                # for a number with a glued sub-part marker (Issue 118). This fallback applies to
                # the recovery pass only; the blank-prepend check above still uses full matches.
                prefix_match = _GLUED_SUBPART_LABEL_PREFIX_PATTERN.match(origin_text.strip())
                if not prefix_match:
                    continue
                origin_number = prefix_match.group(1) + (prefix_match.group(2) or "")
                extracted_via_prefix = True
            recovered_parts: list[str] = []
            if extracted_via_prefix:
                # When the origin's row has exactly one other real cell, the row is a clean (label,
                # content) pair and that cell holds the first part of the answer (Nan-Hua's
                # "10.\n(a)" row), so it is included (Issue 118). Rows with more real cells
                # (Catholic High's "Q17\na)", Raffles-Girls' "Q7 a)") have a sub-table header
                # beside the origin rather than an answer, so the adjacent cell is left out.
                origin_row_cells = fixed_rows[origin_row_index]
                origin_row_tail = fixed_rows_tail[origin_row_index]
                if origin_row_tail.count(False) == 2:
                    real_positions = [i for i, is_tail in enumerate(origin_row_tail) if not is_tail]
                    adjacent_text = _clean_answer_cell_text(origin_row_cells[real_positions[1]])
                    if adjacent_text:
                        recovered_parts.append(adjacent_text)
            for shadow_row_index in range(origin_row_index + 1, origin_row_index + rowspan):
                if shadow_row_index >= len(fixed_rows):
                    break
                shadow_cells = fixed_rows[shadow_row_index]
                shadow_tail = fixed_rows_tail[shadow_row_index]
                # A whole-row-spanning row has exactly one non-blank real cell (Issue 117). This
                # counts non-blank cells rather than using tail.count(False) == 1, because the
                # blank-prepend step above also runs on odd shadow rows: a single-cell continuation
                # row (Nan-Hua p37's Q17) gains a blank cell and would otherwise count as two.
                non_blank_real_positions = [
                    i for i, is_tail in enumerate(shadow_tail)
                    if not is_tail and _clean_answer_cell_text(shadow_cells[i])
                ]
                if len(non_blank_real_positions) == 1:
                    recovered_parts.append(_clean_answer_cell_text(shadow_cells[non_blank_real_positions[0]]))

            # Sub-part rows just after the rowspan's nominal span (Issue 122). The rowspan does not
            # cover them (Raffles-Girls p34: rowspan="4" on "Q7 a)" covers three data rows, and the
            # next row, "b)", holds the part (b) answer). Their labels have no digit, so the main
            # loop skips them. The scan takes consecutive rows with exactly two non-blank real
            # cells whose label matches _BARE_SUBPART_MARKER_PATTERN. Blank cells are not counted,
            # because such rows can end in an empty colspan cell. A numbered label never matches
            # that pattern, so the scan stops at the next question.
            trailing_row_index = origin_row_index + rowspan
            while trailing_row_index < len(fixed_rows):
                if trailing_row_index in rowspan_origin_row_indices:
                    # A different rowspan origin: never consumed here (Issue 122). This loop
                    # handles it separately.
                    break
                trailing_cells = fixed_rows[trailing_row_index]
                trailing_tail = fixed_rows_tail[trailing_row_index]
                trailing_non_blank = [
                    i for i, is_tail in enumerate(trailing_tail)
                    if not is_tail and _clean_answer_cell_text(trailing_cells[i])
                ]
                if len(trailing_non_blank) != 2:
                    break
                trailing_label = trailing_cells[trailing_non_blank[0]].strip()
                if not _BARE_SUBPART_MARKER_PATTERN.match(trailing_label):
                    break
                recovered_parts.append(_clean_answer_cell_text(trailing_cells[trailing_non_blank[1]]))
                trailing_row_index += 1

            if not recovered_parts:
                continue
            recovered_text = "\n".join(recovered_parts)

            if extracted_via_prefix:
                # No existing GridRow can represent this question (Issue 118). The origin label
                # never fully matched, so the main loop paired it with neighbouring sub-table
                # header noise and skipped it. Recognising this label shape in the main loop was
                # rejected for the same reason: position points at header noise, not content. A new
                # GridRow is created from the recovered content instead, unless another row on this
                # table already has the same number, so no existing answer is overwritten or
                # duplicated.
                existing = next(
                    (r for r in rows[rows_before_this_table:] if r.question_number == origin_number),
                    None,
                )
                if existing is not None:
                    logger.debug(
                        "parse_answer_key_grid: page %d — Issue 118 prefix-recovered Q%s "
                        "already has a real GridRow from elsewhere on this table — left alone, "
                        "not guessed onto an unrelated existing row.", page_index, origin_number,
                    )
                    continue
                # Use a column_index outside every column this table can produce. An in-range
                # column (`max_pairs - pair_count`) can share a (section, column) key with an open
                # cross-page chain, which would close it early and take its continuation (on
                # Nan-Hua p35, a new Q10 row took Q9's continuation this way). `this_column` in the
                # main loop is always below max_pairs, so max_pairs is out of range, and adding
                # origin_row_index keeps several recovered rows on one table apart. The row is
                # already a complete answer and never needs to join a chain.
                origin_column_offset = max_pairs + origin_row_index
                rows.append(
                    GridRow(
                        question_number=origin_number,
                        answer_text=recovered_text,
                        raw_cell_content=recovered_text,
                        page_index=page_index,
                        column_index=origin_column_offset,
                        section_label=current_section_label,
                    )
                )
                logger.debug(
                    "parse_answer_key_grid: page %d — Issue 118: recovered a new GridRow for "
                    "Q%s from a rowspan origin whose own label only matched the prefix pattern "
                    "(a real number with an embedded sub-part marker glued on).",
                    page_index, origin_number,
                )
                continue

            target = next(
                (r for r in rows[rows_before_this_table:] if r.question_number == origin_number),
                None,
            )
            if target is None:
                logger.debug(
                    "parse_answer_key_grid: page %d — Shape 3 recovered content found for Q%s "
                    "but no existing GridRow to attach it to at all — left unattached, not "
                    "guessed onto the wrong row.", page_index, origin_number,
                )
                continue
            if not target.answer_text.strip():
                target.answer_text = recovered_text
                target.raw_cell_content = recovered_text
                logger.debug(
                    "parse_answer_key_grid: page %d — recovered Shape 3 shadow content for Q%s "
                    "(Rule A, blank target) from a whole-row-spanning row inside its own rowspan "
                    "span.", page_index, origin_number,
                )
            elif candidates_per_row.get(origin_row_index, 0) == 1:
                target.answer_text = f"{target.answer_text}\n{recovered_text}"
                target.raw_cell_content = f"{target.raw_cell_content}\n{recovered_text}"
                logger.debug(
                    "parse_answer_key_grid: page %d — recovered Shape 3 shadow content for Q%s "
                    "(Rule B, Issue 117: appended, no sibling rowspan number on this row to "
                    "misattribute it to) from a whole-row-spanning row inside its own rowspan "
                    "span.", page_index, origin_number,
                )
            else:
                logger.debug(
                    "parse_answer_key_grid: page %d — Shape 3 recovered content found for Q%s "
                    "but its existing content is non-blank AND a sibling rowspan number shares "
                    "this row (candidates_per_row=%d) — left unattached, not guessed onto the "
                    "wrong number.", page_index, origin_number, candidates_per_row.get(origin_row_index, 0),
                )

    return rows


# =================================================================================================
# parse_narrative_mcq_answers / _has_narrative_mcq_shape, Issue 126 (development log, MCQ half of
# Issue 110's own narrative-answer-booklet gap, PSLE Math 2014-2018 Answer+Solution.pdf). These
# 5 real files' own Paper 1 Booklet A/B pages carry no table/grid at all, prose only, with a real,
# consistently-printed "N. (M)" marker per MCQ question (N = question number, M = the correct
# option's own index, e.g. "1. (4)"), confirmed identical to the bare option-index value 2019/2020's
# own grid-based answer key already stores for the same real Booklet A/B questions (e.g. 2019's
# grid: Q1 -> "4"), matching an already-trusted real representation, not inventing a new one.
#
# Deliberately scoped to MCQ content only (development log, Issue 110 investigation + explicit
# instruction): Paper 2's own open-ended narrative content (bare "N." marker, full-prose worked
# solution, no short answer-key value printed at all) is NOT recognized here and stays exactly as
# unmatched as it is today, a separate, deferred design decision (distilling a short answer_value
# from free prose is real, lower-confidence work, not attempted alongside this scoped MCQ fix).
# =================================================================================================

# Matches at the START of a real line only (re.MULTILINE), "N. (M)", e.g. "1. (4)", "12. (1)".
# Confirmed real, zero false positives (development log, Issue 126): re-scanned all 24 real,
# already-extracted non-narrative files' own raw MinerU2.5 output for this exact pattern, 0 pages
# anywhere produced even a single coincidental match, let alone the >=2 threshold
# _has_narrative_mcq_shape() below actually requires.
_MCQ_MARKER_LABEL_PATTERN = re.compile(r"(?m)^(\d{1,3})\.\s*\((\d{1,2})\)")


def _has_narrative_mcq_shape(extracted_text: str) -> bool:
    """
    Real, dedicated positive signal for a narrative Booklet A/B page, deliberately NOT folded into
    classify_page()'s own general-purpose question_page/other decision (development log, Issue 110
    investigation: that existing signal was confirmed unreliable for this exact shape, several real
    marker-rich pages still classified 'other'). Requires >=2 real matches, not >=1, the same
    real-evidence-backed safety margin this file's other STRICT shape checks use elsewhere
    (_has_answer_grid_table_shape() etc.), real narrative pages carry 3-11 markers each; nothing in
    the real batch has ever produced even one coincidental match, so this margin is conservative,
    not load-bearing.
    """
    return len(_MCQ_MARKER_LABEL_PATTERN.findall(extracted_text)) >= 2


def parse_narrative_mcq_answers(page_index: int, extracted_text: str) -> list[GridRow]:
    """
    Real, table-free counterpart to parse_answer_key_grid() for a narrative Booklet A/B page. Walks
    extracted_text line by line (real, confirmed reading-order-faithful for MinerU2.5's own flat
    text on these pages, Issue 126 investigation) rather than bounding_boxes, since
    DeepSeek-OCR's own real output for these pages carries no structured regions at all (Issue
    100's own real finding), but in practice the representative output reaching this function is
    always MinerU2.5's (page-level compare_readers() agreement_status is 'disagree' on 100% of
    these real pages, confirmed directly against all 45 real pages across all 5 files, never
    single_reader_only), so this is not a theoretical hedge only.

    Emits ONE GridRow per real MCQ marker: question_number=N (bare, matching parse_answer_key_grid's
    own normalized shape), answer_text=M (the bare option index, e.g. "4", matching the real,
    already-trusted representation 2019/2020's own grid answer key already uses for the same real
    questions, confirmed directly, Issue 126). column_index is always 0, narrative
    MCQ content is a real, single, self-contained stream (never a multi-column grid, never a
    cross-page continuation fragment: every real row here always carries its own question_number),
    so no other column value is ever needed to keep locate_embedded_answer_section()'s own per-key
    chain-commit logic correct.

    section_label is only ever set on a row when a real "Paper 1 Booklet A"/"Booklet B" heading line
    is seen ON THIS SAME PAGE (via the shared _normalize_section_label() vocabulary, the same one
    the question side's own _extract_question_page_section() resolves to via these files' real
    footer paper codes, e.g. "0008/1(A)/2015" -> "BOOKLET A"), left None otherwise, exactly like
    parse_answer_key_grid()'s own table_caption-derived section_label, so
    locate_embedded_answer_section()'s EXISTING cross-page last_real_section_label carry-forward
    (unchanged, already handles this for grid rows) applies here too, unmodified, confirmed real
    and necessary: these files' own page 1 continues Q12-15 with no section heading printed again at
    all (Issue 126 investigation).
    """
    rows: list[GridRow] = []
    current_section: Optional[str] = None
    for line in extracted_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        section = _normalize_section_label(stripped)
        if section is not None:
            current_section = section
            continue
        match = _MCQ_MARKER_LABEL_PATTERN.match(stripped)
        if not match:
            continue
        number, option = match.group(1), match.group(2)
        rows.append(GridRow(
            question_number=number,
            answer_text=option,
            raw_cell_content=stripped,
            page_index=page_index,
            column_index=0,
            section_label=current_section,
        ))
    return rows


# Real sub-part marker as printed in a QUESTION's own text, confirmed real shape across this
# whole subset, e.g. chij's Q26 ("(a) Which 2 views... (b) The four shaded squares..."). Single
# letter only (never digits, so a real MCQ option "(1)" can't match), the one known residual risk
# is a real question using a bare single-letter VARIABLE in parens (e.g. "(x)"), not confirmed
# present anywhere in the real subset examined, not chased further per this project's "don't guess
# past real evidence" discipline.
_SUBPART_MARKER_IN_QUESTION_PATTERN = re.compile(r"\(([a-zA-Z])\)")


def _combine_lettered_siblings(
    sibling_source: dict, sibling_keys_to_letters: list[tuple[object, str]], question_raw_text: str,
) -> tuple[tuple[int, str], Optional[str]]:
    """
    Shared combination step (Issue 3 real fix), used by BOTH the
    section-scoped and flat lookup paths in _lookup_answer_for_span() below, so the actual
    combination/completeness logic exists in exactly one place. sibling_keys_to_letters is a
    sorted list of (map_key, letter) pairs; sibling_source is whichever map those keys came from.
    """
    combined_page = sibling_source[sibling_keys_to_letters[0][0]][0]
    combined_parts = [f"({letter}) {sibling_source[key][1]}" for key, letter in sibling_keys_to_letters]
    combined_text = "\n".join(combined_parts)

    # Distinct letters, not raw occurrence count, real, confirmed reason found testing this
    # against real PSLE Math 2019.pdf data: a real question's own text legitimately repeats each
    # sub-part letter twice, once in the question stem ("(a) 15 + 2y") and again on its own answer
    # line ("Ans:(a)\n(b)"), a real, common PSLE convention, confirmed on both Q22 and Q28. Raw
    # occurrence counting doubled both counts (4 instead of 2) and wrongly flagged two genuinely
    # complete matches as partial.
    real_marker_count = len(set(_SUBPART_MARKER_IN_QUESTION_PATTERN.findall(question_raw_text)))
    if real_marker_count != len(sibling_keys_to_letters):
        return (combined_page, combined_text), "partial_subpart_answer_match"
    return (combined_page, combined_text), None


def _lookup_answer_for_span(
    answer_map: dict[str, tuple[int, str]],
    section_answer_map: dict[tuple[Optional[str], str], tuple[int, str]],
    question_number: str,
    question_section: Optional[str] = None,
    question_raw_text: str = "",
) -> tuple[Optional[tuple[int, str]], Optional[str]]:
    """
    Looks up a segmented question's answer, section-scoped first when the question's own section
    is known (Issue 3's real fix), the flat cross-section map as a
    fallback otherwise. Real, confirmed real grid shape: segment_questions() only ever produces
    bare top-level question_numbers (a multi-part question correctly stays ONE span, Piece 2 step
    2's confirmed gap, not touched here), but a real grid's own sub-part rows come in TWO real,
    confirmed shapes depending on the paper: chij folds sub-parts into ONE combined answer cell
    against a single bare label ("Q26" -> "a) ...\nb) ..."); GS005/PSLE Math 2019 lists them as
    SEPARATE lettered rows with no bare row at all ("5a." -> "-", "5b." -> "10").

    Real, confirmed finding this round (checked directly against BOTH real grid shapes' own
    (section, column, base_number) groupings, not assumed): within the CORRECT section, a bare row
    and lettered sibling rows for the same base number never coexist, every base number is either
    bare-only or lettered-only, no exceptions found anywhere. This is what makes section-scoped
    resolution safe: once the question's own real section is known, section_answer_map can only
    ever hold ONE real answer shape for that base number in that section (a bare row, or a
    lettered family, never both, and if a real duplicate somehow occurred,
    locate_embedded_answer_section()'s own commit-time collision check already caught and logged
    it before this function ever sees the map).

    Three real paths:
      - question_section is known: looked up directly in section_answer_map. A bare row there is
        returned as a real, resolved match, no flag, nothing left to be ambiguous against within
        a section that's known. No bare row but lettered siblings IN THAT SAME SECTION , 
        combined (via _combine_lettered_siblings), same completeness check as before. Neither
        found, check the section=None bucket next (Issue 146, the development log) before
        giving up: some real answer-key pages carry NO section-caption signal at all anywhere on or
        near them (not even Issue 137's short-text-region shape, a real, confirmed case,
        Rosyth's own WA/CA family, where every grid row on the page files under section=None
        because there was nothing to detect, not a parsing miss). Using that bucket here
        does NOT reopen the Issue 3 collision this section-scoped design exists to prevent , 
        it's only used when this exact question_number is NOT ALSO claimed by a DIFFERENT, real
        (non-None) section anywhere in the file; a genuine cross-section collision candidate is
        still left unresolved, not guessed at, same as the section-known case above. The None
        bucket is already collision-protected the same way every real section is
        (_commit_grid_chain's own same-section duplicate check runs on it identically), so this
        doesn't weaken that protection, only extends where a resolved value can come from. If
        question_section is literally "PAPER 1" (this module's own documented least-specific,
        last-resort signal) and still nothing found, one more check (Issue 147, development log
): a real paper sometimes prints "Paper 1" as the umbrella covering Booklet A +
        Booklet B with no per-page sub-booklet label on the question side at all, while the grid
        side still separates by "BOOKLET A"/"BOOKLET B", same real paper, different granularity,
        not a cross-paper collision. Used only when exactly ONE other real, named, non-"PAPER 1"
        section claims this exact number (two or more, or a genuine independent "PAPER 1" grid
        entry elsewhere in the file, leaves it unresolved, not guessed at). Neither the known
        section NOR (safely) the None bucket NOR (safely, "PAPER 1" only) another section has
        anything, no answer-key row for this question anywhere it's safe to look,
        returns no match at all; the caller's existing "no_matching_answer_key_row" flag applies.
      - question_section is None (the question's own page carried no detectable section signal , 
        a real, expected case, explicit instruction not to guess past it): falls back to the
        ORIGINAL flat-map behavior, UNCHANGED, a bare match found alongside cross-section lettered
        siblings stays flagged `ambiguous_bare_vs_subpart_answer_key_match` rather than guessed at,
        exactly as before this round. Not a regression of a known-working flagged case into a wrong
        silent match.

    Returns (match_or_None, extra_flag_or_None). The value returned is still real audit
    information (never dropped) even when flagged.
    """
    if question_section is not None:
        section_bare_match = section_answer_map.get((question_section, question_number))
        if section_bare_match is not None:
            return section_bare_match, None

        section_sibling_pairs = sorted(
            (key, key[1][len(question_number):])
            for key in section_answer_map
            if key[0] == question_section
            and re.fullmatch(re.escape(question_number) + r"[a-zA-Z]", key[1])
        )
        if section_sibling_pairs:
            return _combine_lettered_siblings(section_answer_map, section_sibling_pairs, question_raw_text)

        # Known section, nothing there, before giving up, check the section=None bucket
        # (Issue 146, see this function's own docstring for the real case and why this is safe):
        # only used when no DIFFERENT real, named section also claims this exact number, a genuine
        # cross-section collision candidate is still left unresolved, not guessed at.
        none_bare_match = section_answer_map.get((None, question_number))
        if none_bare_match is not None and not any(
            key[0] is not None and key[1] == question_number for key in section_answer_map
        ):
            return none_bare_match, None

        none_sibling_pairs = sorted(
            (key, key[1][len(question_number):])
            for key in section_answer_map
            if key[0] is None
            and re.fullmatch(re.escape(question_number) + r"[a-zA-Z]", key[1])
        )
        if none_sibling_pairs and not any(
            key[0] is not None and re.fullmatch(re.escape(question_number) + r"[a-zA-Z]", key[1])
            for key in section_answer_map
        ):
            return _combine_lettered_siblings(section_answer_map, none_sibling_pairs, question_raw_text)

        # question_section == "PAPER 1" specifically, still nothing (Issue 147, development log
        #), "PAPER 1" is this function's OWN documented least-specific signal
        # (_extract_question_page_section()'s own docstring: checked LAST, only after Booklet A/B
        # and Paper 2 have already been ruled out on that same page) for a real, confirmed reason:
        # some real papers print "Paper 1" as the umbrella name covering Booklet A + Booklet B with
        # no per-page sub-booklet label on the QUESTION side at all, while the GRID side's own
        # answer key still separates its rows into "BOOKLET A"/"BOOKLET B", real, different
        # granularity for the SAME real paper, not a genuine cross-paper collision. Safe the same
        # way the None-bucket check above is: only used when exactly ONE other real, named,
        # non-"PAPER 1" section claims this exact number, two or more (or zero) leaves it
        # unresolved rather than guessed at. Does NOT trigger for any other section pairing, a
        # real, confirmed counter-example exists (`P6-Maths-2021-SA1-Nan-Hua`,
        # `P6-Maths-2021-SA1-Raffles-Girls` both have their own independent "PAPER 1"
        # grid entries elsewhere in the same file), so this is scoped to "PAPER 1" only, not
        # generalized to "try any other section when known section fails."
        if question_section == "PAPER 1":
            other_named_sections = {
                key[0] for key in section_answer_map
                if key[0] is not None and key[0] != "PAPER 1" and key[1] == question_number
            }
            if len(other_named_sections) == 1:
                return section_answer_map[(other_named_sections.pop(), question_number)], None

            other_named_sibling_sections = {
                key[0] for key in section_answer_map
                if key[0] is not None and key[0] != "PAPER 1"
                and re.fullmatch(re.escape(question_number) + r"[a-zA-Z]", key[1])
            }
            if len(other_named_sibling_sections) == 1:
                only_sec = other_named_sibling_sections.pop()
                sibling_pairs = sorted(
                    (key, key[1][len(question_number):])
                    for key in section_answer_map
                    if key[0] == only_sec
                    and re.fullmatch(re.escape(question_number) + r"[a-zA-Z]", key[1])
                )
                return _combine_lettered_siblings(section_answer_map, sibling_pairs, question_raw_text)

        # Not safely resolvable anywhere, not a fallback-to-flat-map case: we KNOW the right
        # section (and checked the unscoped None bucket too), so falling back to the flat map here
        # could silently succeed via an UNRELATED section's bare match, which would be wrong given
        # we have positive knowledge of the correct section already.
        return None, None

    # question_section unknown, original flat-map behavior, unchanged.
    bare_match = answer_map.get(question_number)
    sibling_keys = sorted(
        k for k in answer_map
        if re.fullmatch(re.escape(question_number) + r"[a-zA-Z]", k)
    )
    if not sibling_keys:
        return bare_match, None
    if bare_match is not None:
        return bare_match, "ambiguous_bare_vs_subpart_answer_key_match"
    sibling_pairs = [(k, k[len(question_number):]) for k in sibling_keys]
    return _combine_lettered_siblings(answer_map, sibling_pairs, question_raw_text)


def _commit_grid_chain(
    answer_map: dict[str, tuple[int, str]],
    section_answer_map: dict[tuple[Optional[str], str], tuple[int, str]],
    section: Optional[str],
    chain: tuple[str, int, str],
    pdf_path: Path,
    discrepancy_log: Optional[Path],
) -> None:
    """
    Commits one completed (question_number, page_index, accumulated_text) chain from
    locate_embedded_answer_section()'s per-key chain tracking into BOTH real answer maps, the
    one place collision detection actually happens, run once per chain, not per row (see that
    function's own docstring for the real bug this separation fixes: a chain's continuation
    fragments must accumulate against ITS OWN open chain, never against whatever answer_map
    already holds for that bare number from a different, unrelated chain).

    Same collision discipline as everywhere else in this file (Bug 3's leave-unresolved
    precedent): first real value for a given key wins, a later different one is logged via
    log_ocr_discrepancy and left unused, never silently overwritten either way, applied
    independently to BOTH maps, since a real (section, number) collision (a genuine duplicate row
    within the same section, not observed in real data, but not assumed impossible either) is a
    different, narrower event than a flat, cross-section collision (Issue 3, confirmed common).
    """
    number, page_index, text = chain

    prior = answer_map.get(number)
    if prior is not None and prior[1] != text and not values_equivalent(prior[1], text):
        message = (
            f"page {page_index}: answer-key grid question_number {number!r} already has a "
            f"different real answer from page {prior[0]} ({prior[1]!r}) — this page's value "
            f"({text!r}) NOT used, first one kept. Likely cause: Issue 3 (paper/booklet "
            f"question-number reuse — segment_questions() has the identical limitation, confirmed "
            f"real on GS005: Booklet A's Q2 vs Paper 2's Q2 share a bare number for two different "
            f"real questions). Needs human review, not silently resolved."
        )
        logger.warning("locate_embedded_answer_section: %s", message)
        log_ocr_discrepancy(pdf_path.stem, page_index, message, discrepancy_log)
    else:
        answer_map[number] = (page_index, text)

    section_key = (section, number)
    section_prior = section_answer_map.get(section_key)
    if section_prior is not None and section_prior[1] != text and not values_equivalent(section_prior[1], text):
        message = (
            f"page {page_index}: answer-key grid question_number {number!r} in section "
            f"{section!r} already has a different real answer from page {section_prior[0]} "
            f"({section_prior[1]!r}) — this page's value ({text!r}) NOT used, first one kept. "
            f"A real, same-section duplicate — not the expected Issue 3 cross-section case — "
            f"needs human review."
        )
        logger.warning("locate_embedded_answer_section: %s", message)
        log_ocr_discrepancy(pdf_path.stem, page_index, message, discrepancy_log)
        return
    section_answer_map[section_key] = (page_index, text)


def locate_embedded_answer_section(
    pdf_path: Path,
    reader_output_by_page: dict[int, RawReaderOutput],
    discrepancy_log: Optional[Path] = None,
) -> tuple[dict[str, tuple[int, str]], dict[tuple[Optional[str], str], tuple[int, str]]]:
    """
    Topology B's core requirement (Section 3.4): "associate each answer to its question by
    content, not page-position." Returns TWO maps (the development log follow-up, Issue 3's
    real fix, return type changed from a single flat map): `answer_map`, question_number ->
    (page_index, answer_text), unchanged in shape from before; and NEW, `section_answer_map`,
    (section, question_number) -> (page_index, answer_text), the section-scoped view that makes
    Issue 3's real cross-section collisions resolvable by _lookup_answer_for_span() instead of
    permanently flagged.

    Real implementation (the development log follow-up), built and validated against actual
    School papers/ pages (chij's four real grid pages, p44-47) before trusting it, not a guessed
    signature. Finds every page classify_page() calls `answer_key_grid`, parses each with
    parse_answer_key_grid(), and merges all of them into both maps across the whole file (chij
    has four grid pages spanning one continuous multi-table sequence).

    Three real, confirmed things this merge must get right, all found designing this against real
    data, none hypothetical:

    1. Cross-page continuation (GridRow.question_number == ""): resolved here using
       (section_label, column_index) as the key, the last REAL question_number seen at that same
       key. section_label carries forward across a page with no table_caption of its own (chij
       p46/p47 have none, Q30's answer at p45's Q30-only table, PAPER 2's continuation cells at
       p46/p47, inheriting whatever real caption was last seen, "PAPER 2"), so continuation
       resolution stays correctly scoped even when captions aren't repeated on every page. A
       continuation fragment with NO prior real question at its (section, column) key is a genuine
       anomaly (no real page in this subset produced one), logged via log_ocr_discrepancy and its
       content left unattached, not guessed onto the wrong question.

    2. Same bare question_number, two different real answers (confirmed real: GS005 , 
       Booklet A's Q2 = "3", Paper 2's Q2 = "-", same grid, same file, both real questions that
       merely share a bare number). This is Issue 3 (paper/booklet question-number reuse)
       surfacing here concretely, real, but no longer a dead end: `answer_map` keeps its existing
       flat, first-wins-and-log behavior (still needed as a fallback for pages whose own section is
       undetectable, see _lookup_answer_for_span()), but `section_answer_map` keeps BOTH values,
       correctly scoped, so a question whose own section IS known can be matched to the right one
       instead of whichever happened to win the flat collision.

    3. GS005's own grid has no `table_caption` regions at all, its three sections' names are
       printed as ordinary cells in the table's own header row instead, one per column
       (`parse_answer_key_grid()`'s `section_by_column`, not this function's own concern), but
       CONTINUATION rows on caption-less pages still need this function's own section
       carry-forward (`last_real_section_label`) exactly as before, since `GridRow.section_label`
       is still None for those rows regardless of which real mechanism populated the ORIGINAL
       row's section.
    """
    answer_map: dict[str, tuple[int, str]] = {}
    section_answer_map: dict[tuple[Optional[str], str], tuple[int, str]] = {}
    open_chain_by_key: dict[tuple[Optional[str], int], tuple[str, int, str]] = {}
    last_real_section_label: Optional[str] = None
    grid_pages_found = 0
    narrative_mcq_pages_found = 0  # Issue 126, real narrative Booklet A/B pages, tracked
                                     # separately from grid_pages_found so the "no answer source at
                                     # all" warning below stays honest when a file has one shape but
                                     # not the other.
    previous_page_was_grid = False

    for page_index in sorted(reader_output_by_page.keys()):
        reader_output = reader_output_by_page[page_index]
        bounding_boxes = reader_output.bounding_boxes or []
        classification = classify_page(bounding_boxes, reader_output.extracted_text)

        # Issue 100's grid-side follow-up : also recognizes a
        # flat-text-embedded <table> (DeepSeek-OCR's real shape, no structured bounding_boxes at
        # all), same real reason as classify_page()'s own _has_answer_grid_table_shape() call.
        page_has_table = (
            any(bb.get("type") == "table" for bb in bounding_boxes)
            or bool(_EMBEDDED_HTML_TABLE_PATTERN.search(reader_output.extracted_text))
        )
        # Real, confirmed narrow case classify_page() itself cannot resolve alone (chij p47, see
        # that function's own docstring note): a continuation page whose ONLY content is an orphan
        # fragment with no grid-label cell at all, nothing about the page in isolation
        # distinguishes it from any other page with a table region. Carried forward here instead:
        # if the immediately preceding page (by page_index, this loop's own order) was treated as
        # part of the grid, AND this page classifies as 'other' but still has a table region, treat
        # it as still part of the same grid too. Stops the instant a page is anything OTHER than
        # 'other'-with-a-table (a real question_page/cover_page ends the carry-forward immediately)
        #, deliberately narrow, grounded in the real adjacency observed across all four of chij's
        # real grid pages, not a blanket "everything after a grid page is also a grid page" rule.
        is_narrative_mcq_page = False
        if classification != "answer_key_grid":
            # Issue 145 fix (development log, rowspan/colspan position-signal investigation brief,
            #): this carry-forward is the REAL mechanism that swept Nan-Hua p38's own
            # bar-model diagram table into grid parsing in the first place, it classifies 'other'
            # (correctly) but has a table and immediately follows a real grid page (p37), so without
            # this exclusion it would still be treated as a continuation. See
            # `_is_solo_diagram_noise_table_region()`'s own docstring for the corpus-validated
            # position signal (zero false positives across the full 4,423-page corpus) that now
            # excludes exactly this one real, confirmed case from the carry-forward.
            if (
                previous_page_was_grid
                and classification == "other"
                and page_has_table
                and not _is_solo_diagram_noise_table_region(bounding_boxes)
            ):
                logger.debug(
                    "locate_embedded_answer_section: page %d classified 'other' by classify_page() "
                    "but immediately follows a grid page and still has a table region — treating as "
                    "a continuation of the same grid (chij p47's real confirmed shape).",
                    page_index,
                )
            elif _has_narrative_mcq_shape(reader_output.extracted_text):
                # Issue 126 (MCQ half of Issue 110): a real narrative Booklet A/B page, no
                # table/grid at all, just a real "N. (M)" marker per MCQ question. Deliberately
                # checked here, not folded into classify_page() itself (see
                # _has_narrative_mcq_shape()'s own docstring), never reached for a real
                # answer_key_grid page (classification != "answer_key_grid" already excludes those,
                # zero regression risk by construction).
                is_narrative_mcq_page = True
                logger.debug(
                    "locate_embedded_answer_section: page %d classified %r by classify_page() but "
                    "matches the real narrative Booklet A/B MCQ marker shape ('N. (M)') — parsing "
                    "via parse_narrative_mcq_answers() instead (Issue 126).",
                    page_index, classification,
                )
            else:
                previous_page_was_grid = False
                continue
        if is_narrative_mcq_page:
            previous_page_was_grid = False  # narrative pages don't participate in the grid-only
                                              # chij-style carry-forward heuristic above
            narrative_mcq_pages_found += 1
        else:
            previous_page_was_grid = True
            grid_pages_found += 1

        rows = (
            parse_narrative_mcq_answers(page_index, reader_output.extracted_text)
            if is_narrative_mcq_page
            else parse_answer_key_grid(bounding_boxes, page_index, reader_output.extracted_text)
        )
        for row in rows:
            effective_section = row.section_label if row.section_label is not None else last_real_section_label
            if row.section_label is not None:
                last_real_section_label = row.section_label
            key = (effective_section, row.column_index)

            if row.question_number:
                # A new real row at this key CLOSES whatever chain was open there (if any), must
                # commit it into answer_map BEFORE overwriting open_chain_by_key with the new one,
                # or its accumulated continuation text is lost. This is the real fix for a bug
                # found testing this against chij's actual data: without committing the OLD chain
                # first, a later continuation fragment that belongs to THIS row's own chain could
                # instead be looked up against whatever answer_map already held for this bare
                # number (a DIFFERENT section's row that happened to collide and win), confirmed
                # real: Paper 2's Q11 continuation (p46) was landing on Booklet A's kept Q11 answer
                # instead of Paper 2's own (collision-rejected) Q11, silently corrupting a real,
                # correct answer with unrelated content.
                if key in open_chain_by_key:
                    _commit_grid_chain(
                        answer_map, section_answer_map, key[0], open_chain_by_key.pop(key),
                        pdf_path, discrepancy_log,
                    )
                open_chain_by_key[key] = (row.question_number, row.page_index, row.answer_text)
            else:
                if key not in open_chain_by_key:
                    message = (
                        f"page {page_index}: answer-key grid continuation fragment at column "
                        f"{row.column_index} (section {effective_section!r}) has no open chain at "
                        f"that position on this page or an earlier one — content NOT attached "
                        f"anywhere, needs human review: {row.answer_text[:100]!r}"
                    )
                    logger.warning("locate_embedded_answer_section: %s", message)
                    log_ocr_discrepancy(pdf_path.stem, page_index, message, discrepancy_log)
                    continue
                number, page, text = open_chain_by_key[key]
                open_chain_by_key[key] = (number, page, f"{text}\n{row.answer_text}")

    # Commit every chain still open at the end of the file, a chain with no later row at its
    # (section, column) key to close it is still a real, complete answer (e.g. the very last
    # question in the grid), not an error.
    for key, chain in open_chain_by_key.items():
        _commit_grid_chain(answer_map, section_answer_map, key[0], chain, pdf_path, discrepancy_log)

    if grid_pages_found == 0 and narrative_mcq_pages_found == 0:
        logger.warning(
            "locate_embedded_answer_section: no answer_key_grid page and no narrative Booklet A/B "
            "MCQ page found in %s — every question in this file will have no matched answer "
            "(flagged individually downstream, not a crash here). Confirm this is real (e.g. a "
            "format without a grid or a recognized narrative shape at all) rather than a "
            "classify_page()/Issue 126 miss before trusting an empty result.",
            pdf_path,
        )

    return answer_map, section_answer_map


# =================================================================================================
# build_per_reader_grid_maps / _apply_grid_cross_check, Issue 101's real fix (development log
#): same architectural move Issue 99 already made for question segmentation (segment
# EACH reader's own output independently, don't merge first), now applied to grid parsing. The
# EXISTING single/representative answer_map (built once per file, above) is UNCHANGED and stays
# the authoritative source for the real, stored answer_value on every ExtractedQuestion, these two
# functions add an ADDITIONAL cross-check signal for agreement_status/ocr_confidence only.
# =================================================================================================

def build_per_reader_grid_maps(
    file_path: Path, page_count: int, raw_dir: Path,
) -> tuple[
    dict[str, tuple[dict[str, tuple[int, str]], dict[tuple[Optional[str], str], tuple[int, str]]]],
    dict[str, set[int]],
]:
    """
    Builds TWO independent (answer_map, section_answer_map) pairs, one per real reader, by running
    locate_embedded_answer_section() against EACH reader's own normalized page output separately , 
    not the single "representative" output the existing (unchanged) call above uses. Keyed by
    reader name ("mineru25"/"deepseek_ocr"), used only by _apply_grid_cross_check() below.

    Since Issue 126, this transparently ALSO covers real narrative Booklet A/B MCQ answers, not
    just grid answers, locate_embedded_answer_section() itself now recognizes both real shapes, so
    no separate per-reader builder was needed for the narrative case. Confirmed real and necessary,
    not just theoretically consistent (development log, Issue 126 investigation): page-level
    compare_readers() reports 'disagree' on 100% of these real narrative pages (whole-page text
    similarity is a known-noisy signal here, same reason Issue 101 needed this same treatment for
    grid answers) even though the two readers' own independently-extracted MCQ marker VALUES agree
    100% of the time (0 mismatches across all 75 real MCQ questions checked), without this reuse,
    every recovered narrative MCQ answer would incorrectly show ocr_confidence='low'.

    Issue 251's real fix : ALSO returns, per reader, the real set of
    page indices where that reader produced literally NO usable output at all (no Stage B file, or
    an empty/whitespace-only extracted_text), distinct from "this reader had output on the page
    but its own grid parsing didn't find THIS question's row on it". `_apply_grid_cross_check()`
    needs this to tell those two real, different cases apart: a whole-page reader failure on the
    ANSWER page is honest single_reader_only evidence for that answer's own provenance, regardless
    of what the (different) QUESTION-text page's own agreement_status found; a normal per-question
    grid-lookup miss on a page BOTH readers actually produced output for is not the same claim at
    all, and must not be conflated with it.
    """
    maps: dict[str, tuple[dict, dict]] = {}
    missing_pages: dict[str, set[int]] = {}
    for reader in ("mineru25", "deepseek_ocr"):
        reader_output_by_page: dict[int, RawReaderOutput] = {}
        reader_missing_pages: set[int] = set()
        for page_index in range(page_count):
            raw = load_raw_output(reader, file_path.stem, page_index, raw_dir)
            if raw is None:
                reader_missing_pages.add(page_index)
                continue
            norm = RawReaderOutput(
                extracted_text=normalize_glyphs(raw.extracted_text),
                structure=raw.structure, bounding_boxes=raw.bounding_boxes,
                field_confidence=raw.field_confidence,
            )
            if norm.extracted_text.strip():
                reader_output_by_page[page_index] = norm
            else:
                reader_missing_pages.add(page_index)
        maps[reader] = locate_embedded_answer_section(file_path, reader_output_by_page, None)
        missing_pages[reader] = reader_missing_pages
    return maps, missing_pages


def _apply_grid_cross_check(
    agreement_status: str,
    disagreement_detail: Optional[str],
    disagreement_type: Optional[str],
    per_reader_grid_maps: dict,
    question_section: Optional[str],
    question_number: str,
    question_raw_text: str,
    missing_pages: "dict[str, set[int]]",
    answer_page_index: int,
) -> tuple[str, Optional[str], Optional[str]]:
    """
    Real, highest-priority override (explicit instruction, the development log): if BOTH
    readers have a confident, independently-parsed grid answer for this real question AND those
    answers conflict (_grid_answers_agree() returns False), that overrides everything
    else the question-text comparison already decided, the single most safety-relevant signal
    this whole two-reader design exists to produce, not something a high text-similarity score
    should be allowed to mask. Reuses _lookup_answer_for_span() (not a raw dict lookup) against
    EACH reader's own maps, so real sub-part combination ("5a."/"5b." -> "(a) ...\n(b) ...") works
    identically to how the actual stored answer_value is already resolved, not a simplified,
    less-capable re-derivation.

    If the grid answers instead AGREE but the existing agreement_status is still
    "disagree" (the question-text comparison found a real difference), that is its own real,
    distinct case (explicit instruction), NOT silently upgraded to "agree": the question STEM
    itself may be corrupted even though the final answer happens to line up, so it stays flagged
    for review, just with an enriched detail message distinguishing it from a genuine content
    mismatch (mirroring the same "already correctly disagree" reasoning as the existing structural-
    fabrication case).

    When a confident grid answer isn't available on one or both sides at all (`_grid_answers_agree`
    returns None), returns the existing agreement_status/disagreement_detail completely unchanged , 
    falls back to the question-text-only result, exactly as before Issue 101, not
    silently treated as either agreement or disagreement it can't actually support.

    CHANGED (Issue 212), also threads `disagreement_type` through.
    A real grid-answer conflict is, by its own real nature, a VALUE disagreement (two independently
    parsed real answer VALUES conflict), classified `"value_mismatch"`, confirmed as a real cause
    of 58 of the 130 real, existing `disagree` rows in the live corpus at migration time (the
    other 72 came from the page-text comparison, also a value-level check, see
    `scripts/migrations/migrate_ocr_extraction_records_disagreement_type.py`'s own real backfill logic).
    The "grid agrees, stem still differs" enrichment branch does NOT change what kind of
    disagreement the original question-text comparison found, it only adds a caveat that the
    final value happens to match anyway, so `disagreement_type` passes through UNCHANGED there,
    same as the untouched fallback branch.

    CHANGED (Issue 251), a real, confirmed gap in the "falls back to
    the existing agreement_status unchanged" behavior when `grid_result is None`: that None covers
    TWO real, DIFFERENT situations the old code treated identically. (a) both readers produced
    output for the answer page but neither/one's own grid parsing found THIS question's row on
    it, a genuine, more ambiguous per-question gap, correctly left as a fallback. (b) one reader
    produced literally ZERO usable output for the WHOLE answer page (Stage B never processed it, or
    a Issue 221 known-bad-page exclusion), an unambiguous, whole-page single-reader fact about
    THIS question's own answer content specifically, real and confirmed on a live held-out row
    (`P6_Maths_2025_SA2_rosyth/37`, question ids 273-276: the QUESTION-text page agreed, but the
    ANSWER page was MinerU2.5-only, and the old code silently kept the question-text page's own
    "agree" status, giving no indication the served `worked_solution_text` was ever single-reader).
    Case (b) is now distinguished directly via `missing_pages` (the real, page-level "did this
    reader produce ANY usable output here at all" set `build_per_reader_grid_maps()` now returns)
    and `answer_page_index` (the actual, resolved page the served answer came from), checked BEFORE
    falling back, and only overrides when EXACTLY ONE reader is missing that whole page (both
    missing is a real, different, already-separately-handled "no_matching_answer_key_row" case , 
    see this function's callers).
    """
    mineru_answer_map, mineru_section_map = per_reader_grid_maps["mineru25"]
    deepseek_answer_map, deepseek_section_map = per_reader_grid_maps["deepseek_ocr"]
    mineru_match, _ = _lookup_answer_for_span(
        mineru_answer_map, mineru_section_map, question_number, question_section, question_raw_text,
    )
    deepseek_match, _ = _lookup_answer_for_span(
        deepseek_answer_map, deepseek_section_map, question_number, question_section, question_raw_text,
    )
    a = mineru_match[1] if mineru_match is not None else None
    b = deepseek_match[1] if deepseek_match is not None else None
    grid_result = _grid_answers_agree(a, b)

    if grid_result is False:
        return "disagree", (
            f"real grid-answer conflict for question {question_number!r} (section "
            f"{question_section!r}): mineru25's own grid answer={a!r}, deepseek_ocr's own grid "
            f"answer={b!r} — overrides the question-text comparison (was {agreement_status!r}: "
            f"{disagreement_detail!r})"
        ), "value_mismatch"
    if grid_result is True and agreement_status == "disagree":
        return "disagree", (
            f"both readers' own independently-parsed grid answers agree ({a!r}) for question "
            f"{question_number!r}, but the question-text comparison still found a real "
            f"difference — the question stem itself may be corrupted even though the final "
            f"answer matches; flagged for review, not silently trusted as clean agreement. "
            f"Original detail: {disagreement_detail!r}"
        ), disagreement_type

    # Issue 251's real fix, case (b) above: exactly one reader produced zero usable output for
    # the WHOLE answer page this question's answer actually came from. Honest single-reader
    # evidence about the ANSWER content specifically, regardless of what the (different)
    # question-TEXT page's own agreement_status found. Never fires when both are missing (a real,
    # different case, already handled upstream as "no_matching_answer_key_row", this question
    # would never have reached here with a real `answer_page_index` sourced from a page neither
    # reader produced anything on) or when neither is missing (a real, more ambiguous per-question
    # grid-lookup gap, correctly left as the existing fallback).
    mineru_missing_page = answer_page_index in missing_pages.get("mineru25", set())
    deepseek_missing_page = answer_page_index in missing_pages.get("deepseek_ocr", set())
    if mineru_missing_page != deepseek_missing_page and agreement_status != "single_reader_only":
        surviving_reader = "deepseek_ocr" if mineru_missing_page else "mineru25"
        missing_reader = "mineru25" if mineru_missing_page else "deepseek_ocr"
        return "single_reader_only", (
            f"question {question_number!r}'s own answer content (page {answer_page_index}) came "
            f"from {surviving_reader} alone — {missing_reader} produced no usable output at all "
            f"for this page; overrides the question-TEXT page's own {agreement_status!r} status, "
            f"which reflects a DIFFERENT page (original detail: {disagreement_detail!r})"
        ), None
    return agreement_status, disagreement_detail, disagreement_type


# =================================================================================================
# Topology A / Topology B extraction (Step 3 main branches)
# =================================================================================================

def extract_topology_a(
    pdf_path: Path, answer_file: Path, school_name: Optional[str], raw_dir: Path,
    discrepancy_log: Optional[Path] = None, school_name_verified: bool = False,
) -> ExtractionRunResult:
    """
    Topology A: question and answer live in different files (confirmed for Yearly/ and
    Hold out/small eval/). Populates all four provenance fields:
      source_page_index / source_page_label   <- from pdf_path (the question itself)
      answer_source_file                      <- answer_file, not pdf_path
      answer_source_page_index                <- from answer_file, a separate lookup

    Stage C (post-Issue 81): reads Stage B's already-saved reader output via
    merge_page_with_agreement(pdf_stem, page_index, ..., raw_dir), does not render pages or call
    a reader itself, both already happened in Stage A/Stage B. segment_questions() still takes a
    PageImage per its existing (unchanged) signature; a minimal stand-in with empty image_bytes is
    built here since Stage C has no need for actual pixel data, only page_index, real rendering
    already happened in Stage A and isn't re-done here.

    Returns an ExtractionRunResult (a list of QuestionWithOcrRecord pairs), not a bare list , 
    single-writer database design : each question travels paired with its
    own ocr_extraction_record, persisted together by save() into the one JSON checkpoint for this
    file, rather than being written to a database from inside this (parallel worker) call at all.
    """
    import fitz  # PyMuPDF, for page counts

    pairs: list[QuestionWithOcrRecord] = []

    with fitz.open(pdf_path) as question_doc:
        question_page_count = question_doc.page_count
    with fitz.open(answer_file) as answer_doc:
        answer_page_count = answer_doc.page_count

    # Real answer-key-grid lookup (the development log follow-up), replacing
    # _placeholder_locate_answer_page entirely, not patching around it. Same parse_answer_key_grid
    # / locate_embedded_answer_section machinery Topology B uses, run here against answer_file's
    # own pages instead of pdf_path's, the same task (find the grid page(s) in a file,
    # parse them, look up by question_number), Topology A's only real difference is which file the
    # grid lives in. answer_file's pages must already be rendered and read (Piece 1 fix, Stage A/B
    # now include Topology A's matched answer_file in their own work list, not just pdf_path).
    answer_reader_output_by_page: dict[int, RawReaderOutput] = {}
    for page_index in range(answer_page_count):
        usable_output, _, _ = merge_page_with_agreement(
            answer_file.stem, page_index, f"{answer_file.stem}:p{page_index}", raw_dir
        )
        if usable_output is not None:
            answer_reader_output_by_page[page_index] = usable_output
    answer_map, section_answer_map = locate_embedded_answer_section(
        answer_file, answer_reader_output_by_page, discrepancy_log
    )
    # Issue 101's real fix : TWO independent, per-reader grid maps for
    # the cross-check below, separate from the single/representative answer_map/section_answer_map
    # just built above, which stays the unchanged, authoritative source for the real answer_value.
    per_reader_grid_maps, per_reader_missing_pages = build_per_reader_grid_maps(
        answer_file, answer_page_count, raw_dir,
    )
    used_answer_numbers: set[str] = set()
    last_real_question_section: Optional[str] = None

    for page_index in range(question_page_count):
        # question_id_prefix here is a temporary key for logging/agreement-record association
        # BEFORE the real questions.id exists, see merge_page_with_agreement's docstring on this,
        # same reasoning applies to segment_and_compare_page()'s own per-question question_id.
        provisional_id = f"{pdf_path.stem}:p{page_index}"
        span_results, representative = segment_and_compare_page(
            pdf_path.stem, page_index, provisional_id, raw_dir, discrepancy_log,
        )
        if representative is None:
            continue  # both_failed, or no Stage B output at all, already logged, skip this page

        # Real, secondary has_diagram signal (Issue 233), see
        # _apply_deepseek_ocr_diagram_secondary_signal()'s own docstring and
        # _deepseek_ocr_page_has_diagram_marker()'s own module-level comment for the full real
        # evidence and design.
        _apply_deepseek_ocr_diagram_secondary_signal(span_results, pdf_path.stem, page_index, raw_dir)

        # Real section signal (the development log follow-up, Issue 3's real fix), same
        # carry-forward as extract_topology_b, see its own comment / _extract_question_page_section
        # for both real signal shapes this checks. Page-level by nature (which booklet/paper a
        # PHYSICAL page belongs to), unaffected by Issue 98's per-question comparison fix, the
        # representative output returned alongside span_results is exactly for this real, still
        # page-level need.
        page_section = _extract_question_page_section(
            representative.bounding_boxes, representative.extracted_text
        )
        if page_section is not None:
            last_real_question_section = page_section
        question_section = last_real_question_section

        for span, ocr_confidence, ocr_record in span_results:
            match, subpart_flag = _lookup_answer_for_span(
                answer_map, section_answer_map, span.question_number, question_section, span.raw_text,
            )
            flags = [f for f in (span.flagged, subpart_flag) if f]
            if match is not None:
                answer_source_page_index, answer_text = match
                used_answer_numbers.add(span.question_number)
            else:
                # A real, expected possibility, not a crash (explicit instruction), same
                # both-directions discipline extract_topology_b applies, see its own comment.
                answer_source_page_index, answer_text = 0, None
                flags.append("no_matching_answer_key_row")
            flag = "; ".join(flags) if flags else None

            # Issue 101's real fix: cross-check both readers' own independently-parsed grid
            # answers for this real question, the highest-priority real signal, overrides the
            # question-text comparison's own result when the two readers' grid answers conflict.
            grid_agreement_status, grid_disagreement_detail, grid_disagreement_type = (
                _apply_grid_cross_check(
                    ocr_record["agreement_status"], ocr_record["disagreement_detail"],
                    ocr_record["disagreement_type"],
                    per_reader_grid_maps, question_section, span.question_number, span.raw_text,
                    per_reader_missing_pages, answer_source_page_index,
                )
            )
            # Issue 213, a real, pre-existing bug found and fixed
            # while making this exact change, not left silently broken: the original condition
            # here (`if grid_agreement_status != ocr_record["agreement_status"]`) only merged the
            # override back into ocr_record when the STATUS itself changed, but
            # _apply_grid_cross_check()'s own "grid agrees, stem still differs" enrichment branch
            # returns the SAME status ("disagree") with an ENRICHED detail message, so that
            # condition was always False for it and the enriched message was computed but never
            # actually stored, confirmed empirically, not just by reading the code: zero real
            # rows in the live corpus match this branch's own real message shape. Always merge
            # unconditionally instead, _apply_grid_cross_check()'s own fallback branch already
            # guarantees it returns all three values completely unchanged when no override
            # applies, so this is safe and strictly more correct, not just simpler.
            ocr_record = {
                **ocr_record, "agreement_status": grid_agreement_status,
                "disagreement_detail": grid_disagreement_detail,
                "disagreement_type": grid_disagreement_type,
            }
            ocr_confidence = "high" if grid_agreement_status == "agree" else "low"

            # Part 3 (Issue 102): split a prose-shaped matched
            # answer into (distilled_answer_value, full_worked_solution_text), reuses Issue
            # 101's own real prose-vs-short-value distinction, not a new heuristic.
            answer_value, worked_solution_text = _split_answer_and_worked_solution(answer_text)

            question = ExtractedQuestion(
                source_paper=pdf_path,
                source_page_index=page_index,
                source_page_label=span.page_label,
                answer_source_file=answer_file,
                answer_source_page_index=answer_source_page_index,
                question_number=span.question_number,
                question_text=span.raw_text,
                answer_value=answer_value,
                worked_solution_text=worked_solution_text,
                has_diagram=span.has_diagram,  # real, PRIMARY signal (the development log,
                                                 # Issue 102 part 2), set by
                                                 # _segment_from_regions() via MinerU2.5's own
                                                 # "image"-type regions, attributed by reading-
                                                 # order position. False when span came
                                                 # from the flat-text fallback path (DeepSeek-OCR,
                                                 # no region-type signal available). May ALSO
                                                 # already be True here from the real, SECONDARY
                                                 # DeepSeek-OCR-marker signal applied above, per
                                                 # page, before this per-span loop runs (Decision
                                                 # Log Issue 233), either way,
                                                 # `span.has_diagram` is the one, final, real value
                                                 # by the time it reaches here, not re-derived.
                ocr_confidence=ocr_confidence,
                school_name=school_name,
                school_name_verified=school_name_verified,
                extraction_flag=flag,
                question_section=question_section,
            )
            # Issue 98's fix: ocr_record is now this span's own real per-question record
            # (segment_and_compare_page()), not a page-level record reused across every span.
            # Issue 101: may additionally reflect the real grid-answer cross-check above.
            pairs.append(QuestionWithOcrRecord(question=question, ocr_record=ocr_record))

    # The other real, expected direction (explicit instruction, same as extract_topology_b), an
    # answer-key row with no matching question anywhere in pdf_path. Logged, not silently dropped.
    for number, (answer_page_index, answer_text) in answer_map.items():
        if number in used_answer_numbers:
            continue
        message = (
            f"answer-key grid question_number {number!r} in {answer_file.name} (page "
            f"{answer_page_index}, answer {answer_text!r}) has no matching segmented question "
            f"anywhere in {pdf_path.name} — needs human review."
        )
        logger.warning("extract_topology_a: %s", message)
        log_ocr_discrepancy(pdf_path.stem, answer_page_index, message, discrepancy_log)

    return ExtractionRunResult(pairs=pairs)


def extract_topology_b(
    pdf_path: Path, school_name: Optional[str], raw_dir: Path,
    discrepancy_log: Optional[Path] = None, school_name_verified: bool = False,
) -> ExtractionRunResult:
    """
    Topology B: question and answer live in the same file (confirmed for School papers/ and the
    Topical book). Populates all four provenance fields:
      source_page_index / source_page_label   <- from pdf_path (the question itself)
      answer_source_file                      <- pdf_path (same file as source_paper)
      answer_source_page_index                <- from pdf_path, the embedded answer section's own
                                                   page, not assumed equal to source_page_index

    Stage C (post-Issue 81): reads Stage B's already-saved reader output, see
    extract_topology_a's docstring for the same PageImage-stand-in note, applies here too. Also
    returns an ExtractionRunResult now (a list of QuestionWithOcrRecord pairs), not a bare list , 
    see extract_topology_a's docstring for the single-writer database reasoning (development log
), same here.
    """
    import fitz

    pairs: list[QuestionWithOcrRecord] = []

    with fitz.open(pdf_path) as doc:
        page_count = doc.page_count

    reader_output_by_page: dict[int, RawReaderOutput] = {}
    all_span_results_by_page: dict[int, list[tuple[QuestionSpan, Literal["high", "low"], dict]]] = {}
    section_by_page: dict[int, Optional[str]] = {}
    last_real_question_section: Optional[str] = None
    # Issue 144 residual fix, tracks whether the immediately
    # preceding page (this loop's own strict page_index order) was itself classify_page()==
    # 'answer_key_grid', so segment_and_compare_page() can pass it down to segment_questions()'s
    # own adjacency check. Topology B's single-file, single-loop shape (question and grid pages
    # interleaved in the same page sequence) is exactly the real shape this fixes (acsprimary's
    # own p43/p44), Topology A's own separate answer_file loop never has this adjacency at all,
    # so extract_topology_a's call site deliberately leaves this at its default False.
    previous_page_was_grid = False

    for page_index in range(page_count):
        provisional_id = f"{pdf_path.stem}:p{page_index}"
        span_results, representative = segment_and_compare_page(
            pdf_path.stem, page_index, provisional_id, raw_dir, discrepancy_log,
            previous_page_was_grid,
        )
        if representative is None:
            previous_page_was_grid = False  # no usable output at all -- can't have been a grid
                                              # page either, and doesn't extend a prior one across
                                              # a skipped page (no real evidence either way)
            continue  # both_failed, or no Stage B output at all, skip this page

        # representative also covers the real, still page-level need locate_embedded_answer_section
        # has below (grid-page table parsing, not a per-question comparison), see
        # segment_and_compare_page()'s own docstring for why one call now serves both needs. Reused
        # here too (zero extra file I/O) to compute this page's own classification for the NEXT
        # iteration's adjacency check, classify_page() is pure/cheap, calling it again is fine.
        previous_page_was_grid = (
            classify_page(representative.bounding_boxes, representative.extracted_text)
            == "answer_key_grid"
        )

        # Real, secondary has_diagram signal (Issue 233), see
        # extract_topology_a's own identical call for the shared reasoning.
        _apply_deepseek_ocr_diagram_secondary_signal(span_results, pdf_path.stem, page_index, raw_dir)

        reader_output_by_page[page_index] = representative
        all_span_results_by_page[page_index] = span_results

        # Real section signal (the development log follow-up, Issue 3's real fix), carried
        # forward across pages exactly like parse_answer_key_grid()'s own section_label, since
        # chij's real shape only prints it once, on that section's own cover page (see
        # _extract_question_page_section()'s own docstring for both real signal shapes). Page-level
        # by nature, unaffected by Issue 98's per-question comparison fix.
        page_section = _extract_question_page_section(
            representative.bounding_boxes, representative.extracted_text
        )
        if page_section is not None:
            last_real_question_section = page_section
        section_by_page[page_index] = last_real_question_section

    answer_map, section_answer_map = locate_embedded_answer_section(
        pdf_path, reader_output_by_page, discrepancy_log
    )
    # Issue 101's real fix : TWO independent, per-reader grid maps for
    # the cross-check below, separate from the single/representative answer_map/section_answer_map
    # just built above, which stays the unchanged, authoritative source for the real answer_value.
    per_reader_grid_maps, per_reader_missing_pages = build_per_reader_grid_maps(
        pdf_path, page_count, raw_dir,
    )
    used_answer_numbers: set[str] = set()  # tracks which answer_map entries a real question
                                             # actually claimed, anything left over at the end is
                                             # a real orphan answer-key row (see below).

    for page_index, span_results in all_span_results_by_page.items():
        question_section = section_by_page.get(page_index)
        for span, ocr_confidence, ocr_record in span_results:
            match, subpart_flag = _lookup_answer_for_span(
                answer_map, section_answer_map, span.question_number, question_section, span.raw_text,
            )
            flags = [f for f in (span.flagged, subpart_flag) if f]
            if match is not None:
                answer_page_index, answer_text = match
                used_answer_numbers.add(span.question_number)
            else:
                # A real, expected possibility, not a crash (explicit instruction), no grid row
                # matched this question's number at all. Flagged for step 5's human review rather
                # than silently left with a None answer_value indistinguishable from "not looked
                # up yet".
                answer_page_index, answer_text = page_index, None
                flags.append("no_matching_answer_key_row")
            flag = "; ".join(flags) if flags else None

            # Issue 101's real fix: cross-check both readers' own independently-parsed grid
            # answers for this real question, the highest-priority real signal, overrides the
            # question-text comparison's own result when the two readers' grid answers conflict.
            grid_agreement_status, grid_disagreement_detail, grid_disagreement_type = (
                _apply_grid_cross_check(
                    ocr_record["agreement_status"], ocr_record["disagreement_detail"],
                    ocr_record["disagreement_type"],
                    per_reader_grid_maps, question_section, span.question_number, span.raw_text,
                    per_reader_missing_pages, answer_page_index,
                )
            )
            # Issue 213, a real, pre-existing bug found and fixed
            # while making this exact change, not left silently broken: the original condition
            # here (`if grid_agreement_status != ocr_record["agreement_status"]`) only merged the
            # override back into ocr_record when the STATUS itself changed, but
            # _apply_grid_cross_check()'s own "grid agrees, stem still differs" enrichment branch
            # returns the SAME status ("disagree") with an ENRICHED detail message, so that
            # condition was always False for it and the enriched message was computed but never
            # actually stored, confirmed empirically, not just by reading the code: zero real
            # rows in the live corpus match this branch's own real message shape. Always merge
            # unconditionally instead, _apply_grid_cross_check()'s own fallback branch already
            # guarantees it returns all three values completely unchanged when no override
            # applies, so this is safe and strictly more correct, not just simpler.
            ocr_record = {
                **ocr_record, "agreement_status": grid_agreement_status,
                "disagreement_detail": grid_disagreement_detail,
                "disagreement_type": grid_disagreement_type,
            }
            ocr_confidence = "high" if grid_agreement_status == "agree" else "low"

            # Part 3 (Issue 102), see extract_topology_a's same note.
            answer_value, worked_solution_text = _split_answer_and_worked_solution(answer_text)

            question = ExtractedQuestion(
                source_paper=pdf_path,
                source_page_index=page_index,
                source_page_label=span.page_label,
                answer_source_file=pdf_path,  # same file as source_paper, per Topology B
                answer_source_page_index=answer_page_index,
                question_number=span.question_number,
                question_text=span.raw_text,
                answer_value=answer_value,
                worked_solution_text=worked_solution_text,
                has_diagram=span.has_diagram,  # real signal, see extract_topology_a's same note
                # Issue 98's fix: per-question ocr_confidence now
                # (segment_and_compare_page()), not a page-level value reused across every span on
                # the page, see that function's own docstring for why this no longer requires
                # re-running the readers per span (it doesn't; segment_questions() is pure CPU
                # parsing of already-saved Stage B output). Issue 101: may additionally reflect
                # the real grid-answer cross-check above.
                ocr_confidence=ocr_confidence,
                school_name=school_name,
                school_name_verified=school_name_verified,
                extraction_flag=flag,
                question_section=question_section,
            )
            # Issue 98's fix: ocr_record is this span's own real per-question record
            # (segment_and_compare_page()), not a page-level record reused across every span.
            # Issue 101: may additionally reflect the real grid-answer cross-check above.
            pairs.append(QuestionWithOcrRecord(question=question, ocr_record=ocr_record))

    # The other real, expected direction (explicit instruction): an answer-key row with no
    # matching question at all. There's no ExtractedQuestion row to attach a flag to for
    # something that was never segmented as a question in the first place, so this is logged
    # durably instead, the same log_ocr_discrepancy discipline used throughout this file,
    # not a new mechanism, not silently dropped.
    for number, (answer_page_index, answer_text) in answer_map.items():
        if number in used_answer_numbers:
            continue
        message = (
            f"answer-key grid question_number {number!r} (page {answer_page_index}, answer "
            f"{answer_text!r}) has no matching segmented question anywhere in this file — needs "
            f"human review: a genuinely missing question, a segment_questions() miss, or (per "
            f"Issue 3) a paper/booklet-scoped number this file's segmented questions never "
            f"produced in that exact bare form."
        )
        logger.warning("extract_topology_b: %s", message)
        log_ocr_discrepancy(pdf_path.stem, answer_page_index, message, discrepancy_log)

    return ExtractionRunResult(pairs=pairs)


# =================================================================================================
# Discovery, Section 3.1's real folder layout
# =================================================================================================

# Issue 96 : known-bad source files, excluded from discovery rather
# than silently extracted under a wrong or unconfirmed school attribution. Confirmed real, not
# guessed, every entry here was rendered and read page-by-page before being added. Never remove
# an entry without re-reading its own comment and the issue it references.
EXCLUDED_SOURCE_FILES: frozenset[str] = frozenset({
    # Filed under the "taonan" normalized key by filename alone, but confirmed (all 46 real pages
    # checked, not just a sample) to be a merged compilation of two originally-separate real
    # documents: an SCGS-branded worked-solutions section (~pages 0-9, its own Q1-Q17 numbering)
    # and a completely unbranded exam paper + answer key (the bulk of the file, pages ~10-45, a
    # DIFFERENT Q1-Q17 numbering with different real answers, e.g. two distinct real "Q17"s).
    # Zero Tao Nan branding appears anywhere in it, so it does NOT belong in the taonan group
    # (taonan's own canonical name stays verified on the strength of its other 6 real files), but
    # the unbranded majority of this specific file also cannot be confidently attributed to SCGS
    # or any other real school, so re-keying the whole file under "scgs" would be its own guess,
    # not a fix. Excluded entirely rather than extracted under either wrong or unconfirmed
    # attribution, see Issue 96 for the full evidence trail.
    "P6_Maths_2024_SA2_taonan.pdf",
})


def discover_pdfs(
    data_dir: Path, subfolders: "Sequence[str]" = ("School papers", "Yearly"),
) -> list[Path]:
    """
    Walks the given subfolders of data_dir (default: data/School papers/ and data/Yearly/,
    Section 3.1's confirmed real layout, the training pool), returning question-source PDFs
    only, answer-suffixed files are excluded here since they're discovered as siblings via
    find_matching_answer_file(), not as independent sources. Also skips anything in
    EXCLUDED_SOURCE_FILES (Issue 96), known-bad files confirmed unusable/unattributable by
    direct evidence, not silently extracted.

    `subfolders` (the development log, Phase 6 Step 0a): the real, minimal entrypoint this
    project's own docstring long referenced as "--extra-files" but never actually built (confirmed
    directly, no such CLI argument existed anywhere in this file's argument parser before this).
    Pass e.g. `("Hold out/200", "Hold out/small eval")` to point discovery at the held-out set
    instead of the training pool, the DEFAULT still walks School papers/Yearly exactly as before,
    so every existing caller (training-pool extraction, render_pages.py's own Stage A) is
    unaffected unless it explicitly opts in to a different subfolder list. Does NOT walk Hold
    out/ by default (Issue 27 / Issue 74's source restriction), that restriction is what
    this parameter exists to let a caller deliberately and explicitly override, not silently
    bypass.
    """
    pdfs: list[Path] = []
    for subfolder in subfolders:
        folder = data_dir / subfolder
        if not folder.exists():
            logger.warning("Expected folder not found, skipping: %s", folder)
            continue
        for pdf_path in sorted(folder.rglob("*.pdf")):
            if pdf_path.name in EXCLUDED_SOURCE_FILES:
                logger.warning(
                    "Excluding known-bad source file from discovery (Issue 96): %s", pdf_path,
                )
                continue
            if "answer" in pdf_path.stem.lower():
                continue  # discovered as a sibling of its question file, not independently
            pdfs.append(pdf_path)
    return pdfs


# =================================================================================================
# Save / logging, JSON checkpoint only. The real DB writes are NOT done here (development log
# "Single-writer database design decided", see write_extracted_data_to_db() further
# below). Stage C's parallel workers (DEFAULT_WORKERS = 22) never hold their own DB session:
# SQLite only supports one true writer at a time even in WAL mode, so 22 processes each opening a
# session would risk real "database is locked" contention for zero throughput benefit, this
# corpus is nowhere near large enough for DB writes to be the bottleneck; reading and comparing
# already are, and parallelism already speeds those up. Each worker's only job is to write its
# own JSON checkpoint (questions AND ocr_extraction_records together, see ExtractionRunResult) , 
# a single, later, non-parallel pass reads every checkpoint in file order and performs the actual
# database writes. This reverses an earlier version of this file (parallelization-pass commit,
# earlier same day) that had each worker call its own DB session directly, that
# design is wrong and was corrected before Phase 2 step 1 ever wired it up for real.
# =================================================================================================

def save(result: ExtractionRunResult, output_path: Path) -> None:
    """
    Writes this file's ExtractionRunResult to its one per-file JSON checkpoint, the resumability
    mechanism (Issue 62) AND, per the single-writer design above, the ONLY thing a Stage C
    worker does with this data. No DB write happens here, from a worker process, ever.
    write_extracted_data_to_db() reads this same checkpoint later, in a separate single-writer
    pass, to perform the actual database writes.

    On-disk shape: a flat JSON list of {"question": {...}, "ocr_record": {...} | null} pairs, one
    per QuestionWithOcrRecord (the development log, resolving the question_id linking gap
    flagged after commit 214aa39), NOT two separate "questions"/"ocr_extraction_records" lists
    matched by a provisional string id, which is what the previous version of this function wrote
    and which is exactly the fragile design that gap came from. Each pair already carries both
    halves together, so nothing downstream needs to re-match them by anything.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            [
                {"question": _question_to_dict(p.question), "ocr_record": p.ocr_record}
                for p in result.pairs
            ],
            indent=2, default=str,
        ),
        encoding="utf-8",
    )


class MarkerDatabaseDesyncError(RuntimeError):
    """
    Real, fail-loud guard (Issue 158's own real follow-up FIX, not
 the original incident itself). Raised by write_extracted_data_to_db() when its own
    marker state disagrees with real, live database content: a checkpoint has NO db_write_log
    marker (meaning "not yet written, safe to insert"), but the database ALREADY has real
    Question rows for that file's own source_paper. This is exactly the precondition of the
    original incident (a stale/cleared db_write_log directory letting this function re-run
    against already-populated tables, 15x over, ~81,479 duplicate rows found and fixed via a
    clean truncate-and-reload), this guard exists specifically so that shape cannot silently
    reproduce. Fails BEFORE any row is inserted for this run (see _find_marker_db_desyncs(),
    called as a pre-flight check over every checkpoint, not inline per-file), nothing partial is
    left behind by this refusal; files a PRIOR run already committed are untouched either way.
    """


def _find_marker_db_desyncs(
    checkpoint_paths: "list[Path]", db_write_log_dir: Path, session,
) -> "list[dict]":
    """
    Real, live check against actual database content, the fix for Issue 158's own real,
 still-open follow-up (the development log, closed for real): the marker alone
    ("this process believes it already wrote this file") was never checked against what the
    database actually contains. For every checkpoint whose db_write_log marker is ABSENT
    (meaning this pass believes it has never written that file), checks whether the database
    already has real Question rows for that file's own source_paper. A hit means the marker and
    the database have gone out of sync, the exact precondition for the original incident (a
    cleared/stale db_write_log directory letting this function re-insert into already-populated
    tables), never a normal, safe state during an ordinary resumed or incremental run (files
    already committed keep BOTH their marker and their DB rows together; brand-new files have
    NEITHER).

    Deliberately checks source_paper ONLY, never the fuller natural key (source_paper,
    source_page_index, question_number, question_section, answer_source_page_index). Candidate
    fix (a) from this investigation (a natural-key-based upsert) was considered and REJECTED for
    exactly this reason: the real, confirmed 5-group residual (the development log26-09-07)
    already has different real questions sharing an identical full natural key (e.g.
    `nanyang` 2022 SA2 p42's own `Q4`, one row a garbled OMR-instruction-block read, the other an
    unrelated rounding-calculation fragment). An upsert keyed on that natural key would either
    silently overwrite one real row's content with the other's, or need real row-level content
    comparison to tell them apart, new complexity and a new real data-loss risk, for a mechanism
    whose only job here is deciding whether to REFUSE a run, never whether to insert, skip, merge,
    or overwrite any individual row. This check never merges or drops anything, it only ever
    gates a fail-loud abort of the whole pre-flight pass; the 5-group residual is untouched by it
    either way, exactly as intended.
    """
    desyncs = []
    for checkpoint_path in checkpoint_paths:
        pdf_stem = checkpoint_path.stem
        marker_path = db_write_log_dir / f"{pdf_stem}.done"
        if marker_path.exists():
            continue  # normal, expected state, already committed, nothing to verify

        pairs = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if not pairs:
            continue  # nothing this checkpoint would ever write, nothing to desync
        source_paper = pairs[0]["question"]["source_paper"]

        existing_row_count = (
            session.query(Question).filter(Question.source_paper == source_paper).count()
        )
        if existing_row_count > 0:
            desyncs.append({
                "pdf_stem": pdf_stem,
                "source_paper": source_paper,
                "existing_row_count": existing_row_count,
            })
    return desyncs


def _find_stale_checkpoint_desyncs(
    checkpoint_paths: "list[Path]", db_write_log_dir: Path,
) -> "list[dict]":
    """
    Real, second desync shape for the SAME staleness class `_checkpoint_is_stale()` fixes on the
    extraction side (Issue 257), this is the DATABASE-side half. A
    checkpoint CAN now be refreshed (overwritten with newer content) after
    `write_extracted_data_to_db()` already committed real rows FROM the old version of that same
    checkpoint, the exact real, live gap that let `agreement_status` freeze stale for thousands
    of real rows for days, undetected. Deliberately does NOT attempt to re-insert or upsert
    anything here, matches `_find_marker_db_desyncs()`'s own established real reasoning (Issue
    158/230's own rejected upsert candidate: the confirmed 5-group natural-key residual makes a
    keyed upsert unsafe), this only ever detects the real condition and REFUSES the run, the same
    fail-loud philosophy already established for the sibling marker-absent-but-DB-populated shape.
    A real human then decides how to reconcile (a narrow, targeted refresh script, as was done
    for Issue 257, or a deliberate, reviewed re-extraction), never silently, automatically, inside this single-writer DB pass.
    """
    desyncs = []
    for checkpoint_path in checkpoint_paths:
        pdf_stem = checkpoint_path.stem
        marker_path = db_write_log_dir / f"{pdf_stem}.done"
        if not marker_path.exists():
            continue  # the OTHER desync shape (_find_marker_db_desyncs) already covers this case
        if checkpoint_path.stat().st_mtime > marker_path.stat().st_mtime:
            desyncs.append({"pdf_stem": pdf_stem, "checkpoint_path": str(checkpoint_path),
                             "marker_path": str(marker_path)})
    return desyncs


def write_extracted_data_to_db(extracted_dir: Path, db_write_log_dir: Path) -> None:
    """
    The single, non-parallel DB-writing pass (the development log, "Single-writer database
    design decided"; design specification Section 15's matching Stage C note). Reads every worker's
    JSON checkpoint under extracted_dir IN FILE ORDER (not the order workers happened to finish in
   , real, deterministic ordering, since checkpoint filenames are the PDF stems and glob+sorted
    gives a stable order) and performs the actual `questions`/`ocr_extraction_records` writes, one
    transaction per file, run after Stage C's parallel workers have all finished writing their own
    checkpoints, never called from inside a worker.

    REAL, not a stub (Issue 102), Question/OCRExtractionRecord/
    get_session/init_db are wired to real SQLAlchemy models and a real SQLite engine now
    (backend/app/models/, backend/app/db/session.py). This docstring's account of the design
    below (single-writer, flush-then-link, marker-based resumability) was written and validated
    against the shape before the real layer existed; re-validated against a REAL SQLite database
    file when the real layer was wired in, not just re-confirmed to still import cleanly.

    RESOLVED : the question_id linking gap flagged in an earlier version
    of this docstring. Per pair: insert the Question row, `session.flush()` (assigns a real id
    WITHOUT a full commit, flush and commit are different operations; flush is the only one
    needed to make the id available), THEN set the ocr_record's `question_id` to that real id
    before inserting the OCRExtractionRecord row. This establishes the actual FK relationship
    correctly, no more provisional "pdf_stem:pN" string surviving into the database. Commits once
    per FILE (all of that file's pairs in one transaction), not once per pair, flush is what
    makes the id available for linking, commit is only about durability, conflating the two would
    mean either needless per-pair I/O or (worse) trying to link against an uncommitted, and
    therefore not guaranteed durable, id.

    RESOLVED (the development log, later): the double-insert-on-rerun gap flagged in the next
    earlier version of this docstring. Marker-based resumability, the SAME pattern already used
    everywhere else in this pipeline (Stage A/B's own per-page "skip if output already exists"
    checks, Issue 62's principle), not a new mechanism. Before processing a checkpoint, checks
    whether `db_write_log_dir/{pdf_stem}.done` already exists; if so, skips that file entirely , 
    already committed, don't re-insert. The marker is written ONLY after that file's
    `session.commit()` succeeds, never before: if anything fails partway through a file's pairs
    (before commit), no marker gets written, and a retry correctly reprocesses the whole file from
    scratch, consistent with the one-transaction-per-file commit already established above, since
    a failed file has nothing durable in the database to conflict with on retry anyway. Marker
    content is minimal (a timestamp and the pair count written), enough for debugging later, not
    elaborate.

    RESOLVED (Issue 158's own real follow-up, closed for real, not
 just the immediate mess cleaned up at the time): the marker-vs-real-database-
    content desync gap. The ORIGINAL incident was the mirror-image case, db_write_log cleared
    (or never populated) while the database already had real rows, causing a stale-negative
    marker check to re-insert everything 15x over (~81,479 duplicate rows, fixed at the time via
    a clean truncate-and-reload, but the mechanism itself was left unchanged). Now checked for
    real, live, before any row is written this run: _find_marker_db_desyncs() queries the actual
    database for every checkpoint whose marker is absent, if the database already has real
    Question rows for that file's own source_paper despite no marker existing, this function
    raises MarkerDatabaseDesyncError and writes nothing at all for this run, rather than silently
    re-inserting. This is a REFUSE-AND-REPORT design, not an automatic upsert/repair: candidate
    fix (a) (a natural-key-based upsert) was considered and rejected, see
    _find_marker_db_desyncs()'s own docstring for why the real, confirmed 5-group natural-key
    residual makes that specific approach unsafe here. The one remaining known limitation (not
    the same shape as before): a manual DB wipe that ALSO leaves stale marker files AND happens to
    remove every row for that source_paper is not distinguishable from a legitimate first-ever
    run by this check, that combination is a deliberate, correctly-authorized wipe (the operator
    controls both the database and the marker directory together), not the accidental-partial-
    clear shape this fix targets, so it is intentionally out of scope, not an oversight.

    Still open, not closed by either fix above: if a flush succeeds for some pairs in a
    file but a LATER pair in that same file fails (a DB error partway through), the single
    per-file transaction means everything flushed-but-not-yet-committed in it rolls back together
    with the failure, so a file lands completely or not at all, never partially linked, and (per
    the marker fix above) gets correctly retried whole on the next run. That part is fine.
    """
    init_db()  # idempotent (create_all's own checkfirst=True), safe to call on every real run,
               # not just the first; see init_db()'s own docstring.

    db_write_log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = sorted(extracted_dir.glob("*.json"))
    logger.info("Single-writer DB pass: %d checkpoint(s) found, in file order.", len(checkpoint_paths))

    with get_session() as session:  # type: ignore
        # Real, live pre-flight check (Issue 158's own follow-up fix)
        #, BEFORE any row is written this run, not per-file inline, so one refusal reports every
        # affected file in one pass rather than requiring a re-run per file to discover the next
        # one. See MarkerDatabaseDesyncError's and _find_marker_db_desyncs()'s own docstrings.
        desyncs = _find_marker_db_desyncs(checkpoint_paths, db_write_log_dir, session)
        if desyncs:
            detail = "; ".join(
                f"{d['pdf_stem']} (source_paper={d['source_paper']!r}, "
                f"{d['existing_row_count']} existing row(s) in the database, no db_write_log "
                f"marker present)"
                for d in desyncs
            )
            logger.error(
                "REFUSING TO WRITE — marker/database desync detected for %d checkpoint(s): %s. "
                "This is the exact precondition of the earlier duplication incident "
                "(Issue 158): a checkpoint has no db_write_log marker, but the "
                "database already has real rows for its source_paper. Either restore the correct "
                "marker file(s) for these files, or deliberately truncate their rows from "
                "questions/ocr_extraction_records before re-running — do not clear db_write_log "
                "without also clearing the database rows it was guarding.",
                len(desyncs), detail,
            )
            raise MarkerDatabaseDesyncError(
                f"{len(desyncs)} checkpoint(s) have no db_write_log marker but the database "
                f"already has rows for their source_paper — refusing to write any data this run. "
                f"{detail}"
            )

        # Real, second desync shape (Issue 257), the sibling check to
        # the one above: a checkpoint that WAS already committed (marker present) but has since
        # been refreshed with newer real content (Issue 62's own resumability skip, now fixed
        # by `_checkpoint_is_stale()`, correctly overwriting a checkpoint whose underlying Stage B
        # raw data changed), the database still holds whatever the OLD checkpoint said. Same
        # fail-loud philosophy as the check above: refuse, don't silently re-insert or upsert.
        stale_desyncs = _find_stale_checkpoint_desyncs(checkpoint_paths, db_write_log_dir)
        if stale_desyncs:
            stale_detail = "; ".join(d["pdf_stem"] for d in stale_desyncs)
            logger.error(
                "REFUSING TO WRITE — %d checkpoint(s) were refreshed with newer real content "
                "AFTER their own db_write_log marker was written (Issue 257): %s. "
                "The database's own rows for these files reflect a STALE version of the "
                "checkpoint. Do not clear the marker and let this function re-insert (real "
                "duplicate-row risk, Issue 158) — reconcile with a real, narrow, targeted "
                "refresh script instead (as was done for Issue 257), or a deliberate, reviewed "
                "re-extraction.",
                len(stale_desyncs), stale_detail,
            )
            raise MarkerDatabaseDesyncError(
                f"{len(stale_desyncs)} checkpoint(s) refreshed after their own db_write_log "
                f"marker was written — refusing to write any data this run. {stale_detail}"
            )

        for checkpoint_path in checkpoint_paths:
            pdf_stem = checkpoint_path.stem
            marker_path = db_write_log_dir / f"{pdf_stem}.done"
            if marker_path.exists():
                continue  # resumability, already committed on a prior run, don't re-insert

            pairs = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            for pair in pairs:
                q = pair["question"]
                question_row = Question(  # type: ignore
                    source_paper=q["source_paper"],
                    source_page_index=q["source_page_index"],
                    source_page_label=q["source_page_label"],
                    answer_source_file=q["answer_source_file"],
                    answer_source_page_index=q["answer_source_page_index"],
                    ocr_confidence=q["ocr_confidence"],
                    school_name=q.get("school_name"),  # real bug fix (the development log,
                                                          # Issue 102 step 1): school_name and
                                                          # school_name_verified were genuinely
                                                          # missing from this constructor call , 
                                                          # present on ExtractedQuestion and in
                                                          # every JSON checkpoint the whole time
                                                          # (Issue 92/93's real verification
                                                          # work), silently dropped on the way
                                                          # into the database because this
                                                          # function had never run against a real
                                                          # DB layer before to surface the gap.
                    school_name_verified=q.get("school_name_verified", False),
                    verification_status=q["verification_status"],
                    verified_at=q["verified_at"],
                    superseded_by=q["superseded_by"],
                    extraction_flag=q.get("extraction_flag"),  # .get(), not [], tolerate older
                                                                  # checkpoints written before this
                                                                  # field existed (development log
                                                                  # follow-up)
                    question_section=q.get("question_section"),  # same .get() tolerance, same
                                                                    # reason, this round's new field
                    question_number=q["question_number"],
                    question_text=q["question_text"],
                    answer_value=q["answer_value"],
                    worked_solution_text=q["worked_solution_text"],
                    has_diagram=q["has_diagram"],
                )
                session.add(question_row)
                session.flush()  # assigns question_row.id, no full commit yet, see docstring

                ocr_record = pair.get("ocr_record")
                if ocr_record is not None:
                    ocr_record = dict(ocr_record)  # don't mutate the loaded JSON in place
                    ocr_record["question_id"] = question_row.id  # real id now, not provisional
                    session.add(OCRExtractionRecord(**ocr_record))  # type: ignore
            session.commit()  # once per file, not once per pair, see docstring

            # Marker written ONLY after commit succeeds, never before, see docstring.
            marker_path.write_text(
                json.dumps(
                    {
                        "committed_at": datetime.now(timezone.utc).isoformat(),
                        "pairs_written": len(pairs),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    logger.info("Single-writer DB pass complete.")


def log_failure(pdf_path: Path, error: Exception, failures_log: Path) -> None:
    """Per-file failure logging (Issue 62), never silently swallowed."""
    failures_log.parent.mkdir(parents=True, exist_ok=True)
    with failures_log.open("a", encoding="utf-8") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()}\t{pdf_path}\t{type(error).__name__}: {error}\n")
    logger.error("FILE FAILED: %s — %s: %s", pdf_path, type(error).__name__, error)


def log_question_failure(question_id: object, reason: str) -> None:
    """Per-question failure logging (Issue 62's principle, one level down)."""
    logger.error("QUESTION FAILED: %s — %s", question_id, reason)


def log_ocr_discrepancy(
    pdf_stem: str, page_index: int, message: str, discrepancy_log: Optional[Path],
) -> None:
    """
    Durable log for real, confirmed segmentation discrepancies that get flagged for human review
    rather than silently resolved either way, Bug 1 (low-confidence region-vs-flat-text mismatch)
    and Bug 3's collision-detection safety net (the development log follow-up), both explicit
    instructions: "make sure this gets logged somewhere durable, not just printed and lost."

    logger.warning() ALONE is not durable enough here: Stage C's ProcessPoolExecutor workers use
    Windows' spawn start method, each getting a fresh interpreter that does not inherit the
    parent's logging.basicConfig(), console visibility of a worker's own log lines is already
    documented as unreliable (see _extract_one_file's docstring). Writes directly to a file via
    plain I/O instead, the same durable-logging pattern log_failure() above already uses.

    discrepancy_log is Optional so segment_questions() and its helpers can still be called
    directly (unit tests, ad hoc spot-checks) without requiring a real log path, the finding is
    still surfaced via logger.warning() at the call site either way, just not durably in that case.
    """
    if discrepancy_log is None:
        return
    discrepancy_log.parent.mkdir(parents=True, exist_ok=True)
    with discrepancy_log.open("a", encoding="utf-8") as f:
        f.write(
            f"{datetime.now(timezone.utc).isoformat()}\t{pdf_stem}\tpage={page_index}\t{message}\n"
        )


def _question_to_dict(q: ExtractedQuestion) -> dict:
    d = q.__dict__.copy()
    d["source_paper"] = str(q.source_paper)
    d["answer_source_file"] = str(q.answer_source_file)
    return d


# =================================================================================================
# Main loop, Stage C only (Issue 81). Reads Stage A/B's already-saved output; does not render
# pages or call a reader itself. Run Stage A (render_pages.py) and Stage B (each reader's own
# run_on_directory, in its own venv) first, this stage's output depends on both having run.
#
# PARALLELIZED (speed pass): pure CPU work, no GPU, no model loaded, safe to
# parallelize across files freely, same reasoning as render_pages.py's Stage A (see that file's
# module docstring for the real core-count numbers this is sized from). Parallelized across FILES,
# not pages within a file, simpler unit, and each file's extract_topology_a/b call is already a
# single independent unit of work with its own try/except (Issue 62).
#
# Single-writer database design (the development log, "Single-writer database design
# decided", reverses this file's own earlier same-day parallelization-pass commit, which had
# each worker call its own DB session directly, reasoning that SQLite's WAL mode was designed for
# concurrent writers; that reasoning was wrong, WAL improves read/write concurrency, it does NOT
# allow multiple simultaneous writers, SQLite still serializes writers regardless of journal
# mode). Stage C's parallel workers (`_extract_one_file` below, DEFAULT_WORKERS = 22) each write
# ONLY their own JSON checkpoint (via save(), see above), never a database row, never their own
# DB session. A single separate, non-parallel pass (write_extracted_data_to_db(), see above) reads
# every checkpoint afterward, in file order, and performs the actual `questions`/
# `ocr_extraction_records` writes. This corpus is nowhere near large enough for DB writes to be
# the bottleneck regardless, reading and comparing already are, and those are exactly what this
# parallelism speeds up, so there was no throughput reason to risk concurrent writers in the
# first place.
# =================================================================================================

def _checkpoint_is_stale(checkpoint_path: Path, raw_dir: Path, pdf_stem: str) -> bool:
    """
    Real fix for Issue 257 : a checkpoint's own "already exists"
    skip-check (Issue 62's own resumability pattern) had no concept of the underlying Stage B
    raw data having changed SINCE that checkpoint was written, confirmed as a real, live,
    corpus-wide problem, not theoretical: every one of the training pool's 132 Stage C checkpoints
 was written in one ~2.6s window (real file mtimes), while DeepSeek-OCR's own
    Stage B pass continued producing real new page output for almost 6 more days, until
 Stage C was never re-run, so `agreement_status` for thousands of real rows froze
 at whatever fraction of Stage B had completed, never reflecting the real,
    complete data that arrived days later. This is a structurally RECURRING risk, not a one-time
    accident: ANY time Stage B produces new/updated raw output for a file that already has a
    Stage C checkpoint (a resumed OCR run, a re-run reader, a newly-added page), the OLD
    unconditional `if output_path.exists(): return skipped` would silently keep serving the STALE
    checkpoint forever, with no signal anything was wrong, exactly the real, observed symptom.

    Real, minimal fix: compare the checkpoint's own mtime against the newest real raw Stage B file
    mtime for this PDF, across BOTH readers, every real page, if any raw file is newer than the
    checkpoint, the checkpoint no longer reflects reality and must be recomputed. Cheap (a
    directory walk + `stat()` per real page, no file content read) and safe: this function ONLY
    ever affects whether `_extract_one_file()` recomputes and OVERWRITES its own checkpoint JSON , 
    it never touches the database, so it carries none of `write_extracted_data_to_db()`'s own real
    duplicate-insert risk (Issue 158/230's own established caution). See
    `_find_marker_db_desyncs()`'s own docstring for the matching, deliberately separate, fail-loud
    guard on the DATABASE side of this same real staleness class.
    """
    checkpoint_mtime = checkpoint_path.stat().st_mtime
    for reader in ("mineru25", "deepseek_ocr"):
        reader_dir = raw_dir / reader / pdf_stem
        if not reader_dir.is_dir():
            continue
        for raw_file in reader_dir.glob("*.json"):
            if raw_file.stat().st_mtime > checkpoint_mtime:
                return True
    return False


def _extract_one_file(
    args: "tuple[str, str, str, str, str]",
) -> "tuple[str, str, int, str | None]":
    """
    Module-level, not a nested closure, required to be picklable for ProcessPoolExecutor on
    Windows (spawn start method, not fork). One task = one PDF file, processed end-to-end
    (topology detection, extraction, save), mirrors exactly what the old sequential loop below
    did per iteration, just dispatched to a worker process instead of run inline. Writes only its
    own JSON checkpoint (see save() above), never touches the database itself, single-writer
    design, see the module comment block above this function.

    Returns (status, pdf_path_str, question_count, error_or_None). Note: logger.* calls made from
    inside a worker process (deep inside extract_topology_a/b, save(), etc.) may not reach this
    process's console, Windows' spawn start method gives each worker a fresh interpreter that
    doesn't inherit the parent's logging.basicConfig() configuration. This does NOT affect
    correctness: log_failure() writes directly to failures_log via plain file I/O regardless of
    logging state, and the JSON checkpoint / this function's return value are what run_extraction
    actually relies on for its own summary, only the live console visibility of per-file log
    lines from inside a worker is reduced, a cosmetic trade-off, not a data-loss one.
    """
    pdf_path_str, raw_dir_str, extracted_dir_str, failures_log_str, discrepancy_log_str = args
    pdf_path = Path(pdf_path_str)
    raw_dir = Path(raw_dir_str)
    extracted_dir = Path(extracted_dir_str)
    failures_log = Path(failures_log_str)
    discrepancy_log = Path(discrepancy_log_str)

    output_path = extracted_dir / f"{pdf_path.stem}.json"
    if output_path.exists() and not _checkpoint_is_stale(output_path, raw_dir, pdf_path.stem):
        return ("skipped", pdf_path_str, 0, None)

    try:
        # Issue 71: check topology BEFORE extracting, don't assume one
        answer_file = find_matching_answer_file(pdf_path)
        school_name, school_name_verified = normalized_school_name(pdf_path)

        if answer_file is not None:
            result = extract_topology_a(
                pdf_path, answer_file, school_name, raw_dir, discrepancy_log,
                school_name_verified=school_name_verified,
            )
        else:
            result = extract_topology_b(
                pdf_path, school_name, raw_dir, discrepancy_log,
                school_name_verified=school_name_verified,
            )

        save(result, output_path)
        return ("ok", pdf_path_str, len(result.pairs), None)

    except Exception as e:  # noqa: BLE001, intentionally broad, Issue 62
        log_failure(pdf_path, e, failures_log)
        return ("failed", pdf_path_str, 0, f"{type(e).__name__}: {e}")


def run_extraction(
    data_dir: Path, raw_dir: Path, extracted_dir: Path, failures_log: Path,
    limit: Optional[int] = None, workers: Optional[int] = None,
    db_write_log_dir: Optional[Path] = None, discrepancy_log: Optional[Path] = None,
    source_subdirs: "Optional[Sequence[str]]" = None,
) -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if workers is None:
        workers = DEFAULT_WORKERS
    if db_write_log_dir is None:
        db_write_log_dir = extracted_dir.parent / "db_write_log"
    if discrepancy_log is None:
        # Bug 1/Bug 3 follow-up fix, durable log for real, confirmed
        # segmentation discrepancies flagged for human review rather than silently resolved
        # either way. See log_ocr_discrepancy()'s docstring for why this can't just be
        # logger.warning() alone under Stage C's spawned worker processes.
        discrepancy_log = extracted_dir.parent / "ocr_discrepancies.log"

    # the development log, Phase 6 Step 0a: discover_pdfs()'s own new `subfolders` param,
    # defaulted here to its own School-papers/Yearly default when not explicitly overridden, see
    # discover_pdfs()'s docstring for why this exists (the real "--extra-files" gap).
    pdf_paths = discover_pdfs(data_dir, subfolders=source_subdirs) if source_subdirs else discover_pdfs(data_dir)
    if limit is not None:
        pdf_paths = pdf_paths[:limit]
        logger.info("Limiting run to first %d file(s) for a small-subset test.", limit)

    logger.info("Discovered %d source PDF(s) to process.", len(pdf_paths))

    tasks = [
        (str(p), str(raw_dir), str(extracted_dir), str(failures_log), str(discrepancy_log))
        for p in pdf_paths
    ]
    workers = max(1, min(workers, len(tasks) or 1))
    logger.info("Dispatching %d file task(s) across %d worker process(es).", len(tasks), workers)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_extract_one_file, t) for t in tasks]
        for future in as_completed(futures):
            status, pdf_path_str, n_questions, error = future.result()
            if status == "ok":
                logger.info("OK: %s (%d questions)", Path(pdf_path_str).name, n_questions)
            elif status == "skipped":
                pass  # resumability, already done
            else:
                logger.error("FILE FAILED: %s — %s", Path(pdf_path_str).name, error)

    # Single-writer database pass, deliberately OUTSIDE the parallel pool above, run once, after
    # every worker has finished writing its own JSON checkpoint. See the module comment block
    # above _extract_one_file for why this isn't done from inside the workers themselves.
    write_extracted_data_to_db(extracted_dir, db_write_log_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Job A extraction pipeline — Stage C (merge). Run Stage A and Stage B first.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Root data/ folder (Section 3.1 layout).")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/extracted/raw"), help="Stage B's saved per-reader, per-page JSON root.")
    parser.add_argument("--extracted-dir", type=Path, default=Path("data/extracted/questions"), help="Per-file JSON checkpoint output.")
    parser.add_argument("--failures-log", type=Path, default=Path("data/extracted/failures.log"))
    parser.add_argument(
        "--db-write-log-dir", type=Path, default=None,
        help="Marker directory for the single-writer DB pass's own resumability "
             "(default: <extracted-dir>/../db_write_log). One {pdf_stem}.done file per "
             "file successfully committed to the database.",
    )
    parser.add_argument(
        "--discrepancy-log", type=Path, default=None,
        help="Durable log for real, confirmed segmentation discrepancies flagged for human "
             "review (low-confidence region-vs-flat-text mismatches, ambiguous unnumbered-span "
             "number collisions) — default: <extracted-dir>/../ocr_discrepancies.log.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N discovered files — use this for the small real-subset test "
             "run before scaling up to the full ~140-PDF corpus.",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help=f"Parallel worker processes (default {DEFAULT_WORKERS}, sized off this machine's "
             "24 physical cores minus 2 reserved for the OS — re-measure on different hardware).",
    )
    parser.add_argument(
        "--source-subdirs", type=str, nargs="+", default=None,
        help="Phase 6 Step 0a: override the default training-pool "
             "subfolder walk (School papers, Yearly) with an explicit list of subfolders under "
             "--data-dir to discover PDFs from instead — e.g. "
             "--source-subdirs \"Hold out/200\" \"Hold out/small eval\" to extract the held-out "
             "set specifically. Omit for normal training-pool extraction (unchanged default "
             "behavior). This is the real, minimal entrypoint this file's own discover_pdfs() "
             "docstring long referenced as \"--extra-files\" but never actually built.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    run_extraction(
        args.data_dir, args.raw_dir, args.extracted_dir, args.failures_log,
        limit=args.limit, workers=args.workers, db_write_log_dir=args.db_write_log_dir,
        discrepancy_log=args.discrepancy_log, source_subdirs=args.source_subdirs,
    )


if __name__ == "__main__":
    main()
