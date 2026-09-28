"""
Regression tests for Issue 224 and its follow-up, Issue 226.

Issue 224 addressed the 98.7% `single_reader_only` misalignment found while investigating the
Issue 222 reconciliation measurement. The tests are built from the 15 hand-checked cases in
that investigation (both readers' spans for each page), reduced to their structural shape rather
than byte-exact page dumps, as in `test_segment_questions_bare_number_adjacency.py`.

Of the three fixes first designed for Issue 224, full-corpus validation (132/132 files)
showed that one, a general "ascending question-number sequence" phantom filter, collided with
Issue 3: question numbers legitimately reused across booklets, sections or columns. It merged two
different, correctly numbered questions on `P6_Maths_2022_SA2_chij` p13 (numbers 17/18/18 reused
across sections), so it was reverted. That left the fraction-numerator phantom as an open gap.

Issue 226 closes the gap with a narrow, fraction-specific replacement,
`_fraction_numerator_phantom_evidence()` and `_suppress_fraction_numerator_phantoms()`, wired into
`segment_and_compare_page()`. It has to work across readers, not in `segment_questions()` alone:
by the time DeepSeek-OCR's text is examined, the denominator is already gone, so the only evidence
is in the other reader's more complete transcription.

Full-corpus validation found three false positives, each fixed in turn:
(1) A question that mentions a fraction partway through its sentence (Singapore-Chinese-Girls
    p16 Q1, `\\frac{1}{5}`). Fixed by requiring the matched overlap to start at position 0 of the
    candidate.
(2) An unrelated later question that follows a fraction after other content (nanhua p2 Q6, after
    a `\\frac{6}{7}` MCQ option in Q5). Fixed by also requiring the match to start near the
    beginning of the window just after the fraction.
(3) A fraction that is the last of an MCQ question's four options, followed directly by the next
    question. PSLE question numbers are small integers, so a numerator matching the next number
    is common; this recurred in four files (Methodist-Girls p3, Henry-Park-SA2 p3, Nanyang p4,
    redswastika p3). Fixed by rejecting any fraction immediately preceded by an MCQ option marker.
After all three, a diff of the complete per-question record set before and after showed zero
`agree` -> `disagree`/`single_reader_only` regressions. None of the checks fire on the Issue 3
page (`test_fraction_numerator_phantom_never_fires_on_real_issue_3_reuse_case`): it has
fractions, but neither reused number is one of their numerators.

The three shipped fixes:
  1. **Boilerplate phrase recognition** (`_MARGIN_NOTE_PATTERNS`, applied to the region-based
     path via `_is_recognized_boilerplate()` and to the flat-text path via
     `_strip_trailing_boilerplate_lines()`). Cause: MinerU2.5 emits a printed page-footer numeral
     as a detached bare-number region, followed by navigation text ("Go on to Booklet B", the OCR
     typo "Go on to Booket B", "(Go on to the next page)"), seen across
     `P6-Maths-2021-CA1-Rosyth` pp3-7. The flat-text check was added after validation showed a
     regression from the asymmetry (see `test_flat_text_trailing_boilerplate_stripped*`). It also
     fixes cause 1 of the label-drop family below and restores parity between the two paths.
  2. **Cross-reader content recovery** (`_find_recovered_span_text()`, wired into
     `segment_and_compare_page()`). Causes: (a) label drop, where a reader captures the question
     content but drops the leading number (Henry-Park p11/p21/p23/p25/p29/p30); (b) region merge
     miss, where a reader folds a second question into the still-open previous span with no
     number for the second question (Henry-Park p8). In both, `segment_questions()`, working from
     one reader's data, cannot tell a new unnumbered question from a continuation. The other
     reader's independent segmentation is direct evidence that the content exists.
  3. **Fraction-numerator phantom suppression** (`_fraction_numerator_phantom_evidence()` and
     `_suppress_fraction_numerator_phantoms()`, wired into `segment_and_compare_page()`). Cause:
     DeepSeek-OCR renders a LaTeX fraction without its denominator, leaving a bare numerator that
     its flat-text regex reads as a new question (Henry-Park p13). It requires a
     `\\frac{N}{...}` structure in the other reader's text, an overlap starting at the candidate's
     position 0 and near the start of the window after the fraction, and a fraction that is not
     an MCQ option. It never relies on sequence position alone, which was the reverted filter's
     mistake.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.pipeline.extract import (
    PageImage,
    RawReaderOutput,
    _find_recovered_span_text,
    _is_recognized_boilerplate,
    _strip_trailing_boilerplate_lines,
    segment_and_compare_page,
    segment_questions,
)


def _text_region(content: str, region_type: str = "text") -> dict:
    return {"type": region_type, "content": content}


def _write_raw_page(
    raw_dir, reader_name, pdf_stem, page_index, extracted_text, bounding_boxes=None, structure=None,
):
    page_dir = raw_dir / reader_name / pdf_stem
    page_dir.mkdir(parents=True, exist_ok=True)
    (page_dir / f"{page_index}.json").write_text(
        json.dumps({
            "extracted_text": extracted_text,
            "structure": structure,
            "bounding_boxes": bounding_boxes,
            "field_confidence": None,
            "error": None,
        }),
        encoding="utf-8",
    )


# =================================================================================================
# Fix 1: boilerplate phrase recognition, region-based path (Rosyth CA1 2021 pp3-7)
# =================================================================================================

def test_footer_numeral_phantom_prevented_region_based():
    """A page-footer numeral followed by navigation boilerplate does not become a question.

    Shape from Rosyth p7: the region list ends with question '15', then a footer numeral '7' that
    looks like a detached question label, then footer and navigation text. Without the fix, '7'
    became a question whose only content was boilerplate. The boilerplate is now recognised
    before it attaches to the bare number, resetting pending_bare_number, so no phantom is made.
    """
    page = PageImage(source_pdf=Path("P6-Maths-2021-CA1-Rosyth"), page_index=7, image_bytes=b"")
    bounding_boxes = [
        _text_region("15 A piece of wire is bent to form 1 quarter circle."),
        _text_region("(1) 22 cm"),
        _text_region("(2) 36 cm"),
        _text_region("(3) 47 cm"),
        _text_region("(4) 75 cm"),
        _text_region("7"),  # the phantom page-footer numeral
        _text_region("Go on to Booket B"),  # OCR typo ("Booket" not "Booklet")
        _text_region("Scanned with CamScanner", region_type="footer"),
        _text_region("www.testpapersfree.com", region_type="footer"),
    ]
    reader_output = RawReaderOutput(extracted_text="dummy", bounding_boxes=bounding_boxes)
    spans = segment_questions(page, reader_output)
    numbers = [s.question_number for s in spans]
    assert "7" not in numbers, f"phantom footer numeral '7' should never be created, got {numbers!r}"
    assert numbers == ["15"]
    assert "Go on to Booket B" not in spans[0].raw_text  # discarded, not merged into Q15
    assert spans[0].raw_text.endswith("(4) 75 cm")


def test_footer_numeral_and_boilerplate_combined_in_one_region_prevented():
    """A numeral and navigation text in one region do not become a question (Catholic-High p11).

    The region "9 (Go on to the next page)" starts with a digit, so the whole-content check in
    `_is_recognized_boilerplate()` misses it, while `_match_question_start()` reads "9" plus a
    content-length rest. Both readers used to produce the same phantom "question 9", which then
    "agreed" on meaningless content. Checking the matched rest for boilerplate prevents it.
    """
    page = PageImage(source_pdf=Path("P6-Maths-2021-SA1-Catholic-High"), page_index=11, image_bytes=b"")
    bounding_boxes = [
        _text_region("20. Express 0.9% as a fraction."),
        _text_region("Ans:"),
        _text_region("9 (Go on to the next page)"),  # combined numeral and boilerplate
        _text_region("94", region_type="page_number"),
        _text_region("www.testpapersfree.com", region_type="footer"),
    ]
    reader_output = RawReaderOutput(extracted_text="dummy", bounding_boxes=bounding_boxes)
    spans = segment_questions(page, reader_output)
    numbers = [s.question_number for s in spans]
    assert "9" not in numbers, f"phantom '9' should never be created, got {numbers!r}"
    assert numbers == ["20"]


def test_boilerplate_pattern_matches_real_confirmed_variants():
    """The _MARGIN_NOTE_PATTERNS entry matches the observed phrase variants, outside the pipeline."""
    assert _is_recognized_boilerplate("Go on to Booket B")  # OCR typo
    assert _is_recognized_boilerplate("Go on to Booklet B")  # correct spelling
    assert _is_recognized_boilerplate("(Go on to the next page)")
    assert _is_recognized_boilerplate("(Do on to the next page)")  # OCR typo in Catholic-High
                                                                     # papers ("Go" read as "Do")
    assert _is_recognized_boilerplate("{Go on to the next page}")  # curly-brace variant, also
                                                                     # Catholic-High
    assert not _is_recognized_boilerplate("Go on holiday")  # must not over-match


# =================================================================================================
# Fix 1 (extended): boilerplate phrase recognition, flat-text path
# (P6-Maths-2021-SA1-Rosyth p14 q23).
# =================================================================================================

def test_flat_text_trailing_boilerplate_stripped():
    """Trailing navigation boilerplate is stripped from the last flat-text span on a page.

    DeepSeek-OCR's flat-text span for the last question runs to the end of the page, including
    text such as "(Go on to the next page)" that the region-based path already drops. On
    P6-Maths-2021-SA1-Rosyth p14 q23 this asymmetry lowered text similarity from 0.701 (agree) to
    0.585 (disagree), a spurious disagreement.
    """
    page = PageImage(source_pdf=Path("P6-Maths-2021-SA1-Rosyth"), page_index=14, image_bytes=b"")
    text = (
        "23. Kelly bought a dress for $50 and 3 blouses at $w each, find the value of w.\n\n\n"
        "(Go on to the next page)"
    )
    reader_output = RawReaderOutput(extracted_text=text, bounding_boxes=[])
    spans = segment_questions(page, reader_output)
    assert len(spans) == 1
    assert "Go on to the next page" not in spans[0].raw_text
    assert spans[0].raw_text.endswith("find the value of w.")


def test_strip_trailing_boilerplate_lines_stops_at_first_real_content():
    """Only trailing boilerplate is stripped; earlier content, even short phrases, survives."""
    content = "A real question ending normally.\nAns:\n\nDo not write in this space"
    stripped = _strip_trailing_boilerplate_lines(content)
    assert stripped == "A real question ending normally.\nAns:"


# =================================================================================================
# Fix 3 (Issue 226): the fraction-numerator phantom, handled by a narrow cross-reader signal that
# replaces the reverted ascending-number filter. See _fraction_numerator_phantom_evidence().
# =================================================================================================

def test_fraction_numerator_phantom_suppressed_via_cross_reader_evidence(tmp_path):
    """A fraction numerator read as 'question 5' is suppressed using the other reader.

    Shape from Henry-Park p13: DeepSeek-OCR renders '\\frac{5}{7}' without its denominator, so
    the flat-text regex reads '5' as a question between questions 24 and 25. Only MinerU2.5's
    text still has the fraction, so the fix works across readers in
    `segment_and_compare_page()`; `segment_questions()` cannot resolve it from one reader alone.
    """
    raw_dir = tmp_path / "raw"
    mineru_text = (
        "24 The line graph shows the number of students who were late for school from January to "
        "May.\n"
        "\\(\\frac{5}{7}\\) of all the students who were late were girls. How many boys were late?\n"
        "Ans:\n"
        "25 Find the angle c in the figure below.\n\nAns:"
    )
    deepseek_text = (
        "24 The line graph shows the number of students who were late for school from January to "
        "May.\n\n"
        "5 of all the students who were late were girls. How many boys were late?\n\n"
        "Ans:\n\n"
        "25 Find the angle c in the figure below.\n\nAns:"
    )
    _write_raw_page(
        raw_dir, "mineru25", "henrypark", 13, mineru_text,
        bounding_boxes=[_text_region(line) for line in mineru_text.split("\n") if line.strip()],
    )
    _write_raw_page(raw_dir, "deepseek_ocr", "henrypark", 13, deepseek_text)
    results, _representative = segment_and_compare_page("henrypark", 13, "henrypark:p13", raw_dir)
    numbers = sorted(r[2]["question_id"].rsplit(":q", 1)[1] for r in results)
    assert "5" not in numbers, f"phantom '5' should be suppressed, got {numbers!r}"
    assert numbers == ["24", "25"]


def test_fraction_numerator_evidence_direct_positive_and_negative():
    """`_fraction_numerator_phantom_evidence()` fires only with both a fraction and an overlap.

    It needs a `\\frac{N}{...}` structure and substantial content overlap, not either alone.
    """
    from app.pipeline.extract import _fraction_numerator_phantom_evidence

    candidate_text = "of all the students who were late were girls. How many boys were late?"
    other_text_with_frac_and_overlap = (
        "24 The line graph...\n\\frac{5}{7}\\) of all the students who were late were girls. "
        "How many boys were late?\nAns:"
    )
    assert _fraction_numerator_phantom_evidence("5", candidate_text, other_text_with_frac_and_overlap)

    # A \frac{5}{...} with no content overlap nearby (an unrelated fraction on a busy page) must
    # not fire on the fraction match alone.
    other_text_frac_no_overlap = "\\frac{5}{8} is a completely unrelated real fraction on this page."
    assert not _fraction_numerator_phantom_evidence("5", candidate_text, other_text_frac_no_overlap)

    # Content overlap with no fraction structure must not fire. Firing here was the mistake of
    # the reverted general filter; this signal requires the fraction structure.
    other_text_overlap_no_frac = candidate_text
    assert not _fraction_numerator_phantom_evidence("5", candidate_text, other_text_overlap_no_frac)


def test_fraction_numerator_evidence_tolerates_ocr_digit_splitting():
    """A numerator split by OCR spaces ("1 4" in `\\frac {1 4}{5}`) is still read as "14".

    MinerU2.5 does this in the P6_Maths_2022_SA2_chij p13 region data.
    """
    from app.pipeline.extract import _fraction_numerator_phantom_evidence

    candidate_text = "of the remaining pupils passed the test on their first attempt this year."
    other_text = "\\frac {1 4}{5} of the remaining pupils passed the test on their first attempt this year."
    assert _fraction_numerator_phantom_evidence("14", candidate_text, other_text)


def test_fraction_numerator_evidence_rejects_genuine_question_with_internal_fraction():
    """A question containing a fraction whose numerator equals its label is not a phantom.

    False positive 1 (Issue 226): `P6-Maths-2021-SA1-Singapore-Chinese-Girls` p16, question 1
    ("Eliza ate \\frac{1}{5} of a pizza..."), mentions a fraction with numerator "1" partway
    through its sentence. Requiring overlap anywhere merged Q1 into a blank-numbered "0" span.
    The overlap must now start near the beginning of the candidate. Q1 starts "Eliza ate...",
    unlike the text after "\\frac{1}{" ("5}\\) of a pizza..."), so the result is False.
    """
    from app.pipeline.extract import _fraction_numerator_phantom_evidence

    candidate_text = (
        "Eliza ate \\(\\frac{1}{5}\\) of a pizza and 3 friends shared the remaining pizza equally. "
        "What fraction of the pizza did each of her friends receive?\nAns:"
    )
    other_reader_text = candidate_text  # the other reader captured the same question, which is
                                          # what a correct match looks like, not a phantom
    assert not _fraction_numerator_phantom_evidence("1", candidate_text, other_reader_text)


def test_fraction_numerator_evidence_rejects_coincidental_later_real_question():
    """An unrelated question that follows a fraction later on the page is not a phantom.

    False positive 2 (Issue 226), `P6_Maths_2024_SA2_nanhua` p2: `\\frac{6}{7}` is part of an MCQ
    option in Q5, and Q6 follows after closing LaTeX, an image reference and the "6" label. The
    `match.a` check passes, because Q6's text matches from its position 0. The `match.b` check
    also requires the match to start near the beginning of the window after the fraction.
    """
    from app.pipeline.extract import _fraction_numerator_phantom_evidence

    candidate_text = "Which of the following shows \\(\\frac{1}{2}\\) of the figure shaded?\n(1)\n(2)\n(3)\n(4)"
    other_reader_text = (
        "5 In the number line, what is the mixed number represented by A?\n"
        "(1) \\(2 \\frac{2}{3}\\)\n(2) \\(2 \\frac{3}{4}\\)\n(3) \\(2 \\frac{5}{6}\\)\n"
        "(4) \\(2 \\frac{6}{7}\\)\n\n![](images/0.jpg)\n\n\n"
        "6 Which of the following shows \\(\\frac{1}{2}\\) of the figure shaded?\n"
        "![](images/1.jpg)\n(1)\n(2)\n(3)\n(4)"
    )
    assert not _fraction_numerator_phantom_evidence("6", candidate_text, other_reader_text)


def test_fraction_numerator_evidence_rejects_mcq_option_fraction():
    """A fraction that is an MCQ option is never treated as a phantom source.

    False positive 3 (Issue 226), seen in four files: `P6-Maths-2021-SA1-Methodist-Girls` p3,
    `P6-Maths-2021-SA2-Henry-Park` p3, `P6_Maths_2023_WA1_Nanyang` p4 and
    `P6_Maths_2025_SA2_redswastika` p3. The last of four options is a fraction
    ("(4) `\\frac{5}{4}`"), followed directly by the next question ("5 Arrange..."). PSLE
    question numbers are small integers, so such coincidences are common, and the next question
    starts close enough to pass both position checks. A dropped-denominator phantom needs the
    fraction to be the sentence, not one item in a list, so MCQ options are excluded.
    """
    from app.pipeline.extract import _fraction_numerator_phantom_evidence

    candidate_text = "Arrange the following numbers from the smallest to the largest.\n<table>"
    other_reader_text = (
        "Find the value of \\(\\frac{4}{7} - \\frac{1}{3}\\).\n"
        "(1) \\(\\frac{3}{4}\\)\n(2) \\(\\frac{3}{21}\\)\n(3) \\(\\frac{5}{21}\\)\n"
        "(4) \\(\\frac{5}{4}\\)\n\n"
        "5 Arrange the following numbers from the smallest to the largest.\n<table>"
    )
    assert not _fraction_numerator_phantom_evidence("5", candidate_text, other_reader_text)


def test_fraction_numerator_phantom_never_fires_on_real_issue_3_reuse_case():
    """The phantom check never fires on the Issue 3 number-reuse page (Issue 226).

    P6_Maths_2022_SA2_chij p13 broke the reverted general filter. "18" labels two questions in
    different sections, and "17" is out of reading order (a two-column layout artefact). The page
    has fractions (\\frac{9}{10}, \\frac{14}{5}, \\frac{9}{8}, all in question 17), so the test
    is not trivially safe, but neither 17 nor 18 is a numerator. Both calls must return False
    against MinerU2.5's page text, so the collision the reverted filter caused cannot recur.
    """
    mineru_text = (
        "Questions 16 to 20 carry 1 mark each. Show your working clearly and write your answers "
        "in the spaces provided. For questions which require units, give your answers in the "
        "units stated. (5 marks)\n"
        "16. Write a decimal that is between 8.4 and 8.5\n"
        "Ans:\n"
        "17. Arrange the following from the greatest to the smallest.\n"
        "\\[\n\\frac {9}{1 0}, \\quad \\frac {1 4}{5}, \\quad \\frac {9}{8}, \\quad 2\n\\]\n"
        "Ans:\n"
        "18. Express \\(0.1\\%\\) as a fraction.\n"
        "Ans:"
    )
    from app.pipeline.extract import _fraction_numerator_phantom_evidence

    q17_text = "Arrange the following from the greatest to the smallest."
    q18_text = "Express 0.1% as a fraction."
    assert not _fraction_numerator_phantom_evidence("17", q17_text, mineru_text)
    assert not _fraction_numerator_phantom_evidence("18", q18_text, mineru_text)


# =================================================================================================
# Fix 2: cross-reader content recovery, label drop (Henry-Park p11, p21, p23, p25, p29, p30).
# One reader's segment_questions() returns [] or drops the number, while the other reader
# captured it.
# =================================================================================================

def test_label_drop_recovered_not_single_reader_only(tmp_path):
    """A question whose number one reader dropped is compared, not left single_reader_only.

    Shape from Henry-Park p11, Q20: MinerU2.5 drops the leading '20', so classify_page() does not
    see a question_page and segment_questions() returns []. DeepSeek-OCR captured '20'. Recovery
    finds the content in MinerU2.5's whole-page text, and the pair goes to a value comparison.
    """
    raw_dir = tmp_path / "raw"
    shared_text = (
        "In the figure, shade 4 more squares to form a symmetric figure with AB as the line of "
        "symmetry."
    )
    _write_raw_page(
        raw_dir, "mineru25", "henrypark", 11,
        f"BP-14\n{shared_text}\nDo not write in this space\n\nPage 2\n(Do on to the next page)",
    )
    _write_raw_page(raw_dir, "deepseek_ocr", "henrypark", 11, f"20 {shared_text}")
    results, _representative = segment_and_compare_page("henrypark", 11, "henrypark:p11", raw_dir)
    statuses = {r[2]["question_id"]: r[2]["agreement_status"] for r in results}
    q20_ids = [qid for qid in statuses if qid.endswith(":q20")]
    assert q20_ids, f"expected a recovered q20 record, got {list(statuses)!r}"
    assert statuses[q20_ids[0]] != "single_reader_only", (
        f"label-drop case should be recovered into a real comparison, got "
        f"{statuses[q20_ids[0]]!r}"
    )


def test_label_drop_with_no_real_cross_reader_content_stays_single_reader_only(tmp_path):
    """Recovery does not fire when the other reader's page text lacks the span's content.

    That is a true single_reader_only case, not a label drop, so the classification stays.
    """
    raw_dir = tmp_path / "raw"
    _write_raw_page(raw_dir, "mineru25", "testfile", 0, "20 Completely unrelated real question content here.")
    _write_raw_page(raw_dir, "deepseek_ocr", "testfile", 0, "Some other real page text with no overlap at all.")
    results, _representative = segment_and_compare_page("testfile", 0, "testfile:p0", raw_dir)
    assert len(results) == 1
    assert results[0][2]["agreement_status"] == "single_reader_only"


# =================================================================================================
# Fix 2: cross-reader content recovery, region merge miss (Henry-Park p8).
# =================================================================================================

def test_merge_miss_recovered_not_single_reader_only(tmp_path):
    """A question merged into the previous span by one reader is recovered for comparison.

    Shape from Henry-Park p8: MinerU2.5's region "There are 21 lamp posts..." has no leading '15',
    so Q15 is glued onto the open Q14 span. DeepSeek-OCR captured '14' and '15' separately.
    Recovery finds Q15's content in MinerU2.5's merged Q14 text for a value comparison.
    """
    raw_dir = tmp_path / "raw"
    q14_text = "After giving 3 boxes of pencils to Molly, Aaron had 45 pencils left."
    q15_text = "There are 21 lamp posts along a straight path between the first and the last post."
    _write_raw_page(
        raw_dir, "mineru25", "henrypark", 8,
        f"14 {q14_text}\n(1) 41\n(2) 56\n\n{q15_text}\n(1) 22.4 m\n(2) 29.4 m",
        bounding_boxes=[
            _text_region(f"14 {q14_text}"),
            _text_region("(1) 41"), _text_region("(2) 56"),
            _text_region(q15_text),  # no leading "15"
            _text_region("(1) 22.4 m"), _text_region("(2) 29.4 m"),
        ],
    )
    _write_raw_page(
        raw_dir, "deepseek_ocr", "henrypark", 8,
        f"14 {q14_text}\n(1) 41\n(2) 56\n\n15 {q15_text}\n(1) 22.4 m\n(2) 29.4 m",
    )
    results, _representative = segment_and_compare_page("henrypark", 8, "henrypark:p8", raw_dir)
    statuses = {r[2]["question_id"]: r[2]["agreement_status"] for r in results}
    q15_ids = [qid for qid in statuses if qid.endswith(":q15")]
    assert q15_ids, f"expected a recovered q15 record, got {list(statuses)!r}"
    assert statuses[q15_ids[0]] != "single_reader_only", (
        f"merge-miss case should be recovered into a real comparison, got "
        f"{statuses[q15_ids[0]]!r}"
    )
    # Q14 is unaffected by the recovery pass and still gets a normal comparison.
    q14_ids = [qid for qid in statuses if qid.endswith(":q14")]
    assert q14_ids and statuses[q14_ids[0]] in ("agree", "disagree")


# =================================================================================================
# Safety property: recovery must never fire on a coincidental short overlap.
# =================================================================================================

def test_recovery_does_not_fire_on_coincidental_short_overlap():
    """A short generic shared fragment ('Ans:') does not trigger recovery.

    The absolute (_RECOVERY_MIN_MATCH_CHARS) and proportional (_RECOVERY_MIN_MATCH_RATIO)
    thresholds together prevent it.
    """
    needle = "A completely different, unrelated real question of substantial real length.\nAns:"
    haystack = "Some other totally unrelated page content that happens to also end with.\nAns:"
    assert _find_recovered_span_text(needle, haystack) is None


def test_recovery_fires_on_a_real_substantial_match():
    needle = "There are 21 lamp posts along a straight path between the first and the last post."
    haystack = f"14 Some other question.\n(1) 41\n\n{needle}\n(1) 22.4 m"
    recovered = _find_recovered_span_text(needle, haystack)
    assert recovered is not None
    assert "lamp posts" in recovered
