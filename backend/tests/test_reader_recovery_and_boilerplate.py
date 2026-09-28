"""
Tests for Issues 227 and 228: two narrow fixes to the Issue 224 machinery.

Both were found while breaking down the 621 `single_reader_only` records. Each fix follows the
same pattern as the earlier Issue 224 fixes: confirm the cause, make a narrow fix, validate on
the full corpus and check for regressions.

  1. **Issue 227, gaps in the cover-page and margin-boilerplate whitelists.**
     (a) `_COVER_PAGE_KEYWORDS` in `classify_page()` did not match the cover-page template of
     `P6_Maths_2022_SA2_nanchiau` ("Name / Index #", "This paper consists of N pages
     altogether."), so those two substrings were added. A third keyword, "duration for booklet",
     was reverted: `P6_Maths_2022_SA2_nanyang` p44 is a question page (OCR-corrupted) with a
     mid-page booklet-transition block, and the keyword would have discarded 11 question
     records. `test_nanyang_p44_shaped_mixed_content_page_not_misclassified_as_cover_page`
     guards against reintroducing it.
     (b) `_MARGIN_NOTE_PATTERNS` recognised only "do not write in this space". The sibling phrase
     "do not write in this column" (`P6-Maths-2021-SA2-Singapore-Chinese-Girls` p22 and three
     more pages in the same file family) is now recognised too.

  2. **Issue 228, two threshold gaps in the Issue 224 cross-reader recovery
     (`_find_recovered_span_text()`).**
     (a) The longest-contiguous-match check ran on raw text. MinerU2.5's region-based `raw_text`
     contains blank-line separators, `<table>`/`<tr>`/`<td>` markup and `![](images/N.jpg)`
     placeholders that DeepSeek-OCR's flat text does not, which broke a substantial match into
     pieces below the recovery floor. `_normalize_layout_for_recovery()` now decides whether a
     match exists; it never changes the text a successful recovery returns. It treats each
     removed artefact as a word boundary, because deleting it would fuse neighbours
     ("68" + "Ans:" -> "68Ans:") and break the match.
     (b) The absolute floor `_RECOVERY_MIN_MATCH_CHARS = 40` blocked two exact (100% containment)
     matches shorter than 40 characters. A separate floor,
     `_RECOVERY_MIN_MATCH_CHARS_FOR_FULL_CONTAINMENT = 15`, applies only to full containment. The
     40-character floor for partial matches is unchanged.
     Validation exposed a related gap: `_text_similarity()` did not normalise layout, so
     a good, longer recovery could score below a shorter one because of leftover `<table>` markup.
     `_text_similarity_layout_normalized()` is used only when at least one side is a
     `cross_reader_recovered` span. Normal pairs keep the 0.65 threshold, tuned on 78 matched
     pairs.

Full-corpus validation (132/132 files, `measure_reader_reconciliation.py`, read-only): before,
5,704 records with 621 `single_reader_only` (10.9%); after, 5,679 with 403 (7.1%). Of the 620
unique `single_reader_only` questions, 219 were resolved (200 to agree, 3 to disagree, 16 removed
as spurious pseudo-questions). 25 records were removed, all phantoms (21 nanchiau cover-page
instruction fragments, 4 "do not write in this column" margin notes), and none were added. Two
other transitions were checked and judged correct: a `disagree -> agree` that removed margin-note
contamination (`Singapore-Chinese-Girls` SA1 p10 q23), and an `agree -> single_reader_only`
(`Henry-Park` p25 q10) where the old "agree" came from matching HTML markup rather than prose.
Falling back to `single_reader_only` there is consistent with the "never a worse outcome"
contract of `_find_recovered_span_text()`. The Issue 3 case (`P6_Maths_2022_SA2_chij` p13) and
the fraction-denominator regression suite (Issue 226) are unchanged byte for byte.
"""
from __future__ import annotations

from app.pipeline.extract import (
    _find_recovered_span_text,
    _is_recognized_boilerplate,
    _normalize_layout_for_recovery,
    _text_similarity,
    _text_similarity_layout_normalized,
    classify_page,
)


# =================================================================================================
# Issue 227(a), cover-page keyword whitelist
# =================================================================================================

def test_nanchiau_shaped_cover_page_now_classified_correctly():
    """A nanchiau-style cover page is classified as `cover_page`.

    Shape from `P6_Maths_2022_SA2_nanchiau` p0, p8 and p20 (MinerU2.5). Its template ("Name /
    Index #" cell, "This paper consists of N pages altogether.") matched none of the earlier keywords.
    """
    text = (
        "Founded 1947\nNAN CHIAU PRIMARY SCHOOL\nPRELIMINARY EXAMINATION\n2022\n"
        "MATHEMATICS PAPER 1\nPRIMARY 6\nBOOKLET A\n"
        '<table><tr><td>Name / Index #</td><td></td><td>( )</td></tr>'
        "<tr><td>Class</td><td>Primary 6</td><td></td></tr></table>\n"
        "Instructions\nto students\n"
        "1. Do NOT open this bookdef until you are told to do so.\n"
        "2. Follow all Instructions carefully.\n3. Answer all questions.\n"
        "This paper consists of 5 pages altogether.\n"
        "More papers at www.testpapersfree.com"
    )
    assert classify_page(bounding_boxes=None, extracted_text=text) == "cover_page"


def test_nanyang_p44_shaped_mixed_content_page_not_misclassified_as_cover_page():
    """Guards the reverted Issue 227(a) keyword "duration for booklet".

    `P6_Maths_2022_SA2_nanyang` p44 is a question page, with questions before and after a
    mid-page booklet-transition block ("Total Duration for Booklets A and B", "Do not turn over
    this page..."). It must stay `question_page`; classing it as `cover_page` would discard its
    (OCR-corrupted) question content.
    """
    text = (
        "13 What is the length of the path shown below?\n(1) 0.3mol\n(2) 5.4cm\n"
        "NANYANG PRIMARY SCHOOL\nPRELIMINARY EXAMINATION\n2022\nPRIMARY6\nMATHEMATICS\n"
        "PAPER 1\n(BOOKLET B)\nTotal Duration for Booklets A and B: 1 hour\n"
        "INSTRUMENTS TO FUPLIS\n1. Do not turn over this page until you are told to do so.\n"
        "2. Follow all instructions carefully.\n3. Answer all questions.\n"
        "14 Vv, Wandry and Xyl each had some beads. What is the ratio of the number of beads?"
    )
    assert classify_page(bounding_boxes=None, extracted_text=text) == "question_page"


# =================================================================================================
# Issue 227(b), margin-note phrase whitelist
# =================================================================================================

def test_do_not_write_in_this_column_recognized_as_boilerplate():
    """The margin note from `P6-Maths-2021-SA2-Singapore-Chinese-Girls` p22 is boilerplate."""
    assert _is_recognized_boilerplate("Do not write in this column") is True
    assert _is_recognized_boilerplate("do not write in this column") is True  # case-insensitive


def test_do_not_write_in_this_space_still_recognized():
    """The original Issue 224 phrase must be unaffected by adding the new one."""
    assert _is_recognized_boilerplate("Do not write in this space") is True


def test_do_not_write_in_this_row_not_recognized():
    """Only the two confirmed phrases are whitelisted, not a 'do not write in this X' wildcard.

    Entries on the boilerplate list must be confirmed in the corpus.
    """
    assert _is_recognized_boilerplate("Do not write in this row") is False


# =================================================================================================
# Issue 228(a), _normalize_layout_for_recovery() itself
# =================================================================================================

def test_normalize_layout_strips_tags_and_image_placeholders():
    text = "Find the value.<table><tr><td>68</td></tr></table>![](images/0.jpg)Ans: 5"
    normalized, index_map = _normalize_layout_for_recovery(text)
    assert "<table>" not in normalized and "<td>" not in normalized and "images/0.jpg" not in normalized
    assert "68" in normalized and "Ans: 5" in normalized
    assert len(index_map) == len(normalized)


def test_normalize_layout_treats_artifact_as_word_boundary_not_deletion():
    """Removed markup becomes one space, so "68" and "Ans:" do not fuse (PSLE Math 2019 p9 q16).

    The raw HTML has `</td><td></td></tr><tr><td></td><td>` between them and no whitespace.
    Deleting it outright gives "68Ans:" and breaks the match.
    """
    text = "68</td><td></td></tr><tr><td></td><td>Ans:"
    normalized, _ = _normalize_layout_for_recovery(text)
    assert normalized == "68 Ans:"


def test_normalize_layout_index_map_points_back_to_real_original_positions():
    text = "abc<table>def"
    normalized, index_map = _normalize_layout_for_recovery(text)
    assert normalized == "abc def"
    # Each mapped index must hold the character it represents, or be the artefact's start
    # position for a synthesised word-boundary space.
    for i, ch in enumerate(normalized):
        if ch != " " or text[index_map[i]] == " ":
            assert text[index_map[i]] == ch


# =================================================================================================
# Issue 228(a), _find_recovered_span_text() finds matches fragmented by layout
# =================================================================================================

def test_recovery_finds_real_match_fragmented_by_table_markup():
    """A match broken up by table markup is recovered (`P6-Maths-2021-SA1-Methodist-Girls` p22).

    Raw-text comparison reduced it to a partial hit of about 33%, below the recovery floor.
    """
    needle = "In the diagram below, CDE and FGH are straight lines. DG = GE.\n\n\n(a) Find \\(\\angle\\) DCH.\n\n\n(b) Find \\(\\angle\\) EGF\n\n![](images/0.jpg)\n\n\nAns:"
    haystack = (
        "Do not write in this space\n\n![](images/1.jpg)\n"
        "In the diagram below, CDE and FGH are straight lines. DG = GE.\n"
        "(a) Find \\(\\angle DCH\\).\n(b) Find \\(\\angle EGF\\)\n<table><tr><td>Ans:</td></tr></table>"
    )
    recovered = _find_recovered_span_text(needle, haystack)
    assert recovered is not None
    assert "CDE and FGH are straight lines" in recovered


def test_recovery_finds_real_match_fragmented_by_missing_whitespace_between_cells():
    """A match spanning two table cells with no whitespace between tags is recovered.

    Shape from `PSLE Math 2019` p9 q16. This relies on the word-boundary handling above, not
    only on tag stripping.
    """
    needle = "Find the value of 1056 - 68  \n\n\nAns:"
    haystack = (
        "Do not write in this space</td></tr><tr><td>16</td><td>Find the value of 1056 - 68"
        "</td><td></td></tr><tr><td></td><td>Ans:</td></tr><tr><td>17</td>"
    )
    recovered = _find_recovered_span_text(needle, haystack)
    assert recovered is not None
    assert "1056 - 68" in recovered


# =================================================================================================
# Issue 228(b), the separate, lower full-containment floor
# =================================================================================================

def test_recovery_full_containment_below_old_floor_now_recovers():
    """An 18-character fully contained needle is recovered (`P6_Maths_2022_SA2_nanhua` p45 q23).

    The flat 40-character floor used to block it despite the 100% ratio.
    """
    needle = "a) North-West b) C"
    haystack = "22 a) 4 9 9 2 5 b) 8 3 9\n23. a) North-West\nb) C\n24. some other real question content here"
    recovered = _find_recovered_span_text(needle, haystack)
    assert recovered is not None


def test_recovery_still_rejects_trivial_full_matches_below_the_new_floor():
    """Trivial short strings stay rejected even at a perfect ratio.

    "Ans:" (4 characters) and "(a)" (3 characters) are likely coincidental, which is the same
    reason the 40-character floor exists for partial matches.
    """
    assert _find_recovered_span_text("Ans:", "blah blah blah Ans: more filler text padding here") is None
    assert _find_recovered_span_text("(a)", "some question (a) find the value of something") is None


def test_recovery_partial_match_still_needs_the_original_40_char_floor():
    """The 40-character floor still applies to partial matches.

    Full containment has a separate path; the general floor is not lowered.
    """
    needle = "A completely different, unrelated real question of substantial real length that is long."
    haystack = "Some other totally unrelated page content that happens to share only a little bit."
    assert _find_recovered_span_text(needle, haystack) is None


# =================================================================================================
# Issue 228(a), similarity-scoring gap, _text_similarity_layout_normalized()
# =================================================================================================

def test_layout_normalized_similarity_scores_higher_than_plain_for_markup_polluted_recovery():
    """Layout-normalised similarity scores a markup-polluted recovery above 0.65.

    Shape from `P6_Maths_2024_SA2_methodistgirls` p8 q17. `_find_recovered_span_text()` returns
    a verbatim slice of the haystack, markup included, by design. Plain `_text_similarity()`
    scores this below 0.65 only because of that markup.
    """
    needle = "Express \\(6 \\div 7\\) as a decimal correct to 2 decimal places.  \n\n\nAns:"
    recovered = " as a decimal correct to 2 decimal places.</td><td></td></tr><tr><td></td><td>Ans:"
    plain = _text_similarity(needle, recovered)
    normalized = _text_similarity_layout_normalized(needle, recovered)
    assert normalized > plain
    assert normalized >= 0.65 > plain


def test_plain_text_similarity_unchanged_for_a_normal_non_recovered_pair():
    """The normal comparison path is unchanged.

    _text_similarity() is not modified; the layout-normalised version is a separate function.
    """
    a = "Find the value of 8.09 times 7."
    b = "Find the value of 8.09 times 7."
    assert _text_similarity(a, b) == 1.0
