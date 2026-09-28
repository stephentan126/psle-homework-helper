"""
Tests for the previous_page_was_grid check in segment_questions() (Issue 144).

The check stops the "6)" on acsprimary p44 becoming a fake sub-question. A full scan of the
132-file corpus found 194 pages whose classify_page() == 'question_page' verdict rests only on a
bare-number region match. Exactly one of them (P6_Maths_2024_SA2_acsprimary p44) directly follows
an answer_key_grid page, so the other 193, including every chij bare-number case (Bug 2), are
unaffected. The tests cover both shapes: acsprimary p44 (suppressed) and chij p37 (not
suppressed). previous_page_was_grid is the only thing that tells them apart.
"""
from __future__ import annotations

from pathlib import Path

from app.pipeline.extract import PageImage, RawReaderOutput, segment_questions


def _text_region(content: str) -> dict:
    return {"type": "text", "content": content}


def test_bare_number_only_page_suppressed_when_previous_page_was_grid():
    """A bare '6)' region after an answer_key_grid page does not become a question.

    Shape from acsprimary p44: the '6)' region has no question content and is followed by
    worked-solution prose.
    """
    page = PageImage(source_pdf=Path("acsprimary"), page_index=44, image_bytes=b"")
    reader_output = RawReaderOutput(
        extracted_text="6)\nStep 1: Set up the initial conditions\nLet G be the number of coins.",
        bounding_boxes=[
            _text_region("6)"),
            _text_region("Step 1: Set up the initial conditions"),
            _text_region("Let G be the number of coins."),
        ],
    )
    spans = segment_questions(page, reader_output, previous_page_was_grid=True)
    assert spans == []


def test_bare_number_only_page_still_segmented_when_previous_page_was_not_grid():
    """The same page shape not following a grid page is still segmented as a question.

    previous_page_was_grid defaults to False, so existing callers behave as before.
    """
    page = PageImage(source_pdf=Path("acsprimary"), page_index=44, image_bytes=b"")
    reader_output = RawReaderOutput(
        extracted_text="6)\nStep 1: Set up the initial conditions\nLet G be the number of coins.",
        bounding_boxes=[
            _text_region("6)"),
            _text_region("Step 1: Set up the initial conditions"),
            _text_region("Let G be the number of coins."),
        ],
    )
    spans = segment_questions(page, reader_output)  # previous_page_was_grid defaults False
    assert len(spans) >= 1


def test_chij_p37_shaped_bare_title_not_suppressed_with_real_corpus_adjacency():
    """chij p37's detached '14.' label is kept when the previous page was not a grid (Bug 2).

    The bare '14.' region has no content, and the question text sits in a separate region that
    does not start with a number. Its question_page verdict therefore comes from the same
    bare-number branch as acsprimary p44, and content alone cannot tell them apart. What protects
    chij p37 is previous_page_was_grid=False, which matches the corpus: it never follows a grid
    page. Together with the acsprimary test this covers both states seen in the corpus.
    """
    page = PageImage(source_pdf=Path("chij"), page_index=37, image_bytes=b"")
    reader_output = RawReaderOutput(
        extracted_text="14.\nBuy first air fryer at 15% discount. Find the total amount paid.",
        bounding_boxes=[
            # A bare-number region only takes part in the _has_question_like_content() scan (and
            # _is_question_page_via_bare_number_only) when its type is in
            # _BODY_CONTENT_REGION_TYPES. "text" is in that set and "title" is not. MinerU2.5
            # emitted the bare number here as a "text" region. Title-region checks such as
            # _ANSWER_KEY_STYLE_TITLE_LABEL_PATTERN are a separate, later mechanism.
            _text_region("14."),
            _text_region("Buy first air fryer at 15% discount. Find the total amount paid."),
        ],
    )
    spans = segment_questions(page, reader_output, previous_page_was_grid=False)
    assert len(spans) >= 1
