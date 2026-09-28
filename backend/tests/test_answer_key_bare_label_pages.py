"""
Tests for answer-key pages whose leading label is a plain `text` region (Issue 372).

Issue 371 traced `id=3265` to a worked-solution page that was not recognised as an answer key.
`_ANSWER_KEY_STYLE_TITLE_LABEL_PATTERN` only checks `title` regions, but a corpus-wide survey found
the same page shape with the label typed as plain `text` in 10 of 12 cases. The labels are not
always "Q"-prefixed either: bare "9", lettered "13a)" and dotted "19." all occur.

`_has_narrative_worked_solution_shape()` handles these. A label alone is not enough, because chij
p37 has a bare "14." `text` region that starts a question. The check only fires when the rest of
the page narrates two or more numbered derivation steps and ends in an "Answer:" or "So,"
conclusion line.

The positive fixtures are taken from corpus reader output (region type and content pairs), and
test names note the paper and page each one comes from.
"""
from __future__ import annotations

from app.pipeline.extract import (
    PageImage,
    RawReaderOutput,
    _has_narrative_worked_solution_shape,
    classify_page,
    segment_questions,
)


# =================================================================================================
# _has_narrative_worked_solution_shape(): unit-level gating logic
# =================================================================================================

def test_label_alone_with_no_narrative_steps_is_false():
    """A label followed by ordinary, non-narrative content does not trigger.

    The label shape alone is not a safe signal (the false-positive risk noted in Issue 144).
    """
    boxes = [
        {"type": "text", "content": "14."},
        {"type": "text", "content": "Some girls and boys took part in a race."},
        {"type": "text", "content": "Ans: [4]"},
    ]
    assert _has_narrative_worked_solution_shape(boxes) is False


def test_narrative_steps_with_no_matching_label_is_false():
    """Numbered steps and an answer line do not trigger if the first region is not a label."""
    boxes = [
        {"type": "text", "content": "A long real question stem with no leading label at all."},
        {"type": "text", "content": "1. Calculate the first thing:"},
        {"type": "text", "content": "2. Calculate the second thing:"},
        {"type": "text", "content": "Answer: 42"},
    ]
    assert _has_narrative_worked_solution_shape(boxes) is False


def test_label_plus_only_one_narrative_step_is_false():
    """A single numbered step does not trigger.

    All 12 corpus matches have at least two steps, so the threshold is two.
    """
    boxes = [
        {"type": "text", "content": "9"},
        {"type": "text", "content": "1. Calculate the first thing:"},
        {"type": "text", "content": "Answer: 42"},
    ]
    assert _has_narrative_worked_solution_shape(boxes) is False


def test_ordinary_instruction_text_containing_the_word_answers_does_not_false_trigger():
    """Instruction boilerplate containing "answers" is not a conclusion line.

    Standard PSLE boilerplate ("Write your answers in the spaces provided", as on
    `P6_Maths_2024_SA2_nanyang.pdf` p44) contains the substring "answer". A substring test would
    fire on it, so the check uses `.startswith("answer")`.
    """
    boxes = [
        {"type": "text", "content": "10"},
        {"type": "text", "content": "1) A real MCQ sub-option, not a narrative step."},
        {"type": "text", "content": "2) Another real MCQ sub-option."},
        {
            "type": "text",
            "content": "Questions 16 to 20 carry 1 mark each. Write your answers in the spaces "
                       "provided.",
        },
    ]
    assert _has_narrative_worked_solution_shape(boxes) is False


def test_real_label_plus_narrative_plus_answer_line_is_true():
    boxes = [
        {"type": "text", "content": "9"},
        {"type": "text", "content": "1. Calculate the Total Distance Traveled:"},
        {"type": "text", "content": "2. Combine Distances:"},
        {"type": "text", "content": "Answer: The distance between City A and City B is 420 km."},
    ]
    assert _has_narrative_worked_solution_shape(boxes) is True


def test_real_conclusion_line_may_also_start_with_so_comma():
    boxes = [
        {"type": "text", "content": "13a)"},
        {"type": "text", "content": "1. Calculate Half of the Side Length:"},
        {"type": "text", "content": "2. Calculate the Area of One Quarter Circle:"},
        {"type": "text", "content": "3. Calculate the Area of the Square:"},
        {"type": "text", "content": "So, the area of the square is 12.5 cm2."},
    ]
    assert _has_narrative_worked_solution_shape(boxes) is True


def test_header_footer_page_number_furniture_is_skipped_when_finding_the_label():
    """The label is looked for among body regions only.

    A header, footer or page number earlier in the list must not hide it.
    """
    boxes = [
        {"type": "header", "content": "BP-401"},
        {"type": "text", "content": "13a)"},
        {"type": "text", "content": "1. Calculate Half of the Side Length:"},
        {"type": "text", "content": "2. Calculate the Area of One Quarter Circle:"},
        {"type": "text", "content": "Answer: 12.5 cm2"},
        {"type": "page_number", "content": "Pg16"},
    ]
    assert _has_narrative_worked_solution_shape(boxes) is True


# =================================================================================================
# classify_page() end to end: corpus content, positive cases
# =================================================================================================

def test_id3265_real_page_now_classifies_as_answer_key_grid():
    """The source page of `id=3265` (`P6_Maths_2024_SA2_acsprimary.pdf` p54) is an answer key.

    Its label "13a)" is typed `text`, not `title`, and has no "Q" prefix, so the title-label
    pattern alone missed it.
    """
    boxes = [
        {"type": "header", "content": "BP~398"},
        {"type": "text", "content": "13a)"},
        {"type": "text", "content": "1. Calculate Half of the Side Length:"},
        {"type": "text", "content": "- Side length of the square is equal to the diameter of the "
                                     "small quarter circles."},
        {"type": "text", "content": "The radius of the small quarter circles is 5cm, so:"},
        {"type": "text", "content": "Half of the side length = 5/2 = 2.5 cm"},
        {"type": "text", "content": "2. Calculate the Area of One Quarter Circle:"},
        {"type": "text", "content": "Area of one small quarter circle:"},
        {"type": "equation", "content": "Area = 1/2 x 2.5 x 2.5 = 3.125 cm2"},
        {"type": "text", "content": "3. Calculate the Area of the Square:"},
        {"type": "text", "content": "Since there are four quarter circles making up the square:"},
        {"type": "text", "content": "Area of the Square = 3.125 x 4 = 12.5 cm2"},
        {"type": "text", "content": "Answer: 12.5 cm2"},
        {"type": "page_number", "content": "Pg16"},
    ]
    text = "\n".join(b["content"] for b in boxes)
    assert classify_page(bounding_boxes=boxes, extracted_text=text) == "answer_key_grid"


def test_id3265_real_page_now_produces_zero_question_spans():
    """The same page produces no `QuestionSpan`s, which is the downstream effect that matters."""
    boxes = [
        {"type": "header", "content": "BP~398"},
        {"type": "text", "content": "13a)"},
        {"type": "text", "content": "1. Calculate Half of the Side Length:"},
        {"type": "text", "content": "2. Calculate the Area of One Quarter Circle:"},
        {"type": "text", "content": "3. Calculate the Area of the Square:"},
        {"type": "text", "content": "Answer: 12.5 cm2"},
    ]
    text = "\n".join(b["content"] for b in boxes)
    page = PageImage(source_pdf=__import__("pathlib").Path("x"), page_index=54, image_bytes=b"")
    reader_output = RawReaderOutput(
        extracted_text=text, structure=None, bounding_boxes=boxes, field_confidence=None,
    )
    spans = segment_questions(page, reader_output, None, None, False)
    assert spans == []


def test_real_page47_bare_undecorated_number_label_now_classifies_as_answer_key_grid():
    """A bare "9" label with no punctuation is recognised (`P6_Maths_2024_SA2_acsprimary.pdf` p47).

    This is the loosest label shape in the survey, on a different question from the p54 case, so
    it shows the check is not tuned to one page's label.
    """
    boxes = [
        {"type": "header", "content": "BP~392"},
        {"type": "text", "content": "9"},
        {"type": "text", "content": "1. Calculate the Total Distance Traveled:"},
        {"type": "text", "content": "2. Combine Distances:"},
        {"type": "text", "content": "3. Find Time Relation:"},
        {"type": "text", "content": "4. Calculate:"},
        {"type": "text", "content": "Answer: The distance between City A and City B is 420 km."},
        {"type": "page_number", "content": "Pg10"},
    ]
    text = "\n".join(b["content"] for b in boxes)
    assert classify_page(bounding_boxes=boxes, extracted_text=text) == "answer_key_grid"


def test_real_page55_letter_suffixed_dotted_label_now_classifies_as_answer_key_grid():
    """The label "13b)" on `P6_Maths_2024_SA2_acsprimary.pdf` p55 is recognised.

    It is the letter-suffixed sibling of the p54 case and a distinct label format in the survey.
    """
    boxes = [
        {"type": "header", "content": "BP~399"},
        {"type": "text", "content": "13b)"},
        {"type": "text", "content": "1. Calculate the Area of the Big Quarter Circle:"},
        {"type": "text", "content": "2. Calculate the Area of One Small Quarter Circle:"},
        {"type": "text", "content": "3. Subtract the Area of One Small Quarter Circle from the "
                                     "Big Quarter Circle:"},
        {"type": "text", "content": "Answer: 53.5 cm2"},
        {"type": "page_number", "content": "Pg17"},
    ]
    text = "\n".join(b["content"] for b in boxes)
    assert classify_page(bounding_boxes=boxes, extracted_text=text) == "answer_key_grid"


# =================================================================================================
# classify_page() end to end: false-positive regression guards
# =================================================================================================

def test_real_chij_p37_bare_number_dot_label_stays_question_page_not_answer_key_grid():
    """A bare "14." label that starts a question stays `question_page`.

    On `P6_Maths_2022_SA2_chij.pdf` (and `P6_Maths_2025_SA2_chij.pdf`) p37, the "14." `text`
    region starts an air-fryer discount word problem. The case is also documented in the
    `_ANSWER_KEY_STYLE_TITLE_LABEL_PATTERN` docstring.
    """
    boxes = [
        {"type": "header", "content": "BP~80"},
        {"type": "text", "content": "14."},
        {"type": "title", "content": "Membership Promotion!"},
        {"type": "image", "content": ""},
        {"type": "image_caption", "content": "Buy first air fryer at 15% discount"},
        {"type": "image_caption", "content": "Buy second air fryer at 30% discount"},
        {"type": "text", "content": "For non-members, enjoy a 10% discount for each air fryer."},
        {"type": "text", "content": "Mrs Wong paid $341 for two air fryers by using the "
                                     "membership promotion shown above. How much would "},
        {"type": "text", "content": "Ans: [4]"},
        {"type": "aside_text", "content": "Do not write in this space"},
        {"type": "page_number", "content": "14"},
        {"type": "footer", "content": "More papers at www.testpapersfree.com"},
    ]
    text = "\n".join(b["content"] for b in boxes)
    assert classify_page(bounding_boxes=boxes, extracted_text=text) == "question_page"


def test_real_nanyang_p44_instruction_boilerplate_does_not_false_trigger():
    """An MCQ page whose boilerplate contains "answers" is not classified as `answer_key_grid`.

    `P6_Maths_2024_SA2_nanyang.pdf` p44 (label "10") is the case the `.startswith()` check exists
    to reject. Which other classification it gets is outside the scope of this test.
    """
    boxes = [
        {"type": "header", "content": "BP~697"},
        {"type": "text", "content": "10"},
        {"type": "text", "content": "Which of the four"},
        {"type": "text", "content": "1) A real MCQ sub-option glued together oddly by OCR"},
        {"type": "text", "content": "2) Make the au"},
        {"type": "text", "content": "Questions 16 to 20 carry 1 mark each. Write your answers "
                                     "in the spaces provided."},
    ]
    text = "\n".join(b["content"] for b in boxes)
    assert classify_page(bounding_boxes=boxes, extracted_text=text) != "answer_key_grid"


def test_acsprimary_p43_title_region_q_prefixed_case_still_works_unchanged():
    """The Issue 144 title-region case still classifies as `answer_key_grid`.

    `P6_Maths_2024_SA2_acsprimary.pdf` p43 has a `title` label "Q5)". The new check is additive
    and must not change the existing title-region path.
    """
    text = "Q5)\n1. A real, long worked-solution paragraph with no <table> markup anywhere on this page at all."
    boxes = [{"type": "title", "content": "Q5)"}]
    assert classify_page(bounding_boxes=boxes, extracted_text=text) == "answer_key_grid"
