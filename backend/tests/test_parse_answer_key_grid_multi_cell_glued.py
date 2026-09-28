"""
Tests for multi-cell glued rows in parse_answer_key_grid() (Issue 154).

In this shape every cell in a row carries a complete glued "label. value" pair (Tao-Nan p32,
Booklet A). This differs from the shape in Issues 142, 150 and 153, where exactly one cell holds
a glued pair and the rest are blank or colspan padding.

Also covers a false positive seen on `acsprimary` p42 during full-corpus validation: a row
already split into (label, content), whose content starts with a numbered worked-solution step
("1. Assume..."), must not be treated as this shape.
"""
from __future__ import annotations

from app.pipeline.extract import parse_answer_key_grid


def _table_region(html: str) -> dict:
    return {"type": "table", "content": html}


def test_multi_cell_glued_row_recovers_every_real_answer():
    """Tao-Nan p32's Booklet A row shape, where every cell is glued, yields every answer."""
    html = "<table><tr><td>Q11. 2</td><td>Q12. 2</td><td>Q13. 1</td><td>Q14. 3</td><td>Q15. 3</td></tr></table>"
    rows = parse_answer_key_grid([_table_region(html)], page_index=32)
    by_number = {r.question_number: r.answer_text for r in rows}
    assert by_number == {"11": "2", "12": "2", "13": "1", "14": "3", "15": "3"}


def test_multi_cell_glued_row_preserves_column_index_order():
    html = "<table><tr><td>Q1. 4</td><td>Q2. 2</td><td>Q3. 4</td></tr></table>"
    rows = parse_answer_key_grid([_table_region(html)], page_index=32)
    assert [r.column_index for r in rows] == [0, 1, 2]
    assert [r.question_number for r in rows] == ["1", "2", "3"]


def test_already_split_label_and_worked_solution_content_not_treated_as_multi_cell_glued():
    """A bare label cell plus a content cell starting "1. ..." is not split as glued cells.

    Seen on acsprimary p42: the label cell ("Q2)") has no content, and the content cell opens
    with a numbered step. Treating them as two glued cells would destroy the answer, so the full
    worked-solution text must survive intact.
    """
    html = (
        "<table><tr><td>Q2)</td><td>1. Assume the side length of the garden is 5 meters.\n"
        "2. The total side length including the path is 5 + 4 = 9 meters.</td></tr></table>"
    )
    rows = parse_answer_key_grid([_table_region(html)], page_index=42)
    assert len(rows) == 1
    assert rows[0].question_number == "2"
    assert rows[0].answer_text.startswith("1. Assume the side length")
    assert "2. The total side length" in rows[0].answer_text


def test_row_with_one_blank_real_cell_still_uses_the_issue_153_path_not_154():
    """A row with a blank cell still uses the Issue 153 path, not the multi-cell branch.

    A blank cell cannot satisfy the glued pattern's non-empty content requirement.
    """
    html = "<table><tr><td>Q16) 967</td><td></td></tr></table>"
    rows = parse_answer_key_grid([_table_region(html)], page_index=34)
    assert len(rows) == 1
    assert rows[0].question_number == "16"
    assert rows[0].answer_text == "967"
