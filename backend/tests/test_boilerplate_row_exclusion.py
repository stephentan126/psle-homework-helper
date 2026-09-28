"""
Tests for excluding cover-page boilerplate rows from matching and hints (Issue 209).

Some corpus rows are exam cover-page boilerplate extracted as if it were a question, sometimes with
an unrelated `answer_value` attached. These tests check both enforcement points,
`find_best_match()` in `question_matching.py` and `request_hint()` in `routes.py`. They also check
that unflagged questions are unaffected. About 98% of the corpus has a NULL `extraction_flag`, so a
NULL-handling bug in this filter would silently break match-first for nearly the whole corpus. That
is tested directly rather than assumed from the SQL.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.db.session import get_session
from app.main import app
from app.models.question import NON_QUESTION_EXTRACTION_FLAG, Question
from app.models.question_embedding import QuestionEmbedding
from app.pipeline.question_matching import EMBEDDING_MODEL_ID, _get_model, find_best_match


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _insert_question_with_embedding(text: str, extraction_flag: str | None, answer_value: str) -> int:
    with get_session() as session:
        question = Question(
            source_paper="data/School papers/2023/P6_Maths_2023_WA2_AiTong.pdf",
            source_page_index=1, source_page_label=None,
            answer_source_file="data/School papers/2023/P6_Maths_2023_WA2_AiTong.pdf",
            answer_source_page_index=1, question_number="1",
            question_text=text, answer_value=answer_value, worked_solution_text=None,
            has_diagram=False, ocr_confidence="high", extraction_flag=extraction_flag,
        )
        session.add(question)
        session.flush()
        question_id = question.id

        model = _get_model()
        embedding = model.encode([text], convert_to_numpy=True)[0]
        session.add(QuestionEmbedding(
            question_id=question_id, embedding_model=EMBEDDING_MODEL_ID,
            embedding_json=json.dumps(embedding.tolist()), computed_at=_now_iso(),
        ))
        return question_id


_BOILERPLATE_TEXT = (
    "The use of an approved calculator is allowed. Parent's signature: Paper 1 / 45 Paper 2 / 55 "
    "Total / 100"
)
_ORDINARY_TEXT = "A shop sold 156 apples on Monday and 89 more apples on Tuesday. How many apples were sold in total over the two days?"


def test_find_best_match_excludes_a_row_flagged_non_question_boilerplate():
    """A flagged boilerplate row is never returned, even when queried with its exact text.

    The exact text scores about 1.0, so this shows the row is removed from the candidate pool
    rather than merely falling below the threshold.
    """
    _insert_question_with_embedding(_BOILERPLATE_TEXT, NON_QUESTION_EXTRACTION_FLAG, "36")
    result = find_best_match(_BOILERPLATE_TEXT)
    assert result.matched_question_id is None
    assert result.match_method == f"no_embeddings_cached_for_{EMBEDDING_MODEL_ID}"


def test_find_best_match_excludes_a_row_with_a_combined_flag_string():
    """A row whose flag string combines the boilerplate flag with another flag is still excluded.

    The corpus separates multiple flags with "; ", so the filter cannot rely on an exact string
    match.
    """
    combined_flag = f"no_matching_answer_key_row; {NON_QUESTION_EXTRACTION_FLAG}"
    _insert_question_with_embedding(_BOILERPLATE_TEXT, combined_flag, "36")
    result = find_best_match(_BOILERPLATE_TEXT)
    assert result.matched_question_id is None


def test_find_best_match_still_matches_an_ordinary_question_with_a_null_extraction_flag():
    """An unflagged question (extraction_flag=None) still matches its own text.

    About 98% of the corpus has no flag. In SQL, `!=` against NULL yields NULL, not TRUE, so a
    careless filter would exclude nearly the whole corpus from matching.
    """
    qid = _insert_question_with_embedding(_ORDINARY_TEXT, None, "245")
    result = find_best_match(_ORDINARY_TEXT)
    assert result.matched_question_id == qid
    assert result.match_confidence >= 0.90


def test_find_best_match_still_matches_a_question_with_an_unrelated_real_extraction_flag():
    """A row with an unrelated extraction_flag still matches.

    The exclusion targets only the boilerplate marker, not any flag. Flags such as
    "no_matching_answer_key_row" are carried by 1,487 corpus rows (about 27%).
    """
    qid = _insert_question_with_embedding(_ORDINARY_TEXT, "no_matching_answer_key_row", "245")
    result = find_best_match(_ORDINARY_TEXT)
    assert result.matched_question_id == qid


def test_request_hint_rejects_a_flagged_non_question_directly():
    """`request_hint()` rejects a flagged row requested directly by `question_id`.

    The exclusion in `find_best_match()` does not cover a direct lookup, so this is the second
    enforcement point.
    """
    qid = _insert_question_with_embedding(_BOILERPLATE_TEXT, NON_QUESTION_EXTRACTION_FLAG, "36")
    with TestClient(app) as client:
        response = client.post(
            f"/api/questions/{qid}/hint", params={"request_id": "issue-209-test-1"},
        )
    assert response.status_code == 422
    assert "non-question extraction content" in response.json()["detail"]


def test_request_hint_rejects_a_combined_flag_question_directly_too():
    combined_flag = f"partial_subpart_answer_match; {NON_QUESTION_EXTRACTION_FLAG}"
    qid = _insert_question_with_embedding(_BOILERPLATE_TEXT, combined_flag, "36")
    with TestClient(app) as client:
        response = client.post(
            f"/api/questions/{qid}/hint", params={"request_id": "issue-209-test-2"},
        )
    assert response.status_code == 422
    assert "non-question extraction content" in response.json()["detail"]


def test_request_hint_still_serves_an_ordinary_unflagged_question():
    """`request_hint()` still serves an ordinary question with no extraction_flag."""
    qid = _insert_question_with_embedding(_ORDINARY_TEXT, None, "245")
    with TestClient(app) as client:
        response = client.post(
            f"/api/questions/{qid}/hint", params={"request_id": "issue-209-test-3"},
        )
    assert response.status_code == 200
    assert response.json()["state"] in ("hint", "capped_tier1")
