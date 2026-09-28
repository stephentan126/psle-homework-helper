"""
Tests for the Job A dedup pass (Issue 231).

Covers the tie-break scoring rule, the threshold decision boundary and clustering for groups of
three or more. Above all, it checks the data-safety requirement: setting `Question.superseded_by`
is the only change `run_dedup_pass(apply=True)` makes. No row is deleted and no
`OCRExtractionRecord` is modified.
"""
from __future__ import annotations

import inspect
from datetime import datetime, timezone

import pytest

from app.db.session import get_session
from app.models.ocr_extraction_record import OCRExtractionRecord
from app.models.question import Question
from app.models.question_embedding import QuestionEmbedding
from app.pipeline import dedup
from app.pipeline.dedup import (
    AUTO_MARK_THRESHOLD,
    REVIEW_FLOOR,
    CandidatePair,
    _numeric_multiset_conflict,
    build_clusters,
    compute_full_similarity_matrix,
    find_candidate_pairs,
    load_corpus_embedding_matrix,
    run_dedup_pass,
    score_question_for_survivorship,
)

import numpy as np


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_question(**overrides) -> Question:
    defaults = dict(
        source_paper="fake.pdf", source_page_index=1, source_page_label="p1",
        answer_source_file="fake.pdf", answer_source_page_index=1,
        ocr_confidence="high", school_name="fakeschool", school_name_verified=False,
        verification_status="unverified", verified_at=None, superseded_by=None,
        extraction_flag=None, question_section=None, question_number="1",
        question_text="A real, sufficiently long fake question text for a test fixture.",
        answer_value="42", worked_solution_text=None, has_diagram=False,
    )
    defaults.update(overrides)
    return Question(**defaults)


def _add_embedding(session, question_id: int, vector) -> None:
    import json
    session.add(QuestionEmbedding(
        question_id=question_id, embedding_model=dedup.EMBEDDING_MODEL_ID,
        embedding_json=json.dumps(list(vector)), computed_at=_now(),
    ))


# =================================================================================================
# Tie-break scoring rule: tests of score_question_for_survivorship()
# =================================================================================================

def test_worked_solution_text_presence_is_the_top_tier_signal():
    with_ws = _make_question(id=1, worked_solution_text="1+1=2")
    without_ws = _make_question(id=2, worked_solution_text=None)
    cluster = [with_ws, without_ws]
    assert score_question_for_survivorship(with_ws, cluster) > \
        score_question_for_survivorship(without_ws, cluster)


def test_agreement_status_outranks_lower_tiers_even_without_worked_solution():
    """Agreement status ranks below worked_solution_text but orders records that tie on it.

    An 'agree' record without worked_solution_text loses to one with it (tier 1 beats tier 2).
    Among records that tie on tier 1, 'agree' beats 'disagree', which beats 'single_reader_only'.
    """
    agree_q = _make_question(id=1, worked_solution_text="x", extraction_flag=None)
    disagree_q = _make_question(id=2, worked_solution_text="x", extraction_flag=None)
    single_q = _make_question(id=3, worked_solution_text="x", extraction_flag=None)
    agree_q.ocr_records = [OCRExtractionRecord(
        question_id=1, agreement_status="agree", extracted_at=_now(),
    )]
    disagree_q.ocr_records = [OCRExtractionRecord(
        question_id=2, agreement_status="disagree", extracted_at=_now(),
    )]
    single_q.ocr_records = [OCRExtractionRecord(
        question_id=3, agreement_status="single_reader_only", extracted_at=_now(),
    )]
    cluster = [agree_q, disagree_q, single_q]
    scores = {q.id: score_question_for_survivorship(q, cluster) for q in cluster}
    assert scores[1] > scores[2] > scores[3]

    # Tier 1 (worked_solution_text) dominates tier 2 (agreement): a worse agreement status with
    # worked_solution_text still beats a better one without it.
    single_with_ws = _make_question(id=4, worked_solution_text="x")
    single_with_ws.ocr_records = [OCRExtractionRecord(
        question_id=4, agreement_status="single_reader_only", extracted_at=_now(),
    )]
    agree_without_ws = _make_question(id=5, worked_solution_text=None)
    agree_without_ws.ocr_records = [OCRExtractionRecord(
        question_id=5, agreement_status="agree", extracted_at=_now(),
    )]
    pair_cluster = [single_with_ws, agree_without_ws]
    assert score_question_for_survivorship(single_with_ws, pair_cluster) > \
        score_question_for_survivorship(agree_without_ws, pair_cluster)


def test_has_diagram_scored_against_cluster_majority_not_an_absolute_preference():
    """has_diagram is scored by agreement with the cluster majority, not by its value.

    In a 3-member cluster where 2 say has_diagram=True and 1 says False, the two in the majority
    score higher on that tier. In a cluster with a False majority, False is rewarded instead.
    """
    a = _make_question(id=1, has_diagram=True)
    b = _make_question(id=2, has_diagram=True)
    c = _make_question(id=3, has_diagram=False)
    cluster = [a, b, c]
    # Isolate the has_diagram tier by keeping every other tier identical across all three.
    score_a = score_question_for_survivorship(a, cluster)
    score_c = score_question_for_survivorship(c, cluster)
    assert score_a > score_c  # a agrees with the 2-of-3 majority (True), c does not

    # Inverted cluster: the majority is now False, so the outlier is the one that disagrees.
    d = _make_question(id=4, has_diagram=False)
    e = _make_question(id=5, has_diagram=False)
    f = _make_question(id=6, has_diagram=True)
    inverted_cluster = [d, e, f]
    assert score_question_for_survivorship(d, inverted_cluster) > \
        score_question_for_survivorship(f, inverted_cluster)


def test_has_diagram_even_split_contributes_no_signal():
    a = _make_question(id=1, has_diagram=True)
    b = _make_question(id=2, has_diagram=False)
    cluster = [a, b]
    # Compare only the has_diagram tier (index 2). The full tuples differ at the final -id
    # tiebreak (id=1 vs id=2), which is expected and not under test here.
    score_a = score_question_for_survivorship(a, cluster)
    score_b = score_question_for_survivorship(b, cluster)
    assert score_a[2] == score_b[2] == 0


def test_has_diagram_even_split_does_not_affect_final_survivorship_when_it_is_the_only_difference():
    """With an even has_diagram split as the only difference, the lower id wins.

    The decision falls through to the deterministic id tiebreak rather than the has_diagram value.
    """
    higher_id_true = _make_question(id=9, has_diagram=True)
    lower_id_false = _make_question(id=3, has_diagram=False)
    cluster = [higher_id_true, lower_id_false]
    assert score_question_for_survivorship(lower_id_false, cluster) > \
        score_question_for_survivorship(higher_id_true, cluster)


def test_extraction_flag_none_beats_a_flagged_row_when_earlier_tiers_tie():
    clean = _make_question(id=1, extraction_flag=None)
    flagged = _make_question(id=2, extraction_flag="no_matching_answer_key_row")
    cluster = [clean, flagged]
    assert score_question_for_survivorship(clean, cluster) > \
        score_question_for_survivorship(flagged, cluster)


def test_verified_metadata_breaks_ties_below_the_completeness_tiers():
    verified = _make_question(id=1, school_name_verified=True, verification_status="verified")
    unverified = _make_question(id=2, school_name_verified=False, verification_status="unverified")
    cluster = [verified, unverified]
    assert score_question_for_survivorship(verified, cluster) > \
        score_question_for_survivorship(unverified, cluster)


def test_id_is_only_a_last_resort_tiebreak_not_insertion_order_as_a_real_signal():
    """A higher id (inserted later) can win when its data is better.

    The id decides only when every other tier ties exactly.
    """
    later_but_better = _make_question(id=99, worked_solution_text="x")
    earlier_but_worse = _make_question(id=1, worked_solution_text=None)
    cluster = [later_but_better, earlier_but_worse]
    assert score_question_for_survivorship(later_but_better, cluster) > \
        score_question_for_survivorship(earlier_but_worse, cluster)

    # Only when every tier ties does id decide. The lower id wins, an arbitrary but
    # deterministic tiebreak documented in the dedup.py docstring.
    tie_a = _make_question(id=5)
    tie_b = _make_question(id=2)
    cluster2 = [tie_a, tie_b]
    assert score_question_for_survivorship(tie_b, cluster2) > \
        score_question_for_survivorship(tie_a, cluster2)


# =================================================================================================
# The dedup-specific third secondary signal (Issue 231), which threshold calibration showed to
# be necessary.
# =================================================================================================

_MUFFINS_741 = (
    "<table><tr><td>Muffins</td><td>Number Sold</td></tr><tr><td>A</td><td>80</td></tr>"
    "<tr><td>B</td><td>100</td></tr><tr><td>C</td><td>60</td></tr><tr><td>D</td><td>100</td></tr></table>"
)
_MUFFINS_742 = (
    "<table><tr><td>Muffins</td><td>Number Sold</td></tr><tr><td>A</td><td>60</td></tr>"
    "<tr><td>B</td><td>80</td></tr><tr><td>C</td><td>100</td></tr><tr><td>D</td><td>80</td></tr></table>"
)


def test_numeric_multiset_conflict_catches_the_real_muffins_mcq_false_positive():
    """The numeric multiset check catches the muffins MCQ pair.

    The corpus pair question_id=741 and 742 (`P6-Maths-2021-SA2-ACS-Junior.pdf` p4) has a cosine
    similarity of 0.998185, and both existing secondary signals in question_matching.py miss it
    (see the dedup.py module docstring). It showed during threshold calibration that this signal
    was needed.
    """
    assert _numeric_multiset_conflict(_MUFFINS_741, _MUFFINS_742) is True


def test_numeric_multiset_conflict_does_not_fire_on_a_real_exact_duplicate():
    assert _numeric_multiset_conflict(_MUFFINS_741, _MUFFINS_741) is False


def test_numeric_multiset_conflict_set_based_signals_are_blind_but_this_one_is_not():
    """The two existing secondary signals in question_matching.py miss this case.

    This documents the gap the new signal closes.
    """
    from app.pipeline.question_matching import (
        _distinguishing_terms_conflict,
        _numeric_terms_conflict,
    )
    assert _distinguishing_terms_conflict(_MUFFINS_741, _MUFFINS_742) is False
    assert _numeric_terms_conflict(_MUFFINS_741, _MUFFINS_742) is False
    assert _numeric_multiset_conflict(_MUFFINS_741, _MUFFINS_742) is True


def test_run_dedup_pass_blocks_the_real_muffins_case_from_auto_mark_even_above_threshold():
    """The muffins pair goes to the review queue even above AUTO_MARK_THRESHOLD.

    This shows the third signal is wired into run_dedup_pass().
    """
    with get_session() as session:  # type: ignore
        q1 = _make_question(question_text=_MUFFINS_741)
        q2 = _make_question(question_text=_MUFFINS_742)
        session.add(q1)
        session.add(q2)
        session.flush()
        q1_id, q2_id = q1.id, q2.id
        # Identical vectors give a cosine similarity of 1.0, above AUTO_MARK_THRESHOLD, so only
        # the secondary-signal gate is under test.
        vec = [1.0] + [0.0] * 1023
        _add_embedding(session, q1_id, vec)
        _add_embedding(session, q2_id, vec)
        session.commit()

        result = run_dedup_pass(session, apply=False)

    assert result.auto_marks == []
    assert len(result.review_queue) == 1
    entry = result.review_queue[0]
    assert {entry.question_id_a, entry.question_id_b} == {q1_id, q2_id}
    assert "number_multiset" in entry.reason


# =================================================================================================
# Clustering for groups of three or more
# =================================================================================================

def test_build_clusters_groups_a_real_chain_transitively():
    """Edges A-B and B-C form one 3-member cluster even though A-C is not an edge."""
    edges = [CandidatePair(1, 2, 0.999), CandidatePair(2, 3, 0.998)]
    clusters = build_clusters(edges)
    assert clusters == [{1, 2, 3}]


def test_build_clusters_keeps_unrelated_pairs_separate():
    edges = [CandidatePair(1, 2, 0.999), CandidatePair(3, 4, 0.998)]
    clusters = build_clusters(edges)
    assert {frozenset(c) for c in clusters} == {frozenset({1, 2}), frozenset({3, 4})}


def test_build_clusters_empty_input_returns_empty():
    assert build_clusters([]) == []


# =================================================================================================
# Similarity computation: tests of the vectorised matrix path
# =================================================================================================

def test_compute_full_similarity_matrix_identical_vectors_score_one():
    matrix = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    sim = compute_full_similarity_matrix(matrix)
    assert sim[0, 1] == pytest.approx(1.0, abs=1e-5)
    assert sim[0, 2] == pytest.approx(0.0, abs=1e-5)


def test_find_candidate_pairs_respects_floor_and_excludes_diagonal():
    question_ids = [10, 20, 30]
    sim = np.array([
        [1.0, 0.95, 0.5],
        [0.95, 1.0, 0.5],
        [0.5, 0.5, 1.0],
    ], dtype=np.float32)
    pairs = find_candidate_pairs(question_ids, sim, floor=0.9)
    assert len(pairs) == 1
    assert (pairs[0].question_id_a, pairs[0].question_id_b) == (10, 20)
    assert pairs[0].similarity == pytest.approx(0.95, abs=1e-5)


def test_find_candidate_pairs_below_floor_returns_nothing():
    question_ids = [1, 2]
    sim = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float32)
    assert find_candidate_pairs(question_ids, sim, floor=0.9) == []


# =================================================================================================
# Threshold decision boundary: constant checks and a controlled end-to-end pass
# =================================================================================================

def test_review_floor_reuses_match_confidence_threshold_not_a_separate_number():
    from app.pipeline.question_matching import MATCH_CONFIDENCE_THRESHOLD
    assert REVIEW_FLOOR == MATCH_CONFIDENCE_THRESHOLD


def test_auto_mark_threshold_is_strictly_above_review_floor():
    assert AUTO_MARK_THRESHOLD > REVIEW_FLOOR


def _seed_two_question_cluster(session, sim_pair_texts, has_diagram=(False, False)):
    """Seed two Question rows with identical text and identical embedding vectors.

    The similarity is 1.0 by construction, so run_dedup_pass() can be tested end to end without
    loading the encoder.
    """
    q1 = _make_question(question_text=sim_pair_texts[0], has_diagram=has_diagram[0])
    q2 = _make_question(question_text=sim_pair_texts[1], has_diagram=has_diagram[1])
    session.add(q1)
    session.add(q2)
    session.flush()
    vec = [1.0] + [0.0] * 1023
    _add_embedding(session, q1.id, vec)
    _add_embedding(session, q2.id, vec)  # identical vector, so cosine similarity 1.0
    session.commit()
    return q1.id, q2.id


def test_run_dedup_pass_dry_run_auto_marks_a_real_certain_duplicate_pair_without_writing():
    with get_session() as session:  # type: ignore
        q1_id, q2_id = _seed_two_question_cluster(
            session, ("Find the value of 3 + 4 x 5.", "Find the value of 3 + 4 x 5."),
        )
        result = run_dedup_pass(session, apply=False)

    assert any(
        set([m.survivor_id, *m.superseded_ids]) == {q1_id, q2_id} for m in result.auto_marks
    )
    with get_session() as session:  # type: ignore
        q1 = session.get(Question, q1_id)
        q2 = session.get(Question, q2_id)
        assert q1.superseded_by is None  # dry-run: no write happened
        assert q2.superseded_by is None


def test_run_dedup_pass_apply_sets_superseded_by_pointing_at_the_real_scored_survivor():
    with get_session() as session:  # type: ignore
        q1 = _make_question(question_text="Find the value of 3 + 4 x 5.", worked_solution_text=None)
        q2 = _make_question(question_text="Find the value of 3 + 4 x 5.", worked_solution_text="3+4x5=23")
        session.add(q1)
        session.add(q2)
        session.flush()
        q1_id, q2_id = q1.id, q2.id
        vec = [1.0] + [0.0] * 1023
        _add_embedding(session, q1_id, vec)
        _add_embedding(session, q2_id, vec)
        session.commit()

        run_dedup_pass(session, apply=True)

    with get_session() as session:  # type: ignore
        q1 = session.get(Question, q1_id)
        q2 = session.get(Question, q2_id)
        # q2 has worked_solution_text (tier 1), so it survives.
        assert q2.superseded_by is None
        assert q1.superseded_by == q2_id


def test_run_dedup_pass_below_review_floor_produces_neither_auto_mark_nor_review_entry():
    with get_session() as session:  # type: ignore
        q1 = _make_question(question_text="Find the value of 3 + 4 x 5.")
        q2 = _make_question(question_text="A completely different real question about symmetry.")
        session.add(q1)
        session.add(q2)
        session.flush()
        _add_embedding(session, q1.id, [1.0] + [0.0] * 1023)
        _add_embedding(session, q2.id, [0.0, 1.0] + [0.0] * 1022)  # orthogonal, so sim = 0.0
        session.commit()

        result = run_dedup_pass(session, apply=False)

    assert result.auto_marks == []
    assert result.review_queue == []


# =================================================================================================
# Data-safety guarantee: superseded_by is the only mutation.
# =================================================================================================

def test_dedup_module_source_contains_no_delete_call_at_all():
    """The dedup module source contains no `x.delete(...)` call.

    The module's AST is parsed rather than searched as text, so a docstring that mentions the
    guarantee is not a match. Adding `session.delete(...)` or `query.delete()` would fail this
    test even before a behavioural test caught it.
    """
    import ast

    tree = ast.parse(inspect.getsource(dedup))
    delete_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "delete"
    ]
    assert delete_calls == []


def test_apply_never_deletes_or_touches_ocr_records_the_only_mutation_is_superseded_by():
    """An apply=True pass changes only superseded_by.

    The pass runs on a 3-member cluster (covering the tie-break path) with OCRExtractionRecord
    rows. Afterwards (a) every Question row still exists, (b) every OCRExtractionRecord row and
    field is unchanged, and (c) superseded_by is the only Question field that changed.
    """
    with get_session() as session:  # type: ignore
        q1 = _make_question(question_text="Find the value of 3 + 4 x 5.", worked_solution_text=None)
        q2 = _make_question(question_text="Find the value of 3 + 4 x 5.", worked_solution_text="x")
        q3 = _make_question(question_text="Find the value of 3 + 4 x 5.", worked_solution_text=None)
        session.add_all([q1, q2, q3])
        session.flush()
        ids = [q1.id, q2.id, q3.id]
        vec = [1.0] + [0.0] * 1023
        for qid in ids:
            _add_embedding(session, qid, vec)
        for qid in ids:
            session.add(OCRExtractionRecord(
                question_id=qid, primary_reader_output="real primary output",
                secondary_reader_output="real secondary output", agreement_status="agree",
                disagreement_detail=None, disagreement_type=None, extracted_at=_now(),
            ))
        session.commit()

        before_question_count = session.query(Question).count()
        before_ocr_snapshot = {
            r.question_id: (
                r.primary_reader_output, r.secondary_reader_output, r.agreement_status,
                r.disagreement_detail, r.disagreement_type, r.extracted_at,
            )
            for r in session.query(OCRExtractionRecord).filter(
                OCRExtractionRecord.question_id.in_(ids),
            ).all()
        }
        before_question_snapshot = {
            q.id: {
                col: getattr(q, col) for col in (
                    "source_paper", "source_page_index", "question_number", "question_text",
                    "answer_value", "worked_solution_text", "has_diagram", "extraction_flag",
                    "school_name", "school_name_verified", "verification_status", "verified_at",
                )
            }
            for q in session.query(Question).filter(Question.id.in_(ids)).all()
        }

        run_dedup_pass(session, apply=True)

    with get_session() as session:  # type: ignore
        after_question_count = session.query(Question).count()
        assert after_question_count == before_question_count  # no row deleted

        after_ocr_snapshot = {
            r.question_id: (
                r.primary_reader_output, r.secondary_reader_output, r.agreement_status,
                r.disagreement_detail, r.disagreement_type, r.extracted_at,
            )
            for r in session.query(OCRExtractionRecord).filter(
                OCRExtractionRecord.question_id.in_(ids),
            ).all()
        }
        assert after_ocr_snapshot == before_ocr_snapshot  # every OCR record byte-identical

        after_questions = session.query(Question).filter(Question.id.in_(ids)).all()
        assert len(after_questions) == 3  # every row still exists
        superseded_count = 0
        for q in after_questions:
            for col, before_value in before_question_snapshot[q.id].items():
                assert getattr(q, col) == before_value, (
                    f"question {q.id}: field {col!r} changed — only superseded_by may change"
                )
            if q.superseded_by is not None:
                superseded_count += 1
        # q2 (with worked_solution_text) survives; q1 and q3 point at it.
        assert superseded_count == 2
        survivor = [q for q in after_questions if q.superseded_by is None]
        assert len(survivor) == 1 and survivor[0].worked_solution_text == "x"
        for q in after_questions:
            if q.superseded_by is not None:
                assert q.superseded_by == survivor[0].id


def test_boilerplate_text_is_excluded_from_dedup_candidacy_entirely():
    """Page-continuation boilerplate is never a dedup candidate.

    Strings such as '(Do on to the next page)' repeat exactly across dozens of corpus files. The
    first corpus-wide preview run treated them as duplicate questions.
    """
    with get_session() as session:  # type: ignore
        q1 = _make_question(question_text="(Do on to the next page)")
        q2 = _make_question(question_text="(Do on to the next page)")
        real_q = _make_question(question_text="Find the value of 3 + 4 x 5.")
        session.add_all([q1, q2, real_q])
        session.flush()
        vec = [1.0] + [0.0] * 1023
        _add_embedding(session, q1.id, vec)
        _add_embedding(session, q2.id, vec)
        _add_embedding(session, real_q.id, [0.0, 1.0] + [0.0] * 1022)
        session.commit()
        q1_id, q2_id, real_q_id = q1.id, q2.id, real_q.id

        question_ids, _ = load_corpus_embedding_matrix(session)
    assert q1_id not in question_ids
    assert q2_id not in question_ids
    assert real_q_id in question_ids


def test_short_real_question_is_not_mistaken_for_boilerplate():
    """A short PSLE question is still a dedup candidate.

    The boilerplate filter matches phrases, not a length cutoff.
    """
    with get_session() as session:  # type: ignore
        q = _make_question(question_text="Find the value of \\(68.2 \\div 4\\)\nAns:")
        session.add(q)
        session.flush()
        _add_embedding(session, q.id, [1.0] + [0.0] * 1023)
        session.commit()
        q_id = q.id

        question_ids, _ = load_corpus_embedding_matrix(session)
    assert q_id in question_ids


def test_apply_never_reconsiders_an_already_superseded_row():
    """A row with superseded_by already set is excluded from later runs.

    It is never re-superseded or re-clustered, so the pass is safe to re-run (Issue 231).
    """
    with get_session() as session:  # type: ignore
        survivor = _make_question(question_text="Find the value of 3 + 4 x 5.", worked_solution_text="x")
        session.add(survivor)
        session.flush()
        already_superseded = _make_question(
            question_text="Find the value of 3 + 4 x 5.", superseded_by=survivor.id,
        )
        session.add(already_superseded)
        session.flush()
        vec = [1.0] + [0.0] * 1023
        _add_embedding(session, survivor.id, vec)
        _add_embedding(session, already_superseded.id, vec)
        session.commit()
        survivor_id, superseded_id = survivor.id, already_superseded.id

        question_ids, _ = load_corpus_embedding_matrix(session)
    assert superseded_id not in question_ids
    assert survivor_id in question_ids
