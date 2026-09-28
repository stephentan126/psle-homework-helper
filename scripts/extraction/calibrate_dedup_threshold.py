r"""
Calibrate AUTO_MARK_THRESHOLD for the Job A dedup pass from the live corpus (Issue 231).

The validation set is built directly from corpus text, with nothing invented, as in
`evaluation/model_selection/embedding/build_gold_pairs.py` (Issue 185). The script reports the
score separation that the threshold decision rests on, and recommends a value.

Known-duplicate (positive) class: every pair within each substantive exact-text duplicate group,
built the same way as `same_pairs` in `build_gold_pairs.py` (MIN_LEN=60, boilerplate and
instruction text excluded), but queried from the live corpus rather than that file's
one-pair-per-group snapshot, which may be stale. Limitation (also noted in Issue 185): these are
exact duplicates, the easiest case. A duplicate that differs by one OCR character is not in this
class. The corpus-wide run in run_job_a_dedup.py covers that case by spot-checking auto-marked
and review-queue pairs against the source.

Known-different but similar (hard negative) class: every pair within a sibling group, meaning
rows with the same `(source_paper, source_page_index)` but a different `question_number`. This is
how a multi-part question appears on a page, and different question numbers on one page are
different content by construction (the method of Issue 199(a)). Issue 199(a) used a 140-pair
stratified sample; this script uses the full sibling-pair population, which is stronger evidence
for a threshold that drives automatic, persistent changes.

Easy negative (sanity) class: the 42 `diff_pairs` in
`evaluation/model_selection/embedding/gold_pairs.json` (Issue 185), if that file is present. These
are substantive but unrelated pairs, not expected anywhere near the threshold, and are only a broad
sanity check.

The top 20 known-different pairs are also run through the three secondary signals, to show the
system-level protection and not just the raw embedding score. A false auto-mark changes data
persistently, which is worse than an extra review-queue entry, so the recommendation sits 70% of
the way from the known-different maximum to the known-same minimum rather than at the midpoint
(the same reasoning as Issue 186).

Requires `question_embeddings` to cover the ids involved for the current model. Pairs without a
cached embedding are reported and skipped, never embedded here.

Usage:
    backend/.venv/Scripts/python.exe scripts/extraction/calibrate_dedup_threshold.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "backend"))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.db.session import get_session, init_db  # noqa: E402
from app.models.question import NON_QUESTION_EXTRACTION_FLAG, Question  # noqa: E402
from app.models.question_embedding import QuestionEmbedding  # noqa: E402
from app.pipeline.dedup import (  # noqa: E402
    _numeric_multiset_conflict,
)
from app.pipeline.question_matching import (  # noqa: E402
    _distinguishing_terms_conflict,
    _numeric_terms_conflict,
)

EMBEDDING_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"  # matches question_matching.py's constant;
                                                    # not imported from there, so this small
                                                    # read-only script avoids the heavy
                                                    # sentence-transformers/torch import.

MIN_LEN = 60
_BOILERPLATE_PREFIXES = (
    "all diagrams in this paper",
    "use a dark blue or black ballpoint pen",
    "for each question, four options are given",
    "(go on to the next page)",
    "(do on to the next page)",
    "use the information below to answer questions",
    "answer all questions",
    "write your answers",
)


def _is_boilerplate(text: str) -> bool:
    lowered = text.strip().lower()
    return any(lowered.startswith(p) for p in _BOILERPLATE_PREFIXES)


def _looks_substantive(text: str) -> bool:
    return bool(re.search(r"\?|\(1\)|\d", text))


def main() -> None:
    init_db()
    with get_session() as session:
        rows = session.execute(
            select(
                Question.id, Question.question_text, Question.source_paper,
                Question.source_page_index, Question.question_number,
            ).where(
                Question.question_text.is_not(None),
                (Question.extraction_flag.is_(None))
                | (Question.extraction_flag.notlike(f"%{NON_QUESTION_EXTRACTION_FLAG}%")),
            ),
        ).all()
        embeddings = {
            r.question_id: json.loads(r.embedding_json)
            for r in session.query(QuestionEmbedding).filter(
                QuestionEmbedding.embedding_model == EMBEDDING_MODEL_ID,
            ).all()
        }
    texts_by_id = {qid: text for qid, text, *_ in rows}
    print(f"Real corpus rows considered: {len(rows)}. Real cached embeddings available: "
          f"{len(embeddings)}.")

    # --- Known-duplicate (positive) class: exact-text duplicate groups ------------------------
    dup_groups: dict[str, list[int]] = {}
    for qid, text, *_ in rows:
        norm = text.strip()
        if len(norm) < MIN_LEN or _is_boilerplate(norm) or not _looks_substantive(norm):
            continue
        dup_groups.setdefault(norm, []).append(qid)
    dup_groups = {t: ids for t, ids in dup_groups.items() if len(ids) > 1}
    same_pairs = [
        (a, b) for ids in dup_groups.values()
        for i, a in enumerate(ids) for b in ids[i + 1:]
    ]
    group_sizes = sorted((len(ids) for ids in dup_groups.values()), reverse=True)
    print(f"Real exact-duplicate groups: {len(dup_groups)} (sizes: {group_sizes[:10]}"
          f"{'...' if len(group_sizes) > 10 else ''}), real known-same pairs (all real "
          f"combinations, not one per group): {len(same_pairs)}.")

    # --- Known-different but similar (hard negative) class: full sibling population ----------
    sibling_groups: dict[tuple, list[tuple]] = {}
    for qid, text, source_paper, source_page_index, question_number in rows:
        norm = text.strip()
        if len(norm) < MIN_LEN or _is_boilerplate(norm) or not _looks_substantive(norm):
            continue
        key = (source_paper, source_page_index)
        sibling_groups.setdefault(key, []).append((qid, question_number))
    sibling_pairs: list[tuple] = []
    for members in sibling_groups.values():
        for i, (qid_a, num_a) in enumerate(members):
            for qid_b, num_b in members[i + 1:]:
                if num_a != num_b:  # different question_number, different content
                    sibling_pairs.append((qid_a, qid_b))
    print(f"Real sibling groups (same page, different question_number): {len(sibling_groups)}, "
          f"real FULL sibling-pair population (not sampled): {len(sibling_pairs)}.")

    # --- Easy-negative class: reuse the diff_pairs saved for Issue 185 -----------------------
    gold_path = _REPO_ROOT / "evaluation" / "model_selection" / "embedding" / "gold_pairs.json"
    easy_diff_pairs: list[tuple] = []
    if gold_path.exists():
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        easy_diff_pairs = [(p["id_a"], p["id_b"]) for p in gold["diff_pairs"]]
    print(f"Real easy-negative pairs (Issue 185's own gold_pairs.json): {len(easy_diff_pairs)}.")

    def score_pairs(pairs: list[tuple]) -> list[tuple]:
        scored = []
        skipped = 0
        for a, b in pairs:
            va, vb = embeddings.get(a), embeddings.get(b)
            if va is None or vb is None:
                skipped += 1
                continue
            va_arr, vb_arr = np.array(va, dtype=np.float64), np.array(vb, dtype=np.float64)
            sim = float(
                np.dot(va_arr, vb_arr) / (np.linalg.norm(va_arr) * np.linalg.norm(vb_arr)),
            )
            scored.append((a, b, sim))
        if skipped:
            print(f"  (skipped {skipped}/{len(pairs)} pairs — no real cached embedding yet for "
                  f"one or both ids; run the embedding precompute backfill to completion first "
                  f"for full coverage.)")
        return scored

    print("\nScoring real known-same (exact-duplicate) pairs...")
    same_scored = score_pairs(same_pairs)
    print("Scoring real sibling (known-different) pairs...")
    sibling_scored = score_pairs(sibling_pairs)
    print("Scoring real easy-negative pairs...")
    easy_scored = score_pairs(easy_diff_pairs)

    if same_scored:
        same_scores = [s for _, _, s in same_scored]
        print(f"\nReal known-same (n={len(same_scores)}): min={min(same_scores):.6f} "
              f"max={max(same_scores):.6f} mean={sum(same_scores)/len(same_scores):.6f}")
    else:
        print("\nNo real known-same pairs could be scored (no cached embeddings yet).")

    all_diff_scored = sibling_scored + easy_scored
    if all_diff_scored:
        diff_scores = [s for _, _, s in all_diff_scored]
        worst = max(all_diff_scored, key=lambda t: t[2])
        print(f"Real known-different (n={len(diff_scores)}, sibling+easy combined): "
              f"min={min(diff_scores):.6f} max={max(diff_scores):.6f} "
              f"mean={sum(diff_scores)/len(diff_scores):.6f}")
        print(f"  Real WORST (highest-scoring) known-different pair: "
              f"question_id={worst[0]} vs question_id={worst[1]}, score={worst[2]:.6f}")
        # The 20 highest-scoring known-different pairs, with the three secondary signals
        # applied, to show the system-level result rather than the raw embedding score alone.
        top20 = sorted(all_diff_scored, key=lambda t: t[2], reverse=True)[:20]
        print("  Real top 20 highest-scoring known-different pairs, WITH the real secondary-"
              "signal gate applied (for direct spot-check):")
        gate_failures = []
        for a, b, s in top20:
            ta, tb = texts_by_id.get(a, ""), texts_by_id.get(b, "")
            term_c = _distinguishing_terms_conflict(ta, tb)
            num_c = _numeric_terms_conflict(ta, tb)
            multi_c = _numeric_multiset_conflict(ta, tb)
            blocked = term_c or num_c or multi_c
            print(f"    question_id={a} vs question_id={b}: {s:.6f} -- gate: "
                  f"{'BLOCKED (safe)' if blocked else 'NOT BLOCKED -- real residual risk'} "
                  f"(terms={term_c}, numbers={num_c}, multiset={multi_c})")
            if not blocked:
                gate_failures.append((a, b, s))
        if gate_failures:
            print(f"\n  REAL RESIDUAL RISK: {len(gate_failures)}/{len(top20)} of the top-20 "
                  f"highest-scoring known-different pairs clear ALL THREE secondary signals "
                  f"undetected -- these specific real ids need a threshold set above their score, "
                  f"or they remain a real auto-mark risk regardless of the gate.")
        else:
            print(f"\n  All top-20 highest-scoring known-different pairs are correctly BLOCKED by "
                  f"the real 3-signal gate -- the gate, not the raw embedding score alone, is "
                  f"real, additional, confirmed protection at this corpus's real worst known "
                  f"cases.")
    else:
        print("No real known-different pairs could be scored.")

    if same_scored and all_diff_scored:
        same_min = min(s for _, _, s in same_scored)
        diff_max = max(s for _, _, s in all_diff_scored)
        print(f"\nReal separation: known-same min={same_min:.6f}, known-different max="
              f"{diff_max:.6f}, gap={same_min - diff_max:.6f}")
        if same_min > diff_max:
            midpoint = (same_min + diff_max) / 2
            # As for MATCH_CONFIDENCE_THRESHOLD (Issue 186): a false auto-mark (a persistent
            # data change) is worse than a pair only sent to the review queue (recoverable, just
            # slower), so bias towards the higher side of the observed gap, not the midpoint.
            recommended = diff_max + (same_min - diff_max) * 0.7
            print(f"CLEAN REAL SEPARATION EXISTS. Midpoint would be {midpoint:.6f}. Recommended "
                  f"AUTO_MARK_THRESHOLD (biased toward the safer/higher side, 70% of the way from "
                  f"diff_max to same_min, same asymmetric reasoning as Issue 186): "
                  f"{recommended:.6f}")
        else:
            print("NO CLEAN REAL SEPARATION — known-different max >= known-same min. Per "
                  "explicit instruction: do NOT force an auto-mark tier that isn't evidenced. "
                  "Recommend REVIEW-QUEUE-ONLY for this round.")


if __name__ == "__main__":
    main()
