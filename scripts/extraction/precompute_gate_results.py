r"""
Precompute gate results for corpus questions so the hint endpoint can skip live LLM work.

Runs the same gate pipeline as the live endpoint (`gate.run_full_gate_pipeline()`, shared rather
than duplicated) once per corpus question, and writes the result to `precomputed_gate_results`.
`POST /api/questions/{id}/hint` can then serve any question in the known question bank without
the expensive live LLM calls.

Single trial only, not a 3-trial majority vote. The self-consistency step inside
`run_full_gate_pipeline()` already makes its own 2 passes (Issues 46, 167, 170). This script runs
the pipeline once per question, matching what one live request would produce. A 3-trial vote was
considered and remains an open design option; it is not built here.

Checkpointing and resumability matter here: Issue 172's audit estimated a full single-trial run at
about 21.5 hours on the development machine, likely across several sessions.
  - Each question's result is committed in its own transaction, so a kill leaves every computed
    row in place.
  - A question is skipped if it already has an active row for the current `gate.GATE_VERSION`,
    `llm_service.LLM_MODEL_PATH` and adapter. Re-running after an interruption, or after a full
    run, does only the remaining work, with no resume flag or state file: the question table and
    the version-stamped result rows are the checkpoint.
  - Progress (question id, running count, elapsed time and ETA) is printed after every question.

A scope is required (--all, --source-paper-contains or --question-ids), so a multi-hour job cannot
start by accident.

Usage:
    # Small bounded slice first, to check the path end to end:
    backend/.venv/Scripts/python.exe scripts/extraction/precompute_gate_results.py \
        --source-paper-contains acsjunior,aitong,chij,mgs,mgspayalebar,redswastika,scgs --limit 20

    # Full corpus, once the slice has been checked:
    backend/.venv/Scripts/python.exe scripts/extraction/precompute_gate_results.py --all
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "backend"))

from sqlalchemy import select  # noqa: E402

from app.db.session import get_session, init_db  # noqa: E402
from app.models.precomputed_gate_result import PrecomputedGateResult  # noqa: E402
from app.models.question import Question  # noqa: E402
from app.pipeline.gate import (  # noqa: E402
    GATE_VERSION, compute_gate_content_fingerprint, run_full_gate_pipeline,
)
from app.services.llm_service import LLM_ADAPTER_PATH, LLM_MODEL_PATH  # noqa: E402


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _select_candidates(
    source_paper_contains: list[str] | None, limit: int | None, question_ids: list[int] | None,
) -> list[tuple[int, str, str, bool, str | None]]:
    """Select candidate questions in id order.

    A fixed order means repeated runs process questions in the same sequence, so progress and ETA
    stay meaningful across resumed runs.
    """
    with get_session() as session:
        query = select(Question).where(Question.extraction_flag.is_(None)).order_by(Question.id)
        questions = session.scalars(query).all()
        rows = [
            (q.id, q.question_text, q.answer_value, q.has_diagram, q.worked_solution_text)
            for q in questions
            if q.answer_value
        ]
    if question_ids:
        wanted = set(question_ids)
        rows = [r for r in rows if r[0] in wanted]
    if source_paper_contains:
        # Re-query to filter on source_paper, which the tuple above leaves out. Simple is fine
        # for a batch job.
        with get_session() as session:
            questions = session.scalars(
                select(Question).where(Question.extraction_flag.is_(None)).order_by(Question.id),
            ).all()
            wanted_ids = {
                q.id for q in questions
                if q.answer_value
                and any(needle in q.source_paper for needle in source_paper_contains)
            }
        rows = [r for r in rows if r[0] in wanted_ids]
    if limit is not None:
        rows = rows[:limit]
    return rows


def _already_current(session, question_id: int) -> bool:
    existing = session.scalar(
        select(PrecomputedGateResult).where(
            PrecomputedGateResult.question_id == question_id,
            PrecomputedGateResult.is_active.is_(True),
            PrecomputedGateResult.gate_version == GATE_VERSION,
            PrecomputedGateResult.llm_model_path == LLM_MODEL_PATH,
            # The adapter must also match the configured one (Issue 369, following Issue 352).
            # A row computed with a different adapter is not current, even if the code and base
            # model are unchanged.
            PrecomputedGateResult.adapter_path == LLM_ADAPTER_PATH,
        ),
    )
    return existing is not None


def _classify_for_storage(question_text: str, answer_value: str, has_diagram: bool) -> str:
    from app.pipeline.gate import classify_question_type
    return classify_question_type(question_text, answer_value, has_diagram=has_diagram)


def run(candidates: list[tuple[int, str, str, bool, str | None]]) -> None:
    total = len(candidates)
    print(f"Precompute batch job starting: {total} candidate questions, "
          f"GATE_VERSION={GATE_VERSION!r}, LLM_MODEL_PATH={LLM_MODEL_PATH!r}, "
          f"LLM_ADAPTER_PATH={LLM_ADAPTER_PATH!r}")

    skipped_already_current = 0
    processed = 0
    start_all = time.time()

    for i, (qid, qtext, answer, has_diagram, worked_solution) in enumerate(candidates):
        with get_session() as session:
            if _already_current(session, qid):
                skipped_already_current += 1
                continue

        q_start = time.time()
        q_type = _classify_for_storage(qtext, answer, has_diagram)
        effective_gate_result, hint, _ = run_full_gate_pipeline(
            qtext, answer, has_diagram=has_diagram, worked_solution_text=worked_solution,
        )
        q_elapsed = time.time() - q_start

        with get_session() as session:
            # Keep history rather than overwrite: deactivate any earlier active row for this
            # question (from an older gate_version or model), then insert the new active one.
            # Both happen in one transaction, so a kill between them cannot leave a question that
            # had an active row with none.
            old_active = session.scalars(
                select(PrecomputedGateResult).where(
                    PrecomputedGateResult.question_id == qid,
                    PrecomputedGateResult.is_active.is_(True),
                ),
            ).all()
            for row in old_active:
                row.is_active = False

            new_row = PrecomputedGateResult(
                question_id=qid,
                gate_version=GATE_VERSION,
                llm_model_path=LLM_MODEL_PATH,
                # Record the adapter used for this run (Issue 369), so a later adapter change is
                # caught by the read-time comparison rather than relying on a GATE_VERSION bump.
                adapter_path=LLM_ADAPTER_PATH,
                content_fingerprint=compute_gate_content_fingerprint(
                    qtext, answer, has_diagram, worked_solution,
                ),
                question_type=q_type,
                tier_allowed=effective_gate_result.tier_allowed,
                gate_passed=effective_gate_result.passed,
                gate_reason=effective_gate_result.reason,
                gate_detail_json=json.dumps(effective_gate_result.detail, default=str),
                hint_tier=hint["tier"],
                hint_text=hint["hint_text"],
                computed_at=_now_iso(),
                is_active=True,
            )
            session.add(new_row)
            session.flush()

        processed += 1
        elapsed_all = time.time() - start_all
        done = skipped_already_current + processed
        rate = elapsed_all / processed if processed else 0
        remaining = total - done
        eta_s = remaining * rate if rate else 0
        print(
            f"[{done}/{total}] id={qid} type={q_type} tier_allowed={effective_gate_result.tier_allowed} "
            f"q_time={q_elapsed:.2f}s total_elapsed={elapsed_all:.0f}s "
            f"eta_remaining={eta_s/60:.1f}min (skipped {skipped_already_current} already-current)",
        )

    print(
        f"\nDone. {processed} newly computed, {skipped_already_current} already current "
        f"(skipped), {total} total candidates. Real wall time this run: "
        f"{time.time()-start_all:.1f}s",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Process the full real corpus.")
    parser.add_argument(
        "--source-paper-contains", type=str, default=None,
        help="Comma-separated substrings; only questions whose source_paper contains one of "
             "these are candidates (for a small, bounded real slice).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap the candidate list to N.")
    parser.add_argument(
        "--question-ids", type=str, default=None,
        help="Comma-separated real question ids to process (overrides other filters' selection, "
             "still subject to --limit).",
    )
    args = parser.parse_args()

    if not args.all and not args.source_paper_contains and not args.question_ids:
        parser.error(
            "Refusing to run with no real scope specified -- pass --all for the full corpus, "
            "or --source-paper-contains/--question-ids for a bounded real slice. This is "
            "deliberate: a multi-hour job should never start by accident.",
        )

    source_paper_contains = (
        [s.strip() for s in args.source_paper_contains.split(",")]
        if args.source_paper_contains else None
    )
    question_ids = (
        [int(s.strip()) for s in args.question_ids.split(",")] if args.question_ids else None
    )

    init_db()
    candidates = _select_candidates(source_paper_contains, args.limit, question_ids)
    run(candidates)


if __name__ == "__main__":
    main()
