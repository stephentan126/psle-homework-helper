"""
Precompute the corpus embedding cache used by match-first question lookup (Issues 185, 186).

Embeds every corpus question with `Qwen/Qwen3-Embedding-0.6B` (`EMBEDDING_MODEL_ID` in
`app/pipeline/question_matching.py`, the winner of the embedding model comparison) and writes the
vectors to `question_embeddings`. A live photo submission then never has to embed the whole corpus
at request time: the comparison measured about 1.8 s per text on this machine's CPU, so embedding
all 5,425 corpus rows on each request would add about 2.7 hours.

Checkpointing and resumability, as in `precompute_gate_results.py` (Issue 172), since a full run
is slow:
  - Each batch is committed as it completes, not in one transaction at the end, so a kill leaves
    every computed row in place.
  - Questions that already have a row for the current `EMBEDDING_MODEL_ID` are skipped, so a
    re-run after an interruption does only the remaining work, with no resume flag.
  - Progress (running count, elapsed time, rate and ETA) is printed after every batch.

Run a small `--limit` slice first to confirm the path works end to end and to measure the
rate, before a multi-hour `--all` run.

Usage:
    # Small slice first, to check the path and measure the rate:
    backend/.venv/Scripts/python.exe scripts/extraction/precompute_question_embeddings.py --limit 50

    # Full corpus, once the slice has been checked:
    backend/.venv/Scripts/python.exe scripts/extraction/precompute_question_embeddings.py --all
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
from app.models.question import Question  # noqa: E402
from app.models.question_embedding import QuestionEmbedding  # noqa: E402
from app.pipeline.question_matching import EMBEDDING_MODEL_ID, _get_model  # noqa: E402

_BATCH_SIZE = 32


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--limit", type=int, help="Embed at most this many real, not-yet-embedded "
                        "corpus questions (a small, bounded slice to prove the path).")
    group.add_argument("--all", action="store_true", help="Embed the full remaining real corpus.")
    args = parser.parse_args()

    init_db()

    with get_session() as session:
        already_done = set(
            session.scalars(
                select(QuestionEmbedding.question_id).where(
                    QuestionEmbedding.embedding_model == EMBEDDING_MODEL_ID,
                ),
            ).all(),
        )
        all_questions = session.scalars(
            select(Question).where(Question.question_text.is_not(None)).order_by(Question.id),
        ).all()
        candidates = [(q.id, q.question_text) for q in all_questions if q.id not in already_done]

    print(f"Real corpus: {len(all_questions)} rows, {len(already_done)} already embedded with "
          f"{EMBEDDING_MODEL_ID}, {len(candidates)} remaining.")
    if args.limit:
        candidates = candidates[: args.limit]

    # Sort by text length before batching (Issue 231). A transformer batch pads every item to the
    # length of its longest member, so one outlier makes the whole batch pay its cost. A corpus
    # outlier (question_id=3561, 16,473 characters, a Paper 2 `no_matching_answer_key_row` page
    # from the known family of Issues 110, 131 and 134) was seen to slow a 32-item batch to
    # several minutes. Sorting puts long rows into their own batches, which removed the stall on
    # this corpus. The sort is stable, so equal-length texts keep corpus order.
    candidates.sort(key=lambda pair: len(pair[1]))

    print(f"Will embed {len(candidates)} real question(s) this run (sorted shortest-to-longest "
          f"text, to avoid one long outlier padding-dragging an entire batch of short ones).")

    if not candidates:
        print("Nothing to do.")
        return

    model = _get_model()
    start = time.time()
    done = 0
    for batch_start in range(0, len(candidates), _BATCH_SIZE):
        batch = candidates[batch_start: batch_start + _BATCH_SIZE]
        texts = [text for _, text in batch]
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

        with get_session() as session:
            now = _now_iso()
            for (qid, _), emb in zip(batch, embeddings):
                session.add(QuestionEmbedding(
                    question_id=qid, embedding_model=EMBEDDING_MODEL_ID,
                    embedding_json=json.dumps(emb.tolist()), computed_at=now,
                ))
            session.commit()

        done += len(batch)
        elapsed = time.time() - start
        rate = elapsed / done
        remaining = len(candidates) - done
        eta_s = remaining * rate
        print(f"  {done}/{len(candidates)} embedded — elapsed {elapsed:.1f}s, "
              f"rate {rate:.3f}s/question, ETA {eta_s / 60:.1f} min")

    total_elapsed = time.time() - start
    print(f"\nDone: {done} real question(s) embedded in {total_elapsed:.1f}s "
          f"({total_elapsed / max(done, 1):.3f}s/question real measured rate).")


if __name__ == "__main__":
    main()
