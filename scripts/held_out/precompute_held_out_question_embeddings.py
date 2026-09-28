r"""
Embed the held-out questions with the same model as the training pool.

Uses Qwen/Qwen3-Embedding-0.6B (`question_matching.EMBEDDING_MODEL_ID`), so the overlap check in
`check_held_out_training_overlap.py` compares vectors in the same space. Run after
stage_c_held_out_extraction.py and before the overlap check.

Embeddings are written to held_out.db's own `question_embeddings` table. Because held_out.db is a
separate SQLite file from psle.db, the two id spaces cannot collide; the tables are only compared
in memory by the overlap check.

Resumable, like `precompute_question_embeddings.py`: questions already embedded with the current
model are skipped, and each batch is committed.

Usage:
    backend/.venv/Scripts/python.exe scripts/held_out/precompute_held_out_question_embeddings.py
"""
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "backend"))

from app.pipeline.question_matching import EMBEDDING_MODEL_ID, _get_model  # noqa: E402

HELD_OUT_DB = _REPO_ROOT / "data" / "extracted" / "held_out.db"
_BATCH_SIZE = 32


def main() -> None:
    if not HELD_OUT_DB.exists():
        print(f"REFUSING: {HELD_OUT_DB} does not exist — run stage_c_held_out_extraction.py first.")
        sys.exit(1)

    con = sqlite3.connect(HELD_OUT_DB)
    cur = con.cursor()
    cur.execute("SELECT id, question_text FROM questions WHERE question_text IS NOT NULL ORDER BY id")
    rows = cur.fetchall()
    print(f"Real held-out questions with non-null question_text: {len(rows)}")

    cur.execute(
        "SELECT question_id FROM question_embeddings WHERE embedding_model = ?", (EMBEDDING_MODEL_ID,),
    )
    already = {r[0] for r in cur.fetchall()}
    candidates = [(qid, text) for qid, text in rows if qid not in already]
    print(f"Already embedded: {len(already)}. Remaining to embed: {len(candidates)}.")

    if candidates:
        model = _get_model()
        start = time.time()
        for batch_start in range(0, len(candidates), _BATCH_SIZE):
            batch = candidates[batch_start: batch_start + _BATCH_SIZE]
            texts = [t for _, t in batch]
            embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            now = datetime.now(timezone.utc).isoformat()
            for (qid, _), emb in zip(batch, embeddings):
                cur.execute(
                    "INSERT INTO question_embeddings (question_id, embedding_model, embedding_json, computed_at) "
                    "VALUES (?, ?, ?, ?)",
                    (qid, EMBEDDING_MODEL_ID, json.dumps(emb.tolist()), now),
                )
            con.commit()
            done = min(batch_start + _BATCH_SIZE, len(candidates))
            print(f"  {done}/{len(candidates)} embedded — elapsed {time.time()-start:.1f}s")
        print(f"Total elapsed: {time.time()-start:.1f}s")
    else:
        print("Nothing to do — all held-out questions already embedded.")

    cur.execute("SELECT COUNT(*) FROM question_embeddings WHERE embedding_model = ?", (EMBEDDING_MODEL_ID,))
    print(f"Final real count in held_out.db.question_embeddings: {cur.fetchone()[0]}")
    con.close()


if __name__ == "__main__":
    main()
