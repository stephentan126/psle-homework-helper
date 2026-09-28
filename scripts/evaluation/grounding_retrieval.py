r"""
Retrieval of few-shot examples for the RAG conditions (B and C) of the grounding experiment.

Candidate pool: `training_export_v1.jsonl`, the curated, human-reviewed training export, rather
than the raw live corpus of about 5,819 rows. Two reasons: (1) the method retrieves the k nearest
labelled examples (as in KATE), and most of the raw corpus has no verified answer or working to
label with; (2) the export already excludes the 5 held-out-overlap ids and the 56
corrupted-source ids of Issue 314, since neither passed its upstream filters, so no extra
query-time exclusion is needed.

The export has 1,349 rows but only 1,124 unique question_id values: 225 questions have both a
Tier 1 and a Tier 2 entry, and 1,124 have Tier 1. The pool is the 1,124 unique questions, with one
embedding per question, so the same question cannot be retrieved twice under two tier phrasings.

Embeddings: every pool id already has a cached Qwen/Qwen3-Embedding-0.6B vector in
`psle.db.question_embeddings` (Issues 185, 231), which is looked up rather than recomputed. The 446
held-out rows have no cached embeddings under this model, so they are embedded here.

The frozen held-out database is read-only by design (Issue 251; file mode `-r--r--r--`). This
module never writes to `held_out_frozen_2026-09-09.db`, not even a cache. Held-out embeddings are
cached in a separate file under `data/extracted/slice9_eval/`.

Run directly for a retrieval-only smoke test on 8 held-out questions (no generation):
    backend/.venv/Scripts/python.exe scripts/evaluation/grounding_retrieval.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "backend"))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_DB_PATH = _REPO_ROOT / "psle.db"
_HELD_OUT_DB_PATH = _REPO_ROOT / "data" / "extracted" / "held_out_frozen" / "held_out_frozen_2026-09-09.db"
_EXPORT_PATH = _REPO_ROOT / "data" / "extracted" / "slice6_step0c" / "training_export_v1.jsonl"
_EVAL_DIR = _REPO_ROOT / "data" / "extracted" / "slice9_eval"
_HELD_OUT_EMBED_CACHE_PATH = _EVAL_DIR / "held_out_446_embeddings_qwen3-0.6b.json"

EMBEDDING_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"  # must match app.pipeline.question_matching's
                                                    # constant, so both use the same space.
DEFAULT_K = 3  # A conventional small few-shot count: enough to show a worked pattern without
               # inflating prompt length across 446 x 3 generations. Revisit if the smoke test
               # shows the retrieved neighbours are too thin or noisy at this k.

# Minimum-similarity floor, added after the smoke test. Retrieved neighbours scored 0.45-0.70, well
# below the 0.90 match-first threshold. That is expected, since the aim is similar but different
# examples, a range this embedding model was not validated for (Issue 185 covered only 0.90 and
# above). The top retrieved similarity of each of the 8 smoke-test queries was judged by hand for
# topical relevance:
# - id=1 (0.5493): irrelevant. A "smallest decimal" multiple-choice question pulled a
#   digit-forming question, an averages question and a stray units-instruction fragment.
# - id=4 (0.5125): irrelevant. A bicycle-length estimate pulled wheel-circumference and
#   rectangle-area questions, related only by the word "bicycle".
# - ids 2, 3, 5, 6, 7, 8 (0.6014-0.6950, top-1 each): all on topic.
# The sample shows a clean gap between 0.5493 and 0.6014, so the floor is the midpoint, 0.575.
# It is applied to every retrieved candidate, not just the top one, so a query can keep 0 to k
# examples. A query with none left gets no RAG context, and conditions B and C use condition A's
# prompt for that question; the `used_fallback` flag of `retrieve_k_nearest_floored()` records
# this. With n=8 this is a judgement call, not a statistically strong derivation.
SIMILARITY_FLOOR = 0.575

_model = None  # lazy singleton, same pattern as question_matching._get_model()


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL_ID, device="cpu")
    return _model


def load_pool() -> list[dict]:
    """Load the 1,124 unique pool questions with their cached embeddings.

    The pool ids come from the export's Tier 1 rows, but answer_value and worked_solution_text are
    read from psle.db. A Tier 1 hint never states the final answer, so the grounding content for
    the few-shot block has to come from the corpus row, not from the export.
    """
    with open(_EXPORT_PATH, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    tier1_by_qid = {r["question_id"]: r for r in rows if r["tier"] == 1}
    pool_ids = sorted(tier1_by_qid.keys())
    assert len(pool_ids) == 1124, f"expected 1,124 unique pool questions, got {len(pool_ids)}"

    con = sqlite3.connect(str(_DB_PATH))
    cur = con.cursor()
    cur.execute(
        "SELECT id, question_text, answer_value, worked_solution_text FROM questions "
        f"WHERE id IN ({','.join('?' * len(pool_ids))})",
        pool_ids,
    )
    db_rows = {r[0]: r for r in cur.fetchall()}
    cur.execute(
        f"SELECT question_id, embedding_json FROM question_embeddings "
        f"WHERE embedding_model = ? AND question_id IN ({','.join('?' * len(pool_ids))})",
        [EMBEDDING_MODEL_ID] + pool_ids,
    )
    embed_by_qid = {r[0]: json.loads(r[1]) for r in cur.fetchall()}
    con.close()

    missing_embed = [qid for qid in pool_ids if qid not in embed_by_qid]
    assert not missing_embed, f"pool ids missing a cached embedding: {missing_embed}"

    pool = []
    for qid in pool_ids:
        _, qt, av, wst = db_rows[qid]
        pool.append({
            "question_id": qid,
            "question_text": qt,
            "answer_value": av,
            "worked_solution_text": wst,
            "embedding": embed_by_qid[qid],
        })
    return pool


def load_held_out() -> list[dict]:
    """Load all 446 frozen held-out rows through a read-only connection."""
    con = sqlite3.connect(f"file:{_HELD_OUT_DB_PATH}?mode=ro", uri=True)
    cur = con.cursor()
    cur.execute(
        "SELECT id, question_text, answer_value, has_diagram FROM questions ORDER BY id"
    )
    rows = [
        {"question_id": r[0], "question_text": r[1], "answer_value": r[2], "has_diagram": bool(r[3])}
        for r in cur.fetchall()
    ]
    con.close()
    assert len(rows) == 446, f"expected 446 held-out rows, got {len(rows)}"
    return rows


def compute_or_load_held_out_embeddings(held_out: list[dict]) -> dict[int, list[float]]:
    """Return embeddings for the 446 held-out questions, computing them on first use.

    The frozen database has none cached under this model, so they are cached in a separate file,
    never in the frozen database. Later runs reuse the cache: at about 1.8 s per text (Issue 185),
    embedding all 446 takes about 13 minutes.
    """
    if _HELD_OUT_EMBED_CACHE_PATH.exists():
        with open(_HELD_OUT_EMBED_CACHE_PATH, encoding="utf-8") as f:
            cached = json.load(f)
        cached_ids = {int(k) for k in cached}
        if cached_ids == {r["question_id"] for r in held_out}:
            print(f"Reusing cached held-out embeddings: {_HELD_OUT_EMBED_CACHE_PATH}")
            return {int(k): v for k, v in cached.items()}

    print(f"Computing fresh embeddings for {len(held_out)} held-out questions "
          f"({EMBEDDING_MODEL_ID}, CPU) -- one-time cost...")
    model = _get_model()
    texts = [r["question_text"] for r in held_out]
    embeds = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    result = {r["question_id"]: embeds[i].tolist() for i, r in enumerate(held_out)}

    _EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(_HELD_OUT_EMBED_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in result.items()}, f)
    print(f"Cached to {_HELD_OUT_EMBED_CACHE_PATH} (NOT written to the frozen held-out db)")
    return result


def _cosine_sim(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = query / np.linalg.norm(query)
    matrix_norm = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix_norm @ query_norm


def retrieve_k_nearest(
    query_embedding: list[float], pool: list[dict], k: int = DEFAULT_K,
) -> list[dict]:
    """Return the k pool entries most similar to the query, each with its similarity score.

    Plain NumPy cosine similarity, as in find_best_match() (Issue 52); a vector database is not
    needed at this scale. No floor is applied; the generation scripts use
    `retrieve_k_nearest_floored()`.
    """
    matrix = np.array([p["embedding"] for p in pool], dtype=np.float32)
    query = np.array(query_embedding, dtype=np.float32)
    sims = _cosine_sim(query, matrix)
    top_idx = np.argsort(-sims)[:k]
    return [{**pool[i], "similarity": float(sims[i])} for i in top_idx]


def retrieve_k_nearest_floored(
    query_embedding: list[float], pool: list[dict], k: int = DEFAULT_K,
    floor: float = SIMILARITY_FLOOR,
) -> tuple[list[dict], bool]:
    """Return up to k nearest pool entries that clear `floor`, and a fallback flag.

    The whole sorted pool is filtered by `floor` before the top k are taken, so a good candidate
    ranked 4th or 5th is not lost because some of the top 3 were weak. Returns
    (examples, used_fallback). `used_fallback=True` means no candidate cleared the floor,
    `examples` is empty, and the caller should use a no-RAG (condition A) prompt for this row.
    """
    matrix = np.array([p["embedding"] for p in pool], dtype=np.float32)
    query = np.array(query_embedding, dtype=np.float32)
    sims = _cosine_sim(query, matrix)
    order = np.argsort(-sims)
    survivors = [i for i in order if sims[i] >= floor][:k]
    if not survivors:
        return [], True
    return [{**pool[i], "similarity": float(sims[i])} for i in survivors], False


def format_few_shot_block(examples: list[dict]) -> str:
    """Render retrieved examples as a few-shot block for the generation prompt.

    Each example shows its verified question, working and answer, not the Tier 1 restatement
    that the export holds for these ids.
    """
    parts = []
    for i, ex in enumerate(examples, 1):
        working = ex["worked_solution_text"] or "(no printed working for this example)"
        parts.append(
            f"Example {i}:\nQuestion: {ex['question_text']}\nWorking: {working}\n"
            f"Final Answer: {ex['answer_value']}"
        )
    return "\n\n".join(parts)


def main() -> None:
    """Smoke test: retrieval only, on 8 held-out questions, before the full generation run."""
    pool = load_pool()
    held_out = load_held_out()
    print(f"Pool: {len(pool)} unique questions. Held-out: {len(held_out)} questions.\n")

    held_out_embeds = compute_or_load_held_out_embeddings(held_out)

    smoke_sample = held_out[:8]
    for hq in smoke_sample:
        neighbors, used_fallback = retrieve_k_nearest_floored(
            held_out_embeds[hq["question_id"]], pool, k=DEFAULT_K,
        )
        print(f"=== held-out id={hq['question_id']} (has_diagram={hq['has_diagram']}) "
              f"[FALLBACK: {used_fallback}] ===")
        print(f"Q: {hq['question_text'][:100]!r}")
        print(f"real answer_value: {hq['answer_value']!r}")
        for n in neighbors:
            print(f"  -> pool id={n['question_id']} sim={n['similarity']:.4f} "
                  f"Q={n['question_text'][:70]!r} A={n['answer_value']!r}")
        if used_fallback:
            print("  (0 candidates cleared the similarity floor -- falls back to no-RAG prompt)")
        print()


if __name__ == "__main__":
    main()
