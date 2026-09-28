"""
Real, small supplementary check for the Section 4 slot 5 (embedding model) bake-off.

`build_gold_pairs.py`'s own known-same pairs are EXACT-STRING duplicates (the real corpus's own
naturally-reused questions), a limitation: encoding the identical string twice
trivially favours any deterministic model and does not test the harder, more realistic Phase 4
case (a live VLM transcription of a photographed question, with real OCR-style textual variation,
matched against its corpus counterpart).

This script reuses REAL VLM transcription output already generated this session (Job B
preprocessing validation, Issue 184, `evaluation/model_selection/photo_transcription/
validate_preprocessing.py`'s AFTER-run output on `data/extracted/pages/P6-Maths-2021-CA1-
Henry-Park/2.png`) as noisy real queries, no new GPU inference needed, and no
fabricated text: this is the real Qwen3-VL-8B-Instruct transcription of a real photographed page,
copied verbatim from that real run's own log output.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

DB_PATH = Path(__file__).resolve().parents[3] / "psle.db"
OUT_PATH = Path(__file__).resolve().parent / "noisy_query_scores.json"

CANDIDATES = [
    "Qwen/Qwen3-Embedding-0.6B",
    "BAAI/bge-m3",
    "microsoft/harrier-oss-v1-0.6b",
]

# Real VLM transcription output (verbatim, Issue 184's own AFTER run), paired with the real
# DB id each one is a genuine (noisy, not exact-string) match for.
NOISY_QUERIES = [
    {
        "true_id": 1,
        "query": "1 Which digit in 15.89 is in the tenths place?\n(1) 1\n(2) 5\n(3) 8\n(4) 9",
    },
    {
        "true_id": 2,
        "query": "2 There were 585 640 visitors to a museum last year. Round this number to the\n"
                 "nearest thousand.\n(1) 585 000\n(2) 586 000\n(3) 590 000\n(4) 600 000",
    },
    {
        "true_id": 3,
        "query": "3 In the figure, ABC is a straight line. Find ∠w.\n(1) 74°\n(2) 90°"
                  "\n(3) 106°\n(4) 286°\n[DIAGRAM PRESENT]",
    },
]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return a_norm @ b_norm.T


def main() -> None:
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    # Real distractor pool: the 3 true-match rows themselves, plus 30 other real, distinct,
    # substantive corpus questions sampled deterministically (every 180th row, avoiding the 3
    # true-match ids), a real, fixed, reproducible distractor set, not cherry-picked to flatter
    # any candidate.
    true_ids = {q["true_id"] for q in NOISY_QUERIES}
    cur.execute("SELECT id, question_text FROM questions WHERE question_text IS NOT NULL "
                "AND LENGTH(question_text) > 60 ORDER BY id")
    all_rows = [(qid, text) for qid, text in cur.fetchall() if qid not in true_ids]
    distractors = all_rows[::180][:30]

    cur.execute(f"SELECT id, question_text FROM questions WHERE id IN "
                f"({','.join(str(i) for i in true_ids)})")
    true_rows = dict(cur.fetchall())

    pool = list(true_rows.items()) + distractors  # [(id, text), ...]
    pool_ids = [p[0] for p in pool]
    pool_texts = [p[1] for p in pool]
    print(f"Real distractor pool size: {len(pool)} ({len(true_rows)} true matches + "
          f"{len(distractors)} real distractors)")

    all_results = []
    for model_id in CANDIDATES:
        print(f"\nLoading {model_id} (should be cached locally now, no re-download) ...")
        model = SentenceTransformer(model_id, device="cpu", trust_remote_code=True)
        pool_emb = model.encode(pool_texts, convert_to_numpy=True, show_progress_bar=False)
        query_emb = model.encode(
            [q["query"] for q in NOISY_QUERIES], convert_to_numpy=True, show_progress_bar=False,
        )
        sims = cosine_sim(query_emb, pool_emb)

        model_result = {"model_id": model_id, "queries": []}
        for i, q in enumerate(NOISY_QUERIES):
            true_idx = pool_ids.index(q["true_id"])
            true_score = float(sims[i, true_idx])
            ranked = np.argsort(-sims[i])
            rank_of_true = int(np.where(ranked == true_idx)[0][0]) + 1  # 1-indexed
            best_distractor_score = float(
                max(sims[i, j] for j in range(len(pool_ids)) if j != true_idx)
            )
            print(f"  true_id={q['true_id']}: true_score={true_score:.4f} "
                  f"rank_of_true={rank_of_true} best_distractor_score={best_distractor_score:.4f}")
            model_result["queries"].append({
                "true_id": q["true_id"], "true_score": true_score,
                "rank_of_true": rank_of_true, "best_distractor_score": best_distractor_score,
            })
        all_results.append(model_result)

    OUT_PATH.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nWrote real noisy-query results to {OUT_PATH}")


if __name__ == "__main__":
    main()
