"""
Section 4 slot 5 (embedding model) bake-off, real evaluation of 3 real candidates against the
real gold pairs built by `build_gold_pairs.py` (85 real known-same pairs, 42 real known-different
pairs, drawn directly from the real 5,425-row corpus, no fabricated text).

Candidates (all Apache 2.0 / MIT, NV-Embed-v2 and jina-embeddings-v3 already rejected on licence,
CC-BY-NC, per Section 4 slot 5's own existing text, not retested here):
- Qwen/Qwen3-Embedding-0.6B (named in the original plan)
- BAAI/bge-m3 (named in the original plan)
- microsoft/harrier-oss-v1-0.6b (found via a live model-landscape check this session, MIT,
  released 2026-04, NOT in the original plan, same discovery pattern as Job B's own
  Qwen3-VL-8B find, Issue 183)

Real metrics computed, not assumed:
- retrieval recall@1 / recall@3 over a real candidate pool (all real "b" question texts from
  both same_pairs and diff_pairs combined, so a same-pair query's true match sits among real
  distractors, not compared 1-on-1 in isolation)
- known-same vs. known-different real cosine-similarity score distributions (mean/min/max) , 
  this is also the real, direct input to the match-confidence-threshold decision (a separate,
  later issue), not just a bake-off-only number
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

GOLD_PATH = Path(__file__).resolve().parent / "gold_pairs.json"
RESULTS_PATH = Path(__file__).resolve().parent / "bakeoff_scores.json"

CANDIDATES = [
    "Qwen/Qwen3-Embedding-0.6B",
    "BAAI/bge-m3",
    "microsoft/harrier-oss-v1-0.6b",
]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return a_norm @ b_norm.T


def evaluate_model(model_id: str, gold: dict) -> dict:
    print(f"\n{'=' * 90}\nLoading {model_id} ...")
    t0 = time.time()
    model = SentenceTransformer(model_id, device="cpu", trust_remote_code=True)
    load_time = time.time() - t0
    print(f"Real load time: {load_time:.1f}s")

    same_pairs = gold["same_pairs"]
    diff_pairs = gold["diff_pairs"]

    # Real candidate pool: every real "b" text across both same_pairs and diff_pairs (deduplicated
    # by id), so retrieval has real distractors, not a trivial 2-choice comparison.
    pool_texts, pool_ids, seen_ids = [], [], set()
    for p in same_pairs + diff_pairs:
        if p["id_b"] not in seen_ids:
            pool_texts.append(p["text_b"])
            pool_ids.append(p["id_b"])
            seen_ids.add(p["id_b"])

    t0 = time.time()
    pool_emb = model.encode(pool_texts, convert_to_numpy=True, show_progress_bar=False)
    query_texts = [p["text_a"] for p in same_pairs]
    query_emb = model.encode(query_texts, convert_to_numpy=True, show_progress_bar=False)
    encode_time = time.time() - t0
    print(f"Real encode time ({len(pool_texts)} pool + {len(query_texts)} query texts): "
          f"{encode_time:.1f}s")

    sims = cosine_sim(query_emb, pool_emb)  # [n_queries, n_pool]
    recall_at_1 = 0
    recall_at_3 = 0
    for i, p in enumerate(same_pairs):
        true_idx = pool_ids.index(p["id_b"])
        ranked = np.argsort(-sims[i])
        if ranked[0] == true_idx:
            recall_at_1 += 1
        if true_idx in ranked[:3]:
            recall_at_3 += 1
    n = len(same_pairs)

    # Real, direct same-vs-different score distributions (the input to the real threshold
    # decision later): use the DIAGONAL same-pair score (query vs. its own true match) and the
    # diff-pair score (query vs. a confirmed-different text), not the full retrieval matrix.
    same_scores = []
    for p in same_pairs:
        a_emb = model.encode([p["text_a"]], convert_to_numpy=True, show_progress_bar=False)
        b_emb = model.encode([p["text_b"]], convert_to_numpy=True, show_progress_bar=False)
        same_scores.append(float(cosine_sim(a_emb, b_emb)[0, 0]))

    diff_scores = []
    for p in diff_pairs:
        a_emb = model.encode([p["text_a"]], convert_to_numpy=True, show_progress_bar=False)
        b_emb = model.encode([p["text_b"]], convert_to_numpy=True, show_progress_bar=False)
        diff_scores.append(float(cosine_sim(a_emb, b_emb)[0, 0]))

    result = {
        "model_id": model_id,
        "load_time_s": load_time,
        "encode_time_s": encode_time,
        "n_queries": n,
        "n_pool": len(pool_texts),
        "recall_at_1": recall_at_1 / n,
        "recall_at_3": recall_at_3 / n,
        "same_score_mean": float(np.mean(same_scores)),
        "same_score_min": float(np.min(same_scores)),
        "diff_score_mean": float(np.mean(diff_scores)),
        "diff_score_max": float(np.max(diff_scores)),
        "separation_gap": float(np.min(same_scores) - np.max(diff_scores)),
        "same_scores": same_scores,
        "diff_scores": diff_scores,
    }
    print(f"recall@1={result['recall_at_1']:.3f} recall@3={result['recall_at_3']:.3f} "
          f"same_mean={result['same_score_mean']:.4f} diff_mean={result['diff_score_mean']:.4f} "
          f"separation_gap(min_same - max_diff)={result['separation_gap']:.4f}")
    return result


def main() -> None:
    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    print(f"Real gold set: {len(gold['same_pairs'])} same-pairs, {len(gold['diff_pairs'])} "
          f"diff-pairs")

    all_results = []
    for model_id in CANDIDATES:
        try:
            all_results.append(evaluate_model(model_id, gold))
        except Exception as exc:  # capture the failure, don't let one candidate's
            # crash silently drop it from the record
            print(f"REAL FAILURE loading/evaluating {model_id}: {exc!r}")
            all_results.append({"model_id": model_id, "error": repr(exc)})

    RESULTS_PATH.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nWrote real results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
