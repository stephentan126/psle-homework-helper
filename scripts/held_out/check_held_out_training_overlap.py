r"""
Check whether any held-out question overlaps in content with the training pool (Issues 27, 252).

Run after precompute_held_out_question_embeddings.py. Computes the full held-out by training-pool
cosine-similarity matrix in one matrix multiplication, in the same embedding space
(Qwen/Qwen3-Embedding-0.6B) as the corpus dedup pass in dedup.py.

The same exclusions as `dedup.load_corpus_embedding_matrix()` are applied to both sides first:
NON_QUESTION_EXTRACTION_FLAG rows and `_is_boilerplate()` text. Without them, exam instructions
("For questions which require units...", "The use of calculators is NOT allowed.") fill the top
of the candidate list without being content overlaps.

REVIEW_FLOOR is 0.90, the "worth a closer look" threshold already validated for this corpus and
model (Issues 186, 231). The higher AUTO_MARK_THRESHOLD is not used, because this script only
reports. The three secondary signals from dedup.py (_distinguishing_terms_conflict,
_numeric_terms_conflict, _numeric_multiset_conflict) are recorded per pair as supporting evidence,
never used as a filter (Issues 231, 250). A pair that trips a signal is very likely a different
question. A pair that trips none is not necessarily a duplicate: Issue 250 found that short,
diagram-dependent multiple-choice stems can score up to 1.0 while being different questions,
because the distinguishing content is in the diagram. Only comparing the page images settles a
pair, which this script does not do; it lists candidates with ids on both sides.

Output: data/extracted/held_out_training_overlap_candidates.json, sorted by similarity.

Usage:
    backend/.venv/Scripts/python.exe scripts/held_out/check_held_out_training_overlap.py
"""
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "backend"))

from app.pipeline.question_matching import (  # noqa: E402
    EMBEDDING_MODEL_ID,
    _distinguishing_terms_conflict,
    _numeric_terms_conflict,
)
from app.pipeline.dedup import _numeric_multiset_conflict, _is_boilerplate  # noqa: E402
from app.models.question import NON_QUESTION_EXTRACTION_FLAG  # noqa: E402

REVIEW_FLOOR = 0.90
HELD_OUT_DB = _REPO_ROOT / "data" / "extracted" / "held_out.db"
TRAINING_DB = _REPO_ROOT / "psle.db"
OUT_PATH = _REPO_ROOT / "data" / "extracted" / "held_out_training_overlap_candidates.json"


def load(db_path: Path):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(
        "SELECT e.question_id, q.question_text, q.source_paper, q.source_page_index, "
        "q.question_number, q.extraction_flag, e.embedding_json "
        "FROM question_embeddings e JOIN questions q ON q.id = e.question_id "
        "WHERE e.embedding_model = ? ORDER BY e.question_id",
        (EMBEDDING_MODEL_ID,),
    )
    rows = cur.fetchall()
    con.close()
    return rows


def real_filter(rows):
    out, n_flagged, n_boiler = [], 0, 0
    for qid, text, sp, spi, qn, flag, emb in rows:
        if flag and NON_QUESTION_EXTRACTION_FLAG in flag:
            n_flagged += 1
            continue
        if text and _is_boilerplate(text):
            n_boiler += 1
            continue
        out.append((qid, text, sp, spi, qn, emb))
    return out, n_flagged, n_boiler


def main() -> None:
    ho_rows_raw = load(HELD_OUT_DB)
    tp_rows_raw = load(TRAINING_DB)
    print(f"Held-out raw rows: {len(ho_rows_raw)}")
    print(f"Training-pool raw rows: {len(tp_rows_raw)}")

    ho_rows, ho_flagged, ho_boiler = real_filter(ho_rows_raw)
    tp_rows, tp_flagged, tp_boiler = real_filter(tp_rows_raw)
    print(f"Held-out: excluded {ho_flagged} NON_QUESTION-flagged, {ho_boiler} boilerplate-prefix "
          f"-> {len(ho_rows)} real rows compared")
    print(f"Training pool: excluded {tp_flagged} NON_QUESTION-flagged, {tp_boiler} boilerplate-prefix "
          f"-> {len(tp_rows)} real rows compared")

    ho_ids = [r[0] for r in ho_rows]
    ho_text = {r[0]: r[1] for r in ho_rows}
    ho_meta = {r[0]: (r[2], r[3], r[4]) for r in ho_rows}
    ho_matrix = np.array([json.loads(r[5]) for r in ho_rows], dtype=np.float32)

    tp_ids = [r[0] for r in tp_rows]
    tp_text = {r[0]: r[1] for r in tp_rows}
    tp_meta = {r[0]: (r[2], r[3], r[4]) for r in tp_rows}
    tp_matrix = np.array([json.loads(r[5]) for r in tp_rows], dtype=np.float32)

    ho_norm = ho_matrix / np.linalg.norm(ho_matrix, axis=1, keepdims=True)
    tp_norm = tp_matrix / np.linalg.norm(tp_matrix, axis=1, keepdims=True)
    sim = ho_norm @ tp_norm.T
    print(f"Real cross-similarity matrix shape: {sim.shape}")

    mask = sim >= REVIEW_FLOOR
    idx_ho, idx_tp = np.where(mask)
    candidates = []
    for i, j in zip(idx_ho.tolist(), idx_tp.tolist()):
        ho_id, tp_id, score = ho_ids[i], tp_ids[j], float(sim[i, j])
        ho_t, tp_t = ho_text[ho_id] or "", tp_text[tp_id] or ""
        dist_c = _distinguishing_terms_conflict(ho_t, tp_t)
        num_c = _numeric_terms_conflict(ho_t, tp_t)
        multi_c = _numeric_multiset_conflict(ho_t, tp_t)
        candidates.append({
            "held_out_id": ho_id, "held_out_meta": ho_meta[ho_id], "held_out_text": ho_t,
            "training_pool_id": tp_id, "training_pool_meta": tp_meta[tp_id],
            "training_pool_text": tp_t, "similarity": score,
            "distinguishing_terms_conflict": dist_c, "numeric_terms_conflict": num_c,
            "numeric_multiset_conflict": multi_c,
            "any_secondary_conflict": dist_c or num_c or multi_c,
        })
    candidates.sort(key=lambda c: -c["similarity"])

    OUT_PATH.write_text(json.dumps(candidates, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(candidates)} real candidate(s) to {OUT_PATH}")

    n_no_conflict = sum(1 for c in candidates if not c["any_secondary_conflict"])
    print(f"\n=== SUMMARY ===")
    print(f"Total candidate pairs >= {REVIEW_FLOOR}: {len(candidates)}")
    print(f"No secondary conflict tripped (higher real risk): {n_no_conflict}")


if __name__ == "__main__":
    main()
