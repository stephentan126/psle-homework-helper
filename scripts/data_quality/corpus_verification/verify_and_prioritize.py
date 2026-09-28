r"""
Combine extraction-quality signals, set `verification_status` in psle.db, and rank the rows that
still need a person to check them (Issue 361).

Runs after check_consistency.py and reads its `consistency_results.json`; it does not recompute it.

Signals: in the live corpus (`psle.db`, 5,819 rows) `ocr_confidence` has two values, 'high' (4,452)
and 'low' (1,367), and `ocr_extraction_records.agreement_status` has 'agree' (4,452), 'disagree'
(730) and 'single_reader_only' (637). These are not independent: 730 + 637 = 1,367 exactly, because
`ocr_confidence='low'` is derived from `agreement_status != 'agree'`. They are treated as one
signal, not counted twice.

Prioritisation reuses the precomputed corpus embeddings (`question_embeddings`,
Qwen3-Embedding-0.6B, built by `scripts/extraction/precompute_question_embeddings.py`) as a rough
cluster-size proxy. For each row needing review, it counts the other corpus rows above a similarity
of 0.75. That is looser than `question_matching.MATCH_CONFIDENCE_THRESHOLD` (0.90), which is
calibrated for "the same question"; 0.75 aims at "the same question type". It is a rough proxy,
not a validated threshold.

Definition applied:
  - 'verified' (automated): the consistency check returned MATCH and ocr_confidence is 'high'.
  - 'needs_review': the check returned MISMATCH or UNPARSEABLE, or ocr_confidence is 'low'
    whatever the check said, since a low-confidence OCR read can still pass the numeric check
    by coincidence.
  - 'unverified' (unchanged): rows without both `answer_value` and `worked_solution_text`. The
    check never ran on them, so there is no automated basis to move them.

This writes to the live `psle.db`, so a timestamped backup is made first. Counts are printed and
saved to data/extracted/corpus_verification/verification_summary.json.

Usage: python scripts/data_quality/corpus_verification/verify_and_prioritize.py
"""
import json
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "backend"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_DB = _REPO_ROOT / "psle.db"
_OUT_DIR = _REPO_ROOT / "data" / "extracted" / "corpus_verification"
_SIMILARITY_FLOOR = 0.75  # looser than the 0.90 "same question" threshold: similar type


def main() -> None:
    consistency = json.load(open(_OUT_DIR / "consistency_results.json", encoding="utf-8"))

    con = sqlite3.connect(f"file:{_DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    all_rows = {str(r["id"]): dict(r) for r in con.execute(
        "select id, ocr_confidence, question_section, answer_value, worked_solution_text, superseded_by from questions")}
    agreement = {str(r["question_id"]): r["agreement_status"] for r in con.execute(
        "select question_id, agreement_status from ocr_extraction_records")}
    low_conf_ids = {k for k, r in all_rows.items() if r["ocr_confidence"] == "low"}
    derived_check = low_conf_ids == {k for k, s in agreement.items() if s != "agree"}
    print(f"=== Step 2: extraction-quality signals (real distributions, {len(all_rows)} total rows) ===")
    print(f"  ocr_confidence: high={sum(1 for r in all_rows.values() if r['ocr_confidence']=='high')}, low={len(low_conf_ids)}")
    from collections import Counter
    print(f"  agreement_status: {dict(Counter(agreement.values()))}")
    print(f"  ocr_confidence=='low' EXACTLY equals agreement_status!='agree': {derived_check} (one consolidated signal, not two)")

    # Classify every row that has both fields (the 1,262 rows the consistency check covered).
    verified, needs_review, reasons = [], [], {}
    for qid, res in consistency.items():
        r = all_rows[qid]
        low_conf = r["ocr_confidence"] == "low"
        if res["status"] == "MATCH" and not low_conf:
            verified.append(qid)
        else:
            needs_review.append(qid)
            why = []
            if res["status"] != "MATCH":
                why.append(f"consistency={res['status']}")
            if low_conf:
                why.append("ocr_confidence=low")
            reasons[qid] = ", ".join(why)
    print(f"\n=== Step 4: operational verification_status ===")
    print(f"  eligible for the automated check (both fields present): {len(consistency)}")
    print(f"  -> VERIFIED (automated, no human needed): {len(verified)}")
    print(f"  -> NEEDS_REVIEW: {len(needs_review)}")
    still_unverified = [k for k, r in all_rows.items() if k not in consistency and not r["superseded_by"]]
    print(f"  left as UNVERIFIED (no answer_value+worked_solution_text pair to check): {len(still_unverified)}")

    # Rough cluster-size proxy for the needs_review rows, used only to prioritise them.
    print(f"\n=== Step 3: cluster-size proxy (Qwen3-Embedding-0.6B, similarity > {_SIMILARITY_FLOOR}, precomputed embeddings reused) ===")
    emb_rows = con.execute("select question_id, embedding_json from question_embeddings").fetchall()
    ids = [str(r["question_id"]) for r in emb_rows]
    idx = {qid: i for i, qid in enumerate(ids)}
    mat = np.array([json.loads(r["embedding_json"]) for r in emb_rows], dtype=np.float32)
    mat_norm = mat / np.linalg.norm(mat, axis=1, keepdims=True)
    sims_available = [qid for qid in needs_review if qid in idx]
    print(f"  {len(sims_available)}/{len(needs_review)} needs_review rows have a precomputed embedding")
    cluster_size = {}
    for qid in sims_available:
        sims = mat_norm @ mat_norm[idx[qid]]
        cluster_size[qid] = int((sims > _SIMILARITY_FLOOR).sum()) - 1  # exclude the row itself
    prioritized = sorted(sims_available, key=lambda q: -cluster_size[q])
    print("  top 10 highest-priority needs_review rows (largest estimated similar-question cluster):")
    for qid in prioritized[:10]:
        print(f"    id={qid} cluster_size~{cluster_size[qid]:3d} section={all_rows[qid]['question_section']!r:14s} reason={reasons[qid]}")

    # Write the update to the live database, after taking a backup.
    backup = _DB.with_name(f"psle.db.bak_verify_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}")
    shutil.copy2(_DB, backup)
    print(f"\nBacked up {_DB.name} -> {backup.name} before writing")
    wcon = sqlite3.connect(str(_DB))
    wcon.executemany("update questions set verification_status='verified' where id=?", [(int(q),) for q in verified])
    wcon.executemany("update questions set verification_status='needs_review' where id=?", [(int(q),) for q in needs_review])
    wcon.commit()
    real_counts = dict(wcon.execute("select verification_status, count(*) from questions group by 1").fetchall())
    wcon.close()
    print(f"\n=== Step 5: real post-write verification_status counts in psle.db ===\n  {real_counts}")

    summary = {"total_rows": len(all_rows), "eligible_for_step1": len(consistency),
              "verified_count": len(verified), "needs_review_count": len(needs_review),
              "still_unverified_count": len(still_unverified), "post_write_status_counts": real_counts,
              "cluster_priority_top20": [{"id": q, "cluster_size": cluster_size[q], "reason": reasons[q]} for q in prioritized[:20]],
              "needs_review_reasons": reasons, "backup_file": backup.name}
    (_OUT_DIR / "verification_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print("Saved ->", _OUT_DIR / "verification_summary.json")


if __name__ == "__main__":
    main()
