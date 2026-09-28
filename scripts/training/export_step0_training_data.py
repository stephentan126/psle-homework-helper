r"""
Export the QLoRA training data from the closed hint review pool.

Reads data/extracted/slice6_step0c/hint_review_queue_full_pool.jsonl (1,416 rows as of Issue 307)
and writes one training example per line to data/extracted/slice6_step0c/training_export_v1.jsonl.
Run after review_hint_queue.py and confirm_hint_prescreen.py have closed the pool, and before
training.

This covers the QLoRA condition only. The RAG and few-shot comparison conditions are handled by the
grounding evaluation scripts. The script does not train anything.

There is no psle.db write to put behind an --apply flag, so the "preview" is the per-step counts
and a 5-row sample printed before the file is written. Re-running is safe: the output is fully
regenerated from the queue file each time.

Filter order (each step's surviving count is printed):
  1. Exclude rows whose source_paper is one of the 7 held-out SA2 2025 papers, matched by exact
     basename (case-insensitive) against the files under data/Hold out/200/. A substring or
     school-name match would wrongly exclude other years' papers from the same 7 schools; 49 such
     PDFs exist in the pool. For example, data/School papers/2021/P6-Maths-2021-SA2-Nan-Hua.pdf is
     a valid training row, while data/Hold out/200/SAP/P6_Maths_2025_SA2_nanhua.pdf is held out.
  2. Keep only review_status in {accepted, edited}.
  3. Keep only rows whose reviewed_hint is non-empty after stripping.

Usage:
    backend/.venv/Scripts/python.exe scripts/training/export_step0_training_data.py
"""
from __future__ import annotations

import json
import ntpath
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_QUEUE_PATH = _REPO_ROOT / "data" / "extracted" / "slice6_step0c" / "hint_review_queue_full_pool.jsonl"
_OUT_PATH = _REPO_ROOT / "data" / "extracted" / "slice6_step0c" / "training_export_v1.jsonl"

# Exact basenames of the 7 held-out SA2 2025 papers under data/Hold out/200/, taken from that
# folder's contents rather than from school names.
_HELD_OUT_BASENAMES = {
    "p6_maths_2025_sa2_henrypark.pdf",   # Hold out/200/Non_SAP
    "p6_maths_2025_sa2_raffles.pdf",     # Hold out/200/Non_SAP
    "p6_maths_2025_sa2_rosyth.pdf",      # Hold out/200/Non_SAP
    "p6_maths_2025_sa2_catholichigh.pdf",  # Hold out/200/SAP
    "p6_maths_2025_sa2_nanhua.pdf",      # Hold out/200/SAP
    "p6_maths_2025_sa2_nanyang.pdf",     # Hold out/200/SAP
    "p6_maths_2025_sa2_taonan.pdf",      # Hold out/200/SAP
}

_KEEP_STATUSES = {"accepted", "edited"}


def is_held_out(source_paper: str | None) -> bool:
    if not source_paper:
        return False
    return ntpath.basename(source_paper).lower() in _HELD_OUT_BASENAMES


def main() -> None:
    if not _QUEUE_PATH.exists():
        print(f"*** No such file: {_QUEUE_PATH} ***")
        sys.exit(1)

    rows: list[dict] = []
    with _QUEUE_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    step0 = len(rows)
    print(f"step 0  -- loaded from {_QUEUE_PATH.name}                         : {step0}")

    after_held_out = [r for r in rows if not is_held_out(r.get("source_paper"))]
    excluded_held_out = [r for r in rows if is_held_out(r.get("source_paper"))]
    print(f"step 1  -- exclude the 7 held-out SA2 2025 papers (exact basename): {len(after_held_out)}"
          f"   (-{len(rows) - len(after_held_out)})")
    if excluded_held_out:
        print("  *** excluded rows (unexpected -- report these ids explicitly, do not proceed silently):")
        for r in excluded_held_out:
            print(f"      question_id={r.get('question_id')} tier={r.get('tier')} "
                  f"source_paper={r.get('source_paper')!r}")

    after_status = [r for r in after_held_out if r.get("review_status") in _KEEP_STATUSES]
    print(f"step 2  -- keep review_status in {{accepted, edited}}             : {len(after_status)}"
          f"   (-{len(after_held_out) - len(after_status)})")

    after_hint = [r for r in after_status if (r.get("reviewed_hint") or "").strip()]
    print(f"step 3  -- keep non-empty (stripped) reviewed_hint                : {len(after_hint)}"
          f"   (-{len(after_status) - len(after_hint)})")

    print(f"\nFINAL row count: {len(after_hint)}")

    out_records = []
    for r in after_hint:
        out_records.append({
            "question_id": r["question_id"],
            "tier": r["tier"],
            "messages": [
                {"role": "system", "content": r.get("system_prompt")},
                {"role": "user", "content": r.get("question_text")},
                {"role": "assistant", "content": r.get("reviewed_hint").strip()},
            ],
            "meta": {
                "answer_value": r.get("answer_value"),
                "question_type": r.get("question_type"),
                "answer_value_class": r.get("answer_value_class"),
                "has_diagram": r.get("has_diagram"),
                "extraction_flag": r.get("extraction_flag"),
                "source_paper": r.get("source_paper"),
                "review_status": r.get("review_status"),
                "confirmed_via": r.get("confirmed_via"),
            },
        })

    # Final check, independent of the filter counts: no output row may reference a held-out paper.
    held_out_leaks = [rec for rec in out_records if is_held_out(rec["meta"]["source_paper"])]
    print(f"\nASSERTION -- output rows with a held-out source_paper: {len(held_out_leaks)} "
          f"(must be 0)")
    if held_out_leaks:
        print("*** REFUSING TO WRITE -- held-out leak detected in final output ***")
        for rec in held_out_leaks:
            print(f"    question_id={rec['question_id']} tier={rec['tier']} "
                  f"source_paper={rec['meta']['source_paper']!r}")
        sys.exit(1)

    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _OUT_PATH.open("w", encoding="utf-8") as fh:
        for rec in out_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nWROTE {len(out_records)} row(s) to {_OUT_PATH}")

    print("\n=== random 5-row sample for spot-check ===")
    random.seed(20260914)
    sample = random.sample(out_records, min(5, len(out_records)))
    for rec in sample:
        print(json.dumps(rec, ensure_ascii=False, indent=2))
        print("-" * 100)


if __name__ == "__main__":
    main()
