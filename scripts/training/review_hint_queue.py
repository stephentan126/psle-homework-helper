r"""
Interactive command-line review of a hint review-queue JSONL written by
`scripts/training/build_hint_review_queue.py`.

Read-only against `psle.db`: it looks up `question_number` and `source_page_index` for display,
since the queue file does not carry them. The only files written are the queue JSONL (rewritten in
place after every row) and a small timing sidecar. No GPU, no training.

For each pending row it shows the question_id, tier, source paper, page and question number, the
full `question_text`, the `bootstrapped_hint`, the `answer_value` (raw, plus the gate-normalised
value checked if different, Issue 291) and the `worked_solution_text`, so grounding can be judged
as well as readability. For a Tier 2 or higher row with `has_diagram=1` it shows the
`source_page_image` path so the figure can be checked. A Tier 1 row, or a diagram row with no
image field, says why no image is shown: a Tier 1 restatement never refers to figure or solution
content.

Actions per row:
  [a]ccept  review_status="accepted"; reviewed_hint is the bootstrapped hint unchanged.
  [e]dit    enter replacement text over several lines; review_status="edited" and reviewed_hint
            is the edited text.
  [r]eject  a short reason is required. Use this when the question itself is broken (a
            truncated stem, questions glued together, a wrong answer key, as with ids 23 and 77
            in the 25-row trial), not for the hint's wording. reviewed_hint stays null and the
            row is excluded from training.
  [s]kip    leave for a later pass; review_status="skipped", reported separately from rows not
            yet seen ("pending").
  [q]uit    stop now. Every decision made so far is already saved.

Resumable and crash-safe: after every decision the whole queue is written to a temporary file and
atomically swapped in, so closing the terminal cannot leave a partial file. Re-running on the same
`--queue` path continues where it stopped: pending and skipped rows are shown again, while
accepted, edited and rejected rows are not.

Timing: the session's active time (from the first row shown to exit) is printed at the end and
appended to `<queue path>.review_timing.json`, so the review rate is visible across many short
sessions.

Usage:
    backend/.venv/Scripts/python.exe scripts/training/review_hint_queue.py \
        --queue data/extracted/slice6_step0c/hint_review_queue_20260911_022226.jsonl
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

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_STATUS_PENDING = "pending"
_STATUS_ACCEPTED = "accepted"
_STATUS_EDITED = "edited"
_STATUS_REJECTED = "rejected"
_STATUS_SKIPPED = "skipped"
_DONE_STATUSES = {_STATUS_ACCEPTED, _STATUS_EDITED, _STATUS_REJECTED}
_ALL_STATUSES = [_STATUS_PENDING, _STATUS_ACCEPTED, _STATUS_EDITED, _STATUS_REJECTED, _STATUS_SKIPPED]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_queue(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    for r in rows:
        r.setdefault("review_status", _STATUS_PENDING)
        r.setdefault("review_notes", None)
        r.setdefault("reviewed_hint", None)
        r.setdefault("reviewed_at", None)
    return rows


def save_queue(path: Path, rows: list[dict]) -> None:
    """Rewrite the queue atomically: write a sibling temp file, then replace the original.

    A kill mid-write never leaves a partial file on disk.
    """
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def lookup_db_meta(question_ids: list[int]) -> dict[int, dict]:
    """Look up question_number and source_page_index for display (read-only).

    The review-queue JSONL does not carry these two fields.
    """
    if not question_ids:
        return {}
    from sqlalchemy import text

    from app.db.session import get_session, init_db

    init_db()
    ids = sorted(set(question_ids))
    with get_session() as s:
        rows = s.execute(text(
            f"SELECT id, question_number, source_page_index FROM questions "
            f"WHERE id IN ({','.join(str(i) for i in ids)})"
        )).all()
    return {
        r.id: {"question_number": r.question_number, "source_page_index": r.source_page_index}
        for r in rows
    }


def _read_multiline(label: str) -> str:
    print(f"  {label}")
    print("  (type the replacement hint; finish with a single '.' on its own line)")
    lines: list[str] = []
    while True:
        try:
            line = input("  > ")
        except EOFError:
            break
        if line.strip() == ".":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def _print_row(shown_idx: int, shown_total: int, row: dict, meta: dict) -> None:
    paper = Path(row.get("source_paper", "") or "").name
    qnum = meta.get("question_number")
    page = meta.get("source_page_index")
    print("\n" + "=" * 100)
    print(f"[{shown_idx}/{shown_total}]  question_id={row['question_id']}   tier={row['tier']}"
          f"   status={row.get('review_status')}")
    print(f"  source: {paper}   page={page}   q#={qnum!r}   has_diagram={row.get('has_diagram')}")
    print("-" * 100)
    print("QUESTION TEXT:")
    print(row.get("question_text", ""))
    print("-" * 100)
    av, av_gate = row.get("answer_value"), row.get("answer_value_gate")
    if av_gate is not None and av_gate != av:
        print(f"answer_value: {av!r}   (gate used {av_gate!r} — normalization: "
              f"{row.get('answer_value_normalization_rules')})")
    else:
        print(f"answer_value: {av!r}")
    ws = row.get("worked_solution_text")
    print(f"worked_solution_text: {ws!r}" if ws else "worked_solution_text: (none recorded)")
    print(f"gate_passed={row.get('gate_passed')}   gate_reason: {row.get('gate_reason')}")
    print("-" * 100)
    grounded = row.get("grounded_in")
    print(f"BOOTSTRAPPED HINT (Tier {row['tier']}{f', grounded_in={grounded}' if grounded else ''}):")
    print(row.get("bootstrapped_hint", ""))
    print("-" * 100)
    img = row.get("source_page_image")
    if img:
        print(f"[diagram] source_page_image: {img}   "
              f"(exists on disk: {row.get('source_page_image_exists')})")
        print("          -> open this file yourself and confirm the hint above doesn't")
        print("             reference anything only visible in the figure.")
    elif row.get("has_diagram"):
        print("[diagram] has_diagram=True but no source_page_image on this row "
              "(Tier 1 hints never reference figure content, so none was attached).")
    print("=" * 100)


def counts_by_status(rows: list[dict]) -> dict[str, int]:
    c = {k: 0 for k in _ALL_STATUSES}
    for r in rows:
        c[r.get("review_status", _STATUS_PENDING)] = c.get(r.get("review_status", _STATUS_PENDING), 0) + 1
    return c


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue", required=True, help="path to a bootstrap review-queue JSONL")
    args = ap.parse_args()

    path = Path(args.queue)
    if not path.exists():
        print(f"*** No such file: {path} ***")
        sys.exit(1)

    rows = load_queue(path)
    meta = lookup_db_meta([r["question_id"] for r in rows])

    print(f"Loaded {len(rows)} row(s) from {path.name}")
    print(f"  current state: {counts_by_status(rows)}")

    pending_idx = [
        i for i, r in enumerate(rows)
        if r.get("review_status", _STATUS_PENDING) not in _DONE_STATUSES
    ]
    if not pending_idx:
        print("\nNothing left to review — every row is accepted/edited/rejected.")
        return

    print(f"\n{len(pending_idx)} row(s) to review this session (pending + skipped).\n")

    session_start = time.time()
    session_reviewed = 0
    for shown, i in enumerate(pending_idx, 1):
        row = rows[i]
        _print_row(shown, len(pending_idx), row, meta.get(row["question_id"], {}))

        while True:
            try:
                choice = input(
                    "Action? [a]ccept  [e]dit  [r]eject  [s]kip  [q]uit-and-save > ",
                ).strip().lower()
            except EOFError:
                choice = "q"
            if choice in ("a", "accept"):
                try:
                    note = input("  optional note (Enter to skip): ").strip()
                except EOFError:
                    note = ""
                row.update(review_status=_STATUS_ACCEPTED, reviewed=True,
                           reviewed_hint=row.get("bootstrapped_hint"),
                           review_notes=note or None, reviewed_at=_now_iso())
                session_reviewed += 1
                break
            if choice in ("e", "edit"):
                edited = _read_multiline("Enter the replacement hint:")
                if not edited:
                    print("  (empty edit — nothing saved; choose an action again)")
                    continue
                try:
                    note = input("  optional note (Enter to skip): ").strip()
                except EOFError:
                    note = ""
                row.update(review_status=_STATUS_EDITED, reviewed=True,
                           reviewed_hint=edited, review_notes=note or None,
                           reviewed_at=_now_iso())
                session_reviewed += 1
                break
            if choice in ("r", "reject"):
                reason = ""
                while not reason:
                    try:
                        reason = input("  reason (required) > ").strip()
                    except EOFError:
                        reason = "(no reason given — EOF)"
                        break
                row.update(review_status=_STATUS_REJECTED, reviewed=True,
                           reviewed_hint=None, review_notes=reason, reviewed_at=_now_iso())
                session_reviewed += 1
                break
            if choice in ("s", "skip"):
                row["review_status"] = _STATUS_SKIPPED
                break
            if choice in ("q", "quit"):
                break
            print("  not understood — enter a, e, r, s, or q.")

        save_queue(path, rows)  # save after every decision, so an interruption loses nothing
        if choice in ("q", "quit"):
            break

    session_elapsed = time.time() - session_start

    timing_path = path.with_name(path.name + ".review_timing.json")
    timing = {"total_review_seconds": 0.0, "total_rows_reviewed": 0, "sessions": []}
    if timing_path.exists():
        try:
            timing = json.loads(timing_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    timing["sessions"].append({
        "started_at": datetime.fromtimestamp(session_start, tz=timezone.utc).isoformat(),
        "ended_at": _now_iso(),
        "seconds": round(session_elapsed, 1),
        "rows_reviewed": session_reviewed,
    })
    timing["total_review_seconds"] = round(timing.get("total_review_seconds", 0.0) + session_elapsed, 1)
    timing["total_rows_reviewed"] = timing.get("total_rows_reviewed", 0) + session_reviewed
    timing_path.write_text(json.dumps(timing, indent=2, ensure_ascii=False), encoding="utf-8")

    final_counts = counts_by_status(rows)
    per_row = (session_elapsed / session_reviewed) if session_reviewed else 0.0
    cum_seconds, cum_rows = timing["total_review_seconds"], timing["total_rows_reviewed"]
    cum_per_row = (cum_seconds / cum_rows) if cum_rows else 0.0

    print("\n" + "=" * 100)
    print("SESSION SUMMARY")
    print(f"  total rows in queue: {len(rows)}")
    print(f"  status now: {final_counts}")
    print(f"  this session: {session_reviewed} row(s) decided "
          f"({session_elapsed:.1f}s wall-clock" + (f", {per_row:.1f}s/row)" if session_reviewed else ")"))
    print(f"  cumulative across all sessions ({timing_path.name}): "
          f"{cum_seconds:.1f}s over {cum_rows} row(s)"
          + (f" ({cum_per_row:.1f}s/row avg)" if cum_rows else ""))
    print("=" * 100)


if __name__ == "__main__":
    main()
