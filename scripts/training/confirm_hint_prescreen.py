r"""
Confirm mode for a hint review-queue JSONL: fast human sign-off on a prescreen, not a replacement
for human review.

A prescreen is a structured first pass over every bootstrapped hint against the same rubric as
the human review (tier-appropriateness, groundedness, no leakage, unit notation, restating shared
context), plus the two checks added in Issue 296 (glued questions, mismatched ground truth). It
gives a verdict per `(question_id, tier)`, but a prescreen verdict is not a review. A row's
`review_status`, `reviewed_hint` and `reviewed_at` change only when a person presses a key in this
tool, never because a prescreen file says so. This matters for the methodology (Issue 291's hybrid
design: bootstrap, then review; never bootstrap and assume).

It reuses the queue load, save, lookup and status code from `review_hint_queue.py`: the same JSONL,
the same atomic rewrite after every decision, the same resumability, and the same
`<queue>.review_timing.json` sidecar. Sessions here add to the same cumulative timer but are
tagged as "confirm" mode, so from-scratch review time and confirm time can be reported
separately; the two are not comparable.

Prescreen file format (JSON or JSONL), one entry per row:
    {"question_id": <int>, "tier": <int>, "verdict": "accept"|"edit"|"reject"|"uncertain",
     "reason": "<one-line reason>", "suggested_edit": "<text>"}   # suggested_edit only for "edit"
Entries are keyed by `(question_id, tier)`. Queue rows with no prescreen entry are not shown here;
use `review_hint_queue.py` for those.

Each row shows the same data as `review_hint_queue.py` (question_text, answer_value,
worked_solution_text, gate information, the bootstrapped_hint and any diagram image), plus the
prescreen verdict, reason and suggested edit, displayed prominently so the reviewer is checking a
stated claim rather than reading the hint cold.

Actions:
  [c]onfirm  apply the prescreen verdict as the reviewer's decision:
               accept -> review_status="accepted", reviewed_hint = bootstrapped_hint.
               edit   -> review_status="edited", reviewed_hint = the suggested_edit (entries
                         without one are skipped when the file is loaded).
               reject -> review_status="rejected", reviewed_hint = None,
                         review_notes = the prescreen reason.
             Not offered for verdict="uncertain": there is nothing to confirm, so the reviewer
             must edit, reject or skip, as in ordinary review.
  [e]dit     override: the reviewer types their own replacement text, ignoring any
             suggested_edit. review_status="edited" and `overridden_prescreen=True`.
  [r]eject   override: the reviewer types their own reason. review_status="rejected" and
             `overridden_prescreen=True`.
  [s]kip     leave for later.
  [q]uit     stop now; every decision made so far is already saved.
Bulk confirm: after a row with a confirmable verdict is decided, the tool offers to confirm the
following run of rows with the same verdict (never "uncertain"). The reviewer has just checked
one row and chooses to trust the prescreen for the rest of that run. It shows the count and the
ids first, and a smaller number can be entered. Bulk-confirmed rows get `confirmed_via="bulk"`
(otherwise `"individual"`), so the saved queue records how many rows were read individually and
how many were approved in bulk after a spot check.

Usage:
    backend/.venv/Scripts/python.exe scripts/training/confirm_hint_prescreen.py \
        --queue data/extracted/slice6_step0c/hint_review_queue_full_pool.jsonl \
        --prescreen data/extracted/slice6_step0c/prescreen_465.json
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
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "training"))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from review_hint_queue import (  # noqa: E402
    _DONE_STATUSES,
    _STATUS_ACCEPTED,
    _STATUS_EDITED,
    _STATUS_PENDING,
    _STATUS_REJECTED,
    _STATUS_SKIPPED,
    _now_iso,
    counts_by_status,
    load_queue,
    lookup_db_meta,
    save_queue,
)

_VERDICTS = {"accept", "edit", "reject", "uncertain"}


def load_prescreen(path: Path) -> dict[tuple[int, int], dict]:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
        entries = data if isinstance(data, list) else data.get("entries", data.get("rows", []))
    except json.JSONDecodeError:
        entries = [json.loads(line) for line in text.splitlines() if line.strip()]

    by_key: dict[tuple[int, int], dict] = {}
    bad = 0
    for e in entries:
        try:
            qid, tier, verdict = int(e["question_id"]), int(e["tier"]), e["verdict"]
        except (KeyError, TypeError, ValueError):
            bad += 1
            continue
        if verdict not in _VERDICTS:
            bad += 1
            continue
        if verdict == "edit" and not e.get("suggested_edit"):
            bad += 1
            continue
        by_key[(qid, tier)] = {
            "verdict": verdict,
            "reason": e.get("reason", ""),
            "suggested_edit": e.get("suggested_edit"),
            "confidence": e.get("confidence"),
        }
    if bad:
        print(f"  ({bad} prescreen entr(y/ies) skipped — missing/invalid question_id, tier, "
              f"verdict, or an 'edit' verdict with no suggested_edit)")
    return by_key


def _print_row_with_prescreen(shown: int, total: int, row: dict, meta: dict, pre: dict) -> None:
    paper = Path(row.get("source_paper", "") or "").name
    print("\n" + "=" * 100)
    print(f"[{shown}/{total}]  question_id={row['question_id']}   tier={row['tier']}"
          f"   status={row.get('review_status')}")
    print(f"  source: {paper}   page={meta.get('source_page_index')}   "
          f"q#={meta.get('question_number')!r}   has_diagram={row.get('has_diagram')}")
    print("-" * 100)
    print("QUESTION TEXT:")
    print(row.get("question_text", ""))
    print("-" * 100)
    av, av_gate = row.get("answer_value"), row.get("answer_value_gate")
    if av_gate is not None and av_gate != av:
        print(f"answer_value: {av!r}   (gate used {av_gate!r})")
    else:
        print(f"answer_value: {av!r}")
    ws = row.get("worked_solution_text")
    print(f"worked_solution_text: {ws!r}" if ws else "worked_solution_text: (none recorded)")
    print("-" * 100)
    print(f"BOOTSTRAPPED HINT (Tier {row['tier']}):")
    print(row.get("bootstrapped_hint", ""))
    img = row.get("source_page_image")
    if img:
        print(f"[diagram] source_page_image: {img}  (exists: {row.get('source_page_image_exists')})")
    print("=" * 100)
    print(f">>> PRESCREEN VERDICT: {pre['verdict'].upper()}")
    print(f">>> reason: {pre['reason']}")
    if pre["verdict"] == "edit":
        print(">>> suggested_edit:")
        print(pre["suggested_edit"])
    print("=" * 100)


def _apply_confirm(row: dict, pre: dict, confirmed_via: str) -> None:
    v = pre["verdict"]
    if v == "accept":
        row.update(review_status=_STATUS_ACCEPTED, reviewed=True,
                    reviewed_hint=row.get("bootstrapped_hint"), review_notes=pre["reason"] or None)
    elif v == "edit":
        row.update(review_status=_STATUS_EDITED, reviewed=True,
                    reviewed_hint=pre["suggested_edit"], review_notes=pre["reason"] or None)
    elif v == "reject":
        row.update(review_status=_STATUS_REJECTED, reviewed=True,
                    reviewed_hint=None, review_notes=pre["reason"])
    else:
        raise ValueError(f"cannot confirm verdict={v!r}")
    row["reviewed_at"] = _now_iso()
    row["confirmed_via"] = confirmed_via
    row["overridden_prescreen"] = False


_VERDICT_PRIORITY = {"reject": 0, "edit": 1, "uncertain": 2, "accept": 3}
# Secondary sort key for the edit verdict only (Issue 303). High-confidence edits, whose
# suggested_edit was checked by hand against the facts the reason cites, come before
# low-confidence ones (method leakage, vagueness or factual contradiction), which need a normal
# individual read. A missing or unrecognised confidence counts as low, so nothing is assumed
# checked. The reject, uncertain and accept blocks are not affected.
_EDIT_CONFIDENCE_PRIORITY = {"high": 0, "low": 1}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue", required=True, help="path to a bootstrap review-queue JSONL")
    ap.add_argument("--prescreen", required=True, help="path to the prescreen JSON/JSONL file")
    ap.add_argument(
        "--order", choices=("file", "verdict"), default="file",
        help="row order for this confirm session. 'file' (default): the queue's own row order, "
             "unchanged. 'verdict': reject, then edit, then uncertain, then accept — surfaces the "
             "rows most likely to need real human attention (rejects, then edits) before the "
             "large, usually-safe accept run at the end, which is also where bulk-confirm is "
             "most useful. Within the edit block only (Issue 303), a secondary key sorts "
             "high-confidence edits (already manually verified) before low-confidence/unspecified "
             "ones, so the fast bulk-confirmable stretch comes first. A stable sort otherwise: "
             "relative order within each verdict/confidence group is preserved from the "
             "underlying queue file, not re-randomized.",
    )
    args = ap.parse_args()

    qpath = Path(args.queue)
    if not qpath.exists():
        print(f"*** No such file: {qpath} ***")
        sys.exit(1)
    ppath = Path(args.prescreen)
    if not ppath.exists():
        print(f"*** No such file: {ppath} ***")
        sys.exit(1)

    rows = load_queue(qpath)
    for r in rows:
        r.setdefault("prescreen_verdict", None)
        r.setdefault("prescreen_reason", None)
        r.setdefault("prescreen_suggested_edit", None)
        r.setdefault("prescreen_confidence", None)
        r.setdefault("confirmed_via", None)
        r.setdefault("overridden_prescreen", None)

    prescreen = load_prescreen(ppath)
    meta = lookup_db_meta([r["question_id"] for r in rows])

    print(f"Loaded {len(rows)} row(s) from {qpath.name}; {len(prescreen)} prescreen entr(y/ies) from {ppath.name}")
    print(f"  current queue state: {counts_by_status(rows)}")

    # Tag every row with its prescreen fields, visible for audit whatever its review_status.
    matched = 0
    for r in rows:
        pre = prescreen.get((r["question_id"], r["tier"]))
        if pre:
            r["prescreen_verdict"] = pre["verdict"]
            r["prescreen_reason"] = pre["reason"]
            r["prescreen_suggested_edit"] = pre["suggested_edit"]
            r["prescreen_confidence"] = pre.get("confidence")
            matched += 1
    save_queue(qpath, rows)
    print(f"  {matched} queue row(s) matched to a prescreen entry (tagged, NOT marked reviewed)")

    pending_idx = [
        i for i, r in enumerate(rows)
        if r.get("review_status", _STATUS_PENDING) not in _DONE_STATUSES and r.get("prescreen_verdict")
    ]
    if not pending_idx:
        print("\nNothing left to confirm — every prescreened row is already accepted/edited/rejected, "
              "or no queue row matches a prescreen entry.")
        return

    def _verdict_sort_key(i: int) -> tuple[int, int]:
        verdict = rows[i]["prescreen_verdict"]
        vp = _VERDICT_PRIORITY.get(verdict, 99)
        # Confidence only breaks ties within the edit block. Every other verdict gets a constant
        # 0, so the stable sort keeps their relative order.
        cp = _EDIT_CONFIDENCE_PRIORITY.get(rows[i].get("prescreen_confidence"), 1) if verdict == "edit" else 0
        return (vp, cp)

    if args.order == "verdict":
        pending_idx.sort(key=_verdict_sort_key)
    print(f"Row order this session: {args.order!r} "
          f"({'reject, edit (high-confidence first), uncertain, accept' if args.order == 'verdict' else 'queue file order, unchanged'}).")

    print(f"\n{len(pending_idx)} prescreened row(s) to confirm this session (pending + skipped).\n")

    session_start = time.time()
    session_confirmed = 0
    session_bulk = 0
    pos = 0
    quit_now = False
    while pos < len(pending_idx):
        i = pending_idx[pos]
        row = rows[i]
        pre = {"verdict": row["prescreen_verdict"], "reason": row["prescreen_reason"],
               "suggested_edit": row["prescreen_suggested_edit"]}
        _print_row_with_prescreen(pos + 1, len(pending_idx), row, meta.get(row["question_id"], {}), pre)

        can_confirm = pre["verdict"] != "uncertain"
        # Bulk confirm is offered after this row is decided (below the save_queue() call). "After
        # a spot check" means the reviewer has just read this row; bulk is not an alternative to
        # deciding it.
        prompt = "Action? "
        if can_confirm:
            prompt += "[c]onfirm  "
        prompt += "[e]dit-override  [r]eject-override  [s]kip  [q]uit-and-save > "

        acted = False
        while not acted:
            try:
                choice = input(prompt).strip().lower()
            except EOFError:
                choice = "q"

            if choice in ("q", "quit"):
                quit_now = True
                acted = True
            elif choice in ("c", "confirm") and can_confirm:
                _apply_confirm(row, pre, "individual")
                session_confirmed += 1
                acted = True
            elif choice in ("e", "edit"):
                print("  Enter the replacement hint; finish with a single '.' on its own line")
                lines: list[str] = []
                while True:
                    try:
                        line = input("  > ")
                    except EOFError:
                        break
                    if line.strip() == ".":
                        break
                    lines.append(line)
                edited = "\n".join(lines).strip()
                if not edited:
                    print("  (empty edit — nothing saved; choose an action again)")
                    continue
                row.update(review_status=_STATUS_EDITED, reviewed=True, reviewed_hint=edited,
                           review_notes=None, reviewed_at=_now_iso(),
                           confirmed_via="individual", overridden_prescreen=True)
                session_confirmed += 1
                acted = True
            elif choice in ("r", "reject"):
                reason = ""
                while not reason:
                    try:
                        reason = input("  reason (required) > ").strip()
                    except EOFError:
                        reason = "(no reason given — EOF)"
                        break
                row.update(review_status=_STATUS_REJECTED, reviewed=True, reviewed_hint=None,
                           review_notes=reason, reviewed_at=_now_iso(),
                           confirmed_via="individual", overridden_prescreen=True)
                session_confirmed += 1
                acted = True
            elif choice in ("s", "skip"):
                row["review_status"] = _STATUS_SKIPPED
                acted = True
            else:
                print("  not understood.")

        save_queue(qpath, rows)
        pos += 1
        if quit_now:
            break

        # Offer bulk confirm for the run of same-verdict rows that follows, if there is one.
        if not can_confirm:
            continue
        run_len = 0
        for j in pending_idx[pos:]:
            if rows[j]["prescreen_verdict"] == pre["verdict"]:
                run_len += 1
            else:
                break
        if run_len == 0:
            continue
        ids_preview = [rows[j]["question_id"] for j in pending_idx[pos: pos + run_len]]
        print(f"\n  {run_len} upcoming row(s) also carry verdict={pre['verdict']!r}: {ids_preview[:20]}"
              + (" ..." if len(ids_preview) > 20 else ""))
        try:
            resp = input(f"  Bulk-confirm all {run_len} after that one spot check? "
                         f"[Enter=all / number / n=no] > ").strip().lower()
        except EOFError:
            resp = "n"
        if resp in ("n", "no"):
            continue
        if resp == "" or resp == "all":
            n = run_len
        else:
            try:
                n = max(0, min(run_len, int(resp)))
            except ValueError:
                print("  not understood — skipping bulk-confirm.")
                continue
        for _ in range(n):
            j = pending_idx[pos]
            rj = rows[j]
            prej = {"verdict": rj["prescreen_verdict"], "reason": rj["prescreen_reason"],
                    "suggested_edit": rj["prescreen_suggested_edit"]}
            _apply_confirm(rj, prej, "bulk")
            print(f"    bulk-confirmed id={rj['question_id']} tier={rj['tier']} verdict={prej['verdict']}")
            session_confirmed += 1
            session_bulk += 1
            pos += 1
        save_queue(qpath, rows)

    session_elapsed = time.time() - session_start

    timing_path = qpath.with_name(qpath.name + ".review_timing.json")
    timing = {"total_review_seconds": 0.0, "total_rows_reviewed": 0, "sessions": []}
    if timing_path.exists():
        try:
            timing = json.loads(timing_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    timing["sessions"].append({
        "mode": "confirm",
        "started_at": datetime.fromtimestamp(session_start, tz=timezone.utc).isoformat(),
        "ended_at": _now_iso(), "seconds": round(session_elapsed, 1),
        "rows_reviewed": session_confirmed, "rows_bulk_confirmed": session_bulk,
    })
    timing["total_review_seconds"] = round(timing.get("total_review_seconds", 0.0) + session_elapsed, 1)
    timing["total_rows_reviewed"] = timing.get("total_rows_reviewed", 0) + session_confirmed
    timing_path.write_text(json.dumps(timing, indent=2, ensure_ascii=False), encoding="utf-8")

    final_counts = counts_by_status(rows)
    print("\n" + "=" * 100)
    print("CONFIRM-MODE SESSION SUMMARY")
    print(f"  total rows in queue: {len(rows)}")
    print(f"  status now: {final_counts}")
    print(f"  this session: {session_confirmed} row(s) decided "
          f"({session_confirmed - session_bulk} individually, {session_bulk} via bulk-confirm), "
          f"{session_elapsed:.1f}s wall-clock")
    print(f"  cumulative timing sidecar ({timing_path.name}) now includes this 'confirm' mode "
          f"session TAGGED as such — do not blend with 'review' (from-scratch) session numbers "
          f"in any per-row-time methodology claim.")
    print("=" * 100)


if __name__ == "__main__":
    main()
