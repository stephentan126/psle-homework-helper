"""
evaluation/model_selection/ocr_extraction/runners/run_batched.py

Stage B batched-subprocess driver (Issue 83's mitigation, docs/DEVELOPMENT_LOG.md).

MITIGATION, NOT A ROOT-CAUSE FIX. Issue 82 found MinerU2.5's Stage B leaking GPU memory across
a long-running in-process page loop and fixed it with a per-page torch.cuda.empty_cache() +
gc.collect(). Applying that same fix to DeepSeek-OCR only partially worked (Issue 83): a page
that should take ~10-60s took 634s while VRAM climbed to 96.6% of the 8GB budget, with the actual
output confirmed normal-length (not a runaway generation), pointing at internal state the
external empty_cache() call can't reach. Rather than read DeepSeek-OCR's real trust_remote_code
infer() implementation to find what it retains (real work, uncertain payoff, deferred), this
script sidesteps the problem structurally: each batch of pages runs in a separate
subprocess. A full process exit reclaims everything an in-process fix might miss, regardless of
the actual internal cause, the model, any generation cache, any leaked CUDA context, all of it,
because the process itself is gone.

If a stall shows up even at small batch granularity, that's a sign this mitigation alone isn't
enough and the real fix (reading the actual infer() source) can't be deferred forever, see the
per-batch log this script writes for exactly that evidence.

Works for either reader (--reader mineru25 or deepseek_ocr), the orchestration logic (pending
page discovery, batching, subprocess invocation + timeout, per-batch logging) is identical between
them, only the venv python and target script differ, so this is one shared driver rather than two
near-duplicate ones.

Usage:
    python run_batched.py --reader mineru25 --image-dir <Stage A output> --output-dir <Stage B output>
    python run_batched.py --reader deepseek_ocr --image-dir ... --output-dir ... --batch-size 5 --timeout-s 600
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
RUNNERS_DIR = REPO_ROOT / "evaluation" / "model_selection" / "ocr_extraction" / "runners"
ENVS_DIR = REPO_ROOT / "evaluation" / "model_selection" / "ocr_extraction" / "envs"

VALID_READERS = ("mineru25", "deepseek_ocr")


def get_pending_pages(image_dir: Path, output_dir: Path) -> list[tuple[str, str]]:
    """Same resumability check run_on_directory() already does, skip a page if its output JSON
    already exists. Returns (pdf_stem, page_index) pairs for everything still missing, in sorted
    order (deterministic batch composition across repeated driver runs)."""
    pending = []
    for page_path in sorted(image_dir.rglob("*.png")):
        pdf_stem = page_path.parent.name
        page_index = page_path.stem
        if not (output_dir / pdf_stem / f"{page_index}.json").exists():
            pending.append((pdf_stem, page_index))
    return pending


def get_known_bad_pages(batch_log_path: Path, threshold: int) -> dict[str, dict]:
    """Real, computed-from-history detection of a page that fails deterministically and never
    completes (Issue 221, the development log), NOT a separate, hand-maintained state
    file, so it can never drift from what the log actually shows.

    Deliberately trusts ONLY single-page batch records (`len(pages) == 1`) for blame. A page
    listed in `missing_pages` of a MULTI-page batch's failure is NOT necessarily itself the
    cause, it may just be an innocent later page in the same subprocess call that never got a
    turn because an earlier page in that same batch hung first (confirmed real behavior: the
    original stuck 10-page batch showed `completed_pages: []` on every attempt, which is
    consistent with the very first page alone hanging and blocking the other 9, not all 10 being
    bad). Only a single-page batch gives unambiguous per-page attribution, so multi-page timeout/
    crash history contributes nothing here, it takes a real `--batch-size 1` isolation run (see
    this project's own Issue 221 investigation) to ever produce qualifying evidence.

    A page that has EVER appeared in any record's `completed_pages` is never treated as bad,
    however many times it failed on the way there (a real, if slower, success is still success).
    """
    if not batch_log_path.exists():
        return {}
    failures: dict[str, dict] = {}
    ever_completed: set[str] = set()
    for line in batch_log_path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        ever_completed.update(record.get("completed_pages", []))
        if len(record.get("pages", [])) != 1:
            continue  # multi-page batch, no unambiguous per-page attribution, see docstring
        if record.get("status") not in ("timed_out", "crashed"):
            continue
        page = record["pages"][0]
        info = failures.setdefault(page, {
            "failure_count": 0,
            "first_failure_ts": record["timestamp"],
            "last_failure_ts": record["timestamp"],
            "batches": [],
        })
        info["failure_count"] += 1
        info["last_failure_ts"] = record["timestamp"]
        info["batches"].append(record["batch"])
    return {
        page: info for page, info in failures.items()
        if info["failure_count"] >= threshold and page not in ever_completed
    }


def default_timeout_s(batch_size: int) -> int:
    """
    Scaled per-batch timeout, not one fixed number for every batch size (speed-pass review,
). Real evidence this is sized from: DeepSeek-OCR's worst observed SINGLE page
    latency was 634.33s (Issue 83), the flat 600s default this replaces was already BELOW
    that, meaning a repeat of that exact scenario inside a batch would have been killed as if
    stuck, even though it was still progressing. Base of 700s covers one such spike
    with real margin (not the barely-over 634s it would otherwise be); +60s for each additional
    page in the batch, since the rest of a real batch is expected to run at the normal ~15-60s/
    page range, not another worst-case outlier.
    """
    return 700 + 60 * (batch_size - 1)


def run_batched(
    reader: str, image_dir: Path, output_dir: Path, batch_size: int = 5,
    timeout_s: "int | None" = None,
    exclude_known_bad: bool = False, known_bad_threshold: int = 2,
) -> dict:
    if timeout_s is None:
        timeout_s = default_timeout_s(batch_size)
    if reader not in VALID_READERS:
        raise ValueError(f"reader must be one of {VALID_READERS}, got {reader!r}")

    venv_python = ENVS_DIR / reader / ".venv" / "Scripts" / "python.exe"
    script = RUNNERS_DIR / f"run_{reader}.py"
    if not venv_python.exists():
        raise FileNotFoundError(f"Reader venv not found: {venv_python}")
    if not script.exists():
        raise FileNotFoundError(f"Reader script not found: {script}")

    output_dir.mkdir(parents=True, exist_ok=True)
    batch_log_path = output_dir / "_batch_log.jsonl"
    known_bad_path = output_dir / "_known_bad_pages.json"
    scratch_dir = output_dir / "_batch_scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    # Batches are computed ONCE, up front, from the pending list at driver-start, not
    # recomputed after each batch. This is deliberate: it guarantees a page that times out in
    # one batch is never retried later in the SAME driver run (it would still show up as
    # "pending" on a re-scan, causing an immediate repeat timeout in a tight loop). It gets
    # picked up on the next FULL invocation of this driver instead, same as the instruction.
    pending = get_pending_pages(image_dir, output_dir)

    # Issue 221 skip-guard, opt-in (default off), so this only ever changes behavior for a
    # caller that explicitly asks for it (the deepseek_ocr resilient wrapper); mineru25 and any
    # other caller of this shared driver keeps its exact current behavior.
    known_bad: dict[str, dict] = {}
    if exclude_known_bad:
        known_bad = get_known_bad_pages(batch_log_path, known_bad_threshold)
        if known_bad:
            # Written every run this flag is set, not just when the set changes, keeps
            # first/last-failure timestamps and batch history current and makes the file's own
            # mtime a real, live signal that the guard is active, not stale leftover state.
            known_bad_path.write_text(
                json.dumps(known_bad, indent=2, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            skip_set = set(known_bad)
            excluded = [p for p in pending if f"{p[0]}/{p[1]}" in skip_set]
            pending = [p for p in pending if f"{p[0]}/{p[1]}" not in skip_set]
            print(
                f"Issue 221 skip-guard: excluding {len(excluded)} known-bad page(s) "
                f"(>= {known_bad_threshold} confirmed single-page failure(s), never completed) "
                f"-- see {known_bad_path}: {[f'{s}/{i}' for s, i in excluded]}",
            )

    batches = [pending[i:i + batch_size] for i in range(0, len(pending), batch_size)]

    print(f"=== run_batched: {reader} — {len(pending)} page(s) pending, {len(batches)} batch(es) of up to {batch_size} ===")

    summary = {
        "reader": reader, "total_pending_at_start": len(pending), "batch_size": batch_size,
        "timeout_s": timeout_s, "batches_run": 0, "batches_completed": 0,
        "batches_timed_out": 0, "batches_crashed": 0,
        "pages_completed": 0, "pages_missing": [],
        "known_bad_pages_excluded": sorted(known_bad.keys()),
    }

    for batch_num, batch_pages in enumerate(batches, start=1):
        pages_file = scratch_dir / f"batch_{batch_num:04d}.tsv"
        pages_file.write_text(
            "\n".join(f"{pdf_stem}\t{page_index}" for pdf_stem, page_index in batch_pages),
            encoding="utf-8",
        )

        cmd = [
            str(venv_python), str(script),
            "--image-dir", str(image_dir), "--output-dir", str(output_dir),
            "--pages-file", str(pages_file),
        ]
        print(f"\n--- batch {batch_num}/{len(batches)}: {batch_pages} ---")

        t0 = time.time()
        status = "completed"
        stdout_tail = ""
        try:
            result = subprocess.run(
                cmd, timeout=timeout_s, capture_output=True, text=True,
                encoding="utf-8", errors="replace",
            )
            stdout_tail = "\n".join(result.stdout.splitlines()[-15:])
            if result.returncode != 0:
                status = "crashed"
                print(f"  subprocess exited {result.returncode}. stderr tail:\n{result.stderr[-1500:]}")
        except subprocess.TimeoutExpired as e:
            # subprocess.run's own timeout handling kills the child process (and, on Windows,
            # its process tree via taskkill under the hood) before raising, no manual
            # terminate()/kill() needed here, but confirmed logged as timed_out either way.
            status = "timed_out"
            stdout_tail = (e.stdout or "")[-1500:] if e.stdout else ""
            print(f"  TIMED OUT after {timeout_s}s — killed, moving to next batch.")
        t1 = time.time()

        completed_pages = [p for p in batch_pages if (output_dir / p[0] / f"{p[1]}.json").exists()]
        missing_pages = [p for p in batch_pages if p not in completed_pages]

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "batch": batch_num, "reader": reader,
            "pages": [f"{s}/{i}" for s, i in batch_pages],
            "wall_time_s": round(t1 - t0, 2),
            "status": status,
            "completed_pages": [f"{s}/{i}" for s, i in completed_pages],
            "missing_pages": [f"{s}/{i}" for s, i in missing_pages],
            "stdout_tail": stdout_tail,
        }
        with batch_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"  status={status} wall={t1-t0:.1f}s completed={len(completed_pages)}/{len(batch_pages)}")

        summary["batches_run"] += 1
        summary[f"batches_{status}"] += 1
        summary["pages_completed"] += len(completed_pages)
        summary["pages_missing"].extend(f"{s}/{i}" for s, i in missing_pages)

    summary["pages_missing_count"] = len(summary["pages_missing"])
    print(f"\n=== run_batched ({reader}) done: {summary['pages_completed']}/{len(pending)} pages completed "
          f"this run, {summary['pages_missing_count']} still missing, "
          f"{summary['batches_timed_out']} batch(es) timed out, "
          f"{summary['batches_crashed']} batch(es) crashed ===")
    if known_bad:
        # Real, deliberate, never-silent: a run that excluded known-bad pages must never report
        # itself as fully done without saying so, even when pages_missing_count is 0 (Issue
        # 221, the whole point of this guard is to unblock the REST of the corpus, not to make
        # the skipped page(s) invisible).
        print(
            f"    NOTE: {len(known_bad)} known-bad page(s) were EXCLUDED from this run entirely "
            f"(not attempted) -- see {known_bad_path}: {sorted(known_bad.keys())}. "
            f"These remain a REAL, OPEN gap, not resolved by this run.",
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reader", required=True, choices=VALID_READERS)
    parser.add_argument("--image-dir", type=Path, required=True, help="Stage A output (rendered pages).")
    parser.add_argument("--output-dir", type=Path, required=True, help="Stage B output (per-page common-schema JSON).")
    parser.add_argument("--batch-size", type=int, default=5, help="Pages per subprocess invocation (default 5).")
    parser.add_argument(
        "--timeout-s", type=int, default=None,
        help="Wall-clock timeout per batch, seconds. Default: scaled to batch size "
             "(700 + 60*(batch_size-1) — see default_timeout_s()), not one fixed number for "
             "every batch size. Pass explicitly to override.",
    )
    parser.add_argument(
        "--exclude-known-bad", action="store_true",
        help="Issue 221 skip-guard (opt-in, default off): before batching, exclude any page "
             "that has failed at least --known-bad-threshold times in its OWN single-page batch "
             "(never completed) per the real history in _batch_log.jsonl, so a deterministically "
             "stuck page can't block the rest of the corpus forever. Written to "
             "<output-dir>/_known_bad_pages.json, and always reported in the final summary — "
             "never silently dropped. Default off so this driver's behavior is unchanged for any "
             "existing caller (e.g. mineru25) unless explicitly requested.",
    )
    parser.add_argument(
        "--known-bad-threshold", type=int, default=2,
        help="Number of distinct single-page-batch timed_out/crashed failures (never completed) "
             "before a page is treated as known-bad. Default 2: a single-page batch gives "
             "unambiguous per-page attribution already (no batch-mate confusion possible), and "
             "with deterministic (do_sample=False) decoding a real stuck page fails identically "
             "every time, so 2 is real confirmation without wasting a 3rd full timeout on a page "
             "that's already shown itself twice. Only used with --exclude-known-bad.",
    )
    args = parser.parse_args()

    summary = run_batched(
        args.reader, args.image_dir, args.output_dir,
        batch_size=args.batch_size, timeout_s=args.timeout_s,
        exclude_known_bad=args.exclude_known_bad, known_bad_threshold=args.known_bad_threshold,
    )
    (args.output_dir / "_batch_run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if summary["pages_missing_count"] > 0 or summary["known_bad_pages_excluded"]:
        # Non-zero whenever the corpus isn't fully done, either pages are still
        # missing from this run, or some were deliberately excluded as known-bad (Issue 221).
        # A calling script/wrapper must never read exit 0 here as "100% complete" when pages were
        # quietly carved out before batching even started.
        sys.exit(1)


if __name__ == "__main__":
    main()
