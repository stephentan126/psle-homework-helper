"""
Stage A of the extraction pipeline: render PDF pages to images (Issue 81). Pure CPU, no model
is loaded.

Every page of each source PDF found by discover_pdfs() under data/School papers/ and data/Yearly/
is saved as data/extracted/pages/{pdf_stem}/{page_index}.png. discover_pdfs() excludes answer
files, so each Topology A file's matched answer file (find_matching_answer_file()) is added to the
same task list. Answer files are rendered like any other file, with the same output layout and
resumability.

Resumable per page: a page is skipped if its PNG already exists, so a partial run (crash, Ctrl-C
or a --limit subset test) can be re-run and only fills in what is missing (the same principle as
Issue 62's per-file skip). Adding answer files only adds new tasks; existing page tasks are built
identically and hit the same `out_path.exists()` check.

Reuses discover_pdfs() and render_pdf_page() from extract.py (PyMuPDF-based) rather than
duplicating the PDF walking and rendering logic.

Parallelism: this is CPU and memory work with no GPU, so it runs across processes, unlike Stage B
(GPU-bound, see run_batched.py). The development machine has 24 physical cores, 32 logical
processors and about 31.75GB of RAM (measured with `wmic`). Workers are sized on physical cores
because PyMuPDF rendering is compute-bound and hyperthreading adds little throughput for
compute-bound work. Re-measure on different hardware.

Race safety: the full task list is built up front with one entry per (pdf, page) pair, so no two
tasks share an output path and the check-then-write resumability check cannot race with itself.
This avoids the check-then-act race a shared queue would have.
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from app.pipeline.extract import discover_pdfs, find_matching_answer_file, render_pdf_page

logger = logging.getLogger("pipeline.render_pages")

# 24 physical cores on the development machine, minus 2 reserved for the OS and other processes.
# os.cpu_count() (32) is not used because it counts logical processors, and for compute-bound
# work the physical core count is the better limit.
DEFAULT_WORKERS = 22


def _render_one_page(task: "tuple[str, int, str]") -> "tuple[str, str, int, str | None]":
    """
    Renders one page task: (pdf_path_str, page_index, out_path_str).

    Defined at module level so it can be pickled for ProcessPoolExecutor on Windows, which uses
    the spawn start method; a closure over render_all_pages() state could not cross the process
    boundary. Returns (status, pdf_path_str, page_index, error_or_None).
    """
    pdf_path_str, page_index, out_path_str = task
    out_path = Path(out_path_str)
    if out_path.exists():
        return ("skipped", pdf_path_str, page_index, None)
    try:
        page = render_pdf_page(Path(pdf_path_str), page_index)
        out_path.write_bytes(page.image_bytes)
        return ("rendered", pdf_path_str, page_index, None)
    except Exception as e:  # noqa: BLE001, isolate per-page failures (Issue 62)
        return ("failed", pdf_path_str, page_index, f"{type(e).__name__}: {e}")


def render_all_pages(
    data_dir: Path, pages_dir: Path, limit: int | None = None, workers: int | None = None,
    source_subdirs: "list[str] | None" = None,
) -> dict:
    """
    Renders every page of every discovered PDF to pages_dir/{pdf_stem}/{page_index}.png.

    Work runs in parallel across `workers` processes (default DEFAULT_WORKERS). `source_subdirs`
    is passed to discover_pdfs() as `subfolders`; None keeps the default School papers and Yearly
    training pool.

    Returns:
        A summary dict with files_processed, answer_files_included, pages_rendered,
        pages_skipped, pages_failed and workers.
    """
    import fitz  # PyMuPDF, used here only for page counts

    if workers is None:
        workers = DEFAULT_WORKERS

    pdf_paths = discover_pdfs(data_dir, subfolders=source_subdirs) if source_subdirs else discover_pdfs(data_dir)
    if limit is not None:
        pdf_paths = pdf_paths[:limit]
        logger.info("Limiting run to first %d file(s) for a small-subset test.", limit)

    logger.info("Discovered %d source PDF(s) to render.", len(pdf_paths))

    # discover_pdfs() excludes "*Answer*" files, which Stage C looks up as siblings. Answer-key
    # extraction (extract_topology_a, locate_embedded_answer_section) needs their OCR'd content,
    # so they are rendered here like question files. Deduplicated in case one answer file matches
    # more than one question file; find_matching_answer_file()'s glob matching (Issue 71) should
    # prevent this, but it is not assumed.
    seen_answer_files: set[Path] = set()
    answer_files: list[Path] = []
    for pdf_path in pdf_paths:
        answer_file = find_matching_answer_file(pdf_path)
        if answer_file is not None and answer_file not in seen_answer_files:
            seen_answer_files.add(answer_file)
            answer_files.append(answer_file)
    if answer_files:
        logger.info("Also rendering %d matched Topology A answer file(s).", len(answer_files))

    # Question and answer files are assumed to have distinct stems, as in Issue 71's folder layout
    # (for example "PSLE Math 2019" and "PSLE Math 2019 Answer").
    all_paths = pdf_paths + answer_files

    # Build the page-unique task list up front (fitz.open() for page counts only, no rendering),
    # which rules out the race described in the module docstring.
    files_processed = 0
    tasks: list[tuple[str, int, str]] = []
    for pdf_path in all_paths:
        out_dir = pages_dir / pdf_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            with fitz.open(pdf_path) as doc:
                page_count = doc.page_count
        except Exception as e:  # noqa: BLE001, one bad file must not stop the batch
            logger.error("FILE FAILED (could not open): %s — %s: %s", pdf_path, type(e).__name__, e)
            continue

        files_processed += 1
        for page_index in range(page_count):
            out_path = out_dir / f"{page_index}.png"
            tasks.append((str(pdf_path), page_index, str(out_path)))

    logger.info("Dispatching %d page task(s) across %d worker process(es).", len(tasks), workers)

    pages_rendered = 0
    pages_skipped = 0
    pages_failed = 0
    workers = max(1, min(workers, len(tasks) or 1))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_render_one_page, t) for t in tasks]
        for future in as_completed(futures):
            status, pdf_path_str, page_index, error = future.result()
            if status == "rendered":
                pages_rendered += 1
            elif status == "skipped":
                pages_skipped += 1
            else:
                pages_failed += 1
                logger.error("PAGE FAILED: %s page %d — %s", pdf_path_str, page_index, error)

    summary = {
        "files_processed": files_processed,
        "answer_files_included": len(answer_files),
        "pages_rendered": pages_rendered,
        "pages_skipped": pages_skipped,
        "pages_failed": pages_failed,
        "workers": workers,
    }
    logger.info("Stage A summary: %s", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Job A extraction pipeline — Stage A (render).")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Root data/ folder (Section 3.1 layout).")
    parser.add_argument("--pages-dir", type=Path, default=Path("data/extracted/pages"), help="Rendered-page output root.")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N discovered files — use this for the small real-subset test "
             "run before scaling up to the full ~140-PDF corpus.",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help=f"Parallel worker processes (default {DEFAULT_WORKERS}, sized off this machine's "
             "24 physical cores minus 2 reserved for the OS — re-measure on different hardware).",
    )
    parser.add_argument(
        "--source-subdirs", type=str, nargs="+", default=None,
        help="Phase 6 Step 0a: override the default training-pool "
             "subfolder walk with an explicit list, e.g. "
             "--source-subdirs \"Hold out/200\" \"Hold out/small eval\" — see extract.py's "
             "discover_pdfs() docstring. Omit for normal training-pool rendering (unchanged).",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    render_all_pages(
        args.data_dir, args.pages_dir, limit=args.limit, workers=args.workers,
        source_subdirs=args.source_subdirs,
    )


if __name__ == "__main__":
    main()
