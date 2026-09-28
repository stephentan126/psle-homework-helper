r"""
Completion check used by `scripts/dev/run_pytest_resilient.sh` (Issue 238).

The wrapper follows the kill-detect-and-relaunch pattern of
`scripts/extraction/run_deepseek_ocr_resilient.sh` (Issue 215), with one important difference.
The OCR wrapper never needs to know why its child exited: after any exit it recounts pending pages
on disk, and a completed page is never attempted again, so an uncapped retry loop is safe. pytest
has no per-item, resumable checkpoint, so a killed run must restart from scratch, and blind
retries would hide test failures. This module separates the two cases: a run
that reached its end (whether tests passed or failed) must never be retried; only a run that never
finished is a candidate for relaunch.

Detection signal: the saved output of the three killed runs from Issue 237 was either empty (the
process died before writing anything) or truncated mid-run (one stopped in the progress dots at
"[ 66%]"). In every case, pytest's `-q` final summary line, such as
"218 passed, 5 warnings in 472.50s" or "3 failed, 215 passed, 5 warnings in 480.12s", was
missing. A run that completes, pass or fail, always ends with this line; an external kill never
lets pytest reach it. The check therefore looks for that line rather than the exit code, since
the exit code of an externally killed python.exe in this environment could not be established
reliably.

Usage: backend/.venv/Scripts/python.exe scripts/dev/pytest_resilient_helpers.py --check <log_file>
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

# pytest `-q` summary line: "N passed[, M failed][, K warnings] in X.XXs", in each variant this
# project's test runs have produced (bare "passed", "failed, ... passed", with or without a
# warnings clause). Both an outcome word (passed/failed/error/skipped) and the trailing
# "in <duration>s" are required, which keeps false positives low.
_PYTEST_SUMMARY_LINE_PATTERN = re.compile(
    r"\d+\s+(?:passed|failed|error(?:s)?|skipped).*?\bin\s+[\d.]+s\b",
)


def find_real_pytest_summary_line(log_text: str) -> Optional[str]:
    """Return the pytest `-q` summary line if the log contains one, else None.

    The whole text is searched, not just the tail. The line is normally near the end, but its
    exact position is not assumed.
    """
    match = _PYTEST_SUMMARY_LINE_PATTERN.search(log_text)
    return match.group(0) if match else None


def is_real_pytest_completion(log_text: str) -> bool:
    """Return True if the pytest run reached its end (pass or fail).

    False means the run looks cut off, which is how every observed external kill appeared.
    """
    return find_real_pytest_summary_line(log_text) is not None


def main() -> int:
    """Command-line entry point for the bash wrapper.

    `--check <log_file>` prints the matched summary line, if any. Exits 0 if the run completed,
    and 1 if the log looks like an external kill or crash, which tells the wrapper to relaunch.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path, required=True, help="Path to a real pytest log file.")
    args = parser.parse_args()

    if not args.check.exists():
        print(f"REAL COMPLETION: NOT FOUND -- log file {args.check} does not exist at all.")
        return 1

    log_text = args.check.read_text(encoding="utf-8", errors="replace")
    summary = find_real_pytest_summary_line(log_text)
    if summary is not None:
        print(f"REAL COMPLETION FOUND: {summary!r}")
        return 0
    print(
        f"REAL COMPLETION: NOT FOUND -- {args.check} ({len(log_text)} real chars) has no pytest "
        f"summary line; looks like an external kill or crash, not a genuine test-suite completion.",
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
