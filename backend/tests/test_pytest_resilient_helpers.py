"""
Tests for `scripts/dev/pytest_resilient_helpers.py` (Issue 238).

`scripts/dev/run_pytest_resilient.sh` uses this logic to tell a completed test run (passing or
failing) from one killed externally. The fixtures are saved output, not synthesised:
`backend/tests/fixtures/real_killed_pytest_output_*.log` are byte-for-byte copies of killed runs
(from the three occurrences in Issue 237), and a successful completion log is also included.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "dev"))

from pytest_resilient_helpers import (  # noqa: E402
    find_real_pytest_summary_line,
    is_real_pytest_completion,
)

_FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _read_fixture(name: str) -> str:
    return (_FIXTURES / name).read_text(encoding="utf-8", errors="replace")


def test_real_historical_kill_truncated_mid_run_is_not_a_completion():
    """A saved Issue 237 log cut off at "[ 66%]" with no summary line is not a completion."""
    log_text = _read_fixture("real_killed_pytest_output_truncated_mid_run.log")
    assert is_real_pytest_completion(log_text) is False


def test_real_historical_kill_truncated_early_is_not_a_completion():
    """A saved log killed earlier, with only the startup warning header, is not a completion."""
    log_text = _read_fixture("real_killed_pytest_output_truncated_early.log")
    assert is_real_pytest_completion(log_text) is False


def test_real_successful_completion_log_is_detected():
    """A saved clean full-suite log (218 passed, 0 failures) is recognised as a completion."""
    log_text = _read_fixture("real_successful_pytest_completion.log")
    assert is_real_pytest_completion(log_text) is True
    summary = find_real_pytest_summary_line(log_text)
    assert summary is not None
    assert "218 passed" in summary
    assert "472.50s" in summary


def test_empty_output_is_not_a_completion():
    """A process killed before writing anything is not a completion."""
    assert is_real_pytest_completion("") is False


def test_a_genuine_test_failure_summary_line_IS_a_real_completion():
    """A run that finished with test failures is still a completion.

    This is the distinction the module exists for: failures must not be retried and hidden, only
    externally killed runs should be. The log uses the pytest -q summary format seen in this
    project's output.
    """
    log_text = (
        "..F..F.F.................................................. [100%]\n"
        "=================================== FAILURES ===================================\n"
        "3 failed, 215 passed, 5 warnings in 480.12s (0:08:00)\n"
    )
    assert is_real_pytest_completion(log_text) is True


def test_a_genuine_all_failed_summary_is_also_a_real_completion():
    log_text = "5 failed in 12.34s\n"
    assert is_real_pytest_completion(log_text) is True


def test_progress_dots_alone_with_no_summary_line_is_not_a_completion():
    """Progress dots and percentage markers without a summary line are not a completion.

    This is a minimal synthetic version of the truncated-mid-run fixture above.
    """
    log_text = "....................................... [ 45%]\n"
    assert is_real_pytest_completion(log_text) is False
