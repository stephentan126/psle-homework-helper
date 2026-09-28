#!/usr/bin/env bash
# Self-restarting wrapper around the full backend pytest suite (Issues 237, 238).
#
# Why: on the shared development machine, a background memory guard killed full `pytest -q` runs
# repeatedly (6 consecutive kills in one case, 7 in another). The cause was memory pressure from
# other processes, not the project's code, and running the suite on its own reduced but did not
# remove the kills. Each kill previously had to be noticed and relaunched by hand.
#
# The structure follows scripts/extraction/run_deepseek_ocr_resilient.sh (Issue 215): a small bash
# loop with timestamped logging. Detection and retries differ, because pytest has no resumable
# per-item checkpoint: a killed run must restart from scratch, and unlimited retries would have no
# safety net. See scripts/dev/pytest_resilient_helpers.py for the reasoning.
#
# Detection: after each attempt, pytest_resilient_helpers.py checks this attempt's own log file
# for pytest's `-q` summary line. Found: the suite completed (all passed, or some tests failed), so
# stop and never retry a test failure. Not found: an external kill or crash, so relaunch, up to
# MAX_ATTEMPTS.
#
# MAX_ATTEMPTS defaults to 8, one more than the worst run of consecutive kills observed (7,
# Issue 236). That would have been enough in every recorded case, while still stopping a run that
# crashes the same way every time.
#
# Usage (from the repository root):
#   scripts/dev/run_pytest_resilient.sh

set -u
cd "$(dirname "$0")/../.."  # repository root (this script lives in scripts/<group>/)

PY="../backend/.venv/Scripts/python.exe"   # relative to backend/, where pytest is actually run
HELPER_PY="backend/.venv/Scripts/python.exe"  # relative to repo root, for the detection helper
LOG_DIR="scripts/_pytest_resilient_logs"
MAX_ATTEMPTS="${PYTEST_RESILIENT_MAX_ATTEMPTS:-8}"

mkdir -p "$LOG_DIR"

attempt=0
while [ "$attempt" -lt "$MAX_ATTEMPTS" ]; do
  attempt=$((attempt + 1))
  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  run_id=$(date -u +%Y%m%dT%H%M%SZ)
  log_file="$LOG_DIR/attempt_${attempt}_${run_id}.log"
  echo "[$ts] attempt $attempt/$MAX_ATTEMPTS: launching full pytest suite (backend/), logging to $log_file..."

  # Opt-in test hook (Issue 238). If PYTEST_RESILIENT_TEST_CMD is set, it runs instead of pytest,
  # so the retry loop can be tested end to end with a controllable fake command (for example one
  # that mimics a kill on its first N runs) without reproducing an out-of-memory kill. When it
  # is unset, the default, the pytest suite runs.
  if [ -n "${PYTEST_RESILIENT_TEST_CMD:-}" ]; then
    # The subshell (parentheses) makes an `exit` in the fake command end only that attempt, not
    # the wrapper. A bare `eval` in the current shell would exit the wrapper; the hook's
    # end-to-end test caught this.
    (eval "$PYTEST_RESILIENT_TEST_CMD") > "$log_file" 2>&1
  else
    (cd backend && "$PY" -m pytest -q) > "$log_file" 2>&1
  fi
  exit_code=$?
  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "[$ts] attempt $attempt process exited with code $exit_code. Checking for a real completion..."

  if "$HELPER_PY" scripts/dev/pytest_resilient_helpers.py --check "$log_file"; then
    ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    if [ "$exit_code" -eq 0 ]; then
      echo "[$ts] REAL, GENUINE COMPLETION on attempt $attempt -- all tests passed (exit 0). Done."
    else
      echo "[$ts] REAL, GENUINE COMPLETION on attempt $attempt -- real test failure(s) reported (exit $exit_code). NOT relaunching -- a real failure must surface, not be retried. See $log_file for the real failing test(s)."
    fi
    exit "$exit_code"
  fi

  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "[$ts] attempt $attempt looks like an external kill or crash (no real pytest summary line found in $log_file) -- $((MAX_ATTEMPTS - attempt)) attempt(s) remaining."
done

ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "[$ts] GAVE UP after $MAX_ATTEMPTS attempts, none reached a real completion. This is now real, genuine evidence of either a persistent environment problem beyond this wrapper's own real mitigation, or a real, repeatable crash -- inspect the real logs under $LOG_DIR directly, do not assume either cause."
exit 1
