r"""
Start the backend under a simple supervisor that restarts it when it exits (Issue 366).

This is the deployment shape chosen for a single-machine prototype: one process-manager script
that starts the backend, watches for crashes and writes a plain log file, with no container
orchestration. It gives the swap-crash workaround in `backend/app/services/swap_tracker.py`
something to restart into. That module exits the worker process (`os._exit(1)`) before a second
model swap in the same process would be attempted (Issues 364 and 365). This script notices the
exit and relaunches the server, so the student's retried request lands in a fresh process with no
swaps yet, the state Issue 365 showed to be safe.

Log: data/extracted/server_supervisor.log (each start and exit, with a timestamp and exit code).

Restart-loop protection: if the server starts more than MAX_RESTARTS_PER_WINDOW times within
RESTART_WINDOW_S, the supervisor stops relaunching it and exits non-zero. A crash loop, such as the
process dying on start-up, needs a person to investigate rather than endless restarts.

Usage:  python scripts/server/run_server_supervised.py [--host 127.0.0.1] [--port 8000]
"""
import argparse
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_DIR = _REPO_ROOT / "backend"
_LOG_PATH = _REPO_ROOT / "data" / "extracted" / "server_supervisor.log"

MAX_RESTARTS_PER_WINDOW = 5
RESTART_WINDOW_S = 300  # 5 minutes


def _log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
    print(line)
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default="8000")
    args = ap.parse_args()

    python = sys.executable
    cmd = [python, "-m", "uvicorn", "app.main:app", "--host", args.host, "--port", str(args.port)]
    _log(f"supervisor starting -- cmd={cmd!r} cwd={_BACKEND_DIR}")

    recent_starts: deque = deque()
    while True:
        recent_starts.append(time.time())
        while recent_starts and time.time() - recent_starts[0] > RESTART_WINDOW_S:
            recent_starts.popleft()
        if len(recent_starts) > MAX_RESTARTS_PER_WINDOW:
            _log(f"ABORTING: {len(recent_starts)} restarts within {RESTART_WINDOW_S}s -- a real crash loop, "
                 "not the expected single-swap-restart pattern. Needs a human, not more auto-restarts.")
            sys.exit(1)

        _log("launching uvicorn...")
        t0 = time.time()
        proc = subprocess.Popen(cmd, cwd=str(_BACKEND_DIR))
        exit_code = proc.wait()
        uptime = time.time() - t0
        _log(f"uvicorn exited: code={exit_code} uptime={uptime:.1f}s -- relaunching")


if __name__ == "__main__":
    main()
