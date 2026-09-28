"""Keep each server process to a single large model.

Loading the vision model and the language model one after the other in the same process crashes
`transformers` on Windows with an access violation (Issues 364 to 366). The end-to-end dry run showed
that even the first switch crashes (Issue 385), so no switch is allowed inside a process.

`guard()` is called before any code that needs a model. If a different model is already loaded it
raises `SwapRestartRequired`, and the route returns a 503 "retry shortly" response. After a successful
photo read, `trigger_restart()` ends the process so that `scripts/server/run_server_supervised.py`
starts a fresh one, and the language model loads there for the first hint.

The model services (`llm_service.py`, `vlm_service.py`) are not changed; this module only reads their
`is_llm_loaded()` and `is_vlm_loaded()` status.
"""
from __future__ import annotations

import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_swap_count = 0


class SwapRestartRequired(Exception):
    """Raised by `guard()` when a request needs a model other than the one already loaded.

    Route handlers must catch it and return a retry response. It must never be swallowed, because
    that would let the crashing model load go ahead."""

    def __init__(self, needs: str):
        self.needs = needs
        super().__init__(f"a second same-process model swap (to serve {needs!r}) would be required -- refusing and restarting")


def guard(needs: str) -> None:
    """Refuse a model switch inside this process.

    Call immediately before code that may call `get_llm()` or `get_vlm()`. `needs` is `"llm"` or
    `"vlm"`. Loading into an empty process is allowed; loading a different model from the one already
    loaded raises `SwapRestartRequired`. Call sites: `vlm_service.transcribe_photo()` (needs `"vlm"`),
    `_resolve_gate_result_and_hint()` and `run_weak_mode_gate_pipeline()` (both need `"llm"`)."""
    global _swap_count
    from app.services import llm_service, vlm_service  # Imported here to avoid an import cycle.

    with _lock:
        currently_loaded = "llm" if llm_service.is_llm_loaded() else ("vlm" if vlm_service.is_vlm_loaded() else None)
        would_swap = currently_loaded is not None and currently_loaded != needs
        if not would_swap:
            return
        # Every switch is refused: even the first one crashed the process (Issue 385).
        _swap_count += 1
        raise SwapRestartRequired(needs)


def trigger_restart(delay_s: float = 0.5) -> None:
    """End this process after `delay_s` seconds so the supervisor starts a fresh one.

    The delay gives the current HTTP response time to reach the client, and the exit runs in a
    background thread so the request returns at once. `os._exit()` is used rather than `sys.exit()`
    because a `SystemExit` could be caught by the web framework, leaving the process alive."""

    def _exit() -> None:
        time.sleep(delay_s)
        logger.warning("swap_tracker: exiting process now for a supervised restart (avoiding a 2nd same-process model swap)")
        os._exit(1)

    threading.Thread(target=_exit, daemon=True).start()


def reset_for_tests() -> None:
    """Reset the switch counter. Used by tests only."""
    global _swap_count
    with _lock:
        _swap_count = 0
