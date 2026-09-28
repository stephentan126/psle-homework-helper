"""
Request-latency instrumentation for the worst-case request flow (Issue 232).

This module only observes. It does not change gate, safety or retry behaviour, control flow or
timing: it reads `time.monotonic()` before and after each stage and writes one structured JSON log
line per stage and one per request.

The call chain in `backend/app/api/routes.py` and `submission_routes.py`:
  1. `_resolve_gate_result_and_hint()` on the verified-match path: either a version-matching
     `PrecomputedGateResult` cache hit (near-instant) or the live self-consistency gate
     (`gate.run_full_gate_pipeline()`, Issues 167 and 170, measured worst case 14.98-38.47s).
     The weak-mode branch of `confirm_photo_submission()` calls
     `gate.run_weak_mode_gate_pipeline()` instead, which is instrumented at its own call site.
  2. `_finalize_and_persist_hint()`: the output safety check (`self_harm_detection.check_safety()`),
     retried up to `_MAX_HARASSMENT_RETRY_ATTEMPTS` times (3 attempts in total) on a harassment
     flag, with each retry preceded by hint regeneration (`gate.generate_hint()`). The
     20.86-32.82s worst-case tail in Issue 207 is dominated by regeneration, not the classifier,
     so regeneration is timed as its own stage.

Issue 211 combined these into a worst case of about 139s for a single request, measured once.
This module turns that into a per-request metric, so a change that slows a stage shows up in the
logs.

Warning threshold: `LATENCY_WARNING_THRESHOLD_SECONDS = 90.0`, based on the ranges combined in
Issue 211. The gate alone reaches 38.47s at worst, and one harassment retry (classifier plus
regeneration) has measured up to about 16.4s (32.82s over 2 retries), so one gate call plus two
retries approaches 90s well before the 139s ceiling. The value is well above a normal request
(a cache hit is near-instant; a live gate run with no retries takes 15-38s), so ordinary traffic
does not trigger it, and well below 139s, so a slow request is flagged before it reaches the
ceiling. Update Issue 232 if this value changes.
"""
from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

logger = logging.getLogger("api.latency")

# Reasoning in the module docstring (Issue 232).
LATENCY_WARNING_THRESHOLD_SECONDS = 90.0


@dataclass
class RequestLatencyTracker:
    """Collects stage timings for one request.

    One instance is created at the top of each endpoint (`request_hint()` in `routes.py`,
    `confirm_photo_submission()` in `submission_routes.py`) and passed explicitly to the shared
    gate and finalise functions. It is never global or thread-local, because FastAPI serves
    requests concurrently on a thread pool and timings must not mix between requests (Issue 168).

    Stage names are a convention, not an enum. Those in use are `"gate"`, `"weak_mode_gate"`,
    `"safety_check"` and `"hint_regeneration"`. Extra keyword arguments (for example
    `cache_hit=True` or `attempt=2`) are added to the JSON record as they are, so new stages or
    fields need no change here.
    """

    request_id: str
    _start: float = field(default_factory=time.monotonic)
    _stages: list = field(default_factory=list)

    @contextmanager
    def stage(self, stage_name: str, **detail) -> Iterator[None]:
        """Times one stage and logs a JSON record when it finishes.

        Uses `time.monotonic()`, which is unaffected by wall-clock adjustments. The record is
        written in a `finally` block, so a stage that raises still has its duration logged.
        Exceptions are re-raised unchanged.
        """
        stage_start = time.monotonic()
        try:
            yield
        finally:
            duration_s = time.monotonic() - stage_start
            record = {
                "event": "latency_stage",
                "request_id": self.request_id,
                "stage": stage_name,
                "duration_s": round(duration_s, 4),
            }
            if detail:
                record["detail"] = detail
            self._stages.append(record)
            logger.info(json.dumps(record))

    def finish(self) -> float:
        """Logs a per-request summary and returns the total elapsed seconds.

        The summary is one JSON log line with the cumulative total since the tracker was created
        and the per-stage breakdown. When the total reaches `LATENCY_WARNING_THRESHOLD_SECONDS` it
        is logged at `WARNING` level with an `alert` field. Calling it more than once is safe, but
        each call site calls it once in a `finally` block, so it runs on every exit path of
        `_finalize_and_persist_hint()`.
        """
        total_s = time.monotonic() - self._start
        summary = {
            "event": "latency_summary",
            "request_id": self.request_id,
            "cumulative_s": round(total_s, 4),
            "warning_threshold_s": LATENCY_WARNING_THRESHOLD_SECONDS,
            "stage_breakdown": [
                {"stage": s["stage"], "duration_s": s["duration_s"], **(
                    {"detail": s["detail"]} if "detail" in s else {}
                )}
                for s in self._stages
            ],
        }
        if total_s >= LATENCY_WARNING_THRESHOLD_SECONDS:
            summary["alert"] = (
                f"cumulative request latency {total_s:.1f}s reached the real soft-alert threshold "
                f"({LATENCY_WARNING_THRESHOLD_SECONDS}s) — approaching the known ~139s historical "
                f"worst case (Issue 211); investigate which stage(s) grew, do not assume it is "
                f"just this one slow request."
            )
            logger.warning(json.dumps(summary))
        else:
            logger.info(json.dumps(summary))
        return total_s
