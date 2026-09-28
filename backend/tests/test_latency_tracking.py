"""
Tests for the request-latency instrumentation (Issue 232).

Covers per-stage timing accuracy, the structured (JSON) log shape, the WARNING soft-alert
threshold, that an exception inside a stage is never swallowed, and that `_stage_ctx()` is a no-op
without a tracker. It also measures and reports the overhead of the instrumentation itself rather
than assuming it is negligible.
"""
from __future__ import annotations

import json
import logging
import time

import pytest

from app.pipeline.latency_tracking import (
    LATENCY_WARNING_THRESHOLD_SECONDS,
    RequestLatencyTracker,
)


def test_stage_records_a_real_duration_close_to_actual_elapsed_time(caplog):
    tracker = RequestLatencyTracker("req-1")
    with caplog.at_level(logging.INFO, logger="api.latency"):
        with tracker.stage("gate", cache_hit=False):
            time.sleep(0.05)
    records = [json.loads(r.message) for r in caplog.records if r.message.startswith("{")]
    stage_records = [r for r in records if r.get("event") == "latency_stage"]
    assert len(stage_records) == 1
    assert stage_records[0]["stage"] == "gate"
    assert stage_records[0]["request_id"] == "req-1"
    assert stage_records[0]["detail"] == {"cache_hit": False}
    # Not exact: with Windows timer resolution, time.sleep(0.05) can return a few ms early, so
    # the lower bound has some slack. The important check is that the duration is not much
    # larger, which would point to double-counting or an unaccounted delay.
    assert 0.04 <= stage_records[0]["duration_s"] < 1.0


def test_multiple_stages_all_recorded_in_order():
    tracker = RequestLatencyTracker("req-2")
    with tracker.stage("gate", cache_hit=True):
        pass
    with tracker.stage("safety_check", attempt=1):
        pass
    with tracker.stage("hint_regeneration", attempt=1):
        pass
    assert [s["stage"] for s in tracker._stages] == ["gate", "safety_check", "hint_regeneration"]


def test_stage_never_swallows_a_real_exception():
    tracker = RequestLatencyTracker("req-3")
    with pytest.raises(ValueError, match="real failure"):
        with tracker.stage("safety_check", attempt=1):
            raise ValueError("real failure")
    # The stage duration is still recorded even though it raised, so a crash does not lose its
    # timing data.
    assert len(tracker._stages) == 1
    assert tracker._stages[0]["stage"] == "safety_check"


def test_finish_returns_the_real_cumulative_total_and_logs_a_structured_summary(caplog):
    tracker = RequestLatencyTracker("req-4")
    with tracker.stage("gate", cache_hit=True):
        time.sleep(0.02)
    with caplog.at_level(logging.INFO, logger="api.latency"):
        total = tracker.finish()
    # Same Windows timer-resolution slack as the stage-duration test above.
    assert total >= 0.015 - 1e-6  # float rounding at the boundary
    summaries = [
        json.loads(r.message) for r in caplog.records
        if r.message.startswith("{") and json.loads(r.message).get("event") == "latency_summary"
    ]
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["request_id"] == "req-4"
    # The logged summary is rounded to 4 decimal places.
    assert summary["cumulative_s"] == pytest.approx(total, abs=5e-5)
    assert summary["warning_threshold_s"] == LATENCY_WARNING_THRESHOLD_SECONDS
    assert summary["stage_breakdown"] == [
        {"stage": "gate", "duration_s": tracker._stages[0]["duration_s"], "detail": {"cache_hit": True}},
    ]
    assert "alert" not in summary  # well under the threshold


def test_finish_logs_at_warning_and_includes_a_real_alert_once_threshold_is_reached(caplog, monkeypatch):
    """Reaching the threshold logs at WARNING and includes an alert.

    The threshold is lowered to a tiny value rather than sleeping for about 90s, but the
    comparison in `finish()` is the one used in production.
    """
    monkeypatch.setattr(
        "app.pipeline.latency_tracking.LATENCY_WARNING_THRESHOLD_SECONDS", 0.01,
    )
    tracker = RequestLatencyTracker("req-5")
    with tracker.stage("gate", cache_hit=False):
        time.sleep(0.02)
    with caplog.at_level(logging.INFO, logger="api.latency"):
        tracker.finish()
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warning_records) == 1
    summary = json.loads(warning_records[0].message)
    assert summary["event"] == "latency_summary"
    assert "alert" in summary
    assert "0.01" in summary["alert"] or "soft-alert threshold" in summary["alert"]


def test_finish_stays_at_info_below_threshold(caplog):
    tracker = RequestLatencyTracker("req-6")
    with tracker.stage("gate", cache_hit=True):
        pass
    with caplog.at_level(logging.INFO, logger="api.latency"):
        tracker.finish()
    assert not any(r.levelno == logging.WARNING for r in caplog.records)


def test_stage_ctx_is_a_real_noop_when_no_tracker_supplied():
    from app.api.routes import _stage_ctx

    ran = False
    with _stage_ctx(None, "gate", cache_hit=True):
        ran = True
    assert ran is True  # the wrapped code still runs: a no-op wrapper, not a skip


def test_stage_ctx_delegates_to_the_real_tracker_when_one_is_supplied():
    from app.api.routes import _stage_ctx

    tracker = RequestLatencyTracker("req-7")
    with _stage_ctx(tracker, "gate", cache_hit=True):
        pass
    assert len(tracker._stages) == 1
    assert tracker._stages[0]["stage"] == "gate"


# =================================================================================================
# Overhead measurement: the cost is measured and reported rather than assumed negligible.
# =================================================================================================

def test_real_instrumentation_overhead_is_negligible_compared_to_a_real_request():
    """The overhead of `stage()` and `finish()` is small compared with any request stage.

    The wall-clock overhead around a no-op body is measured over many iterations. The fastest
    stage in the chain, a classifier check, takes tens to hundreds of milliseconds, so the
    instrumentation must cost far less. The per-request overhead is printed (visible with
    `pytest -s`) so the figure can be checked.
    """
    iterations = 2000

    def _instrumented_run() -> None:
        tracker = RequestLatencyTracker("req-overhead")
        with tracker.stage("gate", cache_hit=True):
            pass
        with tracker.stage("safety_check", attempt=1):
            pass
        tracker.finish()

    def _uninstrumented_run() -> None:
        pass  # "do nothing" baseline, to separate plain Python call overhead from the
              # instrumentation cost rather than comparing against zero.

    start = time.perf_counter()
    for _ in range(iterations):
        _instrumented_run()
    instrumented_total = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(iterations):
        _uninstrumented_run()
    baseline_total = time.perf_counter() - start

    real_overhead_per_request_ms = (instrumented_total - baseline_total) / iterations * 1000
    print(
        f"\nReal measured instrumentation overhead: {real_overhead_per_request_ms:.4f} ms/request "
        f"(2 stages + 1 summary, {iterations} real iterations, "
        f"{instrumented_total:.3f}s instrumented vs {baseline_total:.3f}s baseline total).",
    )
    # A generous ceiling: even 5ms per request is negligible next to a chain whose fastest stage
    # (a classifier check) takes tens of milliseconds and whose worst case is about 139s. This is
    # a sanity check, not a tight timing assertion.
    assert real_overhead_per_request_ms < 5.0
