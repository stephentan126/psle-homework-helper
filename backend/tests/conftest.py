"""
Shared fixtures for the integration tests (design specification Section 10.6).

`DATABASE_URL` is pointed at a throwaway file, separate from the gitignored `psle.db`, before any
`app.*` module is imported. `app/db/session.py` builds its engine at import time (a module-level
`create_engine()` call), so the override has to come first. pytest collects conftest.py before any
test module in its directory, which makes this reliable. Issue 158 shows what happens when writes
to `psle.db` are not controlled.
"""
from __future__ import annotations

import os
from pathlib import Path

TEST_DB_PATH = Path(__file__).parent / "_test_psle.db"
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB_PATH.as_posix()}"

import pytest  # noqa: E402, must follow the DATABASE_URL override above

from app.db.session import engine, init_db  # noqa: E402
from app.models.base import Base  # noqa: E402


@pytest.fixture(autouse=True)
def no_real_process_restart(request, monkeypatch):
    """Replace the process restart and, outside test_swap_tracker.py, disable the swap guard.

    A successful photo read schedules a process restart (swap_tracker.trigger_restart). Inside
    pytest that would kill the test run, so it is replaced with a recorder. The one test that
    checks the exit itself runs it in a subprocess.

    The swap guard refuses any model swap in production. Across a test session, a model loaded by
    an earlier test would make the guard refuse unrelated later tests, so it is disabled everywhere
    except test_swap_tracker.py, which tests the guard.
    """
    from app.services import swap_tracker

    calls: list[float] = []
    monkeypatch.setattr(swap_tracker, "trigger_restart", lambda delay_s=0.5: calls.append(delay_s))
    if Path(str(request.node.fspath)).name != "test_swap_tracker.py":
        monkeypatch.setattr(swap_tracker, "guard", lambda needs: None)
    return calls


@pytest.fixture(autouse=True)
def clean_test_db():
    """Drop and recreate all tables around each test.

    No test can then depend on another test's leftover rows or on run order.
    """
    Base.metadata.drop_all(bind=engine)
    init_db()
    yield
    Base.metadata.drop_all(bind=engine)


# --- Public-repo note -------------------------------------------------------
# The phone-photo fixtures in tests/_fixtures/ are photographs of purchased
# exam pages and are not redistributed (Copyright Act 2021 s244 covers analysis,
# not redistribution). Tests that need them are skipped when the folder is absent.
_PHOTO_FIXTURE_DIR = Path(__file__).resolve().parent / "_fixtures"
_PHOTO_FIXTURE_TEST_FILES = {
    "test_skew_detection.py",
    "test_extraction_failed_reasons.py",
    "test_live_photo_pipeline.py",
}


# --- GPU marker ---------------------------------------------------------------
# Test files that load a model (Phi-4-mini, Qwen3-VL, ShieldGemma or the embedding model).
# A clean install without the models showed that the last three files also need them.
# They are marked "gpu" so `pytest -m "not gpu"` runs the rest on any machine.
_GPU_TEST_FILES = {
    "test_hint_endpoint.py",
    "test_safety_wiring.py",
    "test_boilerplate_row_exclusion.py",
    "test_live_photo_pipeline.py",
    "test_matching_and_weak_mode.py",
    "test_multihint_escalation_endpoint.py",
    "test_multihint_tier3_and_persistence.py",
    "test_confirm_and_edit.py",
    "test_diagram_degrade_path.py",
    "test_self_harm_detection.py",
    "test_precompute_cache.py",
    "test_safety_classifier_crash_fails_closed.py",
    "test_calibration_logging_audit.py",
}


def pytest_collection_modifyitems(config, items):
    import pytest as _pytest

    gpu = _pytest.mark.gpu
    for item in items:
        if Path(str(item.fspath)).name in _GPU_TEST_FILES:
            item.add_marker(gpu)
    if not _PHOTO_FIXTURE_DIR.exists():
        skip = _pytest.mark.skip(reason="photo fixtures (copyrighted exam pages) not included in public repo")
        for item in items:
            if Path(str(item.fspath)).name in _PHOTO_FIXTURE_TEST_FILES:
                item.add_marker(skip)
