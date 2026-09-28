"""
CPU-only tests for the branch logic in `swap_tracker.guard()` (Issue 366).

No model is loaded and no GPU is used: `llm_service.is_llm_loaded()` and
`vlm_service.is_vlm_loaded()` are monkeypatched, as in `test_llm_adapter_wiring.py`. The
`os._exit()` call in `trigger_restart()` is tested in a subprocess so that a bug cannot kill the
test runner.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from app.services import swap_tracker


@pytest.fixture(autouse=True)
def _reset():
    swap_tracker.reset_for_tests()
    yield
    swap_tracker.reset_for_tests()


def _set_loaded(monkeypatch, llm: bool, vlm: bool) -> None:
    monkeypatch.setattr("app.services.llm_service.is_llm_loaded", lambda: llm)
    monkeypatch.setattr("app.services.vlm_service.is_vlm_loaded", lambda: vlm)


def test_first_ever_load_into_an_empty_process_is_not_a_swap(monkeypatch):
    """With nothing loaded, guard() allows the load and does not count it as a swap."""
    _set_loaded(monkeypatch, llm=False, vlm=False)
    swap_tracker.guard("vlm")  # no raise
    swap_tracker.guard("vlm")  # needing the model about to load is not a swap either


def test_needing_the_already_loaded_model_is_never_a_swap(monkeypatch):
    _set_loaded(monkeypatch, llm=True, vlm=False)
    swap_tracker.guard("llm")
    swap_tracker.guard("llm")
    swap_tracker.guard("llm")  # repeatable: the same model is never a swap


def test_any_real_swap_is_refused(monkeypatch):
    """No in-process swap is allowed: with the LLM loaded, needing the VLM is refused."""
    _set_loaded(monkeypatch, llm=True, vlm=False)
    with pytest.raises(swap_tracker.SwapRestartRequired) as exc_info:
        swap_tracker.guard("vlm")
    assert exc_info.value.needs == "vlm"


def test_vlm_to_llm_swap_is_refused(monkeypatch):
    """With the VLM loaded, needing the LLM is refused. This is the swap that crashed a dry run."""
    _set_loaded(monkeypatch, llm=False, vlm=True)
    with pytest.raises(swap_tracker.SwapRestartRequired) as exc_info:
        swap_tracker.guard("llm")
    assert exc_info.value.needs == "llm"


def test_repeated_use_of_the_loaded_model_never_triggers_a_restart(monkeypatch):
    _set_loaded(monkeypatch, llm=False, vlm=True)
    swap_tracker.guard("vlm")
    swap_tracker.guard("vlm")
    swap_tracker.guard("vlm")  # no raise, VLM is already loaded


def test_trigger_restart_really_exits_the_process(tmp_path: Path):
    """trigger_restart() exits the process with code 1.

    It runs unmocked in a subprocess so a bug cannot kill the test runner. The exit code shows the
    process ended, not just that a flag was set.
    """
    script = tmp_path / "_trigger.py"
    script.write_text(
        "import sys, time\n"
        "sys.path.insert(0, r'" + str(Path(__file__).resolve().parents[1]) + "')\n"
        "from app.services import swap_tracker\n"
        "swap_tracker.trigger_restart(delay_s=0.05)\n"
        "time.sleep(2)\n"
        "print('SHOULD NOT REACH HERE')\n",
        encoding="utf-8",
    )
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 1
    assert "SHOULD NOT REACH HERE" not in result.stdout
