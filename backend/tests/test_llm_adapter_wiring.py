"""
CPU-only tests for the adapter-loading branch logic in `app.services.llm_service` (Issue 352).

No model is loaded and no GPU is used. Whether the adapter is applied and changes generations
on the full model is checked by `scripts/evaluation/verify_llm_adapter_wiring.py`.
"""
from __future__ import annotations

import importlib
import logging

import pytest

from app.services import llm_service


@pytest.fixture(autouse=True)
def _restore_module():
    """Reload llm_service after each test, since the tests reload it under different env values."""
    yield
    importlib.reload(llm_service)


def test_default_serves_base_weights_only(monkeypatch):
    """With no env var set, the service serves base weights only (Issue 363).

    The default was changed from step0_qlora_v2_r16_a32 after the adapter showed no improvement on
    any measure it was tested against (Issues 351, 354 and 359). The override is covered by
    test_an_explicit_path_still_loads_any_adapter_on_disk.
    """
    monkeypatch.delenv("LLM_ADAPTER_PATH", raising=False)
    mod = importlib.reload(llm_service)
    assert mod.LLM_ADAPTER_PATH == ""


def test_an_explicit_path_still_loads_any_adapter_on_disk(monkeypatch):
    monkeypatch.setenv("LLM_ADAPTER_PATH", "models/adapters/step0_qlora_v2_r16_a32")
    mod = importlib.reload(llm_service)
    assert mod.LLM_ADAPTER_PATH == "models/adapters/step0_qlora_v2_r16_a32"


def test_an_explicitly_empty_env_var_disables_the_adapter_and_says_so(monkeypatch):
    monkeypatch.setenv("LLM_ADAPTER_PATH", "")
    mod = importlib.reload(llm_service)
    assert mod.LLM_ADAPTER_PATH == ""
    sentinel = object()
    mod._model = sentinel
    mod._apply_adapter_if_configured()
    status = mod.get_llm_adapter_status()
    assert status["applied"] is False and "disabled" in status["reason"] and "BASE" in status["reason"]
    assert mod._model is sentinel, "with the adapter disabled the loaded base model must be left untouched"


def test_a_configured_but_missing_adapter_is_not_silent(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("LLM_ADAPTER_PATH", str(tmp_path / "no_such_adapter"))
    mod = importlib.reload(llm_service)
    sentinel = object()
    mod._model = sentinel
    with caplog.at_level(logging.WARNING, logger=mod.logger.name):
        mod._apply_adapter_if_configured()
    status = mod.get_llm_adapter_status()
    assert status["applied"] is False and "NOT FOUND" in status["reason"] and "BASE" in status["reason"]
    assert any("NOT applied" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)
    assert mod._model is sentinel


def test_a_directory_without_adapter_config_json_counts_as_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_ADAPTER_PATH", str(tmp_path))  # exists, but holds no adapter_config.json
    mod = importlib.reload(llm_service)
    mod._model = object()
    mod._apply_adapter_if_configured()
    assert mod.get_llm_adapter_status()["applied"] is False


def test_status_before_any_load_says_so_and_the_accessor_returns_a_copy(monkeypatch):
    monkeypatch.delenv("LLM_ADAPTER_PATH", raising=False)
    mod = importlib.reload(llm_service)
    status = mod.get_llm_adapter_status()
    assert status["applied"] is False and "not loaded" in status["reason"]
    status["applied"] = True  # mutating the returned dict must not change the module's record
    assert mod.get_llm_adapter_status()["applied"] is False
