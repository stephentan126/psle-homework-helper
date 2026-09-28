"""Tests for `app.services.anthropic_config.get_anthropic_api_key()`.

The function loads only the Anthropic key from `.env`, for the API-route model evaluation. The tests
use temporary files and fake values, and never touch the project `.env` or a live key.
"""
from __future__ import annotations

import os

from app.services.anthropic_config import get_anthropic_api_key

_FAKE = "sk-ant-FAKE-test-value-not-a-real-key"


def test_reads_the_key_from_an_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    f = tmp_path / ".env"
    f.write_text(f"ANTHROPIC_API_KEY={_FAKE}\n", encoding="utf-8")
    assert get_anthropic_api_key(f) == _FAKE


def test_a_real_environment_variable_wins_over_the_file(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-FAKE-from-env")
    f = tmp_path / ".env"
    f.write_text(f"ANTHROPIC_API_KEY={_FAKE}\n", encoding="utf-8")
    assert get_anthropic_api_key(f) == "sk-ant-FAKE-from-env"


def test_a_blank_value_counts_as_not_configured(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    f = tmp_path / ".env"
    f.write_text("ANTHROPIC_API_KEY=\n", encoding="utf-8")  # the shipped .env placeholder
    assert get_anthropic_api_key(f) is None


def test_a_missing_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert get_anthropic_api_key(tmp_path / "does-not-exist.env") is None


def test_surrounding_quotes_and_whitespace_are_handled(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    f = tmp_path / ".env"
    f.write_text(f'ANTHROPIC_API_KEY="{_FAKE}"\n', encoding="utf-8")
    assert get_anthropic_api_key(f) == _FAKE


def test_no_other_env_variable_is_loaded_and_the_key_is_not_copied_into_os_environ(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("SOME_OTHER_SETTING", raising=False)
    f = tmp_path / ".env"
    f.write_text(f"ANTHROPIC_API_KEY={_FAKE}\nSOME_OTHER_SETTING=should-stay-unloaded\n", encoding="utf-8")
    assert get_anthropic_api_key(f) == _FAKE
    assert "SOME_OTHER_SETTING" not in os.environ, "explicit loading must not change how any other value is loaded"
    assert "ANTHROPIC_API_KEY" not in os.environ, "the key must not be copied into the process environment"
