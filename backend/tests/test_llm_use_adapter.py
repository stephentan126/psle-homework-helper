"""
CPU-only tests for scoping the LoRA adapter to hint writing (Issue 353).

Covers the `use_adapter` parameter on `llm_service.generate()` and `generate_multiple()`, the lock
that stops concurrent requests switching each other's adapter on or off, and which `gate.py` call
sites pass `use_adapter=False`. The fake models mimic PEFT's shared, in-place `disable_adapter()`
semantics. The PEFT model itself is checked on the GPU by
`scripts/evaluation/verify_llm_use_adapter.py`.
"""
from __future__ import annotations

import contextlib
import threading
import time

import pytest
import torch

from app.pipeline import gate
from app.services import llm_service


class FakeTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt=True, return_tensors="pt"):
        return {"input_ids": torch.tensor([[1, 2, 3]])}

    def decode(self, ids, skip_special_tokens=True):
        return "decoded"


class FakePlainModel:
    """A base-only model with no disable_adapter(), as AutoModelForCausalLM returns."""
    device = "cpu"

    def generate(self, input_ids, **kwargs):
        return torch.tensor([[1, 2, 3, 4], [1, 2, 3, 5]])[: kwargs.get("num_return_sequences", 1)]


class FakePeftModel:
    """Mimics PeftModel.disable_adapter() with one shared flag.

    As in peft 0.20.0, exit re-enables the adapter unless it was already disabled at entry, so
    overlapping contexts can re-enable it under a call that wanted base weights.
    """
    device = "cpu"

    def __init__(self, hold_seconds: float = 0.0):
        self.adapter_enabled = True
        self.hold_seconds = hold_seconds
        self.seen: list[bool] = []  # adapter state observed by each generate() call

    @contextlib.contextmanager
    def disable_adapter(self):
        was_enabled = self.adapter_enabled
        try:
            self.adapter_enabled = False
            yield
        finally:
            if was_enabled:
                self.adapter_enabled = True

    def generate(self, input_ids, **kwargs):
        self.seen.append(self.adapter_enabled)
        if self.hold_seconds:
            time.sleep(self.hold_seconds)
        return torch.tensor([[1, 2, 3, 4]])


@pytest.fixture
def install(monkeypatch):
    def _install(model):
        monkeypatch.setattr(llm_service, "_model", model)
        monkeypatch.setattr(llm_service, "_tokenizer", FakeTokenizer())
        return model
    return _install


def test_use_adapter_false_runs_with_the_adapter_off_and_restores_it(install):
    m = install(FakePeftModel())
    llm_service.generate("s", "u", use_adapter=False)
    assert m.seen == [False] and m.adapter_enabled is True


def test_default_and_true_keep_the_adapter_on(install):
    m = install(FakePeftModel())
    llm_service.generate("s", "u")
    llm_service.generate("s", "u", use_adapter=True)
    assert m.seen == [True, True]


def test_generate_multiple_honours_use_adapter(install):
    m = install(FakePeftModel())
    llm_service.generate_multiple("s", "u", num_return_sequences=1, use_adapter=False)
    llm_service.generate_multiple("s", "u", num_return_sequences=1)
    assert m.seen == [False, True]


def test_a_base_only_model_is_unaffected_and_never_serialized(install):
    """With no adapter loaded (LLM_ADAPTER_PATH=""), use_adapter is a no-op and the lock is not taken.

    Base-only behaviour is therefore unchanged by the parameter.
    """
    install(FakePlainModel())
    llm_service._generate_lock.acquire()  # taking the lock here would block forever
    try:
        results = []
        t = threading.Thread(target=lambda: results.append(
            (llm_service.generate("s", "u", use_adapter=False), llm_service.generate("s", "u"))))
        t.start()
        t.join(timeout=3)
        assert not t.is_alive(), "base-only generation must not wait on the adapter lock"
        assert results == [("decoded", "decoded")]
    finally:
        llm_service._generate_lock.release()


def _race(model, hint_thread_delay=0.05):
    """Start a hint call (adapter on) while a base-weights call (use_adapter=False) runs."""
    solve = threading.Thread(target=lambda: llm_service.generate("s", "u", use_adapter=False))
    hint = threading.Thread(target=lambda: llm_service.generate("s", "u", use_adapter=True))
    solve.start()
    time.sleep(hint_thread_delay)
    hint.start()
    solve.join()
    hint.join()


def test_concurrent_requests_never_run_on_the_wrong_weights_with_the_lock(install):
    m = install(FakePeftModel(hold_seconds=0.3))
    _race(m)
    # One call saw the adapter on (the hint) and one saw it off (the solve).
    assert sorted(m.seen, reverse=True) == [True, False]
    # Each call saw the state it asked for. The solve runs first because it holds the lock.
    assert m.seen == [False, True]


def test_the_race_is_real_without_the_lock(install, monkeypatch):
    """Without the lock, the same scenario runs the hint on base weights.

    This sensitivity check shows that the lock prevents the race and that the previous test can
    detect it.
    """
    m = install(FakePeftModel(hold_seconds=0.3))
    monkeypatch.setattr(llm_service, "_generate_lock", contextlib.nullcontext())
    _race(m)
    assert m.seen == [False, False], "without serialization the hint call ran while the adapter was switched off"


# ---- Which gate.py call sites scope the adapter ----

@pytest.fixture
def recorded_calls(monkeypatch):
    calls = []

    def fake_generate(system_prompt, user_prompt, **kwargs):
        calls.append(("generate", kwargs))
        return {"expr": "2+3", "align": "MATCH"}.get(fake_generate.reply, fake_generate.reply)

    fake_generate.reply = "2+3"

    def fake_generate_multiple(system_prompt, user_prompt, **kwargs):
        calls.append(("generate_multiple", kwargs))
        return ["FINAL ANSWER: 5", "FINAL ANSWER: 5"]

    monkeypatch.setattr(gate, "generate", fake_generate)
    monkeypatch.setattr(gate, "generate_multiple", fake_generate_multiple)
    return calls, fake_generate


def test_the_three_solving_and_judgment_sites_run_on_base_weights(recorded_calls):
    calls, fake_generate = recorded_calls
    gate.sympy_exact_check("Q: 2 plus 3?", "5")
    gate.self_consistency_check("Q: 2 plus 3?", "5")
    fake_generate.reply = "MATCH"
    gate.method_alignment_check("model solution", "worked solution")
    assert [c[0] for c in calls] == ["generate", "generate_multiple", "generate"]
    assert all(kw.get("use_adapter") is False for _, kw in calls), calls


def test_the_two_hint_writing_sites_keep_the_adapter(recorded_calls):
    calls, fake_generate = recorded_calls
    fake_generate.reply = "a hint"
    gate.generate_hint("Q", gate.GateResult(passed=False, tier_allowed=1, reason="x", detail={}))
    gate.generate_hint("Q", gate.GateResult(passed=True, tier_allowed=2, reason="x", detail={"expr_str": "2+3"}))
    assert len(calls) == 2
    assert all(kw.get("use_adapter", True) is True for _, kw in calls), calls
