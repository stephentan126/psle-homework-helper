"""A right answer reached by a wrong method is capped at Tier 2 (report Table 5.5, R2).

`generate` and `generate_multiple` are replaced with fakes, so no model is loaded and these tests
run without a GPU. The fakes make both self-consistency passes agree with the answer key, so the
only thing that changes between the two tests is the method-alignment verdict.
"""
from __future__ import annotations

import pytest

from app.pipeline import gate

_QUESTION = (
    "Clark and Daniel had some sweets in the ratio 5 : 9. After each of them bought 8 more "
    "sweets, the ratio became 3 : 5. How many sweets does Clark have now?"
)
_WORKED = "5u + 8 : 9u + 8 = 3 : 5, so 25u + 40 = 27u + 24, u = 8. Clark now has 5 x 8 + 8 = 48."


@pytest.fixture
def fake_models(monkeypatch):
    def fake_generate_multiple(system_prompt, user_prompt, **kwargs):
        return ["Working... FINAL ANSWER: 48", "Other working... FINAL ANSWER: 48"]

    def fake_generate(system_prompt, user_prompt, **kwargs):
        # The method-alignment call asks for one word; every other call is hint writing.
        return fake_generate.verdict if kwargs.get("max_new_tokens") == 10 else "A hint."

    fake_generate.verdict = "MATCH"
    monkeypatch.setattr(gate, "generate", fake_generate)
    monkeypatch.setattr(gate, "generate_multiple", fake_generate_multiple)
    return fake_generate


def test_method_mismatch_caps_help_at_tier_2(fake_models):
    fake_models.verdict = "MISMATCH"
    result, hint, _ = gate.run_full_gate_pipeline(_QUESTION, "48", worked_solution_text=_WORKED)
    assert result.tier_allowed == 2
    assert gate.tier_ceiling(result.tier_allowed) == 2
    assert "Method alignment" in result.reason
    assert hint["tier"] <= 2


def test_method_match_allows_tier_3(fake_models):
    fake_models.verdict = "MATCH"
    result, _, _ = gate.run_full_gate_pipeline(_QUESTION, "48", worked_solution_text=_WORKED)
    assert result.tier_allowed == 4
    assert gate.tier_ceiling(result.tier_allowed) == 3
