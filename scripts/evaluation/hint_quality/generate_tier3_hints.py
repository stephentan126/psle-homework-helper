r"""
Generate Tier 3 hints, as the app serves them, for 50 question-bank questions (Tier 3 evaluation).

Tier 1 and Tier 2 were scored on held-out questions (generate_hints.py). Tier 3 cannot be scored
there: it retrieves a related worked example through the question's stored embedding, which only
question-bank rows have, and the app shows it only when the cached gate result allows Tiers 2 and
3 (tier_allowed == 4). This script therefore samples 50 such bank questions with a fixed seed.

Each hint is made exactly as the live endpoint makes it: `gate.generate_hint(requested_tier=3)`
with the served weights (LLM_ADAPTER_PATH unset, so the base model), then the answer-leak guard
(`answer_leak.guard_hint`) and `tidy_hint_text`. Both the raw hint and the shown hint are kept, so
the raw leak rate and the guard's effect can be reported separately.

Output: data/extracted/hint_quality_eval/tier3_hints.json (resumable; saved after every row).

Usage: backend/.venv/Scripts/python.exe scripts/evaluation/hint_quality/generate_tier3_hints.py
"""
import json
import random
import sqlite3
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from leak_check import check_leak, numeric_values  # noqa: E402

_N, _SEED = 50, 20260928
_DIR = _REPO_ROOT / "data" / "extracted" / "hint_quality_eval"


def _sample() -> list[dict]:
    con = sqlite3.connect(f"file:{_REPO_ROOT / 'psle.db'}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT q.id, q.question_text, q.answer_value, q.worked_solution_text, q.has_diagram "
        "FROM precomputed_gate_results g JOIN questions q ON q.id = g.question_id "
        "WHERE g.is_active = 1 AND g.tier_allowed = 4 AND q.superseded_by IS NULL "
        "AND q.worked_solution_text IS NOT NULL AND q.worked_solution_text != '' "
        "ORDER BY q.id"
    ).fetchall()
    eligible = [dict(r) for r in rows if numeric_values(r["answer_value"] or "")]
    print(f"eligible bank questions with a Tier 3 gate result: {len(eligible)}")
    return random.Random(_SEED).sample(eligible, _N)


def main() -> None:
    from app.pipeline import gate
    from app.pipeline.answer_leak import guard_hint, tidy_hint_text
    from app.services import llm_service

    _DIR.mkdir(parents=True, exist_ok=True)
    out_path = _DIR / "tier3_hints.json"
    results = json.load(open(out_path, encoding="utf-8")) if out_path.exists() else {}
    print("adapter status:", llm_service.get_llm_adapter_status())
    sample = _sample()
    for i, r in enumerate(sample):
        k = str(r["id"])
        if k in results:
            continue
        gr = gate.GateResult(passed=True, tier_allowed=4, reason="Tier 3 eval: cached gate allows Tiers 2 and 3")

        def make(steering=None):
            return gate.generate_hint(
                r["question_text"], gr, worked_solution_text=r["worked_solution_text"], extra_steering=steering,
                has_diagram=bool(r["has_diagram"]), question_id=r["id"], requested_tier=3,
            )

        t0 = time.time()
        raw = make()
        shown = guard_hint(raw, [str(r["answer_value"])], r["question_text"], lambda s: make(s))
        shown["hint_text"] = tidy_hint_text(shown.get("hint_text", ""))
        results[k] = {
            "id": r["id"], "question_text": r["question_text"], "answer_value": r["answer_value"],
            "worked_solution_text": r["worked_solution_text"], "has_diagram": bool(r["has_diagram"]),
            "tier_returned": raw["tier"], "tier3_similarity_score": raw.get("tier3_similarity_score"),
            "tier3_degrade_reason": raw.get("tier3_degrade_reason"),
            "raw_hint_text": raw["hint_text"],
            "raw_leak": check_leak(r["answer_value"], r["question_text"], raw["hint_text"]),
            "leak_guard": shown["leak_guard"], "hints": {"3": {"hint_text": shown["hint_text"]}},
            "seconds": round(time.time() - t0, 1),
        }
        out_path.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"[{i + 1}/{_N}] id={k} tier={raw['tier']} raw_leak={results[k]['raw_leak']['status']} "
              f"guard={shown['leak_guard']} {results[k]['seconds']}s")
    print("DONE ->", out_path)


if __name__ == "__main__":
    main()
