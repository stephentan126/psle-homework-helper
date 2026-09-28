r"""Judge the revised (v2) Tier 1 prompt candidates (Issue 362).

Runs after generate_tier1_candidates_v2.py, with the rubric, system prompt, model and pricing from
judge_hints.py. By default only A_open_strategy is judged, since it came closest to the ship rule.
For candidates B and C the question was whether fixing the echo also removed their leaks; the leak
check in generate_tier1_candidates_v2.py already answered that (0 leaks each, as in the baseline),
so they are not re-judged, to control cost. Pass --candidates to judge others.

Inputs (data/extracted/hint_quality_eval/): sample_rows.json, tier1_candidates_v2.json.
Output: tier1_candidate_v2_judgments.json in the same folder.

Usage:
    backend/.venv/Scripts/python.exe scripts/evaluation/hint_quality/judge_tier1_candidates_v2.py \
        [--max-usd 2.00] [--candidates A_open_strategy ...]
"""
import argparse
import json
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

import anthropic  # noqa: E402
from app.services.anthropic_config import get_anthropic_api_key  # noqa: E402
from judge_hints import MODEL, PRICE_IN, PRICE_OUT, _SYSTEM, _parse  # noqa: E402

_DIR = _REPO_ROOT / "data" / "extracted" / "hint_quality_eval"
MAX_TOKENS = 1500


def _user(question_text: str, worked_solution_text: str, hint: str) -> str:
    return f"QUESTION:\n{question_text}\n\nVERIFIED WORKED SOLUTION (reference only):\n{worked_solution_text}\n\nHINT TO GRADE:\n{hint}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-usd", type=float, default=2.00)
    ap.add_argument("--candidates", nargs="+", default=["A_open_strategy"])
    args = ap.parse_args()
    key = get_anthropic_api_key()
    if not key:
        sys.exit("No ANTHROPIC_API_KEY found.")
    client = anthropic.Anthropic(api_key=key, max_retries=0)
    rows = {str(r["id"]): r for r in json.load(open(_DIR / "sample_rows.json", encoding="utf-8"))}
    cands = json.load(open(_DIR / "tier1_candidates_v2.json", encoding="utf-8"))
    out_path = _DIR / "tier1_candidate_v2_judgments.json"
    res = json.load(open(out_path, encoding="utf-8")) if out_path.exists() else {}
    jobs = [(name, k) for name in args.candidates for k in cands[name]]
    for n, (name, k) in enumerate(jobs, 1):
        jid = f"{name}:{k}"
        if jid in res and res[jid].get("status") == "ok":
            continue
        spent = sum(r.get("cost_usd", 0) for r in res.values())
        if spent >= args.max_usd:
            print(f"STOP: spend ceiling ${args.max_usd:.2f} reached (${spent:.4f})")
            break
        r = rows[k]
        hint = cands[name][k]["hint_text"]
        t0 = time.time()
        try:
            resp = client.messages.create(model=MODEL, max_tokens=MAX_TOKENS, system=_SYSTEM,
                                          messages=[{"role": "user", "content": _user(r["question_text"], r["worked_solution_text"], hint)}])
        except Exception as e:
            res[jid] = {"status": "api_error", "error": repr(e)[:300]}
            out_path.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
            sys.exit(f"API error on {jid}: {e!r}")
        text = "".join(b.text for b in resp.content if b.type == "text")
        u = resp.usage
        cost = (u.input_tokens * PRICE_IN + u.output_tokens * PRICE_OUT) / 1e6
        entry = {"status": "ok", "candidate": name, "id": r["id"], "input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                 "cost_usd": round(cost, 6), "latency_s": round(time.time() - t0, 1), "raw": text}
        try:
            entry["scores"] = _parse(text)
        except Exception as e:
            entry["status"] = "unparseable"
            entry["parse_error"] = repr(e)[:200]
        res[jid] = entry
        out_path.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        if n % 10 == 0 or n == len(jobs):
            print(f"[{n}/{len(jobs)}] {jid} {entry['status']} ${cost:.4f}")
    total = sum(r.get("cost_usd", 0) for r in res.values())
    print(f"TOTAL cost so far ${total:.4f} over {sum(1 for r in res.values() if r.get('status') == 'ok')} ok calls")


if __name__ == "__main__":
    main()
