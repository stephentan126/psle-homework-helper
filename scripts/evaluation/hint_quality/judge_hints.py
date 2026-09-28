r"""
Score generated hints on dimensions 2 to 5 of the hint-quality rubric (RUBRIC.md) (Issue 357).

Runs after generate_hints.py. The judge is a model called through the Anthropic API (MODEL below).
It sees only the question, the worked_solution_text and an already generated hint from the
held-out set; no student data is involved. Dimension 1 (answer leak) is the deterministic check
from generate_hints.py and is not judged here; the judge never sees it.

Safeguards, as in the Issue 348 paid run: no automatic retries, the run stops at the first API
error, results are saved after every call, and a spend ceiling (--max-usd, default 3.00) stops new
calls. Cost is computed from the reported token usage at the per-MTok prices recorded for Issue
348. No extended thinking, since each call is a short rubric score; default sampling. Resumable.

Input: data/extracted/hint_quality_eval/hints.json. Output: judgments.json in the same folder.

Usage:
    backend/.venv/Scripts/python.exe scripts/evaluation/hint_quality/judge_hints.py [--max-usd 3.00]
    Tier 3: ... judge_hints.py --hints tier3_hints.json --out tier3_judgments.json --tiers 3
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "backend"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import anthropic  # noqa: E402
from app.services.anthropic_config import get_anthropic_api_key  # noqa: E402

MODEL = "claude-opus-5"
MAX_TOKENS = 1500  # 700 cut off 4 of 100 replies; those were re-run at 1500
PRICE_IN, PRICE_OUT = 5.00, 25.00  # USD per million tokens, as recorded for Issue 348
_DIR = _REPO_ROOT / "data" / "extracted" / "hint_quality_eval"

_SYSTEM = (
    "You are grading a HINT written by a homework helper for a Singapore Primary 6 student (about 12 years old). The helper must give a hint, "
    "never the final answer. You are given the question, the verified WORKED SOLUTION (for your reference only; the student never sees it), and the HINT. "
    "Score four dimensions from 1 (worst) to 5 (best) using these anchors:\n"
    "scaffolding: 1 = only restates the question, or is generic enough to fit any problem; 3 = points at a topic or step but gives no guiding question or method; "
    "5 = names the right method or first step for THIS problem and/or asks a guiding question, without doing it for the student.\n"
    "age_appropriate: 1 = technical, textbook-formal or confusing for a 12-year-old; 3 = understandable but stiff; 5 = plain, friendly wording and terms taught in Singapore primary maths.\n"
    "consistency: 1 = contradicts the worked solution or sends the student down a wrong path; 3 = not wrong but not aligned with the worked method; "
    "5 = fully consistent with the worked solution's method and quantities. A hint that only restates the question and makes no method claim should be scored on "
    "whether it misstates the question; say so in the rationale.\n"
    "length_clarity: 1 = empty, garbled, rambling, or a full worked solution; 3 = usable but too thin or too long; 5 = the right size for one hint.\n"
    "Reply with ONLY a JSON object, no other text, of exactly this shape: "
    '{"scaffolding": {"score": <int 1-5>, "why": "<one sentence>"}, "age_appropriate": {"score": <int>, "why": "..."}, '
    '"consistency": {"score": <int>, "why": "..."}, "length_clarity": {"score": <int>, "why": "..."}}'
)


def _user(rec: dict, hint: str) -> str:
    return (f"QUESTION:\n{rec['question_text']}\n\nVERIFIED WORKED SOLUTION (reference only):\n{rec['worked_solution_text']}\n\n"
            f"HINT TO GRADE:\n{hint}")


def _parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    d = json.loads(m.group(0))
    for k in ("scaffolding", "age_appropriate", "consistency", "length_clarity"):
        assert isinstance(d[k]["score"], int) and 1 <= d[k]["score"] <= 5, (k, d[k])
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-usd", type=float, default=3.00)
    ap.add_argument("--hints", default="hints.json", help="input file in the eval folder (tier3_hints.json for Tier 3)")
    ap.add_argument("--out", default="judgments.json", help="output file in the eval folder")
    ap.add_argument("--tiers", default="1,2", help="comma-separated tiers to score, e.g. 3")
    args = ap.parse_args()
    key = get_anthropic_api_key()
    if not key:
        sys.exit("No ANTHROPIC_API_KEY found (environment or repo-root .env).")
    client = anthropic.Anthropic(api_key=key, max_retries=0)
    hints = json.load(open(_DIR / args.hints, encoding="utf-8"))
    out_path = _DIR / args.out
    res = json.load(open(out_path, encoding="utf-8")) if out_path.exists() else {}
    jobs = [(k, t) for k in hints for t in args.tiers.split(",")]
    for n, (k, t) in enumerate(jobs, 1):
        jid = f"{k}:{t}"
        if jid in res and res[jid].get("status") == "ok":
            continue
        spent = sum(r.get("cost_usd", 0) for r in res.values())
        if spent >= args.max_usd:
            print(f"STOP: spend ceiling ${args.max_usd:.2f} reached (${spent:.4f})")
            break
        rec = hints[k]
        t0 = time.time()
        try:
            resp = client.messages.create(model=MODEL, max_tokens=MAX_TOKENS, system=_SYSTEM,
                                          messages=[{"role": "user", "content": _user(rec, rec["hints"][t]["hint_text"])}])
        except Exception as e:  # stop at the first API error; nothing is retried
            res[jid] = {"status": "api_error", "error": repr(e)[:300]}
            out_path.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
            sys.exit(f"API error on {jid}: {e!r}")
        text = "".join(b.text for b in resp.content if b.type == "text")
        u = resp.usage
        cost = (u.input_tokens * PRICE_IN + u.output_tokens * PRICE_OUT) / 1e6
        entry = {"status": "ok", "id": rec["id"], "tier": int(t), "stop_reason": resp.stop_reason, "input_tokens": u.input_tokens,
                 "output_tokens": u.output_tokens, "cost_usd": round(cost, 6), "latency_s": round(time.time() - t0, 1), "raw": text}
        try:
            entry["scores"] = _parse(text)
        except Exception as e:
            entry["status"] = "unparseable"
            entry["parse_error"] = repr(e)[:200]
        res[jid] = entry
        out_path.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"[{n}/{len(jobs)}] {jid} {entry['status']} in={u.input_tokens} out={u.output_tokens} ${cost:.4f}")
    total = sum(r.get("cost_usd", 0) for r in res.values())
    print(f"TOTAL cost so far ${total:.4f} over {sum(1 for r in res.values() if r.get('status') == 'ok')} ok calls")


if __name__ == "__main__":
    main()
