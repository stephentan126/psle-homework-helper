"""Bucket 2 API-route evaluation: the 17 scored gold rows (the 5 multi-part N/A rows are skipped, as in every local-model run)
sent, one call per row, to the Claude API with vision. Same prompt convention, model, call parameters and scorer as the single-row
check (api_single_row_check.py): question text + run_bakeoff._QA_PROMPT_SUFFIX ("FINAL ANSWER: <value>"), scored with the
harness's own fixed scorer (reused, not reimplemented).

Row id=3 already has a real paid result from the single-row check (identical prompt and settings); it is REUSED, not bought again,
so the total is exactly 17 real calls (16 new + the earlier one), and the report labels it.

Safeguards (real money):
  * Refuses to run if the results file exists (never overwrites paid results). --resume continues an interrupted run and skips
    every row already recorded; --force overrides deliberately.
  * max_retries=0 (via the same client settings) and the run STOPS at the first API error, nothing is retried automatically.
  * All 17 images are checked (file size <= 5 MB, each side <= 8000 px) BEFORE any call is made.
  * Results are written to disk after every call, so an interruption cannot lose paid results.
  * A hard spend ceiling (--max-usd, default 5.00): no new call is started once cumulative cost reaches it.
  * The API key is read via app.services.anthropic_config and never printed, logged or written.

Diagram buckets: Bucket 1 is charts and data, Bucket 2 is simple geometry with measurements,
and Bucket 3 is bar models and number lines.
"""
import argparse
import base64
import json
import sys
import time
from pathlib import Path

from PIL import Image

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1] / "backend"))

import anthropic  # noqa: E402
import run_bakeoff as rb  # noqa: E402
from api_single_row_check import (  # noqa: E402  (same model, limits and prices as the single-row check)
    MAX_TOKENS, MODEL, PRICE_CACHE_READ_PER_MTOK, PRICE_CACHE_WRITE_5M_PER_MTOK, PRICE_INPUT_PER_MTOK,
    PRICE_OUTPUT_PER_MTOK,
)
from app.services.anthropic_config import get_anthropic_api_key  # noqa: E402

OUT = _HERE / "api_claude_opus5_17row_results.json"
SINGLE_ROW_RESULT = _HERE / "api_claude_opus5_id3_single_row_result.json"
MEDIA_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


def scored_rows() -> list:
    rows = [r for r in json.load(open(rb.GOLD_SET_PATH, encoding="utf-8"))["rows"] if r["status"] != "excluded"]
    return [r for r in rows if not rb._is_multi_part_gold(r.get("corrected_answer_value") or r["stored_answer_value"])]


def preflight(rows: list) -> None:
    problems = []
    for r in rows:
        p = Path(r["image_path_abs"])
        if not p.exists():
            problems.append((r["id"], "missing file")); continue
        if p.suffix.lower() not in MEDIA_TYPES:
            problems.append((r["id"], f"unsupported type {p.suffix}")); continue
        if p.stat().st_size > 5_000_000:
            problems.append((r["id"], f"file {p.stat().st_size / 1e6:.1f} MB > 5 MB"))
        w, h = Image.open(p).size
        if max(w, h) > 8000:
            problems.append((r["id"], f"{w}x{h} exceeds 8000 px"))
    if problems:
        sys.exit(f"PRE-FLIGHT FAILED, no call made: {problems}")


def cost_of(usage: dict) -> float:
    return (usage["input_tokens"] * PRICE_INPUT_PER_MTOK + usage["output_tokens"] * PRICE_OUTPUT_PER_MTOK
            + usage["cache_read_input_tokens"] * PRICE_CACHE_READ_PER_MTOK
            + usage["cache_creation_input_tokens"] * PRICE_CACHE_WRITE_5M_PER_MTOK) / 1e6


def score(row: dict, text: str) -> dict:
    claimed = rb.extract_claimed_answer(text)
    ok = bool(rb._score_final_answer(row, claimed if claimed is not None else text))
    return {"reached_final_answer_line": claimed is not None, "claimed_answer": claimed, "matches_gold_harness": ok,
            "correct_on_parsed_line": ok and claimed is not None}


def save(results: list, status: str) -> None:
    done = [r for r in results if r.get("status") == "ok"]
    summary = {
        "status": status, "model": MODEL, "rows_recorded": len(done),
        "correct_on_parsed_line": sum(r["correct_on_parsed_line"] for r in done),
        "correct_harness_style": sum(r["matches_gold_harness"] for r in done),
        "rows_with_final_answer_line": sum(r["reached_final_answer_line"] for r in done),
        "total_input_tokens": sum(r["usage"]["input_tokens"] for r in done),
        "total_output_tokens": sum(r["usage"]["output_tokens"] for r in done),
        "total_cost_usd": round(sum(r["cost_usd"] for r in done), 6),
        "total_latency_s": round(sum(r["latency_s"] for r in done), 1),
    }
    OUT.write_text(json.dumps({"summary": summary, "rows": results}, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true", help="continue an interrupted run; skip rows already recorded")
    ap.add_argument("--force", action="store_true", help="overwrite an existing results file")
    ap.add_argument("--max-usd", type=float, default=5.00, help="stop before starting a new call once cumulative cost reaches this")
    args = ap.parse_args()

    results: list = []
    if OUT.exists():
        if args.resume:
            results = json.load(open(OUT, encoding="utf-8"))["rows"]
        elif not args.force:
            sys.exit(f"Refusing to run: {OUT.name} exists (paid results). Use --resume to continue or --force to overwrite.")
    have = {r["id"] for r in results if r.get("status") == "ok"}

    rows = scored_rows()
    assert len(rows) == 17, f"expected 17 scored rows, got {len(rows)}"
    preflight(rows)

    if "3" not in have and SINGLE_ROW_RESULT.exists():  # reuse the real, already-paid id=3 call (identical prompt/settings)
        s = json.load(open(SINGLE_ROW_RESULT, encoding="utf-8"))
        g = next(r for r in rows if str(r["id"]) == "3")
        results.append({
            "id": "3", "status": "ok", "reused_from_single_row_check": True, "gold_answer": s["gold_answer"],
            "stop_reason": s["stop_reason"], "content_block_types": s["content_block_types"], "latency_s": s["latency_s"],
            "usage": s["usage"], "cost_usd": s["cost_usd"], "response_text": s["response_text"], **score(g, s["response_text"]),
        })
        have.add("3")

    key = get_anthropic_api_key()
    if not key:
        sys.exit("No ANTHROPIC_API_KEY found (environment or repo-root .env).")
    client = anthropic.Anthropic(api_key=key, max_retries=0)

    for row in rows:
        rid = str(row["id"])
        if rid in have:
            continue
        spent = sum(r["cost_usd"] for r in results if r.get("status") == "ok")
        if spent >= args.max_usd:
            save(results, f"stopped: spend ceiling ${args.max_usd:.2f} reached at ${spent:.4f}")
            print(f"STOP: spend ceiling reached (${spent:.4f}); rows not run: {[str(r['id']) for r in rows if str(r['id']) not in have]}", flush=True)
            return
        p = Path(row["image_path_abs"])
        b64 = base64.standard_b64encode(p.read_bytes()).decode("utf-8")
        prompt = row["question_text"] + rb._QA_PROMPT_SUFFIX
        t0 = time.time()
        try:
            resp = client.messages.create(model=MODEL, max_tokens=MAX_TOKENS, messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": MEDIA_TYPES[p.suffix.lower()], "data": b64}},
                {"type": "text", "text": prompt}]}])
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
            save(results, f"stopped: API error on id={rid}")
            sys.exit(f"STOP at id={rid}: {type(e).__name__} {getattr(e, 'status_code', '')} {str(getattr(e, 'message', ''))[:200]} "
                     f"(no retry; results so far are saved; use --resume to continue)")
        elapsed = round(time.time() - t0, 2)
        text = "".join(b.text for b in resp.content if b.type == "text")
        u = resp.usage
        usage = {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                 "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
                 "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0}
        entry = {"id": rid, "status": "ok", "reused_from_single_row_check": False, "gold_answer":
                 row.get("corrected_answer_value") or row["stored_answer_value"], "model_served": resp.model,
                 "stop_reason": resp.stop_reason, "content_block_types": [b.type for b in resp.content],
                 "latency_s": elapsed, "usage": usage, "cost_usd": round(cost_of(usage), 6), "response_text": text,
                 **score(row, text)}
        results.append(entry)
        save(results, "in progress")
        print(f"id={rid}: stop={resp.stop_reason} in={usage['input_tokens']} out={usage['output_tokens']} "
              f"cost=${entry['cost_usd']:.4f} {elapsed}s final_line={entry['reached_final_answer_line']} "
              f"claimed={entry['claimed_answer']!r} gold={entry['gold_answer']!r} match={entry['matches_gold_harness']}", flush=True)

    save(results, "complete")
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
