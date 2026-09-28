"""One-off, SINGLE-CALL evidence run (Bucket 2 API-route evaluation): send gold row id=3, the same page
image and question text used throughout this investigation, to the Claude API with vision, and record the real answer
and the real cost. Not wired into run_bakeoff.py or diagram_service.py.

Deliberate guards against accidental spend:
  * ONE row, ONE call. The script refuses to run if the result file already exists (pass --force to override).
  * max_retries=0, so a transient failure is surfaced instead of being silently retried and possibly billed twice.
  * The API key is read via app.services.anthropic_config.get_anthropic_api_key() and passed straight to the client; it is
    never printed, logged or written to the result file.

Model: claude-opus-5 (chosen as the evaluation judge model). Thinking is left at
the model default (adaptive) and effort at its default. Prompt = the harness's own question text + the same
`FINAL ANSWER: <value>` suffix the local-model tests used (run_bakeoff._QA_PROMPT_SUFFIX), scored with the harness's own
fixed scorer.

Cost = usage.input_tokens * $5/MTok + usage.output_tokens * $25/MTok (Claude Opus 5 base prices, checked against
platform.claude.com/docs/en/about-claude/pricing); output_tokens includes any thinking tokens; the image is billed as
ordinary input tokens and is already inside input_tokens.

Diagram buckets: Bucket 1 is charts and data, Bucket 2 is simple geometry with measurements,
and Bucket 3 is bar models and number lines.
"""
import base64
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_REPO / "backend"))

import anthropic  # noqa: E402
import run_bakeoff as rb  # noqa: E402  (imports only: prompt suffix + scorer)
from app.services.anthropic_config import get_anthropic_api_key  # noqa: E402

MODEL = "claude-opus-5"
ROW_ID = "3"
MAX_TOKENS = 16000  # non-streaming ceiling per the API reference; a single short derivation needs far less
PRICE_INPUT_PER_MTOK = 5.00
PRICE_OUTPUT_PER_MTOK = 25.00
PRICE_CACHE_READ_PER_MTOK = 0.50
PRICE_CACHE_WRITE_5M_PER_MTOK = 6.25
OUT = _HERE / "api_claude_opus5_id3_single_row_result.json"


def main() -> None:
    if OUT.exists() and "--force" not in sys.argv:
        sys.exit(f"Refusing to run: {OUT.name} already exists (this script makes ONE paid call). Use --force to override.")

    key = get_anthropic_api_key()
    if not key:
        sys.exit("No ANTHROPIC_API_KEY found (environment or repo-root .env).")

    gold = json.load(open(rb.GOLD_SET_PATH, encoding="utf-8"))["rows"]
    row = {str(r["id"]): r for r in gold}[ROW_ID]
    image_path = Path(row["image_path_abs"])
    image_b64 = base64.standard_b64encode(image_path.read_bytes()).decode("utf-8")
    prompt = row["question_text"] + rb._QA_PROMPT_SUFFIX

    client = anthropic.Anthropic(api_key=key, max_retries=0)
    t0 = time.time()
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
    except anthropic.APIStatusError as e:
        sys.exit(f"API error: HTTP {e.status_code} {type(e).__name__}: {str(e.message)[:300]}")
    except anthropic.APIConnectionError as e:
        sys.exit(f"Connection error: {type(e).__name__}")
    elapsed = round(time.time() - t0, 2)

    text = "".join(b.text for b in response.content if b.type == "text")
    block_types = [b.type for b in response.content]
    usage = response.usage
    in_tok, out_tok = usage.input_tokens, usage.output_tokens
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cost = (in_tok * PRICE_INPUT_PER_MTOK + out_tok * PRICE_OUTPUT_PER_MTOK
            + cache_read * PRICE_CACHE_READ_PER_MTOK + cache_write * PRICE_CACHE_WRITE_5M_PER_MTOK) / 1e6

    claimed = rb.extract_claimed_answer(text)
    gold_answer = row.get("corrected_answer_value") or row["stored_answer_value"]
    matches = bool(rb._score_final_answer(row, claimed if claimed is not None else text))

    result = {
        "model_requested": MODEL, "model_served": response.model, "response_id": response.id,
        "row_id": ROW_ID, "gold_answer": gold_answer, "image_file": image_path.name,
        "max_tokens": MAX_TOKENS, "stop_reason": response.stop_reason,
        "stop_details": getattr(response, "stop_details", None) and str(response.stop_details),
        "content_block_types": block_types, "latency_s": elapsed,
        "usage": {"input_tokens": in_tok, "output_tokens": out_tok,
                  "cache_read_input_tokens": cache_read, "cache_creation_input_tokens": cache_write},
        "prices_per_mtok_usd": {"input": PRICE_INPUT_PER_MTOK, "output": PRICE_OUTPUT_PER_MTOK},
        "cost_usd": round(cost, 6),
        "reached_final_answer_line": claimed is not None, "claimed_answer": claimed, "matches_gold": matches,
        "response_text": text,
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"model served: {response.model} | stop_reason: {response.stop_reason} | blocks: {block_types} | {elapsed}s")
    print(f"usage: input {in_tok} tokens, output {out_tok} tokens (cache read {cache_read}, cache write {cache_write})")
    print(f"cost: ${cost:.6f}")
    print(f"FINAL ANSWER line reached: {claimed is not None} | claimed: {claimed!r} | gold: {gold_answer!r} | matches gold: {matches}")
    print("--- response text ---")
    print(text)


if __name__ == "__main__":
    main()
