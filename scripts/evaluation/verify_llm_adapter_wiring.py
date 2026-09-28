r"""
Verify that the live serving path applies the LoRA adapter, rather than just not crashing
(Issue 352).

Checks backend/app/services/llm_service.py (get_llm() and generate()). Each mode runs as a
separate process so that each gets a fresh model load:
  --mode adapter   Default environment (LLM_ADAPTER_PATH unset, so the v2 adapter). Loads through
                   get_llm(), reports get_llm_adapter_status(), and for a fixed prompt set
                   generates (a) with the adapter on and (b) with it off, using PEFT's
                   disable_adapter() on the same loaded weights. It then regenerates a few
                   held-out rows through llm_service.generate() with the prompts and decoding
                   of the grounding experiment harness, and compares them with the recorded C2
                   generations (same adapter and prompt). A difference would mean the live path
                   differs from the harness (chat template, tokenisation or arguments).
  --mode base      Run with LLM_ADAPTER_PATH="" set by the caller: a separate base-only load with
                   the same prompt set.
  --mode compare   Reads both outputs and prints the verdicts.

Full generated texts are written under data/ (not published). Only the compact summary without
texts, verification_summary.json, is meant to be kept.

Usage:
    backend/.venv/Scripts/python.exe scripts/evaluation/verify_llm_adapter_wiring.py --mode adapter
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_OUT_DIR = _REPO_ROOT / "data" / "extracted" / "llm_adapter_wiring"
_EVAL_DIR = _REPO_ROOT / "data" / "extracted" / "slice9_eval"
_EXPORT = _REPO_ROOT / "data" / "extracted" / "slice6_step0c" / "training_export_v1.jsonl"
_C2_FILE = _EVAL_DIR / "generations_C2_qlora_r16a32_rag_fewshot.json"


def _prompt_set() -> list[dict]:
    """Return the fixed prompt set.

    Four Tier 1 hint prompts are taken verbatim from the training export (the task the adapter was
    trained on), and three held-out questions use the harness's no-RAG solve prompt (which
    resembles the gate's solving calls).
    """
    rows = [json.loads(l) for l in _EXPORT.read_text(encoding="utf-8").splitlines() if l.strip()]
    hint_rows = [r for r in rows if r["tier"] == 1][:4]
    prompts = []
    for r in hint_rows:
        sysm = next(m["content"] for m in r["messages"] if m["role"] == "system")
        user = next(m["content"] for m in r["messages"] if m["role"] == "user")
        prompts.append({"kind": "tier1_hint", "id": r["question_id"], "system": sysm, "user": user, "max_new_tokens": 150})
    import grounding_generate as eg
    from grounding_retrieval import load_held_out
    for hq in load_held_out()[:3]:
        s, u = eg.build_prompt(hq["question_text"], None)
        prompts.append({"kind": "solve_no_rag", "id": hq["question_id"], "system": s, "user": u, "max_new_tokens": 200})
    return prompts


def _eval_rows() -> list[dict]:
    """Return a fixed mix of recorded C2 rows: complete, cut-off and no-RAG fallback rows."""
    c2 = {int(k): v for k, v in json.load(open(_C2_FILE, encoding="utf-8")).items()}
    ids = sorted(c2)
    done = [i for i in ids if "Final Answer:" in c2[i]["generated_text"] and not c2[i]["used_similarity_floor_fallback"]][:4]
    cut = [i for i in ids if "Final Answer:" not in c2[i]["generated_text"] and not c2[i]["used_similarity_floor_fallback"]][:2]
    fb = [i for i in ids if c2[i]["used_similarity_floor_fallback"]][:2]
    return [{"id": i, "bucket": b, "recorded": c2[i]["generated_text"]} for b, group in (("complete", done), ("cutoff", cut), ("fallback_no_rag", fb)) for i in group]


def _harness_prompt(qid: int) -> tuple[str, str]:
    """Rebuild the prompt the C2 run used for this row (same retrieval, floor, k and few-shot)."""
    import grounding_generate as eg
    from grounding_retrieval import (DEFAULT_K, SIMILARITY_FLOOR, compute_or_load_held_out_embeddings, load_held_out, load_pool,
                                       retrieve_k_nearest_floored)
    held = {h["question_id"]: h for h in load_held_out()}
    embeds = compute_or_load_held_out_embeddings(list(held.values()))
    neighbors, fallback = retrieve_k_nearest_floored(embeds[qid], load_pool(), k=DEFAULT_K, floor=SIMILARITY_FLOOR)
    return eg.build_prompt(held[qid]["question_text"], None if fallback else neighbors)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["adapter", "base", "compare"], required=True)
    args = ap.parse_args()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "compare":
        a = json.load(open(_OUT_DIR / "adapter_mode.json", encoding="utf-8"))
        b = json.load(open(_OUT_DIR / "base_mode.json", encoding="utf-8"))
        summary = {"adapter_status": a["status"], "base_status": b["status"], "prompts": []}
        for pa, pb in zip(a["prompt_outputs"], b["prompt_outputs"]):
            assert pa["id"] == pb["id"] and pa["kind"] == pb["kind"]
            summary["prompts"].append({
                "kind": pa["kind"], "id": pa["id"],
                "adapter_on_differs_from_base": pa["adapter_on"] != pb["base_only"],
                "adapter_off_context_equals_separate_base_load": pa["adapter_off_via_context"] == pb["base_only"],
                "adapter_on_chars": len(pa["adapter_on"]), "base_chars": len(pb["base_only"])})
        summary["eval_harness_spot_check"] = [
            {k: v for k, v in r.items() if k not in ("live", "recorded")} for r in a["eval_spot_check"]]
        n = len(summary["prompts"])
        summary["totals"] = {
            "prompts": n,
            "adapter_on_differs_from_base": sum(p["adapter_on_differs_from_base"] for p in summary["prompts"]),
            "adapter_off_context_matches_separate_base_load": sum(p["adapter_off_context_equals_separate_base_load"] for p in summary["prompts"]),
            "eval_spot_check_rows": len(a["eval_spot_check"]),
            "eval_spot_check_identical_to_recorded": sum(r["identical"] for r in a["eval_spot_check"]),
            "adapter_peak_vram_gb": a["peak_vram_gb"], "base_peak_vram_gb": b["peak_vram_gb"]}
        (_OUT_DIR / "verification_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return

    import torch
    from app.services import llm_service

    prompts = _prompt_set()
    model, tok = llm_service.get_llm()
    status = llm_service.get_llm_adapter_status()
    print("get_llm_adapter_status():", status)
    print("model class:", type(model).__name__)
    out = {"mode": args.mode, "status": status, "model_class": type(model).__name__, "prompt_outputs": []}

    for p in prompts:
        gen = lambda: llm_service.generate(p["system"], p["user"], max_new_tokens=p["max_new_tokens"], do_sample=False)  # noqa: E731
        rec = {"kind": p["kind"], "id": p["id"]}
        if args.mode == "adapter":
            rec["adapter_on"] = gen()
            with model.disable_adapter():  # same weights, adapter switched off
                rec["adapter_off_via_context"] = gen()
        else:
            rec["base_only"] = gen()
        out["prompt_outputs"].append(rec)
        print(f"  {p['kind']} id={p['id']} done")

    if args.mode == "adapter":
        spot = []
        for row in _eval_rows():
            s, u = _harness_prompt(row["id"])
            live = llm_service.generate(s, u, max_new_tokens=600, do_sample=False)
            rec_txt = row["recorded"]
            same = live.strip() == rec_txt.strip()
            first_div = next((i for i, (x, y) in enumerate(zip(live, rec_txt)) if x != y), None if same else min(len(live), len(rec_txt)))
            spot.append({"id": row["id"], "bucket": row["bucket"], "identical": same, "first_divergence_char": first_div,
                         "live_chars": len(live), "recorded_chars": len(rec_txt), "live": live, "recorded": rec_txt})
            print(f"  eval spot-check id={row['id']} [{row['bucket']}] identical={same} first_divergence={first_div}")
        out["eval_spot_check"] = spot
    out["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    (_OUT_DIR / f"{args.mode}_mode.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved -> {_OUT_DIR / (args.mode + '_mode.json')} | peak VRAM {out['peak_vram_gb']} GB")


if __name__ == "__main__":
    main()
