r"""
Verify that `use_adapter` on llm_service.generate() and generate_multiple() scopes the adapter to
hint writing (Issue 353).

Uses the same method and the same 7 fixed prompts as verify_llm_adapter_wiring.py (Issue 352), and
compares against the outputs recorded there:

  --mode adapter  (default environment, v2 adapter loaded)
     (1) greedy generate(use_adapter=False) must equal the recorded output of a separate base-only
         load (data/extracted/llm_adapter_wiring/base_mode.json, "base_only");
     (2) greedy generate() with the default must still equal the recorded adapter-on output
         (adapter_mode.json, "adapter_on"), showing the hint path is unchanged;
     (3) sampled generate_multiple() with use_adapter=False and True on fixed seeds, the shape of
         the gate's solving call.
  --mode base     (LLM_ADAPTER_PATH="", a separate base-only load): the same sampled calls and
                  seeds.
  --mode compare  compares (3) across the two processes: the adapter-loaded process with
                  use_adapter=False must equal the separate base load.

Run --mode adapter and --mode base in separate processes, then --mode compare. Full texts are
written under data/ (not published); the compare step prints and saves a summary without them.

Usage:
    backend/.venv/Scripts/python.exe scripts/evaluation/verify_llm_use_adapter.py --mode adapter
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

from verify_llm_adapter_wiring import _OUT_DIR, _prompt_set  # noqa: E402

_SEEDS = [1000, 1001, 1002]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["adapter", "base", "compare"], required=True)
    args = ap.parse_args()

    if args.mode == "compare":
        a = json.load(open(_OUT_DIR / "use_adapter_adapter_mode.json", encoding="utf-8"))
        b = json.load(open(_OUT_DIR / "use_adapter_base_mode.json", encoding="utf-8"))
        s = {"adapter_status": a["status"], "base_status": b["status"],
             "greedy_use_adapter_false_equals_recorded_separate_base_load": a["greedy_false_vs_recorded_base"],
             "greedy_default_equals_recorded_adapter_on_hint_path_unchanged": a["greedy_default_vs_recorded_adapter"],
             "sampled_use_adapter_false_equals_separate_base_load": [x == y for x, y in zip(a["sampled_false"], b["sampled_base"])],
             "sampled_use_adapter_true_differs_from_base": [x != y for x, y in zip(a["sampled_true"], b["sampled_base"])]}
        s["totals"] = {k: f"{sum(v)}/{len(v)}" for k, v in s.items() if isinstance(v, list)}
        (_OUT_DIR / "use_adapter_verification_summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
        print(json.dumps(s, indent=2))
        return

    import torch
    from app.services import llm_service

    prompts = _prompt_set()
    model, _tok = llm_service.get_llm()
    status = llm_service.get_llm_adapter_status()
    print("status:", {k: v for k, v in status.items() if k != "configured_path"}, "| model class:", type(model).__name__)
    out = {"mode": args.mode, "status": status}

    def sampled(use_adapter: bool) -> list[list[str]]:
        res = []
        for p, seed in zip(prompts[4:7], _SEEDS):  # the 3 solve-style prompts
            torch.manual_seed(seed)
            res.append(llm_service.generate_multiple(p["system"], p["user"], num_return_sequences=2, max_new_tokens=200,
                                                     temperature=0.8, use_adapter=use_adapter))
        return res

    if args.mode == "adapter":
        rec_base = json.load(open(_OUT_DIR / "base_mode.json", encoding="utf-8"))["prompt_outputs"]
        rec_adapter = json.load(open(_OUT_DIR / "adapter_mode.json", encoding="utf-8"))["prompt_outputs"]
        g_false, g_default = [], []
        for p, rb, ra in zip(prompts, rec_base, rec_adapter):
            assert p["id"] == rb["id"] == ra["id"]
            f = llm_service.generate(p["system"], p["user"], max_new_tokens=p["max_new_tokens"], do_sample=False, use_adapter=False)
            d = llm_service.generate(p["system"], p["user"], max_new_tokens=p["max_new_tokens"], do_sample=False)
            g_false.append(f == rb["base_only"])
            g_default.append(d == ra["adapter_on"])
            print(f"  {p['kind']} id={p['id']}: use_adapter=False == recorded separate base load: {g_false[-1]} | "
                  f"default == recorded adapter-on: {g_default[-1]}")
        out["greedy_false_vs_recorded_base"], out["greedy_default_vs_recorded_adapter"] = g_false, g_default
        out["sampled_false"], out["sampled_true"] = sampled(False), sampled(True)
    else:
        out["sampled_base"] = sampled(True)  # no adapter loaded: base weights
    out["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    (_OUT_DIR / f"use_adapter_{args.mode}_mode.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved; peak VRAM", out["peak_vram_gb"], "GB")


if __name__ == "__main__":
    main()
