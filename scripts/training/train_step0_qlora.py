r"""
Train the v1 QLoRA hint adapter on the reviewed hint export.

Input: data/extracted/slice6_step0c/training_export_v1.jsonl (1,349 rows), produced by
export_step0_training_data.py. Output: models/adapters/step0_qlora_v1.

Base model: the same local snapshot the live serving stack uses (`LLM_MODEL_PATH` in
backend/app/services/llm_service.py, models/phi-4-mini-instruct). That local directory is the
project's pin for Phi-4-mini-instruct; no Hugging Face Hub revision hash is recorded for this
model. Loading from the same path means training uses exactly the weights the app serves.

LoRA target_modules: `peft.utils.TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING` has no entry
for this model_type ("phi3") in the installed PEFT version, so there is no default. The targets
were found by loading the 4-bit model and listing its named_modules(): the only Linear4bit layers
are qkv_proj, o_proj, gate_up_proj and down_proj (Phi-3 packs QKV and gate/up projections, unlike
the separate q_proj/k_proj/v_proj of Llama-style models). lm_head is a full-precision Linear and is
excluded, as is conventional.

LoRA rank, alpha and dropout: PEFT's library defaults (`LoraConfig()` with no arguments), r=8,
lora_alpha=8, lora_dropout=0.0, with no sweep. This is not the r=16/alpha=32 often recommended in
the QLoRA paper and tutorials; the documented default was chosen deliberately.

`assistant_only_loss=True` (loss on the hint only, not the repeated system prompt and question)
was tried first and rejected in a smoke test before the full run. This model's chat template lacks
the `{% generation %}` markers that TRL's assistant_only_loss needs to find the assistant span, and
TRL will not patch it (`ValueError: The chat template is not training-compatible...`). Patching the
template by hand would be an untested customisation, so TRL's default (`assistant_only_loss=False`)
is kept. Every SFTConfig value is TRL's default except epochs, batch size and gradient
accumulation, chosen below.

Output: the LoRA adapter only (`model.save_pretrained()` on the PeftModel). merge_and_unload() is
never called, by design.

Pre-flight checks refuse to run if the RTX 4060 is not visible, another large model process is
resident (only one large model at a time), less than 4,000 MiB of VRAM is free, or the export does
not have the expected row count.

Usage: backend/.venv/Scripts/python.exe scripts/training/train_step0_qlora.py
"""
from __future__ import annotations

import gc
import json
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_EXPORT_PATH = _REPO_ROOT / "data" / "extracted" / "slice6_step0c" / "training_export_v1.jsonl"
_MODEL_PATH = str(_REPO_ROOT / "models" / "phi-4-mini-instruct")
_ADAPTER_OUT = _REPO_ROOT / "models" / "adapters" / "step0_qlora_v1"

# Row count of the committed export (`git show HEAD:<export path> | wc -l`). Re-check it if the
# export is ever regenerated.
_EXPECTED_ROW_COUNT = 1349

_LORA_TARGET_MODULES = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]  # see module docstring


def _nvidia_smi_text() -> str:
    try:
        return subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=20).stdout
    except Exception as e:  # noqa: BLE001
        return f"(nvidia-smi unavailable: {e})"


def _free_vram_mib() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip()
        return int(out.splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


def preflight() -> None:
    print("=== PRE-FLIGHT: nvidia-smi ===")
    smi = _nvidia_smi_text()
    print(smi)
    if "GeForce RTX 4060" not in smi:
        print("*** REFUSING: RTX 4060 not visible to nvidia-smi -- do not proceed. ***")
        sys.exit(1)
    if smi.count("python") > 0 or "run_batched" in smi or "mineru" in smi.lower():
        print("*** REFUSING: another python/OCR ML process looks resident -- one big model at a "
              "time (Section 2). Close it and retry. ***")
        sys.exit(1)
    free_mib = _free_vram_mib()
    print(f"  free VRAM before load: {free_mib} MiB")
    if free_mib is not None and free_mib < 4000:
        print("*** REFUSING: < 4000 MiB free -- not enough headroom for Phi-4-mini NF4 + LoRA "
              "training activations. Free the GPU and retry. ***")
        sys.exit(1)


def load_and_assert_dataset() -> list[dict]:
    if not _EXPORT_PATH.exists():
        print(f"*** No such file: {_EXPORT_PATH} ***")
        sys.exit(1)
    rows = [json.loads(l) for l in _EXPORT_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"\n=== Loaded {_EXPORT_PATH.name}: {len(rows)} row(s) ===")
    if len(rows) != _EXPECTED_ROW_COUNT:
        print(f"*** REFUSING TO TRAIN: expected {_EXPECTED_ROW_COUNT} rows (git HEAD's committed "
              f"count, verified before this script was written), got {len(rows)}. This is the "
              f"second, independent held-out/row-count safeguard on top of the export-time check "
              f"-- do not proceed on a mismatch, re-derive the expected count instead. ***")
        sys.exit(1)
    print(f"  row-count assertion PASSED ({len(rows)} == {_EXPECTED_ROW_COUNT})")
    return rows


def main() -> None:
    preflight()
    rows = load_and_assert_dataset()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    ds = Dataset.from_list([{"messages": r["messages"]} for r in rows])
    print(f"\n=== Dataset built: {len(ds)} examples ===")

    print("\n=== Loading tokenizer + base model (4-bit NF4) ===")
    tokenizer = AutoTokenizer.from_pretrained(_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
    )
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        _MODEL_PATH, quantization_config=bnb_config, device_map="cuda:0",
    )
    load_s = time.time() - t0
    print(f"  base model loaded in {load_s:.1f}s")
    print(f"  VRAM after base load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False  # required with gradient checkpointing

    lora_config = LoraConfig(
        r=8, lora_alpha=8, lora_dropout=0.0,  # PEFT's library default
        target_modules=_LORA_TARGET_MODULES,
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    epochs = 3
    per_device_batch = 1
    grad_accum = 8  # effective batch size 8

    sft_config = SFTConfig(
        output_dir=str(_REPO_ROOT / "models" / "adapters" / "_train_run_tmp"),
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        bf16=True,
        # assistant_only_loss stays at TRL's default (False): this model's chat template does not
        # support it (see module docstring).
        logging_steps=10,
        save_strategy="no",  # the adapter is saved at the end; no intermediate checkpoints
        report_to=[],
        max_length=1024,
    )

    print(f"\n=== Training config ===")
    print(f"  epochs={epochs}  per_device_batch={per_device_batch}  grad_accum={grad_accum}  "
          f"effective_batch={per_device_batch * grad_accum}")
    print(f"  LoRA: r={lora_config.r} alpha={lora_config.lora_alpha} "
          f"dropout={lora_config.lora_dropout} target_modules={_LORA_TARGET_MODULES}")

    torch.cuda.reset_peak_memory_stats()
    trainer = SFTTrainer(model=model, args=sft_config, train_dataset=ds, processing_class=tokenizer)

    print("\n=== TRAINING START ===")
    train_start = time.time()
    result = trainer.train()
    train_elapsed = time.time() - train_start
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9

    print(f"\n=== TRAINING DONE ===")
    print(f"  wall-clock time: {train_elapsed:.1f}s ({train_elapsed / 60:.1f} min)")
    print(f"  peak VRAM during training: {peak_vram_gb:.2f} GB")
    print(f"  final training loss (trainer-reported): {result.training_loss:.4f}")
    print(f"  full log history (loss curve):")
    for entry in trainer.state.log_history:
        if "loss" in entry:
            print(f"    step={entry.get('step')} epoch={entry.get('epoch'):.2f} loss={entry['loss']:.4f}")

    _ADAPTER_OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(_ADAPTER_OUT))
    tokenizer.save_pretrained(str(_ADAPTER_OUT))
    print(f"\n=== Adapter saved to {_ADAPTER_OUT} (adapter only, base model NOT merged) ===")

    print("\n=== Cleanup ===")
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    print(f"  VRAM after cleanup (torch): {torch.cuda.memory_allocated() / 1e9:.3f} GB")
    print("  nvidia-smi after cleanup:")
    print(_nvidia_smi_text())


if __name__ == "__main__":
    main()
