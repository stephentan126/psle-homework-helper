"""
GPU memory helpers for evaluation scripts that load more than one model.

Loading several candidate models in one process without freeing GPU memory between them
eventually fails with an out-of-memory error that looks unrelated to its cause, stopping a
bake-off partway through. Scripts should free memory between candidates as described below.

`free_gpu()` cannot delete a caller's variables: `del obj` inside a function only removes the
local parameter binding (Issue 163). A test allocating a 1GB CUDA tensor confirmed the memory
stayed allocated until the caller also ran `del x`. The OCR bake-off runners hid this because
each runs one candidate per process, and the operating system reclaims VRAM on exit. It matters
when a script loops over several candidates in one process, as the LLM bake-off does.

Callers must therefore `del` their own `model` and `tokenizer` variables before calling
`free_gpu()`, as `scripts/setup/environment_canary.py` does. `free_gpu()` runs
`gc.collect()` and `torch.cuda.empty_cache()`, then warns if VRAM did not drop, so a missing
`del` is reported rather than passing silently.
"""

import gc
import time
import warnings
from contextlib import contextmanager

import torch


def free_gpu(*objects) -> None:
    """Runs garbage collection and empties the CUDA cache.

    Callers must `del` their own model and tokenizer variables before calling this, because a
    function cannot remove its caller's bindings (Issue 163). A warning is raised if VRAM did not
    drop, which indicates that a reference is still held somewhere.

    Args:
        *objects: Accepted so existing call sites such as `free_gpu(model, tokenizer)` still
            work. They are not freed by this function.
    """
    vram_before = torch.cuda.memory_allocated() / (1024**3)
    gc.collect()
    torch.cuda.empty_cache()
    vram_after = torch.cuda.memory_allocated() / (1024**3)
    if objects and vram_after >= vram_before - 1e-6:
        warnings.warn(
            f"free_gpu() called with {len(objects)} object(s) but VRAM did not drop "
            f"({vram_before:.3f} GB -> {vram_after:.3f} GB) — this almost certainly means the "
            f"caller still holds its own reference (e.g. a `model`/`tokenizer` local variable) "
            f"that was never `del`'d. free_gpu() cannot delete a caller's own variable for it "
            f"(Issue 163) — `del model; del tokenizer` etc. in the CALLER's own scope BEFORE "
            f"calling free_gpu(), the same pattern environment_canary.py already uses.",
            stacklevel=2,
        )


def current_vram_gb() -> float:
    """Returns the currently allocated VRAM in GB."""
    return torch.cuda.memory_allocated() / (1024**3)


@contextmanager
def measure_load(candidate_name: str):
    """Times a model load and reports the change in allocated VRAM.

    Example:
        with measure_load("dots.mocr") as m:
            model = load_dots_mocr()
        # m.load_time_s and m.vram_delta_gb are now populated
    """

    class _Measurement:
        load_time_s: float = 0.0
        vram_before_gb: float = 0.0
        vram_after_gb: float = 0.0
        vram_delta_gb: float = 0.0

    measurement = _Measurement()
    measurement.vram_before_gb = current_vram_gb()
    start = time.time()
    try:
        yield measurement
    finally:
        measurement.load_time_s = time.time() - start
        measurement.vram_after_gb = current_vram_gb()
        measurement.vram_delta_gb = measurement.vram_after_gb - measurement.vram_before_gb
        print(
            f"[{candidate_name}] load_time={measurement.load_time_s:.2f}s "
            f"vram_delta={measurement.vram_delta_gb:.2f}GB"
        )
