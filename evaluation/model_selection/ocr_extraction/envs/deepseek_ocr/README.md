# deepseek_ocr env, DONE

Isolated venv for **DeepSeek-OCR** (`deepseek-ai/DeepSeek-OCR`, 3B params, DeepSeek-OCR family),
per docs/MODEL_SELECTION.md Section 2. Built with `uv` (Python 3.12, per the README).

**flash-attn==2.7.3 (README-pinned) deliberately NOT installed**, fails to build on this machine
(`OSError: CUDA_HOME environment variable is not set`, no system-wide CUDA toolkit). Verified
working fallback: omit `_attn_implementation='flash_attention_2'` entirely, model runs on its
default attention path. See `requirements.txt` for the full writeup (three gotchas: CPU-only
torch wheel, torchvision silently upgrading the torch pin, and this one).

**Run order (smallest-to-largest, run plan Section 4): 5 of 7, COMPLETE.**

Real numbers:
- Load time: 18-20s, peak VRAM **6.39 GB** (fits 8GB gate: YES, but noticeably higher than every
  other candidate so far, likely the cost of running without flash-attn's memory efficiency)
- 18/18 gold pages OK via `model.infer()` with the grounding prompt

**Real bug found and fixed (own code, not the model):** first run, 2/18 pages (GS010, GS013)
failed with `UnicodeEncodeError`, the model's own internal logging prints extracted Unicode
content (checkmarks, arrows) to stdout, which crashes on Windows' default cp1252 console
encoding. Fixed by forcing UTF-8 stdout at the top of the runner; re-ran, 18/18 OK. Second bug,
also own code: `infer()` saves to `result.mmd`, not `*.md` as first assumed, the glob matched
nothing, so every page's `markdown` field came back empty despite `error: None`. Caught by
spot-checking real content rather than trusting the None-error status alone (same discipline as
every other candidate). Fixed the glob pattern and backfilled all 18 records from the
already-saved `.mmd` files, no model re-run needed.

Raw output: `../raw/deepseek_ocr/*.json` (gitignored, same policy as prior candidates).
