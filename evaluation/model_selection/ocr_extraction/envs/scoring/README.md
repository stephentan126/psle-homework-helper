# scoring env, DONE

Shared, lightweight venv for the normalize+score step (Phase 1 step 2), NOT one of the 7
per-model bake-off candidate envs. CPU-only, no GPU, no model loading, this step only parses
already-saved raw output (`../../raw/{model}/*.json`) and scores it against
the gold-set table in docs/DATA_AND_EVALUATION.md. Per-model isolation was for conflicting ML dependency stacks
(torch/transformers version pins per model); that problem doesn't exist here, so one shared venv
is correct, not a shortcut.

Built with `uv` (Python 3.12, matching the rest of the bake-off's Python line).

Dependency: `jiwer` (CER/WER against `question_text_clean`). Everything else the scoring script
uses is stdlib (`json`, `csv`, `re`, `pathlib`).

Run: `envs/scoring/.venv/Scripts/python.exe normalize_and_score.py` (from `evaluation/model_selection/ocr_extraction/`).
