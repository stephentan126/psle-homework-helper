#!/usr/bin/env bash
# Self-restarting wrapper around (Issue 215):
#   evaluation/model_selection/ocr_extraction/runners/run_batched.py --reader deepseek_ocr
#
# Why: the full-corpus DeepSeek-OCR secondary-reader pass (Issue 214) was killed three times in its
# first 70 minutes or so by the environment's background-task supervisor for low system memory,
# after varying times (55 minutes, then about 7). No data was lost, because run_batched.py
# checkpoints each page, but each kill sat idle until someone relaunched it by hand. This loop is
# itself only a few MB. The varying time to kill suggests the supervisor targets the large
# python/model process (a fixed memory budget would kill at a more consistent point), so the loop
# survives its child dying and relaunches it straight away.
#
# Resumability: the pending-page count is recomputed from disk on every iteration, using the same
# check as get_pending_pages() in run_batched.py, and the loop exits as soon as nothing is left.
#
# Known-bad pages (Issue 221): a page with a ruler/tick-mark diagram
# (P6_Maths_2022_SA2_nanyang/44) sends the model into a repetition loop that never completes. Run
# alone at --batch-size 1, it timed out twice with no output, while every other page in its
# original batch completed. Decoding is deterministic (do_sample=False), so the page would fail the
# same way forever and this loop would retry it indefinitely. `--exclude-known-bad` makes
# run_batched.py skip, before batching, any page that has failed at least --known-bad-threshold
# times in a single-page batch (see get_known_bad_pages() in run_batched.py for why only
# single-page batches count as evidence), and always report it in
# data/extracted/raw/deepseek_ocr/_known_bad_pages.json and in the run's output. Such pages remain a
# visible open gap that needs another transcription route; they are never dropped silently.
#
# Usage (from the repository root):
#   scripts/extraction/run_deepseek_ocr_resilient.sh

set -u
cd "$(dirname "$0")/../.."  # repository root (this script lives in scripts/<group>/)

PY="backend/.venv/Scripts/python.exe"
IMAGE_DIR="data/extracted/pages"
OUTPUT_DIR="data/extracted/raw/deepseek_ocr"
KNOWN_BAD_THRESHOLD=2

pending_count() {
  # Excludes known-bad pages the same way run_batched.py's --exclude-known-bad does (by reading
  # the _known_bad_pages.json it writes), so the exit condition agrees with what run_batched.py
  # will attempt. Otherwise pending would never reach 0 and the loop would relaunch forever over a
  # page that is never attempted.
  "$PY" -c "
import json
from pathlib import Path
image_dir = Path('$IMAGE_DIR')
output_dir = Path('$OUTPUT_DIR')
known_bad_path = output_dir / '_known_bad_pages.json'
known_bad = set()
if known_bad_path.exists():
    known_bad = set(json.loads(known_bad_path.read_text(encoding='utf-8')).keys())
n = sum(
    1 for p in image_dir.rglob('*.png')
    if not (output_dir / p.parent.name / f'{p.stem}.json').exists()
    and f'{p.parent.name}/{p.stem}' not in known_bad
)
print(n)
"
}

attempt=0
while true; do
  pending=$(pending_count)
  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  if [ "$pending" -eq 0 ]; then
    known_bad_path="$OUTPUT_DIR/_known_bad_pages.json"
    if [ -f "$known_bad_path" ]; then
      skipped=$("$PY" -c "import json,sys; print(len(json.loads(open(sys.argv[1],encoding='utf-8').read())))" "$known_bad_path")
    else
      skipped=0
    fi
    echo "[$ts] ALL DONE -- 0 pages pending ($skipped known-bad page(s) excluded, see $known_bad_path). Exiting cleanly."
    break
  fi
  attempt=$((attempt + 1))
  echo "[$ts] attempt $attempt: $pending page(s) pending, launching run_batched.py..."
  "$PY" evaluation/model_selection/ocr_extraction/runners/run_batched.py \
    --reader deepseek_ocr \
    --image-dir "$IMAGE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size 10 \
    --exclude-known-bad \
    --known-bad-threshold "$KNOWN_BAD_THRESHOLD"
  exit_code=$?
  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "[$ts] run_batched.py exited with code $exit_code. Re-checking pending state in 5s..."
  sleep 5
done
