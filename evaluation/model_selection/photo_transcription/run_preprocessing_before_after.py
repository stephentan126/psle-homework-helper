"""
Phase 4 step 2 real validation: build a FAIR real before/after comparison.

- 'before_capped.jpg': the raw, degraded phone-photo test image, resized to the SAME size cap
  the real pipeline applies (`resize_for_inference()`) but with NO deskew/dewarp/crop/lighting
  correction, isolates what deskew+crop+lighting specifically contribute, not conflated with
  the separate, real image-size finding (the development log: an uncapped ~20MP raw photo
  made the model run for 60+ real minutes on one image with no completion, confirmed still
  genuinely computing via nvidia-smi, not a hang).
- 'after_full.jpg': the FULL real `preprocess_live_photo()` pipeline output (resize + dewarp +
  crop + lighting), the one Phase 4's own real code path will actually use.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "backend"))

import cv2  # noqa: E402

from app.pipeline.photo_preprocessing import (  # noqa: E402
    preprocess_live_photo,
    resize_for_inference,
)

TEST_DIR = Path(__file__).resolve().parent / "preprocessing_test"
BEFORE_PATH = TEST_DIR / "before_raw_phone_photo.jpg"


def main() -> None:
    image = cv2.imread(str(BEFORE_PATH))
    if image is None:
        raise FileNotFoundError(f"Real before-image not found: {BEFORE_PATH}")

    # Fair baseline: same size cap, no other correction.
    before_capped = resize_for_inference(image)
    before_path = TEST_DIR / "before_capped.jpg"
    cv2.imwrite(str(before_path), before_capped)
    print(f"Before (size-capped only): shape={before_capped.shape} -> {before_path}")

    # Full real pipeline.
    processed, diagnostics = preprocess_live_photo(image)
    after_path = TEST_DIR / "after_full.jpg"
    cv2.imwrite(str(after_path), processed)
    print(f"After (full real pipeline): shape={processed.shape}, diagnostics={diagnostics} "
          f"-> {after_path}")


if __name__ == "__main__":
    main()
