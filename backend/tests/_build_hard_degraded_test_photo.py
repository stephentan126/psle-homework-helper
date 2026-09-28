"""
Build a heavily degraded version of the Q1 crop used by `_build_single_question_test_photo.py`.

Used for Phase 4 step 4 (development log, Issue 44): a photo the model is likely to misread gives a
data point to compare the logprob confidence signal against the easy case, which is already
validated. The degradation combines heavy Gaussian blur (an out-of-focus camera), strong JPEG
re-compression, a harsher lighting gradient and a steeper skew. These are standard phone-photo
failure modes rather than artificial corruption.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PAGE = _REPO_ROOT / "data" / "extracted" / "pages" / "P6-Maths-2021-CA1-Henry-Park" / "2.png"
OUT_PATH = Path(__file__).resolve().parent / "_fixtures" / "hard_degraded_phone_photo.jpg"


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    page = cv2.imread(str(SOURCE_PAGE))
    if page is None:
        raise FileNotFoundError(f"Real source page not found at {SOURCE_PAGE}")

    crop = page[750:1420, 80:2420]
    ph, pw = crop.shape[:2]

    canvas_w, canvas_h = int(pw * 1.6), int(ph * 2.2)
    canvas = np.full((canvas_h, canvas_w, 3), (60, 90, 130), dtype=np.uint8)

    angle_deg = 25.0  # steeper than the easy case's 12 degrees
    rot_matrix = cv2.getRotationMatrix2D((pw / 2, ph / 2), angle_deg, 0.9)
    cos, sin = abs(rot_matrix[0, 0]), abs(rot_matrix[0, 1])
    new_w, new_h = int(ph * sin + pw * cos), int(ph * cos + pw * sin)
    rot_matrix[0, 2] += (new_w / 2) - pw / 2
    rot_matrix[1, 2] += (new_h / 2) - ph / 2
    rotated = cv2.warpAffine(crop, rot_matrix, (new_w, new_h), borderValue=(60, 90, 130))

    mask = np.any(rotated != (60, 90, 130), axis=2)
    y_off = (canvas_h - new_h) // 2
    x_off = (canvas_w - new_w) // 2
    roi = canvas[y_off:y_off + new_h, x_off:x_off + new_w]
    roi[mask] = rotated[mask]
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = roi

    yy, xx = np.mgrid[0:canvas_h, 0:canvas_w]
    gradient = 1.0 - 0.55 * ((xx / canvas_w + yy / canvas_h) / 2.0)  # much stronger than
                                                                       # the easy case's 0.3
    lit = np.clip(canvas.astype(np.float32) * gradient[:, :, None], 0, 255).astype(np.uint8)
    # Heavy out-of-focus blur, much stronger than the easy case's sigmaX=0.6 smoothing.
    blurred = cv2.GaussianBlur(lit, (9, 9), sigmaX=4.0)
    # Resolution loss, as from a photo taken further away or sensor downsampling. Down- then
    # up-sampling destroys thin character strokes far more reliably than blur alone, to which
    # Qwen3-VL-8B-Instruct proved fairly robust (Issue 184).
    small = cv2.resize(blurred, (blurred.shape[1] // 16, blurred.shape[0] // 16),
                        interpolation=cv2.INTER_AREA)
    upscaled = cv2.resize(small, (blurred.shape[1], blurred.shape[0]),
                           interpolation=cv2.INTER_LINEAR)
    success, encoded = cv2.imencode(".jpg", upscaled, [cv2.IMWRITE_JPEG_QUALITY, 15])
    if not success:
        raise RuntimeError("Real JPEG encode failed.")
    final = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    cv2.imwrite(str(OUT_PATH), final)
    print(f"Wrote real hard-degraded test photo to {OUT_PATH} (canvas {canvas_w}x{canvas_h})")


if __name__ == "__main__":
    main()
