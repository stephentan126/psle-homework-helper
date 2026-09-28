"""
Build a single-question test photo for the end-to-end test in Issue 193.

Crops Q1 alone from the corpus page used for the preprocessing validation in Issue 184 and applies
the same phone-photo degradation as
`evaluation/model_selection/photo_transcription/build_preprocessing_test_image.py`. The first
end-to-end run used a whole-page photo and correctly found no confident match (Issue 193), because
the corpus and the `student_submissions.question_text` schema hold one question per submission, not
a whole page. This fixture matches that intended shape.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PAGE = _REPO_ROOT / "data" / "extracted" / "pages" / "P6-Maths-2021-CA1-Henry-Park" / "2.png"
OUT_PATH = Path(__file__).resolve().parent / "_fixtures" / "single_question_phone_photo.jpg"


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    page = cv2.imread(str(SOURCE_PAGE))
    if page is None:
        raise FileNotFoundError(f"Real source page not found at {SOURCE_PAGE}")

    # Visually confirmed crop region for Q1 alone, with a small margin, on the 2480x3507 source
    # page.
    crop = page[750:1420, 80:2420]
    ph, pw = crop.shape[:2]

    canvas_w, canvas_h = int(pw * 1.4), int(ph * 1.6)
    canvas = np.full((canvas_h, canvas_w, 3), (60, 90, 130), dtype=np.uint8)  # "wood" brown

    angle_deg = 12.0
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
    gradient = 1.0 - 0.3 * ((xx / canvas_w + yy / canvas_h) / 2.0)
    lit = np.clip(canvas.astype(np.float32) * gradient[:, :, None], 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(lit, (3, 3), sigmaX=0.6)
    success, encoded = cv2.imencode(".jpg", blurred, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not success:
        raise RuntimeError("Real JPEG encode failed.")
    final = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    cv2.imwrite(str(OUT_PATH), final)
    print(f"Wrote real single-question test photo to {OUT_PATH} (canvas {canvas_w}x{canvas_h})")


if __name__ == "__main__":
    main()
