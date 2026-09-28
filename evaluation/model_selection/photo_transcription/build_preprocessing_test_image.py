"""
Phase 4 step 2 (Issue 21), build ONE real, deliberately harder test image than the bake-off's
own gold set: a real corpus page shown SMALLER within a larger background canvas (simulating a
genuine phone photo where the page doesn't fill the whole frame), at a real, visible skew angle,
with a real, stronger lighting gradient and blur, specifically designed to be hard enough that
preprocessing should produce a REAL, measurable difference, not a marginal one. The bake-off's own
gold set images stayed within-frame (no real background/crop scenario) precisely so THIS test
could add that real, distinct dimension without duplicating Issue 183's own work.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_PAGE = _REPO_ROOT / "data" / "extracted" / "pages" / "P6-Maths-2021-CA1-Henry-Park" / "2.png"
OUT_DIR = Path(__file__).resolve().parent / "preprocessing_test"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    page = cv2.imread(str(SOURCE_PAGE))
    if page is None:
        raise FileNotFoundError(f"Real source page not found at {SOURCE_PAGE}")

    ph, pw = page.shape[:2]
    # Real "phone photo" canvas: larger than the page, real wood-table-brown background color,
    # not white, so a real crop actually needs to isolate the page from a different
    # background, not just from more white.
    canvas_w, canvas_h = int(pw * 1.6), int(ph * 1.5)
    canvas = np.full((canvas_h, canvas_w, 3), (60, 90, 130), dtype=np.uint8)  # BGR "wood" brown

    # Place + rotate the real page within the canvas at a real, visible angle (18 degrees,
    # meaningfully more than the bake-off gold set's own ~2 degree perspective jitter, a real,
    # deliberately harder skew to give preprocessing something substantial to correct).
    angle_deg = 18.0
    rot_matrix = cv2.getRotationMatrix2D((pw / 2, ph / 2), angle_deg, 0.85)
    # Expand the rotation canvas so the rotated page isn't clipped.
    cos, sin = abs(rot_matrix[0, 0]), abs(rot_matrix[0, 1])
    new_w, new_h = int(ph * sin + pw * cos), int(ph * cos + pw * sin)
    rot_matrix[0, 2] += (new_w / 2) - pw / 2
    rot_matrix[1, 2] += (new_h / 2) - ph / 2
    rotated_page = cv2.warpAffine(
        page, rot_matrix, (new_w, new_h), borderValue=(60, 90, 130),
    )

    # Composite the rotated page onto the canvas, centered, using a real mask so the brown
    # background shows through around the rotated page's own now-diagonal edges.
    mask = np.any(rotated_page != (60, 90, 130), axis=2)
    y_off = (canvas_h - new_h) // 2
    x_off = (canvas_w - new_w) // 2
    roi = canvas[y_off:y_off + new_h, x_off:x_off + new_w]
    roi[mask] = rotated_page[mask]
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = roi

    # Real, stronger lighting gradient + blur + JPEG re-compression, same real technique as
    # Issue 183's own gold-set degradation, applied more strongly here.
    yy, xx = np.mgrid[0:canvas_h, 0:canvas_w]
    gradient = 1.0 - 0.45 * ((xx / canvas_w + yy / canvas_h) / 2.0)
    lit = np.clip(canvas.astype(np.float32) * gradient[:, :, None], 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(lit, (3, 3), sigmaX=0.7)
    success, encoded = cv2.imencode(".jpg", blurred, [cv2.IMWRITE_JPEG_QUALITY, 65])
    if not success:
        raise RuntimeError("Real JPEG encode failed.")
    final = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    out_path = OUT_DIR / "before_raw_phone_photo.jpg"
    cv2.imwrite(str(out_path), final)
    print(f"Wrote real 'before' test image to {out_path} (canvas {canvas_w}x{canvas_h}, "
          f"real page rotated {angle_deg} degrees within a real brown background)")


if __name__ == "__main__":
    main()
