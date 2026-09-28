"""
Build the handwritten-working test photo, one of the layout-type fixtures required by the Phase 4
exit criteria (design specification Section 15: "at least one real test photo per major layout
type"). The source is a purchased student's handwritten working, not a rendered one (Issue 197).

Unlike the earlier fixture builders (Issues 184 and 193), which only apply geometric and lighting
degradation, this script also makes one deliberate content edit before degrading the photo. The
working correctly reaches "Total coins at first = 7 x 10 = 70", but the boxed "Ans:" line is
changed from "70" to "10". This mirrors a common PSLE mistake: writing down the unit value `u=10`
instead of completing the final multiplication. It tests Issue 1 (over-correction): a faithful
reader must transcribe the "10" that is on the page, even though it contradicts the working above
it. "Fixing" it to "70" is the silent answer-key corruption that Issue 1 exists to catch.

The replacement "10" is not drawn or synthesised. It is a handwritten "10" cropped from elsewhere on
the same page (from "No. of 50c coin = 2u+10", same pen and same student), scaled up and pasted over
the erased "70" through an ink-darkness mask so the paper background shows through unchanged.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PAGE = _REPO_ROOT / "data" / "Handwritten" / "GS009_q4_coins_handwritten.png"
OUT_PATH = Path(__file__).resolve().parent / "_fixtures" / "handwritten_error_phone_photo.jpg"


def _introduce_real_deliberate_error(page: np.ndarray) -> np.ndarray:
    """Overwrite the correct 'Ans: 70' with a deliberately wrong 'Ans: 10'.

    See the module docstring for why this value and location were chosen.
    """
    page = page.copy()
    bg_color = np.array([255.0, 255.0, 255.0])
    # Erase the '70' (calibrated bounding box that leaves the 'Ans:' label untouched).
    page[900:960, 748:905] = bg_color.astype(np.uint8)

    # A clean handwritten '10' from elsewhere on the same page ('No. of 50c coin = 2u+10'),
    # scaled to the footprint of the erased '70'.
    glyph = page[738:772, 466:504]
    gh, gw = glyph.shape[:2]
    scale = 1.5
    new_w, new_h = int(gw * scale), int(gh * scale)
    glyph_big = cv2.resize(glyph, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(glyph_big, cv2.COLOR_BGR2GRAY)
    ink_mask = gray < 180  # ink only, so the paper background stays transparent

    x0 = 748 + (157 - new_w) // 2
    y0 = 953 - new_h
    roi = page[y0:y0 + new_h, x0:x0 + new_w]
    roi[ink_mask] = glyph_big[ink_mask]
    page[y0:y0 + new_h, x0:x0 + new_w] = roi
    return page


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    page = cv2.imread(str(SOURCE_PAGE))
    if page is None:
        raise FileNotFoundError(f"Real source page not found at {SOURCE_PAGE}")

    page = _introduce_real_deliberate_error(page)

    # Visually confirmed crop region for Q4 alone, with a small margin.
    crop = page[150:1000, 0:1414]
    ph, pw = crop.shape[:2]

    # Same phone-photo degradation as the printed-MCQ fixture
    # (`_build_single_question_test_photo.py`, Issue 193): wood-desk canvas, skew, lighting
    # gradient, light blur and JPEG re-compression.
    canvas_w, canvas_h = int(pw * 1.4), int(ph * 1.5)
    canvas = np.full((canvas_h, canvas_w, 3), (60, 90, 130), dtype=np.uint8)

    angle_deg = 10.0
    rot_matrix = cv2.getRotationMatrix2D((pw / 2, ph / 2), angle_deg, 0.85)
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
    print(f"Wrote real handwritten-error test photo to {OUT_PATH} (canvas {canvas_w}x{canvas_h})")


if __name__ == "__main__":
    main()
