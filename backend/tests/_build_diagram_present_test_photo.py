"""
Build the diagram-present test photo, one of the layout-type fixtures required by the Phase 4 exit
criteria (design specification Section 15: "at least one real test photo per major layout type").

The source is a purchased student paper with a parallelogram and triangle figure, annotated by hand
with angles marked on the diagram itself.

Purpose (Issue 197): diagram interpretation (Section 6) is not built yet, so this fixture does not
test whether the system can solve a diagram question. It tests whether the live pipeline degrades
safely when a diagram is present. The reader prompt (`vlm_service.FAITHFUL_TRANSCRIPTION_PROMPT`)
tells the model to emit a bare '[DIAGRAM PRESENT]' tag instead of interpreting the figure, so
`has_diagram` should end up True. Unlike the handwritten-error fixture, no content edit is made:
the working here is already correct (83 degrees, matching the key), and the property under test is
diagram handling, not faithfulness under error.

Only part (a) is cropped (diagram, reasoning and answer), leaving out part (b)'s table. This keeps
the fixture to one question per submission, the shape the schema and corpus assume (see Issue 193
and `_build_single_question_test_photo.py`).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PAGE = _REPO_ROOT / "data" / "Handwritten" / "GS010_q9_parallelogram_handwritten.png"
OUT_PATH = Path(__file__).resolve().parent / "_fixtures" / "diagram_present_phone_photo.jpg"


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    page = cv2.imread(str(SOURCE_PAGE))
    if page is None:
        raise FileNotFoundError(f"Real source page not found at {SOURCE_PAGE}")

    # Visually confirmed crop region for Q9(a) alone (diagram, reasoning and answer line),
    # excluding part (b)'s table.
    crop = page[150:1120, 0:1414]
    ph, pw = crop.shape[:2]

    # Same phone-photo degradation as the other two layout-type fixtures.
    canvas_w, canvas_h = int(pw * 1.4), int(ph * 1.5)
    canvas = np.full((canvas_h, canvas_w, 3), (60, 90, 130), dtype=np.uint8)

    angle_deg = 8.0
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
    print(f"Wrote real diagram-present test photo to {OUT_PATH} (canvas {canvas_w}x{canvas_h})")


if __name__ == "__main__":
    main()
