"""
Preprocessing for student photos (design specification, Section 6; Issue 21): deskew, dewarp,
crop and lighting normalisation before the photo reaches the document reader
(Qwen3-VL-8B-Instruct, Issue 183).

A standard document-scanner pipeline built on `opencv-contrib-python` (Issue 63):
1. `detect_and_dewarp_page()` finds the page boundary (the largest roughly quadrilateral contour
   after edge detection) and corrects the perspective to a cropped, top-down rectangle. If no
   confident boundary is found, the original image is returned unchanged with
   `found_boundary=False`, rather than applying a wrong crop.
2. `normalize_lighting()` estimates the background lighting with a large-kernel Gaussian blur
   (which removes text and fine detail but keeps the broad gradient) and divides the image by it.
   This corrects the uneven room lighting of a phone photo, the same artefact that the gold-set
   degradation in Issue 183 simulates.

Geometry runs before lighting: perspective distorts the direction of a warped page's lighting
gradient, so lighting correction works better on an image that is already rectangular. This is
the conventional order for document scanners.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Orders 4 points as top-left, top-right, bottom-right, bottom-left.

    Top-left has the smallest x+y and bottom-right the largest; top-right has the smallest y-x
    difference and bottom-left the largest.
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def detect_and_dewarp_page(image: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Detects the page boundary and corrects it to a cropped, top-down rectangle.

    Returns `(processed_image, found_boundary)`. The flag tells callers, and the logging in
    Section 10.5, whether a correction was applied or the original image was returned because no
    confident boundary was found. A low-confidence detection is never used to crop.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image, False

    largest = max(contours, key=cv2.contourArea)
    image_area = image.shape[0] * image.shape[1]
    # A page boundary covers a substantial part of the photo. A small largest contour (a stray
    # text block or shadow edge) is rejected rather than used to crop.
    if cv2.contourArea(largest) < 0.2 * image_area:
        return image, False

    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
    if len(approx) != 4:
        return image, False  # not a clean quadrilateral, so fall back to the original

    pts = approx.reshape(4, 2).astype(np.float32)

    # Degenerate case found in testing: on a non-page image (confirmed with random noise) the edge
    # map is so dense that Canny and dilation trace the whole image border as the largest contour.
    # Its four corners sit on the image corners and pass the area and point-count checks above.
    # A photographed page's corners almost never land exactly on the image border (the camera
    # captures some background, and perspective keeps corners off the edge), so a detection with
    # all four corners within a few pixels of the border is rejected.
    h, w = image.shape[:2]
    margin = 2
    corners_on_border = np.count_nonzero(
        ((pts[:, 0] <= margin) | (pts[:, 0] >= w - 1 - margin))
        & ((pts[:, 1] <= margin) | (pts[:, 1] >= h - 1 - margin)),
    )
    if corners_on_border == 4:
        return image, False  # whole-image-border artefact, not a page boundary

    rect = _order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    if max_width < 10 or max_height < 10:
        return image, False  # degenerate size

    dst = np.array([
        [0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1],
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
    return warped, True


def normalize_lighting(image: np.ndarray) -> np.ndarray:
    """Corrects uneven lighting by dividing by a blurred background estimate.

    Works on luminance for both greyscale and BGR input, and returns the same number of channels
    it was given.
    """
    is_color = image.ndim == 3
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if is_color else image

    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=25, sigmaY=25)
    background = np.clip(background.astype(np.float32), 1, 255)  # avoid division by zero
    normalized = (gray.astype(np.float32) / background) * 255.0
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    if is_color:
        return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
    return normalized


_MAX_INPUT_DIMENSION = 2000  # see resize_for_inference()


def resize_for_inference(image: np.ndarray, max_dimension: int = _MAX_INPUT_DIMENSION) -> np.ndarray:
    """Scales the image down so its longer side is at most `max_dimension` pixels.

    Image resolution drives the VLM's image-token count. An uncapped phone photo (the test image
    was 3968x5260, about 20 megapixels) ran on the reader for over 60 minutes without finishing,
    with `nvidia-smi` showing 100% GPU utilisation throughout, so it was computing, not hung. A
    size cap is therefore a necessary step for a live request. 2000px keeps small print and fine
    diagram detail legible (well above the roughly 1400px at which the bake-off gold set confirmed
    faithful transcription, Issue 183) while bounding the worst-case token cost.
    """
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_dimension:
        return image
    scale = max_dimension / longest
    return cv2.resize(
        image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA,
    )


def preprocess_live_photo(image: np.ndarray) -> Tuple[np.ndarray, dict]:
    """Preprocesses a student photo: resize, then dewarp and crop, then normalise lighting.

    Resizing runs first so a large upload does not pay the full cost of edge and contour
    detection. Returns `(processed_image, diagnostics)`, where diagnostics records whether a page
    boundary was found, for logging (Section 10.5). Low-confidence detection never raises; it
    falls back to no crop and reports that. Callers that need the fallback path (Issue 56) check
    `diagnostics` themselves.
    """
    resized = resize_for_inference(image)
    dewarped, found_boundary = detect_and_dewarp_page(resized)
    normalized = normalize_lighting(dewarped)
    return normalized, {"page_boundary_found": found_boundary}
