"""
Photo tilt detection for the retake-photo prompt (Issue 368).

The page-boundary detection in `detect_and_dewarp_page()` (`photo_preprocessing.py`) does not
work on a tight single-question crop. It requires the largest contour to cover at least 20% of
the frame, which suits a whole-page photo, and reports "no boundary found" on a crop whether or
not the photo is straight.

Method: Canny edges and a probabilistic Hough line transform find line segments, and the median
of their angles is taken as the dominant text-line angle. The median is used rather than the mean
so that the many short strokes in handwriting or a geometric diagram cannot swing the result.

An earlier design used `cv2.minAreaRect` on dilated text-blob contours and failed on every test
photo, for two reasons:
  1. Otsu's global threshold on a phone photo (paper on a desk, not a clean scan) treated the
     whole darker background as one foreground blob. Adaptive thresholding fixed this.
  2. The paper's physical edge still showed as an outline under adaptive thresholding, and a
     horizontal-only dilation kernel cannot merge characters on a tilted line, because they are
     offset vertically as well as horizontally. Single characters were left unmerged and gave
     meaningless angles near 90 degrees.
Both were confirmed with debug runs (see `scripts/dev/skew_detection_manual_check.py`). The Hough
approach does not need to know the text angle in advance, and was verified end to end against all
5 test photos (Issue 368).
"""
from __future__ import annotations

import cv2
import numpy as np

# Based on measurement (Issue 368): every test photo that the VLM read successfully had between 6.8
# and 17.2 degrees of camera tilt, so a threshold of a few degrees would have asked for retakes of
# legible photos. The value sits above the highest legible tilt measured (17.18 degrees) with a
# margin. Limitation: no photo too tilted to read was available, so the threshold is checked
# against false positives only, not true positives. Revisit when deliberately tilted photos exist.
DEFAULT_SKEW_THRESHOLD_DEGREES = 20.0

# Margin cropped from each edge before detection, so the paper's physical border (a confirmed false
# signal, see module docstring) never enters the Hough line search.
_EDGE_MARGIN_FRACTION = 0.06

_CANNY_LOW, _CANNY_HIGH = 50, 150
_HOUGH_THRESHOLD = 40
_MIN_LINE_LENGTH_PX = 25
_MAX_LINE_GAP_PX = 8
# A segment longer than this fraction of the diagonal is almost certainly the paper's edge or a
# ruled margin line, not a text stroke, so it is excluded.
_MAX_LINE_LENGTH_FRACTION_OF_DIAGONAL = 0.5
_MIN_REAL_LINES_TO_TRUST = 5


def _line_angles(gray: np.ndarray) -> list[float]:
    """Returns the angle of each detected line segment in degrees, folded into (-45, 45].

    Folding makes segments running in opposite directions along the same line give the same angle.
    """
    h, w = gray.shape[:2]
    mx, my = int(w * _EDGE_MARGIN_FRACTION), int(h * _EDGE_MARGIN_FRACTION)
    cropped = gray[my: h - my, mx: w - mx]
    blurred = cv2.GaussianBlur(cropped, (5, 5), 0)
    edges = cv2.Canny(blurred, _CANNY_LOW, _CANNY_HIGH)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=_HOUGH_THRESHOLD,
                            minLineLength=_MIN_LINE_LENGTH_PX, maxLineGap=_MAX_LINE_GAP_PX)
    if lines is None:
        return []

    diagonal = (cropped.shape[0] ** 2 + cropped.shape[1] ** 2) ** 0.5
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if length > _MAX_LINE_LENGTH_FRACTION_OF_DIAGONAL * diagonal:
            continue
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # fold to (-45, 45]
        if angle > 90:
            angle -= 180
        elif angle <= -90:
            angle += 180
        if angle > 45:
            angle -= 90
        elif angle <= -45:
            angle += 90
        angles.append(angle)
    return angles


def measure_skew_angle(image: np.ndarray) -> float | None:
    """Returns the dominant line angle in degrees, where 0 is horizontal.

    Returns None rather than 0.0 when fewer than `_MIN_REAL_LINES_TO_TRUST` segments are found
    (for example a blank or illegible photo), so "could not measure" is never read as "straight".
    The deliberately degraded test photo gives 0 lines and returns None.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    angles = _line_angles(gray)
    if len(angles) < _MIN_REAL_LINES_TO_TRUST:
        return None
    return float(np.median(angles))


def is_photo_too_tilted(image: np.ndarray, threshold_degrees: float = DEFAULT_SKEW_THRESHOLD_DEGREES) -> tuple[bool, float | None]:
    """Returns (is_too_tilted, measured_angle_or_None).

    A None angle is not treated as too tilted. This fails open, as `detect_and_dewarp_page()` does
    when no boundary is found, because asking for a retake of a good photo is worse than
    occasionally missing a tilt on a hard-to-read one, which the VLM confidence gate (Issue 194)
    still catches downstream.
    """
    angle = measure_skew_angle(image)
    if angle is None:
        return False, None
    return abs(angle) > threshold_degrees, angle
