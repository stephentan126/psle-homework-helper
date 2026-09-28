"""
Tests for the skew-angle detector on phone photos rather than synthetic images (Issue 368).

The photos are the `backend/tests/_fixtures/*.jpg` fixtures shared with
`test_live_photo_pipeline.py`. The angle ranges come from measurements; see the detector's module
docstring and Issue 368.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from app.pipeline.skew_detection import DEFAULT_SKEW_THRESHOLD_DEGREES, is_photo_too_tilted, measure_skew_angle

_FIXTURES = Path(__file__).resolve().parent / "_fixtures"


def _load(name: str):
    img = cv2.imread(str(_FIXTURES / name))
    assert img is not None, f"real fixture missing: {name}"
    return img


def test_real_clear_single_question_photo_measures_a_real_tilt_angle():
    # Measured at about -11.3 degrees (Issue 368). A phone photo is never perfectly level, so this
    # asserts a non-trivial angle rather than an exact value, which an opencv or numpy patch
    # release could shift slightly.
    angle = measure_skew_angle(_load("single_question_phone_photo.jpg"))
    assert angle is not None
    assert 5 < abs(angle) < 20


def test_real_handwritten_photo_measures_a_real_tilt_angle():
    angle = measure_skew_angle(_load("handwritten_error_phone_photo.jpg"))
    assert angle is not None
    assert 5 < abs(angle) < 20


def test_real_diagram_photo_measures_a_real_tilt_angle():
    angle = measure_skew_angle(_load("diagram_present_phone_photo.jpg"))
    assert angle is not None
    assert 3 < abs(angle) < 20


def test_real_genuinely_illegible_photo_returns_none_not_a_fabricated_angle():
    # The deliberately degraded fixture: no Hough lines survive the length and margin filters
    # (Issue 368). It must return None, not 0.0. A photo this blurred is not a tilt problem but a
    # different failure mode, handled by the low-confidence gate (Issue 194).
    angle = measure_skew_angle(_load("hard_degraded_phone_photo.jpg"))
    assert angle is None


def test_none_angle_is_not_treated_as_too_tilted():
    too_tilted, angle = is_photo_too_tilted(_load("hard_degraded_phone_photo.jpg"))
    assert angle is None
    assert too_tilted is False


@pytest.mark.parametrize("name", [
    "single_question_phone_photo.jpg", "handwritten_error_phone_photo.jpg", "diagram_present_phone_photo.jpg",
])
def test_real_legible_photos_are_not_flagged_as_too_tilted_at_the_default_threshold(name):
    # The basis for the default threshold (Issue 368): every test photo that MinerU2.5 and the VLM
    # read successfully must stay under it, or the default would reject legible photos.
    too_tilted, angle = is_photo_too_tilted(_load(name))
    assert too_tilted is False, f"{name} measured {angle} degrees and was wrongly flagged at the default threshold"


def test_a_low_threshold_does_flag_the_same_real_photos():
    # Shows the check responds to the threshold rather than always returning False: a strict
    # threshold, below every measured angle above, must flag the photo as too tilted.
    too_tilted, angle = is_photo_too_tilted(_load("single_question_phone_photo.jpg"), threshold_degrees=2.0)
    assert too_tilted is True


def test_default_threshold_constant_is_the_real_evidence_based_value():
    assert DEFAULT_SKEW_THRESHOLD_DEGREES == 20.0


def test_a_synthetic_blank_white_image_returns_none():
    import numpy as np
    blank = np.full((800, 600, 3), 255, dtype="uint8")
    assert measure_skew_angle(blank) is None
