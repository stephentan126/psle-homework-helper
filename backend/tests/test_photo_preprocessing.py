"""
Fast, deterministic tests for the photo preprocessing functions (Issue 21).

No VLM or GPU is used. The before/after Qwen3-VL-8B faithfulness check is in
`evaluation/model_selection/photo_transcription/validate_preprocessing.py`, because it is slow
and needs the GPU-resident model.

Inputs are synthetic with known properties (a white rectangle on black at a known angle, a known
lighting gradient), so the expected output can be checked numerically.
"""
from __future__ import annotations

import cv2
import numpy as np

from app.pipeline.photo_preprocessing import (
    detect_and_dewarp_page,
    normalize_lighting,
    preprocess_live_photo,
    resize_for_inference,
)


def _make_skewed_page(angle_deg: float = 12.0, page_size=(400, 300)) -> np.ndarray:
    """Return a white 'page' rectangle on a black 'table', rotated by a known angle.

    This lets the test check that dewarping recovers an axis-aligned, correctly sized rectangle.
    """
    canvas = np.zeros((700, 700, 3), dtype=np.uint8)
    page_w, page_h = page_size
    page = np.full((page_h, page_w, 3), 255, dtype=np.uint8)
    # A few black text-like marks so the page is not a featureless white blob, which could make
    # the input degenerate.
    cv2.rectangle(page, (20, 20), (page_w - 20, 40), (0, 0, 0), -1)
    cv2.rectangle(page, (20, 60), (page_w - 60, 80), (0, 0, 0), -1)

    center = (350, 350)
    rot_matrix = cv2.getRotationMatrix2D((page_w / 2, page_h / 2), angle_deg, 1.0)
    rotated_page = cv2.warpAffine(
        page, rot_matrix, (page_w + 100, page_h + 100), borderValue=(0, 0, 0),
    )
    rh, rw = rotated_page.shape[:2]
    y_off, x_off = center[1] - rh // 2, center[0] - rw // 2
    canvas[y_off:y_off + rh, x_off:x_off + rw] = np.where(
        rotated_page > 0, rotated_page, canvas[y_off:y_off + rh, x_off:x_off + rw],
    )
    return canvas


def test_detect_and_dewarp_page_finds_a_real_page_boundary():
    skewed = _make_skewed_page()
    result, found = detect_and_dewarp_page(skewed)
    assert found is True
    # The dewarped result should be mostly white, since the page fills the frame after cropping.
    # Mostly black would mean the crop failed to isolate the page.
    mean_brightness = result.mean()
    assert mean_brightness > 150, (
        f"Dewarped result's mean brightness ({mean_brightness:.1f}) is too dark -- the crop "
        f"likely didn't correctly isolate the real white page from the black background."
    )


def test_detect_and_dewarp_page_falls_back_honestly_on_no_boundary():
    """An image with no clean quadrilateral (pure noise) is returned unchanged with found=False.

    A guessed crop would be wrong, so the original must be returned.
    """
    noise = np.random.default_rng(0).integers(0, 255, (200, 200, 3), dtype=np.uint8)
    result, found = detect_and_dewarp_page(noise)
    assert found is False
    assert np.array_equal(result, noise)


def test_normalize_lighting_flattens_a_real_known_gradient():
    """Normalisation at least halves a known left-to-right brightness gradient on a grey image."""
    h, w = 200, 400
    base = np.full((h, w), 180, dtype=np.uint8)
    gradient = np.linspace(0.5, 1.0, w)[None, :]
    gradient_img = np.clip(base.astype(np.float32) * gradient, 0, 255).astype(np.uint8)
    gradient_bgr = cv2.cvtColor(gradient_img, cv2.COLOR_GRAY2BGR)

    normalized = normalize_lighting(gradient_bgr)
    normalized_gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)

    original_left_right_diff = abs(
        int(gradient_img[:, :20].mean()) - int(gradient_img[:, -20:].mean())
    )
    corrected_left_right_diff = abs(
        int(normalized_gray[:, :20].mean()) - int(normalized_gray[:, -20:].mean())
    )
    assert corrected_left_right_diff < original_left_right_diff / 2, (
        f"Real lighting gradient not meaningfully flattened: before={original_left_right_diff}, "
        f"after={corrected_left_right_diff} (expected after < before/2)."
    )


def test_preprocess_live_photo_runs_both_real_steps_in_order():
    skewed = _make_skewed_page()
    result, diagnostics = preprocess_live_photo(skewed)
    assert diagnostics["page_boundary_found"] is True
    assert result.shape[0] > 0 and result.shape[1] > 0


def test_resize_for_inference_caps_a_real_oversized_image():
    """An oversized image is resized down; a small image is left alone.

    Upscaling would add no detail and waste compute.
    """
    huge = np.zeros((3000, 4000, 3), dtype=np.uint8)
    resized = resize_for_inference(huge, max_dimension=2000)
    assert max(resized.shape[:2]) == 2000
    # The aspect ratio is preserved.
    assert abs(resized.shape[0] / resized.shape[1] - 3000 / 4000) < 0.01

    small = np.zeros((400, 300, 3), dtype=np.uint8)
    unchanged = resize_for_inference(small, max_dimension=2000)
    assert unchanged.shape == small.shape
