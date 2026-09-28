"""
Fast tests for the `ExtractionFailedResponse` paths not covered by `test_live_photo_pipeline.py`.

Phase 4 steps 5 and 6 (design specification Section 15; Issue 196). The slow GPU test in
`test_live_photo_pipeline.py` covers `reason="low_confidence"` (Issue 194). A bad decode and a
reader-load failure both happen before the VLM is used, so these tests do not need the ~10GB model
loaded and are kept separate to stay fast.

The decode-failure case feeds invalid bytes through the same `cv2.imdecode()` call the endpoint
uses. For the reader-failure case, `vlm_service.transcribe_photo` is monkeypatched to raise. This
is the same "simulate a load failure" approach the design specification (Phase 5 step 3) asks for
with the safety classifier, because an out-of-memory or corrupted-download failure cannot be
reproduced deterministically any other way.
"""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.api import submission_routes
from app.main import app
from app.services import vlm_service

_REAL_VALID_PHOTO = Path(__file__).resolve().parent / "_fixtures" / "single_question_phone_photo.jpg"


def test_submit_photo_genuinely_corrupt_upload_returns_unreadable_photo_reason():
    """A corrupt or non-image upload returns `reason="unreadable_photo"`.

    `cv2.imdecode()` returns `None` for bytes that are not a decodable image, which is the
    condition this branch guards (development log, Issue 196).
    """
    with TestClient(app) as client:
        response = client.post(
            "/api/submissions/photo",
            files={"file": ("not_a_photo.jpg", b"this is not real jpeg data", "image/jpeg")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "extraction_failed"
    assert body["reason"] == "unreadable_photo"


def test_submit_photo_reader_load_failure_returns_reader_unavailable_reason(monkeypatch):
    """A reader failure on a valid photo returns `reason="reader_unavailable"` (Issue 56).

    The photo decodes and preprocesses normally, but the reader raises (simulated, as above). The
    response carries the "type your question instead" message. The reason differs from a bad
    photo because the Phase 8 frontend needs to tell the two cases apart.
    """
    assert _REAL_VALID_PHOTO.exists(), f"Real test photo missing: {_REAL_VALID_PHOTO}"

    def _simulated_reader_failure(*args, **kwargs):
        raise RuntimeError("Simulated reader load/inference failure (Issue 56).")

    monkeypatch.setattr(vlm_service, "transcribe_photo", _simulated_reader_failure)

    with TestClient(app) as client:
        with open(_REAL_VALID_PHOTO, "rb") as f:
            response = client.post(
                "/api/submissions/photo",
                files={"file": ("photo.jpg", f, "image/jpeg")},
            )

    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "extraction_failed"
    assert body["reason"] == "reader_unavailable"
    assert "type your question" in body["message"].lower()


def test_submit_photo_real_legible_tilted_photo_is_not_blocked_as_too_tilted(monkeypatch):
    """A legible photo with moderate tilt is not blocked by the skew detector (Issue 368).

    The test photo, which the VLM reads correctly, measures about -11.3 degrees of tilt, well
    under the 20-degree default. This guards against the false-positive risk that ruled out a
    stricter threshold (see the skew_detection.py module docstring). The skew check runs unmocked;
    only `transcribe_photo` is stubbed so the test stays fast and CPU-only. The VLM path is covered
    in test_live_photo_pipeline.py.
    """
    assert _REAL_VALID_PHOTO.exists(), f"Real test photo missing: {_REAL_VALID_PHOTO}"
    monkeypatch.setattr(vlm_service, "transcribe_photo", lambda *a, **k: ("stubbed transcription text", 0.0))

    with TestClient(app) as client:
        with open(_REAL_VALID_PHOTO, "rb") as f:
            response = client.post(
                "/api/submissions/photo",
                files={"file": ("photo.jpg", f, "image/jpeg")},
            )

    assert response.status_code == 200
    body = response.json()
    assert not (body["state"] == "extraction_failed" and body.get("reason") == "photo_not_straight")


def test_submit_photo_too_tilted_returns_photo_not_straight_reason(monkeypatch):
    """A photo judged too tilted returns `reason="photo_not_straight"` (Issue 368).

    This checks the endpoint wiring. The tilt is simulated, as in the reader-failure test above,
    because every available photo measures under the 20-degree default (see the
    skew_detection.py module docstring).
    """
    assert _REAL_VALID_PHOTO.exists(), f"Real test photo missing: {_REAL_VALID_PHOTO}"
    monkeypatch.setattr(submission_routes, "is_photo_too_tilted", lambda image: (True, 35.0))

    with TestClient(app) as client:
        with open(_REAL_VALID_PHOTO, "rb") as f:
            response = client.post(
                "/api/submissions/photo",
                files={"file": ("photo.jpg", f, "image/jpeg")},
            )

    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "extraction_failed"
    assert body["reason"] == "photo_not_straight"
    assert "tilted" in body["message"].lower() or "level" in body["message"].lower()
