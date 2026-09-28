"""
Live tests for the photo entry point (`POST /api/submissions/photo`) and the model swap between
`llm_service.py` and `vlm_service.py` (Phase 4, Issues 191 and 193).

Nothing is mocked and the tests are slow: a VLM load and a generation call of several hundred
seconds, then a swap to the LLM for the gate run. This is the first test with two large,
independently loaded models in one process, so it exercises the swap-eviction logic end to end
rather than in isolation.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# The transcriptions printed here contain non-cp1252 characters (for example degree and angle
# symbols), which crash the default Windows console encoding after the expensive work is done.
# The same fix is used in the Issue 184 bake-off scripts.
#
# stdout is reconfigured in place rather than replaced with a new io.TextIOWrapper (Issue 195).
# Replacing it breaks pytest's capture manager, which keeps a reference to the original stdout:
# at teardown it raises `ValueError: I/O operation on closed file`. Without `-s` that crashes
# collection; with `-s` it silently swallows the final "X passed" summary line.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fastapi.testclient import TestClient

from app.db.session import get_session
from app.main import app
from app.models.question import Question
from app.models.question_embedding import QuestionEmbedding
from app.models.student_submission import StudentSubmission
from app.pipeline.question_matching import EMBEDDING_MODEL_ID, _get_model
from app.services import llm_service, vlm_service

_REAL_TEST_PHOTO = Path(__file__).resolve().parent / "_fixtures" / "single_question_phone_photo.jpg"

# The corpus text of Q1 in the test photo. The Issue 185 noisy-query check validated this
# photo and text pairing (development log).
_REAL_MATCHING_QUESTION_TEXT = (
    "Which digit in 15.89 is in the tenths place?\n(1) 1\n(2) 5\n(3) 8\n(4) 9"
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _insert_real_matching_question() -> int:
    with get_session() as session:
        question = Question(
            source_paper="data/School papers/2021/P6-Maths-2021-CA1-Henry-Park.pdf",
            source_page_index=2, source_page_label=None,
            answer_source_file="data/School papers/2021/P6-Maths-2021-CA1-Henry-Park.pdf",
            answer_source_page_index=2, question_number="1",
            question_text=_REAL_MATCHING_QUESTION_TEXT, answer_value="4",
            worked_solution_text=None, has_diagram=False, ocr_confidence="high",
            school_name="Henry Park Primary School",
        )
        session.add(question)
        session.flush()
        question_id = question.id

        model = _get_model()
        embedding = model.encode([_REAL_MATCHING_QUESTION_TEXT], convert_to_numpy=True)[0]
        session.add(QuestionEmbedding(
            question_id=question_id, embedding_model=EMBEDDING_MODEL_ID,
            embedding_json=json.dumps(embedding.tolist()), computed_at=_now_iso(),
        ))
        return question_id


def test_real_photo_upload_end_to_end_through_confirm_with_a_real_model_swap():
    """A degraded phone photo goes through the full pipeline to a hint, with two model swaps.

    The path is: upload, preprocessing, Qwen3-VL-8B-Instruct transcription, match-first finds the
    corpus question, confirm, then the gate runs against the matched question's answer
    (Issue 189) and returns a hint. The LLM is loaded first with a throwaway generation, so
    loading the VLM must evict it; the gate run at confirm then evicts the VLM again (Issue 191).
    Swap timings are measured and printed rather than asserted, because there is no earlier
    measurement to assert against.
    """
    question_id = _insert_real_matching_question()
    assert _REAL_TEST_PHOTO.exists(), f"Real test photo missing: {_REAL_TEST_PHOTO}"

    # Load the LLM first, so the VLM load has something to evict.
    t0 = time.time()
    llm_service.get_llm()
    print(f"\nReal LLM load time: {time.time() - t0:.1f}s")
    assert llm_service.is_llm_loaded() is True
    assert vlm_service.is_vlm_loaded() is False

    with TestClient(app) as client:
        with open(_REAL_TEST_PHOTO, "rb") as f:
            t0 = time.time()
            response = client.post(
                "/api/submissions/photo",
                files={"file": ("phone_photo.jpg", f, "image/jpeg")},
            )
            print(f"Real /submissions/photo call (decode+preprocess+VLM-load+transcribe): "
                  f"{time.time() - t0:.1f}s")

        assert response.status_code == 200
        body = response.json()
        print(f"Real response: {json.dumps(body, indent=2, ensure_ascii=False)[:2000]}")

        # Swap 1: the VLM is now resident and the LLM was evicted.
        assert vlm_service.is_vlm_loaded() is True
        assert llm_service.is_llm_loaded() is False

        assert body["state"] == "needs_confirmation"
        # Match-first found the corpus question from the noisy transcription of the degraded
        # photo, which is the main point of this test.
        assert body["question_id"] == question_id
        assert body["match_confidence"] is not None and body["match_confidence"] >= 0.90
        assert body["matched_question_text"] == _REAL_MATCHING_QUESTION_TEXT
        # The raw VLM transcription, with its OCR artefacts, is shown as the confirmation target
        # (Issue 188), not a copy of the clean corpus text.
        assert body["extracted_text"] != body["matched_question_text"]
        assert len(body["extracted_text"]) > 0
        # The transcription cleared MIN_TRUSTED_LOG_PROB (step 4, Issue 194), so "high"
        # confidence is reported.
        assert body["confidence"] == "high"

        submission_id = body["student_submission_id"]

        t0 = time.time()
        confirm_response = client.post(
            f"/api/submissions/{submission_id}/confirm",
            params={"request_id": "live-photo-e2e-test-1"},
        )
        print(f"Real /submissions/{{id}}/confirm call (real LLM-swap-back + real gate run): "
              f"{time.time() - t0:.1f}s")

    assert confirm_response.status_code == 200
    confirm_body = confirm_response.json()
    print(f"Real confirm response: {json.dumps(confirm_body, indent=2, ensure_ascii=False)}")

    # Swap 2: the LLM is resident again and the VLM was evicted.
    assert llm_service.is_llm_loaded() is True
    assert vlm_service.is_vlm_loaded() is False

    assert confirm_body["gate_path_used"] == "verified_match"
    assert confirm_body["is_unverified"] is False
    # The answer (option 4, digit 9; see Issue 175 on MCQ index versus value) must not leak,
    # whichever tier the gate chose.
    hint_text = confirm_body.get("hint_text", "")
    assert "digit is 9" not in hint_text.lower()


_REAL_ILLEGIBLE_PHOTO = Path(__file__).resolve().parent / "_fixtures" / "hard_degraded_phone_photo.jpg"


def test_real_photo_upload_genuinely_illegible_photo_returns_extraction_failed():
    """An illegible photo is rejected as `extraction_failed` by the confidence gate (Issue 194).

    The fixture is the one used to set the threshold: mean_log_prob=-0.0657, and the model
    hallucinated '[DIAGRAM PRESENT]' over text. It must not reach the student through
    `needs_confirmation` as if the read had succeeded.
    """
    assert _REAL_ILLEGIBLE_PHOTO.exists(), f"Real test photo missing: {_REAL_ILLEGIBLE_PHOTO}"

    with TestClient(app) as client:
        with open(_REAL_ILLEGIBLE_PHOTO, "rb") as f:
            response = client.post(
                "/api/submissions/photo",
                files={"file": ("illegible.jpg", f, "image/jpeg")},
            )

    assert response.status_code == 200
    body = response.json()
    print(f"\nReal response for the genuinely illegible photo: "
          f"{json.dumps(body, indent=2, ensure_ascii=False)}")
    assert body["state"] == "extraction_failed"
    # Steps 5 and 6 (Issue 196): the Phase 8 frontend uses this reason to show "retake the
    # photo" rather than "type instead", since the cause is a low-confidence transcription, not a
    # reader-load failure.
    assert body["reason"] == "low_confidence"


_REAL_HANDWRITTEN_ERROR_PHOTO = (
    Path(__file__).resolve().parent / "_fixtures" / "handwritten_error_phone_photo.jpg"
)


def test_real_handwritten_photo_with_deliberate_error_transcribed_faithfully_not_corrected():
    """A deliberate error in handwritten working is transcribed as written, not corrected.

    This is the handwritten-working layout type from the Phase 4 exit criteria (Issue 197). The
    source is a purchased student's handwritten working with one deliberate error added (see
    `_build_handwritten_error_test_photo.py`): the working reaches 'Total coins at first =
    7 x 10 = 70', but the boxed 'Ans:' line reads '10', pasted from elsewhere on the same page.

    This tests Issue 1 (over-correction). The reader must transcribe the ink on the page even
    though it contradicts the working above. Reading 70 would be the silent answer-key corruption
    Issue 1 exists to catch, and a pass on an error-free photo (Issue 193) cannot show this.
    """
    assert _REAL_HANDWRITTEN_ERROR_PHOTO.exists(), \
        f"Real test photo missing: {_REAL_HANDWRITTEN_ERROR_PHOTO}"

    with TestClient(app) as client:
        with open(_REAL_HANDWRITTEN_ERROR_PHOTO, "rb") as f:
            response = client.post(
                "/api/submissions/photo",
                files={"file": ("handwritten_error.jpg", f, "image/jpeg")},
            )

    assert response.status_code == 200
    body = response.json()
    print(f"\nReal response for the handwritten photo with a deliberate error: "
          f"{json.dumps(body, indent=2, ensure_ascii=False)}")
    assert body["state"] == "needs_confirmation"
    extracted = body["extracted_text"]
    # The key assertion: the reader transcribed the wrong value on the page ('Ans: 10'), not the
    # correct value it could have inferred from the working ('Ans: 70').
    assert "Ans: 10" in extracted, (
        f"Expected the real deliberate error ('Ans: 10') to be transcribed faithfully — the "
        f"reader may have over-corrected it (Issue 1). Real extracted_text: {extracted!r}"
    )
    assert "Ans: 70" not in extracted, (
        f"The reader appears to have 'fixed' the real deliberate error back to the correct "
        f"answer — exactly the over-correction failure mode Issue 1 exists to catch. Real "
        f"extracted_text: {extracted!r}"
    )
    # The correct derivation above the error must also be transcribed, showing the reader can
    # read the rest of the page.
    assert "7" in extracted and "70" in extracted


_REAL_DIAGRAM_PRESENT_PHOTO = (
    Path(__file__).resolve().parent / "_fixtures" / "diagram_present_phone_photo.jpg"
)


def test_real_diagram_present_photo_degrades_honestly_no_interpretation():
    """A photo with a diagram is flagged with '[DIAGRAM PRESENT]' rather than interpreted.

    This is the diagram-present layout type from the Phase 4 exit criteria (Issue 197): a
    parallelogram and triangle figure with handwritten angle annotations. Diagram interpretation
    (Section 6) is not built yet, so the reader prompt (`vlm_service.FAITHFUL_TRANSCRIPTION_PROMPT`)
    asks it to note '[DIAGRAM PRESENT]' and not interpret the figure. The diagram must be flagged,
    neither ignored nor filled in with content the reader cannot verify.
    """
    assert _REAL_DIAGRAM_PRESENT_PHOTO.exists(), \
        f"Real test photo missing: {_REAL_DIAGRAM_PRESENT_PHOTO}"

    with TestClient(app) as client:
        with open(_REAL_DIAGRAM_PRESENT_PHOTO, "rb") as f:
            response = client.post(
                "/api/submissions/photo",
                files={"file": ("diagram_present.jpg", f, "image/jpeg")},
            )

    assert response.status_code == 200
    body = response.json()
    print(f"\nReal response for the diagram-present photo: "
          f"{json.dumps(body, indent=2, ensure_ascii=False)}")
    assert body["state"] == "needs_confirmation"
    extracted = body["extracted_text"]
    # The degrade signal: the reader flagged the diagram instead of describing or inventing its
    # content.
    assert "[DIAGRAM PRESENT]" in extracted
    # The handwritten angle annotations written as text on the page must still be transcribed.
    # This shows only the diagram was degraded, not the whole page.
    assert "83" in extracted and "57" in extracted

    with get_session() as session:
        submission = session.get(StudentSubmission, body["student_submission_id"])
        # Database check, since the field is not on the response: the '[DIAGRAM PRESENT]' tag
        # sets `has_diagram` to True, so later tiers and the UI can rely on it.
        assert submission.has_diagram is True
