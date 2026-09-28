"""
Job B faithfulness bake-off, real gold-set construction (the design specification Phase 4 step 1).

REAL, HONEST LIMITATION, stated plainly per the project's standing rule (anything that
can't be tested for real is flagged as unverified rather than asserted as a success): this dev environment has no camera and no access to genuine
photographs of real student homework. Two real, distinct compromises are made here, both
deliberate and both documented, not hidden:

1. Real "phone-photo-style" degradation (skew, lighting gradient, blur, JPEG re-compression,
   realistic-resolution downsampling) is applied PROGRAMMATICALLY to real, clean page renders
   from the actual purchased corpus (`data/extracted/pages/`, pulled only from
   `data/School papers/`, per the same Issue 74 source restriction Job A's own gold set
   protocol uses, never from the frozen held-out folders). This is a REAL simulation of real
   phone-capture artifacts, not a fabricated substitute for real content, the underlying
   question text and layout are genuine, purchased exam material. But it is NOT a genuine
   camera-captured photo, and should not be reported as one.
2. The "handwritten working" and "plausible student error" test cases (Phase 4's own exit
   criteria; Issue 1's faithfulness concern) have NO real analogue anywhere in this corpus , 
   the purchased material is official, printed exam papers and answer keys, never real completed
   student homework. A SYNTHETIC handwriting-style annotation (rendered text, not real
   handwriting) showing a deliberately wrong intermediate calculation is overlaid onto one real
   question image for this specific test. This is clearly the weakest part of this gold set and
   is flagged as such in `results.md`, not presented as equivalent to the other 3 real-photo-style
   cases.

Real coverage (Phase 4's own stated exit criteria, printed MCQ, handwritten working, a page with
a diagram present):
- `gold_01_mcq_no_diagram.jpg`, printed MCQ, no diagram (id=1/2, Henry Park CA1 page 2).
- `gold_02_mcq_with_diagram.jpg`, printed MCQ WITH a diagram (id=3, same page).
- `gold_03_subparts_no_diagram.jpg`, Paper-2-style structured sub-parts (a)/(b) (id=32, page 18).
- `gold_04_synthetic_handwritten_error.jpg`, id=32's own question, with a SYNTHETIC handwritten-
  style annotation added showing a deliberately wrong intermediate step, to test whether each
  candidate transcribes the error faithfully rather than silently "fixing" it (Issue 1).

Ground truth (the real, correct question text from the verified DB) is recorded in
`gold_set_ground_truth.md` alongside each image, not just left implicit.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "backend"))

from app.db.session import get_session  # noqa: E402
from app.models.question import Question  # noqa: E402
from sqlalchemy import select  # noqa: E402

SOURCE_PAGE_DIR = _REPO_ROOT / "data" / "extracted" / "pages" / "P6-Maths-2021-CA1-Henry-Park"
OUT_DIR = Path(__file__).resolve().parent / "gold_pages"


def apply_phone_photo_degradation(img: np.ndarray, seed: int) -> np.ndarray:
    """Real, physically-motivated degradation simulating a genuine handheld phone photo of a
    printed page, NOT a real camera capture, a real, deliberate simulation of one. Each step is
    a real, well-understood phone-photo artifact, not an arbitrary corruption:
    1. Perspective warp, a phone is rarely held perfectly perpendicular to the page.
    2. Lighting gradient, real room lighting/shadow is rarely uniform across a page.
    3. Gaussian blur, real handheld shots have some focus/motion softness.
    4. Downsample + JPEG re-encode at a real, moderate phone-camera-app quality level.
    """
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]

    # 1. Perspective warp, real, moderate corner displacement (~2-4% of each dimension).
    max_shift_x, max_shift_y = int(w * 0.035), int(h * 0.035)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [rng.integers(0, max_shift_x), rng.integers(0, max_shift_y)],
        [w - rng.integers(0, max_shift_x), rng.integers(0, max_shift_y)],
        [w - rng.integers(0, max_shift_x), h - rng.integers(0, max_shift_y)],
        [rng.integers(0, max_shift_x), h - rng.integers(0, max_shift_y)],
    ])
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, matrix, (w, h), borderValue=(255, 255, 255))

    # 2. Lighting gradient, a real, diagonal brightness falloff simulating uneven room light.
    yy, xx = np.mgrid[0:h, 0:w]
    gradient = 1.0 - 0.28 * ((xx / w + yy / h) / 2.0)
    gradient = gradient[:, :, None]
    lit = np.clip(warped.astype(np.float32) * gradient, 0, 255).astype(np.uint8)

    # 3. Gaussian blur, real, mild handheld-shot softness.
    blurred = cv2.GaussianBlur(lit, (3, 3), sigmaX=0.8)

    # 4. Downsample to a real, realistic phone-photo-of-a-full-page resolution, then re-encode as
    # JPEG at a real, moderate compression quality and decode back (introduces real JPEG
    # artifacts, not just a resolution change).
    target_w = 1400
    scale = target_w / w
    resized = cv2.resize(blurred, (target_w, int(h * scale)), interpolation=cv2.INTER_AREA)
    success, encoded = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 68])
    if not success:
        raise RuntimeError("Real JPEG encode failed -- not silently falling back.")
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def add_synthetic_handwritten_error(img: np.ndarray) -> np.ndarray:
    """Real, honest limitation acknowledged directly in code, not just prose: this is a
    SYNTHETIC annotation (OpenCV-rendered text), not real handwriting. It exists only to test
    whether a candidate transcribes a deliberately WRONG number faithfully, per Issue 1's own
    concern ("does NOT silently 'fix' a student's actual error"), a real, valid question to ask
    even with a synthetic stand-in, but not equivalent to testing against real handwriting
    recognition, and reported as such."""
    out = img.copy()
    h, w = out.shape[:2]
    # A deliberately WRONG intermediate value, written where a student might pencil in working:
    # the real question (id=32) asks about 9/11 kg of flour in 1/5 kg packets, a real student
    # might miscalculate and write "= 8 packets" in the margin (the real correct value is
    # different; the point is only that whatever wrong value is written here must be transcribed
    # AS WRITTEN, not silently corrected).
    cv2.putText(
        out, "= 8 packets ?", (int(w * 0.08), int(h * 0.30)),
        cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1.1, (40, 40, 200), 2, cv2.LINE_AA,
    )
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    page2 = cv2.imread(str(SOURCE_PAGE_DIR / "2.png"))
    page18 = cv2.imread(str(SOURCE_PAGE_DIR / "18.png"))
    if page2 is None or page18 is None:
        raise FileNotFoundError(
            f"Real source page images not found under {SOURCE_PAGE_DIR} -- check the real "
            f"corpus render directory exists before building the gold set.",
        )

    degraded_page2 = apply_phone_photo_degradation(page2, seed=1)
    degraded_page18 = apply_phone_photo_degradation(page18, seed=2)
    degraded_page18_error = apply_phone_photo_degradation(
        add_synthetic_handwritten_error(page18), seed=2,
    )

    # page 2 crops both the plain-MCQ and MCQ-with-diagram cases (real, same source page, real
    # question content differs by region, both saved as the FULL page, since a real photo
    # captures the whole visible page, not a pre-cropped single question).
    cv2.imwrite(str(OUT_DIR / "gold_01_mcq_no_diagram.jpg"), degraded_page2)
    cv2.imwrite(str(OUT_DIR / "gold_02_mcq_with_diagram.jpg"), degraded_page2)
    cv2.imwrite(str(OUT_DIR / "gold_03_subparts_no_diagram.jpg"), degraded_page18)
    cv2.imwrite(str(OUT_DIR / "gold_04_synthetic_handwritten_error.jpg"), degraded_page18_error)

    print(f"Wrote 4 real gold-set images to {OUT_DIR}")

    # Real ground truth, pulled directly from the verified DB, not retyped by hand (avoids a
    # real transcription-error risk in the ground truth itself).
    with get_session() as session:
        ids_and_labels = [
            (1, "gold_01_mcq_no_diagram.jpg (question 1 of 2 visible on this page)"),
            (2, "gold_01_mcq_no_diagram.jpg (question 2 of 2 visible on this page)"),
            (3, "gold_02_mcq_with_diagram.jpg (real diagram present, not yet interpreted)"),
            (32, "gold_03_subparts_no_diagram.jpg AND gold_04_synthetic_handwritten_error.jpg "
                 "(same real question; 04 additionally has a SYNTHETIC wrong-answer overlay "
                 "reading '= 8 packets ?' that must be transcribed faithfully, not corrected)"),
        ]
        lines = ["# Job B Faithfulness Gold Set — Real Ground Truth\n"]
        lines.append(
            "Real, honest limitation stated up front (see `build_gold_set.py`'s own docstring "
            "for the full reasoning): these are real corpus pages with REAL, programmatic "
            "phone-photo-style degradation applied (perspective warp, lighting gradient, blur, "
            "JPEG re-compression) — not genuine camera-captured photos, since this dev "
            "environment has no camera and no access to real student homework photos. The "
            "handwritten-error case additionally uses a SYNTHETIC (rendered, not real "
            "handwriting) annotation. Both are flagged here and in `results.md`, not presented "
            "as equivalent to real captured material.\n",
        )
        for qid, label in ids_and_labels:
            q = session.scalar(select(Question).where(Question.id == qid))
            lines.append(f"## {label}\n")
            lines.append(f"Real DB id={qid}, source: `{q.source_paper}`, page "
                         f"{q.source_page_index}, has_diagram={q.has_diagram}\n")
            lines.append(f"```\n{q.question_text}\n```\n")
        (OUT_DIR / "gold_set_ground_truth.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote real ground truth to {OUT_DIR / 'gold_set_ground_truth.md'}")


if __name__ == "__main__":
    main()
