"""
Validation script for the confidence signal (development log, Issue 44, run as Issue 194).

Issue 44 asked to "measure which of (a) or (b) predicts real misreads better on the gold set before
picking one." This script compares token-level mean-logprob confidence between an easy case (correct
transcription, match_confidence=0.9887 in the Issue 193 end-to-end test) and a much harder case
(heavy blur, resolution loss, steeper skew and a stronger lighting gradient). Both are crops of the
same corpus question (Q1).
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "backend"))

from app.services import vlm_service  # noqa: E402

_FIXTURES = Path(__file__).resolve().parent / "_fixtures"

CASES = [
    ("EASY (already validated, real correct transcription)", _FIXTURES / "single_question_phone_photo.jpg"),
    ("HARD (heavy blur + real resolution loss + steeper skew)", _FIXTURES / "hard_degraded_phone_photo.jpg"),
]


def main() -> None:
    for label, path in CASES:
        assert path.exists(), f"Missing real fixture: {path}"
        text, mean_log_prob = vlm_service.transcribe_photo(str(path))
        print(f"\n{'=' * 90}\n{label}")
        print(f"Real mean token log-probability: {mean_log_prob:.4f}")
        print(f"Real transcription:\n{text}")


if __name__ == "__main__":
    main()
