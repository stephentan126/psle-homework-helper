"""The six typed API response states.

Every response has a `state` field whose value names the state: `needs_confirmation`, `hint`,
`capped_tier1`, `safety_blocked`, `extraction_failed` or `unlocked_solution`. `AnyHomeworkResponse`
combines them into a discriminated union, so the frontend switches on one field and the OpenAPI
schema lists the six variants.
"""
from __future__ import annotations

from typing import Annotated, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


class NeedsConfirmationResponse(BaseModel):
    """Transcribed question text waiting for the student to confirm or edit it.

    `extracted_text` is always the text read from the student's own photo (or typed), even when a
    question-bank match was found, because the student can only check it against their own page
    (Issue 188). A confident match is shown separately in `matched_question_text` as supporting
    context and is never the text being confirmed.

    `student_submission_id` identifies the submission for the follow-up confirm call.
    `question_id` is the matched question-bank id, or `None` when no confident match was found
    (Issues 180, 186). `match_confidence` is the best similarity score found, recorded even for a
    near miss.
    """
    state: Literal["needs_confirmation"] = "needs_confirmation"
    question_id: Optional[int] = None
    student_submission_id: Optional[int] = None
    match_confidence: Optional[float] = None
    extracted_text: str
    matched_question_text: Optional[str] = None
    confidence: Optional[Literal["high", "low"]] = None


class HintResponse(BaseModel):
    """A hint at Tier 1 to 3, returned when the gate allows it.

    `gate_path_used` is `"verified_match"` for question-bank questions. Weak mode is capped at
    Tier 1 and is always returned as `CappedTier1Response`, so the validator below rejects
    `"weak_mode"` here (Issue 180). `is_unverified` mirrors `gate_path_used` so the frontend can
    style unverified help from one flag; the validator keeps the two consistent.

    `can_escalate_further` tells the frontend whether a higher tier is available, given the
    gate's ceiling for this attempt (`tier_allowed` 1, 2 or 4 maps to a highest tier of 1, 2 or
    3). Every call site sets these fields explicitly rather than relying on the defaults.
    """
    state: Literal["hint"] = "hint"
    question_id: int
    attempt_id: Optional[int] = None
    tier: int = Field(ge=1, le=3)
    hint_text: str
    gate_path_used: Literal["verified_match", "weak_mode"] = "verified_match"
    is_unverified: bool = False
    can_escalate_further: bool = False

    @model_validator(mode="after")
    def _weak_mode_never_escalates(self) -> "HintResponse":
        if self.gate_path_used == "weak_mode":
            raise ValueError(
                "A HintResponse (tier >= 2 by definition) can never be gate_path_used="
                "'weak_mode' — weak mode is hard-capped at Tier 1 (the design specification Section 5 "
                "addendum, Issue 180 addendum) and must be a CappedTier1Response instead."
            )
        if self.is_unverified != (self.gate_path_used == "weak_mode"):
            raise ValueError(
                f"is_unverified ({self.is_unverified}) must match gate_path_used=="
                f"'weak_mode' ({self.gate_path_used == 'weak_mode'}) exactly — they must never "
                f"disagree."
            )
        return self


class CappedTier1Response(BaseModel):
    """A Tier 1 hint with no escalation, used when the gate cannot verify the question.

    This covers two cases: a question-bank question the gate could not verify, and a weak-mode
    submission with no answer key (Issue 180). Exactly one of `question_id` (question bank) or
    `student_submission_id` (weak mode) is set, matching `gate_path_used`; the validator enforces
    this. `reason` records the gate outcome for logging and is not shown to the child.
    `can_escalate_further` is always `False` here.
    """
    state: Literal["capped_tier1"] = "capped_tier1"
    question_id: Optional[int] = None
    student_submission_id: Optional[int] = None
    attempt_id: Optional[int] = None
    hint_text: str
    reason: Optional[str] = None
    gate_path_used: Literal["verified_match", "weak_mode"] = "verified_match"
    is_unverified: bool = False
    # Always False here: a capped result has no higher tier to offer.
    can_escalate_further: bool = False

    @model_validator(mode="after")
    def _path_and_ids_are_consistent(self) -> "CappedTier1Response":
        if self.is_unverified != (self.gate_path_used == "weak_mode"):
            raise ValueError(
                f"is_unverified ({self.is_unverified}) must match gate_path_used=="
                f"'weak_mode' ({self.gate_path_used == 'weak_mode'}) exactly — they must never "
                f"disagree."
            )
        has_question = self.question_id is not None
        has_submission = self.student_submission_id is not None
        if has_question == has_submission:  # Both set, or neither set.
            raise ValueError(
                f"CappedTier1Response must have exactly one of question_id/"
                f"student_submission_id set, not {'both' if has_question else 'neither'} "
                f"(question_id={self.question_id!r}, "
                f"student_submission_id={self.student_submission_id!r})."
            )
        if self.gate_path_used == "weak_mode" and not has_submission:
            raise ValueError(
                "gate_path_used='weak_mode' requires student_submission_id to be set "
                "(a weak-mode result has no real questions.id)."
            )
        if self.gate_path_used == "verified_match" and not has_question:
            raise ValueError(
                "gate_path_used='verified_match' requires question_id to be set."
            )
        return self


class SafetyBlockedResponse(BaseModel):
    """The safety layer flagged the input or a generated hint.

    On the self-harm path (`is_self_harm_path=True`) the response carries a caring message,
    support `resources` for Singapore and whether a parent was notified. For other categories it
    carries only the message.
    """
    state: Literal["safety_blocked"] = "safety_blocked"
    message: str
    is_self_harm_path: bool = False
    resources: Optional[List[str]] = None
    parent_notified: bool = False


class ExtractionFailedResponse(BaseModel):
    """The photo could not be used, with a machine-readable `reason` (Issue 196).

    - `unreadable_photo`: the image could not be decoded.
    - `low_confidence`: the transcription's confidence score was too low (Issue 194).
    - `reader_unavailable`: the vision model failed to load or run (Issue 56).
    - `photo_not_straight`: the question is tilted too far to read reliably (Issue 368).

    The first two ask the student to take another photo; the third suggests typing the question
    instead. The frontend branches on `reason` rather than on the message text.
    """
    state: Literal["extraction_failed"] = "extraction_failed"
    message: str
    reason: Literal["unreadable_photo", "low_confidence", "reader_unavailable", "photo_not_straight"]


class UnlockedSolutionResponse(BaseModel):
    """The full answer and worked solution, returned after a correct parent PIN.

    This is the only state allowed to show a final answer. Both fields are optional because not
    every question has a short answer value or a written worked solution (Issue 102).
    """
    state: Literal["unlocked_solution"] = "unlocked_solution"
    question_id: int
    answer_value: Optional[str] = None
    worked_solution_text: Optional[str] = None


# A discriminated union: Pydantic uses `state` to choose the model to validate against, and
# FastAPI documents the six variants in the OpenAPI schema.
AnyHomeworkResponse = Annotated[
    Union[
        NeedsConfirmationResponse,
        HintResponse,
        CappedTier1Response,
        SafetyBlockedResponse,
        ExtractionFailedResponse,
        UnlockedSolutionResponse,
    ],
    Field(discriminator="state"),
]
