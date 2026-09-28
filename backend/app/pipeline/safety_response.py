"""
Category-appropriate safety responses and audit logging (Issues 203, 206 and 207).

Used with `self_harm_lexicon.py` and `self_harm_detection.py`, so that a safety flag from either
signal leads to a suitable response for the student and an audit record in the database. The
check is called from `request_hint()` and `confirm_photo_submission()` (Issue 204).

End-to-end tests showed that every ShieldGemma flag, whatever its category, was routed to the
self-harm crisis response, so a student who submitted sexual content, or received a demeaning
generated hint, saw a message about self-harm and a suicide hotline (Issue 206). Every category
was still blocked, but the message was wrong for the case. `build_safety_response(category)` is
now the dispatcher all call sites use (Issue 207).

Crisis resources (Issue 203), taken from the Samaritans of Singapore official site
(`https://www.sos.org.sg/contact-us/`):
  - 24-hour Hotline: 1767
  - 24-hour CareText (WhatsApp): 9151 1767
Re-check these against the same source before release. Crisis service contact details can
change, and out-of-date information here would cause serious harm.

Automatic parent or guardian notification is not implemented. Section 7 of the design
specification calls for an alert to the linked parent account on a self-harm flag, but this raises
consent, disclosure and PDPA (Singapore's Personal Data Protection Act) questions that are still
open. `SafetyBlockedResponse.parent_notified` (schemas.py) is therefore always False here.
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.api.schemas import SafetyBlockedResponse
from app.models.safety_flag import SafetyFlag

# Taken from https://www.sos.org.sg/contact-us/ (Issue 203). Re-check against the same source
# before relying on these in production.
SAMARITANS_OF_SINGAPORE_RESOURCES: list[str] = [
    "Samaritans of Singapore (SOS) 24-hour Hotline: 1767",
    "SOS 24-hour CareText (WhatsApp): 9151 1767",
]

_CRISIS_CARE_MESSAGE = (
    "It sounds like things feel really hard right now, and I want you to know you don't have to "
    "go through this alone. I'm not able to help with that here, but there are people who can — "
    "please reach out to Samaritans of Singapore, any time, day or night. You matter, and things "
    "can get better."
)

_SEXUALLY_EXPLICIT_MESSAGE = (
    "That's not something I can help you with here. Let's get back to your homework — what "
    "question were you working on?"
)

_HARASSMENT_MESSAGE = (
    "Sorry about that — that response wasn't right, and it's not how this should work. Let's "
    "try that question again."
)


def build_self_harm_crisis_response() -> SafetyBlockedResponse:
    """Builds the response for self-harm flagged input or output.

    Section 7 requires a caring, non-judgemental message with current crisis resources, never a
    generic block message or a capped hint. It uses the existing `SafetyBlockedResponse` schema
    (`app/api/schemas.py`) with `is_self_harm_path=True`; that schema's `is_self_harm_path`,
    `resources` and `parent_notified` fields were designed for this path, so the six-state
    response contract in Section 10.3 needs no new state.

    Also used for `category="dangerous_content"` (Issue 207). ShieldGemma's policy for that
    category covers both self-harm and violence to others, and no available signal separates
    them: corpus case `IN-U4`, a self-harm method question, fires only on this policy. See
    `self_harm_detection.py` for the full reasoning.

    `parent_notified` is always False; see the module docstring.
    """
    return SafetyBlockedResponse(
        message=_CRISIS_CARE_MESSAGE,
        is_self_harm_path=True,
        resources=list(SAMARITANS_OF_SINGAPORE_RESOURCES),
        parent_notified=False,
    )


def build_sexually_explicit_response() -> SafetyBlockedResponse:
    """Builds the response for `category="sexually_explicit"` (Issue 207).

    A calm, neutral boundary and redirect, not the crisis message. `is_self_harm_path` is False
    and `resources` is None: only a self-harm signal carries crisis resources.
    """
    return SafetyBlockedResponse(
        message=_SEXUALLY_EXPLICIT_MESSAGE, is_self_harm_path=False, resources=None,
        parent_notified=False,
    )


def build_harassment_response() -> SafetyBlockedResponse:
    """Builds the response for `category="harassment"` (Issue 207).

    In the gold set every harassment case is on the output side (a generated hint that came out
    demeaning), never the input side. This is a problem with the model's output, not something the
    student did, so the message is an apology and retry rather than a refusal, which would imply
    the student's question was at fault.

    On the output path, `routes.py` first regenerates a harassment-flagged hint up to
    `_MAX_HARASSMENT_RETRY_ATTEMPTS` times, so the student only sees this message if the hint is
    still flagged after the retries. It remains a `SafetyBlockedResponse` rather than a new state,
    because Section 10.3 defines a closed set of six response states.
    """
    return SafetyBlockedResponse(
        message=_HARASSMENT_MESSAGE, is_self_harm_path=False, resources=None,
        parent_notified=False,
    )


def build_classifier_error_response() -> SafetyBlockedResponse:
    """Builds the fail-closed response for when the safety check itself fails (Issue 210).

    This covers a crash in ShieldGemma-2B or the lexicon layer, not a detected flag. The safety
    classifier must fail closed (Issue 58): if it cannot run, the response is blocked and content
    is never let through unchecked.

    It reuses the crisis response, the most protective response available. A crash gives no
    information about the content, so there is no basis for choosing one of the four categories,
    and the most cautious option is the safe one. This follows the same safety-biased grouping
    used for `dangerous_content` (Issue 207).
    """
    return build_self_harm_crisis_response()


def build_safety_response(category: str) -> SafetyBlockedResponse:
    """Returns the response for a safety category (Issue 207).

    All call sites should use this rather than calling the individual builders, so a new category
    cannot be handled in one place and missed in another. Categories, as confirmed against
    `gold_set.json`:
      - "self_harm" and "dangerous_content": the crisis response (see
        `build_self_harm_crisis_response()` for why they share it).
      - "sexually_explicit": `build_sexually_explicit_response()`.
      - "harassment": `build_harassment_response()`.

    Raises:
        ValueError: If `category` is not one of the above.
    """
    if category in ("self_harm", "dangerous_content"):
        return build_self_harm_crisis_response()
    if category == "sexually_explicit":
        return build_sexually_explicit_response()
    if category == "harassment":
        return build_harassment_response()
    raise ValueError(f"Unknown safety category: {category!r}")


def record_safety_flag(
    session: Session, *, student_id: int, attempt_id: int | None,
    flagged_input_or_output: str, category: str,
) -> SafetyFlag:
    """Writes an audit row to the `safety_flags` table (`app/models/safety_flag.py`, Section 10.1).

    The existing model needed no schema change. `attempt_id` is nullable because a flagged input
    can arrive before an `Attempt` row exists. `flagged_input_or_output` is "input" or "output",
    matching the two screening points in Section 7. `parent_notified_at` is left as None, since no
    automatic parent notification is implemented (see the module docstring).

    `category` is supplied by the caller (for example a category from `SafetyCheckResult` in
    `self_harm_detection.py`). Previously every flag was logged as "self_harm", so the audit trail
    was mislabelled in the same way as the response (Issue 206); both were fixed together.

    Raises:
        ValueError: If `flagged_input_or_output` is not "input" or "output". The column is a plain
            string rather than an enum, for the same reason as
            `OCRExtractionRecord.agreement_status`.
    """
    if flagged_input_or_output not in ("input", "output"):
        raise ValueError(
            f"flagged_input_or_output must be 'input' or 'output', got {flagged_input_or_output!r}",
        )
    flag = SafetyFlag(
        student_id=student_id, attempt_id=attempt_id,
        flagged_input_or_output=flagged_input_or_output, category=category,
        parent_notified_at=None,
    )
    session.add(flag)
    session.flush()
    return flag
