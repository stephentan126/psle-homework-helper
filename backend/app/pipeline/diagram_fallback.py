"""
Fallback path for diagram interpretation (Issues 14 and 56; design specification, Section 6).

The fallback was built before any diagram interpreter, so the failure path exists first.

`resolve_diagram_context()` never raises and never guesses. If the interpreter fails to load,
times out or returns a low-confidence result, the hint falls back to the text-only path in
Section 6: refer the student to the figure and work only from the numbers in the question text.
A question flagged as containing a diagram must never be treated as if the diagram were
decorative (Issue 14).

For `has_diagram=False` the function returns at once without calling `diagram_service`, so
text-only questions are unaffected by construction. `test_diagram_degrade_path.py` checks this
with a call count on `diagram_service.interpret_diagram`, not only by comparing outputs.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from app.services.diagram_service import (
    MIN_TRUSTED_DIAGRAM_CONFIDENCE,
    DiagramInterpretationResult,
    DiagramInterpretationTimeoutError,
    DiagramInterpreterUnavailableError,
    interpret_diagram,
)

logger = logging.getLogger("pipeline.diagram_fallback")

# Text-only fallback from Section 6 (Issue 14), appended to the hint prompt when a question has a
# diagram but no trusted diagram data was obtained. It never claims the diagram was read and never
# invents its content.
DIAGRAM_UNAVAILABLE_ACKNOWLEDGMENT = (
    "Note: this question includes a diagram or figure that could not be read. Do not describe or "
    "assume any content of the figure. Base your hint only on the numbers and words in the "
    "question text below, and explicitly tell the student to refer to the actual figure on their "
    "own page to check any values you cannot see."
)


@dataclass(frozen=True)
class DiagramContext:
    """Result of attempting diagram interpretation for a question with a diagram.

    `available=False` covers every failure reason (Issue 56): no interpreter, a load failure, a
    timeout or a low-confidence result. Callers only need to know whether trusted data exists.
    """

    available: bool
    structured_data: Optional[dict] = None
    acknowledgment: Optional[str] = None


_NOT_A_DIAGRAM_QUESTION = DiagramContext(available=False)


def resolve_diagram_context(
    has_diagram: bool, diagram_image: Optional[bytes] = None,
) -> DiagramContext:
    """Returns diagram data for the hint, or the text-only fallback (Issue 56).

    For `has_diagram=False` this makes no call into `diagram_service`. For `has_diagram=True`, no
    interpreter failure propagates as an exception and the diagram is never silently ignored
    (Issue 14). Every failure (unavailable, timeout, low confidence or an unexpected exception)
    returns the same explicit text-only acknowledgement rather than failing the request or
    guessing the figure's content.
    """
    if not has_diagram:
        return _NOT_A_DIAGRAM_QUESTION

    if diagram_image is None:
        # No caller extracts or stores a per-question diagram image yet, so this is handled like
        # any other unavailable case.
        logger.info(
            "has_diagram=True but no diagram_image was supplied (Phase 7 image extraction not "
            "built yet) -- degrading to the honest text-only path (Issue 56).",
        )
        return DiagramContext(available=False, acknowledgment=DIAGRAM_UNAVAILABLE_ACKNOWLEDGMENT)

    try:
        result: DiagramInterpretationResult = interpret_diagram(diagram_image)
    except DiagramInterpreterUnavailableError:
        logger.info(
            "Diagram interpreter unavailable/failed to load -- degrading to the honest text-only "
            "path (Issue 56), never treating the diagram as decorative (Issue 14).",
        )
        return DiagramContext(available=False, acknowledgment=DIAGRAM_UNAVAILABLE_ACKNOWLEDGMENT)
    except DiagramInterpretationTimeoutError:
        logger.info(
            "Diagram interpretation timed out -- degrading to the honest text-only path "
            "(Issue 56).",
        )
        return DiagramContext(available=False, acknowledgment=DIAGRAM_UNAVAILABLE_ACKNOWLEDGMENT)
    except Exception:
        # Catch-all, fail-closed like the safety classifier (Issue 210): an unexpected error in an
        # interpreter must not stop hint generation. It degrades like the named failures above
        # and is logged as an error.
        logger.error(
            "Diagram interpreter raised an unexpected exception -- degrading to the honest "
            "text-only path (Issue 56), never erroring the whole request.", exc_info=True,
        )
        return DiagramContext(available=False, acknowledgment=DIAGRAM_UNAVAILABLE_ACKNOWLEDGMENT)

    if result.confidence < MIN_TRUSTED_DIAGRAM_CONFIDENCE:
        logger.info(
            "Diagram interpretation confidence %.3f below trusted threshold %.3f -- degrading to "
            "the honest text-only path (Issue 56), not serving a low-confidence guess.",
            result.confidence, MIN_TRUSTED_DIAGRAM_CONFIDENCE,
        )
        return DiagramContext(available=False, acknowledgment=DIAGRAM_UNAVAILABLE_ACKNOWLEDGMENT)

    return DiagramContext(available=True, structured_data=result.structured_data)
