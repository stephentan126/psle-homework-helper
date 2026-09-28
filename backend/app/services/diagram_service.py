"""
Call point for the diagram-interpretation model (Section 6 of docs/ARCHITECTURE.md).

No Bucket 1, 2 or 3 diagram interpreter is wired into the pipeline yet, so `interpret_diagram()`
always raises `DiagramInterpreterUnavailableError`. Adding a model means replacing that
function's body only: every caller goes through
`app.pipeline.diagram_fallback.resolve_diagram_context()`, which degrades safely on this
exception (Issues 14 and 56).
"""
from __future__ import annotations

from dataclasses import dataclass


class DiagramInterpreterUnavailableError(Exception):
    """The diagram interpreter failed to load or is not available."""


class DiagramInterpretationTimeoutError(Exception):
    """A diagram-interpretation call exceeded its timeout."""


@dataclass(frozen=True)
class DiagramInterpretationResult:
    structured_data: dict
    confidence: float


# Conservative minimum trusted confidence, following Section 6's rule never to guess silently: a
# result below it is treated as unavailable rather than served as low-quality data. This is a
# placeholder, not a measured value, because no interpreter or gold set exists yet (Issues 242
# and 243). It must be calibrated against a gold set before an interpreter ships, as Issue 194
# did for MIN_TRUSTED_LOG_PROB.
MIN_TRUSTED_DIAGRAM_CONFIDENCE = 0.75


def interpret_diagram(
    diagram_image: bytes, timeout_s: float = 30.0,
) -> DiagramInterpretationResult:
    """Interprets a diagram image into structured data.

    No interpreter is built yet, so this always raises `DiagramInterpreterUnavailableError`,
    which callers handle through `app.pipeline.diagram_fallback.resolve_diagram_context()`.
    `timeout_s` is part of the signature so an implementation can be added without changing it.

    Raises:
        DiagramInterpreterUnavailableError: Always, until an interpreter is added.
    """
    raise DiagramInterpreterUnavailableError(
        "No Section 6 Bucket 1/2/3 diagram interpreter is built yet (Phase 7 not started, "
        "docs/DEVELOPMENT_LOG.md) -- this is the real, current state, not a bug.",
    )
