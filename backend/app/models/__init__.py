"""
SQLAlchemy ORM models (design specification, Section 10.1).

Importing this package registers every model on `Base.metadata`, so `init_db()` creates the full
schema regardless of which modules a caller has imported.
"""
from app.models.attempt import Attempt
from app.models.base import Base
from app.models.hint_event import HintEvent
from app.models.ocr_extraction_record import OCRExtractionRecord
from app.models.parent_account import ParentAccount
from app.models.precomputed_gate_result import PrecomputedGateResult
from app.models.question import Question
from app.models.question_embedding import QuestionEmbedding
from app.models.safety_flag import SafetyFlag
from app.models.student import Student
from app.models.student_submission import StudentSubmission

# Every model is imported and listed here explicitly. Relying on callers to import a model
# (for example `PrecomputedGateResult`) before `init_db()` runs would make the created schema
# depend on import order.
__all__ = [
    "Base",
    "Question",
    "OCRExtractionRecord",
    "Student",
    "ParentAccount",
    "Attempt",
    "HintEvent",
    "SafetyFlag",
    "StudentSubmission",
    "QuestionEmbedding",
    "PrecomputedGateResult",
]
