"""
`questions` table (design specification, Section 10.1): one row per question produced by the
extraction pipeline (`ExtractedQuestion` in `app/pipeline/extract.py`).

The columns follow `ExtractedQuestion`, not the field list in Section 10.1, which is out of date.
Section 10.1 omits later additions (`school_name` and `school_name_verified`, Issues 92-93;
`extraction_flag`; `question_section`, Issue 91) and lists three fields that were never produced
(`paper_type`, `topic`, `diagram_path`).

The `*_FLAG` constants below are specific `extraction_flag` values. Each is checked by exact value
rather than "flag is non-null", because many valid rows carry other flags (for example
`no_matching_answer_key_row`). Flags never delete or edit a row: they mark it for exclusion or
record why it was corrected, so the history stays auditable.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sqlalchemy import Boolean, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.ocr_extraction_record import OCRExtractionRecord

# Marks a row whose `question_text` is not a question at all: exam cover-page or header
# boilerplate extracted as question text, sometimes with an unrelated `answer_value` attached
# (Issue 209). Excluding every row with any flag would instead drop about 27% of the corpus (1,487
# valid rows with other flags). Checked at the serving-time lookups
# (`question_matching.find_best_match()`, `routes.request_hint()`).
NON_QUESTION_EXTRACTION_FLAG = "non_question_boilerplate"

# Marks a row whose `answer_value` was corrected by an individually verified database patch, not a
# pipeline change (Issues 239 and 241). The original value came from the wrong column or section
# of the answer-key grid (a rowspan/colspan parsing gap, which was left unchanged). Each original
# value was confirmed wrong against the raw grid, and each corrected value was checked against the
# question's arithmetic and options. The one-off patch script is not included.
# Unlike `NON_QUESTION_EXTRACTION_FLAG`, the row is a valid question; the flag keeps the
# correction visible for audit.
ANSWER_VALUE_CORRECTED_LANDMINE_239_FLAG = "answer_value_corrected_landmine239"

# Quarantine marker (Issue 248): the row's `answer_value` is confirmed wrong and not yet
# corrected. It is set as soon as the error is found, before any fix, so the app stops serving the
# value immediately. Checked by exact value at the two serving paths that need it:
# `question_matching.find_best_match()` (match-first retrieval) and `auth_routes.unlock_solution()`
# (parent PIN unlock, which has no gate to catch a bad stored value). Applied to the two
# rows a reachability test showed were exposed (ids 1572 and 3019); nine further defective rows
# from the same sample (Issues 246-247) were not flagged under this issue. Unlike
# `ANSWER_VALUE_CORRECTED_LANDMINE_239_FLAG`, which marks a row already fixed, this marks one that
# is not yet fixed.
ANSWER_VERIFICATION_PENDING_FLAG = "answer_key_verification_pending"

# Records why `worked_solution_text` is blank: the printed answer key shows only the final value,
# with no working (Issue 305). This describes the source document, not an extraction defect, so no
# repair script clears it (unlike `ANSWER_VERIFICATION_PENDING_FLAG`). Applied to the 7 rows from
# P6_Maths_2023_WA1_Nanhua.pdf repaired under Issue 305. Other rows with a blank
# `worked_solution_text` and no flag were not backfilled.
NO_PRINTED_WORKING_FLAG = "no_printed_working_shown"

# Same individually verified patch pattern as `ANSWER_VALUE_CORRECTED_LANDMINE_239_FLAG`, but a
# different root cause (Issue 307): the answer was read from the wrong cell or page across Paper 1
# and Paper 2 (the Issue 305 defect family), not from a grid-parsing gap. Applied to ids 3876, 3877
# and 3878 (`P6_Maths_2024_WA1_Nanhua.pdf`), each checked against the printed working first.
ANSWER_VALUE_CORRECTED_LANDMINE_307_FLAG = "answer_value_corrected_landmine307"

# Excludes from the training pool the training-side row of each exact `question_text` match
# between the held-out evaluation set and the training pool (follow-up to Issue 252). The Issue 252
# overlap review (`held_out_training_overlap_candidates.json`, 116 candidates) found the pairs were
# different questions (different diagrams), so the frozen held-out set was not contaminated. A
# text-only retriever would still return the training twin at near-1.0 similarity, since it cannot
# see the diagrams, so the training side is excluded. Non-destructive, following the
# `superseded_by` convention (Issue 231): rows are never deleted and their text and answers are
# never edited. Applied to ids 3336, 3480, 3651, 5193 and 5334, the training side of the 4
# confirmed matches (held-out id 7 matched both 3336 and 5193, a "net of a cuboid?" stem). The
# held-out database is frozen and read-only (Issue 251) and was not changed. This flag covers the
# current pool only; it does not replace a retrieval-time similarity exclusion for new questions.
TRAINING_POOL_EXCLUDED_HELD_OUT_OVERLAP_FLAG = "training_pool_excluded_held_out_overlap_landmine252"

# Same individually verified patch pattern as the 239 and 307 flags, with a third root cause
# (Issue 308). `P6_Maths_2025_WA1_nanhua.pdf` binds two separate exam sittings into one PDF:
# "Revision 1" (pages 0-18, key on page 16) and "Revision 2" (pages 19-35, key on pages 34-35).
# Extraction used the Revision 1 key almost everywhere. Applied to the 26 rows checked against
# both printed keys:
# - Revision 1 rows pointing at the wrong key: 4684, 4686, 4701 (4684 and 4686 already had the
#   correct value by coincidence; only the page pointer was wrong).
# - Revision 2 rows that took Revision 1's answer_value and answer_source_page_index: 4703, 4704,
#   4705, 4708-4723, 4725, 5803, 5804, 5805.
# Not applied to ids 5802, 4706, 4707, 4724 or the paper's other 23 rows, which were confirmed
# correct in a full sweep of all 53 rows. Whether other PDFs bind two sittings together has not
# been checked.
ANSWER_VALUE_CORRECTED_LANDMINE_308_FLAG = "answer_value_corrected_landmine308"

# Permanent exclusion, following the same non-destructive pattern as
# `TRAINING_POOL_EXCLUDED_HELD_OUT_OVERLAP_FLAG` (Issue 314). Pages 42-52 of
# `P6_Maths_2024_SA2_nanyang.pdf` are not a question paper but scanned, graded answer scripts from
# several papers (handwritten MCQ choices, ticks, margin working). The 56 rows extracted from those
# pages already carry `no_matching_answer_key_row` with an empty `answer_value`, so no wrong
# answer was ever served, but most have corrupted `question_text` (OCR noise, merged questions, a
# garbage `question_number`). They are excluded rather than repaired, because repairing them would
# mean transcribing a different document. None of the 56 were in the training pool or the hint
# review queue. Whether other PDFs contain bound-in scripts has not been checked.
TRAINING_POOL_EXCLUDED_CORRUPTED_SOURCE_LANDMINE_314_FLAG = (
    "training_pool_excluded_corrupted_source_landmine314"
)


class Question(Base):
    __tablename__ = "questions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    source_paper: Mapped[str] = mapped_column(String, nullable=False)
    source_page_index: Mapped[int] = mapped_column(Integer, nullable=False)
    source_page_label: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    answer_source_file: Mapped[str] = mapped_column(String, nullable=False)
    answer_source_page_index: Mapped[int] = mapped_column(Integer, nullable=False)

    question_number: Mapped[str] = mapped_column(String, nullable=False)
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    answer_value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    worked_solution_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    has_diagram: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    ocr_confidence: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # "high"/"low"
    school_name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    school_name_verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    verification_status: Mapped[str] = mapped_column(String, nullable=False, default="unverified")
    # Stored as the ISO-8601 string extract.py produces (via the JSON checkpoint in save()), not a
    # DateTime column. SQLite's DateTime type expects "YYYY-MM-DD HH:MM:SS[.ffffff]" with no "T"
    # separator or timezone offset, which isoformat() ("...T...+00:00") does not match; a
    # round-trip test confirmed this (Issue 102). Other timestamp columns follow this convention.
    verified_at: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    superseded_by: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("questions.id"), nullable=True,
    )
    extraction_flag: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    question_section: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    ocr_records: Mapped[list["OCRExtractionRecord"]] = relationship(
        back_populates="question", cascade="all, delete-orphan",
    )
