/**
 * Mirrors backend/app/api/schemas.py exactly: the six-state discriminated union, plus the
 * request bodies each endpoint accepts. Kept as a field-for-field port rather than a reshaped
 * frontend version. The backend schema comments are the single source of truth for what each
 * field means and when it is null; repeating that reasoning here would drift.
 */

export interface NeedsConfirmationResponse {
  state: "needs_confirmation";
  question_id: number | null;
  student_submission_id: number | null;
  match_confidence: number | null;
  extracted_text: string;
  matched_question_text: string | null;
  confidence: "high" | "low" | null;
}

export interface HintResponse {
  state: "hint";
  question_id: number;
  attempt_id: number | null;
  tier: number;
  hint_text: string;
  gate_path_used: "verified_match" | "weak_mode";
  is_unverified: boolean;
  can_escalate_further: boolean;
}

export interface CappedTier1Response {
  state: "capped_tier1";
  question_id: number | null;
  student_submission_id: number | null;
  attempt_id: number | null;
  hint_text: string;
  reason: string | null;
  gate_path_used: "verified_match" | "weak_mode";
  is_unverified: boolean;
  can_escalate_further: boolean;
}

export interface SafetyBlockedResponse {
  state: "safety_blocked";
  message: string;
  is_self_harm_path: boolean;
  resources: string[] | null;
  parent_notified: boolean;
}

export type ExtractionFailedReason =
  | "unreadable_photo"
  | "low_confidence"
  | "reader_unavailable"
  | "photo_not_straight";

export interface ExtractionFailedResponse {
  state: "extraction_failed";
  message: string;
  reason: ExtractionFailedReason;
}

export interface UnlockedSolutionResponse {
  state: "unlocked_solution";
  question_id: number;
  answer_value: string | null;
  worked_solution_text: string | null;
}

export type AnyHomeworkResponse =
  | NeedsConfirmationResponse
  | HintResponse
  | CappedTier1Response
  | SafetyBlockedResponse
  | ExtractionFailedResponse
  | UnlockedSolutionResponse;

// --- Request bodies, matching each endpoint's Pydantic model ---

export interface PhotoQuestionSubmission {
  question_text: string;
  has_diagram?: boolean;
}

export interface EditSubmissionRequest {
  corrected_text: string;
}

export interface PinUnlockRequest {
  pin: string;
}
