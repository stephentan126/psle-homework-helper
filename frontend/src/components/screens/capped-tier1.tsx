"use client";

import { KeyRound } from "lucide-react";
import { Button } from "@/components/ui/button";
import { CautionBadge } from "@/components/ui/badge";
import { QuestionPhoto } from "@/components/ui/question-photo";

export interface CappedTier1ScreenProps {
  questionText: string;
  hintText: string;
  photoUrl?: string | null;
  hasDiagram?: boolean;
  // Weak-mode capped_tier1 responses carry no question_id (backend/app/api/schemas.py sets
  // exactly one of question_id or student_submission_id). The unlock endpoint is
  // /questions/{question_id}/unlock only, so a student_submission_id cannot be unlocked.
  // null here means "nothing to unlock", and the button shows that rather than doing
  // nothing on click.
  onParentUnlock: (() => void) | null;
  /** Optional: leave this question and start again. */
  onNewQuestion?: () => void;
}

/**
 * Fallback for a wobbly, mismatched or missing answer, and the hard ceiling for weak mode.
 * Always Tier 1 and never escalates. It should read as a normal outcome, not a failure.
 */
export function CappedTier1Screen({
  questionText,
  hintText,
  photoUrl,
  hasDiagram,
  onParentUnlock,
  onNewQuestion,
}: CappedTier1ScreenProps) {
  return (
    <div className="flex flex-1 flex-col overflow-y-auto">
      <div className="mx-auto flex w-full max-w-md flex-1 flex-col gap-6 px-6 py-8">
        <div className="flex flex-col gap-3 rounded-md border border-border bg-white/60 px-4 py-4">
          <span className="text-xs font-semibold uppercase tracking-wide text-secondary">
            Your question
          </span>
          <p className="font-mono text-lg leading-relaxed text-text">
            {questionText}
          </p>
          <QuestionPhoto photoUrl={photoUrl} hasDiagram={hasDiagram} />
        </div>

        <div className="flex flex-col items-center gap-3 pt-2 text-center">
          <CautionBadge>We&apos;re not fully sure about this one</CautionBadge>
        </div>

        <div className="flex-1 rounded-lg border border-border bg-white px-5 py-6">
          <p className="whitespace-pre-line text-lg leading-relaxed text-text">{hintText}</p>
        </div>

        <div className="flex flex-col gap-3 pb-4">
          <p className="text-center text-sm text-text/60">
            {onParentUnlock
              ? "We can't safely give more hints on this one yet. A parent can unlock the full answer if you need it."
              : "We can't safely give more hints on this one yet, and there's no stored answer to unlock for this question."}
          </p>
          {onParentUnlock && (
            <Button variant="secondary" fullWidth onClick={onParentUnlock}>
              <KeyRound className="h-5 w-5" strokeWidth={2} />
              Ask a parent for the answer
            </Button>
          )}
          {onNewQuestion && (
            <Button variant="ghost" fullWidth onClick={onNewQuestion}>
              Ask a different question
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
