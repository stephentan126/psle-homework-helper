"use client";

import { motion } from "framer-motion";
import { ChevronRight, KeyRound } from "lucide-react";
import { Button } from "@/components/ui/button";
import { TierDots } from "@/components/ui/tier-dots";
import { QuestionPhoto } from "@/components/ui/question-photo";

export interface HintScreenProps {
  questionText: string;
  tier: number; // 1-3
  hintText: string;
  canEscalateFurther: boolean;
  photoUrl?: string | null;
  hasDiagram?: boolean;
  onEscalate: () => void;
  onParentUnlock: () => void;
  /** Optional: leave this question and start again. */
  onNewQuestion?: () => void;
}

export function HintScreen({
  questionText,
  tier,
  hintText,
  canEscalateFurther,
  photoUrl,
  hasDiagram,
  onEscalate,
  onParentUnlock,
  onNewQuestion,
}: HintScreenProps) {
  return (
    <div className="flex flex-1 flex-col overflow-y-auto">
      <div className="mx-auto flex w-full max-w-md flex-1 flex-col gap-6 px-6 py-8">
        {/* Question, secondary in the hierarchy: the hint is the main task */}
        <div className="flex flex-col gap-3 rounded-md border border-border bg-white/60 px-4 py-4">
          <span className="text-xs font-semibold uppercase tracking-wide text-secondary">
            Your question
          </span>
          <p className="font-mono text-lg leading-relaxed text-text">
            {questionText}
          </p>
          <QuestionPhoto photoUrl={photoUrl} hasDiagram={hasDiagram} />
        </div>

        {/* The primary task: the hint itself */}
        <div className="flex flex-col items-center gap-4 pt-2 text-center">
          <TierDots tier={tier} />
          <span className="text-sm font-semibold text-text/60">
            Hint {tier} of 3
          </span>
        </div>

        <motion.div
          key={tier}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35, ease: "easeOut" }}
          className="flex-1 rounded-lg border border-border bg-white px-5 py-6"
        >
          <p className="whitespace-pre-line text-lg leading-relaxed text-text">{hintText}</p>
        </motion.div>

        <div className="flex flex-col gap-3 pb-4">
          {canEscalateFurther && (
            <Button variant="primary" fullWidth onClick={onEscalate}>
              Still stuck? Get another hint
              <ChevronRight className="h-5 w-5" strokeWidth={2} />
            </Button>
          )}
          <Button variant="secondary" fullWidth onClick={onParentUnlock}>
            <KeyRound className="h-5 w-5" strokeWidth={2} />
            Ask a parent for the answer
          </Button>
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
