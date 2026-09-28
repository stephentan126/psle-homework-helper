"use client";

import { useState } from "react";
import { Camera, Keyboard, ShieldCheck, KeyRound } from "lucide-react";
import { Button } from "@/components/ui/button";
import { CautionBadge } from "@/components/ui/badge";

export type AskMode = "photo" | "type";

export interface LandingAskScreenProps {
  onTakePhoto: () => void;
  onSubmitTyped: (questionText: string) => void;
}

/**
 * Screen 1 of 9. Does two jobs on one page:
 * - Functional, above the fold: the question entry (photo or typed).
 * - Trust and explainer, on the same page: the safety promise, what the owl is,
 *   and a note that the unverified (weak-mode) badge exists.
 * There is no sign-up funnel, testimonials or pricing, because there is no
 * multi-family account system.
 */
export function LandingAskScreen({ onTakePhoto, onSubmitTyped }: LandingAskScreenProps) {
  const [mode, setMode] = useState<AskMode>("photo");
  const [typedText, setTypedText] = useState("");

  return (
    <div className="flex flex-1 flex-col items-center overflow-y-auto px-6 py-12">
      <div className="flex w-full max-w-md flex-col items-center gap-8">
        {/* Owl mascot + name, friendly, not corporate */}
        <div className="flex flex-col items-center gap-3 text-center">
          <div
            className="flex h-16 w-16 items-center justify-center rounded-full bg-primary/10 text-4xl"
            aria-hidden="true"
          >
            🦉
          </div>
          <div className="flex flex-col gap-1">
            <p className="text-lg font-bold text-text">Hi, I&apos;m Hoot</p>
            <p className="max-w-xs text-sm text-text/60">
              Show me your maths question, and I&apos;ll help you work through it step by
              step, I won&apos;t just give you the answer.
            </p>
          </div>
        </div>

        {/* Functional entry: the immediate task, above the fold */}
        <div className="flex w-full flex-col gap-3">
          {mode === "photo" ? (
            <>
              <Button variant="primary" fullWidth onClick={onTakePhoto}>
                <Camera className="h-5 w-5" strokeWidth={2} />
                Take a photo of your question
              </Button>
              <Button variant="secondary" fullWidth onClick={() => setMode("type")}>
                <Keyboard className="h-5 w-5" strokeWidth={2} />
                Type it instead
              </Button>
            </>
          ) : (
            <>
              <textarea
                value={typedText}
                onChange={(e) => setTypedText(e.target.value)}
                placeholder="Type your question here…"
                aria-label="Type your maths question"
                rows={4}
                className="w-full resize-none rounded-md border border-border bg-white px-4 py-3 font-mono text-base text-text focus:border-secondary focus:outline-none"
              />
              <Button
                variant="primary"
                fullWidth
                disabled={typedText.trim().length === 0}
                onClick={() => onSubmitTyped(typedText)}
              >
                Get a hint
              </Button>
              <Button variant="secondary" fullWidth onClick={() => setMode("photo")}>
                <Camera className="h-5 w-5" strokeWidth={2} />
                Take a photo instead
              </Button>
            </>
          )}
        </div>

        {/* Trust / explainer, same page, not hidden behind another screen */}
        <div className="flex w-full flex-col gap-3 rounded-md border border-border bg-white/60 px-4 py-4">
          <div className="flex items-start gap-2.5">
            <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-secondary" strokeWidth={2} />
            <p className="text-sm leading-relaxed text-text/70">
              Hoot gives hints, not answers. The final answer only unlocks with a{" "}
              <span className="font-semibold text-text">parent&apos;s PIN</span>, for one
              question at a time.
            </p>
          </div>
          <div className="flex items-start gap-2.5">
            <KeyRound className="mt-0.5 h-4 w-4 shrink-0 text-secondary" strokeWidth={2} />
            <p className="text-sm leading-relaxed text-text/70">
              Sometimes Hoot isn&apos;t fully sure about a question. When that happens you&apos;ll
              see a badge like this, right on the hint:
            </p>
          </div>
          <div className="pl-6.5">
            <CautionBadge>We&apos;re not fully sure about this one</CautionBadge>
          </div>
        </div>
      </div>
    </div>
  );
}
