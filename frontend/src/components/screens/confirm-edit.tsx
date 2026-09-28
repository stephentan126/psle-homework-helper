"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Pencil, Check, Lightbulb, Camera } from "lucide-react";
import { Button } from "@/components/ui/button";
import { QuestionPhoto } from "@/components/ui/question-photo";

export interface ConfirmEditScreenProps {
  /** The raw text read from the student's photo. This is the only thing being confirmed. */
  extractedText: string;
  photoUrl?: string | null;
  hasDiagram?: boolean;
  /** Separate, clearly-labelled supporting context only, never the thing being confirmed. */
  matchedQuestionText?: string | null;
  /** Called on every edit, debounced by the caller if needed, to re-run match/safety via /edit. */
  onTextChange: (correctedText: string) => void;
  /** Whether an /edit re-check is in flight (short re-match feedback, not a full wait screen). */
  isRechecking?: boolean;
  onConfirm: () => void;
  /** Optional: go back and take a different photo (or type a different question). */
  onRetake?: () => void;
  /** Whether the text came from a photo or was typed, so the wording matches. */
  source?: "photo" | "typed";
}

export function ConfirmEditScreen({
  extractedText,
  photoUrl,
  hasDiagram,
  matchedQuestionText,
  onTextChange,
  isRechecking = false,
  onConfirm,
  onRetake,
  source = "photo",
}: ConfirmEditScreenProps) {
  const [text, setText] = useState(extractedText);
  const [dirty, setDirty] = useState(false);
  const typed = source === "typed";
  const normalise = (s: string) => s.replace(/\s+/g, " ").trim().toLowerCase();
  // Showing the matched question only helps when it differs from what the student sees above.
  const showMatch =
    !!matchedQuestionText && !dirty && normalise(matchedQuestionText) !== normalise(text);

  return (
    <div className="flex flex-1 flex-col overflow-y-auto">
      <div className="mx-auto flex w-full max-w-md flex-1 flex-col gap-6 px-6 py-8">
        <div className="flex flex-col gap-2 text-center">
          <h1 className="text-xl font-bold text-text">
            Does this look right?
          </h1>
          <p className="text-sm text-text/60">
            {typed
              ? "Check your question, and fix anything that's wrong before we help."
              : "Check what we read from your photo, and fix anything that's wrong before we help."}
          </p>
        </div>

        {/* Photo next to the editable text, so the student can compare the two */}
        <QuestionPhoto photoUrl={photoUrl} hasDiagram={hasDiagram} />

        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-secondary">
            <Pencil className="h-3.5 w-3.5" strokeWidth={2} />
            {typed ? "Your question (tap to fix it)" : "What we read (tap to fix it)"}
          </div>
          <textarea
            value={text}
            onChange={(e) => {
              setText(e.target.value);
              setDirty(true);
              onTextChange(e.target.value);
            }}
            rows={5}
            className="w-full resize-none rounded-md border border-border bg-white px-4 py-3 font-mono text-base leading-relaxed text-text focus:border-secondary focus:outline-none"
            placeholder="The question text will appear here"
            aria-label={
              typed
                ? "Your question, edit if it's wrong"
                : "What we read from your photo, edit if it's wrong"
            }
          />
          <AnimatePresence>
            {isRechecking && (
              <motion.p
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="text-xs text-text/50"
              >
                Checking that against our questions…
              </motion.p>
            )}
          </AnimatePresence>
        </div>

        {showMatch && (
          <div className="flex flex-col gap-2 rounded-md border border-border bg-white/60 px-4 py-3">
            <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-secondary">
              <Lightbulb className="h-3.5 w-3.5" strokeWidth={2} />
              We think this might be
            </div>
            <p className="text-sm leading-relaxed text-text/70">
              {matchedQuestionText}
            </p>
          </div>
        )}

        <div className="mt-auto pt-4">
          <Button variant="primary" fullWidth onClick={onConfirm}>
            <Check className="h-5 w-5" strokeWidth={2} />
            Yes, this is right
          </Button>
          {onRetake && (
            <Button variant="ghost" fullWidth onClick={onRetake} className="mt-2">
              {!typed && <Camera className="h-4 w-4" strokeWidth={2} />}
              {typed ? "Start again" : "Take another photo"}
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
