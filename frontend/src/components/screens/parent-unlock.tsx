"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { KeyRound, AlertTriangle, Clock, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/cn";

export type UnlockErrorKind =
  | "wrong_pin"
  | "locked_out"
  | "nothing_to_unlock"
  | "verification_pending";

export interface ParentUnlockScreenProps {
  onSubmit: (pin: string) => void;
  onClose: () => void;
  isSubmitting?: boolean;
  error?: UnlockErrorKind | null;
  lockedUntilLabel?: string | null;
  unlocked?: {
    answerValue?: string | null;
    workedSolutionText?: string | null;
  } | null;
}

const ERROR_COPY: Record<UnlockErrorKind, string> = {
  wrong_pin: "That PIN isn't right. Try again.",
  locked_out: "Too many attempts.",
  nothing_to_unlock:
    "There's nothing to unlock for this question, no stored answer exists yet.",
  verification_pending:
    "This question's stored answer is still being checked and isn't available to unlock yet.",
};

export function ParentUnlockScreen({
  onSubmit,
  onClose,
  isSubmitting = false,
  error,
  lockedUntilLabel,
  unlocked,
}: ParentUnlockScreenProps) {
  const [pin, setPin] = useState("");
  const isLocked = error === "locked_out";
  const isBlocked = error === "nothing_to_unlock" || error === "verification_pending";

  if (unlocked) {
    return (
      <div className="flex flex-1 flex-col items-center overflow-y-auto px-6 py-16 text-center">
        <div className="flex w-full max-w-sm flex-col items-center gap-6">
          <div className="flex h-16 w-16 items-center justify-center rounded-full bg-primary/10 text-primary">
            <KeyRound className="h-8 w-8" strokeWidth={1.75} />
          </div>
          <p className="text-lg font-bold text-text">Unlocked for this question</p>
          {unlocked.answerValue && (
            <div className="w-full rounded-md border border-border bg-white px-4 py-3 text-left">
              <span className="text-xs font-semibold uppercase tracking-wide text-secondary">
                Answer
              </span>
              <p className="font-mono text-lg text-text">{unlocked.answerValue}</p>
            </div>
          )}
          {unlocked.workedSolutionText && (
            <div className="w-full rounded-md border border-border bg-white px-4 py-3 text-left">
              <span className="text-xs font-semibold uppercase tracking-wide text-secondary">
                Worked solution
              </span>
              <p className="mt-1 text-sm leading-relaxed text-text">
                {unlocked.workedSolutionText}
              </p>
            </div>
          )}
          <p className="text-xs text-text/50">
            This unlocked only this one question. Every other question still needs its own PIN.
          </p>
          <Button variant="primary" fullWidth onClick={onClose}>
            Done
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-1 flex-col items-center overflow-y-auto px-6 py-16 text-center">
      <div className="flex w-full max-w-xs flex-col items-center gap-6">
        <button
          onClick={onClose}
          aria-label="Close"
          className="-mr-3 -mt-3 self-end rounded-full p-3 text-text/40 hover:text-text/70"
        >
          <X className="h-5 w-5" strokeWidth={2} />
        </button>

        <div className="flex h-16 w-16 items-center justify-center rounded-full bg-secondary/10 text-secondary">
          <KeyRound className="h-8 w-8" strokeWidth={1.75} />
        </div>

        <div className="flex flex-col gap-2">
          <p className="text-lg font-bold text-text">Parent PIN</p>
          <p className="text-sm text-text/60">
            Enter your PIN to unlock the full answer for this question only.
          </p>
        </div>

        {isBlocked ? (
          <div className="flex flex-col items-center gap-3 rounded-md border border-border bg-white/60 px-4 py-4">
            <AlertTriangle className="h-6 w-6 text-caution-text" strokeWidth={1.75} />
            <p className="text-sm leading-relaxed text-text/70">
              {ERROR_COPY[error as UnlockErrorKind]}
            </p>
          </div>
        ) : (
          <>
            <input
              type="password"
              inputMode="numeric"
              autoComplete="off"
              value={pin}
              onChange={(e) => setPin(e.target.value.replace(/[^0-9]/g, "").slice(0, 8))}
              disabled={isLocked || isSubmitting}
              placeholder="••••"
              aria-label="Parent PIN"
              className={cn(
                "w-full rounded-md border px-4 py-3 text-center font-mono text-2xl tracking-[0.3em] text-text focus:outline-none",
                error === "wrong_pin"
                  ? "border-error focus:border-error"
                  : "border-border bg-white focus:border-secondary",
              )}
            />

            <AnimatePresence>
              {error === "wrong_pin" && (
                <motion.p
                  initial={{ opacity: 0, y: -4 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  className="-mt-3 text-sm font-medium text-error"
                >
                  {ERROR_COPY.wrong_pin}
                </motion.p>
              )}
              {isLocked && (
                <motion.div
                  initial={{ opacity: 0, y: -4 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  className="-mt-3 flex items-center gap-2 text-sm font-medium text-caution-text"
                >
                  <Clock className="h-4 w-4" strokeWidth={2} />
                  {ERROR_COPY.locked_out}
                  {lockedUntilLabel && <span>Try again after {lockedUntilLabel}.</span>}
                </motion.div>
              )}
            </AnimatePresence>

            <Button
              variant="primary"
              fullWidth
              disabled={pin.length < 4 || isLocked || isSubmitting}
              onClick={() => onSubmit(pin)}
            >
              {isSubmitting ? "Checking…" : "Unlock"}
            </Button>
          </>
        )}
      </div>
    </div>
  );
}
