"use client";

import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { RotateCcw, WifiOff } from "lucide-react";
import { Button } from "@/components/ui/button";

export type WaitKind = "photo" | "typed" | "hint";

interface StageDef {
  /** seconds after which this stage's copy becomes active */
  at: number;
  copy: string;
}

const STAGES: Record<WaitKind, StageDef[]> = {
  photo: [
    { at: 0, copy: "Reading your photo…" },
    { at: 15, copy: "Making out the numbers and words…" },
    { at: 28, copy: "Almost there…" },
    { at: 60, copy: "Still working, thanks for waiting…" },
  ],
  typed: [
    { at: 0, copy: "Reading your question…" },
    { at: 15, copy: "Checking it against our questions…" },
    { at: 28, copy: "Almost there…" },
    { at: 60, copy: "Still working, thanks for waiting…" },
  ],
  hint: [
    { at: 0, copy: "Working through your question…" },
    { at: 20, copy: "Checking the answer carefully…" },
    { at: 45, copy: "Making sure the hint is right…" },
    { at: 80, copy: "Still working, thanks for waiting…" },
  ],
};

/** Client-side timeout: above the measured worst case, including one server restart. */
const TIMEOUT_SECONDS: Record<WaitKind, number> = {
  photo: 180,
  typed: 180,
  hint: 180,
};

interface WorkingOnItProps {
  kind: WaitKind;
  /** Reused verbatim on retry, never regenerated. Caller owns this. */
  requestId: string;
  onRetry: () => void;
  /** Set true when the request has failed (network or timeout), to show the failure state. */
  failed?: boolean;
}

export function WorkingOnIt({ kind, onRetry, failed = false }: WorkingOnItProps) {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (failed) return;
    const startedAt = Date.now();
    const id = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startedAt) / 1000));
    }, 1000);
    return () => clearInterval(id);
  }, [failed, kind]);

  const stages = STAGES[kind];
  const currentStage = [...stages].reverse().find((s) => elapsed >= s.at) ?? stages[0];
  const timeoutAt = TIMEOUT_SECONDS[kind];
  const timedOut = !failed && elapsed >= timeoutAt;
  const showFailure = failed || timedOut;

  return (
    <div className="flex flex-1 flex-col items-center overflow-y-auto px-6 py-16 text-center">
      <AnimatePresence mode="wait">
        {!showFailure ? (
          <motion.div
            key="working"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3, ease: "easeOut" }}
            className="flex flex-col items-center gap-8"
          >
            <PatientSpinner />
            <div aria-live="polite" aria-atomic="true">
              <AnimatePresence mode="wait">
                <motion.p
                  key={currentStage.copy}
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -6 }}
                  transition={{ duration: 0.35, ease: "easeOut" }}
                  className="max-w-xs text-lg font-medium text-text"
                >
                  {currentStage.copy}
                </motion.p>
              </AnimatePresence>
            </div>
            <p className="max-w-xs text-sm text-text/60">
              This usually takes up to a minute. We&apos;d rather get it right
              than rush it.
            </p>
          </motion.div>
        ) : (
          <motion.div
            key="failed"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.35, ease: "easeOut" }}
            className="flex flex-col items-center gap-6"
          >
            <div className="flex h-16 w-16 items-center justify-center rounded-full bg-caution-bg text-caution-text">
              <WifiOff className="h-8 w-8" strokeWidth={1.75} />
            </div>
            <div className="flex flex-col gap-2">
              <p className="text-lg font-bold text-text">
                That&apos;s taking too long
              </p>
              <p className="max-w-xs text-sm text-text/60">
                Something went wrong on our end. Your question is safe, just
                tap below to try again.
              </p>
            </div>
            <Button variant="primary" onClick={onRetry} className="mt-2">
              <RotateCcw className="h-5 w-5" strokeWidth={2} />
              Try again
            </Button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

/** A calm spinner. Not a progress bar, which would imply false precision. */
function PatientSpinner() {
  return (
    <div className="relative h-20 w-20" aria-hidden="true">
      <motion.div
        className="absolute inset-0 rounded-full border-4 border-border"
        style={{ borderTopColor: "var(--color-primary)" }}
        animate={{ rotate: 360 }}
        transition={{ duration: 1.4, repeat: Infinity, ease: "linear" }}
      />
    </div>
  );
}
