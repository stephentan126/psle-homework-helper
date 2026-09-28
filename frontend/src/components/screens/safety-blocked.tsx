"use client";

import { Heart, ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";

export interface SafetyBlockedScreenProps {
  message: string;
  isSelfHarmPath?: boolean;
  resources?: string[] | null;
  onBackToStart: () => void;
}

/**
 * The care palette is used only on this screen. It is kept distinct from every other colour
 * role and is not reused elsewhere, so this screen reads as calm and separate: not branded
 * and not alarming.
 */
export function SafetyBlockedScreen({
  message,
  isSelfHarmPath = false,
  resources,
  onBackToStart,
}: SafetyBlockedScreenProps) {
  return (
    <div className="flex flex-1 flex-col items-center overflow-y-auto bg-care-bg px-6 py-16 text-center">
      <div className="flex w-full max-w-xs flex-col items-center gap-6">
        <div className="flex h-16 w-16 items-center justify-center rounded-full bg-white text-care-text">
          <Heart className="h-8 w-8" strokeWidth={1.75} />
        </div>

        <p className="text-lg leading-relaxed text-care-text">{message}</p>

        {isSelfHarmPath && resources && resources.length > 0 && (
          <div className="flex w-full flex-col gap-2 rounded-md border border-care-border bg-white/60 px-4 py-4 text-left">
            {resources.map((resource) => (
              <p key={resource} className="text-sm leading-relaxed text-care-text">
                {resource}
              </p>
            ))}
          </div>
        )}

        <Button variant="secondary" fullWidth onClick={onBackToStart} className="mt-2">
          <ArrowLeft className="h-5 w-5" strokeWidth={2} />
          Back to start
        </Button>
      </div>
    </div>
  );
}
