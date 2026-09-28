"use client";

import { Camera, Keyboard, RotateCw, ServerCrash } from "lucide-react";
import { Button } from "@/components/ui/button";

export type ExtractionFailedReason =
  | "unreadable_photo"
  | "low_confidence"
  | "reader_unavailable"
  | "photo_not_straight";

export interface ExtractionFailedScreenProps {
  reason: ExtractionFailedReason;
  onRetakePhoto: () => void;
  onTypeInstead: () => void;
}

const COPY: Record<
  ExtractionFailedReason,
  { title: string; body: string; icon: React.ComponentType<{ className?: string; strokeWidth?: number }>; primaryLabel: string; primaryAction: "retake" | "type" }
> = {
  unreadable_photo: {
    title: "We couldn't read that photo",
    body: "The photo didn't come through clearly enough. Try taking another one with good light, close up on just the question.",
    icon: Camera,
    primaryLabel: "Take another photo",
    primaryAction: "retake",
  },
  low_confidence: {
    title: "We're not confident we read that right",
    body: "The writing was hard to make out. Try taking another photo with good light, close up on just the question.",
    icon: Camera,
    primaryLabel: "Take another photo",
    primaryAction: "retake",
  },
  photo_not_straight: {
    title: "That photo looks tilted",
    body: "Straighten the page and line the question up flat before taking the photo again.",
    icon: RotateCw,
    primaryLabel: "Retake the photo",
    primaryAction: "retake",
  },
  reader_unavailable: {
    title: "Our photo reader is down right now",
    body: "This isn't about your photo. Our system for reading photos is temporarily unavailable, so please type your question instead.",
    icon: ServerCrash,
    primaryLabel: "Type my question instead",
    primaryAction: "type",
  },
};

export function ExtractionFailedScreen({
  reason,
  onRetakePhoto,
  onTypeInstead,
}: ExtractionFailedScreenProps) {
  const copy = COPY[reason];
  const Icon = copy.icon;
  const primaryOnClick = copy.primaryAction === "retake" ? onRetakePhoto : onTypeInstead;

  return (
    <div className="flex flex-1 flex-col items-center overflow-y-auto px-6 py-16 text-center">
      <div className="flex w-full max-w-xs flex-col items-center gap-6">
        <div className="flex h-16 w-16 items-center justify-center rounded-full bg-caution-bg text-caution-text">
          <Icon className="h-8 w-8" strokeWidth={1.75} />
        </div>
        <div className="flex flex-col gap-2">
          <p className="text-lg font-bold text-text">{copy.title}</p>
          <p className="text-sm leading-relaxed text-text/60">{copy.body}</p>
        </div>
        <div className="flex w-full flex-col gap-3">
          <Button variant="primary" fullWidth onClick={primaryOnClick}>
            {copy.primaryAction === "retake" ? (
              <Camera className="h-5 w-5" strokeWidth={2} />
            ) : (
              <Keyboard className="h-5 w-5" strokeWidth={2} />
            )}
            {copy.primaryLabel}
          </Button>
          {copy.primaryAction === "retake" && (
            <Button variant="secondary" fullWidth onClick={onTypeInstead}>
              <Keyboard className="h-5 w-5" strokeWidth={2} />
              Type it instead
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
