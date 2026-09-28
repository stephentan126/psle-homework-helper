import { ImageOff } from "lucide-react";
import { cn } from "@/lib/cn";

interface QuestionPhotoProps {
  /** URL to the student's photo, when the backend photo-serving route has one to give. */
  photoUrl?: string | null;
  /** Whether this question is known to have a diagram, even when no image is available yet. */
  hasDiagram?: boolean;
  className?: string;
}

/**
 * Handles "no image available" without looking broken. Corpus-matched diagram questions have
 * no stored image, so this is an expected state rather than an error.
 */
export function QuestionPhoto({ photoUrl, hasDiagram, className }: QuestionPhotoProps) {
  if (photoUrl) {
    return (
      <div
        className={cn(
          "overflow-hidden rounded-md border border-border bg-white",
          className,
        )}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={photoUrl}
          alt="Your photo of this question"
          className="block w-full object-contain"
        />
      </div>
    );
  }

  if (hasDiagram) {
    return (
      <div
        className={cn(
          "flex flex-col items-center gap-2 rounded-md border border-dashed border-border bg-white/60 px-4 py-6 text-center",
          className,
        )}
      >
        <ImageOff className="h-6 w-6 text-text/40" strokeWidth={1.75} />
        <p className="text-sm text-text/60">
          This question has a diagram, but we don&apos;t have an image saved
          for it yet.
        </p>
      </div>
    );
  }

  return null;
}
