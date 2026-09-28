import { AlertCircle } from "lucide-react";
import { cn } from "@/lib/cn";

interface CautionBadgeProps {
  children: React.ReactNode;
  className?: string;
}

/** Caution badge: amber, pill radius, always visible. Never a dismissible toast. */
export function CautionBadge({ children, className }: CautionBadgeProps) {
  return (
    <div
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full bg-caution-bg px-3.5 py-1.5 text-xs font-semibold text-caution-text",
        className,
      )}
    >
      <AlertCircle className="h-3.5 w-3.5" strokeWidth={2} />
      {children}
    </div>
  );
}
