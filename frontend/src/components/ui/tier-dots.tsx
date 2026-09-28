"use client";

import { motion } from "framer-motion";
import { cn } from "@/lib/cn";

interface TierDotsProps {
  tier: number; // 1-3, the current tier being shown
  className?: string;
}

/** Three hint-tier dots. Reached tiers are filled; the current tier bounces in. */
export function TierDots({ tier, className }: TierDotsProps) {
  return (
    <div
      className={cn("flex items-center gap-2", className)}
      role="img"
      aria-label={`Hint ${tier} of 3`}
    >
      {[1, 2, 3].map((dot) => {
        const filled = dot <= tier;
        const isCurrent = dot === tier;
        return (
          <motion.span
            key={dot}
            className={cn(
              "h-3 w-3 rounded-full",
              filled ? "bg-primary" : "bg-border",
            )}
            initial={isCurrent ? { scale: 0.4 } : false}
            animate={isCurrent ? { scale: 1 } : {}}
            transition={{ type: "spring", stiffness: 500, damping: 15 }}
          />
        );
      })}
    </div>
  );
}
