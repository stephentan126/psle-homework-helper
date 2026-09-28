"use client";

import { forwardRef } from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/cn";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 rounded-lg px-8 py-4 font-sans font-semibold text-base transition-[transform,box-shadow] duration-100 ease-out select-none disabled:opacity-50 disabled:pointer-events-none active:translate-y-[3px]",
  {
    variants: {
      variant: {
        primary:
          "bg-primary text-white shadow-[0_4px_0_var(--color-primary-shadow)] active:shadow-[0_1px_0_var(--color-primary-shadow)]",
        ghost:
          "bg-transparent text-text/60 px-4 py-2 hover:text-text active:translate-y-0",
        secondary:
          "bg-white text-secondary border-2 border-secondary shadow-[0_4px_0_var(--color-secondary-shadow)] active:shadow-[0_1px_0_var(--color-secondary-shadow)]",
      },
      fullWidth: {
        true: "w-full",
        false: "",
      },
    },
    defaultVariants: {
      variant: "primary",
      fullWidth: false,
    },
  },
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, fullWidth, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return (
      <Comp
        className={cn(buttonVariants({ variant, fullWidth, className }))}
        ref={ref}
        {...props}
      />
    );
  },
);
Button.displayName = "Button";

export { Button, buttonVariants };
