"use client";

/**
 * App entry point, wired to the backend (see src/components/app-flow.tsx and
 * src/lib/api-client.ts) rather than sample data. The preview switcher (all 9 screens
 * reachable via buttons, no backend calls) lives at /preview as a separate dev route,
 * for jumping straight to a specific edge-case screen during design work.
 */
import { AppFlow } from "@/components/app-flow";

export default function Home() {
  return <AppFlow />;
}
