"use client";

/**
 * Main student flow: a client-side state machine that chooses screens from backend
 * responses (AnyHomeworkResponse). The dev preview switcher (page.tsx) is kept separate
 * for jumping directly to individual edge-case screens.
 *
 * Flow: Landing (photo or typed) -> submit -> [confirm] -> hint/capped/safety-blocked/
 * extraction-failed -> [escalate] -> [parent-unlock].
 */
import { useCallback, useState } from "react";
import {
  confirmSubmission,
  editSubmission,
  escalateHint,
  submitPhoto,
  submitPhotoQuestion,
  unlockSolution,
  ApiError,
} from "@/lib/api-client";
import type { AnyHomeworkResponse } from "@/lib/api-types";
import { LandingAskScreen } from "@/components/screens/landing-ask";
import { WorkingOnIt, type WaitKind } from "@/components/screens/working-on-it";
import { ConfirmEditScreen } from "@/components/screens/confirm-edit";
import { HintScreen } from "@/components/screens/hint";
import { CappedTier1Screen } from "@/components/screens/capped-tier1";
import { ExtractionFailedScreen } from "@/components/screens/extraction-failed";
import { SafetyBlockedScreen } from "@/components/screens/safety-blocked";
import {
  ParentUnlockScreen,
  type UnlockErrorKind,
} from "@/components/screens/parent-unlock";

/** Screens this flow can show. Not a mirror of AnyHomeworkResponse['state'], since
 * "waiting" and "parent unlock" are client-side states with no backend equivalent. */
type FlowScreen =
  | { kind: "landing" }
  | { kind: "waiting"; waitKind: WaitKind; failed: boolean }
  | {
      kind: "response";
      response: AnyHomeworkResponse;
      recheckPending?: boolean;
    }
  | { kind: "parent-unlock"; questionId: number; returnTo: AnyHomeworkResponse };

function newRequestId(): string {
  return crypto.randomUUID();
}

/** How long to keep retrying while the server restarts between models. */
const RESTART_RETRY_WINDOW_MS = 150_000;

export function AppFlow() {
  const [screen, setScreen] = useState<FlowScreen>({ kind: "landing" });
  const [unlockError, setUnlockError] = useState<UnlockErrorKind | null>(null);
  const [unlockedResult, setUnlockedResult] = useState<AnyHomeworkResponse | null>(
    null,
  );
  const [isSubmittingPin, setIsSubmittingPin] = useState(false);
  const [lockedUntilLabel, setLockedUntilLabel] = useState<string | null>(null);
  // HintResponse and CappedTier1Response carry only hint_text, not the original question,
  // so the "Your question" card on those screens is held as client-side state.
  const [questionText, setQuestionText] = useState<string>("");
  // Whether the current question came from a photo or was typed, so the confirm screen's
  // wording matches what the student did.
  const [inputSource, setInputSource] = useState<"photo" | "typed">("photo");

  /** Runs a request behind the Working-on-it screen and routes the result or failure to
   * the next screen. Shared by the submit, confirm, hint and escalate calls. */
  const runRequest = useCallback(
    async (waitKind: WaitKind, run: () => Promise<AnyHomeworkResponse>) => {
      setScreen({ kind: "waiting", waitKind, failed: false });
      const startedAt = Date.now();
      for (;;) {
        try {
          const response = await run();
          setScreen({ kind: "response", response });
          return;
        } catch (err) {
          // The server restarts itself instead of swapping models on the GPU (see Issue
          // 366). While it restarts, requests get a 503 or cannot connect, so keep waiting
          // and retry for up to RESTART_RETRY_WINDOW_MS before showing the failure state.
          const restarting =
            err instanceof ApiError &&
            (err.kind === "service_restarting" || err.kind === "network");
          if (restarting && Date.now() - startedAt < RESTART_RETRY_WINDOW_MS) {
            const waitSeconds = Math.min(err.retryAfterSeconds ?? 5, 10);
            await new Promise((resolve) => setTimeout(resolve, waitSeconds * 1000));
            continue;
          }
          // Other failures show Working-on-it's `failed` state. Deliberate 4xx responses
          // (e.g. 422 nothing-to-unlock) are handled at their call sites.
          setScreen({ kind: "waiting", waitKind, failed: true });
          return;
        }
      }
    },
    [],
  );

  const handleTakePhoto = useCallback((file: File) => {
    setInputSource("photo");
    void runRequest("photo", () => submitPhoto(file));
  }, [runRequest]);

  const handleSubmitTyped = useCallback(
    (text: string) => {
      setQuestionText(text);
      setInputSource("typed");
      void runRequest("typed", () => submitPhotoQuestion({ question_text: text }));
    },
    [runRequest],
  );

  const handleConfirm = useCallback(
    (studentSubmissionId: number, confirmedText: string) => {
      setQuestionText(confirmedText);
      void runRequest("hint", () =>
        confirmSubmission(studentSubmissionId, newRequestId()),
      );
    },
    [runRequest],
  );

  const handleEdit = useCallback(
    async (studentSubmissionId: number, correctedText: string) => {
      if (screen.kind !== "response") return;
      setScreen({ ...screen, recheckPending: true });
      try {
        const response = await editSubmission(studentSubmissionId, {
          corrected_text: correctedText,
        });
        setScreen({ kind: "response", response });
      } catch {
        // A failed re-check on /edit does not show the failure screen, since the student is
        // mid-edit. They can keep typing or press confirm, which surfaces any server outage.
        setScreen((prev) =>
          prev.kind === "response" ? { ...prev, recheckPending: false } : prev,
        );
      }
    },
    [screen],
  );

  const handleEscalate = useCallback(
    (attemptId: number) => {
      void runRequest("hint", () => escalateHint(attemptId, newRequestId()));
    },
    [runRequest],
  );

  const handleOpenParentUnlock = useCallback(
    (questionId: number) => {
      if (screen.kind !== "response") return;
      setUnlockError(null);
      setUnlockedResult(null);
      setLockedUntilLabel(null);
      setScreen({ kind: "parent-unlock", questionId, returnTo: screen.response });
    },
    [screen],
  );

  const handleSubmitPin = useCallback(
    async (questionId: number, pin: string) => {
      setIsSubmittingPin(true);
      setUnlockError(null);
      try {
        const response = await unlockSolution(questionId, { pin });
        setUnlockedResult(response);
      } catch (err) {
        if (err instanceof ApiError) {
          if (err.kind === "unauthorized") {
            setUnlockError("wrong_pin");
          } else if (err.kind === "rate_limited") {
            setUnlockError("locked_out");
            // The 429 detail string contains the locked_until timestamp. It is shown as-is
            // rather than reparsed, since the label slot accepts free text.
            setLockedUntilLabel(err.detail ?? null);
          } else if (err.kind === "unprocessable") {
            setUnlockError("nothing_to_unlock");
          } else {
            setUnlockError("wrong_pin");
          }
        } else {
          setUnlockError("wrong_pin");
        }
      } finally {
        setIsSubmittingPin(false);
      }
    },
    [],
  );

  if (screen.kind === "landing") {
    return (
      <LandingAskScreen
        onTakePhoto={() => {
          // A plain file input stands in for a dedicated camera UI, so the submitPhoto()
          // path is exercised end-to-end and not only the typed path.
          const input = document.createElement("input");
          input.type = "file";
          input.accept = "image/*";
          input.capture = "environment";
          input.onchange = () => {
            const file = input.files?.[0];
            if (file) handleTakePhoto(file);
          };
          input.click();
        }}
        onSubmitTyped={handleSubmitTyped}
      />
    );
  }

  if (screen.kind === "waiting") {
    return (
      <WorkingOnIt
        key={`${screen.waitKind}-${screen.failed}`}
        kind={screen.waitKind}
        requestId={newRequestId()}
        failed={screen.failed}
        onRetry={() => setScreen({ kind: "landing" })}
      />
    );
  }

  if (screen.kind === "parent-unlock") {
    if (unlockedResult && unlockedResult.state === "unlocked_solution") {
      return (
        <ParentUnlockScreen
          onSubmit={() => {}}
          // "Done" after an unlock ends this question (the answer is shown, so nothing is
          // left to escalate) and returns to Landing rather than the Tier 3 or capped screen.
          onClose={() => setScreen({ kind: "landing" })}
          unlocked={{
            answerValue: unlockedResult.answer_value,
            workedSolutionText: unlockedResult.worked_solution_text,
          }}
        />
      );
    }
    return (
      <ParentUnlockScreen
        onSubmit={(pin) => void handleSubmitPin(screen.questionId, pin)}
        onClose={() => setScreen({ kind: "response", response: screen.returnTo })}
        isSubmitting={isSubmittingPin}
        error={unlockError}
        lockedUntilLabel={lockedUntilLabel}
      />
    );
  }

  // screen.kind === "response"
  const { response } = screen;

  switch (response.state) {
    case "needs_confirmation":
      return (
        <ConfirmEditScreen
          extractedText={response.extracted_text}
          matchedQuestionText={response.matched_question_text}
          source={inputSource}
          isRechecking={screen.recheckPending ?? false}
          onTextChange={(text) => {
            if (
              response.student_submission_id !== null &&
              text !== response.extracted_text
            ) {
              void handleEdit(response.student_submission_id, text);
            }
          }}
          onRetake={() => setScreen({ kind: "landing" })}
          onConfirm={() => {
            if (response.student_submission_id !== null) {
              handleConfirm(response.student_submission_id, response.extracted_text);
            }
          }}
        />
      );

    case "hint":
      return (
        <HintScreen
          questionText={questionText}
          tier={response.tier}
          hintText={response.hint_text}
          canEscalateFurther={response.can_escalate_further}
          onEscalate={() => {
            if (response.attempt_id !== null) handleEscalate(response.attempt_id);
          }}
          onParentUnlock={() => handleOpenParentUnlock(response.question_id)}
          onNewQuestion={() => setScreen({ kind: "landing" })}
        />
      );

    case "capped_tier1":
      // Weak-mode responses carry student_submission_id instead of question_id, and
      // /questions/{id}/unlock cannot accept that id, so there is nothing to unlock. Pass
      // null rather than a button that does nothing (see Issue 180).
      return (
        <CappedTier1Screen
          questionText={questionText}
          hintText={response.hint_text}
          onParentUnlock={
            response.question_id !== null
              ? () => handleOpenParentUnlock(response.question_id as number)
              : null
          }
          onNewQuestion={() => setScreen({ kind: "landing" })}
        />
      );

    case "safety_blocked":
      return (
        <SafetyBlockedScreen
          message={response.message}
          isSelfHarmPath={response.is_self_harm_path}
          resources={response.resources}
          onBackToStart={() => setScreen({ kind: "landing" })}
        />
      );

    case "extraction_failed":
      return (
        <ExtractionFailedScreen
          reason={response.reason}
          onRetakePhoto={() => setScreen({ kind: "landing" })}
          onTypeInstead={() => setScreen({ kind: "landing" })}
        />
      );

    case "unlocked_solution":
      // Defensive: unlock normally goes through the "parent-unlock" screen kind above,
      // but this state is still rendered if reached directly.
      return (
        <ParentUnlockScreen
          onSubmit={() => {}}
          onClose={() => setScreen({ kind: "landing" })}
          unlocked={{
            answerValue: response.answer_value,
            workedSolutionText: response.worked_solution_text,
          }}
        />
      );

    default: {
      const _exhaustive: never = response;
      return _exhaustive;
    }
  }
}
