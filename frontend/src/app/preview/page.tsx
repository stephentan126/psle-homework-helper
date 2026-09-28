"use client";

import { useState } from "react";
import { WorkingOnIt, type WaitKind } from "@/components/screens/working-on-it";
import { HintScreen } from "@/components/screens/hint";
import { ConfirmEditScreen } from "@/components/screens/confirm-edit";
import { CappedTier1Screen } from "@/components/screens/capped-tier1";
import {
  ExtractionFailedScreen,
  type ExtractionFailedReason,
} from "@/components/screens/extraction-failed";
import { SafetyBlockedScreen } from "@/components/screens/safety-blocked";
import {
  ParentUnlockScreen,
  type UnlockErrorKind,
} from "@/components/screens/parent-unlock";
import { LandingAskScreen } from "@/components/screens/landing-ask";

type View =
  | "landing"
  | "confirm"
  | "wait"
  | "hint"
  | "capped"
  | "extraction-failed"
  | "safety-blocked"
  | "parent-unlock";

const SAMPLE_QUESTIONS = [
  "Ali had 3/5 of a box of marbles. He gave 1/4 of them to his sister. What fraction of the original box did he give away?",
];

export default function Home() {
  const [view, setView] = useState<View>("landing");
  const [kind, setKind] = useState<WaitKind>("hint");
  const [requestId] = useState(() => crypto.randomUUID());
  const [failed, setFailed] = useState(false);
  const [tier, setTier] = useState(1);
  const [withDiagram, setWithDiagram] = useState(false);
  const [extractedText, setExtractedText] = useState(
    "Ali had 3/5 of a box of marbels. He gave 1/4 of them to his sister. What fraction of the orignal box did he give away?",
  );
  const [isRechecking, setIsRechecking] = useState(false);
  const [extractionReason, setExtractionReason] =
    useState<ExtractionFailedReason>("unreadable_photo");
  const [isSelfHarmPath, setIsSelfHarmPath] = useState(false);
  const [unlockError, setUnlockError] = useState<UnlockErrorKind | null>(null);
  const [unlocked, setUnlocked] = useState(false);

  const hints: Record<number, string> = {
    1: "Start by working out how many marbles 1/4 of Ali's 3/5 actually is. What operation combines two fractions like that?",
    2: "Try multiplying the two fractions: 3/5 × 1/4. Multiply the numerators together, then the denominators together.",
    3: "Here is a similar question: Mei had 2/3 of a cake and ate 1/2 of it. What fraction of the whole cake did she eat? 2/3 × 1/2 = 2/6 = 1/3. Now try the same steps with Ali's fractions.",
  };

  return (
    <div className="flex flex-1 flex-col">
      <div className="flex flex-wrap items-center gap-3 border-b border-border px-4 py-3 text-sm">
        <span className="font-semibold text-text/50">Preview:</span>
        <button
          className="rounded-md border border-border px-3 py-1"
          onClick={() => setView("landing")}
        >
          Landing/Ask
        </button>
        <button
          className="rounded-md border border-border px-3 py-1"
          onClick={() => setView("confirm")}
        >
          Confirm/Edit
        </button>
        <button
          className="rounded-md border border-border px-3 py-1"
          onClick={() => setView("hint")}
        >
          Hint screen
        </button>
        <button
          className="rounded-md border border-border px-3 py-1"
          onClick={() => setView("capped")}
        >
          Capped/weak-mode
        </button>
        <button
          className="rounded-md border border-border px-3 py-1"
          onClick={() => setView("wait")}
        >
          Working-on-it
        </button>
        <button
          className="rounded-md border border-border px-3 py-1"
          onClick={() => setView("extraction-failed")}
        >
          Extraction-failed
        </button>
        <button
          className="rounded-md border border-border px-3 py-1"
          onClick={() => setView("safety-blocked")}
        >
          Safety-blocked
        </button>
        <button
          className="rounded-md border border-border px-3 py-1"
          onClick={() => {
            setView("parent-unlock");
            setUnlockError(null);
            setUnlocked(false);
          }}
        >
          Parent-unlock
        </button>
        {view === "confirm" && (
          <>
            <span className="mx-1 text-text/30">|</span>
            <button
              className="rounded-md border border-border px-3 py-1"
              onClick={() => setWithDiagram((v) => !v)}
            >
              Toggle diagram gap
            </button>
            <button
              className="rounded-md border border-border px-3 py-1"
              onClick={() => setIsRechecking((v) => !v)}
            >
              Toggle rechecking
            </button>
          </>
        )}
        {view === "hint" && (
          <>
            <span className="mx-1 text-text/30">|</span>
            <button
              className="rounded-md border border-border px-3 py-1"
              onClick={() => setTier((t) => Math.min(3, t + 1))}
            >
              Escalate tier
            </button>
            <button
              className="rounded-md border border-border px-3 py-1"
              onClick={() => setTier(1)}
            >
              Reset tier
            </button>
            <button
              className="rounded-md border border-border px-3 py-1"
              onClick={() => setWithDiagram((v) => !v)}
            >
              Toggle diagram gap
            </button>
          </>
        )}
        {view === "capped" && (
          <>
            <span className="mx-1 text-text/30">|</span>
            <button
              className="rounded-md border border-border px-3 py-1"
              onClick={() => setWithDiagram((v) => !v)}
            >
              Toggle diagram gap
            </button>
          </>
        )}
        {view === "extraction-failed" && (
          <>
            <span className="mx-1 text-text/30">|</span>
            {(
              [
                "unreadable_photo",
                "low_confidence",
                "photo_not_straight",
                "reader_unavailable",
              ] as ExtractionFailedReason[]
            ).map((r) => (
              <button
                key={r}
                className="rounded-md border border-border px-3 py-1"
                onClick={() => setExtractionReason(r)}
              >
                {r}
              </button>
            ))}
          </>
        )}
        {view === "safety-blocked" && (
          <>
            <span className="mx-1 text-text/30">|</span>
            <button
              className="rounded-md border border-border px-3 py-1"
              onClick={() => setIsSelfHarmPath((v) => !v)}
            >
              Toggle self-harm path
            </button>
          </>
        )}
        {view === "parent-unlock" && (
          <>
            <span className="mx-1 text-text/30">|</span>
            <button
              className="rounded-md border border-border px-3 py-1"
              onClick={() => {
                setUnlocked(false);
                setUnlockError("wrong_pin");
              }}
            >
              Wrong PIN
            </button>
            <button
              className="rounded-md border border-border px-3 py-1"
              onClick={() => {
                setUnlocked(false);
                setUnlockError("locked_out");
              }}
            >
              Locked out
            </button>
            <button
              className="rounded-md border border-border px-3 py-1"
              onClick={() => {
                setUnlocked(false);
                setUnlockError("nothing_to_unlock");
              }}
            >
              Nothing to unlock
            </button>
            <button
              className="rounded-md border border-border px-3 py-1"
              onClick={() => {
                setUnlocked(false);
                setUnlockError("verification_pending");
              }}
            >
              Verification pending
            </button>
            <button
              className="rounded-md border border-border px-3 py-1"
              onClick={() => {
                setUnlockError(null);
                setUnlocked(true);
              }}
            >
              Success
            </button>
          </>
        )}
        {view === "wait" && (
          <>
            <span className="mx-1 text-text/30">|</span>
            <button
              className="rounded-md border border-border px-3 py-1"
              onClick={() => {
                setKind("photo");
                setFailed(false);
              }}
            >
              Photo wait
            </button>
            <button
              className="rounded-md border border-border px-3 py-1"
              onClick={() => {
                setKind("hint");
                setFailed(false);
              }}
            >
              Hint wait
            </button>
            <button
              className="rounded-md border border-border px-3 py-1"
              onClick={() => setFailed(true)}
            >
              Force failure
            </button>
          </>
        )}
      </div>

      {view === "landing" ? (
        <LandingAskScreen
          onTakePhoto={() => setView("wait")}
          onSubmitTyped={(text) => {
            setExtractedText(text);
            setView("wait");
          }}
        />
      ) : view === "confirm" ? (
        <ConfirmEditScreen
          extractedText={extractedText}
          hasDiagram={withDiagram}
          matchedQuestionText="Ali had 3/5 of a box of marbles. He gave 1/4 of them to his sister. What fraction of the original box did he give away?"
          isRechecking={isRechecking}
          onTextChange={setExtractedText}
          onConfirm={() => setView("wait")}
          onRetake={() => setView("landing")}
        />
      ) : view === "hint" ? (
        <HintScreen
          questionText={SAMPLE_QUESTIONS[0]}
          tier={tier}
          hintText={hints[tier]}
          canEscalateFurther={tier < 3}
          hasDiagram={withDiagram}
          onEscalate={() => setTier((t) => Math.min(3, t + 1))}
          onParentUnlock={() => alert("Would navigate to Parent-unlock screen")}
          onNewQuestion={() => setView("landing")}
        />
      ) : view === "capped" ? (
        <CappedTier1Screen
          questionText={SAMPLE_QUESTIONS[0]}
          hintText="Try thinking about what fraction of the marbles Ali gave away, out of the whole box. It might help to draw the box split into fifths first."
          hasDiagram={withDiagram}
          onParentUnlock={() => alert("Would navigate to Parent-unlock screen")}
          onNewQuestion={() => setView("landing")}
        />
      ) : view === "extraction-failed" ? (
        <ExtractionFailedScreen
          reason={extractionReason}
          onRetakePhoto={() => alert("Would return to photo capture")}
          onTypeInstead={() => alert("Would switch to typed-question entry")}
        />
      ) : view === "safety-blocked" ? (
        <SafetyBlockedScreen
          message={
            isSelfHarmPath
              ? "It sounds like things feel really hard right now, and I want you to know you don't have to go through this alone. I'm not able to help with that here, but there are people who can, please reach out to Samaritans of Singapore, any time, day or night. You matter, and things can get better."
              : "That's not something I can help you with here. Let's get back to your homework, what question were you working on?"
          }
          isSelfHarmPath={isSelfHarmPath}
          resources={
            isSelfHarmPath
              ? [
                  "Samaritans of Singapore (SOS) 24-hour Hotline: 1767",
                  "SOS 24-hour CareText (WhatsApp): 9151 1767",
                ]
              : null
          }
          onBackToStart={() => setView("confirm")}
        />
      ) : view === "parent-unlock" ? (
        <ParentUnlockScreen
          onSubmit={() => setUnlockError("wrong_pin")}
          onClose={() => setView("hint")}
          error={unlockError}
          lockedUntilLabel={
            unlockError === "locked_out" ? "3:45pm" : null
          }
          unlocked={
            unlocked
              ? {
                  answerValue: "3/20",
                  workedSolutionText:
                    "3/5 × 1/4 = 3/20. Ali gave away 3/20 of the original box of marbles.",
                }
              : null
          }
        />
      ) : (
        <WorkingOnIt
          key={`${kind}-${requestId}-${failed}`}
          kind={kind}
          requestId={requestId}
          failed={failed}
          onRetry={() => setFailed(false)}
        />
      )}
    </div>
  );
}
