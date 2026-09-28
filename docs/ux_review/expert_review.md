# Expert review

Method: heuristic evaluation against Nielsen's 10 usability heuristics, with Nielsen's 0 to 4 severity
scale (0 not a problem, 1 cosmetic, 2 minor, 3 major, 4 catastrophe), plus a cognitive walkthrough of
the four user-test tasks. Material: 14 screenshots of the /preview screens
(before/). This is an expert review, not user testing.

| # | Screen | Finding | Heuristic | Severity | Fix | Status |
|---|---|---|---|---|---|---|
| 1 | Hint, Tier 3 (preview sample) | The sample showed the final answer (3/20) to the student's own question. The real Tier 3 gives a related worked example, so the sample contradicted the design. | 2 Match with real world; project rule | 3 | Sample replaced with a related example (a different cake question) | Fixed |
| 2 | Capped (weak mode, nothing to unlock) | No action at all when there is no stored answer: a dead end | 3 User control and freedom | 3 | Added "Ask a different question" | Fixed |
| 3 | Hint | No way to leave a question and start again | 3 User control and freedom | 2 | Added "Ask a different question" | Fixed |
| 4 | Confirm | No way back to retake a bad photo | 3 User control and freedom | 2 | Added "Take another photo" | Fixed |
| 5 | Confirm | Typo: "fix anything that'" (missing s) | 4 Consistency and standards | 1 | Corrected | Fixed |
| 6 | Confirm | The text box being editable is only signalled by a small pencil label | 6 Recognition rather than recall | 2 | Label now "What we read (tap to fix it)" | Fixed |
| 7 | Hint | "Tier 1 of 3" is system jargon for a 12-year-old | 2 Match with real world | 2 | Now "Hint 1 of 3" (screen text and screen-reader label) | Fixed |
| 8 | Wait screens | Staged messages exist, but no expected duration for a 15 to 60 s wait | 1 Visibility of system status | 2 | Added "This usually takes up to a minute" | Fixed |
| 9 | Hint, capped (phone width) | Long parent button label wraps, icon separates from text | 8 Aesthetic and minimalist design | 1 | Label shortened to "Ask a parent for the answer" | Fixed |
| 10 | Parent PIN | Disabled Unlock button has low contrast | 4 Consistency; accessibility | 1 | Not changed: disabled state is intentional and the PIN field is the focus | Accepted |
| 11 | Parent PIN | No route for a forgotten PIN | 9 Recover from errors | 2 | Needs parent account recovery, out of scope | Future work |

Verified after the fixes: ESLint clean; `next build` passes.
