# End-to-end dry run, 27 and 28 September 2026

| Task | Result | Waits (s) |
|---|---|---|
| T1 typed, 3 hints | Worked, verified path (no badge) | 51, 4, 15 |
| T4 parent PIN | Worked (wrong PIN message, then unlock) | |
| T3 own question | Worked, weak mode badge | 11 |
| T2 photo | Text read correctly 3 times out of 3 tries (one try failed while the server was restarting), but no hint was returned any time | read: 3 |

Cause (server_supervisor.log): the backend process died with exit code 3221225477 (0xC0000005, a
Windows access violation) three times, each time when the confirm step loaded the language model
after the vision model. The existing guard allowed one swap per process, on the evidence that a
first swap was safe; this dry run showed a first VLM -> LLM swap also crashes. The frontend then
sent the student back to the start, losing the photo.

Fix:
1. Backend: the swap guard now refuses every in-process swap, and a successful photo read restarts
   the process at once, so the confirm step loads the language model into a fresh process.
2. Frontend: while the server restarts (503 or no connection), the app keeps waiting and retries for
   up to 150 s instead of failing; wait-screen time-outs raised to 180 s.
3. Tests: the swap tests updated to the new rule; the restart is stubbed inside pytest.
To confirm: re-run the tests and T2.

## Re-run of T2 after the restart fix

The planned restart worked (supervisor log: `uvicorn exited: code=1`, then relaunch; no access
violation). Photo read, confirm, then hints 1 to 3 all returned. Waits: Hint 1 45 s (includes the
restart and loading the language model), Hint 2 3 s, Hint 3 10 s.

Two new findings from the hint text itself:

1. Answer leak at Tier 3 (severity: critical for the project's core rule). Hint 3 solved the
   student's own question: "1.25 litres/bottle * 8 bottles = 10 litres". The Tier 3 prompt tells the
   model to present a different, related example and never to solve the student's question, but the
   small model ignored it. A deterministic leak checker already existed, but only in the offline
   evaluation scripts. Fix: the checker moved into the app (app/pipeline/answer_leak.py) and now
   guards every hint at the one point all hints pass through before they are saved and shown
   (_finalize_and_persist_hint). A leaking hint is regenerated once with extra steering; if it still
   leaks, it is replaced with a fixed, safe prompt for its tier. The answers checked are the stored
   answer key plus any answer the model produced during the gate. 6 new unit tests.
2. Misleading Tier 2 hint (severity: moderate). "Think about how many times the amount in one bottle
   fits into the total amount you want" points towards division when the question needs
   multiplication. This is hint quality, the known weak point (blind-judge Tier 2 score 3.06 of 5).
   Not fixable overnight; recorded as a limitation and watched for in the user test.

## Re-run after the leak guard

Both Tier 3 hints now show a different, related example (juice question: packs of markers; Mei Ling:
Sarah with $60) instead of the student's own answer. New finding: both hints stopped mid-sentence
("Step 3: Multiply the number of"), because generation hit its token limit, and the steps ran together
in one block. Fix: Tier 3 token limit 180 to 240; every hint is tidied before it is shown (an unfinished
last sentence is dropped, each "Step N:" starts a new line); the hint card keeps those line breaks.
2 more unit tests.

The full test run crashed inside test_matching_and_weak_mode.py with the known Windows access
violation when a second test loads the language model in the same pytest process (the same test and
trace as the earlier coverage run). The README now runs each GPU test file in its own process.

## Re-run after tidy fix (tests: 396 passed, 19 skipped; GPU files one process each)

- Juice, Hint 3: the guard fired ("answer-leak guard replaced a tier 3 hint (question_id=3)") and the
  child saw the safe fallback prompt. The leak was stopped, but the fallback helps less than a good
  worked example would.
- Mei Ling, typed a second time: this time the gate capped it at Tier 1 (yellow badge), although the
  first dry run gave the full ladder. The demo database has no precomputed gate results, so the gate
  runs live each time and its two sampled solutions can disagree. This is the gate failing closed as
  designed, but it makes the user test inconsistent between children. Fix for the test: precompute
  the gate once for the 12 demo questions (the same caching the real corpus uses), then choose task
  questions whose stored result allows Tiers 2 and 3.

## Gate precomputed for the 12 demo questions (131 s, one pass each)

| id | topic | type | tier_allowed |
|---|---|---|---|
| 1 | Whole numbers | reasoning | 4 |
| 2 | Fractions (Mei Ling, task T1) | reasoning | 4 |
| 3 | Decimals (juice, task T2) | sum | 4 |
| 4 | Percentage | reasoning | 4 |
| 5 | Ratio | reasoning | 4 |
| 6 | Rate | sum | 2 |
| 7 | Average | reasoning | 4 |
| 8 | Algebra | sum | 2 |
| 9 | Area and perimeter | sum | 1 |
| 10 | Volume | reasoning | 4 |
| 11 | Circles | sum | 4 |
| 12 | Angles | sum | 4 |

9 of 12 allow the full ladder (Tiers 2 and 3), 2 allow Tier 2 only, 1 is capped at Tier 1 even though
it is a simple, correct question: the gate fails closed when it cannot verify its own solution. Both
user-test questions (ids 2 and 3) allow the full ladder, so every child now gets the same, cached gate
decision and a fast first hint.
