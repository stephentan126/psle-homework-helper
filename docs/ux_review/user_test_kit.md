# User test kit: PSLE Homework Helper (round 1 and round 2)

Participants: children aged 11 to 12 with a parent present. Round 1: 3 to 5 families. Round 2: 1 to 2
new families, after fixes. About 25 minutes per session. Ethics approval and consent forms: already in hand.

## Before the day (do the dry run tonight)

1. Build the demo database and start the backend (PowerShell, from the repository root, venv active):
   ```
   python scripts/dev/seed_demo_db.py
   $env:DATABASE_URL="sqlite:///C:/path/to/repo/demo.db"
   python scripts/server/run_server_supervised.py --port 8000
   ```
2. Start the frontend: `cd frontend`, `pnpm dev`, open http://localhost:3000 (not /preview).
3. Dry run every task below once yourself and note the real wait times. After a photo is read, the
   server restarts itself before the first hint (about 35 seconds), and the first photo after a server
   start takes about two and a half minutes. Tell participants "the app is thinking" during waits.
4. Print: the T2 question sheet, one record sheet per participant, the child and parent questionnaires.
5. Parent PIN for the demo account: 0000.

## Consent and conduct

- Parent signs the consent form; the child gives verbal assent in plain words ("You can stop any time,
  it is the app being tested, not you").
- Use IDs P1, P2, ... only. No names, no photos or video of the child. Do not keep the child's own
  question text if it contains anything personal.
- Do not help unless the child is stuck for 60 seconds or asks. Record every assist.
- Ask the child to think aloud: "Tell me what you are looking at and what you think will happen."

## Tasks

Order: run T2 (photo) first, then T1, T3, T4. After a photo is read, the server restarts itself
(about 40 seconds) so the language model loads into a fresh process; doing the photo first means
that restart overlaps with the child checking the text. Restart the backend (Ctrl+C, run it again)
between participants so each one starts from the same state.

T1 (typed question, full hint ladder). Say: "Type this question into the app and use it to help you
solve it. Get as many hints as you need."
  Question card: "Mei Ling spent 2/5 of her money on a bag and 1/4 of the remaining money on a book. She
  had $36 left. How much money did she have at first?" (answer 80, do not tell)
  Success: the child reaches an answer using at least one hint without the parent PIN.

T2 (photo). Say: "Now take a photo of this question. Check the app read it correctly before you go on."
  Printed sheet: "A bottle holds 1.25 litres of juice. How many litres of juice are there in 8 such
  bottles?"
  Success: the child checks the text, fixes any mistake, confirms, and gets a hint.
  Record: was the text read correctly; did the child notice and fix any error.

T3 (own question, weak mode). Say: "Type a maths question of your own, any question you like."
  Then ask: "What do you think the yellow label means?"
  Success: the child reads the hint and can explain the label in their own words.

T4 (parent). Say to the parent: "Your child wants the full answer to the first question. Please unlock
  it. First type 1234, then the real PIN 0000."
  Success: the parent recovers from the wrong PIN and unlocks the answer.

## Record sheet (one per participant)

| Task | Completed (yes / with help / no) | Time (s) | Hints used | Assists | Errors or confusion seen | Quote |
|---|---|---|---|---|---|---|
| T1 | | | | | | |
| T2 | | | | | | |
| T3 | | | | | | |
| T4 | | | | | | |

Also note: any wait that felt too long (child fidgets, asks "is it broken?"), any server restart.

## Child questionnaire (Smileyometer, 5 faces: Awful, Not very good, Good, Really good, Brilliant)

1. How easy was the app to use?
2. How helpful were the hints?
3. How did you feel while waiting for the app?
4. Would you like to use it for your homework?
5. What was the best thing about it?
6. What is one thing you would change?

## Parent questionnaire (System Usability Scale, 1 = strongly disagree, 5 = strongly agree)

1. I think that I would like to use this system frequently.
2. I found the system unnecessarily complex.
3. I thought the system was easy to use.
4. I think that I would need the support of a technical person to be able to use this system.
5. I found the various functions in this system were well integrated.
6. I thought there was too much inconsistency in this system.
7. I would imagine that most people would learn to use this system very quickly.
8. I found the system very cumbersome to use.
9. I felt very confident using the system.
10. I needed to learn a lot of things before I could get going with this system.

Plus: "Would you let your child use this for homework? Why or why not?" and "What worried you, if
anything?"

SUS score: for odd items subtract 1 from the answer; for even items subtract the answer from 5; add the
ten results and multiply by 2.5 (0 to 100).

## After round 1

The researcher collects the record sheets and questionnaires (typed or photographed, with IDs only), lists each
issue with its severity and a fix, applies the fixes, and runs round 2 with the same tasks.
