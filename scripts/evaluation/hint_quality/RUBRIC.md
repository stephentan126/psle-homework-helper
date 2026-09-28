# Hint-quality rubric (Issue 357)

Applies to the hint text the REAL `gate.generate_hint()` returns (adapter active for hint writing, per Issue 353). Scored per hint,
reported per tier (Tier 1 = the restate-and-stop template; Tier 2 = the grounded, worked-solution-based hint), never only as an average.

| # | Dimension | Scored by | Scale |
|---|---|---|---|
| 1 | **No answer leak** | Deterministic code, `leak_check.py` (validated by `backend/tests/test_hint_leak_check.py`). **Not** judged by the API model. | Hard pass/fail per hint: `CLEAN` / `AMBIGUOUS` (answer numeral also occurs in the question; human-reviewed, not counted as a leak) / `LEAK`. `NO_NUMERIC_ANSWER` rows are excluded from the denominator. |
| 2 | **Socratic scaffolding, not a restatement** | Offline API judge | 1-5 |
| 3 | **Age-appropriate (Singapore Primary 6, ~12 y/o)** | Offline API judge | 1-5 |
| 4 | **Consistent with the verified solution** | Offline API judge, given the real `worked_solution_text` | 1-5 |
| 5 | **Length and clarity (a hint, not a solution)** | Offline API judge | 1-5 |

Anchors given to the judge (identical for every row):

- **2 Scaffolding.** 1 = only restates the question, or is generic enough to fit any problem. 3 = points at a topic or step but gives no
  guiding question or method. 5 = names the right method/first step for THIS problem and/or asks a guiding question, without doing it.
- **3 Age-appropriate.** 1 = technical/textbook-formal or confusing for a 12-year-old. 3 = understandable but stiff. 5 = plain, friendly,
  suitable wording and terms taught in Singapore primary maths.
- **4 Consistency.** 1 = contradicts the worked solution or sends the student down a wrong path. 3 = not wrong but not aligned with the
  worked method. 5 = fully consistent with the worked solution's method and quantities. (A restate-only Tier 1 hint that makes no method
  claim is scored on whether it misstates the question, and the judge must say so in its rationale.)
- **5 Length/clarity.** 1 = empty/garbled/rambling or a full worked solution. 3 = usable but too thin or too long. 5 = right size for one hint.

The judge sees the question, the real `worked_solution_text`, and the hint. It is NOT shown the leak-check result, the tier label, or which
adapter wrote the hint. Its output is strict JSON with an integer score and a one-sentence rationale per dimension.
