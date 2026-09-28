"""
Keyword lexicon for passive self-harm ideation, a secondary signal for the child safety layer
(design specification, Section 7).

Held-out validation showed that ShieldGemma-2B has a structural blind spot: passive or indirect
self-harm phrasing can score lower than safe PSLE content, so no threshold on the classifier
alone separates them (Issue 201). This lexicon is an independent second signal, OR-combined with
the classifier. It does not replace the classifier, which still catches direct phrasing and the
non-self-harm categories this lexicon does not cover.

Every pattern is drawn from a cited clinical or research source (Issue 202):

- C-SSRS (Columbia-Suicide Severity Rating Scale, Baseline/Screening Version,
  cssrs.columbia.edu), Category 1 "Wish to be Dead": "thoughts about a wish to be dead or not
  alive anymore, or wish to fall asleep and not wake up."
- PHQ-9 item 9 (Kroenke, Spitzer & Williams 2001), the suicide and self-harm item of a widely
  validated depression screen: "Thoughts that you would be better off dead or of hurting
  yourself in some way."
- Joiner's Interpersonal Theory of Suicide (Joiner, 2005), the "Perceived Burdensomeness"
  construct: the belief that "others would be better off without them." This is the source for
  the phrasing that defeated the classifier in held-out validation ("everyone would be better off
  without me", Issue 201).

To avoid false positives, patterns require specific word combinations, because several of these
constructs use everyday words that also appear in homework talk or PSLE word problems (the same
problem as the single-letter unit exclusion in `question_matching.py`'s `_UNIT_TERMS`):
  - "give up" is common homework frustration ("I give up on this question"), so the pattern
    requires "life", "living" or "everything".
  - "better off" is PSLE percentage and discount vocabulary ("which deal leaves you better
    off?"), so the pattern requires "dead" or "without me/him/her/us/them".
  - "wake up" appears in PSLE time problems ("Sam wakes up at 6:30am"), so the pattern requires a
    negation ("not", "never", "wouldn't", "don't").
  - "cut" and "cutting" are common PSLE vocabulary (rope, fabric, cake), so the pattern requires
    "myself".

Validated (Issue 202) against `gold_set.json`, `held_out_set.json` and a third set not used in
designing either the classifier threshold or this lexicon (`fresh_validation_set.json`). The
combined false-positive and false-negative rates are measured by
`evaluation/model_selection/safety_classifier/run_combined_signal_validation.py`.

Limitation: the lexicon covers a finite set of cited phrasing patterns and cannot catalogue every
way a student might express self-harm risk. Risk expressed in a form not represented here will
not be caught by this signal; the classifier is the other half of the OR. See Issues 202 and 203
in docs/DEVELOPMENT_LOG.md for what the combined approach does and does not solve.
"""
from __future__ import annotations

import re

# Each category's source is noted in the comment above it. Patterns are matched against a
# lowercased copy of the text and require specific word combinations, not single trigger words.

# C-SSRS Category 1, "Wish to be Dead" descriptor.
_WISH_TO_BE_DEAD_PATTERNS: list[re.Pattern] = [
    re.compile(r"wish(?:ed)?\s+(?:i|I)\s+(?:was|were)\s+dead"),
    re.compile(r"wish(?:ed)?\s+(?:i|I)\s+(?:wasn'?t|weren'?t)\s+alive"),
    re.compile(r"wish(?:ed)?\s+(?:i|I)\s+(?:was|were)\s+never\s+born"),
    # "wish/hope I could fall asleep and not wake up", "wish I wouldn't wake up" and similar. The
    # negation is required so PSLE time phrasing ("Sam wakes up at 6:30am") cannot match.
    re.compile(r"(?:wish|hope)(?:d|ed)?\s+(?:i|I)\s+(?:.{0,25}?)"
               r"(?:not|never|wouldn'?t|don'?t)\s+wake\s+up"),
    re.compile(r"to\s+be\s+here\s+anymore"),
]

# PHQ-9 item 9's "better off dead" clause and Joiner's Perceived Burdensomeness ("without me"),
# the source of the phrasing that defeated the classifier alone (Issue 201). A bare "better off"
# (PSLE discount vocabulary) never matches; "dead" or "without <pronoun>" must follow it.
_BURDENSOMENESS_PATTERNS: list[re.Pattern] = [
    re.compile(r"better\s+off\s+dead"),
    re.compile(r"better\s+off\s+without\s+(?:me|him|her|us|them)"),
    re.compile(r"\bi'?m\s+a\s+burden\b"),
    re.compile(r"\bi\s+am\s+a\s+burden\b"),
    re.compile(r"everyone\s+would\s+be\s+(?:so\s+much\s+)?(?:happier|better\s+off)"),
]

# Hopelessness about living, as distinct from "give up on this question". "life", "living" or
# "everything" must follow "give up".
_GIVING_UP_ON_LIFE_PATTERNS: list[re.Pattern] = [
    re.compile(r"giv(?:e|ing)\s+up\s+on\s+(?:everything|life|living)"),
    re.compile(r"(?:don'?t|do\s+not)\s+want\s+to\s+live\s+anymore"),
    re.compile(r"stopped\s+caring\s+(?:whether|if)\s+i\s+wake\s+up"),
]

# PHQ-9 item 9's second clause ("...or of hurting yourself in some way"), direct self-injury
# disclosure. "myself" must follow the verb, so bare "cut", "cutting" or "hurt" (cutting rope or
# cake, a sports injury) never matches.
_SELF_INJURY_DISCLOSURE_PATTERNS: list[re.Pattern] = [
    re.compile(r"(?:cutting|cut|hurting|hurt|harming|harm|scratch(?:ing)?)\s+my(?:self)?\b"),
    re.compile(r"self[\s-]?harm"),
]

# Direct language about ending one's life. This overlaps with what the classifier already catches
# well; the overlap is intended as defence in depth.
_EXPLICIT_ENDING_LIFE_PATTERNS: list[re.Pattern] = [
    re.compile(r"end(?:ing)?\s+(?:it\s+all|my\s+life)"),
    re.compile(r"kill(?:ing)?\s+myself"),
]

_ALL_CATEGORIES: dict[str, list[re.Pattern]] = {
    "wish_to_be_dead": _WISH_TO_BE_DEAD_PATTERNS,
    "burdensomeness": _BURDENSOMENESS_PATTERNS,
    "giving_up_on_life": _GIVING_UP_ON_LIFE_PATTERNS,
    "self_injury_disclosure": _SELF_INJURY_DISCLOSURE_PATTERNS,
    "explicit_ending_life": _EXPLICIT_ENDING_LIFE_PATTERNS,
}


def check_self_harm_lexicon(text: str) -> str | None:
    """Returns the first lexicon category whose pattern matches `text`, or None.

    Matching is case-insensitive. The result is OR-combined with the safety classifier: a match
    flags possible self-harm whatever the classifier scores, following the instruction to bias
    towards over-flagging (Issue 202). A miss can cause serious harm, while a false positive only
    shows an unneeded crisis-resource message. That asymmetry is why this does not use the
    high-threshold approach of the secondary signals in `question_matching.py` (Issue 199(a)),
    which balance a more symmetric cost (app breakage against match quality).
    """
    lowered = text.lower()
    for category, patterns in _ALL_CATEGORIES.items():
        for pattern in patterns:
            if pattern.search(lowered):
                return category
    return None
