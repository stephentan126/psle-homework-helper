"""
Parent PIN verification with rate limiting and lockout (design specification, Section 10.2;
Issue 244).

The PIN is checked per question: every full-solution unlock request verifies the PIN against the
stored hash again (Issue 244). Neither this module nor its caller (`app/api/auth_routes.py`) stores
an "unlocked" result anywhere (database, memory or session) that a later request could use
instead of calling `verify_pin_attempt()`. A correct PIN for one `question_id` authorises only
that response.

Hashing uses `argon2-cffi`, as Section 10.2 requires the PIN to be hashed with bcrypt or argon2
and never stored in plain text. `ParentAccount.pin_hash` holds argon2's encoded hash string, which
includes a fresh random salt and the hashing parameters. `ParentAccount.pin_salt` is therefore
unused and left as None.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

from app.models.parent_account import ParentAccount

_hasher = PasswordHasher()

# Security control against PIN brute-forcing (Issue 244). Section 10.2 specifies the threshold of 5
# failed attempts. The 15-minute cooldown is chosen here. A short numeric PIN is realistically 4
# digits (10,000 values). At 5 guesses per 15-minute lockout, an attacker averages one guess every
# 3 minutes, so covering the keyspace takes about 2,000 lockout cycles (about 20.8 days) of
# continuous guessing, with the account visibly locked throughout, which a parent would notice.
# The cooldown is short enough that a parent who mistypes a few times is not locked out for long.
LOCKOUT_THRESHOLD = 5
LOCKOUT_COOLDOWN_MINUTES = 15


def hash_pin(pin: str) -> str:
    """Returns a one-way argon2 hash of the PIN for storage.

    A fresh random salt is used on every call. Do not use this to compare PINs; use
    `verify_pin_attempt()`, which also applies the lockout.
    """
    return _hasher.hash(pin)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


@dataclass(frozen=True)
class PinVerificationResult:
    outcome: Literal["unlocked", "wrong_pin", "locked_out"]
    # ISO-8601 UTC timestamp string, set only when outcome is "locked_out". Stored as a string to
    # match the project's timestamp convention (Question.verified_at, ParentAccount.locked_until).
    locked_until: Optional[str] = None


def verify_pin_attempt(parent: ParentAccount, submitted_pin: str) -> PinVerificationResult:
    """
    Verifies a submitted PIN against `parent.pin_hash` and updates the lockout state.

    Each call runs argon2 verification again; no earlier result is cached. Lockout fields are
    updated on the `parent` row, and the caller is responsible for committing (this function never
    touches a session).

    The lockout check runs first, before argon2. A locked account is rejected immediately and
    `failed_attempts` is not incremented, so guessing against a locked account cannot extend the
    lockout.
    """
    if parent.locked_until is not None:
        locked_until_dt = _parse_iso(parent.locked_until)
        if _now() < locked_until_dt:
            return PinVerificationResult(outcome="locked_out", locked_until=parent.locked_until)
        # The lockout has expired: clear it so the next attempt starts a fresh failure count.
        parent.locked_until = None
        parent.failed_attempts = 0

    try:
        _hasher.verify(parent.pin_hash, submitted_pin)
        verified = True
    except VerifyMismatchError:
        verified = False
    except Exception:
        # Fail closed: any other exception (a corrupted stored hash, an internal argon2 error)
        # counts as a failed verification, never a success, as with the safety classifier
        # (Issue 210).
        verified = False

    if verified:
        parent.failed_attempts = 0
        parent.locked_until = None
        return PinVerificationResult(outcome="unlocked")

    parent.failed_attempts += 1
    if parent.failed_attempts >= LOCKOUT_THRESHOLD:
        locked_until_dt = _now() + timedelta(minutes=LOCKOUT_COOLDOWN_MINUTES)
        parent.locked_until = locked_until_dt.isoformat()
        return PinVerificationResult(outcome="locked_out", locked_until=parent.locked_until)

    return PinVerificationResult(outcome="wrong_pin")
