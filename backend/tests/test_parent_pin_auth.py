"""
End-to-end tests for the parent-PIN unlock endpoint via `TestClient` (Issue 244).

The `TestClient(app)` lifespan seeds the demo `Student` and `ParentAccount` rows with an argon2
hash of `routes._DEMO_PARENT_PIN` ("0000"). Every test uses that documented demo PIN rather than
a mock, so `parent_auth.py` is exercised through the endpoint and not only in isolation.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import routes as routes_module
from app.db.session import get_session
from app.main import app
from app.models.parent_account import ParentAccount
from app.models.question import Question
from app.pipeline.parent_auth import LOCKOUT_THRESHOLD, hash_pin

_CORRECT_PIN = routes_module._DEMO_PARENT_PIN  # "0000", the documented demo PIN
_WRONG_PIN = "9999"


def _make_question(**overrides) -> int:
    defaults = dict(
        source_paper="data/School papers/2024/P6_Maths_2024_WA2_nanhua.pdf",
        source_page_index=5, source_page_label=None,
        answer_source_file="data/School papers/2024/P6_Maths_2024_WA2_nanhua.pdf",
        answer_source_page_index=20, question_number="19",
        question_text=r"Find the value of \(\frac{3}{4} \div 12\).",
        answer_value="1/16", has_diagram=False, worked_solution_text=None,
    )
    defaults.update(overrides)
    with get_session() as session:
        question = Question(**defaults)
        session.add(question)
        session.flush()
        return question.id


def _get_demo_parent() -> ParentAccount:
    with get_session() as session:
        student = routes_module._get_or_create_demo_student(session)
        parent = student.parent_account
        session.expunge(parent)
        return parent


def test_correct_pin_unlocks_the_specific_question_requested():
    with TestClient(app) as client:
        qid = _make_question(question_number="19")
        response = client.post(f"/api/questions/{qid}/unlock", json={"pin": _CORRECT_PIN})
    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "unlocked_solution"
    assert body["question_id"] == qid
    assert body["answer_value"] == "1/16"


def test_correct_pin_does_not_leak_a_different_questions_answer():
    """Each unlock response carries the requested question's answer, never another question's.

    The check is on the response body, not only on the call succeeding.
    """
    with TestClient(app) as client:
        qid_a = _make_question(question_number="19", answer_value="1/16")
        qid_b = _make_question(question_number="20", answer_value="42")
        response_a = client.post(f"/api/questions/{qid_a}/unlock", json={"pin": _CORRECT_PIN})
        response_b = client.post(f"/api/questions/{qid_b}/unlock", json={"pin": _CORRECT_PIN})
    assert response_a.json()["question_id"] == qid_a
    assert response_a.json()["answer_value"] == "1/16"
    assert response_b.json()["question_id"] == qid_b
    assert response_b.json()["answer_value"] == "42"


def test_wrong_pin_is_rejected():
    with TestClient(app) as client:
        qid = _make_question()
        response = client.post(f"/api/questions/{qid}/unlock", json={"pin": _WRONG_PIN})
    assert response.status_code == 401
    assert "Incorrect PIN" in response.json()["detail"]


def test_a_previously_unlocked_question_does_not_stay_unlocked_for_a_different_question():
    """PIN scope is per question (Issue 244).

    After a correct unlock of question A, a wrong-PIN request for question B must fail. Any
    cached "this parent is unlocked" state would wrongly return 200 instead of 401.
    """
    with TestClient(app) as client:
        qid_a = _make_question(question_number="19")
        qid_b = _make_question(question_number="20")
        unlock_a = client.post(f"/api/questions/{qid_a}/unlock", json={"pin": _CORRECT_PIN})
        assert unlock_a.status_code == 200
        wrong_b = client.post(f"/api/questions/{qid_b}/unlock", json={"pin": _WRONG_PIN})
    assert wrong_b.status_code == 401


def test_every_call_independently_re_verifies_no_caching_bypass():
    """Every unlock re-verifies the PIN against the stored hash.

    After one successful unlock, the stored `pin_hash` is changed to a hash of a different PIN
    without touching in-process state. The same PIN must then fail for the same question, which
    shows argon2 verification ran again rather than reusing the first result.
    """
    with TestClient(app) as client:
        qid = _make_question()
        first = client.post(f"/api/questions/{qid}/unlock", json={"pin": _CORRECT_PIN})
        assert first.status_code == 200

        with get_session() as session:
            student = routes_module._get_or_create_demo_student(session)
            student.parent_account.pin_hash = hash_pin("5678")
            student.parent_account.failed_attempts = 0
            student.parent_account.locked_until = None

        second = client.post(f"/api/questions/{qid}/unlock", json={"pin": _CORRECT_PIN})
    assert second.status_code == 401


def test_repeated_wrong_pins_trigger_lockout():
    with TestClient(app) as client:
        qid = _make_question()
        for i in range(LOCKOUT_THRESHOLD):
            response = client.post(f"/api/questions/{qid}/unlock", json={"pin": _WRONG_PIN})
            if i < LOCKOUT_THRESHOLD - 1:
                assert response.status_code == 401, f"attempt {i + 1} should be a plain wrong-PIN rejection"
            else:
                assert response.status_code == 429, "the 5th real failed attempt should trigger lockout"
                assert "locked" in response.json()["detail"].lower()

        # Even the correct PIN is rejected while locked out.
        still_locked = client.post(f"/api/questions/{qid}/unlock", json={"pin": _CORRECT_PIN})
    assert still_locked.status_code == 429


def test_lockout_does_not_extend_from_further_attempts_while_already_locked():
    """Guessing against a locked account does not push the lockout window further out.

    Once locked, `failed_attempts` must not increase past the threshold.
    """
    with TestClient(app) as client:
        qid = _make_question()
        for _ in range(LOCKOUT_THRESHOLD):
            client.post(f"/api/questions/{qid}/unlock", json={"pin": _WRONG_PIN})
        parent_after_lockout = _get_demo_parent()
        assert parent_after_lockout.failed_attempts == LOCKOUT_THRESHOLD

        client.post(f"/api/questions/{qid}/unlock", json={"pin": _WRONG_PIN})
        parent_after_extra_attempt = _get_demo_parent()
    assert parent_after_extra_attempt.failed_attempts == LOCKOUT_THRESHOLD
    assert parent_after_extra_attempt.locked_until == parent_after_lockout.locked_until


def test_a_correct_unlock_resets_the_failed_attempts_counter():
    """A successful unlock resets the failure count.

    Three wrong attempts, one correct unlock, then two more wrong attempts must not lock the
    account. Without the reset, 3 + 2 = 5 would reach the threshold.
    """
    with TestClient(app) as client:
        qid = _make_question()
        for _ in range(3):
            client.post(f"/api/questions/{qid}/unlock", json={"pin": _WRONG_PIN})
        success = client.post(f"/api/questions/{qid}/unlock", json={"pin": _CORRECT_PIN})
        assert success.status_code == 200

        for _ in range(2):
            response = client.post(f"/api/questions/{qid}/unlock", json={"pin": _WRONG_PIN})
    assert response.status_code == 401, "should still be a plain rejection, not a lockout"


def test_malformed_pin_shapes_are_rejected_before_any_real_verification():
    with TestClient(app) as client:
        qid = _make_question()
        too_short = client.post(f"/api/questions/{qid}/unlock", json={"pin": "12"})
        non_digit = client.post(f"/api/questions/{qid}/unlock", json={"pin": "abcd"})
    assert too_short.status_code == 422
    assert non_digit.status_code == 422


def test_unlock_returns_404_for_a_nonexistent_question():
    with TestClient(app) as client:
        response = client.post("/api/questions/999999999/unlock", json={"pin": _CORRECT_PIN})
    assert response.status_code == 404


def test_correct_pin_on_a_question_with_nothing_at_all_returns_a_real_error_not_a_confusing_unlock():
    """A question with no answer_value and no worked_solution_text returns 422 on unlock.

    Such questions occur in the corpus, for example weak-mode questions or those with no matching
    answer key. An empty `UnlockedSolutionResponse` would look like a successful unlock with
    nothing in it, so a clear error is returned instead.
    """
    with TestClient(app) as client:
        qid = _make_question(answer_value=None, worked_solution_text=None)
        response = client.post(f"/api/questions/{qid}/unlock", json={"pin": _CORRECT_PIN})
    assert response.status_code == 422
    assert "no real answer_value or worked_solution_text" in response.json()["detail"]


def test_correct_pin_on_a_worked_solution_only_question_still_unlocks_successfully():
    """A question with a worked solution but no answer_value still unlocks.

    The `UnlockedSolutionResponse` docstring anticipates this case, and 276 corpus rows have this
    shape. Copying the narrower `answer_value`-only check from `request_hint()` would have
    blocked a useful worked solution for every one of them.
    """
    with TestClient(app) as client:
        qid = _make_question(
            answer_value=None,
            worked_solution_text="Step 1: ... Step 2: ... Final answer: 42",
        )
        response = client.post(f"/api/questions/{qid}/unlock", json={"pin": _CORRECT_PIN})
    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "unlocked_solution"
    assert body["answer_value"] is None
    assert body["worked_solution_text"] == "Step 1: ... Step 2: ... Final answer: 42"


def test_pin_is_never_stored_in_plaintext():
    """The persisted PIN is an argon2 hash, checked on the stored value itself."""
    parent = _get_demo_parent()
    assert parent.pin_hash != routes_module._DEMO_PARENT_PIN
    assert routes_module._DEMO_PARENT_PIN not in parent.pin_hash
    assert parent.pin_hash.startswith("$argon2")


def test_hash_pin_produces_a_real_verifiable_argon2_hash_with_a_fresh_salt_each_call():
    from app.pipeline.parent_auth import hash_pin as _hash_pin
    h1 = _hash_pin("1234")
    h2 = _hash_pin("1234")
    assert h1 != h2, "argon2 must embed a fresh random salt on every real call, even for the same PIN"
    assert h1.startswith("$argon2")
