"""
Exploratory validation script for the layout-type fixtures (Issue 197).

Submits the handwritten-working (with a deliberate error) and diagram-present fixtures through the
live `/api/submissions/photo` endpoint and prints the full response bodies. This lets the model
behaviour be inspected before the pytest assertions are fixed, following the same approach as
`_validate_confidence_signal.py` (Issue 194).
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Same override as conftest.py (Issue 158). This script imports app.* outside pytest, so the
# conftest DATABASE_URL redirect never runs and a run would otherwise write to the gitignored
# psle.db. It must come before any app.* import, because app/db/session.py builds its engine at
# import time.
_TEST_DB_PATH = Path(__file__).resolve().parent / "_test_psle.db"
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DB_PATH.as_posix()}"

from fastapi.testclient import TestClient  # noqa: E402

from app.db.session import engine, init_db  # noqa: E402
from app.main import app  # noqa: E402
from app.models.base import Base  # noqa: E402

Base.metadata.drop_all(bind=engine)
init_db()

_FIXTURES = Path(__file__).resolve().parent / "_fixtures"

CASES = [
    ("HANDWRITTEN + deliberate error", _FIXTURES / "handwritten_error_phone_photo.jpg"),
    ("DIAGRAM PRESENT", _FIXTURES / "diagram_present_phone_photo.jpg"),
]


def main() -> None:
    with TestClient(app) as client:
        for label, path in CASES:
            assert path.exists(), f"Missing real fixture: {path}"
            with open(path, "rb") as f:
                response = client.post(
                    "/api/submissions/photo",
                    files={"file": (path.name, f, "image/jpeg")},
                )
            print(f"\n{'=' * 90}\n{label}\nstatus_code={response.status_code}")
            print(json.dumps(response.json(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
