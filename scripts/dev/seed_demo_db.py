"""Build a small demo database so the app can be run without the source exam papers.

The demo questions in demo_questions.json are original questions written in PSLE style for this
project. Every answer was checked with SymPy. The script creates demo.db at the repository root, inserts the
questions, then embeds them so typed or photographed questions can be matched.

Usage (from the repository root):
    python scripts/dev/seed_demo_db.py
Then start the server with DATABASE_URL pointing at demo.db (see README.md).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_DB = REPO_ROOT / "demo.db"
QUESTIONS_FILE = Path(__file__).resolve().parent / "demo_questions.json"
DEMO_SOURCE = "demo_questions.json"

# DATABASE_URL must be set before the app's database module is imported.
os.environ["DATABASE_URL"] = f"sqlite:///{DEMO_DB.as_posix()}"
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.db.session import SessionLocal, init_db  # noqa: E402
from app.models.question import Question  # noqa: E402


def seed() -> int:
    init_db()
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    added = 0
    with SessionLocal() as session:
        for q in questions:
            exists = (
                session.query(Question)
                .filter(Question.source_paper == DEMO_SOURCE, Question.question_number == q["number"])
                .first()
            )
            if exists:
                continue
            session.add(
                Question(
                    source_paper=DEMO_SOURCE,
                    source_page_index=0,
                    source_page_label="demo",
                    answer_source_file=DEMO_SOURCE,
                    answer_source_page_index=0,
                    question_number=q["number"],
                    question_text=q["question"],
                    answer_value=q["answer"],
                    worked_solution_text=q["working"],
                    has_diagram=False,
                    ocr_confidence="high",
                    school_name="Demo",
                    school_name_verified=True,
                    verification_status="human_confirmed",
                )
            )
            added += 1
        session.commit()
    return added


def embed() -> None:
    """Embed the demo questions with the same model the app uses for matching."""
    script = REPO_ROOT / "scripts" / "extraction" / "precompute_question_embeddings.py"
    subprocess.run([sys.executable, str(script), "--all"], check=True, env=os.environ.copy(), cwd=REPO_ROOT)


if __name__ == "__main__":
    count = seed()
    print(f"Demo database: {DEMO_DB} ({count} questions added)")
    embed()
    print("Done. Start the server with DATABASE_URL set to this file (see README.md).")
