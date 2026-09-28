"""
FastAPI application entry point.

`init_db()` runs at startup so `uvicorn app.main:app` works from a fresh checkout without a
separate migration step. It is the same idempotent call the extraction pipeline makes in
`write_extracted_data_to_db()` (Issue 102).

Run from `backend/` using its virtual environment (Issue 102):
    backend/.venv/Scripts/python.exe -m uvicorn app.main:app --reload
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.auth_routes import router as auth_router
from app.api.escalation_routes import router as escalation_router
from app.api.routes import router, _get_or_create_demo_student
from app.api.submission_routes import router as submission_router
from app.db.session import get_session, init_db


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    init_db()
    # Seed the demo student once at startup rather than get-or-create inside each request
    # (Issue 168). A concurrency test showed the per-request version races on a fresh database:
    # two concurrent first requests can both see no demo student and both insert one. Each then
    # holds SQLite's single write lock until its final commit, after the LLM call, and the
    # slower one can exceed the busy timeout. Seeding before any request arrives removes the race.
    with get_session() as session:
        _get_or_create_demo_student(session)
    yield


app = FastAPI(title="PSLE AI Homework Helper", lifespan=lifespan)
# The Next.js development frontend (http://localhost:3000) runs on a different origin from this
# API (http://localhost:8000 by default), so the browser blocks requests without a CORS allowance.
# This allowlist is for development only; deployment origins are added when there is a deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router, prefix="/api")
app.include_router(submission_router, prefix="/api")
app.include_router(auth_router, prefix="/api")
app.include_router(escalation_router, prefix="/api")
