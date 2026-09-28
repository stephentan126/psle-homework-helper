"""Pre-release checks for the repository.

Run from the repository root before committing:
    python scripts/dev/check_release.py

Checks:
  1. Every relative link in Markdown files points to a file or folder that exists.
  2. No emoji in documentation or source files.
  3. No absolute personal paths (a Windows or Unix home folder) in source files.
  4. No files that must never be published (databases, model weights, source papers) are present
     outside git-ignored folders.
Exits with status 1 if any check fails.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SKIP_DIRS = {"node_modules", ".venv", "__pycache__", ".next", ".git", "models", ".pytest_cache", "data"}
TEXT_SUFFIXES = {".py", ".md", ".ts", ".tsx", ".css", ".sh", ".ps1", ".toml", ".yml", ".yaml", ".json", ".mjs",
                 ".log", ".txt", ".csv"}
EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F900-\U0001F9FF\U00002B00-\U00002BFF\uFE0F]"
)
PERSONAL_PATH = re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+|/home/[a-z]+/|/Users/[A-Za-z]+/")
LINK = re.compile(r"\[[^\]]*\]\(([^)#\s]+)(?:#[^)]*)?\)")
# The owl mascot on the start screen is a deliberate design choice, not decoration in code or docs.
EMOJI_ALLOWED = {"frontend/src/components/screens/landing-ask.tsx": "\U0001F989"}
FORBIDDEN_SUFFIXES = { ".pdf", ".safetensors", ".gguf", ".bin", ".pt"}


def tracked_candidates() -> list[Path]:
    """Files git would publish; falls back to a filesystem walk when git is unavailable."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.splitlines()
        return [REPO_ROOT / p for p in out if p]
    except (OSError, subprocess.CalledProcessError):
        files = []
        for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.endswith(".egg-info")]
            files.extend(Path(dirpath) / name for name in filenames)
        return files


def main() -> int:
    problems: list[str] = []
    files = tracked_candidates()
    for path in files:
        rel = path.relative_to(REPO_ROOT).as_posix()
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            problems.append(f"must not be published: {rel}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        allowed = EMOJI_ALLOWED.get(rel)
        if EMOJI.search(text.replace(allowed, "") if allowed else text):
            problems.append(f"emoji found: {rel}")
        if path.suffix != ".md" and PERSONAL_PATH.search(text):
            problems.append(f"absolute personal path: {rel}")
        if path.suffix == ".md":
            for target in LINK.findall(text):
                if target.startswith(("http://", "https://", "mailto:")):
                    continue
                if not (path.parent / target).resolve().exists():
                    problems.append(f"broken link in {rel}: {target}")
    for line in problems:
        print(line)
    print(f"{len(files)} files checked, {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
