"""Loads the Anthropic API key used by the API-route evaluation (Bucket 2).

The rest of the backend reads configuration with `os.environ.get(...)` and never loads `.env`,
so a key written there would not be visible to it. This module reads `.env` for this one key
only:

- It does not call `load_dotenv()`, so no other `.env` variable is loaded, and the key is never
  copied into `os.environ` (callers pass it straight to the client).
- A non-empty `ANTHROPIC_API_KEY` environment variable takes precedence; otherwise the
  repo-root `.env` is read (git-ignored, see `.env.example`).
- The key is never logged, printed or included in an error message.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import dotenv_values

# backend/app/services/anthropic_config.py -> services -> app -> backend -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ENV_FILE = _REPO_ROOT / ".env"
_KEY_NAME = "ANTHROPIC_API_KEY"


def get_anthropic_api_key(env_file: Optional[Path] = None) -> Optional[str]:
    """Returns the Anthropic API key, or None if it is not configured.

    A blank value (`ANTHROPIC_API_KEY=` left empty, as `.env` ships) counts as not configured.

    Args:
        env_file: Path to a `.env` file, so tests can use a temporary file. Defaults to the
            repo-root `.env`.
    """
    from_env = os.environ.get(_KEY_NAME)
    if from_env and from_env.strip():
        return from_env.strip()
    path = env_file if env_file is not None else DEFAULT_ENV_FILE
    if not path.is_file():
        return None
    from_file = dotenv_values(path).get(_KEY_NAME)
    if from_file and from_file.strip():
        return from_file.strip()
    return None
