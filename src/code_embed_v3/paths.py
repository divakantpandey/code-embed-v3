"""Public/private config overlay.

The public repo ships example configs. A private checkout at ``private/`` (or ``$CE3_PRIVATE``) holds
the real data mix, tuned hyperparameters and generation prompts under the same relative paths, and
wins wherever it has a file. Nothing in ``private/`` is ever committed to the public repo.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def private_root() -> Path:
    return Path(os.environ.get("CE3_PRIVATE", REPO / "private"))


def resolve(rel: str | Path) -> Path:
    """``private/<rel>`` when it exists, else ``<repo>/<rel>``. Absolute paths pass through."""
    rel = Path(rel)
    if rel.is_absolute():
        return rel
    private = private_root() / rel
    return private if private.exists() else REPO / rel
