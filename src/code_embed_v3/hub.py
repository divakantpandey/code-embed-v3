"""Hugging Face Hub as the durable copy of checkpoints and built data, so a lost pod costs minutes."""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Literal

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import RepositoryNotFoundError

logger = logging.getLogger(__name__)

RepoType = Literal["model", "dataset"]


def remote_has(repo_id: str, subdir: str, repo_type: RepoType = "model") -> bool:
    try:
        files = HfApi().list_repo_files(repo_id, repo_type=repo_type)
    except RepositoryNotFoundError:
        return False
    return any(f.startswith(f"{subdir}/") for f in files)


def fetch_folder(repo_id: str, subdir: str, dest: str | Path, repo_type: RepoType = "model") -> bool:
    """Download ``repo_id/subdir`` to ``dest``; False when the repo or folder does not exist yet.

    Downloads into a sibling temp dir and renames, so an interrupted download never looks complete.
    """
    dest = Path(dest)
    if not remote_has(repo_id, subdir, repo_type):
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent, prefix=f".{dest.name}.") as tmp:
        snapshot_download(repo_id, repo_type=repo_type, allow_patterns=[f"{subdir}/*"], local_dir=tmp)
        if dest.exists():
            shutil.rmtree(dest)
        (Path(tmp) / subdir).rename(dest)
    logger.info("fetched %s/%s -> %s", repo_id, subdir, dest)
    return True


def push_folder(
    folder: str | Path,
    repo_id: str,
    path_in_repo: str | None,
    repo_type: RepoType = "model",
    message: str = "",
) -> None:
    """Upload ``folder`` to ``repo_id/path_in_repo`` (repo root when None), creating a private repo."""
    api = HfApi()
    api.create_repo(repo_id, repo_type=repo_type, private=True, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(folder),
        path_in_repo=path_in_repo,
        repo_type=repo_type,
        commit_message=message or f"upload {path_in_repo or folder}",
    )
    logger.info("pushed %s -> %s/%s", folder, repo_id, path_in_repo or "")


def push_file(
    path: str | Path,
    repo_id: str,
    message: str = "",
    path_in_repo: str | None = None,
    repo_type: RepoType = "model",
) -> None:
    api = HfApi()
    api.create_repo(repo_id, repo_type=repo_type, private=True, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=path_in_repo or Path(path).name,
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=message or f"upload {Path(path).name}",
    )
