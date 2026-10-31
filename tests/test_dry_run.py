from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from code_embed_v3.dry_run import main
from code_embed_v3.modeling import resolve_attn


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the laptop guard")
def test_dry_run_refuses_without_cuda(tmp_path: Path) -> None:
    root = tmp_path / "dryrun"
    root.mkdir()
    (root / "keep").write_text("x")
    with pytest.raises(SystemExit, match="needs a CUDA GPU"):
        main(root, hub=False, wandb=False, rows=10, teacher="t")
    assert (root / "keep").exists()


@pytest.mark.skipif(torch.cuda.is_available(), reason="CPU fallback only")
def test_auto_attention_falls_back_to_sdpa_without_cuda() -> None:
    assert resolve_attn("auto") == "sdpa"
    assert resolve_attn("eager") == "eager"


def test_dry_run_never_wipes_a_directory_not_named_dryrun(tmp_path: Path) -> None:
    (tmp_path / "keep").write_text("x")
    with pytest.raises(SystemExit, match="named dryrun"):
        main(tmp_path, hub=False, wandb=False, rows=10, teacher="t")
    assert (tmp_path / "keep").exists()


def test_dry_run_refuses_stale_hub_repos(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError

    from code_embed_v3.dry_run import DryRun, DryRunError

    hub = {"u/ce3-dryrun-synth": [".gitattributes"], "u/ce3-dryrun-ft": [".gitattributes", "final/x.json"]}

    def files(_: HfApi, repo: str, repo_type: str = "model") -> list[str]:
        if repo not in hub:
            raise RepositoryNotFoundError(repo, response=MagicMock())
        return hub[repo]

    monkeypatch.setattr(HfApi, "list_repo_files", files)
    d = DryRun(root=tmp_path / "dryrun", hub_user="u", wandb=False, rows=10, teacher="t", gpus=1)
    with pytest.raises(DryRunError, match=r"\['u/ce3-dryrun-ft'\]"):
        d.require_fresh_hub_repos()
    hub["u/ce3-dryrun-ft"] = [".gitattributes"]
    d.require_fresh_hub_repos()


def test_hub_has_treats_missing_repo_as_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError

    from code_embed_v3.dry_run import HUB_CHECKPOINT, DryRun

    def files(_: HfApi, repo: str) -> list[str]:
        if repo == "u/missing":
            raise RepositoryNotFoundError(repo, response=MagicMock())
        return [HUB_CHECKPOINT] if repo == "u/done" else ["model.safetensors"]

    monkeypatch.setattr(HfApi, "list_repo_files", files)
    assert DryRun.hub_has("u/done", HUB_CHECKPOINT)
    assert not DryRun.hub_has("u/model-only", HUB_CHECKPOINT)
    assert not DryRun.hub_has("u/missing", HUB_CHECKPOINT)
