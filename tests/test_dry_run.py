from __future__ import annotations

from pathlib import Path

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
