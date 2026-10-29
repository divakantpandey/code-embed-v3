from __future__ import annotations

from pathlib import Path

import pytest

from code_embed_v3.paths import REPO, resolve


def test_private_file_overrides_public_example(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CE3_PRIVATE", str(tmp_path))
    assert resolve("configs/data/mix_v1.yaml") == REPO / "configs/data/mix_v1.yaml"
    (tmp_path / "configs/data").mkdir(parents=True)
    (tmp_path / "configs/data/mix_v1.yaml").write_text("x")
    assert resolve("configs/data/mix_v1.yaml") == tmp_path / "configs/data/mix_v1.yaml"
    assert resolve(tmp_path / "abs.yaml") == tmp_path / "abs.yaml"
