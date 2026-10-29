from __future__ import annotations

import os
from pathlib import Path

import pytest

from code_embed_v3.env import available_cpus, load_dotenv


def test_dotenv_fills_gaps_without_overriding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("CE3_A", "CE3_B", "CE3_C", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("CE3_B", "from-shell")
    env = tmp_path / ".env"
    env.write_text(
        "# comment\n"
        "CE3_A='quoted value'\n"
        "CE3_B=from-file\n"
        'export CE3_C="x=y"\n'
        "not a pair\n"
        "CUDA_VISIBLE_DEVICES=0\n"
    )
    assert load_dotenv(env) == ["CE3_A", "CE3_C"]

    assert os.environ["CE3_A"] == "quoted value"
    assert os.environ["CE3_B"] == "from-shell"
    assert os.environ["CE3_C"] == "x=y"
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_missing_dotenv_is_fine(tmp_path: Path) -> None:
    assert load_dotenv(tmp_path / "absent") == []


def test_available_cpus_is_positive() -> None:
    assert available_cpus() >= 1
