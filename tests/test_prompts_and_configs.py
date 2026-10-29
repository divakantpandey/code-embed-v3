from __future__ import annotations

from pathlib import Path

import mteb
import pytest
from mteb.models.abs_encoder import AbsEncoder

from code_embed_v3.config import DataMixConfig, ModelConfig, StageConfig
from code_embed_v3.data.sources import LOADERS
from code_embed_v3.eval.run_mteb import DEFAULT_BENCHMARKS
from code_embed_v3.paths import private_root
from code_embed_v3.prompts import MTEB_TASK_FAMILY, QUERY_PROMPTS, mteb_model_prompts, training_prompts

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
# The private overlay must satisfy the same schemas as the public examples.
ROOTS = [CONFIGS, *([p] if (p := private_root() / "configs").is_dir() else [])]


def _all(sub: str) -> list[Path]:
    return sorted(f for root in ROOTS for f in (root / sub).glob("*.yaml"))


def test_every_reported_task_has_a_family() -> None:
    tasks = {t.metadata.name for b in DEFAULT_BENCHMARKS for t in mteb.get_benchmark(b).tasks}
    assert tasks - set(MTEB_TASK_FAMILY) == set()


def test_mteb_prompt_keys_are_valid() -> None:
    _, invalid = AbsEncoder.validate_task_to_prompt_name(mteb_model_prompts(), raise_for_invalid_keys=False)
    assert not invalid


def test_training_prompts_shape() -> None:
    assert training_prompts({"a": "text2sql"}) == {"a": {"anchor": QUERY_PROMPTS["text2sql"]}}


@pytest.mark.parametrize("path", _all("models"), ids=lambda p: p.name)
def test_model_configs_load(path: Path) -> None:
    cfg = ModelConfig.from_yaml(path)
    assert cfg.matryoshka_dims == sorted(cfg.matryoshka_dims, reverse=True)


@pytest.mark.parametrize("path", _all("data"), ids=lambda p: p.name)
def test_data_configs_load(path: Path) -> None:
    cfg = DataMixConfig.from_yaml(path)
    names = [s.name for s in cfg.sources]
    assert len(names) == len(set(names))
    for s in cfg.sources:
        assert s.loader in LOADERS
        assert set(s.overlaps_eval) <= set(MTEB_TASK_FAMILY)


@pytest.mark.parametrize("path", _all("stages"), ids=lambda p: p.name)
def test_stage_configs_load(path: Path) -> None:
    stage = StageConfig.from_yaml(path)
    assert (CONFIGS.parent / stage.model).exists()
    if stage.loss == "distill_kl":
        assert stage.student_temperature is not None


def test_unknown_config_key_is_rejected(tmp_path: Path) -> None:
    p = tmp_path / "m.yaml"
    p.write_text("name: x\nbackbone: y\npoolng: mean\n")
    with pytest.raises(ValueError, match="poolng"):
        ModelConfig.from_yaml(p)


@pytest.mark.parametrize("path", [r / "synth" / "prompts.yaml" for r in ROOTS], ids=str)
def test_synth_prompts_load(path: Path) -> None:
    from code_embed_v3.data.synth import PARSERS, load_prompts

    assert set(load_prompts(path).tasks) == set(PARSERS)
