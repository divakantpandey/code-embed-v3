"""Optuna search end to end on CPU: proxy trials, a confirm rerun, and a full-length best stage."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from code_embed_v3.config import StageConfig
from code_embed_v3.tune import TuneConfig, run, trial_stage

from .conftest import write_mix
from .test_train_smoke import _model_yaml, _stage


def _tune_yaml(tmp: Path, base: StageConfig) -> Path:
    stage_yaml = tmp / "base.yaml"
    stage_yaml.write_text(yaml.safe_dump(base.__dict__))
    cfg = {
        "study": "t",
        "base_stage": str(stage_yaml),
        "samples_per_trial": 16,
        "evals_per_trial": 2,
        "n_trials": 3,
        "confirm_top": 1,
        "confirm_seeds": [7],
        "output_dir": str(tmp / "tune"),
        "params": {
            "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True},
            "batch_size": {"type": "categorical", "choices": [4, 8]},
        },
    }
    path = tmp / "tune.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_trials_see_equal_samples_and_never_touch_the_hub(tmp_path: Path, tiny_modernbert: Path) -> None:
    base = _stage(
        tmp_path, _model_yaml(tmp_path, tiny_modernbert), write_mix(tmp_path / "d"), hub_repo="me/x"
    )
    cfg = TuneConfig.from_yaml(_tune_yaml(tmp_path, base))
    small, large = (trial_stage(base, {"batch_size": b}, cfg, 0, 1) for b in (4, 8))
    assert small.max_steps * 4 == large.max_steps * 8 == 16
    assert small.hub_repo is None and small.save_steps > small.max_steps


def test_study_confirms_the_winner_and_writes_a_full_stage(tmp_path: Path, tiny_modernbert: Path) -> None:
    base = _stage(tmp_path, _model_yaml(tmp_path, tiny_modernbert), write_mix(tmp_path / "d"), max_steps=50)
    best_path = run(_tune_yaml(tmp_path, base), gpus=0, device="cpu")

    summary = json.loads((best_path.parent / "summary.json").read_text())
    assert summary["trials"]["COMPLETE"] == 4  # 3 searched + 1 confirm rerun
    assert len(summary["best"]["runs"]) == 2
    best = StageConfig.from_yaml(best_path)
    assert best.learning_rate == summary["best"]["params"]["learning_rate"]
    assert best.max_steps == 50  # the real run's length, not the proxy's
    assert not list((best_path.parent).glob("trial-*"))
