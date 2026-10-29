"""A lost pod must cost minutes: resume locally, fall back to the Hub, never trust a half-written save."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import load_file

from code_embed_v3 import hub
from code_embed_v3.config import StageConfig
from code_embed_v3.modeling import META_FILE
from code_embed_v3.train import (
    latest_complete_checkpoint,
    per_device_batch_size,
    plan_stage,
    run_stage,
    stage_run_id,
)

from .conftest import write_mix
from .test_train_smoke import _model_yaml, _stage


def _weights(model_dir: Path) -> dict[str, torch.Tensor]:
    (path,) = model_dir.rglob("model.safetensors")
    return load_file(str(path))


def _resumable_stage(tmp: Path, backbone: Path, **kw: object) -> StageConfig:
    tmp.mkdir(parents=True, exist_ok=True)
    return _stage(
        tmp, _model_yaml(tmp, backbone), write_mix(tmp / "data"), **{"max_steps": 4, "save_steps": 2, **kw}
    )


def test_resumed_run_matches_uninterrupted_run(tmp_path: Path, tiny_modernbert: Path) -> None:
    stage = _resumable_stage(tmp_path, tiny_modernbert)
    out = Path(stage.output_dir)
    uninterrupted = _weights(run_stage(stage, device="cpu"))

    # Simulate a pod lost after the step-2 save: the step-4 checkpoint and the final model never happened.
    shutil.rmtree(out / "final")
    shutil.rmtree(out / "checkpoint-4")
    plan = plan_stage(stage)
    assert plan.done is None and plan.resume_from == out / "checkpoint-2"

    resumed = _weights(run_stage(stage, device="cpu", resume_from=plan.resume_from))
    assert json.loads((out / "checkpoint-4" / "trainer_state.json").read_text())["global_step"] == 4
    assert uninterrupted.keys() == resumed.keys()
    for name, tensor in uninterrupted.items():
        torch.testing.assert_close(resumed[name], tensor, msg=name)


def test_finished_stage_is_skipped(tmp_path: Path, tiny_modernbert: Path) -> None:
    stage = _resumable_stage(tmp_path, tiny_modernbert, max_steps=2)
    final = run_stage(stage, device="cpu")
    assert plan_stage(stage).done == final


def test_half_written_checkpoint_is_ignored(tmp_path: Path) -> None:
    for step, complete in ((2, True), (4, False)):
        ckpt = tmp_path / f"checkpoint-{step}"
        ckpt.mkdir()
        if complete:
            (ckpt / "trainer_state.json").write_text("{}")
    assert latest_complete_checkpoint(tmp_path) == tmp_path / "checkpoint-2"


def test_fresh_pod_resumes_from_hub(
    tmp_path: Path, tiny_modernbert: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trained = _resumable_stage(tmp_path / "old-pod", tiny_modernbert)
    run_stage(trained, device="cpu")
    hub_copy = Path(trained.output_dir) / "checkpoint-2"

    def fake_fetch(repo_id: str, subdir: str, dest: str | Path, repo_type: str = "model") -> bool:
        assert repo_id == "me/stage"
        if subdir != "last-checkpoint":
            return False
        shutil.copytree(hub_copy, dest)
        return True

    monkeypatch.setattr(hub, "fetch_folder", fake_fetch)
    fresh = _resumable_stage(tmp_path / "new-pod", tiny_modernbert, hub_repo="me/stage")
    plan = plan_stage(fresh)
    assert plan.resume_from == Path(fresh.output_dir) / "checkpoint-2"
    assert (plan.resume_from / META_FILE).is_file()


def test_run_id_is_stable_and_config_sensitive(tmp_path: Path, tiny_modernbert: Path) -> None:
    a = _resumable_stage(tmp_path, tiny_modernbert)
    assert stage_run_id(a) == stage_run_id(StageConfig(**{**a.__dict__}))
    assert stage_run_id(a) != stage_run_id(StageConfig(**{**a.__dict__, "learning_rate": 1.0}))


def test_batch_size_is_global(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tiny_modernbert: Path) -> None:
    stage = _resumable_stage(tmp_path, tiny_modernbert, batch_size=16384)
    monkeypatch.setenv("WORLD_SIZE", "8")
    assert per_device_batch_size(stage) == 2048
    monkeypatch.setenv("WORLD_SIZE", "3")
    with pytest.raises(ValueError, match="divisible"):
        per_device_batch_size(stage)


def test_two_process_ddp_splits_the_global_batch(tmp_path: Path, tiny_modernbert: Path) -> None:
    stage = _resumable_stage(tmp_path, tiny_modernbert, max_steps=2, batch_size=8)
    cfg = tmp_path / "ddp.yaml"
    cfg.write_text(yaml.safe_dump(stage.__dict__))
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}
    # Loopback rendezvous: macOS hostnames often fail torchrun's reverse lookup.
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc-per-node=2", "--master-addr=127.0.0.1"]
    cmd += ["--master-port=29517", "-m", "code_embed_v3.cli", "train-worker", str(cfg), "--device", "cpu"]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-4000:]
    state = json.loads((Path(stage.output_dir) / "checkpoint-2" / "trainer_state.json").read_text())
    assert state["global_step"] == 2 and state["train_batch_size"] == 4
    assert (Path(stage.output_dir) / "final" / "modules.json").is_file()
