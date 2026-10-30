"""End-to-end: every stage type runs a couple of optimizer steps and produces a loadable model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from code_embed_v3.config import StageConfig
from code_embed_v3.modeling import META_FILE, load_model
from code_embed_v3.train import TrainingCollapsed, TrainingShortfall, WeightedLoss, per_source_loss, run_stage

from .conftest import HIDDEN, write_mix


def _model_yaml(tmp: Path, backbone: Path, bidirectional: bool = False) -> Path:
    p = tmp / "model.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "name": "tiny",
                "backbone": str(backbone),
                "max_seq_length": 32,
                "bidirectional": bidirectional,
                "matryoshka_dims": [HIDDEN, 16],
            }
        )
    )
    return p


def _stage(tmp: Path, model_yaml: Path, data: Path, **kw: object) -> StageConfig:
    base: dict[str, object] = {
        "name": "smoke",
        "model": str(model_yaml),
        "data_dir": str(data),
        "dev_suite": str(data),
        "output_dir": str(tmp / "out"),
        "loss": "cached_mnrl",
        "batch_size": 4,
        "mini_batch_size": 2,
        "dataloader_num_workers": 0,
        "max_steps": 2,
        "bf16": False,
        "report_to": [],
        "save_steps": 1,
        "eval_steps": 2,
        "logging_steps": 1,
    }
    base.update(kw)
    tmp.mkdir(parents=True, exist_ok=True)
    p = tmp / "stage.yaml"
    p.write_text(yaml.safe_dump(base))
    return StageConfig.from_yaml(p)


def test_contrastive_stage_with_bidirectional_decoder(tmp_path: Path, tiny_qwen3: Path) -> None:
    data = write_mix(tmp_path / "data", n_neg=1)
    stage = _stage(
        tmp_path,
        _model_yaml(tmp_path, tiny_qwen3, bidirectional=True),
        data,
        directions=["query_to_doc", "doc_to_query"],
    )
    final = run_stage(stage, device="cpu")
    assert json.loads((final / META_FILE).read_text())["bidirectional"] is True
    ckpts = sorted((tmp_path / "out").glob("checkpoint-*"))
    assert ckpts and all((c / META_FILE).exists() for c in ckpts)
    assert load_model(final, device="cpu").encode(["sort list"]).shape == (1, HIDDEN)


def test_distill_stage_from_previous_checkpoint(tmp_path: Path, tiny_modernbert: Path) -> None:
    model_yaml = _model_yaml(tmp_path, tiny_modernbert)
    s1 = run_stage(_stage(tmp_path / "s1", model_yaml, write_mix(tmp_path / "d1")), device="cpu")
    data = write_mix(tmp_path / "d2", n_neg=2, labels=True)
    stage = _stage(
        tmp_path / "s2",
        model_yaml,
        data,
        loss="distill_kl",
        init_from=str(s1),
        student_temperature=0.05,
        teacher_temperature=1.0,
        contrastive_weight=0.1,
    )
    final = run_stage(stage, device="cpu")
    assert load_model(final, device="cpu").encode(["read json"], truncate_dim=16).shape == (1, 16)


def test_unaudited_mix_is_refused(tmp_path: Path, tiny_modernbert: Path) -> None:
    data = write_mix(tmp_path / "data")
    manifest = json.loads((data / "manifest.json").read_text())
    manifest["sources"]["toy"]["audited"] = False
    (data / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="ce3 audit"):
        run_stage(_stage(tmp_path, _model_yaml(tmp_path, tiny_modernbert), data), device="cpu")


def test_collapse_guard_stops_and_refuses_to_save(tmp_path: Path, tiny_modernbert: Path) -> None:
    model_yaml = _model_yaml(tmp_path, tiny_modernbert)
    data = write_mix(tmp_path / "data")
    # A ratio above any reachable score trips at the first evaluation.
    stage = _stage(tmp_path / "tripped", model_yaml, data, collapse_guard=100.0, max_steps=4, eval_steps=1)
    with pytest.raises(TrainingCollapsed, match="step 1"):
        run_stage(stage, device="cpu")
    assert not (tmp_path / "tripped" / "out" / "final").exists()

    final = run_stage(_stage(tmp_path / "ok", model_yaml, data, collapse_guard=0.0), device="cpu")
    assert (final / "modules.json").exists()


def test_source_loss_weight_scales_cached_gradients(tiny_modernbert: Path) -> None:
    from sentence_transformers.sentence_transformer.losses import CachedMultipleNegativesRankingLoss

    model = load_model(tiny_modernbert, device="cpu")
    anchors, positives = (
        ["sort a list", "read a file", "parse json"],
        ["sorted(x)", "open(p).read()", "json.loads(s)"],
    )

    def grads(weight: float) -> torch.Tensor:
        model.zero_grad()
        inner = CachedMultipleNegativesRankingLoss(model, mini_batch_size=1)
        feats = [model.tokenize(anchors), model.tokenize(positives)]
        WeightedLoss(inner, weight)(feats, None).backward()
        return torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])

    full, half = grads(1.0), grads(0.5)
    assert full.abs().sum() > 0
    torch.testing.assert_close(half, 0.5 * full, rtol=1e-4, atol=1e-7)


def test_per_source_loss_rejects_unknown_sources() -> None:
    loss = torch.nn.Identity()
    assert per_source_loss(loss, ["a", "b"], {}) is loss
    mapped = per_source_loss(loss, ["a", "b"], {"b": 0.25})
    assert isinstance(mapped, dict) and mapped["a"] is loss
    assert isinstance(mapped["b"], WeightedLoss) and mapped["b"].weight == 0.25
    with pytest.raises(ValueError, match="unknown sources"):
        per_source_loss(loss, ["a"], {"typo": 0.5})


def test_sampler_starvation_fails_the_stage(tmp_path: Path, tiny_modernbert: Path) -> None:
    # Negatives are other rows' positives (as with mined data), so no 8-row batch is free of duplicates.
    data = write_mix(tmp_path / "data", n_neg=4)
    stage = _stage(tmp_path, _model_yaml(tmp_path, tiny_modernbert), data, batch_size=8, mini_batch_size=4)
    with pytest.raises(TrainingShortfall, match="trained 0 of 2"):
        run_stage(stage, device="cpu")
    assert not (tmp_path / "out" / "final").exists()
