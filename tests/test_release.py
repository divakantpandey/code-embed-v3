"""A public release carries the model and its usage, never the recipe."""

from __future__ import annotations

import json
from pathlib import Path

from code_embed_v3.release import build_release
from code_embed_v3.train import run_stage

from .conftest import write_mix
from .test_train_smoke import _model_yaml, _stage


def test_release_contains_no_training_recipe(tmp_path: Path, tiny_modernbert: Path) -> None:
    final = run_stage(
        _stage(tmp_path, _model_yaml(tmp_path, tiny_modernbert), write_mix(tmp_path / "d")), "cpu"
    )
    # The library's own card is full of training details; prove the release does not inherit them.
    assert "Training Hyperparameters" in (final / "README.md").read_text()
    (final.parent / "trainer_state.json").write_text("{}")
    report = tmp_path / "contamination_report.json"
    report.write_text(
        json.dumps(
            {
                "secret_source_a": {"before": 900, "after": 890, "CosQA": 10},
                "secret_source_b": {"before": 100, "after": 100},
            }
        )
    )

    out = build_release(
        final,
        tmp_path / "release",
        "me/model",
        base_model="Alibaba-NLP/gte-modernbert-base",
        contamination_report=report,
        audited_tasks=["CosQA", "AppsRetrieval"],
    )
    published = {str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()}
    assert {"model.safetensors", "modules.json", "README.md", "1_Pooling/config.json"} <= published
    assert not published & {"code_embed_v3.json", "trainer_state.json", "training_args.bin"}

    card = (out / "README.md").read_text()
    for leak in (
        "Training Hyperparameters",
        "learning_rate",
        "MatryoshkaLoss",
        "CachedMultipleNegatives",
        "secret_source",
        "toy",
        "hardness",
    ):
        assert leak not in card, leak
    assert "10 of 1,000 rows" in card and "2 tasks" in card
    assert 'prompt_name="nl2code"' in card and "base_model: Alibaba-NLP/gte-modernbert-base" in card
