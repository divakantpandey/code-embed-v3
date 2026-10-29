"""Teacher labelling is the most expensive step, so it must resume per shard and keep raw log-odds."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from datasets import Dataset
from sentence_transformers import CrossEncoder

from code_embed_v3.data.mining import ScoreJob, crossencoder_scorer, label_source


def _mined(n: int = 10) -> Dataset:
    return Dataset.from_dict(
        {
            "anchor": [f"q{i}" for i in range(n)],
            "positive": [f"pos{i}" for i in range(n)],
            "negative_1": [f"neg{i}a" for i in range(n)],
            "negative_2": [f"neg{i}bb" for i in range(n)],
        }
    )


class LengthScorer:
    """Scores a pair by document length, so expected labels are known exactly."""

    def __init__(self) -> None:
        self.jobs: list[ScoreJob] = []

    def __call__(self, jobs: list[ScoreJob]) -> None:
        for job in jobs:
            self.jobs.append(job)
            part = Dataset.from_parquet(job["input"])
            assert isinstance(part, Dataset)
            cols = ["positive", "negative_1", "negative_2"]
            arr = np.array([[len(part[c][i]) for c in cols] for i in range(len(part))], dtype=np.float32)
            np.save(job["output"], arr)


def test_labels_align_with_rows(tmp_path: Path) -> None:
    ds = _mined()
    labelled = label_source(ds, LengthScorer(), "nl2code", tmp_path / "src", shard_rows=3)
    assert labelled["label"] == [
        [len(p), len(a), len(b)]
        for p, a, b in zip(ds["positive"], ds["negative_1"], ds["negative_2"], strict=True)
    ]


def test_restart_scores_only_missing_shards(tmp_path: Path) -> None:
    ds = _mined()
    first = LengthScorer()
    expected = label_source(ds, first, "nl2code", tmp_path / "src", shard_rows=3)["label"]
    assert len(first.jobs) == 4

    (tmp_path / "src.shards" / "00002.npy").unlink()
    again = LengthScorer()
    assert label_source(ds, again, "nl2code", tmp_path / "src", shard_rows=3)["label"] == expected
    assert [Path(j["output"]).name for j in again.jobs] == ["00002.npy"]


def test_needs_mined_negatives(tmp_path: Path) -> None:
    ds = Dataset.from_dict({"anchor": ["q"], "positive": ["p"]})
    with pytest.raises(ValueError, match="ce3 mine"):
        label_source(ds, LengthScorer(), "nl2code", tmp_path / "src")


def test_crossencoder_scores_are_raw_log_odds(tmp_path: Path) -> None:
    class FakeTeacher:
        def predict(self, pairs: list[tuple[str, str]], **kw: Any) -> torch.Tensor:
            logits = torch.linspace(-8.0, 8.0, len(pairs))
            return kw["activation_fn"](logits)

    labelled = label_source(
        _mined(2), crossencoder_scorer(cast(CrossEncoder, FakeTeacher()), 8), "nl2code", tmp_path / "s"
    )
    # A sigmoid would squash these into (0, 1); distillation needs the spread.
    assert np.allclose(np.asarray(labelled["label"]).ravel(), np.linspace(-8.0, 8.0, 6))
