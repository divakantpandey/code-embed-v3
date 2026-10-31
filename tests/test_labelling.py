"""Teacher labelling is the most expensive step, so it must resume per shard and keep raw log-odds."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from datasets import Dataset
from sentence_transformers import CrossEncoder

from code_embed_v3.data.mining import ScoreJob, crossencoder_scorer, label_source, label_sources


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


class CountingScorer(LengthScorer):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def __call__(self, jobs: list[ScoreJob]) -> None:
        self.calls += 1
        super().__call__(jobs)


def test_all_sources_share_one_scorer_call(tmp_path: Path) -> None:
    """One call means one vLLM start-up for the whole run instead of one per source."""
    a, b = _mined(5), _mined(7)
    scorer = CountingScorer()
    out = label_sources(
        {"a": (a, "nl2code", tmp_path / "a"), "b": (b, "code2code", tmp_path / "b")}, scorer, shard_rows=3
    )
    assert scorer.calls == 1 and len(scorer.jobs) == 2 + 3
    assert {j["instruction"] for j in scorer.jobs[:2]} != {j["instruction"] for j in scorer.jobs[2:]}
    for name, ds in (("a", a), ("b", b)):
        assert out[name]["label"] == [
            [len(p), len(x), len(y)]
            for p, x, y in zip(ds["positive"], ds["negative_1"], ds["negative_2"], strict=True)
        ]


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


def test_label_command_scores_in_one_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    from types import SimpleNamespace

    from code_embed_v3 import cli
    from code_embed_v3.data.build import MANIFEST

    mined = tmp_path / "mined"
    for name, n in (("a", 4), ("b", 6)):
        _mined(n).save_to_disk(str(mined / name))
    meta = {"family": "nl2code"}
    (mined / MANIFEST).write_text(json.dumps({"sources": {"a": meta, "b": meta}}))
    scorer = CountingScorer()
    monkeypatch.setattr(cli, "_teacher_scorer", lambda _: scorer)
    args = SimpleNamespace(
        data_dir=str(mined), output_dir=str(tmp_path / "out"), max_rows_per_source=None, shard_rows=4,
        teacher="t", backend="vllm",
    )  # fmt: skip
    cli._cmd_label(args)
    assert scorer.calls == 1
    manifest = json.loads((tmp_path / "out" / MANIFEST).read_text())
    assert {k: v["rows"] for k, v in manifest["sources"].items()} == {"a": 4, "b": 6}
    assert not (tmp_path / "out" / "a.shards").exists()

    again = CountingScorer()
    monkeypatch.setattr(cli, "_teacher_scorer", lambda _: again)
    cli._cmd_label(args)
    assert again.calls == 0
