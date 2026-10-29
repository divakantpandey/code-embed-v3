"""The leaderboard flags by provenance, so the default mix must be 100% zero-shot and stay that way."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from code_embed_v3.config import DataMixConfig
from code_embed_v3.data.build import MANIFEST, load_mix
from code_embed_v3.eval.run_mteb import NotZeroShotError, declared_meta, require_zero_shot, zero_shot_report
from code_embed_v3.paths import private_root

from .conftest import write_mix

CONFIGS = Path(__file__).parents[1] / "configs" / "data"
DATA_ROOTS = [CONFIGS, *([p] if (p := private_root() / "configs" / "data").is_dir() else [])]
BENCHMARKS = ("MTEB(Code, v1)", "CoIR", "CoREB(v1)")


def _flag(data: Path, tasks: list[str]) -> None:
    manifest = json.loads((data / MANIFEST).read_text())
    manifest["sources"]["toy"]["overlaps_eval"] = tasks
    (data / MANIFEST).write_text(json.dumps(manifest))


@pytest.mark.parametrize(
    "path", [r / n for r in DATA_ROOTS for n in ("mix_v1.yaml", "synth_v1.yaml") if (r / n).exists()], ids=str
)
def test_default_mixes_declare_no_eval_overlap(path: Path) -> None:
    cfg = DataMixConfig.from_yaml(path)
    assert [s.name for s in cfg.sources if s.overlaps_eval] == []


def test_every_overlapping_source_is_quarantined() -> None:
    cfg = DataMixConfig.from_yaml(CONFIGS / "eval_overlap.yaml")
    assert cfg.output_dir != DataMixConfig.from_yaml(CONFIGS / "mix_v1.yaml").output_dir
    assert all(s.overlaps_eval for s in cfg.sources)


def test_clean_mix_from_gte_is_fully_zero_shot(tmp_path: Path) -> None:
    data = write_mix(tmp_path / "data")
    report = zero_shot_report(declared_meta([data], "Alibaba-NLP/gte-modernbert-base"), BENCHMARKS)
    assert report == {b: 100 for b in BENCHMARKS}
    require_zero_shot(report)


def test_one_overlapping_source_fails_the_gate(tmp_path: Path) -> None:
    data = write_mix(tmp_path / "data")
    _flag(data, ["CodeSearchNetRetrieval", "COIRCodeSearchNetRetrieval"])
    report = zero_shot_report(declared_meta([data], None), BENCHMARKS)
    assert report["MTEB(Code, v1)"] < 100 and report["CoIR"] < 100 and report["CoREB(v1)"] == 100
    with pytest.raises(NotZeroShotError, match="not zero-shot"):
        require_zero_shot(report)


def test_training_refuses_overlapping_sources(tmp_path: Path) -> None:
    data = write_mix(tmp_path / "data")
    _flag(data, ["AppsRetrieval"])
    with pytest.raises(RuntimeError, match="non-zero-shot"):
        load_mix(data)
    datasets, _ = load_mix(data, allow_eval_overlap=True)
    assert "toy" in datasets
