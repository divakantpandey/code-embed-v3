"""Streams a few real rows from every configured source to catch upstream schema drift."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from code_embed_v3.config import DataMixConfig, SourceConfig
from code_embed_v3.data.sources import build_source

_DATA = Path(__file__).resolve().parents[1] / "configs" / "data"
# Synthetic sources are local files produced by `make synth`, so only Hub-backed configs stream here.
SOURCES = [
    s for name in ("mix_v1.yaml", "eval_overlap.yaml") for s in DataMixConfig.from_yaml(_DATA / name).sources
]


@pytest.mark.network
@pytest.mark.parametrize("src", SOURCES, ids=lambda s: s.name)
def test_source_yields_valid_rows(src: SourceConfig) -> None:
    ds = build_source(dataclasses.replace(src, max_rows=3))
    assert len(ds) == 3
    assert {"anchor", "positive"} <= set(ds.column_names)
    assert all(a.strip() and p.strip() for a, p in zip(ds["anchor"], ds["positive"], strict=True))
