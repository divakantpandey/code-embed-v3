"""Option A starts from GTE-ModernBERT, so the rebuilt start must be that model, and its exposure is ours."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from code_embed_v3.config import ModelConfig
from code_embed_v3.eval.run_mteb import declared_meta
from code_embed_v3.modeling import build_model

from .conftest import write_mix

GTE = "Alibaba-NLP/gte-modernbert-base"
GTE_CONFIG = Path(__file__).parents[1] / "configs" / "models" / "gte-modernbert-150m.yaml"


def test_fine_tune_inherits_base_model_training_data(tmp_path: Path) -> None:
    data = write_mix(tmp_path / "data")
    own = declared_meta([data], base_model=None).get_training_datasets()
    inherited = declared_meta([data], base_model=GTE).get_training_datasets()
    # mteb follows `adapted_from` itself; GTE-ModernBERT declares MSMARCO in its registry entry.
    assert "MSMARCO" in inherited and "MSMARCO" not in own


@pytest.mark.network
def test_rebuilt_start_matches_published_gte_modernbert() -> None:
    from sentence_transformers import SentenceTransformer

    texts = ["def retry(fn, attempts=3):\n    for _ in range(attempts):\n        return fn()", "sort a list"]
    reference = SentenceTransformer(GTE, device="cpu").encode(texts, convert_to_tensor=True)
    ours = build_model(ModelConfig.from_yaml(GTE_CONFIG), device="cpu").encode(texts, convert_to_tensor=True)
    # Documents take no prompt, so a correctly rebuilt model reproduces upstream exactly (up to the
    # Normalize module, which cosine similarity already implies).
    cos = torch.nn.functional.cosine_similarity(reference, ours)
    assert torch.all(cos > 0.9999), cos
