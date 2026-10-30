from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from transformers import AutoModel

from code_embed_v3.config import ModelConfig
from code_embed_v3.modeling import (
    build_model,
    enable_bidirectional_attention,
    init_model,
    load_model,
    save_model,
)
from code_embed_v3.prompts import QUERY_PROMPTS

from .conftest import HIDDEN


def _cfg(path: Path, **kw: Any) -> ModelConfig:
    base: dict[str, Any] = {
        "name": "tiny",
        "backbone": str(path),
        "max_seq_length": 64,
        "matryoshka_dims": [HIDDEN, 16, 8],
    }
    base.update(kw)
    return ModelConfig(**base)


def test_encoder_embeddings_are_normalised_and_truncatable(tiny_modernbert: Path) -> None:
    model = build_model(_cfg(tiny_modernbert), device="cpu")
    emb = model.encode(["def sort list", "read json file"], convert_to_tensor=True)
    assert emb.shape == (2, HIDDEN)
    torch.testing.assert_close(emb.norm(dim=-1), torch.ones(2))
    assert model.encode(["sum list"], truncate_dim=8).shape == (1, 8)
    assert set(model.prompts) >= set(QUERY_PROMPTS)


def test_projection_above_hidden_size_is_rejected(tiny_modernbert: Path) -> None:
    with pytest.raises(ValueError, match="hidden size"):
        build_model(_cfg(tiny_modernbert, matryoshka_dims=[HIDDEN * 2, HIDDEN]), device="cpu")


def _first_token_state(model: torch.nn.Module, ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state[0, 0]


def test_bidirectional_patch_lets_early_tokens_see_later_ones(tiny_qwen3: Path) -> None:
    a = torch.tensor([[5, 6, 7, 8]])
    b = torch.tensor([[5, 6, 7, 9]])
    causal = AutoModel.from_pretrained(tiny_qwen3).eval()
    torch.testing.assert_close(_first_token_state(causal, a), _first_token_state(causal, b))

    bidir = enable_bidirectional_attention(AutoModel.from_pretrained(tiny_qwen3).eval())
    assert not torch.allclose(_first_token_state(bidir, a), _first_token_state(bidir, b))
    # The patch is per instance: the causal model loaded earlier is untouched.
    torch.testing.assert_close(_first_token_state(causal, a), _first_token_state(causal, b))


def test_bidirectional_embeddings_ignore_padding(tiny_qwen3: Path) -> None:
    model = build_model(_cfg(tiny_qwen3, bidirectional=True), device="cpu")
    alone = model.encode(["sort list"], convert_to_tensor=True)
    batched = model.encode(["sort list", "read json file parse value return key map"], convert_to_tensor=True)
    torch.testing.assert_close(alone[0], batched[0], atol=1e-5, rtol=1e-4)


def test_bidirectional_survives_save_and_load(tiny_qwen3: Path, tmp_path: Path) -> None:
    cfg = _cfg(tiny_qwen3, bidirectional=True)
    model = build_model(cfg, device="cpu")
    texts = ["sort list python", "http retry error"]
    before = model.encode(texts, convert_to_tensor=True)
    save_model(model, tmp_path / "m", cfg)
    after = load_model(tmp_path / "m", device="cpu").encode(texts, convert_to_tensor=True)
    torch.testing.assert_close(before, after, atol=1e-5, rtol=1e-4)


def test_fp16_checkpoint_trains_with_fp32_master_weights(tiny_modernbert: Path, tmp_path: Path) -> None:
    # GTE-ModernBERT ships float16, and transformers loads the checkpoint dtype by default; fp16 master
    # weights collapsed every fine-tune (AdamW eps underflows, small updates round away).
    fp16 = tmp_path / "fp16"
    AutoModel.from_pretrained(tiny_modernbert).half().save_pretrained(fp16)
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        if (tiny_modernbert / name).exists():
            (fp16 / name).write_bytes((tiny_modernbert / name).read_bytes())
    assert AutoModel.from_pretrained(fp16).dtype == torch.float16

    cfg = _cfg(fp16)
    assert {p.dtype for p in build_model(cfg, device="cpu").parameters()} == {torch.float32}
    saved = tmp_path / "stage1"
    save_model(build_model(cfg, device="cpu").half(), saved, cfg)
    assert {p.dtype for p in init_model(cfg, str(saved), device="cpu").parameters()} == {torch.float32}
    # An explicit choice (the bf16 Qwen configs) is still honoured.
    assert {p.dtype for p in build_model(_cfg(fp16, torch_dtype="bfloat16"), device="cpu").parameters()} == {
        torch.bfloat16
    }
