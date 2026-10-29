"""Offline fixtures: tiny random ModernBERT / Qwen3 backbones with a local word-level tokenizer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from datasets import Dataset
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    ModernBertConfig,
    ModernBertModel,
    PreTrainedTokenizerFast,
    Qwen3Config,
    Qwen3Model,
)

from code_embed_v3.data.build import DEV_DIR, MANIFEST

SPECIAL = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
WORDS = [
    "def",
    "return",
    "sort",
    "list",
    "python",
    "go",
    "java",
    "read",
    "file",
    "parse",
    "json",
    "http",
    "retry",
    "error",
    "value",
    "key",
    "map",
    "reduce",
    "filter",
    "class",
    "import",
    "string",
    "int",
    "float",
    "sum",
    "max",
    "min",
    "loop",
    "for",
    "while",
    "if",
    "else",
    "query",
    "sql",
    "select",
    "from",
    "where",
    "table",
    "fix",
    "bug",
    "add",
    "test",
    "commit",
    "diff",
    "the",
    "a",
    "of",
    "to",
    "in",
    "how",
    "do",
    "i",
]

HIDDEN = 32


def _tokenizer(path: Path) -> None:
    vocab = {tok: i for i, tok in enumerate(SPECIAL + WORDS)}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        model_max_length=128,
    ).save_pretrained(path)


@pytest.fixture(scope="session")
def tiny_modernbert(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("tiny-modernbert")
    torch.manual_seed(0)
    cfg = ModernBertConfig(
        vocab_size=len(SPECIAL) + len(WORDS),
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=128,
        pad_token_id=0,
        cls_token_id=2,
        sep_token_id=3,
        bos_token_id=2,
        eos_token_id=3,
    )
    ModernBertModel(cfg).save_pretrained(path)
    _tokenizer(path)
    return path


@pytest.fixture(scope="session")
def tiny_qwen3(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("tiny-qwen3")
    torch.manual_seed(0)
    cfg = Qwen3Config(
        vocab_size=len(SPECIAL) + len(WORDS),
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        pad_token_id=0,
    )
    Qwen3Model(cfg).save_pretrained(path)
    _tokenizer(path)
    return path


PAIRS = [
    ("how do i sort list python", "def sort list return sorted list"),
    ("read json file", "def read file parse json return value"),
    ("http retry error", "def http retry while error loop"),
    ("sum of list", "def sum list return sum"),
    ("max value in map", "def max map return max value"),
    ("filter list", "def filter list return filter"),
    ("parse int string", "def parse string return int"),
    ("select from table", "select key from table where value"),
]


def write_mix(
    root: Path, name: str = "toy", family: str = "nl2code", n_neg: int = 0, labels: bool = False
) -> Path:
    """A built-and-audited mix on disk, in the exact layout `ce3 build-data` + `ce3 audit` produce."""
    rows: dict[str, list] = {"anchor": [a for a, _ in PAIRS] * 4, "positive": [p for _, p in PAIRS] * 4}
    # Make anchors unique so the no-duplicates sampler can fill batches.
    rows["anchor"] = [f"{a} {WORDS[i % len(WORDS)]}" for i, a in enumerate(rows["anchor"])]
    rows["positive"] = [f"{p} {WORDS[(i * 7) % len(WORDS)]}" for i, p in enumerate(rows["positive"])]
    for k in range(1, n_neg + 1):
        rows[f"negative_{k}"] = [
            rows["positive"][(i + k) % len(rows["positive"])] for i in range(len(rows["anchor"]))
        ]
    if labels:
        rows["label"] = [[1.0] + [0.1 * k for k in range(n_neg)] for _ in rows["anchor"]]
    Dataset.from_dict(rows).save_to_disk(str(root / name))
    Dataset.from_dict({"anchor": [a for a, _ in PAIRS], "positive": [p for _, p in PAIRS]}).save_to_disk(
        str(root / DEV_DIR / name)
    )
    manifest = {
        "sources": {
            name: {"family": family, "rows": len(rows["anchor"]), "dev_rows": len(PAIRS), "audited": True}
        }
    }
    (root / MANIFEST).write_text(json.dumps(manifest))
    return root
