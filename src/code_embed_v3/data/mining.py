"""Hard-negative mining and teacher scoring for stages 2 and 3."""

from __future__ import annotations

import importlib.util
import logging
import os
import shutil
from pathlib import Path
from typing import Protocol, TypedDict

import numpy as np
import torch
from datasets import Dataset, load_from_disk
from sentence_transformers import CrossEncoder, SentenceTransformer
from sentence_transformers.util import mine_hard_negatives
from transformers import PreTrainedTokenizerBase

from code_embed_v3.data.isolated import run_vllm_workers
from code_embed_v3.prompts import QUERY_PROMPTS, Family

logger = logging.getLogger(__name__)

# Teacher instructions, phrased the way Qwen3-Reranker's model card phrases its own default.
TEACHER_INSTRUCTIONS: dict[Family, str] = {
    "nl2code": "Given a natural language query, judge whether the code satisfies it",
    "code2nl": "Given a code snippet, judge whether the text correctly describes it",
    "code2code": "Given a code snippet, judge whether the candidate code is functionally equivalent",
    "techqa": "Given a programming question, judge whether the answer resolves it",
    "text2sql": "Given a natural language request, judge whether the SQL query answers it",
    "code_edit": "Given a change description, judge whether the diff implements it",
    "multiturn": "Given a conversation, judge whether the reply is the correct next turn",
}


def mine(
    ds: Dataset,
    model: SentenceTransformer,
    family: Family,
    num_negatives: int = 7,
    range_min: int = 0,
    range_max: int = 100,
    relative_margin: float = 0.05,
    batch_size: int = 256,
    use_multi_process: bool = False,
    query_prompts: bool = True,
) -> Dataset:
    """Positive-aware mining (NV-Retriever TopK-PercPos): drop candidates scoring within
    ``relative_margin`` of the positive, because duplicated code makes false negatives common.
    """
    return mine_hard_negatives(
        dataset=ds,
        model=model,
        anchor_column_name="anchor",
        positive_column_name="positive",
        range_min=range_min,
        range_max=range_max,
        relative_margin=relative_margin,
        num_negatives=num_negatives,
        sampling_strategy="top",
        # Off when mining with a checkpoint that never saw our prompts (e.g. GTE-ModernBERT).
        query_prompt=QUERY_PROMPTS[family] if query_prompts else None,
        output_format="n-tuple",
        batch_size=batch_size,
        use_faiss=importlib.util.find_spec("faiss") is not None,
        use_multi_process=use_multi_process,
    )


VLLM_WORKER = Path(__file__).with_name("vllm_teacher.py")
# The vLLM teacher's per-side token budget; the cross-encoder reference has none (131k context).
TEACHER_MAX_QUERY_TOKENS = 512
TEACHER_MAX_DOC_TOKENS = 2048


def doc_columns(ds: Dataset) -> list[str]:
    negatives = sorted(
        (c for c in ds.column_names if c.startswith("negative_")), key=lambda c: int(c.split("_")[1])
    )
    if not negatives:
        raise ValueError("teacher scoring needs mined negatives; run `ce3 mine` first")
    return ["positive", *negatives]


class ScoreJob(TypedDict):
    """One shard for a scorer; plain JSON so it can cross into the vLLM worker's environment."""

    input: str  # parquet with anchor + doc columns
    output: str  # .npy of shape [rows, docs]
    instruction: str


class ShardScorer(Protocol):
    def __call__(self, jobs: list[ScoreJob]) -> None: ...


def crossencoder_scorer(teacher: CrossEncoder, batch_size: int) -> ShardScorer:
    def score(jobs: list[ScoreJob]) -> None:
        for job in jobs:
            part = Dataset.from_parquet(job["input"])
            if not isinstance(part, Dataset):
                raise TypeError(job["input"])
            cols = doc_columns(part)
            pairs = [(q, part[c][i]) for i, q in enumerate(part["anchor"]) for c in cols]
            # Identity keeps the raw yes/no log-odds. The default sigmoid squashes them into [0, 1],
            # where a softmax at teacher_temperature=1 is almost uniform and distils nothing.
            scores = teacher.predict(
                pairs,
                prompt=job["instruction"],
                batch_size=batch_size,
                activation_fn=torch.nn.Identity(),
                convert_to_tensor=True,
            )
            arr = torch.as_tensor(scores, dtype=torch.float32).view(len(part), len(cols)).cpu().numpy()
            _save_npy(Path(job["output"]), arr)

    return score


def vllm_scorer(teacher: str, num_gpus: int) -> ShardScorer:
    """One vLLM process per GPU, each taking every ``num_gpus``-th shard."""
    args = [
        "--model", teacher,
        "--max-query-tokens", str(TEACHER_MAX_QUERY_TOKENS),
        "--max-doc-tokens", str(TEACHER_MAX_DOC_TOKENS),
    ]  # fmt: skip

    def score(jobs: list[ScoreJob]) -> None:
        run_vllm_workers(
            VLLM_WORKER,
            jobs,
            args,
            num_gpus,
            Path(jobs[0]["output"]).parent,
        )

    return score


def _save_npy(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def clip_to_teacher_window(ds: Dataset, tokenizer: PreTrainedTokenizerBase) -> Dataset:
    """Cut queries and documents to the vLLM teacher's token budget, the way its worker does.

    Lets the cross-encoder reference score exactly the text vLLM sees: long contest solutions and
    diffs otherwise differ only by truncation, and the parity check reads that as a broken backend.
    """

    def clip(texts: list[str], max_tokens: int) -> list[str]:
        ids = tokenizer(texts, add_special_tokens=False, truncation=True, max_length=max_tokens)["input_ids"]
        return tokenizer.batch_decode(ids)

    docs = doc_columns(ds)

    def clip_batch(batch: dict[str, list[str]]) -> dict[str, list[str]]:
        return {
            "anchor": clip(batch["anchor"], TEACHER_MAX_QUERY_TOKENS),
            **{c: clip(batch[c], TEACHER_MAX_DOC_TOKENS) for c in docs},
        }

    return ds.map(clip_batch, batched=True)


def label_source(
    ds: Dataset,
    scorer: ShardScorer,
    family: Family,
    out: Path,
    shard_rows: int = 20_000,
) -> Dataset:
    """Add ``label`` = [s(q, pos), s(q, neg_1), ...] (teacher log-odds) as DistillKLDivLoss expects.

    Work is split into shards whose scores land in ``<out>.shards/``; a restart only scores the
    shards without a finished ``.npy``, so a lost pod costs at most one shard per GPU.
    """
    cols = doc_columns(ds)
    shard_dir = out.with_name(out.name + ".shards")
    shard_dir.mkdir(parents=True, exist_ok=True)
    starts = list(range(0, len(ds), shard_rows))
    pending: list[ScoreJob] = []
    for i, start in enumerate(starts):
        npy = shard_dir / f"{i:05d}.npy"
        if npy.is_file():
            continue
        parquet = shard_dir / f"{i:05d}.parquet"
        ds.select(range(start, min(start + shard_rows, len(ds)))).select_columns(
            ["anchor", *cols]
        ).to_parquet(str(parquet))
        pending.append(
            ScoreJob(input=str(parquet), output=str(npy), instruction=TEACHER_INSTRUCTIONS[family])
        )
    logger.info("%s: %d/%d shards to score", out.name, len(pending), len(starts))
    if pending:
        scorer(pending)
    scores = np.concatenate([np.load(shard_dir / f"{i:05d}.npy") for i in range(len(starts))])
    if scores.shape != (len(ds), len(cols)):
        raise ValueError(f"{out.name}: scores {scores.shape} do not match {(len(ds), len(cols))}")
    return ds.add_column("label", scores.tolist())


def finish_shards(out: Path) -> None:
    shutil.rmtree(out.with_name(out.name + ".shards"), ignore_errors=True)


def save_dataset_atomic(ds: Dataset, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    ds.save_to_disk(str(tmp))
    if path.exists():
        shutil.rmtree(path)
    tmp.rename(path)


def is_saved_dataset(path: Path) -> bool:
    return (path / "dataset_info.json").is_file()


def load_dataset_dir(path: Path) -> Dataset:
    ds = load_from_disk(str(path))
    if not isinstance(ds, Dataset):
        raise TypeError(f"{path} is not a single Dataset")
    return ds
