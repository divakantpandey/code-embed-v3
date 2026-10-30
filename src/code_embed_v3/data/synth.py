"""Synthetic pairs from real code: developer queries, Q&A, change requests and conversations.

The public mix is mostly docstring -> code, and the zero-shot mix has no source for techqa,
multiturn or commit-style code_edit (those sources share provenance with eval tasks). Here an open
coder LLM writes the query side for CoRNStack code; the code side is always real.

Every row then goes through ``ce3 build-data`` and ``ce3 audit`` like any other source.
"""

from __future__ import annotations

import difflib
import json
import logging
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypedDict

import numpy as np
from datasets import Dataset, load_from_disk

from code_embed_v3.config import load_yaml
from code_embed_v3.data.isolated import run_vllm_workers
from code_embed_v3.paths import resolve
from code_embed_v3.prompts import Family

logger = logging.getLogger(__name__)

GENERATOR = "Qwen/Qwen2.5-Coder-7B-Instruct"  # Apache-2.0, fits a 24 GB GPU in bf16
VLLM_WORKER = Path(__file__).with_name("vllm_generate.py")

Row = dict[str, str]


def _json_obj(text: str) -> dict[str, Any] | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _text(obj: dict[str, Any], key: str, lo: int, hi: int) -> str | None:
    value = obj.get(key)
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value if lo <= len(value) <= hi else None


def parse_nl2code(reply: str, code: str) -> Row | None:
    obj = _json_obj(reply)
    query = _text(obj, "query", 6, 300) if obj else None
    # A query that pastes code is a code2code pair in disguise.
    if query is None or "```" in query or query in code:
        return None
    return {"anchor": query, "positive": code}


def parse_techqa(reply: str, code: str) -> Row | None:
    obj = _json_obj(reply)
    if obj is None:
        return None
    question, answer = _text(obj, "question", 30, 3000), _text(obj, "answer", 40, 12000)
    if question is None or answer is None or code.strip() in question:
        return None
    return {"anchor": question, "positive": answer}


def parse_code_edit(reply: str, code: str, max_changed_lines: int = 60) -> Row | None:
    """Change request -> unified diff, the shape of MTEB's CodeEditSearchRetrieval queries/docs."""
    obj = _json_obj(reply)
    if obj is None:
        return None
    request, new_code = _text(obj, "request", 10, 400), obj.get("new_code")
    if request is None or not isinstance(new_code, str) or new_code.strip() == code.strip():
        return None
    diff = "".join(
        difflib.unified_diff(
            code.splitlines(keepends=True),
            new_code.splitlines(keepends=True),
            fromfile="a/snippet",
            tofile="b/snippet",
            n=3,
        )
    )
    changed = sum(1 for line in diff.splitlines() if line[:1] in "+-" and line[:3] not in ("+++", "---"))
    if not diff or changed > max_changed_lines:
        return None
    return {"anchor": request, "positive": diff}


def parse_multiturn(reply: str, code: str) -> Row | None:
    obj = _json_obj(reply)
    turns = obj.get("turns") if obj else None
    if not isinstance(turns, list) or not 3 <= len(turns) <= 7:
        return None
    lines: list[str] = []
    for i, turn in enumerate(turns):
        role = turn.get("role") if isinstance(turn, dict) else None
        content = turn.get("content") if isinstance(turn, dict) else None
        # Must alternate user/assistant and end on the user request the code answers.
        if (
            role != ("user" if i % 2 == 0 else "assistant")
            or not isinstance(content, str)
            or not content.strip()
        ):
            return None
        lines.append(f"{role}: {content.strip()}")
    if len(turns) % 2 == 0:
        return None
    return {"anchor": "\n".join(lines), "positive": code}


@dataclass(frozen=True)
class SynthTask:
    family: Family
    user: str  # @@CODE@@ and @@STYLE@@ placeholders; str.format would choke on braces in code
    styles: tuple[str, ...]
    parse: Callable[[str, str], Row | None]


PARSERS: dict[str, tuple[Family, Callable[[str, str], Row | None]]] = {
    "nl2code": ("nl2code", parse_nl2code),
    "techqa": ("techqa", parse_techqa),
    "code_edit": ("code_edit", parse_code_edit),
    "multiturn": ("multiturn", parse_multiturn),
}
PROMPTS = "configs/synth/prompts.yaml"  # public example; the real prompts live in the private overlay


@dataclass(frozen=True)
class Prompts:
    system: str
    tasks: dict[str, SynthTask]


def load_prompts(
    path: str | Path | None = None,
    parsers: dict[str, tuple[Family, Callable[[str, str], Row | None]]] | None = None,
) -> Prompts:
    parsers = parsers or PARSERS
    raw = load_yaml(path or resolve(PROMPTS))
    tasks = {}
    for name, spec in raw["tasks"].items():
        if name not in parsers:
            raise ValueError(f"unknown synth task {name!r}; known: {sorted(parsers)}")
        if "@@CODE@@" not in spec["user"] or (spec.get("styles") and "@@STYLE@@" not in spec["user"]):
            raise ValueError(f"synth task {name}: prompt needs @@CODE@@ (and @@STYLE@@ when styles are set)")
        family, parse = parsers[name]
        tasks[name] = SynthTask(family, spec["user"], tuple(spec.get("styles") or ("",)), parse)
    return Prompts(system=raw["system"], tasks=tasks)


class GenJob(TypedDict):
    input: str  # parquet with system / user columns
    output: str  # .jsonl, one {"text": reply} per input row


class Generator(Protocol):
    def __call__(self, jobs: list[GenJob]) -> None: ...


class Encoder(Protocol):
    def __call__(self, texts: list[str]) -> np.ndarray: ...


def vllm_generator(model: str, num_gpus: int, max_tokens: int = 1024, seed: int = 0) -> Generator:
    def generate(jobs: list[GenJob]) -> None:
        args = ["--model", model, "--max-tokens", str(max_tokens), "--seed", str(seed)]
        run_vllm_workers(VLLM_WORKER, jobs, args, num_gpus, Path(jobs[0]["output"]).parent)

    return generate


def seed_code(
    seed_dirs: Sequence[Path], n: int, seed: int, min_chars: int = 80, max_chars: int = 6000
) -> list[str]:
    """Up to ``n`` distinct code snippets, spread evenly over the seed sources (``positive`` column)."""
    rng = random.Random(seed)
    per_dir = -(-n // len(seed_dirs))
    picked: list[str] = []
    for d in seed_dirs:
        ds = load_from_disk(str(d))
        if not isinstance(ds, Dataset):
            raise TypeError(f"{d} is not a single Dataset")
        order = rng.sample(range(len(ds)), len(ds))
        taken = 0
        for start in range(0, len(order), 4096):
            for code in ds.select(order[start : start + 4096])["positive"]:
                if min_chars <= len(code) <= max_chars:
                    picked.append(code)
                    taken += 1
                    if taken == per_dir:
                        break
            if taken == per_dir:
                break
    return list(dict.fromkeys(picked))[:n]


def generate(
    task_name: str,
    codes: Sequence[str],
    generator: Generator,
    work_dir: Path,
    shard_rows: int = 2000,
    seed: int = 0,
    round_shards: int | None = None,
    after_round: Callable[[], None] | None = None,
    prompts: Prompts | None = None,
) -> tuple[list[Row], dict[str, int]]:
    """Generate and parse one reply per snippet. Shards with a finished ``.jsonl`` are not regenerated.

    Pending shards go to the generator ``round_shards`` at a time, and ``after_round`` runs after each
    round (the CLI pushes the work dir to the Hub there), so a lost pod costs at most one round.
    """
    prompts = prompts or load_prompts()
    task = prompts.tasks[task_name]
    rng = random.Random(seed)
    work_dir.mkdir(parents=True, exist_ok=True)
    users = [task.user.replace("@@STYLE@@", rng.choice(task.styles)).replace("@@CODE@@", c) for c in codes]
    starts = list(range(0, len(codes), shard_rows))
    pending: list[GenJob] = []
    for i, start in enumerate(starts):
        out = work_dir / f"{i:05d}.jsonl"
        if out.is_file():
            continue
        parquet = work_dir / f"{i:05d}.parquet"
        part = users[start : start + shard_rows]
        Dataset.from_dict({"system": [prompts.system] * len(part), "user": part}).to_parquet(str(parquet))
        pending.append(GenJob(input=str(parquet), output=str(out)))
    logger.info("%s: %d/%d shards to generate", task_name, len(pending), len(starts))
    step = round_shards or max(1, len(pending))
    for r in range(0, len(pending), step):
        generator(pending[r : r + step])
        if after_round:
            after_round()

    rows: list[Row] = []
    stats = {"seeds": len(codes), "parsed": 0, "rejected": 0}
    for i, start in enumerate(starts):
        replies = [
            json.loads(line)["text"] for line in (work_dir / f"{i:05d}.jsonl").read_text().splitlines()
        ]
        snippets = codes[start : start + shard_rows]
        if len(replies) != len(snippets):
            raise ValueError(f"shard {i}: {len(replies)} replies for {len(snippets)} prompts")
        for reply, code in zip(replies, snippets, strict=True):
            row = task.parse(reply, code)
            if row is None:
                stats["rejected"] += 1
            else:
                rows.append(row)
                stats["parsed"] += 1
    return rows, stats


def dedupe(rows: list[Row]) -> list[Row]:
    seen: set[str] = set()
    out: list[Row] = []
    for row in rows:
        key = " ".join(row["anchor"].lower().split())
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out


def consistency_filter(rows: list[Row], encode: Encoder, top_k: int = 3, chunk: int = 512) -> list[Row]:
    """Keep a pair only if its positive ranks in the top ``top_k`` of ``chunk`` candidates for its anchor.

    The CoRNStack-style round-trip check: a generated query that a strong embedder cannot connect
    back to its own code is usually vague or wrong, and would teach the model noise.
    """
    kept: list[Row] = []
    for start in range(0, len(rows), chunk):
        part = rows[start : start + chunk]
        q = encode([r["anchor"] for r in part])
        d = encode([r["positive"] for r in part])
        sims = q @ d.T
        own = np.diag(sims)
        ranks = (sims > own[:, None]).sum(axis=1)
        kept.extend(r for r, rank in zip(part, ranks, strict=True) if rank < top_k)
    return kept


def sentence_transformer_encoder(model_name: str, batch_size: int = 128) -> Encoder:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def encode(texts: list[str]) -> np.ndarray:
        return np.asarray(model.encode(texts, batch_size=batch_size, normalize_embeddings=True))

    return encode


def write_rows(rows: list[Row], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    Dataset.from_list(rows).to_parquet(str(tmp))
    tmp.replace(out)


def load_or_save_seeds(work_dir: Path, pick: Callable[[], list[str]]) -> list[str]:
    """Freeze the seed snippets on first run, so a resumed run pairs replies with the same code even
    if the seed sources were rebuilt in between."""
    path = work_dir / "seeds.parquet"
    if path.is_file():
        ds = Dataset.from_parquet(str(path))
        if not isinstance(ds, Dataset):
            raise TypeError(str(path))
        return list(ds["code"])
    codes = pick()
    work_dir.mkdir(parents=True, exist_ok=True)
    Dataset.from_dict({"code": codes}).to_parquet(str(path))
    return codes
