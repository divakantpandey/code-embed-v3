"""Synthetic data: strict parsing, per-shard resume, and the round-trip filter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from datasets import Dataset

from code_embed_v3.data import synth
from code_embed_v3.data.synth import GenJob

CODE = "def add(a, b):\n    return a + b\n"


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ('{"query": "add two numbers in python"}', "add two numbers in python"),
        ('Sure!\n```json\n{"query": "sum two ints"}\n```', "sum two ints"),
        ('{"query": "def add(a, b):"}', None),  # pastes the code
        ('{"query": ""}', None),
        ("not json", None),
        ('["a list"]', None),
    ],
)
def test_nl2code_parse(reply: str, expected: str | None) -> None:
    row = synth.parse_nl2code(reply, CODE)
    assert (row and row["anchor"]) == expected
    if row:
        assert row["positive"] == CODE


def test_code_edit_builds_a_unified_diff() -> None:
    reply = json.dumps(
        {"request": "Support adding three numbers", "new_code": "def add(a, b, c=0):\n    return a + b + c\n"}
    )
    row = synth.parse_code_edit(reply, CODE)
    assert row is not None and row["anchor"] == "Support adding three numbers"
    assert "--- a/snippet" in row["positive"] and "+def add(a, b, c=0):" in row["positive"]
    assert synth.parse_code_edit(json.dumps({"request": "no-op change", "new_code": CODE}), CODE) is None


def test_multiturn_must_alternate_and_end_with_user() -> None:
    turns = [
        {"role": "user", "content": "I need a helper"},
        {"role": "assistant", "content": "For what?"},
        {"role": "user", "content": "Adding two numbers"},
    ]
    row = synth.parse_multiturn(json.dumps({"turns": turns}), CODE)
    assert row is not None and row["anchor"].splitlines()[-1] == "user: Adding two numbers"
    assert synth.parse_multiturn(json.dumps({"turns": turns[:2]}), CODE) is None
    assert synth.parse_multiturn(json.dumps({"turns": [turns[1], *turns]}), CODE) is None


def test_techqa_rejects_question_that_contains_the_code() -> None:
    answer = "Return the sum with the + operator:\n" + CODE
    good = json.dumps({"question": "How do I add two numbers in a function?", "answer": answer})
    assert synth.parse_techqa(good, CODE) is not None
    leaked = json.dumps(
        {"question": "Why does this fail?\n" + CODE, "answer": "It does not fail at all, really."}
    )
    assert synth.parse_techqa(leaked, CODE) is None


class EchoGenerator:
    """Replies with a query derived from the prompt, so outputs are deterministic and checkable."""

    def __init__(self) -> None:
        self.jobs: list[GenJob] = []

    def __call__(self, jobs: list[GenJob]) -> None:
        for job in jobs:
            self.jobs.append(job)
            users = Dataset.from_parquet(job["input"])["user"]
            with open(job["output"], "w") as fh:
                for u in users:
                    name = u.split("def ")[1].split("(")[0]
                    fh.write(json.dumps({"text": json.dumps({"query": f"find function {name}"})}) + "\n")


def test_generation_resumes_per_shard(tmp_path: Path) -> None:
    codes = [f"def f{i}(x):\n    return x * {i}\n" for i in range(7)]
    first = EchoGenerator()
    rows, stats = synth.generate("nl2code", codes, first, tmp_path, shard_rows=3)
    assert stats == {"seeds": 7, "parsed": 7, "rejected": 0} and len(first.jobs) == 3
    assert rows[4] == {"anchor": "find function f4", "positive": codes[4]}

    (tmp_path / "00001.jsonl").unlink()
    again = EchoGenerator()
    assert synth.generate("nl2code", codes, again, tmp_path, shard_rows=3)[0] == rows
    assert [Path(j["output"]).name for j in again.jobs] == ["00001.jsonl"]


def test_consistency_filter_drops_pairs_the_embedder_cannot_connect() -> None:
    rows = [{"anchor": f"q{i}", "positive": f"d{i}"} for i in range(4)]
    vectors = {
        "q0": [1, 0],
        "d0": [1, 0],
        "q1": [0, 1],
        "d1": [0, 1],
        "q2": [1, 0],
        "d2": [0, 1],
        "q3": [0, 1],
        "d3": [1, 0],
    }

    def encode(texts: list[str]) -> np.ndarray:
        return np.asarray([vectors[t] for t in texts], dtype=np.float32)

    kept = synth.consistency_filter(rows, encode, top_k=1)
    # q2/q3 point away from their own documents.
    assert [r["anchor"] for r in kept] == ["q0", "q1"]


def test_dedupe_ignores_case_and_spacing() -> None:
    rows = [{"anchor": "Sort a  list", "positive": "a"}, {"anchor": "sort a list", "positive": "b"}]
    assert synth.dedupe(rows) == rows[:1]


def test_rounds_call_the_push_hook_after_each_batch_of_shards(tmp_path: Path) -> None:
    codes = [f"def g{i}(x):\n    return x - {i}\n" for i in range(10)]
    pushes: list[int] = []
    gen = EchoGenerator()
    synth.generate(
        "nl2code", codes, gen, tmp_path, shard_rows=2, round_shards=2, after_round=lambda: pushes.append(1)
    )
    assert len(gen.jobs) == 5 and len(pushes) == 3


def test_seeds_are_frozen_on_first_run(tmp_path: Path) -> None:
    first = synth.load_or_save_seeds(tmp_path, lambda: ["a", "b"])
    assert synth.load_or_save_seeds(tmp_path, lambda: ["changed"]) == first == ["a", "b"]
