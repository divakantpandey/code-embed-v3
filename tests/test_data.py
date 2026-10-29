from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from datasets import Dataset

from code_embed_v3.config import SourceConfig
from code_embed_v3.data import sources
from code_embed_v3.data.contamination import EvalIndex, audit_dataset, text_of
from code_embed_v3.prompts import Family


def _fake(monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, Any]]) -> None:
    def load(_: SourceConfig) -> Iterator[dict[str, Any]]:
        yield from rows

    monkeypatch.setattr(sources, "_load", load)


def _src(loader: str, family: Family = "nl2code", **kw: Any) -> SourceConfig:
    return SourceConfig(name="t", loader=loader, family=family, license="mit", hf_id="x/y", **kw)


def test_commitpackft_yields_message_and_unified_diff(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake(
        monkeypatch,
        [
            {
                "old_contents": "a = 1\n",
                "new_contents": "a = 2\n",
                "old_file": "x.py",
                "new_file": "x.py",
                "message": "bump a",
            },
            {
                "old_contents": "same\n",
                "new_contents": "same\n",
                "old_file": "y.py",
                "new_file": "y.py",
                "message": "noop",
            },
        ],
    )
    rows = list(sources.LOADERS["commitpackft"](_src("commitpackft", "code_edit")))
    assert len(rows) == 1
    assert rows[0]["anchor"] == "bump a"
    assert "-a = 1" in rows[0]["positive"] and "+a = 2" in rows[0]["positive"]


def test_apps_skips_rows_without_valid_solutions(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake(
        monkeypatch,
        [
            {"question": "q1", "solutions": json.dumps(["print(1)", "print(2)"])},
            {"question": "q2", "solutions": ""},
            {"question": "q3", "solutions": "{not json"},
        ],
    )
    assert list(sources.LOADERS["apps"](_src("apps"))) == [{"anchor": "q1", "positive": "print(1)"}]


def test_code_contests_pairs_are_cross_language(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake(
        monkeypatch,
        [
            {"solutions": {"language": [3, 2, 4], "solution": ["py", "cpp", "java"]}},
            {"solutions": {"language": [3, 1], "solution": ["py3", "py2"]}},  # one language only
        ],
    )
    rows = list(
        sources.LOADERS["code_contests_xlang"](
            _src("code_contests_xlang", "code2code", options={"pairs_per_problem": 5})
        )
    )
    assert len(rows) == 5
    assert all(r["anchor"] != r["positive"] for r in rows)


def test_cornstack_negatives(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake(
        monkeypatch,
        [
            {"query": "q", "document": "d", "negatives": ["n1", "n2", "n3"]},
            {"query": "q2", "document": "d2", "negatives": ["n"]},
        ],
    )
    rows = list(sources.LOADERS["cornstack"](_src("cornstack", options={"num_negatives": 2})))
    assert rows == [{"anchor": "q", "positive": "d", "negative_1": "n1", "negative_2": "n2"}]


def test_build_source_filters_and_dedups(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [{"anchor": "query one", "positive": "positive one"}] * 3 + [
        {"anchor": "", "positive": "empty anchor"},
        {"anchor": "query two", "positive": "positive two"},
        {"anchor": "query three", "positive": "positive three"},
    ]

    def load(_: SourceConfig) -> Iterator[dict[str, str]]:
        yield from rows

    monkeypatch.setitem(sources.LOADERS, "fake", load)
    ds = sources.build_source(_src("fake", max_rows=2))
    assert ds["anchor"] == ["query one", "query two"]


def test_unknown_loader_is_an_error() -> None:
    with pytest.raises(KeyError, match="unknown loader"):
        sources.build_source(_src("nope"))


LONG = "def merge sorted lists a b result while i less than len a and j less than len b append the smaller"


def test_eval_index_exact_and_near_duplicates() -> None:
    idx = EvalIndex(threshold=0.7)
    idx.add("CosQA", "sort a list in python")
    idx.add("AppsRetrieval", LONG)
    assert idx.hits("  Sort a LIST in python ") == {"CosQA"}
    assert idx.hits(LONG.replace("smaller", "smaller one")) == {"AppsRetrieval"}
    assert idx.hits("completely unrelated text about http retries and backoff") == set()


def test_audit_drops_rows_hitting_any_text_column() -> None:
    idx = EvalIndex()
    idx.add("CosQA", "leaked test query")
    ds = Dataset.from_dict(
        {
            "anchor": ["clean query", "leaked test query", "another clean query"],
            "positive": ["clean doc", "doc", "doc 2"],
            "negative_1": ["n", "n", "Leaked  test query"],
        }
    )
    kept, counts = audit_dataset(ds, idx)
    assert kept["anchor"] == ["clean query"]
    assert counts["CosQA"] == 2
    assert kept.column_names == ["anchor", "positive", "negative_1"]


def test_text_of_handles_multiturn() -> None:
    assert text_of([{"text": "hi"}, {"text": "there"}]) == "hi there"


def test_cornstack_reverse_gives_code_to_description(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake(
        monkeypatch, [{"query": "sort a list", "document": "def s(x): return sorted(x)", "negatives": ["n"]}]
    )
    rows = list(sources.LOADERS["cornstack"](_src("cornstack", "code2nl", options={"reverse": True})))
    # Negatives are code, so they must not appear as wrong descriptions.
    assert rows == [{"anchor": "def s(x): return sorted(x)", "positive": "sort a list"}]
