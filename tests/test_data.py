from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mteb
import pytest
from datasets import Dataset

from code_embed_v3.cli import _cmd_audit
from code_embed_v3.config import SourceConfig
from code_embed_v3.data import contamination, sources
from code_embed_v3.data.build import DEV_DIR, MANIFEST
from code_embed_v3.data.contamination import EvalIndex, audit_dataset, index_mteb_tasks, text_of
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


def test_index_mteb_tasks_converts_v1_retrieval_format(monkeypatch: pytest.MonkeyPatch) -> None:
    class V1Task:
        metadata = SimpleNamespace(eval_splits=["test"])
        dataset: dict[str, Any] | None = None

        def load_data(self) -> None:
            # v1 tasks leave self.dataset as None until convert_v1_dataset_format_to_v2 is called.
            self.dataset = None

        def convert_v1_dataset_format_to_v2(self, num_proc: int | None = None) -> None:
            self.dataset = {
                "default": {
                    "test": {
                        "queries": [{"id": "q1", "text": "find binary search"}],
                        "corpus": [
                            {"id": "d1", "text": "def bsearch(a, x): pass"},
                            {"id": "d2", "text": "unreferenced doc"},
                        ],
                        "relevant_docs": {"q1": {"d1": 1}},
                    }
                }
            }

    monkeypatch.setattr(mteb, "get_task", lambda name: V1Task())
    idx = index_mteb_tasks(["CodeEditSearchRetrieval"])
    assert idx.hits("find binary search") == {"CodeEditSearchRetrieval"}
    assert idx.hits("def bsearch(a, x): pass") == {"CodeEditSearchRetrieval"}
    assert idx.hits("unreferenced doc") == set()


def test_cmd_audit_cleans_train_and_dev_slices(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    train_ds = Dataset.from_dict({"anchor": ["clean train", "leaked query"], "positive": ["p1", "p2"]})
    dev_ds = Dataset.from_dict({"anchor": ["clean dev", "leaked query"], "positive": ["d1", "d2"]})
    train_ds.save_to_disk(str(tmp_path / "toy"))
    dev_ds.save_to_disk(str(tmp_path / DEV_DIR / "toy"))
    manifest = {"sources": {"toy": {"family": "nl2code", "rows": 2, "dev_rows": 2, "audited": False}}}
    (tmp_path / MANIFEST).write_text(json.dumps(manifest))

    idx = EvalIndex()
    idx.add("FakeTask", "leaked query")
    monkeypatch.setattr(contamination, "index_mteb_tasks", lambda tasks, index=None: idx)

    _cmd_audit(
        SimpleNamespace(
            data_dir=str(tmp_path),
            benchmarks=None,
            tasks=["FakeTask"],
            external=False,
            threshold=0.7,
            num_proc=1,
            report=str(tmp_path / "report.json"),
        )
    )
    updated = json.loads((tmp_path / MANIFEST).read_text())["sources"]["toy"]
    assert updated["audited"] is True
    assert updated["rows"] == 1
    assert updated["dev_rows"] == 1
    assert Dataset.load_from_disk(str(tmp_path / "toy"))["anchor"] == ["clean train"]
    assert Dataset.load_from_disk(str(tmp_path / DEV_DIR / "toy"))["anchor"] == ["clean dev"]


_BUG_PATCH = (
    "diff --git a/mod.py b/mod.py\n"
    "index 1111111..2222222 100644\n"
    "--- a/mod.py\n"
    "+++ b/mod.py\n"
    "@@ -1,4 +1,3 @@\n"
    " def clamp(x, lo):\n"
    "--- x\n"
    "-    return max(x, lo)\n"
    "+    return min(x, lo)\n"
    " \n"
)


def test_reverse_patch_turns_a_bug_patch_into_its_fix(tmp_path: Path) -> None:
    import shutil
    import subprocess

    fix = sources.reverse_patch(_BUG_PATCH)
    assert fix.splitlines()[2:5] == ["--- b/mod.py", "+++ a/mod.py", "@@ -1,3 +1,4 @@"]
    assert sources.reverse_patch(fix) == _BUG_PATCH
    if shutil.which("git") is None:
        pytest.skip("git not installed")
    (tmp_path / "mod.py").write_text("def clamp(x, lo):\n    return min(x, lo)\n\n")
    (tmp_path / "fix.diff").write_text(fix)
    subprocess.run(["git", "apply", "fix.diff"], cwd=tmp_path, check=True)
    assert (tmp_path / "mod.py").read_text() == "def clamp(x, lo):\n-- x\n    return max(x, lo)\n\n"


def test_swe_smith_pairs_issue_with_fixing_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake(
        monkeypatch,
        [
            {"problem_statement": "clamp ignores the lower bound", "patch": _BUG_PATCH},
            {"problem_statement": "", "patch": _BUG_PATCH},
        ],
    )
    rows = list(sources.LOADERS["swe_smith"](_src("swe_smith", "code_edit")))
    assert len(rows) == 1
    assert "+    return max(x, lo)" in rows[0]["positive"]
    assert "-    return min(x, lo)" in rows[0]["positive"]


def test_replace_dir_survives_undeletable_backup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import errno
    import shutil

    from code_embed_v3.cli import _replace_dir

    dst, src = tmp_path / "source", tmp_path / "source.audited"
    dst.mkdir()
    (dst / "old.txt").write_text("old")
    src.mkdir()
    (src / "new.txt").write_text("new")
    (tmp_path / "source.pre-audit").mkdir()  # leftover from an earlier crashed audit
    (tmp_path / "source.pre-audit" / "stale.txt").write_text("stale")

    real_rmtree = shutil.rmtree

    def nfs_rmtree(path: Any, *a: Any, **kw: Any) -> None:
        if (Path(path) / "old.txt").exists():
            raise OSError(errno.ENOTEMPTY, "Directory not empty", str(path))
        real_rmtree(path, *a, **kw)

    monkeypatch.setattr(shutil, "rmtree", nfs_rmtree)
    _replace_dir(src, dst)
    assert (dst / "new.txt").read_text() == "new" and not (dst / "old.txt").exists()
    assert not src.exists()
    assert not (tmp_path / "source.pre-audit" / "stale.txt").exists()


@pytest.mark.parametrize(("write_outputs", "fails"), [(True, False), (False, True)])
def test_vllm_worker_crash_after_writing_outputs_is_kept(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, write_outputs: bool, fails: bool
) -> None:
    from code_embed_v3.data import isolated

    jobs = [{"output": str(tmp_path / f"{i}.npy")} for i in range(2)]

    class Aborting:
        def __init__(self, cmd: list[str], env: dict[str, str]) -> None:
            self.args, self.returncode = cmd, -6
            if write_outputs:
                for job in jobs:
                    Path(job["output"]).write_bytes(b"")

        def wait(self) -> int:
            return self.returncode

    monkeypatch.setattr(isolated.subprocess, "Popen", Aborting)
    if fails:
        with pytest.raises(RuntimeError, match="vLLM workers failed"):
            isolated.run_vllm_workers(tmp_path / "w.py", jobs, [], 1, tmp_path / "spec")
    else:
        isolated.run_vllm_workers(tmp_path / "w.py", jobs, [], 1, tmp_path / "spec")


def test_label_check_clips_both_backends_to_the_teacher_window() -> None:
    from code_embed_v3.data.mining import (
        TEACHER_MAX_DOC_TOKENS,
        TEACHER_MAX_QUERY_TOKENS,
        clip_to_teacher_window,
    )

    class Words:
        def __call__(self, texts: list[str], **kw: Any) -> dict[str, list[list[str]]]:
            return {"input_ids": [t.split()[: kw["max_length"]] for t in texts]}

        def batch_decode(self, ids: list[list[str]]) -> list[str]:
            return [" ".join(i) for i in ids]

    ds = Dataset.from_dict(
        {
            "anchor": ["q " * 600, "short query"],
            "positive": ["d " * 3000, "x"],
            "negative_1": ["y", "d " * 2100],
        }
    )
    out = clip_to_teacher_window(ds, Words())
    assert [len(t.split()) for t in out["anchor"]] == [TEACHER_MAX_QUERY_TOKENS, 2]
    assert [len(t.split()) for t in out["positive"]] == [TEACHER_MAX_DOC_TOKENS, 1]
    assert [len(t.split()) for t in out["negative_1"]] == [1, TEACHER_MAX_DOC_TOKENS]
