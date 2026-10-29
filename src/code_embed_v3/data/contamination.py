"""Train/test contamination audit against the mteb benchmarks we report on.

Only test *queries* and the documents *relevant to them* are indexed: several CoIR corpora (e.g.
CosQA) also contain train-partition documents, and indexing the whole corpus would flag legitimate
training data.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from datasets import Dataset
from datasketch import MinHash, MinHashLSH

logger = logging.getLogger(__name__)

_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"\w+|[^\w\s]")


def normalise(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


def text_of(value: Any) -> str:
    """mteb queries are usually str, but multi-turn tasks hold lists of turns."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(text_of(v) for k, v in value.items() if k in ("title", "text", "content"))
    if isinstance(value, Iterable):
        return " ".join(text_of(v) for v in value)
    return str(value)


def exact_key(text: str) -> str:
    return hashlib.blake2b(normalise(text).encode(), digest_size=16).hexdigest()


def minhash(text: str, num_perm: int, ngram: int) -> MinHash | None:
    tokens = _TOKEN.findall(normalise(text))
    if len(tokens) < ngram:
        return None
    mh = MinHash(num_perm=num_perm)
    mh.update_batch([" ".join(tokens[i : i + ngram]).encode() for i in range(len(tokens) - ngram + 1)])
    return mh


class EvalIndex:
    """Exact + MinHash-LSH index over eval texts, tagged with the task each text came from."""

    def __init__(self, threshold: float = 0.8, num_perm: int = 128, ngram: int = 5) -> None:
        self.threshold, self.num_perm, self.ngram = threshold, num_perm, ngram
        self.exact: dict[str, str] = {}
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self._n = 0

    def add(self, task: str, text: str) -> None:
        self.exact.setdefault(exact_key(text), task)
        mh = minhash(text, self.num_perm, self.ngram)
        if mh is not None:
            self.lsh.insert(f"{task}::{self._n}", mh)
            self._n += 1

    def __len__(self) -> int:
        return len(self.exact)

    def hits(self, text: str) -> set[str]:
        """Eval tasks this text exactly or near-duplicates."""
        found = set()
        if (task := self.exact.get(exact_key(text))) is not None:
            found.add(task)
        mh = minhash(text, self.num_perm, self.ngram)
        if mh is not None:
            found.update(key.split("::", 1)[0] for key in self.lsh.query(mh))
        return found


def index_mteb_tasks(task_names: list[str], index: EvalIndex | None = None) -> EvalIndex:
    import mteb

    index = index or EvalIndex()
    for name in task_names:
        task = mteb.get_task(name)
        task.load_data()
        if task.dataset is None:
            raise RuntimeError(f"{name}: mteb loaded no data")
        for subset in task.dataset.values():
            for split_name in task.metadata.eval_splits:
                split = subset.get(split_name)
                if not split or "queries" not in split:
                    logger.warning("%s/%s has no retrieval layout; skipped", name, split_name)
                    continue
                relevant = split.get("relevant_docs") or {}
                wanted_docs = {d for docs in relevant.values() for d in docs}
                for q in split["queries"]:
                    index.add(name, text_of(q["text"]))
                for d in split["corpus"]:
                    if d["id"] in wanted_docs:
                        index.add(name, text_of(d.get("text", "")))
        logger.info("indexed %s (%d unique eval texts so far)", name, len(index))
    return index


def audit_dataset(ds: Dataset, index: EvalIndex, num_proc: int = 1) -> tuple[Dataset, Counter[str]]:
    """Drop every row whose any text column hits the eval index; return kept rows and hits per task."""
    text_cols = [c for c in ds.column_names if c == "anchor" or c == "positive" or c.startswith("negative_")]

    def flag(row: dict[str, Any]) -> dict[str, Any]:
        tasks: set[str] = set()
        for c in text_cols:
            tasks |= index.hits(row[c])
        return {"_hits": sorted(tasks)}

    flagged = ds.map(flag, num_proc=num_proc, desc="contamination audit")
    counts: Counter[str] = Counter(t for hits in flagged["_hits"] for t in hits)
    kept = flagged.filter(lambda r: not r["_hits"], num_proc=num_proc).remove_columns("_hits")
    return kept, counts


def write_report(path: str | Path, report: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(report, indent=2, sort_keys=True))
