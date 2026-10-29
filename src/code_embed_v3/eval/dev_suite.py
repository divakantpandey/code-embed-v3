"""Fast per-family dev suite built from held-out rows of the training sources.

Never touches benchmark test splits, so it is safe to use for checkpoint selection and ablations.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from datasets import Dataset, load_from_disk
from sentence_transformers.sentence_transformer.evaluation import (
    InformationRetrievalEvaluator,
    SequentialEvaluator,
)

from code_embed_v3.data.build import DEV_DIR, read_manifest
from code_embed_v3.prompts import QUERY_PROMPTS, Family

if TYPE_CHECKING:
    from sentence_transformers.base.modality_types import SingleInput


def build_dev_evaluator(
    data_dir: str | Path, max_queries_per_family: int = 1000, batch_size: int = 64
) -> SequentialEvaluator:
    """One IR evaluator per family; the corpus is every dev positive in that family, so the
    other sources' positives act as distractors."""
    data_dir = Path(data_dir)
    manifest = read_manifest(data_dir)
    by_family: dict[Family, list[Dataset]] = defaultdict(list)
    for name, meta in manifest["sources"].items():
        path = data_dir / DEV_DIR / name
        if meta.get("dev_rows") and path.exists():
            ds = load_from_disk(str(path))
            if isinstance(ds, Dataset):
                by_family[meta["family"]].append(ds)

    evaluators = []
    for family, parts in sorted(by_family.items()):
        queries: dict[str, SingleInput] = {}
        corpus: dict[str, SingleInput] = {}
        relevant: dict[str, set[str]] = {}
        doc_ids: dict[str, str] = {}
        for p_idx, ds in enumerate(parts):
            for r_idx, row in enumerate(ds):
                doc_id = doc_ids.setdefault(row["positive"], f"d{len(doc_ids)}")
                corpus[doc_id] = row["positive"]
                if len(queries) < max_queries_per_family:
                    qid = f"q{p_idx}_{r_idx}"
                    queries[qid] = row["anchor"]
                    relevant[qid] = {doc_id}
        evaluators.append(
            InformationRetrievalEvaluator(
                queries=queries,
                corpus=corpus,
                relevant_docs=relevant,
                name=f"dev-{family}",
                query_prompt=QUERY_PROMPTS[family],
                batch_size=batch_size,
                write_csv=False,
            )
        )
    if not evaluators:
        raise RuntimeError(f"no dev holdouts under {data_dir / DEV_DIR}")
    # Macro-average across families so one large family cannot dominate checkpoint selection.
    return SequentialEvaluator(evaluators, main_score_function=lambda scores: sum(scores) / len(scores))
