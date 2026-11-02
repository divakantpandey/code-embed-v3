"""Paired significance of a candidate against a baseline on the behaviour benchmarks.

Both models answer the same queries, so tests are paired: McNemar's exact test on binary per-query
ExecRetrieval outcomes, and a bootstrap over queries for every metric (CoQuIR pairs of one query share its
embedding and are not independent, so queries are resampled, not pairs). Inputs are ``behaviour-eval``
JSONs carrying ``per_query``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Any

import numpy as np

EXEC_METRICS = ("exec@1", "exec@10", "beats_paired_distractors")


@dataclass(frozen=True)
class Comparison:
    metric: str
    baseline: float
    candidate: float
    ci_low: float
    ci_high: float
    p_bootstrap: float
    # Only for binary per-query outcomes.
    p_mcnemar: float | None = None
    wins: int | None = None
    losses: int | None = None

    @property
    def diff(self) -> float:
        return self.candidate - self.baseline


def mcnemar_exact(wins: int, losses: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n
    return min(1.0, 2 * tail)


def _bootstrap(stat: Any, n: int, n_boot: int, rng: np.random.Generator, chunk: int = 256) -> np.ndarray:
    """``stat(idx)`` maps a (chunk, n) array of resampled row indices to one value per row."""
    out = []
    for start in range(0, n_boot, chunk):
        out.append(stat(rng.integers(0, n, size=(min(chunk, n_boot - start), n))))
    return np.concatenate(out)


def _summarise(metric: str, base: float, cand: float, boot: np.ndarray, **extra: Any) -> Comparison:
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # Two-sided: how often resampling puts the difference on the other side of zero.
    p = min(1.0, 2 * min(float(np.mean(boot <= 0)), float(np.mean(boot >= 0))))
    return Comparison(metric, base, cand, float(lo), float(hi), p, **extra)


def compare(
    baseline: dict[str, Any], candidate: dict[str, Any], n_boot: int = 10_000, seed: int = 0
) -> list[Comparison]:
    rng = np.random.default_rng(seed)
    bq, cq = baseline["per_query"], candidate["per_query"]
    rows: list[Comparison] = []
    for metric in EXEC_METRICS:
        b = np.asarray(bq["ExecRetrieval"][metric], dtype=np.int8)
        c = np.asarray(cq["ExecRetrieval"][metric], dtype=np.int8)
        if b.shape != c.shape:
            raise ValueError(f"ExecRetrieval {metric}: {b.size} vs {c.size} queries")
        delta = (c - b).astype(np.float64)
        wins, losses = int(np.sum(delta > 0)), int(np.sum(delta < 0))
        boot = _bootstrap(lambda idx, d=delta: d[idx].mean(axis=1), b.size, n_boot, rng)
        rows.append(
            _summarise(
                f"ExecRetrieval {metric}",
                float(b.mean()),
                float(c.mean()),
                boot,
                p_mcnemar=mcnemar_exact(wins, losses),
                wins=wins,
                losses=losses,
            )
        )

    subsets = sorted(k for k in bq if k.startswith("CoQuIR-") and k in cq)
    pairwise_boots, pairwise_base, pairwise_cand = [], [], []
    for subset in subsets:
        b, c = bq[subset], cq[subset]
        if b["qid"] != c["qid"]:
            raise ValueError(f"{subset}: baseline and candidate scored different queries")
        pairs = np.asarray(b["pairs"], dtype=np.float64)
        if pairs.sum() == 0:
            continue
        hb = np.asarray(b["pair_hits"], dtype=np.float64)
        hc = np.asarray(c["pair_hits"], dtype=np.float64)
        boot = _bootstrap(
            lambda idx, d=hc - hb, p=pairs: d[idx].sum(axis=1) / np.maximum(p[idx].sum(axis=1), 1),
            pairs.size,
            n_boot,
            rng,
        )
        base_pw, cand_pw = float(hb.sum() / pairs.sum()), float(hc.sum() / pairs.sum())
        rows.append(_summarise(f"{subset} pairwise", base_pw, cand_pw, boot))
        pairwise_boots.append(boot)
        pairwise_base.append(base_pw)
        pairwise_cand.append(cand_pw)

        keep = [i for i, x in enumerate(b["ndcg@10"]) if x is not None]
        nb = np.asarray([b["ndcg@10"][i] for i in keep])
        nc = np.asarray([c["ndcg@10"][i] for i in keep])
        boot = _bootstrap(lambda idx, d=nc - nb: d[idx].mean(axis=1), nb.size, n_boot, rng)
        rows.append(_summarise(f"{subset} ndcg@10", float(nb.mean()), float(nc.mean()), boot))

    if pairwise_boots:
        # Subsets are disjoint query sets resampled independently, so their bootstrap draws average directly.
        rows.append(
            _summarise(
                "CoQuIR pairwise mean",
                float(np.mean(pairwise_base)),
                float(np.mean(pairwise_cand)),
                np.mean(pairwise_boots, axis=0),
            )
        )
    return rows
