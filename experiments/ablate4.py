"""Round 4 (R7): first experiments with the fp32 master-weight fix, on the archived round-3 data (no rebuild).

Does behaviour data move beats_paired / exec@1 above base, does the dry-run recipe work, and do per-source
weights or base merging help. 120 steps, batch 16, collapse guard on (pilot setting). Key cells run 2 seeds.
"""

from __future__ import annotations

import json
import shutil
import sys
import tarfile
import traceback
from pathlib import Path
from typing import Any

import ablate
from ablate import GTE, ROOT, DryRun, evaluate, log, results, upload
from ablate2 import without
from ablate3 import BEHAVIOUR, FULL, PROD, base_in_repo_format, train_eval

DATA_REPO = "divakantpandey/ce3-ablate-data"
ARCHIVE = "dryrun_ablate_r3.tar"
results["round"] = 4
ablate.STEPS = 120
GUARD: dict[str, Any] = dict(collapse_guard=0.5, eval_steps=40)


def restore() -> tuple[Path, Path]:
    from huggingface_hub import hf_hub_download

    tar = hf_hub_download(DATA_REPO, ARCHIVE, repo_type="dataset", local_dir=str(ROOT.parent / "archive"))
    with tarfile.open(tar) as fh:
        # Our own archive; the combined manifest holds absolute paths under ROOT, so restore in place.
        fh.extractall(ROOT.parent, filter="data")
    return ROOT / "mined", ROOT / "combined"


def merge(d: DryRun, name: str, final: Path, base: Path, w: float = 0.5) -> None:
    out = ROOT / "merged" / f"{name}-ft{w:g}"
    key = f"M-{name}-ft{w:g}"
    try:
        d.ce3(f"merge-{key}", "merge", str(final), str(base), "--weights", str(w), str(1 - w), "--output", str(out))
        results["runs"][key] = {"merge_of": name, "ft_weight": w, "eval": evaluate(d, key, str(out))}
    except Exception as e:  # noqa: BLE001
        results["runs"][key] = {"merge_of": name, "ft_weight": w, "error": f"{type(e).__name__}: {str(e)[-1500:]}"}
    log(f"RESULT {key}: {json.dumps(results['runs'][key].get('eval') or results['runs'][key].get('error'))}")
    upload()


def main() -> None:
    if ROOT.exists():
        shutil.rmtree(ROOT)
    d = DryRun(root=ROOT, hub_user=None, wandb=True, rows=300, teacher="Qwen/Qwen3-Reranker-4B")
    try:
        mined, combined = restore()
        results["base"] = evaluate(d, "base", GTE)
        upload()
        no_twins = without(combined, "behaviour_swe_twins")
        finals: dict[str, Path] = {}

        def cell(name: str, data: Path, keep: bool = False, **stage: Any) -> None:
            final = train_eval(d, name, data, FULL, keep=keep, **GUARD, **stage)
            if final is not None:
                finals[name] = final

        # Seed pairs on the central contrast: does adding behaviour data change the target metrics?
        for seed in (42, 43):
            cell(f"replay-plain-s{seed}", mined, keep=seed == 42, seed=seed)
            cell(f"combined-plain-s{seed}", combined, keep=seed == 42, seed=seed)
        cell("combined-prod", combined, keep=True, **PROD)
        cell("replay-prod", mined, **PROD)
        cell("no-twins-plain", no_twins)
        cell("combined-behaviour-x0.25", combined, source_loss_weights={s: 0.25 for s in BEHAVIOUR})
        cell("combined-behaviour-x3", combined, source_loss_weights={s: 3.0 for s in BEHAVIOUR})

        base = base_in_repo_format()
        for name in ("combined-plain-s42", "combined-prod", "replay-plain-s42"):
            if name in finals:
                merge(d, name, finals[name], base)
    except Exception:  # noqa: BLE001 - top-level boundary: log, upload what exists
        results["fatal"] = traceback.format_exc()[-3000:]
        log(f"FATAL\n{results['fatal']}")
    results["done"] = True
    upload()
    log(f"DONE {json.dumps({n: r.get('eval', r.get('error')) for n, r in results['runs'].items()})}")


if __name__ == "__main__":
    sys.exit(main())
