"""Round 6 (R9): seed replicates of the 10x behaviour effect under the prod recipe.

R8 measured combined-prod vs replay-prod once (exec@1 +0.033, beats_paired +0.034). This rebuilds the 10x mix
(R8's was on wiped container disk), archives it to the private data repo so later rounds skip the rebuild, and
trains both cells with seeds 42, 43, 44 on that one build. Seed 42 vs R8's seed 42 also shows rebuild variance.
"""

from __future__ import annotations

import json
import shutil
import tarfile
import traceback
from pathlib import Path

import ablate
import ablate5
from ablate import GTE, ROOT, DryRun, evaluate, log, results, upload
from ablate3 import FULL, PROD, train_eval
from ablate4 import GUARD

DATA_REPO = "divakantpandey/ce3-ablate-data"
ARCHIVE = "dryrun_ablate_x10.tar"
SEEDS = (42, 43, 44)


def archive_data() -> None:
    from huggingface_hub import HfApi

    tar = ROOT.parent / ARCHIVE
    with tarfile.open(tar, "w") as fh:
        for name in ("data", "mined", "behaviour", "combined", "configs"):
            fh.add(ROOT / name, arcname=f"{ROOT.name}/{name}")
    HfApi().upload_file(path_or_fileobj=str(tar), path_in_repo=ARCHIVE, repo_id=DATA_REPO, repo_type="dataset")
    log(f"archived 10x data to {DATA_REPO}/{ARCHIVE} ({tar.stat().st_size / 1e6:.0f} MB)")
    tar.unlink()


def label_check(d: DryRun, mined: Path) -> None:
    """Re-test vLLM teacher parity with both backends clipped to the same window (dry-run attempt 3 failed it)."""
    import re

    try:
        out = d.ce3("label-check", "label-check", str(mined), "--teacher", d.teacher, "--rows", "64",
                    "--work-dir", str(ROOT / "label-check"))
        results["label_check"] = {"passed": True}
    except Exception as e:  # noqa: BLE001 - record the per-source numbers either way
        out = (ROOT / "logs" / "label-check.log").read_text()
        results["label_check"] = {"passed": False, "error": str(e)[-300:]}
    results["label_check"]["spearman"] = {
        m[1]: float(m[2]) for m in re.finditer(r"^(\w+): \{'spearman': ([0-9.]+)", out, re.M)
    }
    log(f"label-check: {json.dumps(results['label_check'])}")
    upload()


def main() -> None:
    results["round"] = 6
    ablate.STEPS = 1200
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True)
    d = DryRun(root=ROOT, hub_user=None, wandb=True, rows=ablate5.ROWS, teacher="Qwen/Qwen3-Reranker-4B")
    try:
        mined, combined = ablate5.build10(d)
        try:
            archive_data()
        except Exception as e:  # noqa: BLE001 - an archive failure must not cost the runs
            log(f"archive failed: {e}")
        results["base"] = evaluate(d, "base", GTE)
        upload()
        for seed in SEEDS:
            # keep=False: only the scores matter here, and 6 checkpoints would fill the container disk.
            train_eval(d, f"x10-combined-prod-s{seed}", combined, FULL, seed=seed, **GUARD, **PROD)
            train_eval(d, f"x10-replay-prod-s{seed}", mined, FULL, seed=seed, **GUARD, **PROD)
        label_check(d, mined)
    except Exception:  # noqa: BLE001 - top-level boundary: log, upload what exists
        results["fatal"] = traceback.format_exc()[-3000:]
        log(f"FATAL\n{results['fatal']}")
    results["done"] = True
    upload()
    log(f"DONE {json.dumps({n: r.get('eval', r.get('error')) for n, r in results['runs'].items()})}")
