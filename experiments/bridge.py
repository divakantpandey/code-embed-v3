"""Round 8 (R15): batch-size bridge before the pilot, plus the first full MTEB(Code, v1) numbers.

The behaviour effect (R8/R9) was measured at batch 16 x 1,200 steps. The pilot trains at batch 128. On the
archived 10x data, train combined and replay-only at batch 64, 128 and 256 with the same 19,200 samples
(prod recipe, seed 42; batch 128 also seed 43), then run MTEB(Code, v1) on base and the batch-128 combined
model. Contest behaviour sources hold ~300 rows here, so at 256 they fill one batch: that is the point.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tarfile
import time
import traceback
from pathlib import Path

import ablate
from ablate import GTE, ROOT, DryRun, evaluate, log, results, upload
from ablate3 import FULL, PROD, train_eval
from ablate4 import GUARD

DATA_REPO = "divakantpandey/ce3-ablate-data"
ARCHIVE = "dryrun_ablate_x10.tar"
SAMPLES = 19_200
CELLS = [(64, 42), (128, 42), (256, 42), (128, 43)]
KEEP = "b128-combined-s42"


def restore() -> tuple[Path, Path]:
    from huggingface_hub import hf_hub_download

    tar = hf_hub_download(DATA_REPO, ARCHIVE, repo_type="dataset", local_dir=str(ROOT.parent / "x10tar"))
    with tarfile.open(tar) as fh:
        fh.extractall(ROOT.parent, filter="data")
    Path(tar).unlink()
    from code_embed_v3.data.build import read_manifest

    combined = ROOT / "combined"
    results["sources"] = {n: m.get("rows") for n, m in read_manifest(combined)["sources"].items()}
    log(f"restored 10x data: {results['sources']}")
    return ROOT / "mined", combined


def mteb(d: DryRun, name: str, model: str, data_dirs: list[str]) -> None:
    out = ROOT / "mteb" / name
    t0 = time.time()
    try:
        extra = ["--data-dirs", *data_dirs, "--base-model", GTE] if data_dirs else []
        d.ce3(f"mteb-{name}", "mteb", model, "--name", f"divakantpandey/ce3-bridge-{name}",
              "--benchmarks", "MTEB(Code, v1)", "--output", str(out), *extra)
        results.setdefault("mteb", {})[name] = {"minutes": round((time.time() - t0) / 60, 1),
                                               "files": [str(p.relative_to(out)) for p in out.rglob("*.json")]}
    except Exception as e:  # noqa: BLE001 - record and keep the other model's run
        results.setdefault("mteb", {})[name] = {"error": f"{type(e).__name__}: {str(e)[-1500:]}"}
    log(f"MTEB {name}: {json.dumps(results['mteb'][name])[:500]}")
    upload()
    from huggingface_hub import HfApi

    try:
        HfApi().upload_folder(folder_path=str(ROOT / "mteb"), repo_id=ablate.REPO,
                              path_in_repo=f"ablate/{ablate.STAMP}/mteb", allow_patterns=["**/*.json"],
                              commit_message=f"bridge mteb {name}")
    except Exception as e:  # noqa: BLE001
        log(f"mteb upload failed: {e}")


def main() -> None:
    results["round"] = 8
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True)
    d = DryRun(root=ROOT, hub_user=None, wandb=True, rows=3000, teacher="Qwen/Qwen3-Reranker-4B")
    try:
        mined, combined = restore()
        results["base"] = evaluate(d, "base", GTE)
        upload()
        final: Path | None = None
        for batch, seed in CELLS:
            steps = SAMPLES // batch
            ablate.STEPS = steps
            guard = {**GUARD, "eval_steps": max(10, steps // 6)}
            for data, path in (("combined", combined), ("replay", mined)):
                name = f"b{batch}-{data}-s{seed}"
                got = train_eval(d, name, path, FULL, keep=name == KEEP, seed=seed, batch_size=batch,
                                 **guard, **PROD)
                if got is not None:
                    final = got
        mteb(d, "base", GTE, [])
        if final is not None:
            mteb(d, KEEP, str(final), [str(ROOT / "data"), str(ROOT / "behaviour")])
    except Exception:  # noqa: BLE001 - top-level boundary: log, upload what exists
        results["fatal"] = traceback.format_exc()[-3000:]
        log(f"FATAL\n{results['fatal']}")
    results["done"] = True
    upload()
    log(f"DONE {json.dumps({n: r.get('eval', r.get('error')) for n, r in results['runs'].items()})}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.exit(main())
