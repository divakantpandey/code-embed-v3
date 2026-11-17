"""Round 5 (R8): the full end-to-end dry-run with every fix, then the behaviour question at 10x scale.

Part 1 runs `ce3 dry-run` exactly as `make dry-run` does (label, train-s3, mteb, release included) and uploads its
logs and reports. Part 2 rebuilds the mix at 3000 rows/source with 400 CodeContests problems and 3000 SWE twins
(10x round 3/4) and trains 1200 steps at batch 16 (10x round 4): does behaviour data move exec@1 at this scale?
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import ablate
from ablate import GTE, ROOT, DRY_TASKS, DryRun, evaluate, log, resolve, results, upload
from ablate3 import FULL, PROD, base_in_repo_format, train_eval
from ablate4 import GUARD, merge

REPO = ablate.REPO
DRY_ROOT = Path("/root/dryrun")
ROWS = 3000
results["round"] = 5
ablate.STEPS = 1200
GUARD["eval_steps"] = 200


def dry_run() -> None:
    log("dry-run: start")
    t0 = time.time()
    proc = subprocess.run(["uv", "run", "ce3", "dry-run", "--root", str(DRY_ROOT)], capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    (DRY_ROOT).mkdir(parents=True, exist_ok=True)
    (DRY_ROOT / "dry_run_stdout.log").write_text(out)
    results["dry_run"] = {"exit": proc.returncode, "minutes": round((time.time() - t0) / 60, 1), "tail": out[-3000:]}
    report = DRY_ROOT / "dry_run_report.json"
    if report.exists():
        results["dry_run"]["report"] = json.loads(report.read_text())
    log(f"dry-run: exit {proc.returncode} in {results['dry_run']['minutes']} min")
    print(out[-4000:], flush=True)
    try:
        from huggingface_hub import HfApi

        HfApi().upload_folder(
            folder_path=str(DRY_ROOT), repo_id=REPO, path_in_repo=f"dryrun/{ablate.STAMP}",
            allow_patterns=["*.json", "*.log", "logs/*.log", "mteb/**/*.json"], commit_message=f"dryrun {ablate.STAMP}",
        )
    except Exception as e:  # noqa: BLE001 - the 10x part must still run
        log(f"dry-run upload failed: {e}")
    upload()
    # Free the disk and GPU caches the 10x part needs.
    shutil.rmtree(DRY_ROOT, ignore_errors=True)


def build10(d: DryRun) -> tuple[Path, Path]:
    data, mined, beh, combined = ROOT / "data", ROOT / "mined", ROOT / "behaviour", ROOT / "combined"
    d.ce3("build-data", "build-data", str(resolve("configs/data/mix_v1.yaml")), "--output-dir", str(data), "--max-rows", str(ROWS))
    seeds = [str(data / "cornstack_python"), str(data / "cornstack_java")]
    for task in ("nl2code", "techqa", "code_edit", "multiturn"):
        d.ce3(f"synth-{task}", "synth", task, "--seed-dirs", *seeds, "--rows", "24", "--oversample", "2",
              "--out", str(ROOT / "synth" / f"{task}.parquet"), "--push-repo", "")
    d.ce3("build-synth", "build-data", str(d.synth_config()))
    d.ce3("audit", "audit", str(data), "--tasks", *DRY_TASKS)
    d.ce3("mine", "mine", str(data), "--model", GTE, "--no-query-prompts", "--num-negatives", "4",
          "--output-dir", str(mined), "--batch-size", "128")
    d.ce3("behaviour-build", "behaviour-build", "--out-dir", str(beh), "--max-problems", "400", "--mutants", "3",
          "--rewrites", "1", "--negatives", "1", "--repeats", "1", "--swe-smith-rows", str(ROWS), "--push-repo", "")
    results["behaviour_stats"] = json.loads((beh / "stats.json").read_text())
    d.ce3("audit-behaviour", "audit", str(beh), "--tasks", *DRY_TASKS)
    d.ce3("behaviour-combine", "behaviour-combine", str(mined), str(beh), "--output-dir", str(combined))
    from code_embed_v3.data.build import read_manifest

    results["sources"] = {n: m.get("rows", m.get("train_rows")) for n, m in read_manifest(combined)["sources"].items()}
    log(f"sources: {results['sources']}")
    upload()
    return mined, combined


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    if os.environ.get("R5_PART") == "seeds":
        import ablate6

        ablate6.main()
        return
    if os.environ.get("R5_PART") == "parity":
        import ablate7

        ablate7.main()
        return
    try:
        dry_run()
    except Exception:  # noqa: BLE001
        results["dry_run_fatal"] = traceback.format_exc()[-3000:]
        log(f"dry-run FATAL\n{results['dry_run_fatal']}")
        upload()
    if os.environ.get("R5_PART") == "dryrun":
        results["done"] = True
        upload()
        return
    shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True)
    d = DryRun(root=ROOT, hub_user=None, wandb=True, rows=ROWS, teacher="Qwen/Qwen3-Reranker-4B")
    try:
        mined, combined = build10(d)
        results["base"] = evaluate(d, "base", GTE)
        upload()
        finals: dict[str, Path] = {}

        def cell(name: str, data: Path, **stage: Any) -> None:
            final = train_eval(d, name, data, FULL, keep=True, **GUARD, **stage)
            if final is not None:
                finals[name] = final

        cell("x10-replay-plain", mined)
        cell("x10-combined-plain", combined)
        cell("x10-combined-prod", combined, **PROD)
        cell("x10-replay-prod", mined, **PROD)
        base = base_in_repo_format()
        for name in ("x10-combined-plain", "x10-combined-prod"):
            if name in finals:
                merge(d, name, finals[name], base)
    except Exception:  # noqa: BLE001 - top-level boundary: log, upload what exists
        results["fatal"] = traceback.format_exc()[-3000:]
        log(f"FATAL\n{results['fatal']}")
    results["done"] = True
    upload()
    log(f"DONE {json.dumps({n: r.get('eval', r.get('error')) for n, r in results['runs'].items()})}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.exit(main())
