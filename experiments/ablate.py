"""Collapse bisection for the 150m_ft_behaviour recipe (DRYRUN_FINDINGS.md P0). Runs from the repo root on a GPU pod.

Rebuilds the dry-run mix (same steps and sizes), checks the twin data, then trains short runs that change one
factor at a time and scores each with behaviour-eval against the base model. Results go to the private
src repo under ablate/<stamp>/ because pod disk is not reachable from outside.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import yaml

from code_embed_v3.config import load_yaml
from code_embed_v3.dry_run import DRY_TASKS, GTE, DryRun
from code_embed_v3.paths import resolve

ROOT = Path(os.environ.get("ABLATE_ROOT", "/root/dryrun_ablate"))
REPO = "divakantpandey/code-embed-v3-src"
STAMP = time.strftime("%Y%m%d-%H%M%S")
STEPS = 60
results: dict[str, Any] = {"stamp": STAMP, "steps": STEPS, "runs": {}}


def log(msg: str) -> None:
    print(f"[ablate] {msg}", flush=True)


def upload() -> None:
    from huggingface_hub import HfApi

    (ROOT / "ablate_results.json").write_text(json.dumps(results, indent=2))
    try:
        HfApi().upload_folder(
            folder_path=str(ROOT),
            repo_id=REPO,
            path_in_repo=f"ablate/{STAMP}",
            allow_patterns=["ablate_results.json", "logs/*.log", "data_check.json"],
            commit_message=f"ablate {STAMP}",
        )
    except Exception as e:  # noqa: BLE001 - an upload failure must not kill the remaining runs
        log(f"upload failed: {e}")


def build(d: DryRun) -> tuple[Path, Path]:
    data, mined = ROOT / "data", ROOT / "mined"
    d.ce3("build-data", "build-data", str(resolve("configs/data/mix_v1.yaml")), "--output-dir", str(data), "--max-rows", str(d.rows))
    seeds = [str(data / "cornstack_python"), str(data / "cornstack_java")]
    for task in ("nl2code", "techqa", "code_edit", "multiturn"):
        d.ce3(f"synth-{task}", "synth", task, "--seed-dirs", *seeds, "--rows", "24", "--oversample", "2",
              "--out", str(ROOT / "synth" / f"{task}.parquet"), "--push-repo", "")
    d.ce3("build-synth", "build-data", str(d.synth_config()))
    d.ce3("audit", "audit", str(data), "--tasks", *DRY_TASKS)
    d.ce3("mine", "mine", str(data), "--model", GTE, "--no-query-prompts", "--num-negatives", "4", "--output-dir", str(mined), "--batch-size", "128")
    return mined, d.behaviour(data, mined, "")


def data_check(combined: Path) -> None:
    """Per source: exact pos==neg rows, and how often base GTE already ranks negative_1 above the positive."""
    import torch

    from code_embed_v3.data.build import read_manifest, source_dir
    from code_embed_v3.data.mining import load_dataset_dir
    from code_embed_v3.modeling import load_model

    model = load_model(GTE, device="cuda")
    out: dict[str, Any] = {}
    for name, meta in read_manifest(combined)["sources"].items():
        ds = load_dataset_dir(source_dir(combined, name, meta))
        negs = sorted((c for c in ds.column_names if c.startswith("negative_")), key=lambda c: int(c.split("_")[1]))
        row: dict[str, Any] = {"rows": len(ds), "family": meta["family"], "negatives": len(negs)}
        if negs:
            row["neg_equals_pos"] = int(sum(p == n for p, *ns in zip(ds["positive"], *(ds[c] for c in negs)) for n in ns))
            sub = ds.select(range(min(512, len(ds))))
            with torch.no_grad():
                a, p, n = (model.encode(sub[c], batch_size=64, normalize_embeddings=True, convert_to_tensor=True).float() for c in ("anchor", "positive", negs[0]))
            ap, an, pn = (a * p).sum(1), (a * n).sum(1), (p * n).sum(1)
            row["neg1_ge_pos"] = float((an >= ap).float().mean())
            row["cos_pos_neg1_mean"] = float(pn.mean())
            row["cos_pos_neg1_gt_0.99"] = float((pn > 0.99).float().mean())
        out[name] = row
        log(f"data {name}: {row}")
    (ROOT / "data_check.json").write_text(json.dumps(out, indent=2))
    results["data_check"] = out
    del model
    torch.cuda.empty_cache()


def model_yaml(name: str, **overrides: Any) -> str:
    raw = load_yaml(resolve("configs/models/gte-modernbert-150m.yaml"))
    raw.update(overrides)
    path = ROOT / "configs" / f"model-{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw))
    return str(path)


def evaluate(d: DryRun, step: str, model: str) -> dict[str, Any]:
    out = ROOT / "evals" / f"{step}.json"
    d.ce3(f"eval-{step}", "behaviour-eval", model, "--subsets", "Defects4J", "--out", str(out))
    r = json.loads(out.read_text())
    return {"exec@1": r["ExecRetrieval"]["exec@1"], "exec@10": r["ExecRetrieval"]["exec@10"],
            "beats_paired": r["ExecRetrieval"]["beats_paired_distractors"],
            "d4j_pairwise": r["CoQuIR-Defects4J"]["pairwise"], "d4j_ndcg@10": r["CoQuIR-Defects4J"]["ndcg@10"]}


def run(d: DryRun, name: str, data_dir: Path, model: dict[str, Any], **stage: Any) -> None:
    overrides: dict[str, Any] = dict(
        data_dir=str(data_dir), dev_suite=str(data_dir), batch_size=16, max_seq_length=256, max_steps=STEPS,
        warmup_ratio=0.2, save_steps=10_000, eval_steps=10_000, temperature=0.05, hardness_mode=None,
        hardness_strength=0.0, learning_rate=2e-5, model=model_yaml(name, **model),
    )
    overrides.update(stage)
    record: dict[str, Any] = {"stage": {k: v for k, v in overrides.items() if k not in ("data_dir", "dev_suite", "model")},
                              "data": data_dir.name, "model": model}
    try:
        cfg = d.stage(name, "150m_ft_behaviour.yaml", **overrides)
        d.ce3(f"train-{name}", "train", str(cfg))
        final = ROOT / "runs" / name / "final"
        record["eval"] = evaluate(d, name, str(final))
        shutil.rmtree(ROOT / "runs" / name, ignore_errors=True)
    except Exception as e:  # noqa: BLE001 - record and keep going with the other cells
        record["error"] = f"{type(e).__name__}: {str(e)[-1500:]}"
    results["runs"][name] = record
    log(f"RESULT {name}: {json.dumps(record.get('eval') or record.get('error'))}")
    upload()


def main() -> None:
    if ROOT.exists():
        shutil.rmtree(ROOT)
    d = DryRun(root=ROOT, hub_user=None, wandb=True, rows=300, teacher="Qwen/Qwen3-Reranker-4B")
    try:
        mined, combined = build(d)
        data_check(combined)
        results["base"] = evaluate(d, "base", GTE)
        log(f"RESULT base: {json.dumps(results['base'])}")
        upload()

        single = {"matryoshka_dims": [768], "attn_implementation": "sdpa"}
        full = {"attn_implementation": "auto"}
        # P reproduces the dry-run recipe; A-C bisect pipeline vs data; D picks lr x tau on the full setup.
        run(d, "P-prod", combined, full, temperature=0.015, hardness_mode="hard_negatives", hardness_strength=5.0, warmup_ratio=0.05)
        run(d, "A-replay-plain-sdpa", mined, single)
        run(d, "B-replay-plain-flash", mined, {"matryoshka_dims": [768], "attn_implementation": "auto"})
        run(d, "C-combined-plain-sdpa", combined, single)
        grid = [(lr, tau) for lr in (5e-6, 1e-5) for tau in (0.02, 0.05)]
        for lr, tau in grid:
            run(d, f"D-lr{lr:g}-tau{tau:g}", combined, full, learning_rate=lr, temperature=tau)
        scored = {n: r["eval"]["exec@10"] for n, r in results["runs"].items() if n.startswith("D-") and "eval" in r}
        if scored:
            best = max(scored, key=lambda n: scored[n])
            s = results["runs"][best]["stage"]
            run(d, f"E-{best[2:]}-b64", combined, full, learning_rate=s["learning_rate"], temperature=s["temperature"], batch_size=64)
    except Exception:  # noqa: BLE001 - top-level boundary: log, upload what exists
        results["fatal"] = traceback.format_exc()[-3000:]
        log(f"FATAL\n{results['fatal']}")
    upload()
    log(f"DONE {json.dumps({n: r.get('eval', r.get('error')) for n, r in results['runs'].items()})}")


if __name__ == "__main__":
    sys.exit(main())
