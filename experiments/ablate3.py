"""Round 3: retest with replay that actually fills batches (4 mined negatives, twins capped at 300).

Every run now fails loudly on sampler starvation (TrainingShortfall). Tests replay-only fine-tuning, the
diluted behaviour mix, per-source down-weighting, the collapse guard, and base/fine-tune weight averaging.
The built data is archived to a private HF dataset so later rounds skip the ~25 min rebuild.
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
from ablate import GTE, ROOT, DryRun, build, data_check, evaluate, log, model_yaml, results, upload
from ablate2 import without

DATA_REPO = "divakantpandey/ce3-ablate-data"
ARCHIVE = "dryrun_ablate_r3.tar"
BEHAVIOUR = ["behaviour_swe_twins", "behaviour_nl2code", "behaviour_code2code"]
results["round"] = 3
FULL = {"attn_implementation": "auto"}
PLAIN: dict[str, Any] = dict(temperature=0.05, hardness_mode=None, hardness_strength=0.0, warmup_ratio=0.2)
PROD: dict[str, Any] = dict(temperature=0.015, hardness_mode="hard_negatives", hardness_strength=5.0, warmup_ratio=0.05)


def archive_data() -> None:
    from huggingface_hub import HfApi

    tar = ROOT.parent / ARCHIVE
    with tarfile.open(tar, "w") as fh:
        for name in ("data", "mined", "behaviour", "combined", "configs"):
            fh.add(ROOT / name, arcname=f"{ROOT.name}/{name}")
    api = HfApi()
    api.create_repo(DATA_REPO, repo_type="dataset", private=True, exist_ok=True)
    api.upload_file(path_or_fileobj=str(tar), path_in_repo=ARCHIVE, repo_id=DATA_REPO, repo_type="dataset")
    log(f"archived built data to {DATA_REPO}/{ARCHIVE} ({tar.stat().st_size / 1e6:.0f} MB)")


def train_eval(d: DryRun, name: str, data_dir: Path, model: dict[str, Any], keep: bool = False, **stage: Any) -> Path | None:
    overrides: dict[str, Any] = dict(
        data_dir=str(data_dir), dev_suite=str(data_dir), batch_size=16, max_seq_length=256, max_steps=ablate.STEPS,
        save_steps=10_000, eval_steps=10_000, learning_rate=2e-5, model=model_yaml(name, **model), **PLAIN,
    )
    overrides.update(stage)
    record: dict[str, Any] = {"stage": {k: v for k, v in overrides.items() if k not in ("data_dir", "dev_suite", "model")},
                              "data": data_dir.name, "model": model}
    final = ROOT / "runs" / name / "final"
    try:
        cfg = d.stage(name, "150m_ft_behaviour.yaml", **overrides)
        out = d.ce3(f"train-{name}", "train", str(cfg))
        record["steps_line"] = next((ln for ln in out.splitlines()[::-1] if "train_runtime" in ln), "")[-300:]
        record["eval"] = evaluate(d, name, str(final))
    except Exception as e:  # noqa: BLE001 - record and keep going with the other cells
        record["error"] = f"{type(e).__name__}: {str(e)[-1500:]}"
    if not keep:
        shutil.rmtree(ROOT / "runs" / name, ignore_errors=True)
    results["runs"][name] = record
    log(f"RESULT {name}: {json.dumps(record.get('eval') or record.get('error'))}")
    upload()
    return final if keep and "eval" in record else None


def base_in_repo_format() -> Path:
    """Base GTE with this repo's module stack (adds Normalize + prompts) so mergekit sees matching configs."""
    from code_embed_v3.config import ModelConfig
    from code_embed_v3.modeling import build_model, save_model
    from code_embed_v3.paths import resolve

    cfg = ModelConfig.from_yaml(resolve("configs/models/gte-modernbert-150m.yaml"))
    cfg.max_seq_length = 256
    out = ROOT / "base_repo_format"
    save_model(build_model(cfg, device="cpu"), out, cfg)
    return out


def merges(d: DryRun, candidates: dict[str, Path]) -> None:
    if not candidates:
        log("no healthy candidate to merge")
        return
    best = max(candidates, key=lambda n: results["runs"][n]["eval"]["exec@10"])
    base = base_in_repo_format()
    for w in (0.5, 0.7):
        name = f"M-{best}-ft{w:g}"
        out = ROOT / "merged" / name
        try:
            d.ce3(f"merge-{name}", "merge", str(candidates[best]), str(base), "--weights", str(w), str(1 - w), "--output", str(out))
            results["runs"][name] = {"merge_of": best, "ft_weight": w, "eval": evaluate(d, name, str(out))}
        except Exception as e:  # noqa: BLE001
            results["runs"][name] = {"merge_of": best, "ft_weight": w, "error": f"{type(e).__name__}: {str(e)[-1500:]}"}
        log(f"RESULT {name}: {json.dumps(results['runs'][name].get('eval') or results['runs'][name].get('error'))}")
        upload()


def main() -> None:
    if ROOT.exists():
        shutil.rmtree(ROOT)
    d = DryRun(root=ROOT, hub_user=None, wandb=True, rows=300, teacher="Qwen/Qwen3-Reranker-4B")
    try:
        mined, combined = build(d)
        data_check(combined)
        try:
            archive_data()
        except Exception as e:  # noqa: BLE001 - an archive failure must not cost the runs
            log(f"archive failed: {e}")
        results["base"] = evaluate(d, "base", GTE)
        upload()
        no_twins = without(combined, "behaviour_swe_twins")

        healthy: dict[str, Path] = {}

        def keep(name: str, final: Path | None) -> None:
            if final is not None and results["runs"][name]["eval"]["exec@10"] >= 0.5 * results["base"]["exec@10"]:
                healthy[name] = final

        # Replay alone: is plain fine-tuning on the replay buffer itself safe, at pilot lr and lower?
        keep("A3-replay-lr2e-5", train_eval(d, "A3-replay-lr2e-5", mined, FULL, keep=True))
        keep("A3-replay-lr5e-6", train_eval(d, "A3-replay-lr5e-6", mined, FULL, keep=True, learning_rate=5e-6))
        # Behaviour diluted by replay that now fills batches.
        keep("C4-combined-lr2e-5", train_eval(d, "C4-combined-lr2e-5", combined, FULL, keep=True))
        keep("C4-combined-lr5e-6", train_eval(d, "C4-combined-lr5e-6", combined, FULL, keep=True, learning_rate=5e-6))
        keep("N4-no-twins-lr2e-5", train_eval(d, "N4-no-twins-lr2e-5", no_twins, FULL, keep=True))
        weights = {s: 0.25 for s in BEHAVIOUR}
        keep("W4-behaviour-x0.25", train_eval(d, "W4-behaviour-x0.25", combined, FULL, keep=True, source_loss_weights=weights))
        # Guard check: the dry-run recipe is known to collapse; the guard should stop it early and refuse to save.
        train_eval(d, "G4-prod-guarded", combined, FULL, collapse_guard=0.5, eval_steps=10, **PROD)
        merges(d, healthy)
    except Exception:  # noqa: BLE001 - top-level boundary: log, upload what exists
        results["fatal"] = traceback.format_exc()[-3000:]
        log(f"FATAL\n{results['fatal']}")
    results["done"] = True
    upload()
    log(f"DONE {json.dumps({n: r.get('eval', r.get('error')) for n, r in results['runs'].items()})}")


if __name__ == "__main__":
    sys.exit(main())
