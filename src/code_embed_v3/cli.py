"""``ce3`` command line: build-data -> audit -> train -> mine -> label -> train -> merge -> mteb."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from datasets import load_from_disk


def _cmd_build_data(a: argparse.Namespace) -> None:
    from dataclasses import replace

    from code_embed_v3.config import DataMixConfig
    from code_embed_v3.data.build import build_mix

    cfg = DataMixConfig.from_yaml(a.config)
    if a.output_dir:
        cfg.output_dir = a.output_dir
    if a.max_rows is not None:
        cfg.sources = [replace(s, max_rows=min(s.max_rows or a.max_rows, a.max_rows)) for s in cfg.sources]
    print(build_mix(cfg, only=a.only))


def _cmd_audit(a: argparse.Namespace) -> None:
    import mteb
    from datasets import Dataset

    from code_embed_v3.data.build import read_manifest, write_manifest
    from code_embed_v3.data.contamination import audit_dataset, index_mteb_tasks, write_report
    from code_embed_v3.env import available_cpus
    from code_embed_v3.eval.run_mteb import DEFAULT_BENCHMARKS

    task_names = sorted(
        set(a.tasks)
        if a.tasks
        else {
            t.metadata.name for b in (a.benchmarks or DEFAULT_BENCHMARKS) for t in mteb.get_benchmark(b).tasks
        }
    )
    manifest = read_manifest(a.data_dir)
    # Re-audit only sources not yet checked against every task requested now.
    pending = {
        n: m
        for n, m in manifest["sources"].items()
        if not set(task_names) <= set(m.get("audited_against", []))
    }
    if not pending:
        print("every source is already audited against these tasks")
        return
    index = index_mteb_tasks(task_names)
    report_path = Path(a.data_dir) / "contamination_report.json"
    report: dict[str, dict[str, int]] = json.loads(report_path.read_text()) if report_path.exists() else {}
    for name, meta in pending.items():
        path = Path(a.data_dir) / name
        ds = load_from_disk(str(path))
        if not isinstance(ds, Dataset):
            raise TypeError(f"{path} is not a single Dataset")
        kept, counts = audit_dataset(ds, index, num_proc=a.num_proc or available_cpus())
        report[name] = {"before": len(ds), "after": len(kept), **dict(counts)}
        tmp = path.with_name(path.name + ".audited")
        kept.save_to_disk(str(tmp))
        del ds
        _replace_dir(tmp, path)
        meta.update(rows=len(kept), audited=True, audited_against=task_names, contamination_hits=dict(counts))
        print(f"{name}: {report[name]}")
        # Per source, so a lost pod does not redo finished sources.
        write_manifest(a.data_dir, manifest)
        write_report(report_path, report)


def _replace_dir(src: Path, dst: Path) -> None:
    import shutil

    backup = dst.with_name(dst.name + ".pre-audit")
    dst.rename(backup)
    src.rename(dst)
    shutil.rmtree(backup)


def _cmd_mine(a: argparse.Namespace) -> None:
    import torch

    from code_embed_v3.data.build import read_manifest, write_manifest
    from code_embed_v3.data.mining import is_saved_dataset, load_dataset_dir, mine, save_dataset_atomic
    from code_embed_v3.modeling import load_model

    model = load_model(a.model, attn_implementation="auto")
    manifest = read_manifest(a.data_dir)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_manifest: dict[str, Any] = read_manifest(out) if (out / "manifest.json").exists() else {"sources": {}}
    for name, meta in manifest["sources"].items():
        if is_saved_dataset(out / name) and name in out_manifest["sources"]:
            print(f"{name}: already mined")
            continue
        ds = load_dataset_dir(Path(a.data_dir) / name)
        if a.max_rows_per_source is not None:
            ds = ds.select(range(min(a.max_rows_per_source, len(ds))))
        mined = mine(
            ds,
            model,
            meta["family"],
            num_negatives=a.num_negatives,
            relative_margin=a.relative_margin,
            batch_size=a.batch_size,
            use_multi_process=torch.cuda.device_count() > 1,
            query_prompts=a.query_prompts,
        )
        save_dataset_atomic(mined, out / name)
        out_manifest["sources"][name] = {**meta, "rows": len(mined), "mined_with": a.model}
        # Written per source so a restart skips everything already mined.
        write_manifest(out, out_manifest)
    _copy_dev(Path(a.data_dir), out)


def _teacher_scorer(a: argparse.Namespace) -> Any:
    import torch

    from code_embed_v3.data.mining import crossencoder_scorer, vllm_scorer

    if a.backend == "vllm":
        return vllm_scorer(a.teacher, num_gpus=max(1, torch.cuda.device_count()))
    from sentence_transformers import CrossEncoder

    return crossencoder_scorer(CrossEncoder(a.teacher), batch_size=a.batch_size)


def _cmd_label(a: argparse.Namespace) -> None:
    from code_embed_v3.data.build import read_manifest, write_manifest
    from code_embed_v3.data.mining import (
        finish_shards,
        is_saved_dataset,
        label_source,
        load_dataset_dir,
        save_dataset_atomic,
    )

    scorer = _teacher_scorer(a)
    manifest = read_manifest(a.data_dir)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_manifest: dict[str, Any] = read_manifest(out) if (out / "manifest.json").exists() else {"sources": {}}
    for name, meta in manifest["sources"].items():
        if is_saved_dataset(out / name) and name in out_manifest["sources"]:
            print(f"{name}: already labelled")
            continue
        ds = load_dataset_dir(Path(a.data_dir) / name)
        if a.max_rows_per_source is not None:
            ds = ds.select(range(min(a.max_rows_per_source, len(ds))))
        labelled = label_source(ds, scorer, meta["family"], out / name, shard_rows=a.shard_rows)
        save_dataset_atomic(labelled, out / name)
        finish_shards(out / name)
        out_manifest["sources"][name] = {
            **meta,
            "rows": len(labelled),
            "teacher": a.teacher,
            "teacher_backend": a.backend,
            "teacher_score": "log-odds",
        }
        write_manifest(out, out_manifest)
    _copy_dev(Path(a.data_dir), out)


def _cmd_label_check(a: argparse.Namespace) -> None:
    """Score the same sample with both backends; vLLM is only trusted if it ranks like the reference."""
    import shutil

    import numpy as np
    import torch
    from scipy.stats import spearmanr
    from sentence_transformers import CrossEncoder

    from code_embed_v3.data.build import read_manifest
    from code_embed_v3.data.mining import crossencoder_scorer, label_source, load_dataset_dir, vllm_scorer

    work = Path(a.work_dir)
    reference = crossencoder_scorer(CrossEncoder(a.teacher), batch_size=a.batch_size)
    fast = vllm_scorer(a.teacher, num_gpus=max(1, torch.cuda.device_count()))
    report: dict[str, dict[str, float]] = {}
    for name, meta in read_manifest(a.data_dir)["sources"].items():
        ds = load_dataset_dir(Path(a.data_dir) / name)
        ds = ds.select(range(min(a.rows, len(ds))))
        ref = np.asarray(label_source(ds, reference, meta["family"], work / "ref" / name)["label"])
        vll = np.asarray(label_source(ds, fast, meta["family"], work / "vllm" / name)["label"])
        rho = float(spearmanr(ref.ravel(), vll.ravel()).statistic)
        report[name] = {"spearman": rho, "max_abs_diff": float(np.abs(ref - vll).max())}
        print(f"{name}: {report[name]}")
    shutil.rmtree(work, ignore_errors=True)
    worst = min(r["spearman"] for r in report.values())
    if worst < a.min_spearman:
        raise SystemExit(f"vLLM disagrees with the reference (min spearman {worst:.4f} < {a.min_spearman})")
    print(f"ok: min spearman {worst:.4f}")


def _copy_dev(src: Path, dst: Path) -> None:
    import shutil

    from code_embed_v3.data.build import DEV_DIR

    if (src / DEV_DIR).exists() and not (dst / DEV_DIR).exists():
        shutil.copytree(src / DEV_DIR, dst / DEV_DIR)


def _cmd_synth(a: argparse.Namespace) -> None:
    import torch

    from code_embed_v3 import hub
    from code_embed_v3.data import synth

    out = Path(a.out)
    work = Path(a.work_dir) if a.work_dir else out.with_name(out.stem + ".work")
    remote = f"{a.task}/work"
    if a.push_repo and not work.exists():
        # A fresh pod continues from the shards the last one pushed.
        hub.fetch_folder(a.push_repo, remote, work, repo_type="dataset")
    codes = synth.load_or_save_seeds(
        work, lambda: synth.seed_code([Path(d) for d in a.seed_dirs], int(a.rows * a.oversample), seed=a.seed)
    )
    num_gpus = max(1, torch.cuda.device_count())
    generator = synth.vllm_generator(a.generator, num_gpus=num_gpus, seed=a.seed)

    def push_work() -> None:
        if a.push_repo:
            hub.push_folder(work, a.push_repo, remote, repo_type="dataset", message=f"{a.task}: shards")

    rows, stats = synth.generate(
        a.task,
        codes,
        generator,
        work,
        shard_rows=a.shard_rows,
        seed=a.seed,
        round_shards=4 * num_gpus,
        after_round=push_work,
    )
    rows = synth.dedupe(rows)
    stats["deduped"] = len(rows)
    if a.filter_model:
        encoder = synth.sentence_transformer_encoder(a.filter_model)
        rows = synth.consistency_filter(rows, encoder, top_k=a.top_k)
    rows = rows[: a.rows]
    stats["kept"] = len(rows)
    synth.write_rows(rows, out)
    meta = {**stats, "task": a.task, "generator": a.generator, "filter_model": a.filter_model}
    task = synth.load_prompts().tasks[a.task]
    meta |= {"top_k": a.top_k, "prompt": task.user, "styles": task.styles}
    stats_path = out.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(meta, indent=2))
    if a.push_repo:
        # Raw replies stay too, so a later model can re-filter or re-parse without regenerating.
        hub.push_file(out, a.push_repo, path_in_repo=f"{a.task}/{out.name}", repo_type="dataset")
        hub.push_file(
            stats_path, a.push_repo, path_in_repo=f"{a.task}/{stats_path.name}", repo_type="dataset"
        )
    print(json.dumps(meta, indent=2))


def _cmd_train(a: argparse.Namespace) -> None:
    import subprocess
    import sys

    import torch

    from code_embed_v3.config import StageConfig
    from code_embed_v3.train import configure_tracking, plan_stage, run_stage

    stage = StageConfig.from_yaml(a.config)
    configure_tracking(stage)
    # Planned once here, before any rank exists, so ranks never race on Hub downloads.
    plan = plan_stage(stage)
    if plan.done is not None:
        print(f"{stage.name}: already finished -> {plan.done}")
        return
    nproc = a.nproc if a.nproc is not None else max(1, torch.cuda.device_count())
    if nproc == 1:
        print(
            run_stage(
                stage, device=a.device, require_audit=not a.skip_audit_check, resume_from=plan.resume_from
            )
        )
        return
    cmd = [sys.executable, "-m", "torch.distributed.run", f"--nproc-per-node={nproc}"]
    cmd += ["-m", "code_embed_v3.cli", "train-worker", a.config]
    if plan.resume_from is not None:
        cmd += ["--resume-from", str(plan.resume_from)]
    if a.skip_audit_check:
        cmd.append("--skip-audit-check")
    subprocess.run(cmd, check=True)
    print(Path(stage.output_dir) / "final")


def _cmd_train_worker(a: argparse.Namespace) -> None:
    from code_embed_v3.config import StageConfig
    from code_embed_v3.train import run_stage

    run_stage(
        StageConfig.from_yaml(a.config),
        device=a.device,
        require_audit=not a.skip_audit_check,
        resume_from=Path(a.resume_from) if a.resume_from else None,
    )


def _cmd_push_data(a: argparse.Namespace) -> None:
    from code_embed_v3.hub import push_folder

    for d in a.data_dirs:
        path = Path(d)
        push_folder(path, a.repo, path.name, repo_type="dataset", message=f"upload {path.name}")


def _cmd_pull_data(a: argparse.Namespace) -> None:
    from code_embed_v3.hub import fetch_folder

    for d in a.data_dirs:
        path = Path(d)
        if (path / "manifest.json").exists():
            print(f"{path}: present locally")
        elif fetch_folder(a.repo, path.name, path, repo_type="dataset"):
            print(f"{path}: fetched")
        else:
            print(f"{path}: not on the Hub yet")


def _cmd_release(a: argparse.Namespace) -> None:
    import tempfile

    from code_embed_v3.data.build import read_manifest
    from code_embed_v3.hub import push_folder
    from code_embed_v3.release import build_release

    audited: list[str] | None = None
    if a.data_dir:
        lists = [m.get("audited_against", []) for m in read_manifest(a.data_dir)["sources"].values()]
        audited = sorted(set.intersection(*(set(x) for x in lists))) if lists else None
    with tempfile.TemporaryDirectory() as tmp:
        out = build_release(
            Path(a.model),
            Path(tmp) / "release",
            a.repo,
            base_model=a.base_model,
            results_dir=Path(a.results) if a.results else None,
            contamination_report=Path(a.contamination_report) if a.contamination_report else None,
            audited_tasks=audited,
        )
        print("publishing:", *sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()))
        if a.dry:
            print((out / "README.md").read_text())
            return
        push_folder(out, a.repo, None, message=a.message)


def _cmd_tune(a: argparse.Namespace) -> None:
    import torch

    from code_embed_v3.tune import run

    gpus = a.gpus if a.gpus is not None else torch.cuda.device_count()
    print(run(Path(a.config), gpus=gpus))


def _cmd_tune_worker(a: argparse.Namespace) -> None:
    from code_embed_v3.tune import TuneConfig, work

    work(TuneConfig.from_yaml(a.config), a.phase, a.quota)


def _cmd_dry_run(a: argparse.Namespace) -> None:
    from code_embed_v3.dry_run import main as dry_run

    dry_run(Path(a.root), hub=a.hub, wandb=a.wandb, rows=a.rows, teacher=a.teacher)


def _cmd_doctor(a: argparse.Namespace) -> None:
    """Fail before paying for GPUs: credentials, GPUs, disk and storage location."""
    import os
    import shutil

    import torch
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    from code_embed_v3.env import available_cpus

    problems: list[str] = []
    try:
        print(f"hf: logged in as {HfApi().whoami()['name']}")
    except (HfHubHTTPError, OSError, KeyError) as e:
        problems.append(f"HF_TOKEN missing or invalid: {e}")
    if not os.environ.get("WANDB_API_KEY"):
        problems.append("WANDB_API_KEY is not set")
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        print(f"warning: CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} limits the visible GPUs")
    gpus = torch.cuda.device_count()
    names = sorted({torch.cuda.get_device_name(i) for i in range(gpus)})
    print(f"gpus: {gpus} {names}  cpus: {available_cpus()}")
    if a.gpus is not None and gpus != a.gpus:
        problems.append(f"expected {a.gpus} GPUs, see {gpus}")
    free_gb = shutil.disk_usage(".").free / 1e9
    print(f"disk free here: {free_gb:.0f} GB")
    if free_gb < a.min_free_gb:
        problems.append(f"only {free_gb:.0f} GB free, need {a.min_free_gb}")
    if os.environ.get("RUNPOD_POD_ID"):
        if not Path.cwd().is_relative_to("/workspace"):
            problems.append(
                f"cwd {Path.cwd()} is container disk; clone into /workspace so runs survive restarts"
            )
        if not os.environ.get("HF_HOME", "").startswith("/workspace"):
            problems.append("HF_HOME should be under /workspace so model and dataset caches persist")
    if problems:
        raise SystemExit("doctor found problems:\n  " + "\n  ".join(problems))
    print("ok")


def _cmd_merge(a: argparse.Namespace) -> None:
    from sentence_transformers import SentenceTransformer

    from code_embed_v3.modeling import META_FILE

    merged = SentenceTransformer.merge(a.models, method=a.method)
    merged.save(a.output)
    meta = Path(a.models[0]) / META_FILE
    if meta.exists():
        (Path(a.output) / META_FILE).write_text(meta.read_text())
    print(a.output)


def _cmd_dev_eval(a: argparse.Namespace) -> None:
    from code_embed_v3.eval.dev_suite import build_dev_evaluator
    from code_embed_v3.modeling import load_model

    model = load_model(a.model, truncate_dim=a.truncate_dim)
    print(json.dumps(build_dev_evaluator(a.data_dir)(model), indent=2))


def _cmd_mteb(a: argparse.Namespace) -> None:
    from code_embed_v3.eval.run_mteb import DEFAULT_BENCHMARKS, run

    run(
        a.model,
        model_name=a.name,
        benchmarks=tuple(a.benchmarks or DEFAULT_BENCHMARKS),
        data_dirs=a.data_dirs,
        output_folder=a.output,
        batch_size=a.batch_size,
        truncate_dim=a.truncate_dim,
        base_model=a.base_model,
        tasks=a.tasks,
        allow_eval_overlap=a.allow_eval_overlap,
    )


def _cmd_zero_shot(a: argparse.Namespace) -> None:
    from code_embed_v3.eval.run_mteb import (
        DEFAULT_BENCHMARKS,
        declared_meta,
        require_zero_shot,
        zero_shot_report,
    )

    report = zero_shot_report(
        declared_meta(a.data_dirs, a.base_model), tuple(a.benchmarks or DEFAULT_BENCHMARKS)
    )
    print(json.dumps(report, indent=2))
    require_zero_shot(report)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from code_embed_v3.env import load_dotenv

    load_dotenv()
    p = argparse.ArgumentParser(prog="ce3")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("build-data", help="materialise a data mix")
    s.add_argument("config")
    s.add_argument("--only", nargs="*")
    s.add_argument("--output-dir", help="override the config's output_dir")
    s.add_argument("--max-rows", type=int, help="cap every source (dry runs)")
    s.set_defaults(fn=_cmd_build_data)

    s = sub.add_parser("audit", help="drop rows that overlap benchmark test data")
    s.add_argument("data_dir")
    s.add_argument("--benchmarks", nargs="*")
    s.add_argument("--tasks", nargs="*", help="audit against these tasks only (dry runs)")
    s.add_argument("--num-proc", type=int, help="default: every CPU this container may use")
    s.set_defaults(fn=_cmd_audit)

    s = sub.add_parser("mine", help="positive-aware hard-negative mining")
    s.add_argument("data_dir")
    s.add_argument("--model", required=True)
    s.add_argument("--output-dir", required=True)
    s.add_argument("--num-negatives", type=int, default=7)
    s.add_argument("--relative-margin", type=float, default=0.05)
    s.add_argument("--batch-size", type=int, default=512, help="encode batch per GPU")
    s.add_argument("--max-rows-per-source", type=int)
    s.add_argument(
        "--no-query-prompts",
        dest="query_prompts",
        action="store_false",
        help="for mining models trained without our prompts",
    )
    s.set_defaults(fn=_cmd_mine)

    s = sub.add_parser("label", help="teacher scores for distillation")
    s.add_argument("data_dir")
    s.add_argument("--teacher", default="Qwen/Qwen3-Reranker-4B")
    s.add_argument("--output-dir", required=True)
    s.add_argument("--backend", choices=["vllm", "crossencoder"], default="vllm")
    s.add_argument("--batch-size", type=int, default=32, help="crossencoder backend only")
    s.add_argument("--max-rows-per-source", type=int, help="the dominant cost lever of the pipeline")
    s.add_argument("--shard-rows", type=int, default=20_000)
    s.set_defaults(fn=_cmd_label)

    s = sub.add_parser("label-check", help="check vLLM teacher scores against the sentence-transformers ones")
    s.add_argument("data_dir")
    s.add_argument("--teacher", default="Qwen/Qwen3-Reranker-4B")
    s.add_argument("--rows", type=int, default=64)
    s.add_argument("--batch-size", type=int, default=16)
    s.add_argument("--min-spearman", type=float, default=0.99)
    s.add_argument("--work-dir", default="data/.label-check")
    s.set_defaults(fn=_cmd_label_check)

    s = sub.add_parser("synth", help="generate synthetic pairs from real code with an open coder LLM")
    s.add_argument("task", choices=["nl2code", "techqa", "code_edit", "multiturn"])
    s.add_argument("--seed-dirs", nargs="+", required=True, help="built sources whose `positive` is code")
    s.add_argument("--rows", type=int, required=True, help="rows to keep after filtering")
    s.add_argument("--oversample", type=float, default=1.6, help="seeds per kept row; covers rejects")
    s.add_argument("--out", required=True, help="parquet read by configs/data/synth_v1.yaml")
    s.add_argument("--generator", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    s.add_argument("--filter-model", default="Alibaba-NLP/gte-modernbert-base", help="'' disables the filter")
    s.add_argument("--top-k", type=int, default=3)
    s.add_argument("--shard-rows", type=int, default=2000)
    s.add_argument("--work-dir")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument(
        "--push-repo",
        default="divakantpandey/code-embed-v3-synth",
        help="private Hub dataset for outputs, raw replies and resume; '' disables",
    )
    s.set_defaults(fn=_cmd_synth)

    s = sub.add_parser("train", help="run one training stage")
    s.add_argument("config")
    s.add_argument("--device", help="single-process runs only")
    s.add_argument("--nproc", type=int, help="processes (GPUs); default: every visible GPU")
    s.add_argument("--skip-audit-check", action="store_true", help="only for smoke tests")
    s.set_defaults(fn=_cmd_train)

    s = sub.add_parser("train-worker", help="internal: one torchrun rank of `ce3 train`")
    s.add_argument("config")
    s.add_argument("--resume-from")
    s.add_argument("--device", help="tests only; ranks pick their GPU from LOCAL_RANK")
    s.add_argument("--skip-audit-check", action="store_true")
    s.set_defaults(fn=_cmd_train_worker)

    for cmd, fn, help_text in (
        ("push-data", _cmd_push_data, "upload built data dirs to a private Hub dataset"),
        ("pull-data", _cmd_pull_data, "download built data dirs missing locally from the Hub dataset"),
    ):
        s = sub.add_parser(cmd, help=help_text)
        s.add_argument("data_dirs", nargs="+")
        s.add_argument("--repo", default="divakantpandey/code-embed-v3-data")
        s.set_defaults(fn=fn)

    s = sub.add_parser("release", help="publish model files and a functional card only (private repo)")
    s.add_argument("model")
    s.add_argument("--repo", default="divakantpandey/code-embed-v3")
    s.add_argument("--contamination-report", help="published as totals only, never per source")
    s.add_argument("--data-dir", help="built mix, to state which benchmark tasks were audited")
    s.add_argument("--results", help="`ce3 mteb --output` folder: scores and zero-shot % for the card")
    s.add_argument("--base-model", help="named in the card and license line (Apache-2.0 attribution)")
    s.add_argument("--dry", action="store_true", help="print the files and card instead of uploading")
    s.add_argument("--message", default="release")
    s.set_defaults(fn=_cmd_release)

    s = sub.add_parser("tune", help="Optuna search on proxy runs scored on the dev suite (never MTEB)")
    s.add_argument("config", help="e.g. configs/tune/150m_ft.yaml")
    s.add_argument("--gpus", type=int, help="parallel trials, one per GPU; default: every visible GPU")
    s.set_defaults(fn=_cmd_tune)

    s = sub.add_parser("tune-worker", help="internal: one GPU's share of `ce3 tune`")
    s.add_argument("config")
    s.add_argument("phase", choices=["search", "confirm"])
    s.add_argument("quota", type=int)
    s.set_defaults(fn=_cmd_tune_worker)

    s = sub.add_parser("dry-run", help="tiny end-to-end rehearsal on the GPU pod; refuses without CUDA")
    s.add_argument("--root", default="dryrun")
    s.add_argument("--no-hub", dest="hub", action="store_false", help="skip scratch Hub repos")
    s.add_argument("--no-wandb", dest="wandb", action="store_false")
    s.add_argument("--rows", type=int, default=300, help="rows per source")
    s.add_argument("--teacher", default="Qwen/Qwen3-Reranker-4B")
    s.set_defaults(fn=_cmd_dry_run)

    s = sub.add_parser("doctor", help="check credentials, GPUs, disk and storage before a paid run")
    s.add_argument("--gpus", type=int)
    s.add_argument("--min-free-gb", type=float, default=200)
    s.set_defaults(fn=_cmd_doctor)

    s = sub.add_parser("merge", help="merge checkpoints (needs the `merge` extra)")
    s.add_argument("models", nargs="+")
    s.add_argument("--method", default="linear")
    s.add_argument("--output", required=True)
    s.set_defaults(fn=_cmd_merge)

    s = sub.add_parser("dev-eval", help="per-family dev suite")
    s.add_argument("model")
    s.add_argument("data_dir")
    s.add_argument("--truncate-dim", type=int)
    s.set_defaults(fn=_cmd_dev_eval)

    s = sub.add_parser("mteb", help="official MTEB(Code)/CoIR/CoREB evaluation")
    s.add_argument("model")
    s.add_argument("--name", required=True, help="e.g. divakantpandey/code-embed-v3")
    s.add_argument("--benchmarks", nargs="*")
    s.add_argument("--data-dirs", nargs="*", help="built mixes, to declare training_datasets")
    s.add_argument("--output", default="results/mteb")
    s.add_argument("--batch-size", type=int, default=64)
    s.add_argument("--truncate-dim", type=int)
    s.add_argument("--base-model", help="checkpoint fine-tuned from; its declared training data is inherited")
    s.add_argument("--tasks", nargs="*", help="evaluate these tasks only (dry runs)")
    s.add_argument("--allow-eval-overlap", action="store_true", help="only for a separately reported run")
    s.set_defaults(fn=_cmd_mteb)

    s = sub.add_parser(
        "zero-shot", help="fail unless mteb would call a model trained on these mixes zero-shot"
    )
    s.add_argument("data_dirs", nargs="+")
    s.add_argument("--base-model")
    s.add_argument("--benchmarks", nargs="*")
    s.set_defaults(fn=_cmd_zero_shot)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
