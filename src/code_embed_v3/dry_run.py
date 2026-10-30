"""End-to-end rehearsal on the GPU pod at tiny scale, before paying for the real run.

Runs every real step (streaming data, vLLM generation and teacher, mining, multi-GPU training,
Hub checkpoints, W&B, mteb) with a few hundred rows, kills a training run mid-way to prove resume,
and deletes the local run to prove the Hub copy restores it. Refuses to run without CUDA, so it can
never start on a laptop.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml

from code_embed_v3.config import load_yaml
from code_embed_v3.paths import resolve

GTE = "Alibaba-NLP/gte-modernbert-base"
# Small, fast mteb tasks; the audit is checked against the same ones.
DRY_TASKS = ["CosQA", "SyntheticText2SQL"]


class DryRunError(RuntimeError):
    pass


@dataclass
class DryRun:
    root: Path
    hub_user: str | None
    wandb: bool
    rows: int
    teacher: str
    gpus: int = field(default_factory=torch.cuda.device_count)
    timings: dict[str, float] = field(default_factory=dict)
    hub_repos: list[str] = field(default_factory=list)

    def ce3(self, step: str, *args: str) -> str:
        log = self.root / "logs" / f"{step}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        with log.open("w") as fh:
            proc = subprocess.run(
                [sys.executable, "-m", "code_embed_v3.cli", *args], stdout=fh, stderr=subprocess.STDOUT
            )
        self.timings[step] = time.perf_counter() - start
        text = log.read_text()
        if proc.returncode != 0:
            raise DryRunError(f"{step} failed (exit {proc.returncode}); last lines of {log}:\n{text[-3000:]}")
        print(f"ok  {step:<22} {self.timings[step]:7.1f}s", flush=True)
        return text

    def stage(self, name: str, base: str, **overrides: Any) -> Path:
        raw = load_yaml(resolve(f"configs/stages/{base}"))
        raw.update(
            name=f"ce3-dryrun-{name}",
            output_dir=str(self.root / "runs" / name),
            wandb_project="code-embed-v3-dryrun",
            report_to=["wandb"] if self.wandb else [],
            hub_repo=f"{self.hub_user}/ce3-dryrun-{name}" if self.hub_user else None,
            logging_steps=1,
            **overrides,
        )
        if raw["hub_repo"]:
            self.hub_repos.append(raw["hub_repo"])
        path = self.root / "configs" / f"{name}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(raw))
        return path

    def synth_config(self) -> Path:
        raw = load_yaml(resolve("configs/data/synth_v1.yaml"))
        raw["output_dir"] = str(self.root / "data")
        for src in raw["sources"]:
            src["data_files"] = str(self.root / "synth" / Path(src["data_files"]).name)
        path = self.root / "configs" / "synth.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(raw))
        return path

    def interrupted_train(
        self, config: Path, output_dir: Path, kill_at_step: int, hub_repo: str | None
    ) -> None:
        """Start training, SIGTERM the whole process group once a checkpoint exists, then resume.

        With a Hub repo the local run is deleted after the kill, as if the pod were lost, so the resume
        must come from the Hub's ``last-checkpoint``.
        """
        log = self.root / "logs" / "train-killed.log"
        with log.open("w") as fh:
            proc = subprocess.Popen(
                [sys.executable, "-m", "code_embed_v3.cli", "train", str(config)],
                stdout=fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            target = output_dir / f"checkpoint-{kill_at_step}" / "trainer_state.json"
            while proc.poll() is None and not target.exists():
                time.sleep(0.5)
            if proc.poll() is not None:
                raise DryRunError(f"training ended before checkpoint-{kill_at_step}; see {log}")
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=120)
        print(f"ok  {'train killed':<22} at checkpoint-{kill_at_step}", flush=True)
        if hub_repo:
            self.wait_for_hub_file(hub_repo, "last-checkpoint/trainer_state.json")
            shutil.rmtree(output_dir)
        out = self.ce3("train-resumed", "train", str(config))
        if "resuming from" not in out:
            raise DryRunError("second run did not resume from a checkpoint")

    def wait_for_hub_file(self, repo: str, path: str, timeout: float = 600) -> None:
        """Hub pushes run in the background, so poll until the file is there."""
        from huggingface_hub import HfApi
        from huggingface_hub.errors import RepositoryNotFoundError

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                if path in HfApi().list_repo_files(repo):
                    return
            except RepositoryNotFoundError:
                pass
            time.sleep(10)
        raise DryRunError(f"{repo}/{path} never appeared on the Hub")

    def require_fresh_hub_repos(self) -> None:
        """Every step resumes from its Hub repo, so a previous dry-run's leftovers would let generation,
        resume and restore "pass" without running. Refuse unless the scratch repos are empty or absent."""
        from huggingface_hub import HfApi
        from huggingface_hub.errors import RepositoryNotFoundError

        api = HfApi()
        stale = []
        for name, repo_type in (("synth", "dataset"), ("ft", "model"), ("s3", "model"), ("release", "model")):
            repo = f"{self.hub_user}/ce3-dryrun-{name}"
            try:
                files = api.list_repo_files(repo, repo_type=repo_type)
            except RepositoryNotFoundError:
                continue
            # A renamed repo's old name redirects to it, so its files show up here too.
            if set(files) - {".gitattributes", "README.md"}:
                stale.append(repo)
        if stale:
            raise DryRunError(f"scratch Hub repos from an earlier dry-run still hold files: {stale}")

    def run(self) -> None:
        data, mined, labelled = self.root / "data", self.root / "mined", self.root / "labelled"
        if self.hub_user:
            self.require_fresh_hub_repos()
        rows = str(self.rows)
        self.ce3("doctor", "doctor", "--min-free-gb", "30")
        self.ce3(
            "build-data",
            "build-data",
            str(resolve("configs/data/mix_v1.yaml")),
            "--output-dir",
            str(data),
            "--max-rows",
            rows,
        )
        seeds = [str(data / "cornstack_python"), str(data / "cornstack_java")]
        synth_repo = f"{self.hub_user}/ce3-dryrun-synth" if self.hub_user else ""
        if synth_repo:
            self.hub_repos.append(synth_repo)
        for task in ("nl2code", "techqa", "code_edit", "multiturn"):
            self.ce3(
                f"synth-{task}",
                "synth",
                task,
                "--seed-dirs",
                *seeds,
                "--rows",
                "24",
                "--oversample",
                "2",
                "--out",
                str(self.root / "synth" / f"{task}.parquet"),
                "--push-repo",
                synth_repo,
            )
        self.ce3("build-synth", "build-data", str(self.synth_config()))
        self.ce3("audit", "audit", str(data), "--tasks", *DRY_TASKS)
        self.ce3("zero-shot", "zero-shot", str(data), "--base-model", GTE)
        self.attention_check()
        self.ce3(
            "mine",
            "mine",
            str(data),
            "--model",
            GTE,
            "--no-query-prompts",
            # The pilot's MINE_NEGS. With the default 7 at 300 rows/source, the no-duplicates sampler
            # cannot fill batches from mined sources, so the dry-run trained almost only on behaviour data.
            "--num-negatives",
            "4",
            "--output-dir",
            str(mined),
            "--batch-size",
            "128",
        )
        combined = self.behaviour(data, mined, synth_repo)

        per_gpu = 16
        ft = self.stage(
            "ft",
            "150m_ft_behaviour.yaml",
            data_dir=str(combined),
            dev_suite=str(combined),
            batch_size=per_gpu * self.gpus,
            max_seq_length=256,
            max_steps=120,
            save_steps=20,
            eval_steps=60,
        )
        ft_out = self.root / "runs" / "ft"
        ft_repo = f"{self.hub_user}/ce3-dryrun-ft" if self.hub_user else None
        self.interrupted_train(ft, ft_out, kill_at_step=40, hub_repo=ft_repo)
        if ft_repo:
            self.wait_for_hub_file(ft_repo, "final/modules.json")
            shutil.rmtree(ft_out)
            out = self.ce3("train-from-hub", "train", str(ft))
            if "already finished" not in out:
                raise DryRunError("a fresh pod did not restore the finished stage from the Hub")

        self.ce3(
            "label-check",
            "label-check",
            str(mined),
            "--teacher",
            self.teacher,
            "--rows",
            "16",
            "--work-dir",
            str(self.root / "label-check"),
        )
        self.ce3(
            "label",
            "label",
            str(mined),
            "--teacher",
            self.teacher,
            "--output-dir",
            str(labelled),
            "--max-rows-per-source",
            "64",
        )
        s3 = self.stage(
            "s3",
            "150m_s3_distill.yaml",
            init_from=str(ft_out / "final"),
            data_dir=str(labelled),
            dev_suite=str(labelled),
            model=str(resolve("configs/models/gte-modernbert-150m.yaml")),
            batch_size=8 * self.gpus,
            max_seq_length=256,
            max_steps=6,
            save_steps=3,
            eval_steps=6,
        )
        self.ce3("train-s3", "train", str(s3))
        base_eval, ft_eval = self.root / "behaviour_eval_base.json", self.root / "behaviour_eval.json"
        for step, model, result in (
            ("behaviour-eval-base", GTE, base_eval),
            ("behaviour-eval", str(ft_out / "final"), ft_eval),
        ):
            self.ce3(step, "behaviour-eval", model, "--subsets", "Defects4J", "--out", str(result))
        self.ce3(
            "mteb",
            "mteb",
            str(ft_out / "final"),
            "--name",
            "dryrun/code-embed-v3",
            "--tasks",
            *DRY_TASKS,
            "--data-dirs",
            str(data),
            "--base-model",
            GTE,
            "--output",
            str(self.root / "mteb"),
        )
        if self.hub_user:
            repo = f"{self.hub_user}/ce3-dryrun-release"
            self.hub_repos.append(repo)
            self.ce3(
                "release",
                "release",
                str(ft_out / "final"),
                "--repo",
                repo,
                "--contamination-report",
                str(data / "contamination_report.json"),
            )
        from code_embed_v3.eval.behaviour_eval import no_collapse

        healthy, kept = no_collapse(json.loads(base_eval.read_text()), json.loads(ft_eval.read_text()))
        self.summary(ft_out, kept)
        # Raised last so one dry-run still exercises every later step on a collapsed model.
        if not healthy:
            raise DryRunError(f"fine-tune collapsed: score kept vs base {kept}")

    def behaviour(self, data: Path, mined: Path, push_repo: str) -> Path:
        """Sandbox, variant generation and verification on a handful of problems, merged into the mix."""
        out = self.root / "behaviour"
        self.ce3(
            "behaviour-build",
            "behaviour-build",
            "--out-dir",
            str(out),
            "--max-problems",
            "40",
            "--mutants",
            "3",
            "--rewrites",
            "1",
            "--negatives",
            "1",
            "--repeats",
            "1",
            # The 5,000-row default would make twins ~60% of a 300-rows-per-source mix (vs ~1% in the
            # pilot), and at that share they collapse even a plain recipe (private/docs/RESEARCH_LOG.md).
            "--swe-smith-rows",
            str(self.rows),
            "--push-repo",
            push_repo,
        )
        stats = json.loads((out / "stats.json").read_text())
        if not stats["mutants_kept"]:
            raise DryRunError(f"no verified near-miss mutants; sandbox or generator broken: {stats}")
        self.ce3("audit-behaviour", "audit", str(out), "--tasks", *DRY_TASKS)
        combined = self.root / "combined"
        self.ce3(
            "behaviour-combine", "behaviour-combine", str(mined), str(out), "--output-dir", str(combined)
        )
        return combined

    def attention_check(self) -> None:
        """flash-attention must give sdpa's embeddings; report the speedup that pays for it."""
        from code_embed_v3.config import ModelConfig
        from code_embed_v3.modeling import build_model

        cfg = ModelConfig.from_yaml(resolve("configs/models/gte-modernbert-150m.yaml"))
        texts = [("def f(x):\n    return x * 2\n" * (1 + i % 40)) for i in range(512)]
        results: dict[str, Any] = {}
        embeddings = {}
        for attn in ("sdpa", "auto"):
            cfg.attn_implementation = attn
            model = build_model(cfg, device="cuda").to(torch.bfloat16)
            model.encode(texts[:32], batch_size=32)
            torch.cuda.synchronize()
            start = time.perf_counter()
            embeddings[attn] = model.encode(texts, batch_size=64, convert_to_tensor=True).float()
            torch.cuda.synchronize()
            results[attn] = {
                "seconds": time.perf_counter() - start,
                "unpadded": bool(getattr(model[0], "can_flatten_inputs", False)),
            }
            del model
            torch.cuda.empty_cache()
        cos = torch.nn.functional.cosine_similarity(embeddings["sdpa"], embeddings["auto"]).min().item()
        results["min_cosine"] = cos
        results["speedup"] = results["sdpa"]["seconds"] / results["auto"]["seconds"]
        (self.root / "attention_check.json").write_text(json.dumps(results, indent=2))
        if cos < 0.999:
            raise DryRunError(f"flash-attention embeddings differ from sdpa (min cosine {cos:.5f})")
        print(
            f"ok  {'attention':<22} speedup x{results['speedup']:.2f}, unpadded={results['auto']['unpadded']}"
        )

    def summary(self, ft_out: Path, kept_vs_base: dict[str, float]) -> None:
        states = sorted(ft_out.glob("checkpoint-*/trainer_state.json"), key=lambda p: int(p.parent.name[11:]))
        perf = (
            [
                h["perf/samples_per_sec"]
                for h in json.loads(states[-1].read_text())["log_history"]
                if "perf/samples_per_sec" in h
            ]
            if states
            else []
        )
        report = {
            "gpus": self.gpus,
            "gpu": torch.cuda.get_device_name(0),
            "seconds_per_step_name": self.timings,
            "ft_samples_per_sec_last": perf[-1] if perf else None,
            "attention": json.loads((self.root / "attention_check.json").read_text()),
            "zero_shot": json.loads((self.root / "mteb" / "zero_shot.json").read_text()),
            "ft_kept_vs_base": kept_vs_base,
            "hub_repos_created": self.hub_repos,
        }
        (self.root / "dry_run_report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        if self.hub_repos:
            print("scratch Hub repos (delete when done): " + " ".join(self.hub_repos))


def main(root: Path, hub: bool, wandb: bool, rows: int, teacher: str) -> None:
    # The run directory is wiped first; never let a typo point that at the repo or anything else.
    if not root.name.startswith("dryrun"):
        raise SystemExit(f"--root must be a directory named dryrun*, got {root}")
    if not torch.cuda.is_available():
        raise SystemExit("dry-run needs a CUDA GPU; it is meant for the cluster, never a laptop")
    hub_user = None
    if hub:
        from huggingface_hub import HfApi

        hub_user = HfApi().whoami()["name"]
    if root.exists():
        shutil.rmtree(root)
    DryRun(root=root, hub_user=hub_user, wandb=wandb, rows=rows, teacher=teacher).run()
