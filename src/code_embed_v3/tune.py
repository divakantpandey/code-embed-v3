"""Hyperparameter search with Optuna on short proxy runs, scored on the dev suite.

Never on MTEB: tuning against the benchmark would overfit its test sets and make the reported
score meaningless. The dev suite is held-out training-source rows, one score per task family,
macro-averaged.

Two phases. Search: TPE proposes settings, each trial trains on the same number of samples (so
batch sizes compare fairly) and a median pruner stops clearly worse trials early. Confirm: the
top trials rerun with extra seeds, and the winner is the best mean, because proxy runs differ by
noise of a similar size to the gaps between good settings.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from statistics import mean
from typing import Any, Literal, TypedDict

import yaml
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from code_embed_v3 import hub
from code_embed_v3.config import StageConfig, _build, load_yaml
from code_embed_v3.train import run_stage

logger = logging.getLogger(__name__)

METRIC = "eval_sequential_score"  # macro-average of per-family dev nDCG@10


@dataclass
class ParamSpec:
    type: Literal["float", "int", "categorical"]
    low: float | None = None
    high: float | None = None
    log: bool = False
    choices: list[Any] | None = None


@dataclass
class TuneConfig:
    study: str
    base_stage: str
    params: dict[str, ParamSpec]
    samples_per_trial: int = 50_000
    evals_per_trial: int = 4
    n_trials: int = 32
    confirm_top: int = 3
    confirm_seeds: list[int] = field(default_factory=lambda: [1, 2])
    # Settings whose mean dev score is within this of the winner are reported as equivalent.
    min_delta: float = 0.002
    output_dir: str = "tune"
    # Private Hub dataset that keeps the study journal, so a lost pod does not lose finished trials.
    storage_repo: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> TuneConfig:
        raw = load_yaml(path)
        raw["params"] = {k: _build(ParamSpec, v, f"{path}:params.{k}") for k, v in raw["params"].items()}
        cfg = _build(cls, raw, str(path))
        tunable = {f.name for f in fields(StageConfig)}
        unknown = set(cfg.params) - tunable
        if unknown:
            raise ValueError(f"{path}: not StageConfig fields: {sorted(unknown)}")
        return cfg

    @property
    def dir(self) -> Path:
        return Path(self.output_dir) / self.study

    @property
    def journal(self) -> Path:
        return self.dir / "journal.log"


def _storage(cfg: TuneConfig) -> Any:
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend

    cfg.dir.mkdir(parents=True, exist_ok=True)
    return JournalStorage(JournalFileBackend(str(cfg.journal)))


def load_study(cfg: TuneConfig) -> Any:
    import optuna

    return optuna.create_study(
        study_name=cfg.study,
        storage=_storage(cfg),
        direction="maximize",
        # constant_liar keeps parallel GPU workers from proposing the same point.
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True, constant_liar=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        load_if_exists=True,
    )


def suggest(trial: Any, specs: dict[str, ParamSpec]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, spec in specs.items():
        if spec.type == "categorical":
            out[name] = trial.suggest_categorical(name, spec.choices)
        elif spec.type == "int":
            out[name] = trial.suggest_int(name, int(spec.low or 0), int(spec.high or 0), log=spec.log)
        else:
            out[name] = trial.suggest_float(name, float(spec.low or 0), float(spec.high or 0), log=spec.log)
    return out


def trial_stage(
    base: StageConfig, params: dict[str, Any], cfg: TuneConfig, number: int, seed: int
) -> StageConfig:
    """Proxy run: same samples for every trial, evaluated ``evals_per_trial`` times, never saved."""
    batch = int(params.get("batch_size", base.batch_size))
    eval_every = max(1, cfg.samples_per_trial // batch // cfg.evals_per_trial)
    steps = eval_every * cfg.evals_per_trial
    return replace(
        base,
        **params,
        name=f"{cfg.study}-t{number}-s{seed}",
        output_dir=str(cfg.dir / f"trial-{number}"),
        hub_repo=None,
        init_from=base.init_from,
        max_steps=steps,
        eval_steps=eval_every,
        save_steps=steps + 1,
        seed=seed,
        wandb_project="code-embed-v3-tune",
    )


class ReportToTrial(TrainerCallback):
    def __init__(self, trial: Any, prune: bool) -> None:
        self.trial = trial
        self.prune = prune
        self.last: float | None = None

    def on_evaluate(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        import optuna

        metrics = kwargs.get("metrics") or {}
        if METRIC not in metrics:
            return
        self.last = float(metrics[METRIC])
        self.trial.report(self.last, state.global_step)
        if self.prune and self.trial.should_prune():
            raise optuna.TrialPruned()


def _finish_wandb() -> None:
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is not None:
        wandb.finish()


def objective(trial: Any, cfg: TuneConfig, base: StageConfig, device: str | None) -> float:
    params = suggest(trial, cfg.params)
    confirm = "confirm_of" in trial.user_attrs
    stage = trial_stage(base, params, cfg, trial.number, int(trial.user_attrs.get("seed", base.seed)))
    report = ReportToTrial(trial, prune=not confirm)
    try:
        run_stage(stage, device=device, callbacks=[report])
    finally:
        shutil.rmtree(stage.output_dir, ignore_errors=True)
        _finish_wandb()
    if report.last is None:
        raise RuntimeError(f"trial {trial.number}: the dev suite never ran; does the stage set dev_suite?")
    return report.last


def _backup(cfg: TuneConfig) -> None:
    if cfg.storage_repo and cfg.journal.exists():
        hub.push_file(
            cfg.journal, cfg.storage_repo, path_in_repo=f"tune/{cfg.study}/journal.log", repo_type="dataset"
        )


def work(cfg: TuneConfig, phase: Literal["search", "confirm"], quota: int, device: str | None = None) -> None:
    """One GPU's share: ask/tell ``quota`` trials. Confirm workers only take queued confirm trials."""
    import optuna
    from optuna.trial import TrialState

    study = load_study(cfg)
    base = StageConfig.from_yaml(cfg.base_stage)
    for _ in range(quota):
        trial = study.ask()
        if phase == "confirm" and "confirm_of" not in trial.user_attrs:
            study.tell(trial, state=TrialState.FAIL)
            break
        try:
            study.tell(trial, objective(trial, cfg, base, device))
        except optuna.TrialPruned:
            study.tell(trial, state=TrialState.PRUNED)
        _backup(cfg)


def _quotas(total: int, workers: int) -> list[int]:
    return [total // workers + (i < total % workers) for i in range(workers)]


def _run_phase(
    cfg_path: Path,
    cfg: TuneConfig,
    phase: Literal["search", "confirm"],
    total: int,
    gpus: int,
    device: str | None,
) -> None:
    if total <= 0:
        return
    if gpus <= 1:
        work(cfg, phase, total, device)
        return
    procs = []
    for gpu, quota in enumerate(_quotas(total, gpus)):
        if quota:
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
            cmd = [sys.executable, "-m", "code_embed_v3.cli", "tune-worker", str(cfg_path), phase, str(quota)]
            procs.append(subprocess.Popen(cmd, env=env))
    failed = [p.args for p in procs if p.wait() != 0]
    if failed:
        raise RuntimeError(f"tune workers failed: {failed}")


class Ranked(TypedDict):
    trial: int
    mean: float
    runs: list[float]
    params: dict[str, Any]


def summarise(cfg: TuneConfig, study: Any) -> dict[str, Any]:
    """Group each searched trial with its confirm reruns and rank parameter sets by mean score."""
    from optuna.trial import TrialState

    done = [t for t in study.trials if t.state == TrialState.COMPLETE]
    groups: dict[int, list[float]] = {}
    params: dict[int, dict[str, Any]] = {}
    for t in done:
        key = int(t.user_attrs.get("confirm_of", t.number))
        groups.setdefault(key, []).append(float(t.value))
        params.setdefault(key, t.params)
    ranked = sorted(groups, key=lambda k: (len(groups[k]) > 1, mean(groups[k])), reverse=True)
    rows = [Ranked(trial=k, mean=mean(groups[k]), runs=groups[k], params=params[k]) for k in ranked]
    best = rows[0]
    return {
        "study": cfg.study,
        "metric": METRIC,
        "best": best,
        "equivalent": [
            r for r in rows[1:] if len(r["runs"]) > 1 and best["mean"] - r["mean"] <= cfg.min_delta
        ],
        "ranking": rows,
        "trials": {s.name: sum(t.state == s for t in study.trials) for s in TrialState},
    }


def run(cfg_path: Path, gpus: int, device: str | None = None) -> Path:
    import optuna
    from optuna.trial import TrialState

    cfg = TuneConfig.from_yaml(cfg_path)
    if cfg.storage_repo and not cfg.journal.exists():
        hub.fetch_folder(cfg.storage_repo, f"tune/{cfg.study}", cfg.dir, repo_type="dataset")
    study = load_study(cfg)

    searched = [t for t in study.trials if "confirm_of" not in t.user_attrs and t.state.is_finished()]
    _run_phase(cfg_path, cfg, "search", cfg.n_trials - len(searched), gpus, device)

    study = load_study(cfg)
    complete = [
        t for t in study.trials if t.state == TrialState.COMPLETE and "confirm_of" not in t.user_attrs
    ]
    top = sorted(complete, key=lambda t: float(t.value), reverse=True)[: cfg.confirm_top]
    have = {(t.user_attrs.get("confirm_of"), t.user_attrs.get("seed")) for t in study.trials}
    queued = 0
    for t in top:
        for seed in cfg.confirm_seeds:
            if (t.number, seed) not in have:
                study.enqueue_trial(t.params, user_attrs={"confirm_of": t.number, "seed": seed})
                queued += 1
    _run_phase(cfg_path, cfg, "confirm", queued, gpus, device)

    summary = summarise(cfg, load_study(cfg))
    best_stage = replace(StageConfig.from_yaml(cfg.base_stage), **summary["best"]["params"])
    stage_path = cfg.dir / "best_stage.yaml"
    stage_path.write_text(
        f"# Tuned by `ce3 tune` (study {cfg.study}): base {cfg.base_stage} with the best mean dev score.\n"
        + yaml.safe_dump(asdict(best_stage), sort_keys=False)
    )
    (cfg.dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    _backup(cfg)
    logger.info("optuna %s: best %s", optuna.__version__, summary["best"])
    if not math.isfinite(summary["best"]["mean"]):
        raise RuntimeError("best trial has no finite score")
    return stage_path
