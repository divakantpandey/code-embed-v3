"""Stage runner: one YAML stage config -> one trained checkpoint, resumable across lost pods."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sentence_transformer.losses import (
    CachedMultipleNegativesRankingLoss,
    DistillKLDivLoss,
    MatryoshkaLoss,
    MultipleNegativesRankingLoss,
)
from sentence_transformers.sentence_transformer.training_args import (
    BatchSamplers,
    MultiDatasetBatchSamplers,
)
from torch import Tensor, nn
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from code_embed_v3 import hub
from code_embed_v3.config import ModelConfig, StageConfig
from code_embed_v3.data.build import MANIFEST, load_mix
from code_embed_v3.data.contamination import full_audit_tasks
from code_embed_v3.env import available_cpus
from code_embed_v3.eval.dev_suite import build_dev_evaluator
from code_embed_v3.modeling import META_FILE, inference_length, init_model, model_meta, save_model
from code_embed_v3.prompts import training_prompts

logger = logging.getLogger(__name__)


class DistillWithContrastive(nn.Module):
    """KL to teacher scores plus an optional in-batch InfoNCE term, sharing one forward pass.

    The contrastive term keeps the in-batch negatives that pure distillation over a handful of mined
    candidates never sees (DenseOn / LateOn recipe).
    """

    def __init__(
        self,
        model: SentenceTransformer,
        scale: float,
        contrastive_weight: float,
        student_temperature: float | None,
        teacher_temperature: float | None,
        gather_across_devices: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.kl = DistillKLDivLoss(
            model, student_temperature=student_temperature, teacher_temperature=teacher_temperature
        )
        self.mnrl = MultipleNegativesRankingLoss(
            model, scale=scale, gather_across_devices=gather_across_devices
        )
        self.contrastive_weight = contrastive_weight

    def forward(self, sentence_features: list[dict[str, Tensor]], labels: Tensor) -> Tensor:
        embeddings = [self.model(f)["sentence_embedding"] for f in sentence_features]
        # DistillKLDivLoss multiplies by student_temperature**2, which shrinks its gradient by 1/tau^2
        # relative to MultipleNegativesRankingLoss(scale=1/tau) when student_temperature << 1.
        loss = self.kl.compute_loss_from_embeddings(embeddings, labels) / (self.kl.student_temperature**2)
        if self.contrastive_weight > 0:
            loss = loss + self.contrastive_weight * self.mnrl.compute_loss_from_embeddings(embeddings, labels)
        return loss

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "kl": self.kl.get_config_dict(),
            "contrastive_weight": self.contrastive_weight,
            "scale": self.mnrl.scale,
        }


class WeightedLoss(nn.Module):
    """One source's loss times a constant. GradCache scales its cached embedding gradients by the
    incoming grad_output, so the weight reaches them as well."""

    def __init__(self, loss: nn.Module, weight: float) -> None:
        super().__init__()
        self.loss = loss
        self.weight = weight

    @property
    def model(self) -> nn.Module:
        # Trainer only re-binds the DDP-wrapped model into losses that expose `model`.
        model = self.loss.model
        if not isinstance(model, nn.Module):
            raise TypeError(f"{type(self.loss).__name__}.model is {type(model).__name__}, not a module")
        return model

    def forward(self, sentence_features: list[dict[str, Tensor]], labels: Tensor | None) -> Tensor:
        loss: Tensor = self.loss(sentence_features, labels)
        return self.weight * loss


def per_source_loss(
    loss: nn.Module, sources: list[str], weights: dict[str, float]
) -> nn.Module | dict[str, nn.Module]:
    unknown = set(weights) - set(sources)
    if unknown:
        raise ValueError(f"source_loss_weights names unknown sources {sorted(unknown)}; have {sources}")
    if not weights:
        return loss
    return {name: WeightedLoss(loss, weights[name]) if name in weights else loss for name in sources}


class TrainingCollapsed(RuntimeError):
    pass


class TrainingShortfall(RuntimeError):
    pass


class CollapseGuard(TrainerCallback):
    """Stops training once the dev score falls below ``ratio`` x the untrained model's score.

    A collapsed model shows a flat loss and small gradients (the first dry-run trained ~90 steps past
    collapse), so only an embedding-quality signal catches it.
    """

    def __init__(self, ratio: float, baseline: float = math.nan) -> None:
        # NaN until scored: every comparison is False, so an unset baseline never stops training.
        self.baseline = baseline
        self.ratio = ratio
        self.tripped: tuple[int, float] | None = None

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        score = next((v for k, v in (metrics or {}).items() if k.endswith("sequential_score")), None)
        if score is not None and score < self.ratio * self.baseline:
            self.tripped = (state.global_step, score)
            control.should_training_stop = True


def build_loss(
    stage: StageConfig,
    model_cfg: ModelConfig,
    model: SentenceTransformer,
    mini_batch_size: int,
    mini_batch_num_tokens: int | None = None,
) -> nn.Module:
    inner: nn.Module
    if stage.loss == "cached_mnrl":
        inner = CachedMultipleNegativesRankingLoss(
            model,
            scale=stage.scale,
            mini_batch_size=mini_batch_size,
            mini_batch_num_tokens=mini_batch_num_tokens,
            gather_across_devices=stage.gather_across_devices,
            directions=tuple(stage.directions),
            hardness_mode=stage.hardness_mode,
            hardness_strength=stage.hardness_strength,
        )
    elif stage.loss == "distill_kl":
        inner = DistillWithContrastive(
            model,
            scale=stage.scale,
            contrastive_weight=stage.contrastive_weight,
            student_temperature=stage.student_temperature,
            teacher_temperature=stage.teacher_temperature,
            gather_across_devices=stage.gather_across_devices,
        )
    else:
        raise ValueError(f"unknown loss {stage.loss!r}")
    return MatryoshkaLoss(model, inner, matryoshka_dims=model_cfg.matryoshka_dims)


class StageTrainer(SentenceTransformerTrainer):
    """Adds the runtime-patch metadata to every save and a throughput metric to every log."""

    def __init__(self, *args: Any, meta: dict[str, Any], **kwargs: Any) -> None:
        self._meta = meta
        self._last_log: tuple[float, int] | None = None
        super().__init__(*args, **kwargs)

    def _save(self, output_dir: str | None = None, state_dict: Any = None) -> None:
        # Writing here rather than in an on_save callback matters: Trainer pushes a checkpoint to the
        # Hub before callbacks run, and a bidirectional checkpoint without this file reloads as causal.
        super()._save(output_dir, state_dict)
        target = output_dir if output_dir is not None else self.args.output_dir
        if target is None:
            return
        (Path(target) / META_FILE).write_text(json.dumps(self._meta, indent=2))

    def evaluation_loop(self, *args: Any, **kwargs: Any) -> Any:
        # sentence-transformers calls the dev evaluator outside autocast; with fp32 master weights that
        # would feed fp32 into flash-attention, which only takes fp16/bf16.
        with self.accelerator.autocast():
            return super().evaluation_loop(*args, **kwargs)

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        if "loss" in logs:
            now, step = time.perf_counter(), self.state.global_step
            if self._last_log is not None and step > self._last_log[1]:
                global_batch = self.args.per_device_train_batch_size * self.args.world_size
                logs["perf/samples_per_sec"] = (
                    (step - self._last_log[1]) * global_batch / (now - self._last_log[0])
                )
            self._last_log = (now, step)
        super().log(logs, start_time)


class LogRunMetadata(TrainerCallback):
    """Ties each W&B run to the exact code, lockfile, configs and audited data it trained on."""

    def __init__(self, stage: StageConfig, model_cfg: ModelConfig) -> None:
        self.stage = stage
        self.model_cfg = model_cfg

    def on_train_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        if not state.is_world_process_zero or "wandb" not in (args.report_to or []):
            return
        import wandb

        if wandb.run is None:
            return
        manifest = Path(self.stage.data_dir) / MANIFEST
        wandb.config.update(
            {
                "ce3/git": _git_state(),
                "ce3/uv_lock_sha256": _sha256(Path("uv.lock")),
                "ce3/stage": asdict(self.stage),
                "ce3/model": asdict(self.model_cfg),
                "ce3/data_manifest": json.loads(manifest.read_text()) if manifest.exists() else None,
            },
            allow_val_change=True,
        )


def _git_state() -> dict[str, Any] | None:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return {"sha": sha.stdout.strip(), "dirty": bool(dirty.stdout.strip())}


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def stage_run_id(stage: StageConfig) -> str:
    """Same config -> same W&B run, so a restarted pod appends to its run instead of starting a new one."""
    digest = hashlib.sha256(json.dumps(asdict(stage), sort_keys=True).encode()).hexdigest()[:8]
    return f"{stage.name}-{digest}"


def configure_tracking(stage: StageConfig) -> None:
    # Assigned, not defaulted: a WANDB_PROJECT inherited from another project's .env must not win.
    os.environ["WANDB_PROJECT"] = stage.wandb_project
    os.environ["WANDB_RUN_ID"] = stage_run_id(stage)
    os.environ["WANDB_RESUME"] = "allow"
    # Checkpoints go to the Hub; logging them to W&B as well would double the upload.
    os.environ["WANDB_LOG_MODEL"] = "false"


@dataclass(frozen=True)
class StagePlan:
    done: Path | None
    resume_from: Path | None


def latest_complete_checkpoint(output_dir: Path) -> Path | None:
    """Newest ``checkpoint-N`` holding ``trainer_state.json``, which Trainer writes after the weights,
    optimizer and RNG state; a pod killed mid-save leaves a checkpoint without it."""
    if not output_dir.is_dir():
        return None
    ckpts = [p for p in output_dir.glob("checkpoint-*") if p.name.split("-")[-1].isdigit()]
    for ckpt in sorted(ckpts, key=lambda p: int(p.name.split("-")[-1]), reverse=True):
        if (ckpt / "trainer_state.json").is_file():
            return ckpt
    return None


def plan_stage(stage: StageConfig) -> StagePlan:
    """Decide where a stage starts: already finished, resume from a checkpoint, or from scratch.

    Local disk is checked before the Hub, so a stopped-and-restarted pod never downloads.
    Run once per job, before any torchrun ranks exist.
    """
    out = Path(stage.output_dir)
    final = out / "final"
    if (final / "modules.json").is_file():
        return StagePlan(done=final, resume_from=None)
    if stage.hub_repo and hub.fetch_folder(stage.hub_repo, "final", final):
        return StagePlan(done=final, resume_from=None)
    local = latest_complete_checkpoint(out)
    if local is not None:
        return StagePlan(done=None, resume_from=local)
    if stage.hub_repo:
        staging = out / "hub-last-checkpoint"
        if hub.fetch_folder(stage.hub_repo, "last-checkpoint", staging):
            step = json.loads((staging / "trainer_state.json").read_text())["global_step"]
            ckpt = out / f"checkpoint-{step}"
            staging.rename(ckpt)
            return StagePlan(done=None, resume_from=ckpt)
    return StagePlan(done=None, resume_from=None)


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def per_device_batch_size(stage: StageConfig) -> int:
    ws = world_size()
    if stage.batch_size % ws:
        raise ValueError(f"batch_size={stage.batch_size} is not divisible by {ws} GPUs")
    return stage.batch_size // ws


def dataloader_workers(stage: StageConfig) -> int:
    if stage.dataloader_num_workers is not None:
        return stage.dataloader_num_workers
    local_ranks = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    # Tokenisation runs in the collator, so workers are what keeps a 16k-row batch off the step time.
    return max(2, min(16, available_cpus() // local_ranks - 1))


def probe_mini_batch_size(
    model: SentenceTransformer, seq_len: int, bf16: bool, upper: int = 4096, headroom: float = 0.85
) -> int:
    """Largest power-of-two GradCache chunk whose forward+backward at full ``seq_len`` fits on this GPU,
    leaving room for AdamW state (8 bytes/param) that is not allocated yet.

    Sized at the worst-case length so a batch of long files cannot OOM hours into the run.
    """
    if not torch.cuda.is_available():
        logger.warning("no CUDA device; mini_batch_size=auto falls back to 32")
        return 32
    device = model.device
    total = torch.cuda.get_device_properties(device).total_memory
    optimizer_bytes = 8 * sum(p.numel() for p in model.parameters() if p.requires_grad)
    vocab = model.tokenizer.vocab_size
    best, size = 0, 1
    while size <= upper:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            ids = torch.randint(5, vocab, (size, seq_len), device=device)
            feats = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                emb = model(feats)["sentence_embedding"]
            emb.float().sum().backward()
            fits = torch.cuda.max_memory_allocated(device) + optimizer_bytes < headroom * total
        except torch.OutOfMemoryError:
            fits = False
        finally:
            model.zero_grad(set_to_none=True)
        if not fits:
            break
        best, size = size, size * 2
    torch.cuda.empty_cache()
    if best == 0:
        raise RuntimeError(f"a single sequence of {seq_len} tokens does not fit; lower max_seq_length")
    logger.info("mini_batch_size=auto -> %d at seq_len %d", best, seq_len)
    return best


def _save_final(model: SentenceTransformer, final: Path, model_cfg: ModelConfig) -> None:
    tmp = final.with_name(final.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    # The stage's training length would otherwise ship as the model's limit, cutting long code that the
    # base handles (the first dry-run's Defects4J "regression" was exactly that 256-token cut).
    model.max_seq_length = inference_length(model, model_cfg)
    save_model(model, tmp, model_cfg)
    if final.exists():
        shutil.rmtree(final)
    tmp.rename(final)


def run_stage(
    stage: StageConfig,
    device: str | None = None,
    require_audit: bool = True,
    resume_from: Path | None = None,
    callbacks: list[TrainerCallback] | None = None,
) -> Path:
    configure_tracking(stage)
    model_cfg = ModelConfig.from_yaml(stage.model)
    if stage.max_seq_length is not None:
        model_cfg.max_seq_length = stage.max_seq_length
    if device is None and world_size() > 1 and torch.cuda.is_available():
        device = f"cuda:{os.environ['LOCAL_RANK']}"
    model = init_model(model_cfg, stage.init_from, device=device)
    model.max_seq_length = model_cfg.max_seq_length

    train_dataset, families = load_mix(
        stage.data_dir,
        require_audit=require_audit,
        allow_eval_overlap=stage.allow_eval_overlap,
        audited_against=full_audit_tasks() if stage.audit_scope == "full" else (),
    )
    evaluator = build_dev_evaluator(stage.dev_suite) if stage.dev_suite else None
    if stage.mini_batch_size != "auto":
        mini_batch = stage.mini_batch_size
    elif stage.loss == "cached_mnrl":
        mini_batch = probe_mini_batch_size(model, model_cfg.max_seq_length, stage.bf16)
    else:
        mini_batch = 0  # distill_kl has no GradCache; the whole per-device batch must fit
    # Token budgets bound memory only when padding is gone; with padded inputs a chunk of short rows
    # could still be padded to one long row.
    flattened = bool(getattr(model[0], "can_flatten_inputs", False))
    num_tokens = (
        mini_batch * model_cfg.max_seq_length
        if stage.pack_mini_batches_by_tokens and flattened and mini_batch > 0
        else None
    )
    logger.info("unpadded inputs: %s, mini batch: %d rows / %s tokens", flattened, mini_batch, num_tokens)
    workers = dataloader_workers(stage)
    cuda = torch.cuda.is_available()

    args = SentenceTransformerTrainingArguments(
        output_dir=stage.output_dir,
        run_name=stage.name,
        per_device_train_batch_size=per_device_batch_size(stage),
        learning_rate=stage.learning_rate,
        max_steps=stage.max_steps,
        num_train_epochs=stage.num_train_epochs,
        warmup_ratio=stage.warmup_ratio,
        lr_scheduler_type=stage.lr_scheduler_type,
        bf16=stage.bf16,
        # Same results, faster kernels; both need a GPU.
        tf32=True if cuda else None,
        optim="adamw_torch_fused" if cuda else "adamw_torch",
        gradient_checkpointing=stage.gradient_checkpointing,
        # Duplicate texts inside one batch become false in-batch negatives; the hashed variant
        # keeps that check O(1) per sample on multi-million-row sources.
        batch_sampler=BatchSamplers.NO_DUPLICATES_HASHED,
        # Each batch is drawn from a single source, so the model cannot separate positives by
        # spotting which dataset a text came from.
        multi_dataset_batch_sampler=MultiDatasetBatchSamplers(stage.multi_dataset_batch_sampler),
        prompts=training_prompts(families),
        dataloader_drop_last=True,
        dataloader_num_workers=workers,
        dataloader_persistent_workers=workers > 0,
        dataloader_prefetch_factor=4 if workers > 0 else None,
        dataloader_pin_memory=cuda,
        ddp_find_unused_parameters=False,
        logging_steps=stage.logging_steps,
        save_steps=stage.save_steps,
        save_total_limit=stage.save_total_limit,
        push_to_hub=stage.hub_repo is not None,
        hub_model_id=stage.hub_repo,
        # Also uploads the newest full checkpoint (optimizer, scheduler, RNG) to `last-checkpoint/`.
        hub_strategy="checkpoint",
        hub_private_repo=True,
        eval_strategy="steps" if evaluator else "no",
        eval_steps=stage.eval_steps if evaluator else None,
        seed=stage.seed,
        report_to=stage.report_to,
        # Otherwise Trainer moves the model to MPS/CUDA regardless of `device`.
        use_cpu=device == "cpu",
    )
    guard = None
    if stage.collapse_guard is not None:
        if evaluator is None:
            raise ValueError("collapse_guard needs a dev_suite to score against")
        guard = CollapseGuard(stage.collapse_guard)
    loss = build_loss(stage, model_cfg, model, mini_batch, num_tokens)
    trainer = StageTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=per_source_loss(loss, list(train_dataset), stage.source_loss_weights),
        evaluator=evaluator,
        callbacks=[LogRunMetadata(stage, model_cfg), *([guard] if guard else []), *(callbacks or [])],
        meta=model_meta(model_cfg),
    )
    if guard is not None and evaluator is not None:
        # Scored before any weights are loaded, so a resumed run still compares to the untrained model.
        with trainer.accelerator.autocast():
            guard.baseline = float(evaluator(model)["sequential_score"])
        logger.info(
            "collapse guard: untrained dev score %.4f, stop below %.4f",
            guard.baseline,
            guard.ratio * guard.baseline,
        )
    if resume_from is not None:
        logger.info("resuming from %s", resume_from)
    trainer.train(resume_from_checkpoint=str(resume_from) if resume_from else None)
    # The no-duplicates sampler gives up quietly (one warning_once) when a source's rows share too many
    # texts, and the epoch then ends early. Short by more than the last partial batches = data starvation.
    planned, done = trainer.state.max_steps, trainer.state.global_step
    stopped_by_guard = guard is not None and guard.tripped is not None
    if not stopped_by_guard and done < stage.min_step_fraction * planned:
        raise TrainingShortfall(
            f"trained {done} of {planned} planned steps; the no-duplicates sampler could not fill batches "
            "(too many shared texts per source, e.g. mined negatives); see the sampler warning in the log"
        )
    if guard is not None and guard.tripped is not None:
        step, score = guard.tripped
        raise TrainingCollapsed(
            f"dev score {score:.4f} at step {step} is below {guard.ratio} x untrained {guard.baseline:.4f}; "
            "no final model saved"
        )

    final = Path(stage.output_dir) / "final"
    if trainer.is_world_process_zero():
        _save_final(model, final, model_cfg)
        if stage.hub_repo:
            hub.push_folder(final, stage.hub_repo, "final", message=f"{stage.name} final")
        logger.info("saved %s", final)
    return final
