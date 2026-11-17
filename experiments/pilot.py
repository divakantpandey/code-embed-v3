"""Behaviour pilot (`make pilot` with a replay-only control arm), resumable on a pod with /workspace.

Each finished step leaves a marker in /workspace/pilot_state, so a restarted pod continues where it stopped
(training itself resumes from its Hub checkpoint). Logs and results go to the private src repo under
pilot/<PILOT_ID>/. The training batch size can be changed before training starts by uploading
pilot/<PILOT_ID>/batch_size.txt to that repo (the bridge round decides it).
"""

from __future__ import annotations

import itertools
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import yaml

REPO = "divakantpandey/code-embed-v3-src"
DATA_REPO = "divakantpandey/code-embed-v3-data"
GTE = "Alibaba-NLP/gte-modernbert-base"
PILOT_ID = os.environ.get("PILOT_ID", "pilot-1")
STATE = Path("/workspace/pilot_state")
LOGS = STATE / "logs"
B = "data/behaviour"
MINED = "data/mix_v1_gte_mined"
COMBINED = "data/mix_v1_gte_behaviour"
CONTROL = "data/mix_v1_gte_control"
STAGE = "configs/stages/150m_ft_behaviour.yaml"
# Encode batch for behaviour-eval and mteb at 8192 tokens with sdpa attention: 128 ran out of memory on the
# fp16 base and 64 on the fp32 fine-tune (one 16 GB attention allocation each).
EVAL_BATCH = os.environ.get("EVAL_BATCH", "32")
status: dict[str, Any] = {"pilot_id": PILOT_ID, "steps": {}}


def log(msg: str) -> None:
    print(f"[pilot] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def upload() -> None:
    from huggingface_hub import HfApi

    (STATE / "status.json").write_text(json.dumps(status, indent=2))
    try:
        HfApi().upload_folder(folder_path=str(STATE), repo_id=REPO, path_in_repo=f"pilot/{PILOT_ID}",
                              allow_patterns=["status.json", "logs/*.log", "results/**/*.json", "configs/*.yaml"],
                              commit_message=f"pilot {PILOT_ID}")
    except Exception as e:  # noqa: BLE001 - an upload failure must not stop the pilot
        log(f"upload failed: {e}")


def ce3(step: str, *args: str, ok_codes: tuple[int, ...] = (0,)) -> str:
    """Runs `ce3 <args>` once: skipped when its marker exists, raises on an unexpected exit code."""
    done = STATE / f"{step}.done"
    if done.exists():
        log(f"{step}: already done")
        return ""
    log(f"{step}: start")
    t0 = time.time()
    status["steps"][step] = {"state": "running", "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    upload()
    with (LOGS / f"{step}.log").open("w") as fh:
        proc = subprocess.Popen(["uv", "run", "ce3", *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        tail: list[str] = []
        for line in proc.stdout:
            fh.write(line)
            fh.flush()
            tail = (tail + [line])[-200:]
        code = proc.wait()
    minutes = round((time.time() - t0) / 60, 1)
    status["steps"][step] = {"state": "done" if code in ok_codes else "failed", "exit": code, "minutes": minutes}
    log(f"{step}: exit {code} in {minutes} min")
    upload()
    if code not in ok_codes:
        raise RuntimeError(f"{step} exited {code}:\n{''.join(tail[-40:])}")
    done.write_text(str(code))
    return "".join(tail)


def sample_utilisation() -> None:
    """Every 60 s: GPU util and VRAM, then this container's CPU use (cores busy vs quota) and RAM from its
    cgroup; /proc/loadavg and /proc/meminfo describe the whole host and say nothing about this pod."""
    from code_embed_v3.env import available_cpus

    cpus = available_cpus()
    cg = Path("/sys/fs/cgroup")

    def first(*paths: str) -> Path:
        return next(cg / p for p in paths if (cg / p).is_file())

    def cpu_usec() -> int:
        if (cg / "cpu.stat").is_file() and "usage_usec" in (cg / "cpu.stat").read_text():  # cgroup v2
            return int(next(ln.split()[1] for ln in (cg / "cpu.stat").read_text().splitlines() if ln.startswith("usage_usec")))
        return int(first("cpuacct/cpuacct.usage", "cpu,cpuacct/cpuacct.usage", "cpuacct.usage").read_text()) // 1000

    def ram_bytes() -> int:
        return int(first("memory.current", "memory/memory.usage_in_bytes").read_text())

    raw: dict[str, str] = {}
    for name in ("cpu.max", "cpu/cpu.cfs_quota_us", "cpu/cpu.cfs_period_us", "cpu,cpuacct/cpu.cfs_quota_us",
                 "cpuset.cpus.effective", "cpuset/cpuset.cpus"):
        if (cg / name).is_file():
            raw[name] = (cg / name).read_text().strip()
    status["cpu_budget"] = {"available_cpus": cpus, "affinity": len(os.sched_getaffinity(0)), "cgroup": raw}
    try:
        prev, t_prev = cpu_usec(), time.time()
    except (OSError, StopIteration, ValueError) as e:
        status["cpu_budget"]["sampler_error"] = str(e)
        return
    with (LOGS / "util.log").open("a") as fh:
        for tick in itertools.count(1):
            time.sleep(60)
            if tick % 5 == 0:  # steps can run for hours; keep the log on the Hub current
                upload()
            try:
                gpu = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=30,
                ).stdout.strip()
                now, t_now = cpu_usec(), time.time()
                busy = (now - prev) / 1e6 / (t_now - t_prev)
                prev, t_prev = now, t_now
                ram = ram_bytes() / 2**30
                step = next((k for k, v in status["steps"].items() if v.get("state") == "running"), "-")
                fh.write(f"{time.strftime('%H:%M:%S')} {step} gpu%,vram_mb,vram_total={gpu} "
                         f"cpu_busy={busy:.1f}/{cpus} ram_gb={ram:.1f}/{status.get('ram_limit_gb')}\n")
                fh.flush()
            except (OSError, subprocess.SubprocessError, ValueError, StopIteration) as e:
                fh.write(f"sample failed: {e}\n")


def memory_limit_bytes() -> int:
    """The container's RAM limit: cgroup memory.max, since /proc/meminfo reports the whole host."""
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        f = Path(path)
        if f.is_file():
            raw = f.read_text().strip()
            if raw.isdigit() and int(raw) < 1 << 60:
                return int(raw)
    return int(Path("/proc/meminfo").read_text().split()[1]) * 1024


def hub_overrides() -> dict[str, Any]:
    """pilot/<PILOT_ID>/stage_overrides.json (or the older batch_size.txt) from the src repo, read when a
    training config is first written, so the bridge result can change the recipe without a restart."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        return dict(json.loads(Path(hf_hub_download(REPO, f"pilot/{PILOT_ID}/stage_overrides.json", force_download=True)).read_text()))
    except EntryNotFoundError:
        pass
    try:
        return {"batch_size": int(Path(hf_hub_download(REPO, f"pilot/{PILOT_ID}/batch_size.txt", force_download=True)).read_text())}
    except EntryNotFoundError:
        return {}


def stage(config: str, data_dir: str, **overrides: Any) -> str:
    raw = yaml.safe_load(Path(STAGE).read_text())
    raw.update(data_dir=data_dir, dev_suite=data_dir, **overrides)
    raw.update(hub_overrides())
    path = STATE / "configs" / f"{config}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():  # keep the first choice so a resumed run trains with the same config
        path.write_text(yaml.safe_dump(raw))
    status["configs"] = {**status.get("configs", {}), config: yaml.safe_load(path.read_text())}
    return str(path)


def control_mix() -> None:
    """The combined mix without behaviour sources: the same (subsampled) replay rows, for a fair control."""
    from code_embed_v3.data.build import DEV_DIR, read_manifest, write_manifest

    out = Path(CONTROL)
    if (out / "manifest.json").exists():
        return
    sources = read_manifest(COMBINED)["sources"]
    kept = {n: m for n, m in sources.items() if not n.startswith("behaviour_")}
    for n, m in kept.items():
        m.setdefault("path", str((Path(COMBINED) / n).resolve()))
        dev = Path(COMBINED) / DEV_DIR / n
        if dev.exists():
            shutil.copytree(dev, out / DEV_DIR / n, dirs_exist_ok=True)
    write_manifest(out, {"sources": kept})
    log(f"control mix: {sorted(kept)}")


def base_repo_format() -> str:
    """Base GTE with this repo's module stack (adds Normalize), so it merges with the fine-tune."""
    out = Path("runs/150m/base_repo_format")
    if not (out / "modules.json").exists():
        from code_embed_v3.config import ModelConfig
        from code_embed_v3.modeling import build_model, save_model

        cfg = ModelConfig.from_yaml("configs/models/gte-modernbert-150m.yaml")
        cfg.max_seq_length = 8192
        save_model(build_model(cfg, device="cpu"), out, cfg)
    return str(out)


def results(name: str) -> str:
    p = STATE / "results" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    prev = STATE / "status.json"
    if prev.exists():
        status.update(json.loads(prev.read_text()))
        status.pop("fatal", None)
    status["boot"] = time.strftime("%Y-%m-%d %H:%M:%S")
    # The fine-tune's eval OOM'd with 12.8 GB reserved but unallocated: fragmentation, not real demand.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # /workspace is a network filesystem: datasets (mixes, benchmark corpora) under a quarter of RAM are read
    # once into memory instead of memory-mapped, so shuffled training reads never hit the network. Every
    # ce3 subprocess inherits this.
    limit = memory_limit_bytes()
    os.environ.setdefault("HF_DATASETS_IN_MEMORY_MAX_SIZE", str(limit // 4))
    status["ram_limit_gb"] = round(limit / 2**30, 1)
    status["in_memory_max_gb"] = round(int(os.environ["HF_DATASETS_IN_MEMORY_MAX_SIZE"]) / 2**30, 1)
    threading.Thread(target=sample_utilisation, daemon=True).start()
    try:
        # Replay mix: capped well above the 25k rows/source mined, then the full contamination audit.
        ce3("build-data", "build-data", "configs/data/mix_v1.yaml", "--max-rows", "30000")
        ce3("audit-mix", "audit", "data/mix_v1")
        ce3("push-mix", "push-data", "data/mix_v1", "--repo", DATA_REPO)
        ce3("zero-shot-mix", "zero-shot", "data/mix_v1", "--base-model", GTE)
        ce3("mine", "mine", "data/mix_v1", "--model", GTE, "--no-query-prompts", "--num-negatives", "4",
            "--max-rows-per-source", "25000", "--output-dir", MINED)
        ce3("push-mined", "push-data", MINED, "--repo", DATA_REPO)
        ce3("behaviour-build", "behaviour-build", "--out-dir", B, "--max-problems", "5000", "--mutants", "6",
            "--rewrites", "2", "--swe-smith-rows", "5000")
        status["behaviour_stats"] = json.loads(Path(B, "stats.json").read_text())
        ce3("audit-behaviour", "audit", B)
        out = ce3("combine", "behaviour-combine", MINED, B, "--output-dir", COMBINED, "--behaviour-share", "0.11")
        if out:
            status["combine"] = [ln for ln in out.splitlines() if "rows" in ln or "share" in ln]
        ce3("zero-shot-combined", "zero-shot", COMBINED, "--base-model", GTE)
        ce3("train-ft", "train", stage("ft", COMBINED))
        ce3("eval-base", "behaviour-eval", GTE, "--batch-size", EVAL_BATCH, "--out", results("behaviour/gte.json"))
        ce3("eval-ft", "behaviour-eval", "runs/150m/ft_behaviour/final", "--batch-size", EVAL_BATCH, "--out", results("behaviour/ft_behaviour.json"))
        ce3("mteb-base", "mteb", GTE, "--name", GTE, "--benchmarks", "MTEB(Code, v1)", "--batch-size", EVAL_BATCH, "--output", results("mteb-gte"))
        ce3("mteb-ft", "mteb", "runs/150m/ft_behaviour/final", "--name", "divakantpandey/code-embed-v3",
            "--benchmarks", "MTEB(Code, v1)", "--batch-size", EVAL_BATCH, "--data-dirs", "data/mix_v1", B, "--base-model", GTE,
            "--output", results("mteb-ft-behaviour"), "--dtype", "float16")
        gate = ce3("gate", "pilot-gate", "--baseline", results("behaviour/gte.json"),
                   "--candidate", results("behaviour/ft_behaviour.json"), "--mteb-baseline", results("mteb-gte"),
                   "--mteb-candidate", results("mteb-ft-behaviour"), ok_codes=(0, 1))
        if gate:
            status["gate"] = gate[-3000:]
        upload()
        # Control arm: identical replay rows and recipe, no behaviour data.
        control_mix()
        ce3("train-control", "train", stage("control", CONTROL, name="ce3-150m-ft-control",
                                            output_dir="runs/150m/ft_control",
                                            hub_repo="divakantpandey/code-embed-v3-150m-ft-control"))
        ce3("eval-control", "behaviour-eval", "runs/150m/ft_control/final", "--batch-size", EVAL_BATCH, "--out", results("behaviour/ft_control.json"))
        ce3("mteb-control", "mteb", "runs/150m/ft_control/final", "--name", "divakantpandey/code-embed-v3-control",
            "--benchmarks", "MTEB(Code, v1)", "--batch-size", EVAL_BATCH, "--data-dirs", "data/mix_v1", "--base-model", GTE,
            "--output", results("mteb-ft-control"), "--dtype", "float16")
        # v3 candidates: weight-average the pilot model with base GTE to buy back MTEB(Code).
        base = base_repo_format()
        for w in (0.5, 0.7):
            tag = f"merge{round(w * 10)}"
            out = f"runs/150m/{tag}"
            ce3(tag, "merge", "runs/150m/ft_behaviour/final", base, "--weights", str(w), str(round(1 - w, 1)),
                "--output", out)
            ce3(f"eval-{tag}", "behaviour-eval", out, "--batch-size", EVAL_BATCH, "--out", results(f"behaviour/{tag}.json"))
            ce3(f"mteb-{tag}", "mteb", out, "--name", f"divakantpandey/code-embed-v3-{tag}",
                "--benchmarks", "MTEB(Code, v1)", "--batch-size", EVAL_BATCH, "--data-dirs", "data/mix_v1", B,
                "--base-model", GTE, "--output", results(f"mteb-{tag}"), "--dtype", "float16")
        status["done"] = True
    except Exception:  # noqa: BLE001 - top-level boundary: record, upload, leave the pod for inspection
        status["fatal"] = traceback.format_exc()[-4000:]
        log(f"FATAL\n{status['fatal']}")
    upload()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.exit(main())
