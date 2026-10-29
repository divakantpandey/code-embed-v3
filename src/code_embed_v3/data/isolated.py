"""Run a standalone vLLM worker script in its own uv environment, one process per GPU."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

# Pinned and isolated: vLLM pins older torch/transformers than this project.
VLLM_SPEC = "vllm==0.32.0"


def run_vllm_workers(
    worker: Path, jobs: Sequence[Mapping[str, object]], args: list[str], num_gpus: int, spec_dir: Path
) -> None:
    """Split ``jobs`` round-robin over GPUs; each worker gets ``--jobs <file>`` plus ``args``."""
    spec_dir.mkdir(parents=True, exist_ok=True)
    procs: list[subprocess.Popen[bytes]] = []
    for gpu in range(num_gpus):
        mine = list(jobs[gpu::num_gpus])
        if not mine:
            continue
        spec = spec_dir / f"jobs-gpu{gpu}.json"
        spec.write_text(json.dumps([dict(j) for j in mine]))
        env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        cmd = [
            "uv", "run", "--no-project", "--python", f"{sys.version_info.major}.{sys.version_info.minor}",
            "--with", VLLM_SPEC, "--with", "pyarrow",
            "python", "-I", str(worker), "--jobs", str(spec), *args,
        ]  # fmt: skip
        procs.append(subprocess.Popen(cmd, env=env))
    failed = [p.args for p in procs if p.wait() != 0]
    if failed:
        raise RuntimeError(f"vLLM workers failed: {failed}")
