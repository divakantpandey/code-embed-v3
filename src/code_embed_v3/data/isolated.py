"""Run a standalone vLLM worker script in its own uv environment, one process per GPU."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

# Pinned and isolated: vLLM pins older torch/transformers than this project.
VLLM_SPEC = "vllm==0.32.0"

logger = logging.getLogger(__name__)


def run_vllm_workers(
    worker: Path, jobs: Sequence[Mapping[str, object]], args: list[str], num_gpus: int, spec_dir: Path
) -> None:
    """Split ``jobs`` round-robin over GPUs; each worker gets ``--jobs <file>`` plus ``args``."""
    spec_dir.mkdir(parents=True, exist_ok=True)
    procs: list[tuple[subprocess.Popen[bytes], list[Mapping[str, object]]]] = []
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
        procs.append((subprocess.Popen(cmd, env=env), mine))
    failed = []
    for proc, mine in procs:
        if proc.wait() == 0:
            continue
        # vLLM 0.32 can abort in its own teardown ("terminate called without an active exception") after every
        # output is written; workers write outputs atomically, so complete outputs mean the work is done.
        if all(Path(str(j["output"])).is_file() for j in mine):
            logger.warning("vLLM worker exited %d after writing all outputs; keeping them", proc.returncode)
            continue
        failed.append(proc.args)
    if failed:
        raise RuntimeError(f"vLLM workers failed: {failed}")
