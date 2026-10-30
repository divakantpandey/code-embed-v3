"""Run untrusted Python programs against stdin/stdout tests.

The programs are contest solutions and LLM-written variants of them, so they are untrusted. Each run
gets a fresh temp directory, an empty environment, no site-packages (``-S``), CPU, memory, file-size
and core limits, its own process group (killed on timeout) and a wall-clock timeout. That bounds
accidents, not attacks: there is no network or filesystem isolation here, so this refuses to run
outside a container unless explicitly allowed.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

ALLOW_ENV = "CE3_ALLOW_UNTRUSTED_EXEC"


class SandboxRefusedError(RuntimeError):
    pass


def require_isolation() -> None:
    """Only execute inside a container (RunPod pod, Docker, Podman) or with an explicit opt-in."""
    if os.environ.get(ALLOW_ENV) == "1":
        return
    if os.environ.get("RUNPOD_POD_ID") or Path("/.dockerenv").exists() or Path("/run/.containerenv").exists():
        return
    raise SandboxRefusedError(
        f"refusing to execute untrusted code outside a container; run on the pod, or set {ALLOW_ENV}=1 "
        "inside a disposable, network-less container"
    )


@dataclass(frozen=True)
class Test:
    input: str
    output: str


@dataclass(frozen=True)
class Outcome:
    passed: int
    failed: int  # wrong output, crash or timeout

    @property
    def all_passed(self) -> bool:
        return self.failed == 0 and self.passed > 0

    @property
    def near_miss(self) -> bool:
        """Runs and gets some tests right, but not all: wrong behaviour, not broken code."""
        return self.passed > 0 and self.failed > 0


# The child applies its own limits before running the program: preexec_fn is unsafe with the
# thread pool in evaluate_many (CPython documents possible deadlocks).
LAUNCHER = (
    "import resource, runpy, sys\n"
    "def _cap(r, v):\n"
    "    _, h = resource.getrlimit(r)\n"
    "    c = v if h == resource.RLIM_INFINITY else min(v, h)\n"
    "    try:\n"
    "        resource.setrlimit(r, (c, c))\n"
    "    except (ValueError, OSError):\n"
    "        if sys.platform != 'darwin' or r != resource.RLIMIT_AS:\n"
    "            raise\n"
    "cpu, mem = int(sys.argv[1]), int(sys.argv[2]) << 20\n"
    "_cap(resource.RLIMIT_CPU, cpu)\n"
    "_cap(resource.RLIMIT_AS, mem)\n"
    "_cap(resource.RLIMIT_FSIZE, 16 << 20)\n"
    "_cap(resource.RLIMIT_CORE, 0)\n"
    "sys.argv = sys.argv[3:]\n"
    "runpy.run_path(sys.argv[0], run_name='__main__')\n"
)


def _tokens_match(got: str, expected: str, tol: float = 1e-6) -> bool:
    """Whitespace-insensitive, with float tolerance: contest judges accept either."""
    a, b = got.split(), expected.split()
    if len(a) != len(b):
        return False
    for x, y in zip(a, b, strict=True):
        if x == y:
            continue
        try:
            fx, fy = float(x), float(y)
        except ValueError:
            return False
        if abs(fx - fy) > tol * max(1.0, abs(fy)):
            return False
    return True


def run_once(code: str, stdin: str, timeout: float, memory_mb: int, workdir: Path) -> str | None:
    """stdout of one run, or None on crash, timeout or limit."""
    script = workdir / "main.py"
    script.write_text(code)
    cpu = max(1, int(timeout) + 1)
    proc = subprocess.Popen(
        [sys.executable, "-I", "-S", "-c", LAUNCHER, str(cpu), str(memory_mb), str(script)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        cwd=workdir,
        env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0", "LANG": "C.UTF-8"},
        start_new_session=True,
    )
    try:
        out, _ = proc.communicate(stdin.encode(), timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        return None
    if proc.returncode != 0:
        return None
    return out.decode(errors="replace")


def evaluate(
    code: str,
    tests: Sequence[Test],
    timeout: float = 4.0,
    memory_mb: int = 512,
    stop: str = "never",
) -> Outcome:
    """Run ``code`` on ``tests``. ``stop="first_fail"`` short-cuts all-pass checks; ``"mixed"`` stops
    once both a pass and a fail were seen, which is all a near-miss check needs."""
    require_isolation()
    passed = failed = 0
    with tempfile.TemporaryDirectory(prefix="ce3-sbx-") as tmp:
        for test in tests:
            out = run_once(code, test.input, timeout, memory_mb, Path(tmp))
            if out is not None and _tokens_match(out, test.output):
                passed += 1
            else:
                failed += 1
                if stop == "first_fail":
                    break
            if stop == "mixed" and passed and failed:
                break
    return Outcome(passed, failed)


def evaluate_many(
    jobs: Sequence[tuple[str, Sequence[Test], float]], workers: int, stop: str = "never"
) -> list[Outcome]:
    """``(code, tests, timeout)`` jobs in parallel. Threads suffice: each run is a subprocess, so the
    GIL is idle while programs execute."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda job: evaluate(job[0], job[1], timeout=job[2], stop=stop), jobs))
