"""Pilot-1 replay-only control on a 2-GPU box, resumable.

Rebuilds the pilot's combined mix from Hub copies (mined replay + behaviour data), refuses to train unless
every source's row count matches the pilot's status.json, then derives the replay-only control mix exactly as
pilot.py did. GPU 0 trains the control (resuming from the Hub checkpoint) and then scores it; GPU 1 re-scores
base GTE and the pilot fine-tune meanwhile, then runs MTEB(Code) on the control. Paired comparisons at the end.
Everything lands in the private src repo under pilot/pilot-1-control/.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

REPO = "divakantpandey/code-embed-v3-src"
DATA_REPO = "divakantpandey/code-embed-v3-data"
SYNTH_REPO = "divakantpandey/code-embed-v3-synth"
FT_REPO = "divakantpandey/code-embed-v3-150m-ft-behaviour"
GTE = "Alibaba-NLP/gte-modernbert-base"
OUT_ID = "pilot-1-control"
STATE = Path("/workspace/control_state")
LOGS = STATE / "logs"
MIX, MINED, B = "data/mix_v1", "data/mix_v1_gte_mined", "data/behaviour"
COMBINED, CONTROL = "data/mix_v1_gte_behaviour", "data/mix_v1_gte_control"
EVAL_BATCH = os.environ.get("EVAL_BATCH", "32")
status: dict[str, Any] = {"id": OUT_ID, "steps": {}}
lock = threading.Lock()


def log(msg: str) -> None:
    print(f"[control] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def upload() -> None:
    from huggingface_hub import HfApi

    with lock:
        (STATE / "status.json").write_text(json.dumps(status, indent=2))
    try:
        HfApi().upload_folder(folder_path=str(STATE), repo_id=REPO, path_in_repo=f"pilot/{OUT_ID}",
                              allow_patterns=["status.json", "logs/*.log", "results/**/*.json", "configs/*.yaml"],
                              commit_message=f"{OUT_ID}")
    except Exception as e:
        log(f"upload failed: {e}")


def ce3(step: str, *args: str, gpu: str | None = None, ok_codes: tuple[int, ...] = (0,)) -> str:
    done = STATE / f"{step}.done"
    if done.exists():
        return done.read_text()
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    with lock:
        status["steps"][step] = {"state": "running", "started": time.strftime("%Y-%m-%d %H:%M:%S"), "gpu": gpu}
    upload()
    t0 = time.time()
    log(f"{step}: ce3 {' '.join(args)} (gpu {gpu})")
    with open(LOGS / f"{step}.log", "w") as f:
        p = subprocess.run(["uv", "run", "ce3", *args], stdout=f, stderr=subprocess.STDOUT, env=env)
    out = (LOGS / f"{step}.log").read_text()
    with lock:
        status["steps"][step] = {"state": "done" if p.returncode in ok_codes else "failed", "exit": p.returncode,
                                 "minutes": round((time.time() - t0) / 60, 1), "gpu": gpu}
    upload()
    if p.returncode not in ok_codes:
        raise RuntimeError(f"{step} exited {p.returncode}; tail:\n{out[-2000:]}")
    done.write_text(out[-20000:])
    return out


def fetch(repo: str, sub: str, dest: str, repo_type: str) -> None:
    from code_embed_v3 import hub

    if not Path(dest, "manifest.json").exists() and not hub.fetch_folder(repo, sub, dest, repo_type=repo_type):
        raise RuntimeError(f"missing {repo}/{sub}")


def expected_counts() -> dict[str, int]:
    from huggingface_hub import hf_hub_download

    pilot = json.loads(Path(hf_hub_download(REPO, "pilot/pilot-1/status.json")).read_text())
    counts = {}
    for line in pilot["combine"]:
        m = re.match(r"(\w+): (\d+) rows \(", line)
        if m:
            counts[m.group(1)] = int(m.group(2))
    return counts


def check_combined(out: str) -> None:
    got = {m.group(1): int(m.group(2)) for m in re.finditer(r"^(\w+): (\d+) rows \(", out, re.M)}
    want = expected_counts()
    status["combine_check"] = {"expected": want, "got": got}
    if got != want:
        raise RuntimeError(f"combined mix differs from the pilot: {set(want.items()) ^ set(got.items())}")
    log("combined mix matches the pilot row for row per source")


def control_mix() -> None:
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
    status["control_rows"] = sum(int(m.get("rows", 0)) for m in kept.values())


def control_config() -> str:
    from huggingface_hub import hf_hub_download

    path = STATE / "configs" / "control.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        shutil.copy(hf_hub_download(REPO, "pilot/pilot-1/configs/control.yaml"), path)
    return str(path)


def ft_model() -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(FT_REPO, allow_patterns=["*.json", "*.safetensors", "1_*/*", "2_*/*", "tokenizer*"],
                             ignore_patterns=["last-checkpoint/*"])


def res(name: str) -> str:
    p = STATE / "results" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def gpu0() -> None:
    ce3("train-control", "train", control_config(), gpu="0")
    ce3("eval-control", "behaviour-eval", "runs/150m/ft_control/final", "--batch-size", EVAL_BATCH,
        "--out", res("behaviour/ft_control.json"), gpu="0")


def gpu1() -> None:
    ce3("eval-base", "behaviour-eval", GTE, "--batch-size", EVAL_BATCH, "--out", res("behaviour/gte.json"), gpu="1")
    ce3("eval-ft", "behaviour-eval", ft_model(), "--batch-size", EVAL_BATCH,
        "--out", res("behaviour/ft_behaviour.json"), gpu="1")
    while not (Path("runs/150m/ft_control/final") / "modules.json").is_file():
        time.sleep(60)
    ce3("mteb-control", "mteb", "runs/150m/ft_control/final", "--name", "divakantpandey/code-embed-v3-control",
        "--benchmarks", "MTEB(Code, v1)", "--batch-size", EVAL_BATCH, "--data-dirs", MIX, "--base-model", GTE,
        "--output", res("mteb-ft-control"), "--dtype", "float16", gpu="1")


def run_parallel(*fns: Any) -> None:
    errors: list[str] = []

    def wrap(fn: Any) -> None:
        try:
            fn()
        except Exception:
            errors.append(traceback.format_exc()[-3000:])

    threads = [threading.Thread(target=wrap, args=(f,)) for f in fns]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise RuntimeError("\n---\n".join(errors))


def main() -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    prev = STATE / "status.json"
    if prev.exists():
        status.update(json.loads(prev.read_text()))
        status.pop("fatal", None)
    status["boot"] = time.strftime("%Y-%m-%d %H:%M:%S")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_DATASETS_IN_MEMORY_MAX_SIZE", str(64 * 2**30))
    try:
        fetch(DATA_REPO, "mix_v1", MIX, "dataset")
        fetch(DATA_REPO, "mix_v1_gte_mined", MINED, "dataset")
        fetch(SYNTH_REPO, "behaviour/data", B, "dataset")
        ce3("audit-behaviour", "audit", B)
        out = ce3("combine", "behaviour-combine", MINED, B, "--output-dir", COMBINED, "--behaviour-share", "0.11")
        check_combined(out)
        control_mix()
        upload()
        run_parallel(gpu0, gpu1)
        pairs = (("control_vs_base", "gte", "ft_control"), ("ft_vs_control", "ft_control", "ft_behaviour"),
                 ("ft_vs_base", "gte", "ft_behaviour"))
        for name, base, cand in pairs:
            ce3(f"compare-{name}", "behaviour-compare", "--baseline", res(f"behaviour/{base}.json"),
                "--candidate", res(f"behaviour/{cand}.json"), "--out", res(f"compare/{name}.json"))
        status["done"] = True
    except Exception:
        status["fatal"] = traceback.format_exc()[-4000:]
        log(f"FATAL\n{status['fatal']}")
    upload()
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.exit(main())
