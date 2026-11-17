"""v3 candidates on a fresh pod: weight-average the pilot model with base GTE, then behaviour-eval and MTEB(Code).

Needs only Hub artifacts (the pilot pod's host died with its volume). Reuses pilot.py's resumable steps;
status and results go to code-embed-v3-src under pilot/<PILOT_ID>/ (default pilot-1-merge).
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback

os.environ.setdefault("PILOT_ID", "pilot-1-merge")

import pilot  # noqa: E402
from pilot import DATA_REPO, EVAL_BATCH, GTE, LOGS, STATE, base_repo_format, ce3, log, results, status, upload  # noqa: E402

FT_REPO = "divakantpandey/code-embed-v3-150m-ft-behaviour"
FT = "runs/150m/ft_behaviour/final"
# MTEB runs the merges in fp16, where batch 64 fits at 8192 tokens (the fp16 base ran it); the behaviour
# eval keeps the fp32 checkpoint and EVAL_BATCH (32).
MTEB_BATCH = "64"


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    prev = STATE / "status.json"
    if prev.exists():
        status.update(json.loads(prev.read_text()))
        status.pop("fatal", None)
        status.pop("done", None)
    status["boot"] = time.strftime("%Y-%m-%d %H:%M:%S")
    # Steps finished on an earlier pod whose results are already on the Hub; a fresh volume has no markers.
    for step in filter(None, os.environ.get("DONE_STEPS", "").split(",")):
        (STATE / f"{step}.done").write_text("0")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    status["ram_limit_gb"] = round(pilot.memory_limit_bytes() / 2**30, 1)
    threading.Thread(target=pilot.sample_utilisation, daemon=True).start()
    try:
        if not os.path.isfile(f"{FT}/modules.json"):
            from huggingface_hub import snapshot_download

            snapshot_download(FT_REPO, allow_patterns=["final/*"], local_dir="runs/150m/ft_behaviour")
            log(f"fetched {FT_REPO}/final")
        # Only the manifest is used: mteb's training-data declaration for the zero-shot check.
        ce3("pull-mix", "pull-data", "data/mix_v1", "--repo", DATA_REPO)
        base = base_repo_format()
        tags = {f"merge{round(w * 10)}": w for w in (0.5, 0.7, 0.4)}
        for tag, w in tags.items():
            ce3(tag, "merge", FT, base, "--weights", str(w), str(round(1 - w, 1)), "--output", f"runs/150m/{tag}")

        def mteb(tag: str) -> None:
            ce3(f"mteb-{tag}", "mteb", f"runs/150m/{tag}", "--name", f"divakantpandey/code-embed-v3-{tag}",
                "--benchmarks", "MTEB(Code, v1)", "--batch-size", MTEB_BATCH, "--data-dirs", "data/mix_v1",
                "--base-model", GTE, "--output", results(f"mteb-{tag}"), "--dtype", "float16")

        def evaluate(step: str, model: str, out: str) -> None:
            ce3(step, "behaviour-eval", model, "--batch-size", EVAL_BATCH, "--out", results(f"behaviour/{out}.json"))

        def compare(tag: str) -> None:
            ce3(f"compare-{tag}", "behaviour-compare", "--baseline", results("behaviour/gte_pq.json"),
                "--candidate", results(f"behaviour/{tag}_pq.json"), "--out", results(f"significance/{tag}.json"))

        # Every step runs alone: one fp16 MTEB run holds the GPU at 100% and peaks at ~24 GB on long documents
        # (two ran out of memory on a 48 GB card), and one fp32 behaviour eval holds it at 100% in ~29 GB.
        for tag in ("merge5", "merge7"):
            mteb(tag)
            evaluate(f"eval-{tag}", f"runs/150m/{tag}", tag)
        # Per-query reruns for the paired significance test; the 0.5 merge first, it is the v3 candidate.
        # Plain GTE, as in the pilot's baseline: the repo-format copy adds this repo's query prompts.
        evaluate("eval-gte-pq", GTE, "gte_pq")
        evaluate("eval-merge5-pq", "runs/150m/merge5", "merge5_pq")
        compare("merge5")
        mteb("merge4")
        evaluate("eval-merge4-pq", "runs/150m/merge4", "merge4_pq")
        compare("merge4")
        status["done"] = True
    except Exception:  # noqa: BLE001 - top-level boundary: record, upload, leave the pod for inspection
        status["fatal"] = traceback.format_exc()[-4000:]
        log(f"FATAL\n{status['fatal']}")
    upload()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.exit(main())
