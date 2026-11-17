"""Round 7 (R10): why does vLLM teacher parity stay at 0.976 on code_contests_xlang after clipping?

On the archived 10x data, clips 128 rows to the teacher window and scores them with the cross-encoder in bf16
(its default, the checkpoint dtype) and in fp32, then with vLLM. If bf16 vs fp32 reference already disagree
about as much as vLLM vs reference, the gap is bf16 precision on long pairs, not the vLLM backend.
"""

from __future__ import annotations

import gc
import json
import tarfile
import traceback
from typing import Any

import numpy as np

from ablate import ROOT, log, results, upload

DATA_REPO = "divakantpandey/ce3-ablate-data"
ARCHIVE = "dryrun_ablate_x10.tar"
SOURCES = ("code_contests_xlang", "swe_smith", "cornstack_python")
ROWS = 128
TEACHER = "Qwen/Qwen3-Reranker-4B"


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(a.ravel(), b.ravel()).statistic)


def main() -> None:
    import torch
    from huggingface_hub import hf_hub_download
    from sentence_transformers import CrossEncoder

    from code_embed_v3.data.build import read_manifest
    from code_embed_v3.data.mining import (
        clip_to_teacher_window,
        crossencoder_scorer,
        label_source,
        load_dataset_dir,
        vllm_scorer,
    )

    results["round"] = 7
    try:
        tar = hf_hub_download(DATA_REPO, ARCHIVE, repo_type="dataset", local_dir=str(ROOT.parent / "archive"))
        with tarfile.open(tar) as fh:
            fh.extractall(ROOT.parent, filter="data")
        mined = ROOT / "mined"
        meta = read_manifest(mined)["sources"]
        work = ROOT / "parity"
        scores: dict[str, dict[str, np.ndarray]] = {s: {} for s in SOURCES}
        samples: dict[str, Any] = {}
        for dtype in ("bfloat16", "float32"):
            teacher = CrossEncoder(TEACHER, model_kwargs={"dtype": dtype})
            log(f"reference {dtype}: model dtype {next(teacher.model.parameters()).dtype}")
            for s in SOURCES:
                if s not in samples:
                    ds = load_dataset_dir(mined / s)
                    samples[s] = clip_to_teacher_window(ds.select(range(min(ROWS, len(ds)))), teacher.tokenizer)
                    tok = teacher.tokenizer
                    q = [len(x) for x in tok(list(samples[s]["anchor"]), add_special_tokens=False)["input_ids"]]
                    cols = [c for c in samples[s].column_names if c != "anchor"]
                    d = [[len(x) for x in tok(list(samples[s][c]), add_special_tokens=False)["input_ids"]] for c in cols]
                    scores[s]["pair_tokens"] = np.asarray(q)[:, None] + np.asarray(d).T
                ref = crossencoder_scorer(teacher, batch_size=8)
                scores[s][dtype] = np.asarray(label_source(samples[s], ref, meta[s]["family"], work / dtype / s)["label"])
            del teacher, ref
            gc.collect()
            torch.cuda.empty_cache()
        fast = vllm_scorer(TEACHER, num_gpus=1)
        out: dict[str, Any] = {}
        for s in SOURCES:
            v = np.asarray(label_source(samples[s], fast, meta[s]["family"], work / "vllm" / s)["label"])
            bf, fp, n = scores[s]["bfloat16"], scores[s]["float32"], scores[s]["pair_tokens"]
            long = n >= 1024
            row = {
                "pairs": int(n.size),
                "long_pairs(>=1024 tok)": int(long.sum()),
                "ref_bf16_vs_ref_fp32": spearman(bf, fp),
                "vllm_vs_ref_bf16": spearman(v, bf),
                "vllm_vs_ref_fp32": spearman(v, fp),
                "max_abs_diff_bf16_fp32": float(np.abs(bf - fp).max()),
                "max_abs_diff_vllm_fp32": float(np.abs(v - fp).max()),
                "mean_abs_diff_vllm_fp32_short": float(np.abs(v - fp)[~long].mean()) if (~long).any() else None,
                "mean_abs_diff_vllm_fp32_long": float(np.abs(v - fp)[long].mean()) if long.any() else None,
                "mean_abs_diff_bf16_fp32_short": float(np.abs(bf - fp)[~long].mean()) if (~long).any() else None,
                "mean_abs_diff_bf16_fp32_long": float(np.abs(bf - fp)[long].mean()) if long.any() else None,
            }
            out[s] = row
            log(f"parity {s}: {json.dumps(row)}")
        results["parity"] = out
    except Exception:  # noqa: BLE001 - top-level boundary: log, upload what exists
        results["fatal"] = traceback.format_exc()[-3000:]
        log(f"FATAL\n{results['fatal']}")
    results["done"] = True
    upload()
