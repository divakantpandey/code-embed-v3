"""Round 2: is it the SWE-smith twins' share of batches or their content?

The dry-run now caps twins at --rows (300), like every other source. Compares, on that mix, the plain recipe and
the dry-run recipe with the twins at 300 rows vs removed, plus the best round-1 cell (lr 5e-6, tau 0.05).
"""

from __future__ import annotations

import json
import shutil
import sys
import traceback
from pathlib import Path

import ablate
from ablate import GTE, ROOT, DryRun, build, data_check, evaluate, log, results, run, upload

from code_embed_v3.data.build import read_manifest, write_manifest

results["round"] = 2


def without(src: Path, source: str) -> Path:
    dst = src.with_name(f"{src.name}_no_{source}")
    shutil.copytree(src, dst)
    manifest = read_manifest(dst)
    manifest["sources"].pop(source)
    write_manifest(dst, manifest)
    return dst


def main() -> None:
    if ROOT.exists():
        shutil.rmtree(ROOT)
    d = DryRun(root=ROOT, hub_user=None, wandb=True, rows=300, teacher="Qwen/Qwen3-Reranker-4B")
    try:
        _, combined = build(d)
        data_check(combined)
        results["base"] = evaluate(d, "base", GTE)
        log(f"RESULT base: {json.dumps(results['base'])}")
        upload()
        noswe = without(combined, "behaviour_swe_twins")

        single = {"matryoshka_dims": [768], "attn_implementation": "sdpa"}
        full = {"attn_implementation": "auto"}
        prod = dict(temperature=0.015, hardness_mode="hard_negatives", hardness_strength=5.0, warmup_ratio=0.05)
        run(d, "C2-plain-swe300", combined, single)
        run(d, "C3-plain-noswe", noswe, single)
        run(d, "P2-prod-swe300", combined, full, **prod)
        run(d, "P3-prod-noswe", noswe, full, **prod)
        run(d, "D2-lr5e-06-tau0.05-swe300", combined, full, learning_rate=5e-6, temperature=0.05)
    except Exception:  # noqa: BLE001 - top-level boundary: log, upload what exists
        results["fatal"] = traceback.format_exc()[-3000:]
        log(f"FATAL\n{results['fatal']}")
    results["done"] = True
    upload()
    log(f"DONE {json.dumps({n: r.get('eval', r.get('error')) for n, r in results['runs'].items()})}")


if __name__ == "__main__":
    assert ablate.STEPS == 60
    sys.exit(main())
