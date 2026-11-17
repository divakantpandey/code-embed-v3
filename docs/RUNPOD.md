# Running on RunPod

## Recommended first run: Lean Behaviour Pilot (`make pilot`, ~USD 2.50–3.00)

**Why this replaces the ~USD 25 / ~USD 200 runs as the starting gate:**
- `Alibaba-NLP/gte-modernbert-base` (149M, Apache-2.0) already scores **71.66 on MTEB(Code, v1)** and is 100% zero-shot. Training from scratch (`s1..s4`, ~USD 200) or mining 2.7M replay rows × 7 negatives (`24.3M` sequence passes at 1024 tokens) mostly re-learns general retrieval that the base model already knows.
- Instead, `make pilot` now defaults to a **lean replay buffer** (`MINE_ROWS=25000`, `MINE_NEGS=4`, `max_seq_length: 512`, `temperature: 0.015`) combined with **execution-verified CodeContests near-misses** and **zero-LLM SWE-smith pre/post-fix patch twins** (`SWE_ROWS=5000`). That cuts sequence passes by ~16x while testing the core novelty (`ExecRetrieval` +0.10 top-1 gain with $\le 0.5$ MTEB drop).

```bash
make dry-run                            # ~20 min rehearsal of every step at tiny scale
make data                               # stream + audit the zero-shot replay mix (run on a cheap CPU pod)
make pilot                              # lean replay mine + behaviour-build + train + ExecRetrieval/CoQuIR/MTEB gate
# Zero-LLM alternative (uses only human contest near-misses + SWE-smith patch twins; $0 vLLM cost):
make pilot MUTANTS=0 REWRITES=0
```

| Pilot Variant | Hardware | Wall Time | Est. GPU Cost |
|---|---|---:|---:|
| Zero-LLM (`MUTANTS=0 REWRITES=0`) | 1x RTX 4090 ($0.34/h) | ~4–5 h | **~USD 1.50** |
| Full Lean Pilot (with vLLM mutants/rewrites + SWE-smith twins) | 1x RTX 4090 ($0.34/h) | ~7–9 h | **~USD 2.50–3.00** |
| Full Lean Pilot | 2x RTX 4090 ($0.68/h) | ~3.5–4.5 h | **~USD 2.50–3.00** |

## Legacy Option A: Full-Scale Replay Fine-Tune (`MINE_ROWS=250000 MINE_NEGS=7`)

Start from `Alibaba-NLP/gte-modernbert-base` (Apache-2.0, 71.66 on MTEB Code v1 in LightOn's comparison) and run a hard-negative fine-tune on a family-balanced ~2.7M-row subset (`make finetune MINE_ROWS=250000 MINE_NEGS=7`). No stage 1, no teacher.

```bash
make dry-run            # ~30-45 min rehearsal of every step at tiny scale; read dryrun/dry_run_report.json
make data               # stream + audit the public mix (CPU-heavy; a CPU pod is cheapest)
make synth              # generate synthetic pairs on the GPUs, audit them, push
make tune               # optional: Optuna on the dev suite (~10 h, ~USD 13 on 4x 4090)
make finetune STAGE=tune/ce3-150m-ft-v1/best_stage.yaml   # or omit STAGE for the default config
make mteb-ft            # refuses to run if the model would not be 100% zero-shot
make release-ft         # model files + functional card only; repo stays private
```

| Setup | Time | GPU cost |
|---|---|---|
| 4x RTX 4090 (Community) | ~14 h | ~USD 19 |
| 1x RTX 4090 | ~55 h | ~USD 19 |
| 1x H100 PCIe | ~12 h | ~USD 24 |
| 1x B200 | ~6-8 h (small models run well below spec) | ~USD 35-45 |

Estimates are about 12 H100-equivalent GPU-hours. Expected score: about 72-73.5. Only add distillation (`make s3` on top) if the result lands below about 73.

## Hardware

**Default: RTX 4090, Community Cloud (USD 0.34/h).** It has the most bf16 compute per dollar (about 485 TFLOPS/USD, against about 380 for H100 PCIe and about 304 for RTX 5090). Faster cards cost more per unit of work, and a 150M model leaves big GPUs underused, so they buy time, not savings.

Use 4x 4090 to cut wall-clock time at about the same cost. Gradient sync is a few hundred MB per step, against steps of seconds to tens of seconds, so PCIe without NVLink is fine. `ce3 train` launches torchrun across every visible GPU and gathers negatives across devices.

Check after about 10 minutes: `perf/samples_per_sec` in W&B should be above about 75 per GPU on stage 1. Well below that means CPU starvation (see the CPU count printed by `make doctor`) or a slow host; resume on an H100 instead, since nothing is lost. That threshold is derived from the H100 estimate below, not measured.

Community Cloud has no network volumes. Pod disk survives a stop but not a terminate, and a stopped pod's GPU can be taken by someone else. The Hub copies (below) are what make that acceptable.

## What a lost pod costs

| Work | Where it is kept | Lost on pod death |
|---|---|---|
| Training stage | Every `save_steps`: local `checkpoint-N` plus the Hub repo `last-checkpoint/` (weights, optimizer, scheduler, RNG) | Up to one save interval (about 30 min; s1 saves every 50 steps) |
| Finished stage | Hub `<stage repo>/final/` | Nothing; `ce3 train` skips it |
| Mined / labelled data | Per source on disk; pushed to the Hub data repo after each step | One source being mined; one 20k-row shard per GPU being labelled |
| W&B | Stable run id per config, with `resume=allow` | Nothing; the run continues |

On a fresh pod, `make pull-data s1 s2 s3 s4` resumes from wherever the last pod stopped.

## Setup

```bash
cd /workspace && git clone <repo> code-embed-v3 && cd code-embed-v3
export HF_HOME=/workspace/.cache/huggingface
uv sync --all-extras
make doctor          # HF login, W&B key, GPUs, free disk, /workspace location
```

Put `HF_TOKEN` and `WANDB_API_KEY` (and optionally `WANDB_ENTITY`) in RunPod secrets rather than copying `.env` onto the pod. Locally, `ce3` reads `.env` without overriding real environment variables. It ignores `CUDA_VISIBLE_DEVICES` from the file and always sets `WANDB_PROJECT` from the stage config, so values left over from another project's `.env` cannot change GPU selection or the W&B project.

Run `make data` (build + MinHash audit) on a CPU pod. The audit loads every benchmark task's test data, so it needs tens of GB of RAM; on a GPU pod it would only waste GPU-hours.

`make synth` generates about 600k pairs with Qwen2.5-Coder-7B-Instruct (vLLM, one process per GPU). The estimate is about 3-4 h on 4x 4090, roughly USD 5-8. It is resumable per 2,000-row shard.

## Projected cost of the full pipeline (in H100 PCIe hours; multiply by about 4.6 for 4090 hours)

On 4090s the full pipeline is about USD 156; on one H100 PCIe at USD 1.99/h it is about USD 200.

These are FLOP-based estimates assuming about 35% bf16 utilisation and about 14M training pairs (the configured caps). Measure `perf/samples_per_sec` in the first 10 minutes and rescale.

| Step | Estimate (GPU-h) | Notes |
|---|---|---|
| Ablations (docs/PLAN.md) | 10-15 | On a ~2M-row subset; each about 1-1.5 h |
| s1 pretrain | ~9 | One epoch, batch 16k, 512 tokens |
| mine | ~3 | Encodes every query and positive once |
| s2 finetune | ~40 | Full mined mix, 1 + 7 negatives per row. **The largest training cost** |
| label (vLLM, Qwen3-Reranker-4B) | ~25 | `LABEL_ROWS=100000` per source (~1.1M rows x 8 pairs) |
| s3 distill | ~2.5 | |
| s4 long context | ~3 | |
| mteb | ~1.5 | |
| **Total** | **~95-100** | **About USD 190-200; budget USD 250-350 for reruns and slower-than-estimated throughput** |

Cost levers, in order of size (each needs an ablation before it is used for a final run, because each can cost accuracy):

- **s2 on a subset** (`ce3 mine ... --max-rows-per-source`): s2 cost scales linearly with rows.
- **Fewer label rows** (`LABEL_ROWS`): labelling cost scales linearly.
- **A cheaper GPU**, if the benchmark above shows better samples per dollar.

Storage: pod disk is USD 0.10/GB/month while running. Mined data with 7 negatives per row can reach roughly 100 GB or more; check your Hub plan's private storage limit before `push-data` of the mined set.

The 0.6B sibling costs roughly 4x the 150M pipeline. Start it only after the 150M dev-suite numbers justify it.
