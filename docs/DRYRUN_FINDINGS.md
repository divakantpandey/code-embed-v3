# Dry-run findings (2026-10-29/30) - hand-off for fix and design work

> Follow-up investigation of P0 is tracked in `RESEARCH_LOG.md` (same folder). Resolved 2026-10-30: the
> collapse came from **float16 master weights** (GTE-ModernBERT's checkpoint is fp16, transformers loads that
> dtype by default, and `build_model` passed none). Fixed in `modeling.py`; the same local run went from
> exec@10 0.001 to 0.957. Two dry-run/pilot mismatches (7 vs 4 mined negatives, uncapped SWE twins) were
> also found and fixed. The recipe and behaviour-data questions below must be re-run with the fix.

Observations only. Nothing below was tuned or redesigned; the one code fix made along the way is listed under "Changes made".
"Measured" means read from pod logs, W&B or eval JSON this session. "Hypothesis" means not tested.

## Verdict

**NO-GO for `make pilot` as configured.** The pipeline plumbing works end to end, but the fine-tune config
(`configs/stages/150m_ft_behaviour.yaml`) destroys the model in the dry-run. Fix the training instability first.

## What ran

- Pod `dwqm6vtduzs1t5` (`ce3-dryrun`): A40 48 GB, Secure Cloud, CA-MTL-1, $0.59/h, 9 vCPU (7 usable), image
  `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, host CUDA 13.2, 50 GB container disk, 50 GB persistent `/workspace`.
  Status now EXITED (stopped). Runtime about 69 min, roughly $0.7 (estimate, not a billing read).
- Code shipped as a tarball in private HF repo `divakantpandey/code-embed-v3-src`; pod bootstrap downloads it with `HF_TOKEN`.
- Secrets on RunPod: `HF_TOKEN` (id `t3b5r0ngj2`), `WANDB_API_KEY` (id `fe29e2twzm`).
- Community 4090 had no stock. Secure 4090 is $0.89/h with LOW stock, A40 Secure $0.59/h with HIGH stock.

## Dry-run step results (A40, 300 rows/source)

| Step | Result |
|---|---|
| doctor, build-data, synth x4, build-synth, zero-shot, attention (flash-attn2 via `kernels`, unpadded, 1.13x), mine (26 s) | pass |
| audit | pass after fix, **505 s** for ~3k rows |
| behaviour-build (vLLM Qwen2.5-Coder mutants/rewrites + sandbox + SWE-smith twins) | pass, 390.6 s |
| audit-behaviour (ExecRetrieval + CoQuIR texts), behaviour-combine | pass, 111 s / 3.7 s |
| train, kill at checkpoint-40, resume from Hub, restore deleted run from Hub | pass (142 s, 11 s) |
| behaviour-eval, mteb (CosQA, SyntheticText2SQL), zero-shot gate 100% on all 3 benchmarks | pass (run by hand after the failure below) |
| **label-check** | **FAIL** (see P1). label, train-s3, release were never run. |

## P0 - fine-tune collapses the model

Same eval harness, base vs dry-run model (120 steps, batch 16, max_seq 256, config `150m_ft_behaviour.yaml`):

| | GTE-ModernBERT base | dry-run ft | ft, hardness weighting off (40 steps) |
|---|---|---|---|
| ExecRetrieval exec@1 | 0.097 | 0.000 | 0.001 |
| exec@10 | 0.967 | 0.003 | 0.003 |
| beats_paired_distractors | 0.063 | 0.024 | 0.028 |
| CoQuIR Defects4J pairwise / ndcg@10 | 0.60 / 0.814 | 0.232 / 0.005 | 0.308 / 0.014 |
| mteb CosQA | 0.427 | 0.000 | not run |
| mteb SyntheticText2SQL | 0.646 | 0.0005 | not run |

exec@10 of 0.003 equals random over the 4,694-document corpus. The harness is sound (base scores are sane).

W&B run `ce3-dryrun-ft-390772c6` (project `code-embed-v3-dryrun`, https://wandb.ai/divakantpandey-ai/code-embed-v3-dryrun/runs/ce3-dryrun-ft-390772c6):
- loss 18.7 (step 0, lr 0) -> 103 (step 13) with grad_norm up to ~2,800, then flat at **25.94** with grad_norm ~0.1-3
  from about step 27 on. Flat loss plus near-zero gradient is consistent with representation collapse.
- dev accuracy@1: code2code 0.121 (step 72) -> 0.030 (step 133); code2nl 0.033 -> 0.022.

Config in that run: `temperature 0.015`, `hardness_mode hard_negatives`, `hardness_strength 5.0`, `lr 2e-5`,
`cached_mnrl`, Matryoshka dims [768,512,256,128,64], bf16, batch 16 (toy).

Ablation (one only): hardness weighting off, otherwise identical, 40 steps. Step-1 loss falls from ~22 to 7.6, but the
loss still spikes to 33 at step 4 (grad_norm 1,136) and the final model is also collapsed. So hardness weighting inflates
the loss but is **not** the sole cause.

Correction to an earlier remark: I first read the step-0 loss of 18.7 as "chance level, so the loss is broken".
With hardness off the step-1 loss is 7.6, which is plausible as a sum over 5 Matryoshka dims against near-identical twin
negatives. Treat the high initial loss as expected, not as evidence of a bug.

Not tested (hypotheses, none verified):
1. lr 2e-5 is too high for the sharp objective (grad norm >100 even at lr ~ 0); try 5e-6 / 1e-5.
2. tau 0.015 (scale 66.7) is too sharp against twins with cos ~0.95+; try 0.02 / 0.05.
3. Batch 16 is a bad proxy for the real batch 512 (cached MNRL); the collapse may not occur at 512. Needs a run with a
   larger batch (the dry-run has only a few thousand rows, so 512 means few steps).
4. Matryoshka loss sums 5 dims, so effective gradient is ~5x; check `matryoshka_weights`.
5. Train/eval path mismatch: confirm that the query prompts used at eval (`nl2code`, etc.) are also applied in
   training, and that the training forward (flash-attn2 unpadded, CLS pooling) matches `load_model` at step 0.
6. Twins in the same batch as each other's negatives (`NO_DUPLICATES` sampler vs near-duplicate anchors).
7. Gradient clipping: confirm `max_grad_norm` is actually applied (loss spiked at grad_norm >1,000).

Suggested decisive experiments (cheap, each about 3-4 min on the pod): (a) step-0 embedding parity train-path vs
`load_model` on the same texts; (b) lr x tau grid with hardness off, 40 steps, check exec@10 after each;
(c) the best cell at a larger batch.

## Gate numbers to know

Base GTE-ModernBERT exec@1 is **0.097** on this harness, not the 20-30% quoted earlier in the conversation (that figure
was unverified and is wrong for this metric). `pilot-gate` requires +0.10 exec@1, i.e. roughly doubling it.
Base CoQuIR Defects4J pairwise is 0.60.

## P1 - other defects found

- **label-check GPU memory** (only affects s3 distillation, not the pilot): `_cmd_label_check` in `cli.py` loads
  `CrossEncoder(teacher)` in-process (holds ~13.5 GiB on the A40), then launches a vLLM subprocess that wants 92% of the
  GPU -> `ValueError: Free memory on device cuda:0 (30.87/44.42 GiB) ... less than desired GPU memory utilization (0.92)`.
  Fix idea: compute all reference labels first, free the model (`del`, `gc.collect()`, `torch.cuda.empty_cache()`), then run vLLM.
- **`/workspace` is NFSv4.1** (`ca-mtl-1-vast.runpod.net:/runpodfs/podvolumes/...`, 47 GB). Consequences seen:
  uv cache cannot live there (`Invalid cross-device link`); deleting memory-mapped Arrow files leaves `.nfsXXXX` stubs so
  `rmtree` fails with ENOTEMPTY. Other renames/atomic saves may hit the same class of problem (training checkpoint dirs
  worked in the dry-run).
- **Disk**: 24 GB of 47 GB used after the dry-run (Qwen coder ~15 GB cached). `ce3 doctor` demands 200 GB free, so it
  fails on this pod; the dry-run itself uses `--min-free-gb 30`. The pilot pod needs a bigger volume or a lower threshold.
- **Cost assumptions in the docs are stale**: they assume the Community 4090 at $0.34/h. Currently only the Secure 4090
  ($0.89/h, low stock) or Secure A40 ($0.59/h) are available.

## P2 - observations

- Audit throughput is poor for tiny inputs: ~12-30 s per source for 30-270 rows (process-pool + index pickling overhead).
  The real `make data` audits millions of rows; measure before trusting the "CPU pod, tens of GB RAM" estimate.
- The dry-run installs vLLM 0.32.0 on demand via `uv run --with vllm==0.32.0` (199 packages) inside `behaviour-build`
  and `label-check`; flashinfer JIT compile adds minutes on first use.
- Behaviour data volume from the dry-run was not collected (`dryrun/behaviour/stats.json` is on the stopped pod's volume).
  Rewrite pass parsed 33 of 38 seeds. Real yields (human near-misses, mutant keep rate, SWE-smith twins per 5k rows) are unmeasured.
- About half of the SWE-smith twins start mid-function (hunk has no def header); only whole-function extraction from the
  repo at the base commit would fix that. ~3.3k twins per 20k SWE-smith rows after filtering.
- Dev-set caveat: SWE-smith issues also sit in the replay mix; `behaviour-combine` now drops replay rows sharing a dev anchor.
- Prior session-level unverified claims (voyage/jina scores, "+0.5-1.8 MTEB points", mLateOn, "100% zero-shot" statements)
  remain unverified. The zero-shot gate itself reported 100/100/100 for the dry-run model.

## Changes made this session (all uncommitted in the working tree)

- `src/code_embed_v3/data/behaviour.py`: SWE-smith twins now `-` side = correct (positive), `+` side = buggy; single-file,
  single-hunk, single-definition only; context trimmed to the changed function; comment/docstring tells filtered.
- `src/code_embed_v3/data/sources.py`: `reverse_patch` + `swe_smith` loader (replay source trains issue -> fixing patch).
  `private/configs/data/mix_v1.yaml` switched to `loader: swe_smith`.
- `src/code_embed_v3/cli.py`: `behaviour-combine` drops replay rows that share a dev anchor; `pilot-gate` adds
  `--max-coquir-drop` (default 0.01); `_replace_dir` tolerates undeletable backups on NFS and clears a stale backup first.
- `src/code_embed_v3/eval/behaviour_eval.py`: `load_coquir` skips rows whose text is NaN (Defects4J, SQLR2); `pilot_gate`
  checks mean CoQuIR pairwise drop.
- Tests added in `tests/test_behaviour.py` and `tests/test_data.py`; last full run: **125 passed**, ruff and mypy clean.
- `private/docs/*.md`: SWE-smith direction wording corrected.
- Not changed: `150m_ft_behaviour.yaml` (still has tau 0.015, hardness 5.0, lr 2e-5), `label-check`, docs cost numbers.

## Artifacts and leftovers

- Pod `dwqm6vtduzs1t5` is **stopped and cannot currently be restarted**: `pod-action start` returned 400 "not enough free
  GPUs on the host machine" (another renter took the A40 after I stopped it). Its 50 GB persistent `/workspace` (with
  `dryrun/` outputs and HF/model caches) is host-local, so it is unreachable until a GPU frees up, and it bills at the
  stopped-disk rate until the pod is terminated (terminating deletes it). Options: retry `start` later; or terminate and
  rebuild on a new pod (the dry-run takes about 25 min; the ft model also exists on the Hub as `ce3-dryrun-ft`).
  The pod's saved start command is the `[rest]` script (label, train-s3, release, summary), not the full dry-run.
  Lesson: do not stop a pod with a host-local volume while more work is planned; leaving it idle was the safer choice.
- **Still unobserved** because of the above: `label`, `train-s3`, `release`, and the final `dry_run_report.json`.
  The vLLM teacher path (`label`) is untested apart from the label-check failure.
- Private HF repos created: `divakantpandey/code-embed-v3-src` (code tarball), `divakantpandey/ce3-dryrun-ft`,
  dataset `divakantpandey/ce3-dryrun-synth`. The dry-run may create `ce3-dryrun-release` on a full run.
- W&B: project `code-embed-v3-dryrun` (one run). Older projects `code-embed`, `pycode-embed` are unrelated.
- Local helper scripts (session scratchpad): `runpod_secrets.py`, `push_src.py`, tarball. The user added two exact-match
  Bash allow rules for them.
- Tarball lesson: `tar --exclude='./data'` also dropped `src/.../data` and `configs/data`; rebuild with explicit include
  lists (src, tests, configs, private, Makefile, pyproject.toml, uv.lock, ...). Verified file counts match disk.
- Monitoring lesson: container stdout only shows a dry-run step after it finishes; per-step output is in
  `dryrun/logs/*.log` on the volume. I once mistook a slow audit for a hang and restarted a healthy run. A background
  `[progress]` tailer in the bootstrap script fixes the visibility.

## Open decisions for the design pass

1. Training recipe: lr, tau, hardness weighting, Matryoshka weighting, batch size - then re-check exec@10 and CosQA
   against the base numbers above before any spend on a full pilot.
2. Hardware: A40 Secure ($0.59/h, 48 GB) vs 4090 Secure ($0.89/h); decide after throughput is measured at real batch size.
3. Whether the pilot gate should also require "no collapse" (e.g. MTEB(Code) subset within X of base) before looking at exec@1.
4. Whether to fix `label-check` now or defer with s3 distillation.
