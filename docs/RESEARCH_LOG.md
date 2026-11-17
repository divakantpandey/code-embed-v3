# Research log: fine-tune collapse (P0 in DRYRUN_FINDINGS.md)

Running record of every investigation step: what was tried, what changed, what worked, what did not.
Newest entry last. "Measured" = read from eval JSON / logs this session. "Hypothesis" = not tested yet.

## Current status (2026-11-17, latest)

- **R16 (2026-11-17): the pilot-scale control is in. Closed as a negative result.** Near-miss data adds +1.4 points
  exec@1 over a replay-only control (p 0.21) and causes the AppsRetrieval collapse (control 53.9, pilot 18.9);
  replay fine-tuning alone costs ~2 MTEB(Code) points. No model released.
- **ROOT CAUSE FOUND (R6): the model trained with float16 master weights.** GTE-ModernBERT's checkpoint is
  float16, transformers 5 loads the checkpoint dtype by default, and `build_model` passed no dtype, so all
  134 trainable tensors were fp16. `bf16: true` only sets autocast; AdamW then ran on fp16 weights (eps 1e-8
  underflows, small updates round away), giving grad_norm 72 at lr 0 and collapse to the ln(#candidates)
  plateau within ~10-20 steps at any lr and on any data. Fixed in `modeling.py` (fp32 master weights unless
  a config sets `torch_dtype`), with a regression test that fails without the fix.
- **Every training result before R6 is invalid as evidence about data or recipe**: all runs had the bug.
  The data/sampler findings (R3 twins near-dups, R4 starvation) stand on their own; the "behaviour data
  causes collapse" and "lr/tau" readings do not, and must be re-run with the fix.
- **R7 (with the fix):** all runs healthy, including the configured recipe (best exec@10 0.969). Behaviour
  data adds a small, seed-consistent gain (D4J pairwise +0.017, beats_paired +0.005) but nothing lifts exec@1
  above base, and every fine-tune loses ~0.11 CoQuIR Defects4J pairwise, which the current pilot gate rejects.
  Open decisions: pilot-scale test of the behaviour effect, and whether the CoQuIR gate threshold is right.
- **R14: dry-run passes with every fix**; batched labelling 3.3x faster; models saved and evaluated at 8192;
  Defects4J ft 0.611 vs base 0.600 on GPU. Research data has no ExecRetrieval/Defects4J contamination (R13).
- **R12: the Defects4J drop was a 256-token truncation artefact** (fine-tune 0.609 vs base 0.606 at equal length);
  no pilot blocker. The saved/eval length mismatch is a real bug to fix.
- **R11 (2026-10-31): dry-run passes end to end** with all fixes; teacher parity min 0.9973. Behaviour data
  beats replay on every seed (exec@1 +0.029, beats_paired +0.028); Defects4J regression is the pilot blocker.
- **R8 10x (2026-10-31): behaviour data + prod recipe beats base on exec@1 (0.111 vs 0.097)** and the effect
  grows with scale; merging with base now hurts; D4J pairwise still ~0.11 below base. Open user decisions: the
  pilot gate's absolute `--min-exec-gain 0.10` and `--max-coquir-drop 0.01`.
- **R8 (paused 2026-10-30):** the full dry-run reached label-check: the fine-tune, kill/resume and Hub restore
  all pass. Two new issues fixed in `9ea27a4`: a vLLM teardown abort failed the step, and stale scratch Hub repos
  had let generation be skipped. Next: rerun the dry-run, then the 10x behaviour run (see the R8 resume checklist).

Earlier status (superseded, kept for the record):

- **The dry-run barely trained on replay data at all** (R4). With 7 mined negatives per row and only ~250 rows
  per source, the no-duplicates sampler cannot fill 16-row batches, so replay sources yield a handful of
  batches and training is mostly behaviour data. The replay-only run (A) stopped at step 7 of 60; ST only
  logged a `warning_once`. **Round 1's "replay trains fine" is not established** (A barely trained).
- **Behaviour near-miss data degrades the model fast when it is not diluted**: SWE twins collapse it even at
  300 rows (C2, P2); contest mutants alone took exec@10 0.97 -> 0.49 in ~4 batches (C3).
- Recipe (tau, hardness, lr, Matryoshka), flash-attn and the train/eval path are not the root cause.
- Pilot is still **NO-GO**. Next: a dry-run mix where replay actually fills batches, then re-test the
  behaviour data at a controlled share and with the new per-source loss weights.

## Baseline (fixed reference for every table)

GTE-ModernBERT base, `ce3 behaviour-eval --subsets Defects4J`. Reproduced exactly on two separate pods.

| exec@1 | exec@10 | beats_paired | Defects4J pairwise | Defects4J ndcg@10 |
|---|---|---|---|---|
| 0.097 | 0.967 | 0.063 | 0.600 | 0.814 |

---

## R0. Dry-run (2026-10-29/30)

See DRYRUN_FINDINGS.md. Outcome: pipeline works end to end; `150m_ft_behaviour.yaml` (tau 0.015, hardness
`hard_negatives` 5.0, lr 2e-5, 5 Matryoshka dims, batch 16 x 120 steps) produced a random-level model
(exec@10 0.003). Seven hypotheses listed there.

## R1. Code reading, no GPU (2026-10-30)

| Hypothesis (DRYRUN_FINDINGS numbering) | Finding | Status |
|---|---|---|
| 7. Gradient clipping not applied | `run_stage` never sets `max_grad_norm`; transformers 5.21.1 default is 1.0, so clipping is on. W&B `grad_norm` is pre-clip. | Ruled out |
| 4. Matryoshka sums 5 dims | `MatryoshkaLoss` built without `matryoshka_weights` -> 5 equal terms | Confirmed (effect tested in R2 via 1-dim runs) |
| 5. Train/eval path mismatch | `build_model` = Transformer + CLS Pooling + Normalize, same as base; prompts applied in train and eval | Ruled out except flash-attn backward (tested in R2) |
| - Warmup | `warmup_ratio 0.05` of 120 steps = 6 steps; full lr at step 6, loss spike at steps 4/13 | Noted |

## R2. Ablation round 1 (2026-10-30, pod `5g0cujs7eh8b1d`, A40, ~48 min, ~$0.50, terminated)

**Applied:** `ablate.py` (session scratchpad, shipped in the src tarball). Rebuilt the dry-run mix with the
same steps and sizes (300 rows/source, 4 synth tasks x 24, behaviour-build 40 problems), checked the data,
then trained short runs changing one factor at a time and scored each with `behaviour-eval`.
Common settings unless stated: 60 steps, batch 16, max_seq 256, warmup_ratio 0.2, no Hub pushes,
W&B project `code-embed-v3-dryrun`. "Plain" = tau 0.05, no hardness weighting, single dim [768].
Results: HF `divakantpandey/code-embed-v3-src` `ablate/20261030-131608/`.

### Data check (combined mix)

- No row in any source has a negative identical to its positive.
- Base model ranks negative_1 >= positive on <=0.4% of replay rows (cornstack, spider, synth, ...).
- On behaviour sources it is a coin flip: `behaviour_swe_twins` 55.7%, `behaviour_nl2code` 66.7%,
  `behaviour_code2code` 44.8% (these used raw dot products, not cosine; see mistake below).
- **`behaviour_swe_twins` had 4,800 rows vs 22-270 for every other source.** The dry-run never passed
  `--swe-smith-rows`, so the 5,000 default applied. With the proportional single-source sampler, about 60%
  of batches were pure twin batches. In the pilot (25k replay rows/source) twins are about 1%.
- Mistake: the `cos_*` columns were raw dot products (base GTE has no Normalize module), so they are
  meaningless. Fixed in `ablate.py` for round 2. `behaviour_eval` normalises, so eval numbers are unaffected.

### Runs

| Run | Data | Recipe | exec@1 | exec@10 | D4J pairwise | D4J ndcg@10 | Verdict |
|---|---|---|---|---|---|---|---|
| P-prod | combined | dry-run recipe (tau 0.015, hardness 5, lr 2e-5, 5 dims, flash) | 0.000 | 0.002 | 0.105 | 0.007 | Collapse reproduced |
| A | replay only | plain, sdpa, lr 2e-5 | 0.100 | 0.932 | 0.462 | 0.783 | Healthy |
| B | replay only | plain, flash-attn2 unpadded | 0.084 | 0.937 | 0.422 | 0.771 | Healthy, = A |
| C | combined | plain, sdpa | 0.000 | 0.002 | 0.161 | 0.010 | **Collapse** |
| D | combined | lr 5e-6, tau 0.02, 5 dims, flash | 0.045 | 0.528 | 0.432 | 0.451 | Degraded |
| D | combined | lr 5e-6, tau 0.05 | 0.069 | 0.667 | 0.409 | 0.609 | Degraded (best D) |
| D | combined | lr 1e-5, tau 0.02 | 0.014 | 0.267 | 0.398 | 0.330 | Degraded |
| D | combined | lr 1e-5, tau 0.05 | 0.021 | 0.310 | 0.409 | 0.397 | Degraded |
| E | combined | best D at batch 64 | 0.060 | 0.806 | 0.454 | 0.745 | Degraded, less |

### What it showed

> **Correction (R4):** A and B trained only 7 of 60 steps (sampler starvation) and C/D/E/P trained mostly
> on twins. The A/B "healthy" verdicts and the flash-attn comparison are weak evidence; the C-vs-A
> contrast is really "many twin steps" vs "7 steps". Original reading kept below for the record.

- **Worked / ruled out:** training pipeline, prompts, pooling, clipping, and flash-attn backward (A, B).
- **Root cause is the behaviour data in this mix:** C differs from A only by data and collapses as badly as P.
  The config's tau, hardness weighting and lr are not the root cause (hypotheses 1, 2 downgraded).
- **Improved but did not fix:** lower lr (5e-6) and a larger batch (64) slow the damage; no D/E cell reaches
  base on any metric. Higher tau (0.05) beat 0.02 at both learning rates.
- **New concern:** healthy A/B still drop D4J pairwise 0.60 -> 0.42-0.46 (ndcg holds at 0.77-0.78).
  `pilot-gate --max-coquir-drop 0.01` would fail any such model, regardless of exec@1.

## R3. Round 2 + twin near-duplicate check (2026-10-30, pod `grnzjxlhh1lz9j`, A40, EU-SE-1) - running

**Question:** do the SWE-smith twins hurt because of their share of batches or because of their content?

**Changed:**
- `src/code_embed_v3/dry_run.py`: behaviour-build now gets `--swe-smith-rows <rows>` (300), so twins are one
  source among equals, closer to pilot proportions.
- `ablate.py` data check now uses normalised embeddings.

**Runs** (`ablate2.py`): C2 plain with twins at 300 rows; C3 plain without twins; P2/P3 the dry-run recipe
with/without twins; D2 lr 5e-6 tau 0.05 with twins at 300.

**Local check** (`twin_dups.py`, no GPU): on the same 5,000 SWE-smith twins, count repeated anchors/positives
and near-duplicates across rows (another row's buggy code that is a near-copy of this row's positive would
be a false in-batch negative).

### Twin near-duplicate check (measured, local, base GTE, normalised cosine, 5,000 twins, seed 0)

| Metric | Value |
|---|---|
| Distinct anchors | 5,000 / 5,000 |
| Distinct positives | 4,763; 450 rows (9%) share their correct code with another row, max 5 per function |
| Positive equals another row's buggy code | 7 rows |
| cos(correct, own buggy twin) | mean 0.977; 67.5% above 0.98 |
| Rows with another row's buggy code at cos > 0.98 | 7.8% |
| Rows with another row's positive at cos > 0.98 | 11.5% |
| Anchors with a neighbour anchor at cos > 0.95 | 3.4% |
| Base ranks own buggy twin >= correct (issue text as query) | 55.1% |

Reading: cross-row near-duplicates exist but rarely meet in one 16-row batch drawn from ~4,800 rows, so
hypothesis 6 (twins as each other's false negatives) is weak. What every twin row does is ask the model to
separate two vectors at cos ~0.98 from an issue text that the base model cannot use (coin flip). When most
batches are such rows, that is the likely collapse driver; round 2 tests it.

Truncation check (measured, tokenizer only): twins are short (positive p50 69 / p90 146 tokens) and first
differ at token p50 30 / p90 50; **0 of 5,000** pairs become identical at 256 tokens. Ruled out.

### Round 2 runs (60-step budget, batch 16; "steps" = steps actually trained, from W&B `train/global_step`)

| Run | Data | Recipe | Steps | exec@1 | exec@10 | D4J pairwise | D4J ndcg@10 |
|---|---|---|---|---|---|---|---|
| C2 | twins capped at 300 | plain, sdpa | 27 | 0.001 | 0.002 | 0.211 | 0.009 |
| C3 | no twins | plain, sdpa | 11 | 0.039 | 0.492 | 0.385 | 0.603 |
| P2 | twins capped at 300 | dry-run recipe | 27 | 0.000 | 0.006 | 0.346 | 0.007 |
| P3 | no twins | dry-run recipe | 11 | 0.003 | 0.080 | 0.434 | 0.157 |
| D2 | twins capped at 300 | lr 5e-6, tau 0.05, 5 dims, flash | 27 | 0.047 | 0.623 | 0.460 | 0.484 |

Pod `grnzjxlhh1lz9j` (EU-SE-1), ~35 min, ~$0.35, terminated. Results: `ablate/<stamp>/` in the src repo.

- Capping twins to 300 did **not** prevent collapse (C2, P2). Share alone is not the explanation.
- Without twins (C3) the model still lost half its exec@10 in 11 steps, mid-warmup.
- The step counts differ wildly although every run asked for 60 steps; see R4.

## R4. Batch-sampler starvation (2026-10-30, W&B + local simulation)

**Observation (measured, W&B `code-embed-v3-dryrun`):** steps actually trained per run:
A 7, B 7, C 60, every D 60, E 60, P 60, C2 27, C3 11, P2 27. The original dry-run ft reached 120.
Runs dominated by twin data reach the budget; replay-heavy runs stop early.

**Mechanism:** `NO_DUPLICATES_HASHED` rejects a row if any of its texts is already in the batch. Mined
negatives are other rows' positives from the same source, so a mined row carries 8 texts drawn from the
same ~250-text pool. ST resamples for a few extra passes, then gives up with `warning_once`; the
proportional sampler skips exhausted sources, and Trainer ends the epoch (and the run) early.

**Simulation** (`sampler_sim.py`, ST 6.2 `NoDuplicatesBatchSampler`, drop_last, uniform random negatives):

| Scale | Negatives | Batches yielded / advertised |
|---|---|---|
| 250 rows, batch 16 | 7 | 3 / 15 |
| 250 rows, batch 16 | 4 | 15 / 15 |
| 250 rows, batch 16 | 1 | 15 / 15 |
| twins, 250 rows, batch 16 | 1 | 15 / 15 |
| 25k rows, batch 512 | 7 | 48 / 48 |
| twins 5k rows, batch 512 | 1 | 9 / 9 |

Real mined negatives cluster on "hub" positives, so real starvation is worse than this uniform simulation
(A got 7 steps from 16 replay sources; the simulation predicts ~48). **Pilot scale looks safe in the
simulation but is unmeasured on real mined data.**

**Consequences:**
- Round 1 twins were ~85% of actual batches, not ~60%.
- Run A/B "healthy" = 7 steps inside a 12-step warmup. Not evidence that replay training is safe.
- The dry-run never exercised the replay-dilution design of the pilot mix.

**What did not catch it:** the zero-shot gate, the dry-run itself, and ST's single `warning_once` line in
the train log.

**Second dry-run vs pilot mismatch found:** the pilot mines 4 negatives (`Makefile MINE_NEGS ?= 4`), the
dry-run used the `mine` default of 7. With 4, the simulation fills every batch at dry-run scale.

**Changed:**
- `dry_run.py` mines with `--num-negatives 4` (the pilot value).
- `train.py`: `TrainingShortfall` raised when fewer than `min_step_fraction` (default 0.95) of the planned
  optimizer steps ran; no final model is saved. Test: a toy mix with 4 shared negatives at batch 8 trains
  0 of 2 steps and raises.

## R5. Round 3 (2026-10-30, pod `3ik7swbm0cmnv2`, A40, CA-MTL-1) - running

**Question:** with replay actually filling batches, is plain fine-tuning safe, and can the behaviour data be
used without damage (diluted, down-weighted, or merged back toward base)?

**Applied** (`ablate3.py`): rebuilt the dry-run mix with 4 mined negatives and twins at 300 rows; archived the
built data to private dataset `divakantpandey/ce3-ablate-data` for reuse. All runs: 60 steps, batch 16,
5 Matryoshka dims, flash-attn, tau 0.05, no hardness, warmup 0.2, and the new shortfall check.

| Run | Purpose |
|---|---|
| A3 replay, lr 2e-5 / 5e-6 | Is fine-tuning on the replay buffer alone safe? (true version of round-1 A) |
| C4 combined, lr 2e-5 / 5e-6 | Behaviour data diluted by replay that fills batches |
| N4 combined without twins | Contest-mutant data alone |
| W4 behaviour sources x0.25 | New per-source loss weights |
| G4 dry-run recipe + `collapse_guard 0.5`, eval every 10 steps | Guard should stop a known-collapsing run early and refuse to save |
| M merges | Best healthy run averaged with base at fine-tune weight 0.5 and 0.7 |

Pod ~55 min, ~$0.55, terminated. All runs trained the full 60 steps (no shortfall).

| Run | exec@1 | exec@10 | beats_paired | D4J pairwise | D4J ndcg@10 |
|---|---|---|---|---|---|
| base | 0.097 | 0.967 | 0.063 | 0.600 | 0.814 |
| A3 replay, lr 2e-5 | 0.000 | 0.004 | 0.011 | 0.299 | 0.008 |
| A3 replay, lr 5e-6 | 0.042 | 0.687 | 0.049 | 0.413 | 0.657 |
| C4 combined, lr 2e-5 | 0.000 | 0.002 | 0.029 | 0.269 | 0.007 |
| C4 combined, lr 5e-6 | 0.055 | 0.613 | 0.073 | 0.445 | 0.673 |
| N4 no twins, lr 2e-5 | 0.000 | 0.001 | 0.018 | 0.262 | 0.005 |
| W4 behaviour x0.25, lr 2e-5 | 0.001 | 0.005 | 0.024 | 0.368 | 0.001 |
| G4 dry-run recipe + guard | stopped by guard at step 20 (dev 0.324 < 0.5 x 0.829); no model saved | | | | |
| M: A3-lr5e-6 merged with base, ft weight 0.5 | 0.060 | 0.874 | 0.078 | 0.490 | 0.744 |
| M: same, ft weight 0.7 | 0.043 | 0.789 | 0.058 | 0.449 | 0.724 |

**What it showed:**
- **Replay-only training collapses too** (A3 lr 2e-5), so behaviour data was never the root cause. This
  redirected the search to the training path (R6).
- **Worked:** the collapse guard stopped a known-bad run at step 20 and refused to save it. The shortfall
  check confirmed every run trained all 60 steps (4 mined negatives fix the starvation at this scale).
- **Improved, but on a broken base:** merging with base recovered exec@10 0.687 -> 0.874 and beats_paired
  0.049 -> 0.078 (above base 0.063). Mechanism sound; must be re-measured after the fp16 fix.
- W&B A3 lr 2e-5: grad_norm 71.7 at lr 0; loss 8.6 -> plateau 21.9 = 5 x ln(80), i.e. uniform similarity over
  all 16 x (1 + 4) candidates. This plateau signature is what pointed to a numerical, not data, cause.

## R6. Root cause: float16 master weights (2026-10-30, local Mac, no GPU spend)

**Procedure:**
1. Downloaded the round-3 data archive and re-ran A3 locally (MPS, fp32 autocast off, sdpa, no token
   packing): collapsed identically (exec@10 0.001), with **`grad_norm` NaN from the first logged step**.
   This excluded flash-attn, token packing and bf16 autocast in one run.
2. Data columns checked: only anchor/positive/negative_1..4, clean text.
3. Step-0 parity, `build_model` vs sentence-transformers loading of GTE: identical weights (134/134,
   max diff 0), cosine 1.0. Not a loading/mapping bug.
4. The parity log said "model was loaded in torch.float16": `build_model` params = 134 x float16; checkpoint
   `config.json` `torch_dtype: float16`; `cfg.torch_dtype` None, so no dtype was passed.

**Changed:** `src/code_embed_v3/modeling.py` `training_dtype()`: master weights float32 unless the model
config sets `torch_dtype` (the Qwen configs keep their explicit bfloat16). Applied in `build_model` and in
`init_model` for stages that start from a checkpoint.

**Verified:** `test_fp16_checkpoint_trains_with_fp32_master_weights` fails without the fix
(`{torch.float16} == {torch.float32}`) and passes with it.

**Local A3 re-run with the fix** (same data, config and seed; MPS, sdpa, fp32):

| | exec@1 | exec@10 | D4J ndcg@10 | Gradients | Loss over 60 steps |
|---|---|---|---|---|---|
| Before fix | 0.000 | 0.001 | 0.005 | NaN from step 5 | stuck at 0 (NaN weights) |
| **After fix** | 0.075 | **0.957** | 0.787 | finite (12-190) | 10.5 -> 3.2, falling |
| Base | 0.097 | 0.967 | 0.814 | | |

The collapse is gone. A small drop remains after 60 steps of batch-16 training on replay only, which is
expected noise at this scale and is what the real experiments now need to measure.

## R7. Round 4: first experiments with the fix (2026-10-30, pod `eb0zmme470osfb`, A40, CA-MTL-1) - running

**Applied** (`ablate4.py`): restored the round-3 data archive (no rebuild); code includes the fp32 fix.
120 steps, batch 16, 5 Matryoshka dims, flash-attn, collapse guard 0.5 (eval every 40 steps), shortfall check.

| Run | Purpose |
|---|---|
| replay-plain / combined-plain, seeds 42 and 43 | Central contrast: does behaviour data change beats_paired / exec@1? Two seeds to size noise |
| combined-prod, replay-prod | The configured dry-run recipe (tau 0.015, hardness 5) with the fix |
| no-twins-plain | Contest-mutant behaviour data only |
| combined-behaviour-x0.25 / x3 | Per-source loss weights, down and up |
| merges (ft weight 0.5) of combined-plain, combined-prod, replay-plain | Does base averaging keep gains and recover robustness? |

**First attempt failed (every cell, ~4 min of pod time):** `FlashAttention only support fp16 and bf16 data
type`. With fp32 master weights, any forward pass outside autocast now feeds fp32 into flash-attn. Training
steps run under the Trainer's bf16 autocast, but sentence-transformers calls the dev evaluator from
`evaluation_loop` without it, and the collapse-guard baseline also ran before training. The pre-fix pipeline
only worked because the weights were accidentally fp16.

**Changed:**
- `train.py`: `StageTrainer.evaluation_loop` runs under `self.accelerator.autocast()`; the guard baseline is
  scored under the same context after the trainer exists (`CollapseGuard` baseline defaults to NaN, which
  never trips).
- `modeling.py` `load_model`: when flash-attn is selected and no dtype is passed, load in bfloat16 (this
  repo now saves fp32 checkpoints; mining loads them with `attn_implementation="auto"`).
- `probe_mini_batch_size` already used autocast. 131 tests pass, ruff/mypy clean.

Uploaded the fixed tree and restarted the same pod's container (start command re-downloads the code),
avoiding a new provision.

**Results** (pod ~40 min incl. the failed attempt, ~$0.40, terminated). Every run trained all 120 steps, no
guard trips. "Plain" = tau 0.05, no hardness; "prod" = configured recipe (tau 0.015, hardness 5).

| Run | exec@1 | exec@10 | beats_paired | D4J pairwise | D4J ndcg@10 |
|---|---|---|---|---|---|
| base | 0.097 | 0.967 | 0.063 | 0.600 | 0.814 |
| replay plain, seed 42 | 0.071 | 0.958 | 0.072 | 0.488 | 0.785 |
| replay plain, seed 43 | 0.075 | 0.960 | 0.071 | 0.484 | 0.787 |
| combined plain, seed 42 | 0.075 | 0.963 | 0.076 | 0.503 | 0.802 |
| combined plain, seed 43 | 0.078 | 0.960 | 0.079 | 0.503 | 0.797 |
| combined prod | 0.077 | **0.969** | 0.077 | 0.480 | 0.792 |
| replay prod | 0.080 | 0.966 | **0.084** | 0.490 | 0.798 |
| no-twins plain (contest mutants only) | 0.068 | 0.964 | 0.068 | 0.488 | 0.793 |
| combined, behaviour x0.25 | 0.073 | 0.962 | 0.069 | 0.486 | 0.793 |
| combined, behaviour x3 | 0.077 | 0.962 | 0.073 | 0.492 | 0.798 |
| merge: combined plain s42 + base, 0.5 | **0.081** | 0.965 | 0.079 | 0.477 | 0.797 |
| merge: combined prod + base, 0.5 | **0.081** | 0.969 | 0.076 | 0.482 | 0.799 |
| merge: replay plain s42 + base, 0.5 | 0.073 | 0.968 | 0.075 | 0.471 | 0.789 |

Seed noise (replay plain, 2 seeds): <= 0.004 on every metric. Treat gaps below ~0.01 as noise.

**What worked:**
- **The fp32 fix holds on GPU with flash-attn**: every run is healthy; replay plain s42 exec@10 0.958 matches
  the local Mac run (0.957). The configured prod recipe no longer collapses and gives the best exec@10 (0.969).
- **Behaviour data (plain recipe) gives a small, seed-consistent gain over replay alone**: D4J pairwise
  +0.017 (0.503 vs 0.486, both seeds), beats_paired +0.005, exec@1 +0.003.
- **Base merging raises exec@1** for behaviour-trained models (0.075-0.077 -> 0.081) and keeps beats_paired.

**What did not:**
- **Nothing raises exec@1 above base** (best 0.081 vs 0.097). The pilot gate needs +0.10. At dry-run scale
  (120 steps x 16 rows) this is not evidence about the pilot, but no signal of the target effect yet.
- **No dose-response for behaviour weight**: x0.25 falls back to replay-only, but x3 (0.073 beats_paired) is
  not better than x1 (0.077). The behaviour effect is small at this scale.
- **Merging does not restore CoQuIR Defects4J pairwise** (0.47-0.48 vs base 0.60). Every fine-tune, merged or
  not, loses 0.10-0.13 pairwise; `pilot-gate --max-coquir-drop 0.01` would reject all of them.

**Mixed / single-seed, needs confirmation:**
- With the prod recipe, replay-only beat combined on beats_paired (0.084 vs 0.077): the opposite sign of the
  plain-recipe result. Hardness weighting itself may matter more for beats_paired than the behaviour data.
- Without SWE twins (contest mutants only) beats_paired drops to 0.068: under the plain recipe the small
  behaviour gain seems to come from the SWE twins, not the contest mutants.

## Plan written before R7 (kept for the record)

Re-run the questions the bug made unanswerable, on the archived round-3 data (no rebuild):
1. Dry-run recipe as configured (tau 0.015, hardness 5, lr 2e-5) vs plain, with the fix.
2. Replay-only vs combined behaviour mix: does behaviour data move `beats_paired` / exec@1 above base?
3. Per-source behaviour weights and base merging, measured on the target metrics, not only no-collapse.
4. Then the full dry-run (`make dry-run`) end to end with all fixes, including `label`, `train-s3`, `release`.

## R8. Round 5: full dry-run with every fix + behaviour at 10x scale (2026-10-30, pod `onn3q3flywx84r`, A40, CA-MTL-1) - running

Code committed first: branch `behaviour-pilot-fixes`, `2014373` (public) and `db93bbf` (private overlay), not
pushed. `collapse_guard: 0.5` added to `150m_ft_behaviour.yaml` (public example and private copy).

**Procedure.** One pod, script `ablate5.py` (session scratchpad), two parts:
1. `ce3 dry-run` exactly as `make dry-run` (Hub on, W&B on, 300 rows/source): doctor, build, synth, mine,
   behaviour, killed + resumed fine-tune with Hub restore, label-check, label, train-s3, behaviour eval
   (base + ft), mteb, release. Logs and reports to `code-embed-v3-src` under `dryrun/<stamp>/`.
2. 10x behaviour question: rebuild at 3,000 rows/source, behaviour-build with 400 CodeContests problems and
   3,000 SWE twins (10x R5/R7), 1,200 steps at batch 16 (10x R7), collapse guard 0.5 every 200 steps.
   Cells: replay vs combined under the plain and prod recipes (one seed each), then 50/50 base merges of the
   two combined runs.

**Setup problems hit before it ran (no results lost, ~$0.05):**
- Stale scratch repo: `divakantpandey/ce3-dryrun-ft` still held the collapsed first dry-run's `final/` and
  `last-checkpoint/`, so the resume/restore checks would have passed on old files. Renamed it to
  `ce3-dryrun-ft-20261029` (kept, not deleted). HF redirects an old name to the renamed repo, so a fresh
  empty private `ce3-dryrun-ft` was created to break the redirect.
- `ce3 doctor` refuses a pod without `/workspace` (repo and `HF_HOME` must persist): first pod had no volume.
  Recreated with a 100 GB persistent `/workspace`.
- On that volume `tar` cannot chown (needs `--no-same-owner`) and uv's cache cannot rename across devices
  (`UV_CACHE_DIR` must stay on container disk). Both are pod-bootstrap issues, not repo bugs; worth noting
  in `docs/RUNPOD.md`.

### Part 1 result: dry-run attempt 1 failed at `label-check` (34.7 min)

Measured (stdout + step logs, `code-embed-v3-src/dryrun/20261030-174122/`):

| Step | Result |
|---|---|
| doctor, build-data (36 s), synth x4 (225 s), build-synth, audit (**605 s**), zero-shot, attention (flash == sdpa, min cos 0.9998, x1.04), mine (61 s), behaviour-build (27 s), audit-behaviour (160 s), behaviour-combine | pass |
| fine-tune killed at checkpoint-40, resumed from the Hub (175 s), restored as finished on a "fresh pod" (48 s) | **pass** (fp32 fix + autocast eval hold through the real dry-run path) |
| label-check | **fail**: vLLM teacher scored all 80 pairs for `cornstack_java`, wrote the output, then aborted in vLLM's own teardown (`terminate called without an active exception`); the non-zero exit failed the step |
| label, train-s3, behaviour-eval, mteb, release | not reached |

What it showed:
- **P1 (vLLM OOM in label-check) is fixed**: after the cross-encoder is freed vLLM gets 32 GB KV cache (43/44 GB free at start).
- **New bug, fixed**: `run_vllm_workers` treated any non-zero exit as failure. Now a worker that wrote every
  output (written atomically, so a present file is complete) counts as done, with a warning. Regression test
  fails on the old code, passes on the new. Commit `9ea27a4`.
- **New gap, fixed**: synth and behaviour generation were *restored from the first dry-run's*
  `ce3-dryrun-synth` Hub dataset (`0/1 shards to generate`), so this dry-run never exercised vLLM generation;
  same for `ce3-dryrun-ft` before it was archived. `DryRun.run()` now refuses to start while any scratch repo
  (`synth`, `ft`, `s3`, `release`) holds files. Stale repos archived as `ce3-dryrun-synth-20261029` and
  `ce3-dryrun-ft-20261030`, fresh empty private ones created. Same commit.
- `audit` is the slowest step (10 min of a 35 min run) at 300 rows/source; worth a look before the pilot.

Part 2 (10x) was stopped early on purpose: paused for the day at 18:23, pod `onn3q3flywx84r` terminated
after build-data and synth-nl2code, so no 10x results. Round 5 spend about $0.70 in total.

### Resume checklist (paused 2026-10-30)

- Code: branch `behaviour-pilot-fixes` (`9ea27a4`), not pushed. The pod tarball
  `divakantpandey/code-embed-v3-src/code-embed-v3-src.tar.gz` already holds this code plus the ablation
  scripts (`ablate.py` .. `ablate5.py`); unpack it to recover the scripts if the local copies are gone.
- Scratch Hub repos `ce3-dryrun-synth` (dataset) and `ce3-dryrun-ft` are empty, as the new guard requires.
- Pod recipe that works: A40 Secure, `mounts.persistent` 100 GB at `/workspace`, repo + `HF_HOME` under
  `/workspace`, `UV_CACHE_DIR=/root/.cache/uv`, `UV_LINK_MODE=copy`, `tar --no-same-owner`, secrets
  `RUNPOD_SECRET_HF_TOKEN` / `RUNPOD_SECRET_WANDB_API_KEY`.
- Run order: `R5_PART=dryrun` first (dry-run only, ~45 min, checks the two fixes), then the 10x part
  (no env var; ~2-3 h).

### Resumed 2026-10-31: dry-run attempt 2 (pod `gi673gmlohffds`, A40, EU-SE-1)

Ran dry-run then 10x on one pod (no env var). Measured:
- Scratch repos empty, guard passed. **Generation now really ran**: synth x4 with vLLM 05:14-05:30, behaviour
  mutate/rewrite pushed 05:52 (attempt 1 had restored both from the stale repo).
- **Failed at the fine-tune kill/resume (57 min):** `ce3-dryrun-ft/last-checkpoint/trainer_state.json never
  appeared on the Hub`. The ft repo got only the step-20 *model* commit.
- Root cause (read in transformers 5.21.1 `Trainer._push_from_checkpoint`): each save starts two background
  uploads, model files and the full checkpoint (~1.8 GB with optimizer state) into `last-checkpoint/`. The
  step-40 save came 15 s after step 20 and was skipped because the step-20 push was still running
  (`hub_always_push=False`), and the dry-run SIGTERMed the process as soon as local `checkpoint-40` existed,
  aborting the only checkpoint upload in flight. **Every earlier "resume from Hub: pass" came from the stale
  `last-checkpoint/` of the 2026-10-29 run**, never from the run under test.
- Fix (`6b9cd8f`): the dry-run kills only after the Hub holds `last-checkpoint/trainer_state.json` (polled
  every 10 s), and the dry-run fine-tune runs 300 steps (was 120) so the upload can finish first. `hub_has`
  is unit-tested; the kill loop itself is only exercised on GPU.
- Implication for the pilot (not a bug): after a preemption, the Hub copy can be several saves behind,
  since in-flight uploads die with the pod and pushes are skipped while one is running. Resume still works,
  it just repeats more steps.
- Scratch repos archived again as `ce3-dryrun-{synth,ft}-20261031`; fresh empty ones created. The archived
  `ce3-dryrun-*-2026103*` repos can be deleted once the dry-run passes (user's call).
- The 10x part started at 06:11 on the same pod. Dry-run attempt 3 follows it (`R5_PART=dryrun`).

### Part 2 result: behaviour data at 10x scale (2026-10-31, same pod, results `ablate/20261031-051212/`)

**Procedure.** Rebuilt at 3,000 rows/source: 29,939 train rows; behaviour 3,340 rows (11%), of which SWE twins
2,756 (9%), contest nl2code 301 + code2code 283. Behaviour-build on 400 CodeContests problems: 78/712
mutants and 105/333 rewrites kept after sandbox verification (same ~11% / ~32% yield as at 40 problems).
1,200 steps, batch 16, lr 2e-5, collapse guard 0.5 every 200 steps (never tripped). One seed per cell.
Base reproduced exactly (exec@1 0.097, exec@10 0.967, beats_paired 0.063, D4J pairwise 0.600).

| Run (10x) | exec@1 | exec@10 | beats_paired | D4J pairwise | D4J ndcg@10 |
|---|---|---|---|---|---|
| base | 0.097 | 0.967 | 0.063 | 0.600 | 0.814 |
| replay, plain | 0.082 | 0.967 | 0.080 | 0.484 | 0.797 |
| replay, prod | 0.078 | 0.968 | 0.077 | **0.512** | 0.810 |
| combined, plain | 0.094 | 0.966 | 0.095 | 0.503 | 0.806 |
| **combined, prod** | **0.111** | **0.970** | **0.111** | 0.490 | 0.796 |
| combined plain, 50/50 merge with base | 0.082 | 0.971 | 0.081 | 0.497 | 0.804 |
| combined prod, 50/50 merge with base | 0.088 | 0.971 | 0.089 | 0.488 | 0.795 |

Behaviour-data effect (combined minus replay, same recipe):

| Recipe | exec@1 | beats_paired | D4J pairwise |
|---|---|---|---|
| plain, 10x | +0.012 | +0.015 | +0.019 |
| prod, 10x | **+0.033** | **+0.034** | -0.022 |
| plain, 1x (R7, 2 seeds) | +0.003 | +0.005 | +0.017 |

**What worked:**
- **The behaviour effect grows with scale** (exec@1 +0.003 at 1x to +0.012 at 10x under the plain recipe)
  and is far above R7's seed noise (<=0.004), though each 10x cell is one seed.
- **Behaviour data + prod recipe is the first configuration above base on exec@1**: 0.111 vs 0.097 (+0.014,
  +14% relative), with beats_paired 1.76x base and the best exec@10 so far (0.970).
- **The recipe and the data interact**: the prod recipe alone does nothing for exec@1 (replay prod 0.078 <
  replay plain 0.082); with behaviour data it nearly triples the behaviour gain. tau 0.015 + hardness 5 is
  what makes near-miss negatives informative. R7's "replay-only beats combined under prod" reversed at scale.

**What did not:**
- **Merging with base now hurts** (exec@1 0.111 -> 0.088, 0.094 -> 0.082). It only helped in R7 because the
  fine-tunes were below base; once the fine-tune carries the signal, averaging dilutes it. Drop merging from
  the pilot plan unless a later run is again below base.
- **CoQuIR Defects4J pairwise stays 0.09-0.12 below base in every run**, and behaviour data under prod
  costs another 0.022 there. Still fails `--max-coquir-drop 0.01` (user decision pending).
- **Far from the pilot gate**: `pilot_gate` defaults `--min-exec-gain 0.10` *absolute* (exec@1 >= 0.197).
  Best is +0.014. Either the gate is mis-set (a relative +10% would be 0.107, which combined-prod passes)
  or the effect must grow ~7x at pilot scale. Needs the user's call.

Checked: exec@1 and beats_paired are both exactly 104/939 for combined-prod. The two are independent
counters in `exec_retrieval_scores`, other runs differ, so this is a coincidence, not a bug.

### Dry-run attempt 3 (2026-10-31 08:04, same pod, `R5_PART=dryrun`, code `6b9cd8f`): failed at label-check (72 min)

- **Fine-tune kill / resume / Hub restore now genuinely pass**: `last-checkpoint/` and `final/` reached the
  fresh `ce3-dryrun-ft` (08:52), the resume came from this run's own checkpoint.
- **vLLM teardown fix works**: every teacher worker exited cleanly after scoring.
- **label-check failed on parity, min Spearman 0.957 < 0.99**: `code_contests_xlang` 0.957 (max abs diff 6.5),
  `synth_techqa` 0.984, `synth_nl2code` 0.988; the other 13 sources 0.991-0.9995.
- Root cause for the worst source (measured on R5 data, teacher tokenizer, 16 rows): the vLLM worker clips
  queries to 512 and documents to 2048 tokens, while the cross-encoder reference has no limit
  (`model_max_length` 131072). `code_contests_xlang`: 4/16 queries over 512 (max 1,964), 5/80 documents over
  2048. `swe_smith`: 1/16 and 6/80. Short sources (spider, techqa) are never clipped. So the check compared
  different inputs, not different backends.
- Fix (`9bbc1fc`): shared `TEACHER_MAX_QUERY_TOKENS` / `TEACHER_MAX_DOC_TOKENS`, passed to the worker, and
  `clip_to_teacher_window` applied to the label-check sample so both backends score identical text. Test added.
- Not yet explained: `synth_techqa` 0.984 and `synth_nl2code` 0.988 have no clipped text. Likely bf16 noise on
  only 80 pairs with near-tied scores. Re-measured with 64 rows/source in R9 before touching the 0.99 bar.

### User decisions (2026-10-31)

- Pilot gate: exec@1 bar is **relative +10%** (`--min-exec-rel-gain 0.10`), CoQuIR pairwise drop limit **0.05**
  with per-subset drops reported. Done in `56ce73d`. Under it, 10x combined-prod passes exec@1 (+14%) but
  still fails CoQuIR (Defects4J -0.11).
- Next: seed replicates before any pilot spend (R9). Delete the archived `ce3-dryrun-*-2026103*` repos once
  a dry-run passes.

## R9. Round 6: seed replicates of the 10x prod effect (2026-10-31, pod `gi673gmlohffds`, results `ablate/20261031-092215/`)

Rebuild the 10x mix (R8's was on container disk the dry-run restart wiped; this time archived to
`ce3-ablate-data/dryrun_ablate_x10.tar`, 234 MB), then combined-prod and replay-prod with seeds 42, 43, 44 on one
build, then label-check with the clipping fix on 64 rows/source. Script `ablate6.py`.

Rebuild matched R8: identical row counts for every replay source and the SWE twins; only vLLM-generated
behaviour rows differ by 1-3 (nl2code 304 vs 301, code2code 284 vs 283). Base identical. `replay-prod-s42`
reproduced R8 to every digit (same data, same seed: training is deterministic); `combined-prod-s42` moved
0.111 -> 0.109 exec@1 from the 4 changed behaviour rows.

| prod recipe, 10x | seed | exec@1 | exec@10 | beats_paired | D4J pairwise |
|---|---|---|---|---|---|
| base | | 0.097 | 0.967 | 0.063 | 0.600 |
| replay | 42 / 43 / 44 | 0.078 / 0.078 / 0.077 | 0.968 / 0.971 / 0.972 | 0.077 / 0.079 / 0.078 | 0.512 / 0.499 / 0.497 |
| combined | 42 / 43 / 44 | 0.109 / 0.096 / 0.114 | 0.972 / 0.974 / 0.967 | 0.111 / 0.099 / 0.108 | 0.480 / 0.501 / 0.482 |
| combined mean +- sd | | **0.106 +- 0.009** | 0.971 | **0.106 +- 0.006** | 0.488 |
| behaviour gain (paired by seed) | | **+0.031 / +0.018 / +0.037** | | **+0.034 / +0.020 / +0.030** | -0.032 / +0.002 / -0.015 |
| mean behaviour gain | | **+0.029 +- 0.010** | | **+0.028 +- 0.007** | -0.015 +- 0.017 |

**What holds (3 seeds):**
- Behaviour data beats replay-only on exec@1 and beats_paired on **every** seed (paired t, df 2, p ~ 0.04).
  Replay-only is very stable across seeds (exec@1 0.077-0.078); the variance comes from the behaviour runs.
- beats_paired 1.7x base on every seed with behaviour data.

**What holds only on average:**
- exec@1 vs base: mean 0.106 = **+9.5% relative**, just under the +10% gate (0.107); 2 of 3 seeds pass
  (seed 43 is at base). R8's single seed (+14%) overstated it.
- Seed spread at 10x (exec@1 sd 0.009) is ~2x R7's 1x estimate (<= 0.004); single-seed 10x cells are not
  enough to call a difference under ~0.02.

**What does not:**
- Defects4J pairwise is ~0.11 below base with or without behaviour data; the behaviour effect on it is
  within noise (-0.015 +- 0.017). **The CoQuIR regression comes from fine-tuning itself** (replay mix or
  recipe), not from the behaviour data. That is the next problem to solve before a pilot can pass the gate.

**label-check with clipping, 64 rows/source** (finished 12:12; the GPU reading of 0 earlier was stale):

| Source | R8 attempt 3 (16 rows, unclipped) | R9 (64 rows, clipped) |
|---|---|---|
| code_contests_xlang | 0.957 | **0.976** (still < 0.99) |
| swe_smith | 0.991 | 0.9987 |
| synth_techqa | 0.984 | 0.9925 |
| synth_nl2code | 0.988 | 0.9944 |
| other 12 | 0.991-0.9995 | 0.996-0.9995 |

Clipping fixed swe_smith; the synth sources were 80-pair sampling noise. code_contests_xlang (long code on
both sides, cross-language) still misses. User chose to investigate before changing the bar (R10).

## R10. Round 7: teacher parity on code_contests_xlang (2026-10-31, same pod, ~20 min, results `ablate/20261031-12*`)

**Procedure.** On the archived 10x data, 128 rows of three sources clipped to the teacher window, scored by the
cross-encoder reference in bf16 (checkpoint dtype) and in fp32, then by vLLM. Script `ablate7.py`.

| Source (640 pairs) | ref bf16 vs fp32 | vLLM vs ref fp32 | vLLM mean abs diff short / long (>=1024 tok) | ref bf16-fp32 mean abs diff |
|---|---|---|---|---|
| code_contests_xlang | **0.9997** | **0.982** | 0.42 / 0.43 | 0.07 |
| swe_smith | 0.9998 | 0.9989 | 0.22 / 0.25 | 0.07 |
| cornstack_python | 0.9999 | 0.9993 | 0.29 / 0.12 | 0.05 |

- **Not precision**: bf16 and fp32 references agree to 0.9997+ everywhere.
- **Not length**: short and long contest pairs carry the same vLLM error.
- vLLM is off by a consistent 0.1-0.4 on every source: a prompt difference, not noise.

**Root cause (found locally, no GPU).** Rendered both prompts with transformers' `apply_chat_template` (the
renderer vLLM uses): byte-identical except the **last token**, the one the yes/no score is read from:
reference (model's `chat_template.jinja`, matching the model card) ends `</think>\n\n` (token 271), vLLM
worker ended `</think>\n` (token 198). Jinja drops one trailing newline; the reference's template file ends
with three newlines, the worker's `TEMPLATE` with two. Fix `TEMPLATE` ends with three, so the rendered
prompts are byte-identical. Test renders it in transformers' Jinja environment; fails before, passes after.
No vLLM teacher labels were ever produced (label never ran), so nothing to regenerate. GPU confirmation is the
label-check in the next dry-run.

## R11. Dry-run attempt 4: PASS end to end (2026-10-31 12:51-14:15, pod `rct3ebgzb41h4t`, A40 CA-MTL-1, terminated)

Code `e8d0d1e`, all fixes, fresh scratch repos. `ce3 dry-run` exit 0 in 84 min; every step ran for real
(report `code-embed-v3-src/dryrun/20261031-125130/`):

doctor, build-data, synth x4 (vLLM generation, 12.3 min), audit (9.0 min), zero-shot, attention (flash == sdpa,
x1.12), mine, behaviour-build (vLLM, 5.8 min), fine-tune killed at checkpoint-40 -> resumed from this run's
Hub `last-checkpoint` (5.5 min) -> restored as finished on a "fresh pod" (1.9 min), **label-check pass**
(17.5 min), **label** (15.1 min), **train-s3** (1.4 min), behaviour-eval base + ft, mteb (4.3 min), **release**.
`label`, `train-s3`, `mteb` and `release` had never been reached by any earlier dry-run.

- **Teacher parity with the prompt fix**: min Spearman **0.9973** over 16 sources; code_contests_xlang
  0.976 -> **0.9992**; max abs diff 0.18-0.37 everywhere (was up to 6.5). R10's root cause confirmed on GPU.
- Fine-tune kept vs base: exec@10 1.001, CoQuIR Defects4J ndcg@10 0.971 (no-collapse check passes).
- Slowest steps: label-check 17.5 min and label 15.1 min (one vLLM start-up per source; worth batching
  sources into one worker before the pilot), audit 9 min.

Cleanup: deleted the four archived repos the user approved (`ce3-dryrun-synth-20261029`, `-synth-20261031`,
`-ft-20261029`, `-ft-20261031`). Remaining, all private: the run's own scratch repos (`ce3-dryrun-synth`,
`-ft`, `-s3`, `-release`) and three archives made after the approval (`ce3-dryrun-ft-20261030`,
`-ft-20261031b`, `-synth-20261031b`), left for the user.

### Where things stand (2026-10-31)

- Pipeline: **dry-run passes end to end**; 12 fixes committed on `behaviour-pilot-fixes` (not pushed), 137 tests.
- Behaviour data (3 seeds, 10x, prod recipe): beats replay-only on exec@1 (+0.029 +- 0.010) and beats_paired
  (+0.028 +- 0.007) on every seed; beats_paired 1.7x base. exec@1 vs base +9.5% (gate +10%): borderline.
- **Blocker for a pilot**: CoQuIR Defects4J pairwise ~0.11 below base in every fine-tune, from fine-tuning itself,
  not behaviour data; fails the 0.05 gate. Next research step: find which part of fine-tuning causes it
  (replay mix, recipe, steps) before paying for the pilot.
- Round 4-7 spend about $6 total on 2026-10-31.

## R12. The Defects4J "regression" is a truncation artefact (2026-10-31, local Mac MPS, no GPU spend)

**Hypothesis.** Fine-tunes are saved with the stage's training `max_seq_length` (256 in dry-run/ablations,
512 in the pilot config) and evaluated at that length, while base GTE is evaluated at 8192. Defects4J pairs are
buggy/fixed versions of one Java method; if the fix sits past the cut, both encode identically.

**Procedure.** `ce3-dryrun-ft/final` (dry-run attempt 4, combined mix, prod recipe, 300 steps) and base GTE,
scored on CoQuIR Defects4J and ExecRetrieval at different `max_seq_length`, locally on MPS (absolute numbers
differ from A40 CUDA, e.g. base exec@1 0.087 vs 0.097; compare within a table only). Scripts `d4j_probe.py`,
`exec_probe.py` (scratchpad).

| Defects4J (465 pairs; 86 identical within 256 tokens; corpus p50 237, p90 794 tokens) | pairwise | ndcg@10 |
|---|---|---|
| base @ 8192 (as evaluated so far) | 0.606 | 0.819 |
| base @ 256 | **0.460** | 0.749 |
| ft @ 256 (as saved / evaluated so far) | **0.488** | 0.801 |
| ft @ 8192 | **0.609** | 0.807 |
| ft @ 8192, no query prompt | 0.637 | 0.818 |

| ExecRetrieval (queries max 136 tok; 10.5% of corpus > 256) | exec@1 | exec@10 | beats_paired |
|---|---|---|---|
| base @ 8192 | 0.087 | 0.966 | 0.058 |
| base @ 256 | 0.083 | 0.953 | 0.054 |
| ft @ 256 | 0.104 | 0.968 | 0.100 |
| ft @ 8192 | 0.105 | 0.969 | 0.110 |

**Findings.**
- **The ~0.11 Defects4J pairwise drop seen in every fine-tune (R7-R9) was the 256-token cut**, not a
  regression: at equal length the fine-tune matches base (0.609 vs 0.606); base cut to 256 scores 0.460.
  Hypothesis tracker: "fine-tuning causes the CoQuIR regression" is **refuted**; the pilot blocker is gone.
- ExecRetrieval conclusions hold; truncation slightly *understated* the fine-tune (beats_paired +0.01 at full
  length).
- Without the nl2code query prompt the fine-tune scores higher on Defects4J (0.637). Small, single model; note
  for prompt design, not acted on.
- **Real bug for the pilot and release**: `train.py` sets `model.max_seq_length` to the training length and
  saves it, so `final/` (and the released model) truncates inputs at 512 tokens in the pilot, against base
  GTE's 8192; behaviour-eval and mteb then compare unequal lengths.

**Fixes (user: saved models at least 8192; evals at 8192 for every model).**
- `08be40b`: final model saved at `ModelConfig.inference_max_seq_length` (default: backbone position
  limit, 8192 for GTE-ModernBERT); training still at the stage length. Test: tiny backbone trained at 32 tokens
  saves at 128 (fails on old code with 32).
- `behaviour-eval` / `mteb` take `--max-seq-length` (default 8192), applied to base and candidate alike and
  written to the output JSON. Test covers it.
- Batched teacher labelling (`5a0296d`): one vLLM start-up per run instead of per source; tests for
  `label_sources` and the `label` command (fail on old code). Speed-up not yet measured on GPU.
- Consequence for earlier tables: R7-R9 Defects4J numbers for fine-tunes are 256-token artefacts and must not
  be compared with base; exec@1 / beats_paired conclusions stand (slightly understated).

## R13. Contamination check of the research data and enforcement gap (2026-10-31, local, no GPU spend)

**Why.** The dry-run and ablation mixes were audited with `--tasks CosQA SyntheticText2SQL` only, which skips the
non-mteb benchmarks, so the 10x data behind R8/R9 had never been checked against ExecRetrieval or CoQuIR.

**Procedure.** The repo's own audit (`EvalIndex`: exact + MinHash-LSH, threshold 0.8, 5-gram; anchor, positive and
every negative column) over the archived 10x mix (`dryrun_ablate_x10.tar`, mined + behaviour, 30,562 rows) against
`external_texts()` (all ExecRetrieval queries + corpus, all 7 CoQuIR subsets). Script `audit_x10.py`.

| Benchmark | Training rows hitting it |
|---|---|
| ExecRetrieval | **0** |
| CoQuIR Defects4J | **0** |
| CoQuIR CodeNet-B / CodeNet-E | 15 rows (0.05%): code_contests_xlang 13, behaviour code2code 1, nl2code 1 |
| everything else (cornstack x9, spider, swe_smith, synth, SWE twins) | 0 |

- **The R8/R9 conclusions (exec@1, beats_paired, Defects4J) are not contaminated.**
- CodeNet overlap is expected (CodeContests and CodeNet both draw on AtCoder); CodeNet was never reported, and the
  full audit drops these rows.
- **Gap fixed (`5fb7d3c`)**: training only checked the `audited` flag, so a 2-task dry-run audit could feed a real
  stage. Stages now require every source's `audited_against` to cover `full_audit_tasks()` (all MTEB(Code, v1),
  CoIR, CoREB(v1) tasks + ExecRetrieval + every CoQuIR subset); the dry-run opts out with `audit_scope: any`.
  Test: the old code trains on a partly audited mix, the new code refuses.

## R14. Dry-run attempt 5: PASS, today's fixes verified on GPU (2026-10-31 14:55-16:12, pod `0wo8h9jpevi189`, terminated)

Code `08be40b` (batched labelling `5a0296d`, full-context save + eval at 8192 `08be40b`; the audit-scope check
`5fb7d3c` came after and was not in this run). Fresh scratch repos. `ce3 dry-run` exit 0 in **76.2 min** (attempt 4:
84.1). Report `code-embed-v3-src/dryrun/20261031-145521/`.

| Check | Attempt 4 | Attempt 5 |
|---|---|---|
| label-check | 17.5 min | **5.9 min** |
| label | 15.1 min | **3.9 min** |
| teacher parity (min Spearman, 16 sources) | 0.9973 | 0.9972 |
| eval length (base / ft) | 8192 / 256 | **8192 / 8192** |
| released model card "Max sequence length" | 256 | **8192 tokens** |
| mteb step | 4.3 min | 6.4 min (longer inputs) |

- **Batched labelling: 32.6 -> 9.8 min (3.3x)** for the two teacher steps, same parity.
- **Defects4J at equal length on GPU: ft 0.611 vs base 0.600 pairwise** (ndcg 0.811 vs 0.814): no regression,
  confirming R12 on the A40.
- The dry-run fine-tune itself (300 rows/source, 300 steps): exec@1 0.106 vs base 0.097 (+10%), beats_paired
  0.111 vs 0.063, exec@10 equal. Tiny scale, one seed; consistent with R9, not new evidence.
- Every pipeline step now passes on code that will run the pilot, except the audit-scope check (`5fb7d3c`), which
  is unit-tested and only adds a refusal before training.

## R15. Pilot-risk fixes, batch-size bridge and the pilot (2026-10-31, started 16:39 UTC) - running

**Fixes (`9b5bc9b`, private `0b88591`), from reading the pilot config against the 10x setup:**
- **Behaviour share.** The 10x effect was measured with behaviour data at 11% of train rows; the pilot mix (25k
  rows/source replay) would have set it by accident. `behaviour-combine --behaviour-share 0.11` subsamples
  whichever side is over the target and prints every source's share; `make pilot` passes `BEH_SHARE=0.11`.
- **Batch size 512 -> 128.** Each batch comes from one source and partial batches are dropped, so at 512 a
  source under 512 rows contributed nothing, silently. 128 keeps compute per row (GradCache) and gives 4x steps.
- **Small-source guard.** `min_batches_per_source: 3` (StageConfig): training refuses a source that fills
  fewer than 3 global batches. The dry-run opts out (its 40-problem sources fill 1-2 batches of 16).
- 146 tests pass, ruff and mypy clean.

**Bridge** (pod `x34h837ontr0ug`, A40, script `bridge.py`): archived 10x data, prod recipe, same 19,200 samples
at batch 64 / 128 / 256 (seed 42; 128 also seed 43), combined vs replay-only; then MTEB(Code, v1) on base and
b128-combined. Guard off here, so 256 shows what happens when contest sources (~300 rows) fill one batch.

**Pilot** (pod `vc0q4mymjp57b3`, A40, script `pilot.py`, status in `code-embed-v3-src/pilot/pilot-1/`):
`make pilot` steps plus a control arm. Replay mix built with `--max-rows 30000` (mined to 25k/source) and the
full audit; `make synth` (600k LLM rows) skipped, so no synth_* sources. Behaviour build 5,000 problems,
6 mutants, 2 rewrites, 5,000 SWE twins; combined at 11%. Train batch 128 (changeable before training via
`pilot/pilot-1/batch_size.txt` once the bridge is in), behaviour-eval and MTEB(Code) on base and ft, pilot-gate,
then a replay-only control trained on the identical (subsampled) replay rows.

### Bridge result (results `ablate/20261031-164208/`): the behaviour effect needs small batches

Same 19,200 samples, prod recipe, evals at 8192 tokens. Base reproduced exactly.

| Batch (steps) | Seed | exec@1 comb / replay | beats_paired comb / replay | D4J pairwise comb / replay |
|---|---|---|---|---|
| 16 (1200), R9 | 42/43/44 | 0.106 / 0.078 (mean) | 0.106 / 0.078 | 0.488 / 0.503 (at 256 tok) |
| 64 (300) | 42 | 0.093 / 0.091 | 0.101 / 0.098 | 0.628 / 0.639 |
| 128 (150) | 42 | 0.089 / 0.081 | 0.097 / 0.089 | 0.641 / 0.624 |
| 128 (150) | 43 | 0.094 / 0.092 | 0.100 / 0.096 | 0.639 / 0.656 |
| 256 (75) | 42 | TrainingShortfall (30/75 steps) / TrainingShortfall (21/75) | | |

- **Behaviour effect (combined - replay) shrinks with batch size:** exec@1 +0.029 (16) -> +0.002 (64) -> +0.005
  mean (128); beats_paired +0.028 -> +0.003 -> +0.006. Larger batches at equal samples mean fewer optimizer
  steps, and replay-only fine-tuning itself already lifts beats_paired to ~0.09-0.10 there.
- No setting beats base exec@1 (0.097) except batch 16.
- **Batch 256 cannot be filled** by the no-duplicates sampler on 10x sources (~2.3k rows, 4 mined negatives
  that are other rows' positives), with or without behaviour data; the shortfall guard refused both.
- Defects4J pairwise at 8192 tokens is above base (0.60) in every run (0.62-0.66): the old "CoQuIR regression"
  is gone for good at equal eval length.
- **Decision (user, 2026-10-31):** train the pilot at batch 16 (`pilot/pilot-1/stage_overrides.json`:
  batch 16, save_steps 2000, eval_steps 1000) and accept the +9.5% exec@1 class of effect as worth a shot.

**Pilot behaviour data (built 2026-10-31 22:30 UTC, 101 min):** 5,000 problems, 4,386 with a verified
solution; mutants kept 1,731 / 17,527 parsed (26,316 requested), rewrites kept 2,906 / 7,945. Rows: nl2code
27,608, code2code 26,360 (repeats 8, 4 negatives), SWE twins 5,000.
**Known deviation from the validated 10x recipe:** at 10x the twins were ~82% of behaviour rows (2,756 of
3,340, repeats 1, 1 negative); here ~8% before the 11% share subsampling, which scales all behaviour sources
alike. R7 (1x) hinted twins matter (no-twins exec@1 0.068 vs 0.075-0.078 with them). Matching both the 11%
share and the 10x composition is impossible with 5,000 twins (it would cap behaviour at ~2% of rows), so the
pilot runs as built; the replay-only control still isolates the behaviour data's effect. If the gate fails,
a twins-heavy mix (more SWE-smith rows, contest repeats 1) is the first thing to try.

### Pilot result: behaviour evals (2026-11-01, pod A40, both models scored in float32 at 8192 tokens)

Training: 14,995 steps at batch 16 (240,054 rows, 11.0% behaviour), 199.5 min, dev score 0.758 -> 0.813, guard
never tripped. Model at `code-embed-v3-150m-ft-behaviour/final`.

| Metric | Base | Pilot ft | Delta |
|---|---|---|---|
| ExecRetrieval exec@1 | 0.0884 | **0.0990** | **+12.0% rel (gate >= +10%: PASS)** |
| exec@10 | 0.9681 | 0.9723 | +0.004 |
| beats_paired | 0.0937 | 0.1054 | +12% rel |
| CoQuIR mean pairwise (7 subsets) | 0.5523 | 0.5530 | +0.0007 (gate drop <= 0.05: PASS) |
| Defects4J pairwise / ndcg | 0.624 / 0.815 | 0.654 / 0.828 | +0.030 / +0.013 |
| SQLR2 / SafeCoder / DepreAPI pairwise | 0.638 / 0.497 / 0.461 | 0.654 / 0.507 / 0.464 | +0.016 / +0.010 / +0.003 |
| CVEFixes / CodeNet-B / CodeNet-E pairwise | 0.604 / 0.542 / 0.500 | 0.592 / 0.525 / 0.476 | -0.012 / -0.018 / -0.024 |
| CodeNet-B / CodeNet-E ndcg@10 | 0.342 / 0.282 | 0.245 / 0.192 | **-0.097 / -0.090** |

- Gate: exec@1 and CoQuIR pass; MTEB(Code) pending.
- **Watch item:** CodeNet ndcg drops ~0.09 on both subsets (contest code, same family as the training
  contest data); pairwise holds. Not in the gate; report it.
- Local Mac check gave the same ft exec@1 (0.0990); base differs by device (fp16 MPS 0.083 vs CUDA 0.088).
- **Gate: NO-GO on MTEB(Code).** Base 71.78 (reproduces the 71.66 reference), fine-tune 66.29: **-5.5 pt**
  against a 0.5 pt limit. Two tasks carry it: AppsRetrieval 57.67 -> 18.84 (-38.8), CodeSearchNetCC
  93.38 -> 74.42 (-19.0). The other ten move -6.8..+7.1 (~-0.8 pt mean); CodeEditSearch +7.1, CodeTransOcean
  Contest +3.0. Suspects: (1) query prompts the base never used (the fine-tune prompts every query in mteb);
  (2) contest-heavy behaviour data (Apps is contest problems -> solutions). The replay-only control and a
  no-prompt rerun of the two tasks separate them.
- **Prompt check (local Mac, fp16, 8192 tokens; matches the pod to 0.02):** the pilot model without query
  prompts scores AppsRetrieval 19.06 (18.83 with) and CodeSearchNetCC-python 82.30 (79.31 with; base 94.18).
  Prompts cost ~3 pt on CSNCC (its `code2code` "equivalent code" prompt does not fit a completion task) and
  nothing on Apps. **The MTEB loss is in the weights, not the prompts.**
- The control arm first failed on a pilot-script bug (`stage()` got `name` twice); fixed, rerun 10:27 UTC.
- **MTEB precision:** base GTE runs fp16 (its checkpoint dtype); the fp32 fine-tune took ~3x as long on the
  behaviour eval, which would have made its MTEB(Code) run 4-5 h. From `1f9fef5`, `ce3 mteb --dtype float16`
  scores the fine-tune (and the control) in fp16 too, so MTEB compares both models at the same precision.
  mteb does **not** upcast numpy embeddings (its `_postprocess_dense_embeddings` only casts torch tensors), so
  both models' MTEB similarities are float16: equal treatment for the gate, but absolute MTEB numbers may
  differ slightly from float32 leaderboard runs. The pod was restarted mid `mteb-base`; mteb's cache
  (`only-missing`) keeps finished tasks.

### Correction (2026-11-01): base GTE was scored in float16; its beats_paired was understated

GTE-ModernBERT ships fp16, so its embeddings came back float16 and `q @ d.T` ran in float16 (numpy has no
fast kernel for it: the pilot's 26k x 53k CoQuIR-DepreAPI product stalled for 90 min). Rounding to float16
turns near-equal scores into exact ties, and `beats_paired` counts a tie as a loss. Fine-tunes are fp32 and
never had the handicap. Same base model, same embeddings (local Mac, MPS, 8192 tokens):

| base scored in | exec@1 | exec@10 | beats_paired | D4J pairwise | D4J ndcg@10 |
|---|---|---|---|---|---|
| float16 (every base number before `3656621`) | 0.0873 | 0.9659 | **0.0575** | 0.6065 | 0.8187 |
| float32 | 0.0831 | 0.9670 | **0.0873** | 0.6258 | 0.8158 |

- **"beats_paired 1.7x base" (R8, R9, bridge) was mostly this artefact.** The GPU base (0.063 in float16)
  should be ~0.09-0.095 in float32, against ~0.10-0.11 for the fine-tunes: a real gain of ~10-20%.
- exec@1 moves slightly the other way (base lower in float32), so exec@1 gains vs base may be a bit larger.
- **Fine-tune vs fine-tune comparisons are unaffected** (both fp32), including every "behaviour data vs
  replay-only" effect in R7-R9 and the bridge.
- Fixed in `3656621` (`_encode` returns float32, regression test). The pilot's base eval was rerun with it.

**Pilot incidents so far:** the full audit had never run at real scale. (1) `_hits` column typed list<null>
when a map worker's first 1,000-row write had no hit, so the first later hit failed to cast; fixed by
declaring the feature (`432e17f`, regression test). (2) MinHash indexing of ~357k eval texts was single-core;
now parallel (`775eb69`). Also: datasets under 1/4 of RAM load into memory (`/workspace` is a network FS),
CPU quota 7.65 now rounds up to 8 workers, eval/mteb encode batch 128, utilisation logged from cgroup counters.

### Host failure and merge run (2026-11-02)

- The pilot pod's host (`vc0q4mymjp57b3`, A40, EU-SE-1) froze on 2026-11-01 ~11:43 UTC during the control arm
  (replay only, step 5,510/13,346; last Hub checkpoint step 2,000). Start/restart on 11-01 and 11-02 left it
  "RUNNING" with no container and no boot; it is stopped, volume kept. The control arm is not rerun yet.
- Weight-averaged candidates, to recover MTEB(Code) while keeping the exec gain: pilot fine-tune x base GTE at
  0.5/0.5 and 0.7/0.3 (`ce3 merge`). Local 0.5/0.5 check (Mac, float32): exec@1 0.1022, exec@10 0.9734,
  beats_paired 0.1086.
- Run on a fresh pod (`4d38g5mt6tph5z`, RTX 6000 Ada, $0.99/h; A40/A6000/L40 out of stock) with `merge.py`,
  results under `pilot/pilot-1-merge/`. Pilot utilisation showed one fp16 MTEB run at 57% GPU in 13 GB and
  one fp32 behaviour eval at 94% in 35 GB, so both MTEB runs share the GPU and the behaviour evals run in turn.
  That was wrong: the 13 GB was a per-minute sample, one fp16 MTEB run peaks at ~24 GB on long documents and
  holds the GPU at 100%, and the concurrent 0.5 run ran out of memory. Rerun one at a time (pod restarted,
  finished steps kept). Total pod time ~3.3 h.

### Merge results (2026-11-02, pod above; behaviour eval float32 at 8192 tokens, MTEB fp16)

| model | exec@1 | exec@10 | beats_paired | CoQuIR pairwise mean | MTEB(Code) | gate |
|---|---|---|---|---|---|---|
| base GTE | 0.0884 | 0.9681 | 0.0937 | 0.5523 | 71.78 | - |
| pilot fine-tune | 0.0990 (+12.1%) | 0.9723 | 0.1054 | 0.5530 | 66.29 | FAIL (MTEB -5.49) |
| merge 0.7 ft / 0.3 base | 0.0852 (-3.6%) | 0.9744 | 0.0927 | 0.5537 | 69.39 | FAIL (exec, MTEB -2.39) |
| merge 0.5 / 0.5 | **0.1022 (+15.7%)** | 0.9734 | **0.1086** | **0.5556** | **70.82** | FAIL (MTEB -0.96 vs 0.5 limit) |

- The 0.5 merge is the best model on every behaviour metric and every CoQuIR ndcg@10 (all >= base), and fails
  only the MTEB limit, by 0.46 pt. Its MTEB loss is AppsRetrieval (57.7 -> 45.2) and CodeSearchNetCC
  (93.4 -> 84.4); it beats base on 7 of 12 tasks (CodeTransOceanContest +6.7, SyntheticText2SQL +3.3).
- exec@1 is noisy: 939 queries, binomial SE ~0.009, so +0.014 is ~1.5 SE, and the 0.7 merge scoring below
  both the fine-tune and the 0.5 merge is a sign of that noise. beats_paired and CoQuIR move the same way
  as exec@1 for the 0.5 merge, which is the stronger evidence.
- The fine-tune's MTEB loss sits in AppsRetrieval (57.7 -> 18.8), contest problems against solutions, the
  same domain as the behaviour data. Hypothesis (untested): learning to separate correct from near-identical
  buggy solutions costs the plain problem-to-solution match. v4 work.

### Significance of the 0.5 merge against base (2026-11-02, pod `tgpmemw829kgc0`, A100; `ce3 behaviour-compare`)

Paired per query: McNemar exact on ExecRetrieval, 10k bootstrap resamples of queries for everything
(`7e9caae`). Base is plain GTE scored again on this pod.

| metric | base | 0.5 merge | diff | 95% CI | p |
|---|---|---|---|---|---|
| exec@1 | 0.0927 | 0.1022 | +0.0096 | [-0.013, +0.032] | 0.45 (McNemar) |
| beats_paired | 0.0990 | 0.1086 | +0.0096 | [-0.014, +0.033] | 0.47 (McNemar) |
| exec@10 | 0.9681 | 0.9734 | +0.0053 | [-0.001, +0.013] | 0.23 (McNemar) |
| CoQuIR pairwise mean | 0.5528 | 0.5554 | +0.0026 | [-0.004, +0.009] | 0.46 |
| CodeNet-B pairwise | 0.5423 | 0.5314 | -0.0109 | [-0.016, -0.006] | <0.001 |
| CodeNet-E pairwise | 0.4996 | 0.4848 | -0.0148 | [-0.019, -0.011] | <0.001 |
| DepreAPI ndcg@10 | 0.6551 | 0.6886 | +0.0334 | [+0.031, +0.036] | <0.001 |
| SQLR2 ndcg@10 | 0.4090 | 0.4364 | +0.0275 | [+0.024, +0.031] | <0.001 |

- **No significant behaviour gain.** The exec@1 and beats_paired gains are within noise (p ~0.45). The claim
  "ranks working code above buggy code better than base" does not hold for the 0.5 merge.
- Significant: retrieval gains on DepreAPI and SQLR2 (ndcg@10), and pairwise losses on CodeNet-B and
  CodeNet-E (the worse variant ranked above the better one more often than base does).
- Base exec@1 moves with hardware: 0.0831 (Mac), 0.0884 (RTX 6000 Ada), 0.0927 (A100); the 0.5 merge scored
  0.1022 on all three. Base GTE loads in fp16 and computes in fp16 (only its embeddings are upcast), so its
  scores carry kernel-dependent rounding; the fp32 merge does not. The "+16%" headline was partly this.

### 0.4 fine-tune / 0.6 base merge (same pod; pod stopped 2026-11-02 ~14:00 UTC, ~2.8 h on the A100)

- MTEB(Code) **70.88** (0.5 merge: 70.82): no better, and still fails the 0.5 pt limit (-0.90). AppsRetrieval
  recovers more (50.1 vs 45.2) but SyntheticText2SQL falls (64.7 vs 70.5), so the mean does not move.
- Behaviour against base: exec@1 0.0969 (+0.004, p 0.77), beats_paired +0.005 (p 0.70), CoQuIR pairwise mean
  +0.0004 (p 0.90). Same significant pattern as the 0.5 merge: DepreAPI and SQLR2 ndcg@10 up ~0.03, CodeNet-B/E
  pairwise down ~0.01, plus CodeNet-B/E ndcg@10 up ~0.02.
- **Conclusion for v3:** no merge ratio gives a significant behaviour gain, and none reaches base MTEB(Code).
  Not released. v4 has to change the training data or objective, not the merge ratio.


## R16. Pilot-scale replay-only control (2026-11-17, Vast.ai 2x A100 40 GB, ~5.5 h incl. idle, results `pilot/pilot-1-control/`)

**Why.** The pilot's control arm died with its host on 2026-11-01, so the MTEB(Code) loss could not be attributed to
the near-miss data versus fine-tuning on the replay mix itself.

**Procedure** (`experiments/control.py`). Rebuilt the pilot's combined mix from Hub copies (mined replay +
behaviour data, re-audited) and refused to train unless every source's row count matched the pilot's
`status.json`: all 15 matched, 213,648 replay rows. Control = the same replay rows and recipe (batch 16, prod
recipe), resumed from its Hub checkpoint at step 2,000 and trained to 13,346 (1.91 it/s). Base, pilot fine-tune and
control were all re-scored on the same box (float32, 8,192 tokens); MTEB(Code) for the pilot and the control both
on the same A100s in fp16. `behaviour-compare` for every pair. The control's behaviour eval ran at batch 8 instead
of 32 (shared GPU); batch size does not change per-item scores beyond float rounding.

| Model (same box) | exec@1 | exec@10 | beats_paired | CoQuIR pairwise mean | MTEB(Code, v1) |
|---|---|---|---|---|---|
| base GTE | 0.0927 | 0.9681 | 0.0990 | 0.5528 | 71.78 (A40) |
| replay-only control | 0.0852 | 0.9755 | 0.0905 | 0.5588 | 69.75 |
| pilot (11% near-miss) | 0.0990 | 0.9723 | 0.1054 | 0.5527 | 66.02 (66.29 on the A40) |

Paired tests (bootstrap 10k over queries; McNemar on ExecRetrieval):

| Comparison | exec@1 diff (p) | beats_paired diff (p) | CoQuIR pairwise mean diff (p) |
|---|---|---|---|
| pilot vs control (effect of near-miss data) | +0.0138 (0.21) | +0.0149 (0.18) | -0.0061 (0.043) |
| control vs base (effect of replay fine-tuning) | -0.0075 (0.59) | -0.0085 (0.54) | +0.0060 (0.16) |
| pilot vs base | +0.0064 (0.67) | +0.0064 (0.69) | -0.0001 (0.98) |

MTEB(Code) per task, the ones that move (A100, fp16):

| Task | base | control | pilot | pilot - control |
|---|---|---|---|---|
| AppsRetrieval | 57.67 | 53.89 | 18.87 | **-35.02** |
| CodeTransOceanContest | 72.38 | 80.97 | 75.37 | -5.60 |
| CodeSearchNetCCRetrieval | 93.38 | 77.06 | 74.42 | -2.64 |
| CodeEditSearchRetrieval | 56.08 | 63.64 | 63.15 | -0.49 |
| CodeFeedbackMT | 86.17 | 78.61 | 79.38 | +0.77 |
| other 7 tasks | | | | -1.1 to +0.2 |

**Findings.**
- **The AppsRetrieval collapse is caused by the near-miss data**: the control keeps 53.9, the pilot falls to 18.9.
  Apps pairs contest problems with solutions, the domain of the contest twins: learning to push away near-identical
  solutions to the same problem breaks problem-to-solution matching.
- **Replay fine-tuning itself costs ~2 MTEB points** (71.78 -> 69.75), almost all on CodeSearchNetCC (-16.3), offset
  by gains on CodeTransOceanContest (+8.6) and CodeEditSearch (+7.6). Of the pilot's 5.8-point loss, ~2.0 comes from
  fine-tuning on the replay mix and ~3.7 from the near-miss data.
- **Near-miss data adds +1.4 points exec@1 over the control, not significant (p 0.21)**, same direction as the
  small-scale seeds (+2.9). The replay control alone sits slightly below base on exec@1. CoQuIR goes the other way:
  the control is better than the pilot (p 0.043), mostly on CodeNet.
- Hardware: the pilot's MTEB(Code) moved 66.29 (A40) -> 66.02 (A100); base exec@1 0.0927 here, as on the earlier A100.
- **Conclusion for v3:** near-miss data as contrastive negatives gives a small, non-significant gain on the target
  metric and a large, specific loss on problem-to-solution retrieval. Closed as a negative result.

**Cost and ops.** Vast.ai 2x A100 40 GB at $1.21/h (RunPod had no 48 GB stock). About 2 h of the bill were an idle
box after a failed first launch (tarball excluded `src/code_embed_v3/data`), and the GPUs were under-used (training at
batch 16 used ~40% of one GPU, 8 GB): one 48 GB card with evals packed alongside training would have been cheaper.
The final Hub push hit the account's private storage limit; old dry-run repos were deleted and the model pushed to
`code-embed-v3-150m-ft-control/final`.

---

## Code changes so far (committed in `2014373`)

| File | Change | Why | Verified |
|---|---|---|---|
| `src/code_embed_v3/eval/behaviour_eval.py` | `no_collapse()`: fails if exec@10 or any CoQuIR ndcg@10 keeps < 50% of base | Zero-shot gate passed a random-level model | Unit test with the dry-run's collapsed numbers |
| `src/code_embed_v3/dry_run.py` | Evaluates base GTE too; records `ft_kept_vs_base`; raises at the end on collapse | Dry-run reported success on a collapsed model | ruff, mypy, tests |
| `src/code_embed_v3/dry_run.py` | `--swe-smith-rows <rows>` for behaviour-build | R2: 4,800 twins = 60% of batches | Round 2 runs on it |
| `src/code_embed_v3/cli.py` | `label-check` scores reference labels, frees the cross-encoder, then starts vLLM | vLLM OOM (P1 in DRYRUN_FINDINGS) | ruff, mypy, tests; not yet run on GPU |
| `tests/test_behaviour.py` | `test_no_collapse_flags_near_random_model` | | 126 passed |
| `src/code_embed_v3/config.py`, `train.py` | `collapse_guard`: scores the dev suite before training; stops and raises `TrainingCollapsed` (no final save) when the dev score drops below that fraction | Dry-run trained ~90 steps past collapse | Test: trips at step 1 with ratio 100, saves normally with 0 |
| `src/code_embed_v3/config.py`, `train.py` | `source_loss_weights` + `WeightedLoss`: per-source loss multiplier via ST's per-dataset loss dict | Lets behaviour data be down-weighted instead of dropped | Test: weight 0.5 halves every gradient through GradCache (exact) |
| `src/code_embed_v3/cli.py` | `ce3 merge --weights` (mergekit linear weights) | Weight averaging fine-tune with base to recover robustness | ruff/mypy; GPU test pending |
| `tests/test_train_smoke.py` | 3 tests for the above | | 129 passed |
| `src/code_embed_v3/config.py`, `train.py` | `min_step_fraction` + `TrainingShortfall` | R4: sampler starvation silently cut runs to 7 of 60 steps | Test: starved toy mix raises; 130 passed |
| `src/code_embed_v3/dry_run.py` | `mine --num-negatives 4` | Match pilot `MINE_NEGS`; 7 starves the sampler at 300 rows/source | Round 3 runs on it |
| `src/code_embed_v3/modeling.py` | `training_dtype()`: fp32 master weights unless `torch_dtype` is set; used by `build_model` and `init_model` | **Root cause of the collapse** (R6) | Regression test fails without / passes with; local A3 0.001 -> 0.957 exec@10; 131 passed |
| `tests/test_modeling.py` | `test_fp16_checkpoint_trains_with_fp32_master_weights` | | |
| `src/code_embed_v3/train.py` | Dev evaluation and guard baseline under `accelerator.autocast()` | fp32 weights + flash-attn outside autocast crashed (R7) | 131 passed; GPU check in R7 rerun |
| `src/code_embed_v3/modeling.py` | `load_model`: bf16 default when flash-attn is selected | fp32 checkpoints + flash-attn for mining/inference | GPU check pending |

Not changed on purpose: `150m_ft_behaviour.yaml` (wait for R3), `pilot_gate` (already fails collapse via the
MTEB(Code) drop limit), docs cost numbers (4090 is back at $0.34/h with low stock on 2026-10-30).

## Hypothesis tracker

| # | Hypothesis | Status |
|---|---|---|
| 1 | lr 2e-5 too high | Contributes, not root cause (R2: lower lr degrades less, still fails) |
| 2 | tau 0.015 too sharp | Contributes, not root cause (R2: tau 0.05 better than 0.02) |
| 3 | batch 16 bad proxy for 512 | Contributes (R2 E: batch 64 better); pilot-scale not tested |
| 4 | Matryoshka 5x gradient | Not root cause (C collapses with one dim) |
| 5 | train/eval mismatch | Ruled out (R1, R2 A/B) |
| 6 | twins as each other's negatives | Weak (R3: 7.8% of rows have a cross-row near-dup, rarely in the same batch) |
| 7 | clipping not applied | Ruled out (R1) |
| 8 | SWE twins share of batches (new) | Not sufficient (R3: collapse at 300 rows too) |
| 9 | Twins identical after truncation | Ruled out (R3: 0 / 5,000) |
| 10 | Sampler starves mined replay sources (new) | Confirmed at dry-run scale (R4); pilot scale unmeasured |
| 11 | Behaviour near-miss data harms fast when undiluted (new) | Unproven: all evidence came from fp16-broken runs; retest |
| 12 | fp16 master weights (new) | **Confirmed root cause** (R6); fix verified on GPU (R7) |
| 13 | Behaviour data improves target metrics (pilot premise) | **Supported, grows with scale** (R8 10x: exec@1 +0.033, beats_paired +0.034 vs replay under prod; first exec@1 above base, 0.111 vs 0.097; 1 seed). Far below the absolute +0.10 gate |
| 14 | Base merging recovers robustness | **No at scale** (R8: lowers exec@1 0.111 -> 0.088); only helped in R7 when fine-tunes were below base; never restores D4J pairwise |
| 15 | Prod recipe (tau 0.015, hardness 5) is what makes behaviour data work (new) | Supported (R8: alone it does nothing, with behaviour data it triples the gain; 1 seed) |
| 16 | Near-miss data causes the MTEB(Code) loss (pilot) | **Partly** (R16): it causes the AppsRetrieval collapse (-35 vs control); replay fine-tuning alone costs ~2 points |
| 17 | Near-miss data improves exec@1 at pilot scale | **Not significant** (R16: +0.014 vs control, p 0.21; small-scale seeds +0.029) |

Note: rows 1-4 and 8 were judged on fp16-broken runs; their "contributes" verdicts only describe how fast
the bug destroyed the model. Re-test after the fix before using them for the recipe.
