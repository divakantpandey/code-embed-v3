# Plan

## Goal

Lead the ≤200M class on MTEB(Code, v1) with a 150M encoder, and be competitive in the ≤600M class with the 0.6B sibling. Both models should also do well on real repository search, not just the leaderboard.

## Landscape (MTEB Code, nDCG@10, as of 2026-10)

| Model | Params | MTEB Code | Notes |
|---|---|---|---|
| jina-code-embeddings-0.5b | 494M | ~78.7 | Causal Qwen2.5-Coder, last-token pooling; CC-BY-NC |
| voyage-4-nano | 340M | 76.4 | Bidirectional Qwen3, Apache-2.0 |
| pplx-embed-v1-0.6b | 0.6B | 75.2 | Bidirectional Qwen3 (diffusion continued pretraining), MIT |
| Qwen3-Embedding-0.6B | 0.6B | 73.8 | |
| mLateOn / mDenseOn | 307M | 73.5 / 71.5 | ModernBERT-class, little code data |
| EmbeddingGemma-300m | 300M | 68.8 | |
| code-embed-v2 (previous) | 131M | ~55 | UniXcoder, CodeSearchNet only |

These numbers come from different papers (jina 2508.21290, DenseOn 2607.27178), so they don't line up exactly. Re-run every baseline with the pinned `mteb` version from `uv.lock` before claiming a rank.

**Targets:** 150M ≥ 74 (best under 200M, matching Qwen3-Embedding-0.6B at a quarter of the size); 0.6B ≥ 79.

## What changed from v2, and why

| v2 | v3 | Reason |
|---|---|---|
| UniXcoder (2021, ~1k context, interpolated to 8k) | ModernBERT / Ettin (8k native), Qwen3-0.6B | Modern pretraining with code, real long context |
| CodeSearchNet + CoSQA | A 100% zero-shot mix (CoRNStack both directions, CodeContests, SWE-smith, Spider) plus synthetic pairs for techqa, multiturn and code_edit | v2's worst tasks were the families it never saw, and v2 was flagged for training on eval-task data |
| Batch 128, in-batch negatives only | 16k batch (GradCache), mined and filtered hard negatives, distillation | The standard recipe of current leaders |
| 768→1536 projection | Native-dimension Matryoshka | An up-projection adds index cost without adding information |
| Internal eval on about 100 candidates | Per-family dev suite plus official mteb runs with a contamination audit | v2's internal numbers did not predict leaderboard results |

## Ablations (one s1 epoch on a ~2M-row subset each, scored on the dev suite)

Run these before any full-scale run. Gaps under 1 point need 3 seeds and bootstrap confidence intervals.

1. Backbone: ModernBERT-base vs Ettin-150m; Qwen3-0.6B (bidirectional, mean) vs Qwen2.5-Coder-0.5B (causal, last-token).
2. Temperature: 0.01 / 0.02 / 0.05.
3. Directions: `query_to_doc` vs `query_to_doc + doc_to_query`.
4. Hardness weighting: off vs `hard_negatives` at strength 5.
5. Data mix: drop one family at a time to measure each family's contribution.
6. Teacher: Qwen3-Reranker-4B vs the trained 0.6B sibling (for the 150M model).
7. In-domain data (`eval_overlap.yaml`, CoIR train splits) on vs off. Never part of a submitted model: it drops MTEB(Code) to 50% zero-shot.
8. Synthetic data on vs off, and `top_k` of the round-trip filter (1 / 3 / 10).

## Testing

- **Dev suite** (`ce3 dev-eval`): per-family nDCG@10 on held-out training rows. Use it for checkpoint selection.
- **Leaderboard** (`ce3 mteb`): MTEB(Code, v1), CoIR, CoREB(v1). Run once per release candidate, with `--data-dirs` so overlapping sources are declared.
- **Contamination**: `contamination_report.json` ships with every release.
- **Usefulness** (to add): SWE-bench Verified issue→file localisation Recall@k, the `CodeRAG` benchmark in mteb, Matryoshka truncation curves (`--truncate-dim`), int8/binary quantisation loss, CPU/GPU throughput.

## Known gaps and follow-ups

- **Not yet run on a GPU**: flash-attention (`attn_implementation: auto` through the `kernels` package), token-packed GradCache chunks, the vLLM teacher and generator, and the mini-batch probe. `make dry-run` exercises all of them on the pod and records the flash-attention speedup and its embedding agreement with sdpa (`dryrun/attention_check.json`). If that check fails, set `attn_implementation: sdpa`.
- **Throughput not yet measured**: the cost table in [RUNPOD.md](RUNPOD.md) is FLOP-based until the dry run and the first 10 minutes of a real run.

- **ONNX export is blocked upstream**: `optimum-onnx` still pins `transformers<4.58`. Revisit when it supports transformers 5.
- **Using the 0.6B model outside this repo**: the bidirectional patch is a runtime change. Publishing to the Hub needs a small custom modeling file (`trust_remote_code`) or upstream support. GGUF/llama.cpp would need non-causal attention for Qwen3.
- **Synthetic data**: developer queries, Q&A, change requests and conversations are built (`make synth`). Not built: near-miss negatives and execution-verified translations.
- **Long-context subset**: stage 4 currently reuses the mined mix. Build a whole-file subset (repo files ≥ 2k tokens) for it.
- **Teacher temperature**: calibrate `teacher_temperature` to the teacher's real score spread before stage 3 (see the `DistillKLDivLoss` docstring on softmax underflow).
- **Late-interaction variant**: sentence-transformers 6 ships `multi_vector_encoder`. A ColBERT-style sibling from the same checkpoint is a cheap extra (LateOn scored about 2 points above DenseOn on code).

## Core Novelty & Lean 150M Strategy (Upgraded 2026-10-29)

### Why not train 4 stages from raw `ModernBERT-base`?
- `Alibaba-NLP/gte-modernbert-base` (149M, Apache-2.0) already completed multi-stage contrastive pretraining and scores **71.66 on MTEB(Code, v1)** with **100% zero-shot compliance**.
- Spending ~100 H100-hours (~USD 200) on `s1 -> s2 -> s3 -> s4` from scratch mostly re-learns general retrieval that `GTE-ModernBERT` already has, for a projected +0.5 to +1.8 MTEB gain.
- Where `GTE-ModernBERT` and every current code embedder fail is **execution-level correctness (`ExecRetrieval`, `CoQuIR`)**: standard hard-negative mining (`mine_hard_negatives` with `relative_margin=0.05`) explicitly discards candidates within 5% cosine similarity of the positive as "false negatives", so standard embedders never see 95%-identical single-edit buggy twins (`<` vs `<=`, off-by-one) as negatives.

### Three-Part Behavioural Recipe (`ce3 behaviour-build` + `make pilot`)
1. **Execution-Verified Contest Twins (`behaviour_nl2code`, `behaviour_code2code`):**
   - Anchor passing solution `p.correct[0]` paired against its own single-edit mutants (`p.near_misses[anchor_code]`) and human near-miss submissions (`p.human_near_misses`) that pass some tests and fail others.
   - Equivalent syntactic rewrites (`p.rewrites[anchor_code]`, $\le 0.80$ line similarity) that pass 100% of tests serve as `code2code` positives.
   - Can run in **zero-LLM mode** (`make pilot MUTANTS=0 REWRITES=0`) using only human solutions and human near-misses from `CodeContests`.
2. **Zero-LLM Real-World Repository Bug-Fix Twins (`behaviour_swe_twins`):**
   - Extracted directly from `SWE-bench/SWE-smith` unified diffs (`extract_patch_twins`): SWE-smith's `patch` introduces the bug, so `problem_statement` is paired with the original correct code (`-` + context lines) as positive and the injected buggy twin (`+` + context lines) as `negative_1`. Only single-file, single-hunk patches whose change stays inside one definition are kept, with context from neighbouring functions trimmed (~3.3k twins per 20k rows).
   - No MTEB(Code, v1)/CoIR/CoREB(v1) task derives from SWE-smith, and a MinHash check of 4.4k twins against those tasks plus ExecRetrieval/CoQuIR found no overlap. FAIL_TO_PASS tests verify each bug; no LLM generation cost.
3. **Lean Replay Buffer & Sharper Twin Temperature (~USD 3 Pilot on 1x RTX 4090):**
   - Replay mix (`make mine-gte`) reduced from `250,000 rows/source x 7 negatives` (~2.7M rows, 24.3M sequences) to `25,000 rows/source x 4 negatives` (~250k rows, 1.5M sequences) at `max_seq_length: 512` and `temperature: 0.015`.
   - Cuts GPU time by **~16x** (from ~55h to ~4-6h on 1x RTX 4090, ~USD 1.50-3.00) while keeping a balanced replay anchor so MTEB(Code) does not regress beyond the 0.5-point pilot gate (`pilot_gate`).

## Decisions

- **Zero-shot by construction.** mteb flags a model by provenance: any declared training dataset that is an eval task, or a base model's (`adapted_from`). Row-level dedup does not clear the flag. The default mix therefore contains no source that shares provenance with a task in MTEB(Code, v1), CoIR or CoREB(v1): 100% zero-shot on all three, checked by `ce3 zero-shot` before training and again by `ce3 mteb`, which refuses to run below 100%. The audit still removes exact and near-duplicate test rows, because GitHub-scraped sources can contain benchmark code without sharing its provenance.
- **Context: 8192 tokens.** That is ModernBERT's and GTE-ModernBERT's trained length. Going to 16k or 32k would need RoPE scaling plus long-document training data, and quality past the trained length is untested. That is the same trap as v2's 1k to 8k interpolation. No MTEB(Code)/CoIR task needs more than 8k.
- **Dimensions: 768, 512, 256, 128, 64** (Matryoshka, native hidden size 768). v2's 1536 came from an up-projection: twice the index size and no extra information, since a linear map of a 768-dim vector still lies in a 768-dim subspace. 384 or 32 can be added to `matryoshka_dims` if a deployment needs them.
- **Hyperparameters come from Optuna on the dev suite, never from MTEB** (`ce3 tune`). Every trial trains on the same number of samples, so batch sizes compare fairly. A median pruner stops clearly worse trials. The top 3 rerun with two more seeds, and the best mean wins. Tuning against MTEB would overfit its test sets.
- **What is public** (superseded 2026-11-17: the toolkit, recipe, research log and experiment drivers are now open source). Originally only `ce3 release` output: weights, tokenizer, configs and a functional card (dimensions, prompts, context length, usage, scores, zero-shot %, contamination totals). sentence-transformers' auto card (datasets with samples, losses, hyperparameters) never leaves the private stage repos. The same goes for the data mix, synthetic prompts, tuning results and this repository. The base model is named in the card for the Apache-2.0 attribution and so mteb's `adapted_from` stays honest.
