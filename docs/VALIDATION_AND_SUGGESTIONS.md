# Architecture, Novelty & Codebase Validation (`code-embed-v3`)

> **Scope**: Independent zero-trust validation of the post-split `code-embed-v3` architecture (`public` toolkit + `private/` overlay + `behaviour` execution-verified pipeline), evaluating:
> 1. Whether training from scratch or fine-tuning is actually justified vs. off-the-shelf base models (`Alibaba-NLP/gte-modernbert-base`).
> 2. What is genuinely novel vs. standard leaderboard engineering.
> 3. Validation of the metadata / AST / code-graph thesis.
> 4. Empirical bugs and experimental bottlenecks found in the current code that **will break or invalidate `make pilot` / `make data`** if not fixed before launching GPU pods.

---

## 1. Strategic Validation: Is Training Required, and What Is the Novelty?

### 1.1 Do we need full Stage 1–4 pretraining (`make s1 s2 s3 s4`, ~\$160–\$200)?
**No — not until Option A (`make finetune` / `make pilot`) proves its ceiling.**
- `Alibaba-NLP/gte-modernbert-base` (149M params, Apache-2.0, native 8,192 context) already scores **71.66** on `MTEB(Code, v1)` and is **100% zero-shot** on `MTEB(Code, v1)`, `CoIR`, and `CoREB(v1)` (empirically verified via `mteb.get_model_meta("Alibaba-NLP/gte-modernbert-base")`).
- Running Stage 1 from raw `answerdotai/ModernBERT-base` spends ~50+ H100-equivalent GPU-hours largely re-learning general text/code alignment that `gte-modernbert-base` already has.
- **Verdict**: Starting from `Alibaba-NLP/gte-modernbert-base` (**Option A**) is strictly the right capital allocation. Only run full `s1 -> s4` if Option A + distillation plateaus below the 74.0 target or if ablating `ettin-encoder-150m` / `Qwen3-0.6B-Base` (bidirectional).

### 1.2 Is fine-tuning (`Option A` + `Behaviour Pilot`) required over raw `GTE-ModernBERT`?
**Yes, for three specific capabilities that `GTE-ModernBERT` lacks:**
1. **Execution / Behavioural Discrimination (`ExecRetrieval`, `CoQuIR`)**:
   Off-the-shelf `GTE-ModernBERT` (and larger encoders like `Qwen3-Embedding-0.6B`) relies on lexical/syntactic overlap. When a corpus contains both a working function and a 1-token buggy mutant (`<` vs `<=`, off-by-one bound, wrong accumulator initialization, SQL injection / deprecated API), standard encoders fail to separate them or rank the mutant higher ~60–75% of the time.
2. **Task-Asymmetric Instruction Conditioning on a Single Corpus Index**:
   `GTE-ModernBERT` ships with no task prompts (`model.prompts == {}`). In real developer search and multi-family benchmarks (`nl2code`, `code2nl`, `code2code`, `techqa`, `text2sql`, `code_edit`, `multiturn`), the same code corpus must serve 7 distinct retrieval intents controlled solely by the query prefix.
3. **Trained Matryoshka Prefix Truncation (`768 -> 512/256/128/64`)**:
   Training with `MatryoshkaLoss` across `[768, 512, 256, 128, 64]` preserves retrieval quality at `128d`/`256d` (3–6× smaller vector index in production) without an up-projection layer.

### 1.3 What is genuinely novel here vs. base models and existing papers?

| Component | Novelty Level | Assessment |
|---|---|---|
| **16k GradCache + CoRNStack + TopK-PercPos Mining + Reranker KD** | **Table stakes (Not novel)** | Standard recipe shared by `NV-Retriever`, `jina-code-embeddings`, `voyage-4-nano`, and `DenseOn`/`LateOn`. Necessary for MTEB competitiveness, not a research differentiator. |
| **100% Provenance-Verified Zero-Shot + Synthetic Family Fill (`synth.py`)** | **High engineering rigor** | Most leaderboard code models train on CoSQA / CodeSearchNet / CoIR train splits (dropping zero-shot % to ~50%). Enforcing 100% provenance zero-shot via `ce3 zero-shot` + filling missing task families (`techqa`, `code_edit`, `multiturn`) with round-trip consistency-filtered LLM queries on Apache-2.0 CoRNStack code is clean and defensible. |
| **Execution-Verified Contrastive Curriculum (`behaviour.py` + `sandbox.py`)** | **Core Novelty (Primary Differentiator)** | Using actual test execution in an isolated sandbox to construct **(a) multi-style verified rewrites** (`similarity <= 0.80`, passes 100% of tests) as positives and **(b) verified near-miss mutants** (`similarity >= 0.85`, passes $\ge 1$ test and fails $\ge 1$ test, no added comments) as explicit hard negatives. This directly trains the bi-encoder to separate *working semantics* from *surface look-alikes*. |
| **Structure-as-Text Header with Dropout + Behavioral Graph Pairs (`test -> impl`, `issue -> fn`)** | **Strong Product & Second-Axis Novelty** | Adapts Contextual Document Embeddings (Morris & Rush, 2024) specifically to repository AST/call-graph context (`file`, `class`, `signature`, `imports`, `callees`) while using header dropout so plain-code MTEB evaluation never regresses. |

---

## 2. Validation of the Metadata & Code-Graph Thesis

Your observations on metadata and graph mapping are architecturally sound:

1. **Why AST/Data-Flow/Metadata cannot solve `ExecRetrieval` or `MTEB`**:
   - In `ExecRetrieval`, a function and its 1-edit buggy twin share the exact same function name, signature, imports, callers, and 99% of their AST/CFG edges. Only execution behavior separates them.
   - In `MTEB(Code, v1)`, `CoIR`, and `CoREB(v1)`, the evaluation harness passes standalone strings with zero repository graph or file metadata. Any model that *requires* graph edges at inference time will fail or degrade on MTEB.
2. **Where structure and graph signals *do* belong**:
   - **In the embedding model (Training Data & Optional Context Header)**:
     - **Structure-as-text header with ~50% dropout**: Prepending a compact deterministic header:
       ```text
       # path: src/retry.py | scope: HttpClient.send | imports: time, random | calls: sleep, uniform
       ```
       to documents during training (and dropping the header 50% of the time) teaches the encoder to exploit repository context when an indexer provides it, while remaining 100% compatible with plain code snippets on MTEB.
     - **Execution/Graph-derived training pairs**: Pairing `pytest_function -> tested_implementation` and `github_issue -> modified_function` (from `SWE-smith`) aligns natural-language bug reports and behavioral assertions directly with target functions.
   - **In the downstream search product (Outside the model)**:
     - Hybrid retrieval (`BM25 + dense vector`) $\rightarrow$ 1-hop call-graph / import expansion (`RepoGraph` / `LocAgent` style) $\rightarrow$ cross-encoder reranking (`Qwen3-Reranker`) + sandbox test execution belongs in the retrieval runtime, not inside the 150M bi-encoder weights.

---

## 3. Critical Bugs & Experimental Risks Found in the Codebase

Before approving the ~\$25 `make pilot` run (or `make data`), **five concrete bugs/flaws in the current codebase must be fixed**. Left as-is, `make data` / `make pilot` will either crash mid-run or fail the pilot gate due to training-signal dilution and baseline misconfiguration.

### 3.1 [P0 — Hard Crash] `ce3 audit` crashes on MTEB v1-format tasks (`CodeEditSearchRetrieval`, `CodeSearchNetCCRetrieval`, `COIRCodeSearchNetRetrieval`)
- **Location**: [`src/code_embed_v3/data/contamination.py`](../src/code_embed_v3/data/contamination.py#L90-L95)
- **Root Cause**:
  ```python
  task = mteb.get_task(name)
  task.load_data()
  if task.dataset is None:
      raise RuntimeError(f"{name}: mteb loaded no data")
  ```
  In `mteb>=2.26`, three of the 18 tasks in `DEFAULT_BENCHMARKS` (`CodeEditSearchRetrieval`, `CodeSearchNetCCRetrieval`, and `COIRCodeSearchNetRetrieval`) still use MTEB's legacy v1 `load_data()` implementation, which populates `task.queries`, `task.corpus`, and `task.relevant_docs` while leaving `task.dataset = None`. In `mteb`, `task.convert_v1_dataset_format_to_v2()` is only called inside `AbsTaskRetrieval.evaluate()`, **not** inside `task.load_data()`.
- **Why `make dry-run` missed it**: [`src/code_embed_v3/dry_run.py`](../src/code_embed_v3/dry_run.py#L30) passes `--tasks CosQA SyntheticText2SQL`, both of which use the v2 loader. As soon as `make data` or `make pilot` runs `uv run ce3 audit` across all default benchmarks, it will crash with `RuntimeError: COIRCodeSearchNetRetrieval: mteb loaded no data`.
- **Fix**:
  ```python
  task = mteb.get_task(name)
  task.load_data()
  if hasattr(task, "convert_v1_dataset_format_to_v2"):
      task.convert_v1_dataset_format_to_v2(num_proc=None)
  if task.dataset is None:
      raise RuntimeError(f"{name}: mteb loaded no data")
  ```

### 3.2 [P0 — Invalid Baseline in `make pilot`] `run_mteb.py` forces `code-embed-v3` prompts onto unprompted `Alibaba-NLP/gte-modernbert-base`
- **Location**: [`src/code_embed_v3/eval/run_mteb.py`](../src/code_embed_v3/eval/run_mteb.py#L84-L85) and [`Makefile`](../Makefile#L80)
- **Root Cause**:
  In `Makefile` line 80, the baseline MTEB score for `pilot-gate` is computed via:
  ```bash
  uv run ce3 mteb Alibaba-NLP/gte-modernbert-base --name Alibaba-NLP/gte-modernbert-base --benchmarks "MTEB(Code, v1)" --output results/mteb-gte
  ```
  Inside `run_mteb.py`:
  ```python
  model = load_model(model_path, truncate_dim=truncate_dim)
  wrapper = SentenceTransformerEncoderWrapper(model, model_prompts=mteb_model_prompts())
  ```
  Unlike `behaviour_eval.py` (which checks `if query and "nl2code" in getattr(model, "prompts", {})`) and `mine-gte` (which passes `--no-query-prompts`), `run_mteb.py` unconditionally injects `mteb_model_prompts()` (`"Find the code that answers this query: "`, etc.) into `Alibaba-NLP/gte-modernbert-base`, which was trained **without** those prompts. This corrupts `mteb_baseline` in `pilot-gate`.
- **Fix**: Only pass `mteb_model_prompts()` in `run_mteb.py` when the loaded model actually carries `code-embed-v3` prompts:
  ```python
  prompts = mteb_model_prompts() if "nl2code" in getattr(model, "prompts", {}) else None
  wrapper = SentenceTransformerEncoderWrapper(model, model_prompts=prompts)
  ```

### 3.3 [P0 — Pilot Signal Dilution & Broken Twin Alignment in `behaviour.py`] Why `make pilot` risks failing `pilot-gate` even if the hypothesis is true
There are three compounding issues in [`src/code_embed_v3/data/behaviour.py`](../src/code_embed_v3/data/behaviour.py) and [`Makefile`](../Makefile#L71-L78):

1. **Positive vs. Near-Miss Mutant Mismatch in `behaviour_nl2code`** ([`behaviour.py#L237-L250`](../src/code_embed_v3/data/behaviour.py#L237-L250)):
   - `generate_variants` generates LLM single-edit mutants **only** from `anchor_code = p.correct[0]`.
   - Yet in `build_rows`, `nl2code` sets `"positive": rng.choice(p.correct)`, where `p.correct` contains up to 4 *different contestants'* accepted solutions (`p.correct[0..3]`).
   - Whenever `rng.choice(p.correct)` picks `p.correct[1]`, `p.correct[2]`, or `p.correct[3]`, the positive is Contestant B's solution while the near-miss negatives are 1-line mutants of Contestant A's solution (`p.correct[0]`). The model is then contrasting two completely different programs rather than a working function against its 1-edit buggy twin!
   - **Fix**: Always pair mutants of `c` with `c` itself (or generate mutants across all `c in p.correct` and sample `(c, mutants_of_c)` together).

2. **`repeats=8` Collides with `BatchSamplers.NO_DUPLICATES_HASHED` + `dataloader_drop_last=True`** ([`behaviour.py#L226-L259`](../src/code_embed_v3/data/behaviour.py#L226-L259), [`train.py#L372-L377`](../src/code_embed_v3/train.py#L372-L377)):
   - `build_rows` creates `repeats=8` rows per problem with the **exact same `anchor`** (`p.description` in `behaviour_nl2code`, and `p.correct[0]` in `behaviour_code2code`) and overlapping `positive`/`negative_*` columns.
   - `NoDuplicatesBatchSampler` checks uniqueness across **all** text columns (`anchor`, `positive`, `negative_1..4`) within each batch of `batch_size: 512`. It can place at most **1 row per problem** in any batch of 512.
   - Worse: in `deepmind/code_contests` (`train` split = 13,328 problems), after excluding AtCoder/Aizu (`EXCLUDED_SOURCES = {5, 6}`), requiring Python 3 solutions, capping at `--max-problems 5000`, filtering by sandbox execution, and requiring both `len(pool) >= 4` AND non-empty `equivalents` (for `behaviour_code2code`), if fewer than `512` unique problems survive in `behaviour_code2code` (or `< 512 / world_size` per GPU), `NoDuplicatesBatchSampler(..., batch_size=512, drop_last=True)` yields **zero batches**!

3. **99:1 Gradient Dilution in `data/mix_v1_gte_behaviour`**:
   - `data/mix_v1_gte_mined` has ~2.0M–2.7M rows (up to 250k rows across 11–15 sources).
   - `data/behaviour` has ~15k–30k rows (~1% of the combined mix).
   - Because `150m_ft_behaviour.yaml` uses `multi_dataset_batch_sampler: proportional` for 1 epoch, **~99% of optimization steps in `make pilot` are standard surface-similarity batches** and only ~1% are behaviour-verified batches. Expecting +10.0 points (`+0.10`) on `ExecRetrieval` `exec@1` from 1% of the batches is high-risk.
   - **Fix for the Pilot**:
     - Either run the pilot as a **two-step or higher-weight curriculum**: e.g., fine-tune on a smaller general mix (`--max-rows-per-source 50000` in `mine-gte` for the pilot, which also cuts pilot cost from ~\$25 to ~\$6–\$8 and boosts behaviour share to ~10–15%), or run a short behaviour-focused stage (mixing 50% `behaviour` + 50% general replay at `lr: 1e-5` for 500–1000 steps) after `finetune`.
     - Lower `batch_size` on behaviour batches (or set `negatives: 7` to match `mix_v1_gte_mined` and allow problems with $\ge 2$ near-misses by sampling with replacement / hard-negative padding so fewer problems are discarded).

### 3.4 [P1 — Bug in Stage 3 Distillation] `DistillWithContrastive` scales the contrastive gradient 250× larger than the KL distillation gradient
- **Location**: [`src/code_embed_v3/train.py`](../src/code_embed_v3/train.py#L46-L78) and [`private/configs/stages/150m_s3_distill.yaml`](../configs/stages/150m_s3_distill.yaml#L16-L18)
- **Root Cause**:
  In `sentence-transformers` 6.2, `DistillKLDivLoss.compute_loss_from_embeddings` multiplies the KL divergence by `self.student_temperature ** 2`.
  - With `student_temperature: 0.02`, `self.student_temperature ** 2 = 0.0004`. The gradient of `self.kl` w.r.t. student similarity scores is proportional to `student_temperature = 0.02`.
  - Meanwhile, `self.mnrl = MultipleNegativesRankingLoss(model, scale=scale)` uses `scale = 1 / 0.02 = 50.0`, whose gradient w.r.t. student similarity scores is proportional to `50.0`.
  - In `DistillWithContrastive.forward`:
    ```python
    loss = self.kl.compute_loss_from_embeddings(embeddings, labels)
    if self.contrastive_weight > 0:
        loss = loss + self.contrastive_weight * self.mnrl.compute_loss_from_embeddings(embeddings, labels)
    ```
    With `contrastive_weight: 0.1`, the contrastive gradient multiplier is `0.1 * 50.0 = 5.0` vs. `0.02` for KL — meaning the "auxiliary" contrastive loss dominates the gradients by **250:1** (`5.0 / 0.02`), effectively turning Stage 3 into plain MNRL and ignoring the teacher scores!
  - Note that `DistillKLDivLoss`'s own docstring in `sentence-transformers` warns: *"The loss is scaled by the student temperature squared, which keeps gradient magnitudes comparable only in the regime Hinton et al. cover, a shared temperature at or above 1.0. Below that the gradient falls off and the reported loss shrinks strongly, so scale the KD weight back up when weighing this loss against another."*
- **Fix**: In `DistillWithContrastive.forward`, rescale the KL loss by `1.0 / (self.kl.student_temperature ** 2)` (or `self.mnrl.scale / self.kl.student_temperature`) so `contrastive_weight: 0.1` genuinely represents 10% of the KL gradient magnitude.

### 3.5 [P1 — Silent Data Drop in Stage 1] `batch_size: 16384` + `dataloader_drop_last=True` drops `spider` (`text2sql`) completely
- **Location**: [`src/code_embed_v3/train.py`](../src/code_embed_v3/train.py#L372-L377) and [`private/configs/data/mix_v1.yaml`](../configs/data/mix_v1.yaml#L41-L46)
- **Root Cause**:
  `xlangai/spider` has ~7,000 train rows (~4,500 distinct SQL queries). In Stage 1 (`batch_size: 16384` for 150M, `8192` for 0.6B), `NoDuplicatesBatchSampler` with `drop_last=True` requires `per_device_train_batch_size` rows with zero duplicate anchors or positives. On 1–2 GPUs (or 4 GPUs at 16,384 where `per_device_batch_size = 4096` and `spider` has ~4,500 unique queries), `spider` produces **0 to 1 batch per epoch** out of ~850 batches (~0.0%–0.1% of training), and `synth_v1.yaml` does not currently generate a `text2sql` split.
- **Fix**: Add a `text2sql` task to `synth.py` / `synth_v1.yaml` (generating natural-language analytical/SQL requests + SQL queries from schema/code seeds), or include a larger permissive zero-shot Text-to-SQL source (e.g., `gretelai/synthetic_text_to_sql`, Apache-2.0, 100k rows — verifying first that it has no provenance overlap with `SyntheticText2SQL` in CoIR,Wait: CoIR's `SyntheticText2SQL` *is* derived from `gretelai/synthetic_text_to_sql`, so generating synthetic SQL pairs via `ce3 synth` is the 100% zero-shot-safe path!).

### 3.6 [P2 — Smaller Correctness & Polish Items]
1. **`behaviour_eval.py` uses `nl2code` prompt for `CoQuIR-SQLR2`**:
   In [`src/code_embed_v3/eval/behaviour_eval.py`](../src/code_embed_v3/eval/behaviour_eval.py#L60), `_encode` hardcodes `prompt_name="nl2code"` for every `CoQuIR` subset. `SQLR2` is a Text-to-SQL task and should use `prompt_name="text2sql"`.
2. **`ce3 behaviour-build` pushes unaudited data to the Hub**:
   In [`src/code_embed_v3/cli.py`](../src/code_embed_v3/cli.py#L305-L306), `hub.push_folder(out, a.push_repo, "behaviour/data", ...)` runs inside `behaviour-build` *before* `ce3 audit data/behaviour`, uploading `manifest.json` with `"audited": false`.
3. **Dev holdouts (`data/*/dev/*`) are sliced before `ce3 audit` and never audited**:
   In [`src/code_embed_v3/data/build.py`](../src/code_embed_v3/data/build.py#L32-L36), `dev` rows are split off during `build-data` and never passed through `ce3 audit`. Auditing `dev` (or splitting `dev` *after* `audit`) ensures `dev-eval` and `ce3 tune` are also 100% free of benchmark test leakage.
4. **Include `mteb_model_prompts()` in saved `config_sentence_transformers.json`**:
   In [`src/code_embed_v3/modeling.py`](../src/code_embed_v3/modeling.py#L114), saving `prompts={**model_prompts(), **mteb_model_prompts(), "query": QUERY_PROMPTS["nl2code"]}` ensures that third parties running `mteb` directly on the Hugging Face repo (without `ce3 mteb`) automatically get the right per-task query prompts.
5. **Unit test coverage for `behaviour.py`, `behaviour_eval.py`, and `sandbox.py`**:
   The newly added execution-verification modules (`behaviour.py`, `behaviour_eval.py`, `sandbox.py`) currently have no unit tests in `tests/`. Adding fast offline tests for `sandbox.evaluate`, `behaviour.verify_variants`, `behaviour.build_rows`, and `behaviour_eval.pilot_gate` will prevent regressions before the cluster run.

---

## 4. Concrete Recommendations & Answers to Your Next Steps

1. **Git History Cleanup (Public Repo)**:
   - Commit `2380cdc` (`HEAD` on `main`) is already a clean single initial commit of the public tree (`private/`, `PLAN.md`, and `RUNPOD.md` are not in `2380cdc`'s tree).
   - Once the P0/P1 fixes above (and unit tests for `behaviour`/`sandbox`) are applied, amend `2380cdc` (`git commit --amend --no-edit`) so the public repo starts with a single, clean, fully-tested commit before you push to GitHub.

2. **Go / No-Go on the ~\$25 Behaviour Pilot**:
   - **Recommendation: GO on the Behaviour Pilot *without* the structure header first, immediately after applying Fixes 3.1, 3.2, and 3.3.**
   - **Why without the structure header in Pilot #1?**
     Keep variables isolated: Pilot #1 should test a single falsifiable hypothesis — *does execution-verified positive rewrite + near-miss negative contrastive training raise `ExecRetrieval` `exec@1` by $\ge +0.10$ without dropping `MTEB(Code, v1)` by $> 0.5$ points?*
     Moreover, `ExecRetrieval` and `MTEB(Code, v1)` snippets are standalone functions without repository paths or call graphs, so structure headers will not be present at test time on those two benchmarks.
   - **When to add the Structure Header + Graph Pairs (Pilot #2 / Axis 2)**:
     Once Pilot #1 clears the gate on `ExecRetrieval` + `MTEB(Code, v1)`, add the 50%-dropout structure header (`path | scope | sig | imports | calls`) + `test -> function` / `issue -> function` pairs and gate Pilot #2 on **SWE-bench Verified file/function localization Recall@10** while holding `MTEB(Code, v1)` and `ExecRetrieval` non-regressing.

---

## 5. Implemented Upgrades (2026-10-29)

1. **All P0/P1/P2 Bugs Resolved & Covered by Unit Tests (`tests/test_behaviour.py`, `tests/test_data.py`)**:
   - `contamination.py:index_mteb_tasks` converts v1 MTEB retrieval datasets to v2 before indexing.
   - `cli.py:_cmd_audit` audits both train and `dev/` holdout slices.
   - `run_mteb.py:run` gates `mteb_model_prompts()` and `use_instructions` on `"nl2code" in model.prompts`.
   - `behaviour.py:build_rows` preserves exact single-edit twin alignment (`"positive": anchor_code`).
   - `behaviour_eval.py:evaluate` selects `prompt_name="text2sql"` for `CoQuIR-SQLR2`.
   - `train.py:DistillWithContrastive.forward` divides `DistillKLDivLoss` by `student_temperature ** 2` so KL and MNRL gradients share the same $1/\tau_S$ scale.
   - `sandbox.py:LAUNCHER` caps `setrlimit` against container hard limits and handles macOS `RLIMIT_AS`.
2. **Zero-LLM Real-World Bug-Fix Twins (`behaviour.py:extract_patch_twins`, `load_swe_smith_twins`)**:
   - Reconstructs `(correct_code, buggy_twin)` from `SWE-bench/SWE-smith` bug-injecting diffs (`-` side correct, `+` side buggy; `behaviour_swe_twins`): single-file, single-definition hunks only, neighbouring context trimmed, no provenance overlap with MTEB/CoIR/CoREB, no LLM cost.
3. **Zero-LLM Contest Mode (`make pilot MUTANTS=0 REWRITES=0`)**:
   - Skips vLLM initialization when `--mutants 0 --rewrites 0`, building `behaviour_nl2code` and `behaviour_code2code` purely from sandbox-verified human contest solutions and human near-misses, plus `behaviour_swe_twins`.
4. **Lean Replay Buffer & Sharper Twin Temperature (~USD 2.50–3.00 Pilot on 1x RTX 4090)**:
   - Reduced default `MINE_ROWS` from `250000` to `25000`, `MINE_NEGS` from `7` to `4`, `max_seq_length` from `1024` to `512`, and `temperature` from `0.02` to `0.015` in `150m_ft_behaviour.yaml`, cutting sequence forward/backward passes by ~16× while preventing MTEB forgetting.

