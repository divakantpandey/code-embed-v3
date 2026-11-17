# code-embed-v3

A toolkit for training compact code embedding models, and the full record of one research question asked with it: does fine-tuning on execution-verified near-miss code (single-edit buggy copies and passing rewrites) teach an embedder to rank working code above code that only looks right?

**Status: negative result, no model released.** Starting from GTE-ModernBERT-base (149M), near-miss data beat a replay-only control on ExecRetrieval exec@1 on every seed in small runs (+0.029 ± 0.010), but at pilot scale the gain over the base model was not significant (McNemar p = 0.45) and MTEB(Code, v1) dropped from 71.78 to 66.29. Every experiment, number and dead end is in [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md).

| Model | ExecRetrieval exec@1 | CoQuIR pairwise | MTEB(Code, v1) |
|---|---|---|---|
| GTE-ModernBERT-base | 0.0884 | 0.5523 | 71.78 |
| Pilot fine-tune (11% near-miss data) | 0.0990 | 0.5530 | 66.29 |
| 50/50 weight average of the two | 0.1022 | 0.5556 | 70.82 |

## What is in this repo

| Path | Contents |
|---|---|
| `src/code_embed_v3/` | The `ce3` toolkit: data building, contamination audit, near-miss construction, mining, training, evaluation |
| `configs/` | The exact data mix, stage hyperparameters, search space and generation prompts used in the experiments |
| `docs/RESEARCH_LOG.md` | Every experiment in order: procedure, results, what worked and what did not |
| `docs/PLAN.md`, `docs/RUNPOD.md`, `docs/DRYRUN_FINDINGS.md` | Strategy, GPU runbook and dry-run findings |
| `experiments/` | The drivers that ran each round on rented GPUs (`ablate*.py`, `bridge.py`, `pilot.py`, `merge.py`, `control.py`) |

The drivers read and write private Hugging Face repos under `divakantpandey/`. To rerun them, point `REPO`, `DATA_REPO` and the model repos at your own namespace.

## Zero-shot by construction

mteb flags a model by provenance: if any declared training dataset is an evaluation task (or is in the base model's declared training data), the model stops counting as zero-shot, however carefully the test rows were removed. This toolkit enforces 100% zero-shot on MTEB(Code, v1), CoIR and CoREB(v1):

- `ce3 zero-shot` computes mteb's own zero-shot percentage for a data mix before training.
- `ce3 train` refuses any source marked `overlaps_eval` unless the stage sets `allow_eval_overlap: true`.
- `ce3 mteb` refuses to evaluate below 100%, and writes the figure to `zero_shot.json`.
- `ce3 audit` also removes exact and near-duplicate (MinHash) matches of benchmark test queries and relevant documents, since scraped code can contain benchmark code without sharing its provenance.

## Toolkit

| Command | What it does |
|---|---|
| `ce3 build-data` / `ce3 audit` | Stream sources into a mix, hold out a dev slice, remove benchmark overlap |
| `ce3 synth` | Generate query-side pairs for real code with an open LLM (vLLM), resumable, round-trip filtered |
| `ce3 mine` / `ce3 label` | Positive-aware hard-negative mining; teacher scores for distillation |
| `ce3 train` | One stage on every visible GPU; resumes from the newest local or Hub checkpoint |
| `ce3 tune` | Optuna search on proxy runs, scored on the dev suite (never on MTEB) |
| `ce3 dev-eval` / `ce3 mteb` | Per-family dev suite; official mteb evaluation |
| `ce3 release` | Publish model files and a functional model card only |
| `ce3 doctor` / `ce3 dry-run` | Pre-flight checks; a tiny end-to-end rehearsal on a GPU machine |

`make check` runs ruff, mypy and the offline tests (tiny random models, no downloads).

## Your own recipe

`configs/` is the recipe used for the experiments. To try your own without editing it, put files at the same relative paths under `private/` (gitignored) or under `$CE3_PRIVATE`; they override `configs/` wherever they exist (`src/code_embed_v3/paths.py`).

## License

Apache-2.0.
