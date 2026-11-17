# code-embed-v3

A toolkit for training compact code embedding models, and the full record of one research question asked with it: does fine-tuning on execution-verified near-miss code (single-edit buggy copies and passing rewrites) teach an embedder to rank working code above code that only looks right?

**Status: negative result, no model released.** Starting from GTE-ModernBERT-base (149M), near-miss data beat a replay-only control on ExecRetrieval exec@1 on every seed in small runs (+0.029 ± 0.010). At pilot scale it adds +0.014 over the same control, which is not significant (p = 0.21), and it alone collapses AppsRetrieval from 53.9 to 18.9. Fine-tuning on the replay mix by itself costs about 2 MTEB(Code) points. Every experiment, number and dead end is in [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md).

Behaviour evals for all three on one machine (float32, 8,192 tokens); MTEB in fp16, control and pilot on that machine, base from the original pilot run:

| Model | ExecRetrieval exec@1 | CoQuIR pairwise | MTEB(Code, v1) |
|---|---|---|---|
| GTE-ModernBERT-base | 0.0927 | 0.5528 | 71.78 |
| Replay-only control | 0.0852 | 0.5588 | 69.75 |
| Pilot fine-tune (11% near-miss data) | 0.0990 | 0.5527 | 66.02 |

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
