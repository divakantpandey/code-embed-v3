# code-embed-v3

A compact (150M) embedding model for code search, and the toolkit that trains it: natural language to code, code to description, code to code, programming Q&A, text to SQL, code changes and multi-turn developer conversations. 768-dimensional Matryoshka embeddings (truncatable to 64), 8k context, runs on CPU.

## Usage

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("divakantpandey/code-embed-v3")
q = model.encode(["retry an http request with exponential backoff"], prompt_name="nl2code")
d = model.encode([open("client.py").read()])  # documents never take a prompt
print(model.similarity(q, d))
```

Query prompt names: `nl2code`, `code2nl`, `code2code`, `techqa`, `text2sql`, `code_edit`, `multiturn` (see `src/code_embed_v3/prompts.py`). One document index serves every task.

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

## Public examples, private recipe

`configs/` holds working example configs and prompts. The data mix, hyperparameters and generation prompts behind the released model live in a separate private checkout at `private/`, which overrides any file at the same relative path (`src/code_embed_v3/paths.py`). Without it, every command runs on the examples.

## License

Apache-2.0.
