"""Public release: the model and how to use it, nothing about how it was made.

sentence-transformers writes a model card listing the training datasets (with sample rows), every
loss and its parameters, and all hyperparameters. That is the recipe, so it never leaves the private
stage repos. A release copies an allow-list of files, writes its own card from what a user needs
(dimensions, prompts, context length, usage, scores), and summarises the contamination audit as
totals only.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

# Top-level files a SentenceTransformer needs to load; everything else (README, trainer state,
# optimizer, eval CSVs, our metadata unless it carries a runtime patch) stays private.
TOP_LEVEL = {
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "sentence_bert_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "merges.txt",
    "added_tokens.json",
}
MODULE_SUFFIXES = {".json", ".safetensors"}


def public_files(model_dir: Path) -> list[Path]:
    files = [model_dir / name for name in sorted(TOP_LEVEL) if (model_dir / name).is_file()]
    for module in json.loads((model_dir / "modules.json").read_text()):
        if module["path"]:
            files += sorted(p for p in (model_dir / module["path"]).iterdir() if p.suffix in MODULE_SUFFIXES)
    meta = model_dir / "code_embed_v3.json"
    if meta.is_file() and json.loads(meta.read_text()).get("bidirectional"):
        files.append(meta)  # required to reapply the bidirectional patch at load time
    return files


def contamination_summary(report: dict[str, dict[str, Any]], audited_tasks: list[str] | None) -> str:
    before = sum(int(r["before"]) for r in report.values())
    removed = before - sum(int(r["after"]) for r in report.values())
    tasks = f"{len(audited_tasks)} tasks of" if audited_tasks else "the tasks of"
    return (
        f"Before training, every training row was checked against the test queries and relevant documents of "
        f"{tasks} MTEB(Code, v1), CoIR and CoREB(v1), by exact match and MinHash near-duplicate search. "
        f"{removed:,} of {before:,} rows ({100 * removed / max(before, 1):.3f}%) were removed."
    )


def model_card(
    repo: str,
    model_dir: Path,
    base_model: str | None,
    zero_shot: dict[str, int | None] | None,
    scores: dict[str, float] | None,
    contamination: str | None,
) -> str:
    from code_embed_v3.modeling import load_model

    # Read from the loaded model: config file keys differ across sentence-transformers versions.
    model = load_model(model_dir, device="cpu")
    dims = model.get_embedding_dimension()
    max_len = model.max_seq_length
    smaller = [d for d in (512, 256, 128, 64) if dims and d < dims]
    prompts = {k: v for k, v in model.prompts.items() if v}
    header = [
        "---",
        "license: apache-2.0",
        "library_name: sentence-transformers",
        "pipeline_tag: sentence-similarity",
    ]
    header += ["tags:", "- sentence-transformers", "- code", "- code-search", "- embeddings"]
    if base_model:
        header += [f"base_model: {base_model}"]
    header += ["---", ""]
    lines = [
        f"# {repo.split('/')[-1]}",
        "",
        "A compact embedding model for code search: natural language to code, code to description, "
        "code to code, programming Q&A, text to SQL, code changes and multi-turn developer conversations.",
        "",
        "| | |",
        "|---|---|",
        f"| Embedding dimensions | {dims} (Matryoshka: truncate to {', '.join(str(d) for d in smaller)}) |",
        f"| Max sequence length | {max_len} tokens |",
        "| Similarity | cosine (embeddings are normalised) |",
        "",
        "## Usage",
        "",
        "Queries take a task prompt; documents never do, so one index serves every task.",
        "",
        "```python",
        "from sentence_transformers import SentenceTransformer",
        "",
        f'model = SentenceTransformer("{repo}")',
        'q = model.encode(["retry an http request with exponential backoff"], prompt_name="nl2code")',
        'd = model.encode([open("client.py").read()])',
        "print(model.similarity(q, d))",
        "",
        "# smaller index: keep the first 256 dimensions",
        f'small = SentenceTransformer("{repo}", truncate_dim=256)',
        "```",
        "",
        "| `prompt_name` | Query type |",
        "|---|---|",
        *(f"| `{name}` | {text.strip()} |" for name, text in sorted(prompts.items())),
        "",
    ]
    if scores:
        lines += ["## Evaluation", "", "| Task | Score |", "|---|---|"]
        lines += [f"| {task} | {score:.4f} |" for task, score in sorted(scores.items())]
        lines += [""]
    if zero_shot or contamination:
        lines += ["## Training data and contamination", ""]
        if zero_shot:
            pct = ", ".join(f"{b}: {v}%" for b, v in zero_shot.items())
            lines += [
                f"Zero-shot by mteb's own check: {pct}. No training source shares provenance with an "
                "evaluation task.",
                "",
            ]
        if contamination:
            lines += [contamination, ""]
    lines += ["## License", "", "Apache-2.0."]
    if base_model:
        lines[-1] += f" Fine-tuned from [{base_model}](https://huggingface.co/{base_model}) (Apache-2.0)."
    return "\n".join(header + lines) + "\n"


def mteb_scores(results_dir: Path) -> dict[str, float]:
    """Main score per task from an mteb results folder (``ce3 mteb --output``)."""
    scores: dict[str, float] = {}
    for path in results_dir.rglob("*.json"):
        if path.name in {"model_meta.json", "zero_shot.json"}:
            continue
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or "task_name" not in data:
            continue
        splits = data.get("scores", {})
        values = [s["main_score"] for split in splits.values() for s in split if "main_score" in s]
        if values:
            scores[data["task_name"]] = sum(values) / len(values)
    return scores


def build_release(
    model_dir: Path,
    out: Path,
    repo: str,
    base_model: str | None = None,
    results_dir: Path | None = None,
    contamination_report: Path | None = None,
    audited_tasks: list[str] | None = None,
) -> Path:
    if out.exists():
        shutil.rmtree(out)
    for src in public_files(model_dir):
        dst = out / src.relative_to(model_dir)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    zero_shot = None
    if results_dir and (results_dir / "zero_shot.json").is_file():
        zero_shot = json.loads((results_dir / "zero_shot.json").read_text())
    summary = None
    if contamination_report:
        summary = contamination_summary(json.loads(contamination_report.read_text()), audited_tasks)
    scores = mteb_scores(results_dir) if results_dir else None
    (out / "README.md").write_text(model_card(repo, model_dir, base_model, zero_shot, scores, summary))
    return out
