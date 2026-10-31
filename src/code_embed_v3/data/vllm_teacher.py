"""Teacher scoring with vLLM, executed in its own environment by ``mining.vllm_scorer``.

vLLM pins older torch/transformers than this project, so this file imports nothing from
``code_embed_v3`` and only needs ``vllm`` and ``pyarrow``::

    uv run --no-project --with vllm==0.32.0 --with pyarrow \
        python -I vllm_teacher.py --model ... --jobs jobs.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from vllm import LLM, PoolingParams

# vLLM's examples/pooling/score/template/qwen3_reranker.jinja with the instruction inlined, matching the
# Qwen3-Reranker model card prompt. The score is read at the last token, which the model card makes
# "</think>\n\n"; chat-template rendering drops one trailing newline, hence three here.
TEMPLATE = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
    'Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
    "<Instruct>: @@INSTRUCTION@@\n"
    '<Query>: {{ messages | selectattr("role", "eq", "query") | map(attribute="content") | first }}\n'
    '<Document>: {{ messages | selectattr("role", "eq", "document") | map(attribute="content") | first }}'
    "<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n\n"
)

# The original checkpoints are causal LMs; this turns the yes/no logits into one classifier output.
ORIGINAL_QWEN3_RERANKER = {
    "architectures": ["Qwen3ForSequenceClassification"],
    "classifier_from_token": ["no", "yes"],
    "is_original_qwen3_reranker": True,
}


def truncate(tokenizer: Any, texts: list[str], max_tokens: int) -> list[str]:
    ids = tokenizer(texts, add_special_tokens=False, truncation=True, max_length=max_tokens)["input_ids"]
    return tokenizer.batch_decode(ids)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--jobs", required=True)
    p.add_argument("--max-query-tokens", type=int, default=512)
    p.add_argument("--max-doc-tokens", type=int, default=2048)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    a = p.parse_args()

    overrides = ORIGINAL_QWEN3_RERANKER if a.model.startswith("Qwen/Qwen3-Reranker") else {}
    llm = LLM(
        model=a.model,
        runner="pooling",
        hf_overrides=overrides,
        max_model_len=a.max_query_tokens + a.max_doc_tokens + 256,
        gpu_memory_utilization=a.gpu_memory_utilization,
    )
    tokenizer = llm.get_tokenizer()
    # Raw log-odds, same scale as sentence-transformers' LogitScore with an Identity activation.
    raw = PoolingParams(use_activation=False)

    for job in json.loads(Path(a.jobs).read_text()):
        out = Path(job["output"])
        if out.is_file():
            continue
        table = pq.read_table(job["input"]).to_pydict()
        doc_cols = [c for c in table if c != "anchor"]
        queries = truncate(tokenizer, table["anchor"], a.max_query_tokens)
        q_rep = [q for q in queries for _ in doc_cols]
        docs = [table[c][i] for i in range(len(queries)) for c in doc_cols]
        docs = truncate(tokenizer, docs, a.max_doc_tokens)
        template = TEMPLATE.replace("@@INSTRUCTION@@", job["instruction"])
        outputs = llm.score(q_rep, docs, pooling_params=raw, chat_template=template, use_tqdm=False)
        scores = np.asarray([o.outputs.score for o in outputs], dtype=np.float32)
        tmp = out.with_name(out.name + ".tmp.npy")
        np.save(tmp, scores.reshape(len(queries), len(doc_cols)))
        os.replace(tmp, out)
        print(f"scored {out.name}: {len(docs)} pairs", flush=True)


if __name__ == "__main__":
    main()
