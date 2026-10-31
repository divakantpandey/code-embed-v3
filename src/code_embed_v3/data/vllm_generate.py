"""Chat generation with vLLM, executed in its own environment by ``synth.vllm_generator``.

Like ``vllm_teacher.py``, this imports nothing from ``code_embed_v3`` (vLLM pins older torch and
transformers) and only needs ``vllm`` and ``pyarrow``. Each job is a parquet of ready-made chat
messages; the reply text is written next to it as JSONL, one line per input row, in order.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
from vllm import LLM, SamplingParams


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--jobs", required=True)
    p.add_argument("--max-model-len", type=int, default=6144)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    a = p.parse_args()

    llm = LLM(
        model=a.model,
        max_model_len=a.max_model_len,
        gpu_memory_utilization=a.gpu_memory_utilization,
        seed=a.seed,
    )
    # No per-request seed: several rows can carry the same prompt (k variants of one snippet), and a
    # fixed per-request seed would make them identical. The engine seed keeps runs reproducible.
    params = SamplingParams(temperature=a.temperature, top_p=0.95, max_tokens=a.max_tokens)
    tokenizer = llm.get_tokenizer()
    # vLLM rejects the whole batch when one prompt exceeds the context; such rows get an empty reply,
    # which callers already reject as unparsable, so replies stay aligned with input rows.
    prompt_limit = a.max_model_len - a.max_tokens

    for job in json.loads(Path(a.jobs).read_text()):
        out = Path(job["output"])
        if out.is_file():
            continue
        rows = pq.read_table(job["input"]).to_pydict()
        conversations = [
            [{"role": "system", "content": s}, {"role": "user", "content": u}]
            for s, u in zip(rows["system"], rows["user"], strict=True)
        ]
        fits = [
            len(
                tokenizer.encode(
                    tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True),
                    add_special_tokens=False,
                )
            )
            <= prompt_limit
            for c in conversations
        ]
        outputs = iter(
            llm.chat([c for c, ok in zip(conversations, fits, strict=True) if ok], params, use_tqdm=False)
        )
        tmp = out.with_name(out.name + ".tmp")
        with tmp.open("w") as fh:
            for ok in fits:
                fh.write(json.dumps({"text": next(outputs).outputs[0].text if ok else ""}) + "\n")
        os.replace(tmp, out)
        print(
            f"generated {out.name}: {len(fits)} replies, "
            f"{fits.count(False)} prompts over {prompt_limit} tokens skipped",
            flush=True,
        )


if __name__ == "__main__":
    main()
