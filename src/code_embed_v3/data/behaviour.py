"""Behaviour-verified pairs: teach the embedder that code which works beats code that only looks right.

Embedders rank a correct implementation below a one-edit buggy copy most of the time
(ExecRetrieval, EMNLP 2026), because similarity tracks surface form. Here every label comes from
running tests:

- positives: correct solutions (other contestants' accepted code, LLM rewrites) that pass every test
- near-miss negatives: human wrong submissions and LLM single-edit mutants that pass some tests but
  fail others, and stay lexically close to the correct code

Rows: ``behaviour_nl2code`` (problem statement -> correct code, near-misses as negatives) and
``behaviour_code2code`` (correct code -> an equivalent rewrite, its own near-misses as negatives).
"""

from __future__ import annotations

import difflib
import json
import logging
import random
import re
import textwrap
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

from code_embed_v3.data import synth
from code_embed_v3.data.build import DEV_DIR, MANIFEST
from code_embed_v3.sandbox import Outcome, Test, evaluate_many

logger = logging.getLogger(__name__)

PROMPTS = "configs/synth/behaviour_prompts.yaml"
PYTHON3 = 3  # deepmind/code_contests `solutions.language` ClassLabel id
# `source` ClassLabel ids for AtCoder and Aizu: Project CodeNet, and so CoQuIR's CodeNet subsets, is
# built from exactly these judges, so they are excluded by provenance rather than by text matching
# (CodeNet ships HTML statements that near-duplicate search would not reliably catch).
EXCLUDED_SOURCES = {5, 6}
SOURCE = "deepmind/code_contests"


@dataclass
class Problem:
    pid: str
    description: str
    tests: list[Test]
    solutions: list[str]
    incorrect: list[str]
    timeout: float
    # filled by verification
    correct: list[str] = field(default_factory=list)
    near_misses: dict[str, list[str]] = field(default_factory=dict)  # correct code -> its near-misses
    rewrites: dict[str, list[str]] = field(default_factory=dict)  # correct code -> verified rewrites
    human_near_misses: list[str] = field(default_factory=list)


def _tests(row: dict[str, Any], limit: int) -> list[Test]:
    tests: list[Test] = []
    for key in ("public_tests", "private_tests", "generated_tests"):
        for inp, out in zip(row[key]["input"], row[key]["output"], strict=True):
            tests.append(Test(inp, out))
            if len(tests) == limit:
                return tests
    return tests


def load_problems(
    max_problems: int,
    split: str = "train",
    max_tests: int = 12,
    max_solutions: int = 4,
    max_incorrect: int = 8,
) -> list[Problem]:
    """Python 3 problems that read stdin and have at least 3 tests, streamed from CodeContests."""
    problems: list[Problem] = []
    for row in load_dataset(SOURCE, split=split, streaming=True):
        if row["input_file"] or row["output_file"] or row["source"] in EXCLUDED_SOURCES:
            continue
        tests = _tests(row, max_tests)
        sols = [
            s
            for lang, s in zip(row["solutions"]["language"], row["solutions"]["solution"], strict=True)
            if lang == PYTHON3
        ][:max_solutions]
        if len(tests) < 3 or not sols:
            continue
        bad = row["incorrect_solutions"]
        incorrect = [s for lang, s in zip(bad["language"], bad["solution"], strict=True) if lang == PYTHON3]
        limit = row.get("time_limit") or {}
        seconds = float(limit.get("seconds") or 2) + float(limit.get("nanos") or 0) / 1e9
        problems.append(
            Problem(
                pid=row["name"],
                description=row["description"],
                tests=tests,
                solutions=sols,
                incorrect=incorrect[:max_incorrect],
                # Python is slower than the C++ the limits were set for.
                timeout=min(10.0, max(2.0, 2 * seconds)),
            )
        )
        if len(problems) == max_problems:
            break
    return problems


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.splitlines(), b.splitlines(), autojunk=False).ratio()


def _adds_comment(original: str, variant: str) -> bool:
    """Mutants that explain themselves ("# bug: off by one") would teach the model to read comments."""
    added = (
        line for line in difflib.ndiff(original.splitlines(), variant.splitlines()) if line.startswith("+ ")
    )
    return any("#" in line for line in added)


def parse_code(reply: str, original: str) -> synth.Row | None:
    obj = synth._json_obj(reply)
    code = obj.get("code") if obj else None
    if not isinstance(code, str):
        return None
    code = code.strip("\n")
    if code.startswith("```"):
        code = code.split("\n", 1)[-1].rsplit("```", 1)[0]
    if not code.strip() or code.strip() == original.strip():
        return None
    return {"anchor": original, "positive": code}


PARSERS: dict[str, tuple[Any, Callable[[str, str], synth.Row | None]]] = {
    "mutate": ("code2code", parse_code),
    "rewrite": ("code2code", parse_code),
}


Job = tuple[str, list[Test], float]  # code, tests, timeout
Runner = Callable[[list[Job], str], list[Outcome]]


def sandbox_runner(workers: int) -> Runner:
    def run(jobs: list[Job], stop: str) -> list[Outcome]:
        return evaluate_many(jobs, workers=workers, stop=stop)

    return run


def verify_humans(problems: list[Problem], run: Runner) -> None:
    """Keep solutions that pass every test (the oracle must agree with the judge) and human wrong
    submissions that are near-misses."""
    # One batch across all problems, so every CPU stays busy.
    sols = [(p, s) for p in problems for s in p.solutions]
    for (p, s), o in zip(sols, run([(s, p.tests, p.timeout) for p, s in sols], "first_fail"), strict=True):
        if o.all_passed:
            p.correct.append(s)
    bad = [(p, s) for p in problems for s in p.incorrect]
    for (p, s), o in zip(bad, run([(s, p.tests, p.timeout) for p, s in bad], "mixed"), strict=True):
        if o.near_miss:
            p.human_near_misses.append(s)


def generate_variants(
    problems: list[Problem],
    generator: synth.Generator,
    work_dir: Path,
    mutants: int,
    rewrites: int,
    prompts_path: str | Path | None = None,
    after_round: Callable[[], None] | None = None,
) -> dict[str, list[synth.Row]]:
    prompts = synth.load_prompts(prompts_path or synth.resolve(PROMPTS), parsers=PARSERS)
    canon = [p.correct[0] for p in problems if p.correct]
    out: dict[str, list[synth.Row]] = {}
    for task, k in (("mutate", mutants), ("rewrite", rewrites)):
        # Justification: skip vLLM calls when k <= 0 so `ce3 behaviour-build --mutants 0 --rewrites 0`
        # can build a zero-LLM dataset from verified human contest submissions alone on a CPU pod.
        if k <= 0:
            out[task] = []
            continue
        rows, stats = synth.generate(
            task,
            [c for c in canon for _ in range(k)],
            generator,
            work_dir / task,
            round_shards=None,
            after_round=after_round,
            prompts=prompts,
        )
        logger.info("%s: %s", task, stats)
        out[task] = rows
    return out


def verify_variants(
    problems: list[Problem],
    variants: dict[str, list[synth.Row]],
    run: Runner,
    min_mutant_similarity: float = 0.85,
    max_rewrite_similarity: float = 0.8,
) -> dict[str, int]:
    """Mutants must look like the original and fail some tests; rewrites must look different and pass all."""
    by_code = {p.correct[0]: p for p in problems if p.correct}
    stats = {"mutants_kept": 0, "mutants_seen": 0, "rewrites_kept": 0, "rewrites_seen": 0}
    for task, stop, keep_key in (("mutate", "mixed", "mutants"), ("rewrite", "first_fail", "rewrites")):
        rows = [r for r in variants.get(task, []) if r["anchor"] in by_code]
        if task == "mutate":
            rows = [
                r
                for r in rows
                if similarity(r["anchor"], r["positive"]) >= min_mutant_similarity
                and not _adds_comment(r["anchor"], r["positive"])
            ]
        else:
            rows = [r for r in rows if similarity(r["anchor"], r["positive"]) <= max_rewrite_similarity]
        stats[f"{keep_key}_seen"] = len(variants.get(task, []))
        jobs = [(r["positive"], by_code[r["anchor"]].tests, by_code[r["anchor"]].timeout) for r in rows]
        outcomes = run(jobs, stop)
        for row, outcome in zip(rows, outcomes, strict=True):
            p = by_code[row["anchor"]]
            if task == "mutate" and outcome.near_miss:
                p.near_misses.setdefault(row["anchor"], []).append(row["positive"])
                stats["mutants_kept"] += 1
            elif task == "rewrite" and outcome.all_passed:
                p.rewrites.setdefault(row["anchor"], []).append(row["positive"])
                stats["rewrites_kept"] += 1
    return stats


def build_rows(
    problems: list[Problem], negatives: int, repeats: int, seed: int = 0
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """``repeats`` resamples per problem (different negative subsets and positives), so a few thousand
    problems carry enough weight next to a multi-million-row mix."""
    rng = random.Random(seed)
    nl2code: list[dict[str, str]] = []
    code2code: list[dict[str, str]] = []
    for p in problems:
        if not p.correct:
            continue
        anchor_code = p.correct[0]
        pool = list(dict.fromkeys(p.near_misses.get(anchor_code, []) + p.human_near_misses))
        if len(pool) < negatives:
            continue
        equivalents = list(dict.fromkeys(p.correct[1:] + p.rewrites.get(anchor_code, [])))
        for _ in range(repeats):
            negs = rng.sample(pool, negatives)
            nl2code.append(
                {
                    "anchor": p.description,
                    # Justification: always pair `anchor_code` (`p.correct[0]`) with its own single-edit
                    # mutants (`p.near_misses[anchor_code]`). Sampling `rng.choice(p.correct)` broke twin
                    # alignment whenever `p.correct` held multiple contestants' distinct solutions.
                    "positive": anchor_code,
                    **{f"negative_{i + 1}": n for i, n in enumerate(negs)},
                }
            )
            if equivalents:
                code2code.append(
                    {
                        "anchor": anchor_code,
                        "positive": rng.choice(equivalents),
                        **{f"negative_{i + 1}": n for i, n in enumerate(negs)},
                    }
                )
    return nl2code, code2code


SWE_SMITH_SOURCE = "SWE-bench/SWE-smith"
_DEF = re.compile(r"\s*(?:async\s+def|def|class)\s")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _trim_to_function(hunk: list[tuple[str, str]]) -> list[tuple[str, str]] | None:
    """Drop context lines that belong to the functions around the changed one; ``None`` if the
    changes span several definitions (several bugs, as in ``combine_*``).

    A hunk carries 3 lines of context either side, which often reach into the previous or next
    definition; eval sets score whole functions, so those neighbours are a distribution mismatch.
    Without a header inside the hunk the enclosing function's extent is unknown, so only lines
    past the next shallower definition are dropped.
    """
    changed = [i for i, (tag, _) in enumerate(hunk) if tag != " "]
    first, last = changed[0], changed[-1]
    body = min((_indent(t) for tag, t in hunk if tag != " " and t.strip()), default=0)
    start, ref = 0, None
    for i in range(first, -1, -1):
        tag, text = hunk[i]
        if _DEF.match(text) and (_indent(text) < body or tag != " "):
            start, ref = i, _indent(text)
            while start > 0 and hunk[start - 1][1].lstrip().startswith("@"):
                start -= 1
            break
    for _, text in hunk[first + 1 : last + 1]:
        if _DEF.match(text) and _indent(text) <= (body - 1 if ref is None else ref):
            return None
    end = len(hunk)
    for i in range(last + 1, len(hunk)):
        text = hunk[i][1]
        if not text.strip():
            continue
        is_def = bool(_DEF.match(text)) or text.lstrip().startswith("@")
        # With a known header, any line at or above its indent has left the function; without one,
        # `else:`/`except:` sit shallower than the changed statement, so only definitions end it.
        if (ref is not None and _indent(text) <= ref) or (ref is None and is_def and _indent(text) < body):
            end = i
            break
    return hunk[start:end]


def extract_patch_twins(
    patch: str, min_lines: int = 4, min_similarity: float = 0.5
) -> tuple[str, str] | None:
    """Reconstruct ``(correct_code, buggy_twin)`` from a SWE-smith bug patch.

    SWE-smith's ``patch`` *introduces* the bug: ``-`` lines are the original (correct) code and ``+``
    lines the injected bug, verified by the instance's FAIL_TO_PASS tests. Only single-file,
    single-hunk patches are kept: ``combine_*`` instances inject several bugs while the problem
    statement describes one, and joining hunks would yield fragments rather than one twin.
    Context from neighbouring functions is trimmed (see ``_trim_to_function``).
    """
    hunk: list[tuple[str, str]] = []
    files = hunks = 0
    for line in patch.splitlines():
        if line.startswith("diff --git"):
            files += 1
        elif line.startswith("@@ "):
            hunks += 1
        elif line.startswith(("index ", "--- ", "+++ ")):
            continue
        elif line[:1] in ("+", "-", " "):
            hunk.append((line[0], line[1:]))
    tags = {tag for tag, _ in hunk}
    if files > 1 or hunks != 1 or not {"+", "-"} <= tags:
        return None
    trimmed = _trim_to_function(hunk)
    if trimmed is None:
        return None
    correct = [text for tag, text in trimmed if tag != "+"]
    buggy = [text for tag, text in trimmed if tag != "-"]
    if len(correct) < min_lines or len(buggy) < min_lines:
        return None
    correct_code = textwrap.dedent("\n".join(correct)).strip("\n")
    buggy_code = textwrap.dedent("\n".join(buggy)).strip("\n")
    if not correct_code or not buggy_code or correct_code == buggy_code:
        return None
    # A comment on only one side, or a dropped docstring block (low similarity), is a lexical tell
    # the model could learn instead of the behavioural difference.
    if _adds_comment(correct_code, buggy_code) or _adds_comment(buggy_code, correct_code):
        return None
    if similarity(correct_code, buggy_code) < min_similarity:
        return None
    return correct_code, buggy_code


def load_swe_smith_twins(
    max_rows: int = 5000, negatives: int = 4, split: str = "train", seed: int = 0
) -> list[dict[str, str]]:
    """Build ``(problem_statement, correct_code, buggy_twin, ...)`` rows from SWE-smith."""
    if max_rows <= 0:
        return []
    rng = random.Random(seed)
    raw_pairs: list[tuple[str, str, str]] = []
    for row in load_dataset(SWE_SMITH_SOURCE, split=split, streaming=True):
        stmt = (row.get("problem_statement") or "").strip()
        patch = row.get("patch") or ""
        if not stmt or not patch:
            continue
        twins = extract_patch_twins(patch)
        if twins is None:
            continue
        raw_pairs.append((stmt, twins[0], twins[1]))
        if len(raw_pairs) >= max_rows:
            break
    if len(raw_pairs) < negatives:
        return []
    buggy_pool = [buggy for _, _, buggy in raw_pairs]
    rows: list[dict[str, str]] = []
    for idx, (stmt, correct, buggy) in enumerate(raw_pairs):
        # negative_1 is the instance's own buggy twin; the rest are other instances' buggy code.
        distractors: list[str] = [buggy]
        while len(distractors) < negatives:
            cand = rng.choice(buggy_pool)
            if cand != correct and cand not in distractors:
                distractors.append(cand)
            elif len(set(buggy_pool)) <= negatives:
                distractors.append(buggy_pool[(idx + len(distractors)) % len(buggy_pool)])
        rows.append(
            {
                "anchor": stmt,
                "positive": correct,
                **{f"negative_{i + 1}": n for i, n in enumerate(distractors[:negatives])},
            }
        )
    return rows


def save(
    out_dir: Path,
    nl2code: list[dict[str, str]],
    code2code: list[dict[str, str]],
    swe_twins: list[dict[str, str]] | None = None,
    dev_rows: int = 200,
) -> None:
    """Write behaviour sources plus a dev slice, in the layout ``ce3 build-data`` produces (audit pending)."""
    manifest: dict[str, Any] = {"sources": {}}
    specs: list[tuple[str, str, str, str, list[dict[str, str]]]] = [
        ("behaviour_nl2code", "nl2code", "cc-by-4.0", SOURCE, nl2code),
        ("behaviour_code2code", "code2code", "cc-by-4.0", SOURCE, code2code),
    ]
    if swe_twins:
        specs.append(("behaviour_swe_twins", "nl2code", "mit", SWE_SMITH_SOURCE, swe_twins))
    for name, family, lic, hf_id, rows in specs:
        if not rows:
            continue
        # Dev rows by anchor so a held-out problem never leaks into training through a repeat.
        anchors = list(dict.fromkeys(r["anchor"] for r in rows))
        held = set(anchors[: min(dev_rows, len(anchors) // 10)])
        train = [r for r in rows if r["anchor"] not in held]
        dev = list(
            {
                r["anchor"]: {"anchor": r["anchor"], "positive": r["positive"]}
                for r in rows
                if r["anchor"] in held
            }.values()
        )
        Dataset.from_list(train).save_to_disk(str(out_dir / name))
        if dev:
            Dataset.from_list(dev).save_to_disk(str(out_dir / DEV_DIR / name))
        manifest["sources"][name] = {
            "name": name,
            "family": family,
            "license": lic,
            "hf_id": hf_id,
            "loader": "behaviour",
            "rows": len(train),
            "dev_rows": len(dev),
            "audited": False,
            "overlaps_eval": [],
        }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=2))


def dump_problems(problems: list[Problem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for p in problems:
            fh.write(json.dumps({**asdict(p), "tests": [asdict(t) for t in p.tests]}) + "\n")


def read_problems(path: Path) -> Iterator[Problem]:
    for line in path.read_text().splitlines():
        raw = json.loads(line)
        raw["tests"] = [Test(**t) for t in raw["tests"]]
        yield Problem(**raw)
