"""Offline unit tests for sandbox execution, behaviour pair construction, and behaviour evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mteb
import numpy as np
import pytest
import torch

from code_embed_v3.data import behaviour
from code_embed_v3.data.behaviour import (
    Problem,
    _adds_comment,
    build_rows,
    extract_patch_twins,
    generate_variants,
    load_swe_smith_twins,
    parse_code,
    save,
    similarity,
    verify_humans,
    verify_variants,
)
from code_embed_v3.data.build import DEV_DIR, MANIFEST
from code_embed_v3.eval import behaviour_eval, run_mteb
from code_embed_v3.modeling import load_model
from code_embed_v3.sandbox import (
    ALLOW_ENV,
    Outcome,
    SandboxRefusedError,
    _tokens_match,
    evaluate,
    evaluate_many,
    require_isolation,
)
from code_embed_v3.sandbox import Test as SandboxTest
from code_embed_v3.train import DistillWithContrastive


def test_sandbox_refuses_without_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ALLOW_ENV, raising=False)
    monkeypatch.delenv("RUNPOD_POD_ID", raising=False)
    with pytest.raises(SandboxRefusedError, match=ALLOW_ENV):
        require_isolation()


def test_tokens_match_whitespace_and_float_tolerance() -> None:
    assert _tokens_match("1  2\n3\n", "1 2 3")
    assert _tokens_match("3.14159265", "3.14159260", tol=1e-6)
    assert not _tokens_match("3.14", "3.15", tol=1e-6)
    assert not _tokens_match("1 2", "1 2 3")
    assert not _tokens_match("abc", "def")


def test_sandbox_evaluate_pass_fail_and_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ALLOW_ENV, "1")
    tests = [SandboxTest("2\n", "4\n"), SandboxTest("5\n", "10\n"), SandboxTest("-3\n", "-6\n")]

    good = evaluate("print(int(input()) * 2)", tests)
    assert good == Outcome(passed=3, failed=0) and good.all_passed and not good.near_miss

    near = evaluate("x = int(input())\nprint(x * 2 if x > 0 else 0)", tests, stop="mixed")
    assert near.near_miss

    first_fail = evaluate("print(0)", tests, stop="first_fail")
    assert first_fail == Outcome(passed=0, failed=1)

    batch = evaluate_many(
        [("print(int(input()) * 2)", tests, 2.0), ("raise RuntimeError()", tests, 2.0)],
        workers=2,
        stop="first_fail",
    )
    assert [b.all_passed for b in batch] == [True, False]


def test_parse_code_and_comment_filter() -> None:
    orig = "def f(x):\n    return x + 1\n"
    assert parse_code('{"code": "def f(x):\\n    return x - 1"}', orig) == {
        "anchor": orig,
        "positive": "def f(x):\n    return x - 1",
    }
    assert parse_code('{"code": "def f(x):\\n    return x + 1"}', orig) is None
    assert _adds_comment(orig, "def f(x):\n    return x - 1  # bug\n")
    assert not _adds_comment(orig, "def f(x):\n    return x - 1\n")
    assert similarity(orig, "def f(x):\n    return x - 1\n") >= 0.5


def test_verify_and_build_rows_preserve_twin_alignment(tmp_path: Path) -> None:
    c0 = "a = int(input())\nb = int(input())\nres = a + b\nprint(res)\n"
    c1 = "import sys\nvals = [int(x) for x in sys.stdin.read().split()]\nprint(sum(vals))\n"
    m1 = "a = int(input())\nb = int(input())\nres = a - b\nprint(res)\n"
    m2 = "a = int(input())\nb = int(input())\nres = a + b + 1\nprint(res)\n"
    rw = "x, y = int(input()), int(input())\nprint(x + y)\n"

    prob = Problem(
        pid="p1",
        description="add two integers",
        tests=[SandboxTest("1\n2\n", "3\n"), SandboxTest("0\n0\n", "0\n"), SandboxTest("5\n-2\n", "3\n")],
        solutions=[c0, c1],
        incorrect=[m1],
        timeout=2.0,
    )

    def fake_run(jobs: list[tuple[str, list[SandboxTest], float]], stop: str) -> list[Outcome]:
        out: list[Outcome] = []
        for code, _, _ in jobs:
            if code in (c0, c1, rw):
                out.append(Outcome(passed=3, failed=0))
            else:
                out.append(Outcome(passed=1, failed=1))
        return out

    verify_humans([prob], fake_run)
    assert prob.correct == [c0, c1]
    assert prob.human_near_misses == [m1]

    variants = {
        "mutate": [{"anchor": c0, "positive": m2}],
        "rewrite": [{"anchor": c0, "positive": rw}],
    }
    stats = verify_variants([prob], variants, fake_run, min_mutant_similarity=0.7, max_rewrite_similarity=0.9)
    assert stats["mutants_kept"] == 1 and stats["rewrites_kept"] == 1

    nl2code, code2code = build_rows([prob], negatives=2, repeats=3, seed=42)
    assert len(nl2code) == 3 and len(code2code) == 3
    # Every nl2code row must pair the anchor_code (c0) with its own single-edit mutants.
    assert all(r["positive"] == c0 for r in nl2code)
    assert all({r["negative_1"], r["negative_2"]} == {m1, m2} for r in nl2code)
    assert all(r["anchor"] == c0 and r["positive"] in {c1, rw} for r in code2code)

    ten_nl = [{"anchor": f"q{i}", "positive": c0, "negative_1": m1} for i in range(10)]
    swe = [{"anchor": f"issue {i}", "positive": c0, "negative_1": m1} for i in range(10)]
    save(tmp_path / "data", ten_nl, code2code, swe_twins=swe, dev_rows=1)
    manifest = json.loads((tmp_path / "data" / MANIFEST).read_text())["sources"]
    assert manifest["behaviour_nl2code"]["rows"] == 9
    assert manifest["behaviour_nl2code"]["dev_rows"] == 1
    assert manifest["behaviour_swe_twins"]["rows"] == 9
    assert manifest["behaviour_swe_twins"]["hf_id"] == "SWE-bench/SWE-smith"
    assert (tmp_path / "data" / DEV_DIR / "behaviour_nl2code").exists()
    assert (tmp_path / "data" / DEV_DIR / "behaviour_swe_twins").exists()


def test_zero_llm_generate_variants_and_swe_smith_twins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def boom_generator(prompts: list[tuple[str, str]]) -> list[str]:
        raise AssertionError("generator must not be called when mutants=0 and rewrites=0")

    out = generate_variants([], boom_generator, tmp_path, mutants=0, rewrites=0)
    assert out == {"mutate": [], "rewrite": []}

    # SWE-smith patches inject the bug: `-` is the original correct code, `+` the buggy version.
    sample_patch = (
        "diff --git a/mod.py b/mod.py\n"
        "--- a/mod.py\n"
        "+++ b/mod.py\n"
        "@@ -1,5 +1,5 @@\n"
        " def clamp(x, lo, hi):\n"
        "-    if x < lo:\n"
        "+    if x <= lo:\n"
        "         return lo\n"
        "     return min(x, hi)\n"
    )
    twins = extract_patch_twins(sample_patch)
    assert twins is not None
    correct, buggy = twins
    assert "if x < lo:" in correct and "if x <= lo:" not in correct
    assert "if x <= lo:" in buggy

    second_hunk = "@@ -10,4 +10,4 @@\n def f(a):\n-    return a + 1\n+    return a\n     pass\n     pass\n"
    assert extract_patch_twins(sample_patch + second_hunk) is None
    assert extract_patch_twins(sample_patch + sample_patch.replace("mod.py", "other.py")) is None
    commented = sample_patch.replace("+    if x <= lo:\n", "+    if x <= lo:  # inclusive\n")
    assert extract_patch_twins(commented) is None
    doc_dropped = sample_patch.replace(
        " def clamp(x, lo, hi):\n",
        " def clamp(x, lo, hi):\n" + "".join(f"-    doc line {i}\n" for i in range(12)),
    )
    assert extract_patch_twins(doc_dropped) is None

    fake_rows = [
        {"problem_statement": f"fix boundary {i}", "patch": sample_patch.replace("lo", f"lo_{i}")}
        for i in range(4)
    ]
    monkeypatch.setattr(behaviour, "load_dataset", lambda *a, **kw: iter(fake_rows))
    rows = load_swe_smith_twins(max_rows=3, negatives=2, seed=0)
    assert len(rows) == 3
    assert "if x < lo_0:" in rows[0]["positive"]
    assert "if x <= lo_0:" in rows[0]["negative_1"]
    assert "negative_2" in rows[0]


def test_exec_retrieval_and_coquir_scoring() -> None:
    sims = np.array([[0.9, 0.8, 0.2], [0.3, 0.7, 0.1]])
    queries = [
        {"query": "q1", "correct_corpus_ids": ["d0"], "paired_distractor_corpus_ids": ["d1"]},
        {"query": "q2", "correct_corpus_ids": ["d0"], "paired_distractor_corpus_ids": ["d1"]},
    ]
    exec_scores = behaviour_eval.exec_retrieval_scores(sims, ["d0", "d1", "d2"], queries)
    assert exec_scores == {"exec@1": 0.5, "exec@10": 1.0, "beats_paired_distractors": 0.5}

    qrels = [{"qid": "q1", "pos-docids": ["d0"], "neg-docids": ["d1", "d2"]}]
    cq_scores = behaviour_eval.coquir_scores(sims[:1], ["q1"], ["d0", "d1", "d2"], qrels)
    assert cq_scores["pairwise"] == 1.0
    assert cq_scores["ndcg@10"] == pytest.approx(1.0)


def test_behaviour_eval_selects_sql_prompt_for_sqlr2(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    class FakeModel:
        def __init__(self) -> None:
            self.prompts = {"nl2code": "p_nl", "text2sql": "p_sql"}

        def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
            calls.append(kwargs)
            return np.ones((len(texts), 4), dtype=np.float32)

    exec_q = [{"query": "q", "correct_corpus_ids": ["c1"], "paired_distractor_corpus_ids": []}]
    monkeypatch.setattr(behaviour_eval, "load_exec_retrieval", lambda: (exec_q, {"c1": "x"}))
    coquir_qrels = [{"qid": "q1", "pos-docids": ["d1"], "neg-docids": []}]
    monkeypatch.setattr(
        behaviour_eval,
        "load_coquir",
        lambda subset: ({"q1": "select 1"}, {"d1": "SELECT 1"}, coquir_qrels),
    )
    behaviour_eval.evaluate(FakeModel(), subsets=("CodeNet-B", "SQLR2"))
    query_prompts = [c.get("prompt_name") for c in calls if "prompt_name" in c]
    assert query_prompts == ["nl2code", "nl2code", "text2sql"]


def test_pilot_gate_decision() -> None:
    nan = float("nan")
    base = {
        "ExecRetrieval": {"exec@1": 0.25},
        "CoQuIR-Defects4J": {"pairwise": 0.60},
        "CoQuIR-CVEFixes": {"pairwise": 0.70},
        "CoQuIR-SQLR2": {"pairwise": nan},
    }
    cand = {
        "ExecRetrieval": {"exec@1": 0.38},
        "CoQuIR-Defects4J": {"pairwise": 0.61},
        "CoQuIR-CVEFixes": {"pairwise": 0.68},
        "CoQuIR-SQLR2": {"pairwise": 0.10},
    }
    ok, report = behaviour_eval.pilot_gate(base, cand, mteb_baseline=0.700, mteb_candidate=0.697)
    assert ok is True
    assert report["exec@1_gain"] == pytest.approx(0.13)
    assert report["exec@1_rel_gain"] == pytest.approx(0.52)
    # SQLR2 is NaN in the baseline, so only the two shared subsets count.
    assert report["coquir_pairwise_drop"] == pytest.approx(0.005)
    assert report["CoQuIR-CVEFixes_pairwise_drop"] == pytest.approx(0.02)
    assert "CoQuIR-SQLR2_pairwise_drop" not in report

    worse = {**cand, "CoQuIR-CVEFixes": {"pairwise": 0.55}}
    ok, report = behaviour_eval.pilot_gate(base, worse, mteb_baseline=0.700, mteb_candidate=0.697)
    assert ok is False and report["coquir_pairwise_drop"] == pytest.approx(0.07)

    # +8% relative (0.25 -> 0.27) misses the +10% bar even though everything else holds.
    small = {**cand, "ExecRetrieval": {"exec@1": 0.27}}
    ok, report = behaviour_eval.pilot_gate(base, small, mteb_baseline=0.700, mteb_candidate=0.697)
    assert ok is False and report["exec@1_rel_gain"] == pytest.approx(0.08)

    with pytest.raises(ValueError, match="no CoQuIR subset"):
        behaviour_eval.pilot_gate(
            {"ExecRetrieval": {"exec@1": 0.25}}, {"ExecRetrieval": {"exec@1": 0.38}}, 0.700, 0.697
        )


def test_no_collapse_flags_near_random_model() -> None:
    nan = float("nan")
    base = {
        "ExecRetrieval": {"exec@10": 0.967},
        "CoQuIR-Defects4J": {"ndcg@10": 0.814},
        "CoQuIR-SQLR2": {"ndcg@10": nan},
    }
    healthy = {
        "ExecRetrieval": {"exec@10": 0.90},
        "CoQuIR-Defects4J": {"ndcg@10": 0.70},
        "CoQuIR-SQLR2": {"ndcg@10": 0.0},
    }
    ok, kept = behaviour_eval.no_collapse(base, healthy)
    # SQLR2 is NaN in the baseline, so it is not compared.
    assert ok is True and set(kept) == {"ExecRetrieval exec@10", "CoQuIR-Defects4J ndcg@10"}

    # Numbers from the first dry-run's collapsed fine-tune.
    collapsed = {"ExecRetrieval": {"exec@10": 0.003}, "CoQuIR-Defects4J": {"ndcg@10": 0.005}}
    ok, kept = behaviour_eval.no_collapse(base, collapsed)
    assert ok is False and kept["ExecRetrieval exec@10"] == pytest.approx(0.003 / 0.967)


def test_run_mteb_does_not_inject_prompts_into_unprompted_baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tiny_modernbert: Path
) -> None:
    captured: dict[str, Any] = {}
    base_model = load_model(tiny_modernbert, device="cpu")
    base_model.prompts = {}

    monkeypatch.setattr(run_mteb, "load_model", lambda path, truncate_dim=None: base_model)
    monkeypatch.setattr(run_mteb, "zero_shot_report", lambda meta, benchmarks=None: {"MTEB(Code, v1)": 100})
    monkeypatch.setattr(
        mteb,
        "get_task",
        lambda name: SimpleNamespace(metadata=SimpleNamespace(name=name)),
    )
    monkeypatch.setattr(
        mteb,
        "evaluate",
        lambda model, tasks, **kwargs: captured.update(
            use_instructions=model.mteb_model_meta.use_instructions,
            model_prompts=model.model_prompts,
        ),
    )
    run_mteb.run(
        "Alibaba-NLP/gte-modernbert-base",
        "Alibaba-NLP/gte-modernbert-base",
        output_folder=tmp_path / "eval",
        tasks=["AppsRetrieval"],
    )
    assert captured["use_instructions"] is False
    assert not captured["model_prompts"]


def test_distill_with_contrastive_unshrinks_kl_temperature(tiny_modernbert: Path) -> None:
    model = load_model(tiny_modernbert, device="cpu")
    tau = 0.02
    loss_fn = DistillWithContrastive(
        model,
        scale=1.0 / tau,
        contrastive_weight=0.0,
        student_temperature=tau,
        teacher_temperature=1.0,
    )
    emb = [
        torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True),
        torch.tensor([[0.8, 0.2], [0.2, 0.8]], requires_grad=True),
        torch.tensor([[0.2, 0.8], [0.8, 0.2]], requires_grad=True),
    ]
    labels = torch.tensor([[0.9, 0.1], [0.9, 0.1]])
    raw_kl = loss_fn.kl.compute_loss_from_embeddings(emb, labels)
    # Without dividing by tau**2, raw_kl is scaled down by 0.0004.
    unshrunk = raw_kl / (loss_fn.kl.student_temperature**2)
    assert torch.isclose(unshrunk, raw_kl * 2500.0)


def test_coquir_drops_rows_without_text(monkeypatch: pytest.MonkeyPatch) -> None:
    files = {
        "S/query.jsonl": [{"query-id": "q1", "text": "find user"}],
        "S/corpus.jsonl": [{"doc-id": "d1", "text": "select 1"}, {"doc-id": "d2", "text": float("nan")}],
        "S/qrels.jsonl": [{"qid": "q1", "pos-docids": ["d1"], "neg-docids": ["d2"]}],
    }
    monkeypatch.setattr(behaviour_eval, "_jsonl", lambda repo, filename: files[filename])
    queries, corpus, qrels = behaviour_eval.load_coquir("S")
    assert corpus == {"d1": "select 1"}
    scores = behaviour_eval.coquir_scores(np.array([[0.9]]), list(queries), list(corpus), qrels)
    assert np.isnan(scores["pairwise"]) and scores["ndcg@10"] == 1.0


def test_patch_twins_drop_neighbouring_functions() -> None:
    patch = (
        "diff --git a/mod.py b/mod.py\n"
        "--- a/mod.py\n"
        "+++ b/mod.py\n"
        "@@ -1,12 +1,12 @@\n"
        "         return self.a\n"
        " \n"
        "     @property\n"
        "     def clamp(self, x):\n"
        "         if x < 0:\n"
        "-            return 0\n"
        "+            return 1\n"
        "         else:\n"
        "             return x\n"
        " \n"
        "     def other(self):\n"
        "         return 2\n"
    )
    twins = extract_patch_twins(patch)
    assert twins is not None
    correct, buggy = twins
    assert correct.splitlines()[0] == "@property"
    assert "return self.a" not in correct and "def other" not in correct
    assert "else:" in buggy and "return 1" in buggy

    two_bugs = patch.replace("         return 2\n", "-        return 2\n+        return 3\n").replace(
        "@@ -1,12 +1,12 @@", "@@ -1,13 +1,13 @@"
    )
    assert extract_patch_twins(two_bugs) is None


def test_behaviour_combine_drops_replay_rows_sharing_a_dev_anchor(tmp_path: Path) -> None:
    from datasets import Dataset, load_from_disk

    from code_embed_v3.cli import _cmd_behaviour_combine

    swe = [
        {"anchor": f"issue {i}", "positive": "def f(): return 1", "negative_1": "def f(): return 2"}
        for i in range(10)
    ]
    save(tmp_path / "behaviour", [], [], swe_twins=swe, dev_rows=1)
    held = load_from_disk(str(tmp_path / "behaviour" / DEV_DIR / "behaviour_swe_twins"))["anchor"]
    assert held == ["issue 0"]

    replay = tmp_path / "replay"
    Dataset.from_dict(
        {"anchor": ["issue 0", "issue 5", "other"], "positive": ["p0", "p5", "po"]}
    ).save_to_disk(str(replay / "swe_smith"))
    (replay / MANIFEST).write_text(
        json.dumps({"sources": {"swe_smith": {"family": "code_edit", "rows": 3, "audited": True}}})
    )

    out = tmp_path / "combined"
    _cmd_behaviour_combine(
        SimpleNamespace(
            data_dirs=[str(replay), str(tmp_path / "behaviour")], output_dir=str(out), behaviour_share=None
        )
    )
    sources = json.loads((out / MANIFEST).read_text())["sources"]
    assert sources["swe_smith"]["rows"] == 2 and sources["swe_smith"]["dev_anchor_rows_dropped"] == 1
    assert load_from_disk(sources["swe_smith"]["path"])["anchor"] == ["issue 5", "other"]
    # Sources without a shared dev anchor stay referenced in place, and inputs are left untouched.
    assert sources["behaviour_swe_twins"]["path"] == str(
        (tmp_path / "behaviour" / "behaviour_swe_twins").resolve()
    )
    assert not list((replay / "swe_smith").glob("cache-*"))
    assert (out / DEV_DIR / "behaviour_swe_twins").exists()


@pytest.mark.parametrize(("replay_rows", "share", "expect"), [(90, 0.25, (90, 30)), (10, 0.5, (8, 8))])
def test_behaviour_combine_holds_the_behaviour_share(
    tmp_path: Path, replay_rows: int, share: float, expect: tuple[int, int]
) -> None:
    """Whichever side is over the target is subsampled; the other is kept whole."""
    from datasets import Dataset, load_from_disk

    from code_embed_v3.cli import _cmd_behaviour_combine

    for name, n in (("replay", replay_rows), ("behaviour", 8 if replay_rows == 10 else 40)):
        Dataset.from_dict({"anchor": [f"{name} {i}" for i in range(n)], "positive": ["p"] * n}).save_to_disk(
            str(tmp_path / name / name)
        )
        (tmp_path / name / MANIFEST).write_text(
            json.dumps({"sources": {name: {"family": "nl2code", "rows": n, "audited": True}}})
        )
    out = tmp_path / "combined"
    _cmd_behaviour_combine(
        SimpleNamespace(
            data_dirs=[str(tmp_path / "replay"), str(tmp_path / "behaviour")],
            output_dir=str(out),
            behaviour_share=share,
        )
    )
    sources = json.loads((out / MANIFEST).read_text())["sources"]
    got = (sources["replay"]["rows"], sources["behaviour"]["rows"])
    assert got == expect
    assert len(set(load_from_disk(sources["behaviour"]["path"])["anchor"])) == got[1]


def test_eval_scores_every_model_at_one_length(monkeypatch: pytest.MonkeyPatch) -> None:
    """A checkpoint saved at its training length must not be scored on cut text against a full-length base."""
    queries = [{"query": "q", "correct_corpus_ids": ["c1"], "paired_distractor_corpus_ids": ["c2"]}]
    monkeypatch.setattr(behaviour_eval, "load_exec_retrieval", lambda: (queries, {"c1": "a", "c2": "b"}))
    monkeypatch.setattr(
        behaviour_eval,
        "load_coquir",
        lambda _: (
            {"q": "q"},
            {"g": "good", "b": "bad"},
            [{"qid": "q", "pos-docids": ["g"], "neg-docids": ["b"]}],
        ),
    )

    class Model:
        def __init__(self) -> None:
            self.prompts: dict[str, str] = {}
            self.max_seq_length = 256
            self.seen: list[int] = []

        def encode(self, texts: list[str], **_: Any) -> np.ndarray:
            self.seen.append(self.max_seq_length)
            return np.eye(len(texts), 4)

    model = Model()
    out = behaviour_eval.evaluate(model, subsets=("Defects4J",))
    assert out["max_seq_length"] == behaviour_eval.EVAL_MAX_SEQ_LENGTH == 8192
    assert set(model.seen) == {8192}
    assert (
        behaviour_eval.evaluate(Model(), subsets=("Defects4J",), max_seq_length=None)["max_seq_length"] == 256
    )
