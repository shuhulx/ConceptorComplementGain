"""Judgment recovery uses saved synthetic scores, without a hosted evaluator."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.broad_study.checkpointing import atomic_write_json
from experiments.broad_study.evaluation import (
    require_complete_persona_corpus,
    score_persona_responses,
    study_judgments_are_complete,
)

KEY = ("sycophancy", "baseline", 0)


def corpus():
    return {KEY: {
        "question": "Question?", "response": "Answer.",
        "trait_score": None, "coherence": None,
    }}


class FakeJudge:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    def judge_score(self, prompt, question, answer, *, legacy_refusal_as_zero):
        self.calls.append((prompt, question, answer, legacy_refusal_as_zero))
        value = next(self.replies)
        if isinstance(value, BaseException):
            raise value
        return value


def evaluate(path, rows, judge, **overrides):
    options = {
        "trait_prompts": {"sycophancy": "Trait rubric"},
        "coherence_prompt": "Coherence rubric",
        "evaluator_specification": {"model": "local-test", "temperature": 0.0},
        "checkpoint_path": path,
        "run_specification_sha256": "same-run",
    }
    options.update(overrides)
    return score_persona_responses(rows, judge=judge, **options)


def test_interrupted_judgment_retains_success_before_next_request(tmp_path):
    path = tmp_path / "judgments.json"
    with pytest.raises(KeyboardInterrupt):
        evaluate(path, corpus(), FakeJudge([15.5, KeyboardInterrupt()]))
    saved = json.loads(path.read_text())
    assert saved["scores"][str(KEY)] == {"trait_score": 15.5, "coherence": None}

    resumed = corpus()
    judge = FakeJudge([80.0])
    assert evaluate(path, resumed, judge) == 1
    assert resumed[KEY]["trait_score"] == 15.5
    assert resumed[KEY]["coherence"] == 80.0
    assert judge.calls == [("Coherence rubric", "Question?", "Answer.", False)]
    assert study_judgments_are_complete(resumed, [KEY])
    assert evaluate(path, corpus(), FakeJudge([])) == 0


def test_failed_score_is_retried_without_rejudging_success(tmp_path):
    path = tmp_path / "judgments.json"
    rows = corpus()
    assert evaluate(path, rows, FakeJudge([None, 90.0])) == 2
    assert not study_judgments_are_complete(rows, [KEY])
    judge = FakeJudge([0.0])
    assert evaluate(path, rows, judge) == 1
    assert judge.calls == [("Trait rubric", "Question?", "Answer.", True)]
    assert study_judgments_are_complete(rows, [KEY])


@pytest.mark.parametrize("invalid", [None, -1, 101, True, "70", float("nan")])
def test_invalid_checkpoint_score_is_not_reused(tmp_path, invalid):
    path = tmp_path / "judgments.json"
    evaluate(path, corpus(), FakeJudge([10.0, 80.0]))
    saved = json.loads(path.read_text())
    saved["scores"][str(KEY)]["trait_score"] = invalid
    path.write_text(json.dumps(saved))
    rows = corpus()
    assert evaluate(path, rows, FakeJudge([20.0])) == 1
    assert rows[KEY]["trait_score"] == 20.0
    assert rows[KEY]["coherence"] == 80.0


@pytest.mark.parametrize("changed", ["question", "response", "rubric", "coherence", "evaluator", "run"])
def test_checkpoint_requires_identical_inputs_and_evaluator(tmp_path, changed):
    path = tmp_path / "judgments.json"
    evaluate(path, corpus(), FakeJudge([10.0, 80.0]))
    rows = corpus()
    overrides = {}
    if changed in {"question", "response"}:
        rows[KEY][changed] = "Changed"
    elif changed == "rubric":
        overrides["trait_prompts"] = {"sycophancy": "Changed rubric"}
    elif changed == "coherence":
        overrides["coherence_prompt"] = "Changed rubric"
    elif changed == "evaluator":
        overrides["evaluator_specification"] = {"model": "different"}
    else:
        overrides["run_specification_sha256"] = "different-run"
    judge = FakeJudge([])
    with pytest.raises(RuntimeError, match="do not match|belongs to run"):
        evaluate(path, rows, judge, **overrides)
    assert not judge.calls


def test_unbound_in_memory_scores_are_not_reused(tmp_path):
    rows = corpus()
    rows[KEY].update(trait_score=99, coherence=99)
    assert evaluate(tmp_path / "judgments.json", rows, FakeJudge([20, 70])) == 2
    assert rows[KEY]["trait_score"] == 20
    assert rows[KEY]["coherence"] == 70


def test_plot_failure_does_not_lose_result_exports(tmp_path, monkeypatch):
    source = Path(__file__).parents[1] / "experiments/broad_study/run_study.py"
    tree = ast.parse(source.read_text())
    start = next(
        index for index, node in enumerate(tree.body)
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "prefix"
    )
    end = next(
        index for index, node in enumerate(tree.body)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "plt"
        and node.value.func.attr == "subplots"
    )

    def failed_plot(*args, **kwargs):
        raise RuntimeError("Plotting failed")

    rows = corpus()
    rows[KEY].update(trait_score=15.5, coherence=80.0)
    namespace = {
        "MODEL_SHORT": "synthetic", "MODEL": "local-test", "NUM_LAYERS": 1,
        "HIDDEN_SIZE": 2, "APERTURE": .05, "BETA_CONCEPTOR": 2.0,
        "BETA_ADDITIVE": 2.3, "SEED": 42,
        "peak_info": {"sycophancy": {"layer": 0, "ccr": .1, "eff_rank": 1.0}},
        "norm_peak_info": {"sycophancy": {"ccr": .01}},
        "stat": 1.0, "pval": .5, "sem_nccr": .01, "per_nccr": .02,
        "semantic_results": {"task": {"baseline_mean": .25}},
        "semantic_best_configs": {"task": {"accuracy": .5}},
        "judge": SimpleNamespace(stats={"total": 2}), "all_scores": rows,
        "PERSONA_RESPONSE_KEYS": [KEY], "PERSONA_EVAL_QUESTIONS": {"sycophancy": ["Question?"]},
        "PERSONA_TRAITS": {"sycophancy": {}}, "SEMANTIC_FUNCTIONS": {"task": {}},
        "require_complete_persona_corpus": require_complete_persona_corpus,
        "atomic_write_json": atomic_write_json, "Path": Path,
        "plt": SimpleNamespace(subplots=failed_plot),
    }
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="Plotting failed"):
        exec(compile(ast.Module(body=tree.body[start:end + 1], type_ignores=[]), str(source), "exec"), namespace)
    results = json.loads((tmp_path / "synthetic_results_broad_study.json").read_text())
    assert results["semantic_results"] == namespace["semantic_results"]
    assert results["spectral"]["sycophancy"]["ccr"] == .1
    assert json.loads((tmp_path / "synthetic_responses_broad_study.json").read_text()) == {
        str(KEY): {"question": "Question?", "response": "Answer."}
    }
    assert json.loads((tmp_path / "synthetic_scores_broad_study.json").read_text()) == {
        str(KEY): {"trait_score": 15.5, "coherence": 80.0}
    }


def test_numerical_outputs_precede_plotting_and_complete_scores_gate_aggregation():
    source = Path(__file__).parents[1] / "experiments/broad_study/run_study.py"
    tree = ast.parse(source.read_text())
    writes = [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "atomic_write_json"
        and isinstance(node.args[0], ast.Call)
        and isinstance(node.args[0].func, ast.Name)
        and node.args[0].func.id == "Path"
    ]
    plots = [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name) and node.func.value.id == "plt"
        and node.func.attr in {"subplots", "figure", "show"}
    ]
    assert len(writes) == 3
    assert max(writes) < min(plots)
    judge_block = next(
        node for node in tree.body if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name) and node.test.id == "judge"
    )
    assert any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "require_complete_evaluation"
        for node in ast.walk(judge_block)
    )
    assert any(isinstance(node, ast.Raise) for node in ast.walk(judge_block))
