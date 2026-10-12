"""Defined and undefined study effects, exercised without loading a model."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/protocol_swap/source/runner.py"
BROAD = ROOT / "experiments/broad_study/run_study.py"


def protocol_module():
    spec = importlib.util.spec_from_file_location("effect_protocol", PROTOCOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.np = np
    return module


def execute_nodes(nodes, namespace, path):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


def protocol_summary(first, second):
    """Execute the actual run_model summary and display, omitting model extraction."""
    tree = ast.parse(PROTOCOL.read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "run_model")
    start = next(i for i, node in enumerate(function.body)
                 if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "sem_scrs")
    end = next(i for i, node in enumerate(function.body)
               if isinstance(node, ast.Delete) and ast.unparse(node.targets[0]) == "model")
    namespace = {
        "np": np, "stats": stats, "cfg": {"short": "synthetic"},
        "protocol_hedges_g": protocol_module().protocol_hedges_g,
        "results": {
            "control_A_semantic_as_chat": {str(i): {"scr": value} for i, value in enumerate(first)},
            "control_B_persona_as_prompt_final": {str(i): {"scr": value} for i, value in enumerate(second)},
        },
    }
    execute_nodes(function.body[start:end], namespace, PROTOCOL)
    return namespace["results"]["summary"]


def broad_displays(first, second):
    """Execute both existing Cohen calculation/display blocks with synthetic groups."""
    tree = ast.parse(BROAD.read_text())
    helper = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == "population_cohens_d")
    normalized = next(i for i, node in enumerate(tree.body)
                      if isinstance(node, ast.Assign)
                      and "cohens_d_norm" in ast.unparse(node.targets[0]))
    comparison = next(node for node in tree.body
                      if isinstance(node, ast.If) and ast.unparse(node.test) == "len(all_nccr) >= 5")
    namespace = {
        "np": np, "mannwhitneyu": stats.mannwhitneyu,
        "sem_nccrs_all": first, "per_nccrs_all": second,
        "sem_nmean": sum(first) / len(first), "per_nmean": sum(second) / len(second),
        "all_nccr": first + second, "all_acc": [None] * (len(first) + len(second)),
        "all_names_corr": [], "all_types_corr": ["semantic"] * len(first) + ["persona"] * len(second),
        "peak_info": {},
    }
    execute_nodes([helper, *tree.body[normalized:normalized + 2], comparison], namespace, BROAD)
    return namespace


@pytest.mark.parametrize("first_value,second_value,status", [
    (0.0, 0.0, "equal_constant_groups"),
    (0.1, 0.1, "equal_constant_groups"),
    (0.1, 0.2, "zero_variance_negative_difference"),
    (0.2, 0.1, "zero_variance_positive_difference"),
])
def test_constant_study_effects_are_null_and_unclassified(first_value, second_value, status, capsys):
    first, second = [first_value] * 6, [second_value] * 7
    summary = protocol_summary(first, second)
    assert summary["hedges_g"] is None
    assert summary["hedges_g_status"] == status
    broad = broad_displays(first, second)
    for name in ("cohens_d_norm", "cohens_d_c"):
        assert broad[name] is None
        assert broad[name + "_status"] == status
    effect_lines = [line for line in capsys.readouterr().out.splitlines()
                    if "Hedges' g" in line or "Cohen's d" in line]
    assert len(effect_lines) == 3
    assert all("N/A" in line and "undefined" in line for line in effect_lines)
    assert not any(label in line for line in effect_lines for label in ("Large", "Medium", "Small"))


@pytest.mark.parametrize("reverse", [False, True])
def test_distinct_finite_estimators_preserve_unequal_group_weighting(reverse, capsys):
    first, second = [1.0, 2.0, 4.0], [-3.0, 0.0, 1.0, 2.0, 3.0]
    if reverse:
        first, second = second, first
    n1, n2 = len(first), len(second)
    first_mean, second_mean = sum(first) / n1, sum(second) / n2
    ss1 = sum((value - first_mean) ** 2 for value in first)
    ss2 = sum((value - second_mean) ** 2 for value in second)
    expected_g = (first_mean - second_mean) / ((ss1 + ss2) / (n1 + n2 - 2)) ** .5
    expected_g *= 1 - 3 / (4 * (n1 + n2) - 9)
    expected_d = (first_mean - second_mean) / ((ss1 + ss2) / (n1 + n2)) ** .5
    summary = protocol_summary(first, second)
    broad = broad_displays(first, second)
    assert summary["hedges_g"] == pytest.approx(expected_g, rel=1e-14)
    assert summary["hedges_g_status"] == "defined"
    for name in ("cohens_d_norm", "cohens_d_c"):
        assert broad[name] == pytest.approx(expected_d, rel=1e-14)
        assert broad[name + "_status"] == "defined"
    assert expected_d != pytest.approx(expected_g)
    assert "N/A" not in "\n".join(line for line in capsys.readouterr().out.splitlines()
                                   if "Cohen's d" in line or "Hedges' g" in line)


@pytest.mark.parametrize("factor,defined", [(0.999, False), (1.001, True)])
def test_protocol_pooled_sd_boundary_is_retained(factor, defined):
    amplitude = factor * 1e-12 / np.sqrt(2)
    first, second = [-amplitude, amplitude], [2 * amplitude, 4 * amplitude]
    summary = protocol_summary(first, second)
    if defined:
        expected = -3 / np.sqrt(2) * (1 - 3 / 7)
        assert summary["hedges_g"] == pytest.approx(expected, rel=1e-14)
        assert summary["hedges_g_status"] == "defined"
    else:
        assert summary["hedges_g"] is None
        assert summary["hedges_g_status"] == "pooled_sd_at_or_below_1e-12"


def test_broad_cohens_d_does_not_inherit_protocol_cutoff():
    first, second = [0.0, 1e-13, 3e-13], [4e-13, 5e-13, 8e-13, 9e-13]
    broad = broad_displays(first, second)
    assert broad["cohens_d_norm_status"] == broad["cohens_d_c_status"] == "defined"
    assert broad["cohens_d_norm"] < 0
    assert protocol_summary(first, second)["hedges_g"] is None


def strict_json(text):
    def reject_constant(value):
        raise ValueError(f"Nonstandard JSON constant: {value}")
    return json.loads(text, parse_constant=reject_constant)


def test_protocol_writer_serializes_undefined_effect_as_null(tmp_path):
    module = protocol_module()
    output = tmp_path.resolve() / "results.json"
    module.save_results_atomically(output, "synthetic", {"summary": protocol_summary([.1] * 6, [.2] * 7)})
    summary = strict_json(output.read_text())["synthetic"]["summary"]
    assert summary["hedges_g"] is None
    assert summary["hedges_g_status"] == "zero_variance_negative_difference"


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("existing", [False, True])
def test_protocol_strict_writer_failure_preserves_existing_output(tmp_path, invalid, existing):
    module = protocol_module()
    output = tmp_path.resolve() / "results.json"
    if existing:
        module.save_results_atomically(output, "original", {"value": 0.1})
    before = output.read_bytes() if existing else None
    with pytest.raises(ValueError, match="JSON compliant"):
        module.save_results_atomically(output, "new", {"hedges_g": invalid})
    assert (output.read_bytes() if output.exists() else None) == before
    assert not list(tmp_path.glob("*.tmp"))
    if existing:
        assert strict_json(output.read_text()) == {"original": {"value": 0.1}}
        with pytest.raises(ValueError, match="already contains"):
            module.save_results_atomically(output, "original", {"value": 99})
        assert output.read_bytes() == before
