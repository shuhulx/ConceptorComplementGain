from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_broad_study_fractional_accuracy_label():
    source = (ROOT / "experiments/broad_study/run_study.py").read_text()
    tree = ast.parse(source)
    labels = [
        ast.literal_eval(node.args[0])
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "ax_j"
        and node.func.attr == "set_ylabel"
    ]
    assert labels == ["Best conceptor prefix-match score (fraction)"]
    assert 'acc_plot.append(semantic_best_configs[name]["accuracy"])' in source
