"""Public import boundaries, extraction validation and CPU entry points."""

import ast
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from extract_activations import last_token_resid_pre

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = [
    "conceptor_tools.py",
    "activation_steer.py",
    "extract_activations.py",
    "eval/compare_activations.py",
    "eval/compare_generations.py",
]


def test_public_modules_do_not_depend_on_examples_or_experiments():
    for relative in PUBLIC:
        tree = ast.parse((ROOT / relative).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not any(name.split(".")[0] in {"examples", "experiments"} for name in names)
        assert "sys.path" not in (ROOT / relative).read_text()


@pytest.mark.parametrize(
    "module",
    [
        "examples.compare_activations",
        "examples.synthetic_complement_gain",
        "examples.extract_activations",
        "experiments.broad_study.run_study",
    ],
)
def test_module_help_runs_without_model_loading(module):
    bootstrap = (
        "import runpy,sys; sys.modules['transformer_lens']=None; "
        f"sys.argv=[{module!r},'--help']; runpy.run_module({module!r},run_name='__main__')"
    )
    result = subprocess.run(
        [sys.executable, "-c", bootstrap], cwd=ROOT, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


@pytest.mark.parametrize("prompts", ["hello", b"hello", [], [""], [1], None])
def test_extraction_rejects_malformed_prompts_before_forward(prompts):
    class NoForward:
        def run_with_cache(self, *args, **kwargs):
            raise AssertionError("invalid prompts reached the model")

    with pytest.raises(ValueError, match="prompt"):
        last_token_resid_pre(NoForward(), prompts, (0,))


def test_extraction_rejects_training_model_and_out_of_range_layer():
    model = torch.nn.Linear(3, 3)
    model.cfg = SimpleNamespace(n_layers=2)
    with pytest.raises(ValueError, match="eval"):
        last_token_resid_pre(model, ["one"], (0,))
    model.eval()
    with pytest.raises(ValueError, match="indices"):
        last_token_resid_pre(model, ["one"], (2,))
