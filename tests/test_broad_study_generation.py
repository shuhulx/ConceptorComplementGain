"""Completion extraction must use token boundaries, not prompt character counts."""

import ast
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch


def load_steerable_model():
    source = Path(__file__).resolve().parents[1] / "experiments/broad_study/run_study.py"
    tree = ast.parse(source.read_text())
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SteerableModel")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["SteerableModel"]


@pytest.mark.parametrize("prompt,prefix,decoded_prefix", [
    ("hello", [0, 1], "[BOS]hello"),
    ("HELLO       WORLD", [1, 2], "hello world"),
    ("HELLO   WORLD", [0, 1, 2], "[BOS]hello world"),
])
def test_completion_boundary_with_bos_and_normalization(prompt, prefix, decoded_prefix):
    class Model:
        def to_tokens(self, text):
            assert text == prompt
            return torch.tensor([prefix])

        def hooks(self, *, fwd_hooks):
            assert fwd_hooks == [("blocks.2.hook_resid_pre", "sentinel")]
            return nullcontext()

        def generate(self, tokens, **options):
            assert options.pop("return_type", "input") in ("tokens", "input")
            assert options == dict(max_new_tokens=3, temperature=0.4, do_sample=False,
                                   top_p=0.8, verbose=False)
            return torch.cat([tokens, torch.tensor([[3, 4]])], dim=1)

        def to_string(self, tokens):
            if tokens.tolist() == [3, 4]:
                return "  answer!  "
            assert tokens.tolist() == prefix + [3, 4]
            return decoded_prefix + "  answer!  "

    # Demonstrate why character slicing is not a valid tokenizer boundary.
    assert (decoded_prefix + "  answer!  ")[len(prompt):].strip() != "answer!"
    wrapper = load_steerable_model()(Model())
    wrapper.add_hook(2, "sentinel")
    assert wrapper.generate(prompt, max_new_tokens=3, temperature=0.4,
                            do_sample=False, top_p=0.8) == "answer!"
