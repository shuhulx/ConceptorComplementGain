"""Broad-study Function Vector next-token scoring and decoding rules.

The scoring rule and the next-token decoder are extracted from
``experiments/broad_study/run_study.py`` with ``ast`` and executed in isolation on
synthetic strings and a fake model, avoiding the runner's model initialization.
"""

from __future__ import annotations

import ast
import contextlib
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments" / "broad_study" / "run_study.py"


def _is_prefix_rule(node: ast.AST) -> bool:
    """Match ``<prediction>.strip().lower().startswith(...)``."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "startswith"
        and isinstance(node.func.value, ast.Call)
        and isinstance(node.func.value.func, ast.Attribute)
        and node.func.value.func.attr == "lower"
        and isinstance(node.func.value.func.value, ast.Call)
        and isinstance(node.func.value.func.value.func, ast.Attribute)
        and node.func.value.func.value.func.attr == "strip"
        and isinstance(node.func.value.func.value.func.value, ast.Name)
    )


def extracted_scorers():
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"), filename=str(RUNNER))
    scorers = []
    for node in ast.walk(tree):
        if not _is_prefix_rule(node):
            continue
        prediction = node.func.value.func.value.func.value.id
        names = sorted(
            {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} - {prediction}
        )
        if len(names) != 1:
            raise AssertionError(f"Unexpected scorer free names at line {node.lineno}")
        function = ast.Expression(
            ast.Lambda(
                args=ast.arguments(
                    posonlyargs=[],
                    args=[ast.arg(prediction), ast.arg(names[0])],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],
                ),
                body=node,
            )
        )
        ast.fix_missing_locations(function)
        code = compile(function, str(RUNNER), "eval")
        scorers.append((node.lineno, eval(code, {"__builtins__": {}})))
    return tree, scorers


class SemanticScorerTests(unittest.TestCase):
    def test_semantic_scorer_is_case_insensitive_prefix_next_token(self) -> None:
        tree, scorers = extracted_scorers()
        # Three accuracy loops plus three per-prompt diagnostic marks.
        self.assertGreaterEqual(len(scorers), 6)
        synthetic_cases = [
            # (decoded top-1 next token, expected answer, scored correct)
            ("paris", "Paris", True),  # case-insensitive
            ("  PAR", "Paris", True),  # whitespace stripped, upper case accepted
            ("Par", "Paris", True),  # one partial token counts: not full-answer match
            ("Parliament", "Paris", True),  # only the first three characters matter
            ("hello", "Hello", True),  # letter case of Capitalize cannot be assessed
            ("Pa", "Paris", False),  # shorter than the three-character prefix
            ("London", "Paris", False),
            ("an", "a", True),  # expected answers shorter than three characters
        ]
        for line, scorer in scorers:
            for prediction, expected, correct in synthetic_cases:
                self.assertIs(
                    bool(scorer(prediction, expected)),
                    correct,
                    f"line {line}: {prediction!r} vs {expected!r}",
                )

        # Every scored prediction list comes from top_1_next_token.
        prediction_sources = {
            target.id: node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name) and target.id.startswith("preds")
        }
        self.assertTrue(prediction_sources)
        for name, value in prediction_sources.items():
            self.assertIsInstance(value, ast.Call, name)
            self.assertEqual(value.func.attr, "top_1_next_token", name)

    def test_top_1_next_token_decodes_one_argmax_token(self) -> None:
        tree = ast.parse(RUNNER.read_text(encoding="utf-8"), filename=str(RUNNER))
        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "top_1_next_token"
        )
        namespace = {
            "torch": SimpleNamespace(argmax=lambda values, dim: np.argmax(values, axis=dim))
        }
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(RUNNER), "exec"), namespace)

        vocabulary = [" Paris", " France", "ian", " the"]
        # Synthetic logits: final position favours token 0 for prompt 0 and token 2
        # for prompt 1; earlier positions favour other tokens and must be ignored.
        logits = np.array(
            [
                [[0, 9, 0, 0], [5, 1, 0, 0]],
                [[0, 0, 0, 9], [0, 1, 7, 0]],
            ],
            dtype=float,
        )
        steerable = SimpleNamespace(hooks=[], model=_FakeModel(vocabulary, logits))
        predictions = namespace["top_1_next_token"](steerable, ["a", "b"])
        self.assertEqual(predictions, ["Paris", "ian"])


class _FakeModel:
    """Synthetic stand-in exposing only what top_1_next_token touches."""

    def __init__(self, vocabulary: list[str], logits: np.ndarray) -> None:
        self.tokenizer = SimpleNamespace(decode=lambda ids: vocabulary[ids[0]])
        self._logits = logits

    def to_tokens(self, prompts, padding_side):
        return np.zeros((len(prompts), self._logits.shape[1]), dtype=int)

    def hooks(self, fwd_hooks):
        return contextlib.nullcontext()

    def __call__(self, tokens):
        return self._logits


if __name__ == "__main__":
    unittest.main()
