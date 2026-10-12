"""Broad-study hosted-score parsing and persona-universe completeness regressions.

GeminiJudge.judge_score is compiled from the source AST and exercised with an
in-memory provider stub and synthetic persona rows.
"""

from __future__ import annotations

import ast
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "broad_study"))

from evaluation import (  # noqa: E402
    LEGACY_BROAD_STUDY_REFUSAL_TRAIT_SCORE,
    InvalidJudgeScore,
    checkpointed_persona_traits,
    decode_persona_checkpoint_responses,
    judgments_are_complete,
    parse_judge_score,
    parse_persona_response_key,
    persona_response_universe,
    reject_duplicate_json_keys,
    require_complete_evaluation,
    require_complete_persona_corpus,
    study_judgments_are_complete,
)


def load_judge_score():
    """Compile only GeminiJudge.judge_score from run_study.py's AST."""
    tree = ast.parse((ROOT / "experiments" / "broad_study" / "run_study.py").read_text(encoding="utf-8"))
    judge = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GeminiJudge"
    )
    method = next(
        node for node in judge.body
        if isinstance(node, ast.FunctionDef) and node.name == "judge_score"
    )
    namespace = {
        "parse_judge_score": parse_judge_score,
        "InvalidJudgeScore": InvalidJudgeScore,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), "run_study.py", "exec"), namespace)
    return namespace["judge_score"]


class StubJudge:
    judge_score = load_judge_score()

    def __init__(self, reply, status="ok"):
        self.reply, self.status = reply, status
        self.stats = {"total": 0, "ok": 0, "safety_blocks": 0, "empty": 0, "errors": 0, "invalid": 0}

    def _call(self, prompt):
        self.stats["total"] += 1
        return self.reply, self.status, None


class HostedScoreParsingTests(unittest.TestCase):
    def test_out_of_range_signed_and_embedded_scores_are_rejected(self) -> None:
        for reply in ("-5", "101", "score 60, not 90"):
            with self.subTest(reply=reply), self.assertRaises(InvalidJudgeScore):
                parse_judge_score(reply, legacy_refusal_as_zero=True)

    def test_text_outside_documented_grammar_is_rejected(self) -> None:
        for reply in (
            "", "   ", "60 90", "60, 90", "Score: 60", "60/100", "80.", ".5", "+5",
            "1e2", "nan", "inf", "100.5", "100.0001", "50\n60", "٥٠",
            "REFUSAL 0", None, 60,
        ):
            with self.subTest(reply=reply), self.assertRaises(InvalidJudgeScore):
                parse_judge_score(reply, legacy_refusal_as_zero=True)

    def test_decimals_and_bounds_are_kept_exactly(self) -> None:
        self.assertEqual(parse_judge_score("15.5"), 15.5)
        self.assertEqual(parse_judge_score("0"), 0.0)
        self.assertEqual(parse_judge_score("100"), 100.0)
        self.assertEqual(parse_judge_score("100.0"), 100.0)
        self.assertEqual(parse_judge_score(" 42\n"), 42.0)

    def test_refusal_is_only_the_named_legacy_trait_policy(self) -> None:
        self.assertEqual(LEGACY_BROAD_STUDY_REFUSAL_TRAIT_SCORE, 0.0)
        for reply in ("REFUSAL", "refusal"):
            self.assertEqual(
                parse_judge_score(reply, legacy_refusal_as_zero=True),
                LEGACY_BROAD_STUDY_REFUSAL_TRAIT_SCORE,
            )
            with self.assertRaises(InvalidJudgeScore):
                parse_judge_score(reply)

    def test_invalid_category_blocks_aggregation(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "invalid=1"):
            require_complete_evaluation(
                {"safety_blocks": 0, "errors": 0, "empty": 0, "invalid": 1}
            )

    def test_runner_judge_counts_parser_failures_as_incomplete(self) -> None:
        for reply in ("-5", "101", "score 60, not 90"):
            judge = StubJudge(reply)
            self.assertIsNone(judge.judge_score("{question} {answer}", "q", "a", legacy_refusal_as_zero=True))
            self.assertEqual((judge.stats["ok"], judge.stats["invalid"]), (0, 1))
            with self.assertRaises(RuntimeError):
                require_complete_evaluation(judge.stats)
        judge = StubJudge("15.5")
        self.assertEqual(judge.judge_score("{question} {answer}", "q", "a"), 15.5)
        judge = StubJudge("REFUSAL")
        self.assertEqual(judge.judge_score("p", "q", "a", legacy_refusal_as_zero=True), 0.0)
        self.assertIsNone(judge.judge_score("p", "q", "a"))
        self.assertEqual((judge.stats["ok"], judge.stats["invalid"]), (1, 1))
        judge = StubJudge(None, status="error")
        self.assertIsNone(judge.judge_score("p", "q", "a"))
        self.assertEqual(judge.stats["errors"], 1)


# Synthetic configuration: two traits, two conditions, four samples per arm.
QUESTIONS = {"trait_a": ["qa0", "qa1", "qa2"], "trait_b": ["qb0", "qb1", "qb2"]}
CONDITIONS = ["baseline", "arm_1"]
N_SAMPLES = 4
KEYS = persona_response_universe(QUESTIONS, CONDITIONS, N_SAMPLES)


def row(key, trait_score=None, coherence=None):
    trait, _condition, index = key
    questions = QUESTIONS[trait]
    return {
        "question": questions[index % len(questions)],
        "response": f"synthetic response {key}",
        "trait_score": trait_score,
        "coherence": coherence,
    }


def corpus(keys=KEYS, **scores):
    return {key: row(key, **scores) for key in keys}


class PersonaUniverseTests(unittest.TestCase):
    def test_universe_is_derived_from_configuration(self) -> None:
        self.assertEqual(len(KEYS), 2 * 2 * 4)
        self.assertEqual(len(set(KEYS)), len(KEYS))
        with self.assertRaises(ValueError):
            persona_response_universe(["t", "t"], CONDITIONS, 1)
        with self.assertRaises(ValueError):
            persona_response_universe(["t"], CONDITIONS, 0)

    def test_historical_configuration_requires_320_rows_per_trait(self) -> None:
        materials = json.loads((ROOT / "experimental_materials.json").read_text(encoding="utf-8"))
        traits = list(materials["traits"])
        keys = persona_response_universe(traits, [f"arm_{i}" for i in range(8)], 40)
        self.assertEqual(len(keys), len(traits) * 320)
        questions = {name: data["eval_questions"] for name, data in materials["traits"].items()}
        one = {keys[0]: {"question": questions[traits[0]][0], "response": "synthetic",
                         "trait_score": 50.0, "coherence": 80.0}}
        with self.assertRaisesRegex(RuntimeError, "incomplete traits"):
            checkpointed_persona_traits(one, keys, questions)

    def test_single_row_is_field_complete_but_not_study_complete(self) -> None:
        single = {KEYS[0]: row(KEYS[0], trait_score=50.0, coherence=80.0)}
        self.assertTrue(judgments_are_complete(single))
        self.assertFalse(study_judgments_are_complete(single, KEYS))
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            require_complete_persona_corpus(single, KEYS, QUESTIONS)
        with self.assertRaisesRegex(RuntimeError, "incomplete traits"):
            checkpointed_persona_traits(single, KEYS, QUESTIONS)

    def test_complete_trait_may_be_skipped(self) -> None:
        trait_a = corpus([key for key in KEYS if key[0] == "trait_a"])
        self.assertEqual(checkpointed_persona_traits(trait_a, KEYS, QUESTIONS), {"trait_a"})
        self.assertEqual(checkpointed_persona_traits({}, KEYS, QUESTIONS), set())
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            require_complete_persona_corpus(trait_a, KEYS, QUESTIONS)

    def test_missing_condition_or_sample_is_rejected(self) -> None:
        for dropped in (
            [key for key in KEYS if key[:2] == ("trait_a", "arm_1")],
            [("trait_b", "baseline", 3)],
        ):
            rows = corpus([key for key in KEYS if key not in dropped])
            with self.subTest(dropped=dropped[0]):
                with self.assertRaisesRegex(RuntimeError, "incomplete traits"):
                    checkpointed_persona_traits(rows, KEYS, QUESTIONS)
                with self.assertRaisesRegex(RuntimeError, "incomplete"):
                    require_complete_persona_corpus(rows, KEYS, QUESTIONS)

    def test_extra_keys_are_rejected(self) -> None:
        for extra in (
            ("trait_a", "baseline", N_SAMPLES),
            ("trait_a", "unknown_arm", 0),
            ("trait_c", "baseline", 0),
            ("trait_a", "baseline", -1),
            "('trait_a', 'baseline', 0)",
        ):
            rows = corpus()
            rows[extra] = {"question": "qa0", "response": "x", "trait_score": None, "coherence": None}
            with self.subTest(extra=extra), self.assertRaisesRegex(RuntimeError, "outside"):
                require_complete_persona_corpus(rows, KEYS, QUESTIONS)

    def test_wrong_question_binding_or_malformed_row_is_rejected(self) -> None:
        rows = corpus()
        rows[("trait_a", "arm_1", 3)]["question"] = "qa1"  # 3 % 3 binds qa0
        with self.assertRaisesRegex(RuntimeError, "question binding"):
            require_complete_persona_corpus(rows, KEYS, QUESTIONS)
        for mutate in (
            lambda value: value.update(extra_field=1),
            lambda value: value.pop("coherence"),
            lambda value: value.update(response=None),
        ):
            rows = corpus()
            mutate(rows[KEYS[0]])
            with self.assertRaisesRegex(RuntimeError, "Malformed"):
                require_complete_persona_corpus(rows, KEYS, QUESTIONS)

    def test_checkpoint_keys_are_canonical_and_unique(self) -> None:
        self.assertEqual(
            parse_persona_response_key("('trait_a', 'baseline', 0)"), ("trait_a", "baseline", 0)
        )
        for text in (
            "('trait_a','baseline',0)",
            "(trait_a, baseline, 0)",
            "('trait_a', 'baseline', True)",
            "('trait_a', 'baseline', 0.0)",
            "('trait_a', 'baseline', 0, 1)",
            "['trait_a', 'baseline', 0]",
            "__import__('os')",
        ):
            with self.subTest(text=text), self.assertRaisesRegex(RuntimeError, "Malformed"):
                parse_persona_response_key(text)
        duplicated = (
            '{"responses": {"(\'trait_a\', \'baseline\', 0)": 1, '
            '"(\'trait_a\', \'baseline\', 0)": 2}}'
        )
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            json.loads(duplicated, object_pairs_hook=reject_duplicate_json_keys)
        with self.assertRaises(RuntimeError):
            decode_persona_checkpoint_responses(None)

    def test_checkpoint_round_trip_and_full_run_export(self) -> None:
        # Same serialization as run_study.py's persona checkpoint writer.
        text = json.dumps({"responses": {str(key): value for key, value in corpus().items()}})
        saved = json.loads(text, object_pairs_hook=reject_duplicate_json_keys)
        rows = decode_persona_checkpoint_responses(saved["responses"])
        self.assertEqual(checkpointed_persona_traits(rows, KEYS, QUESTIONS), set(QUESTIONS))
        require_complete_persona_corpus(rows, KEYS, QUESTIONS)
        self.assertFalse(study_judgments_are_complete(rows, KEYS))
        scored = corpus(trait_score=15.5, coherence=LEGACY_BROAD_STUDY_REFUSAL_TRAIT_SCORE)
        self.assertTrue(study_judgments_are_complete(scored, KEYS))
        for bad in (None, 101.0, -1.0, math.nan, math.inf, True, "50"):
            rows = corpus(trait_score=15.5, coherence=80.0)
            rows[KEYS[-1]]["coherence"] = bad
            with self.subTest(bad=bad):
                self.assertFalse(study_judgments_are_complete(rows, KEYS))


def synthetic_upstream_rows(filename: str) -> list[dict[str, str]]:
    """Synthetic upstream file with unique placeholder text in each row."""
    return [
        {"input": f"in-{filename}-{index}", "output": f"out-{filename}-{index}"}
        for index in range(40)
    ]


class FunctionVectorFetchTests(unittest.TestCase):
    """The live upstream fetch is not exercised here (no network); fetch_json is stubbed."""

    def setUp(self) -> None:
        import fetch_function_vectors

        self.fetch = fetch_function_vectors
        self.rows = {
            filename: synthetic_upstream_rows(filename)
            for filename, _indices in fetch_function_vectors.SELECTIONS.values()
        }
        self.original_fetch_json = fetch_function_vectors.fetch_json
        self.original_digests = dict(fetch_function_vectors.EXPECTED_PAIR_SHA256)
        fetch_function_vectors.fetch_json = lambda filename: (
            self.rows[filename], "0" * 64
        )

    def tearDown(self) -> None:
        self.fetch.fetch_json = self.original_fetch_json
        self.fetch.EXPECTED_PAIR_SHA256.clear()
        self.fetch.EXPECTED_PAIR_SHA256.update(self.original_digests)

    def synthetic_pairs(self, task: str) -> list[list[str]]:
        filename, indices = self.fetch.SELECTIONS[task]
        return [
            [self.rows[filename][index]["input"], self.rows[filename][index]["output"]]
            for index in indices
        ]

    def pin_synthetic_digests(self) -> None:
        self.fetch.EXPECTED_PAIR_SHA256.update(
            {task: self.fetch.pair_sha256(self.synthetic_pairs(task)) for task in self.fetch.SELECTIONS}
        )

    def test_expected_digests_cover_exactly_the_six_tasks(self) -> None:
        expected_tasks = {
            "antonym", "country_capital", "present_past",
            "english_french", "singular_plural", "capitalize",
        }
        self.assertEqual(set(self.original_digests), expected_tasks)
        self.assertEqual(set(self.fetch.SELECTIONS), expected_tasks)
        self.assertEqual(len(set(self.original_digests.values())), 6)
        for task, digest in self.original_digests.items():
            with self.subTest(task=task):
                self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertEqual(
            self.fetch.pair_canonical_json([["a", "é"], ["b", "c"]]), '[["a","é"],["b","c"]]'
        )

    def test_matching_pairs_are_written_to_the_stated_path(self) -> None:
        self.pin_synthetic_digests()
        with tempfile.TemporaryDirectory() as scratch:
            output = Path(scratch) / "nested" / "function_vectors.json"
            self.fetch.main(["--output", str(output)])
            self.assertEqual(sorted(path.name for path in Path(scratch).rglob("*")),
                             ["function_vectors.json", "nested"])
            document = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(set(document["functions"]), set(self.fetch.SELECTIONS))
        for task in self.fetch.SELECTIONS:
            self.assertEqual(document["functions"][task]["pairs"], self.synthetic_pairs(task))
        self.fetch.require_expected_pairs(document["functions"])

    def test_tampered_selected_row_fails_closed_without_output(self) -> None:
        self.pin_synthetic_digests()
        tampered = [
            ("country-capital.json", 5, "output"),
            ("singular-plural.json", 38, "input"),  # last selected singular-plural row
            ("english-french.json", 32, "output"),  # last selected english-french row
        ]
        task_of = {filename: task for task, (filename, _) in self.fetch.SELECTIONS.items()}
        for filename, index, field in tampered:
            with self.subTest(filename=filename, index=index):
                self.rows = {name: synthetic_upstream_rows(name) for name in self.rows}
                self.rows[filename][index][field] += " "
                with tempfile.TemporaryDirectory() as scratch:
                    output = Path(scratch) / "nested" / "function_vectors.json"
                    with self.assertRaisesRegex(ValueError, f"task {task_of[filename]} "):
                        self.fetch.main(["--output", str(output)])
                    self.assertEqual(list(Path(scratch).iterdir()), [])

    def test_unselected_row_changes_are_ignored(self) -> None:
        self.pin_synthetic_digests()
        self.rows["english-french.json"][18]["output"] += " changed"  # skipped source row
        self.rows["singular-plural.json"][15]["output"] += " changed"  # not selected
        with tempfile.TemporaryDirectory() as scratch:
            self.fetch.main(["--output", str(Path(scratch) / "fv.json")])

    def test_real_digests_reject_non_executed_pairs_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            output = Path(scratch) / "function_vectors.json"
            with self.assertRaisesRegex(ValueError, "task antonym .*refusing"):
                self.fetch.main(["--output", str(output)])
            self.assertFalse(output.exists())

    def test_malformed_rows_and_task_sets_are_rejected(self) -> None:
        self.pin_synthetic_digests()
        for bad in ({"input": "x"}, {"output": "y"}, {"input": 1, "output": "y"}, ["x", "y"]):
            with self.subTest(bad=bad):
                self.rows = {
                    filename: synthetic_upstream_rows(filename)
                    for filename, _ in self.fetch.SELECTIONS.values()
                }
                self.rows["antonyms.json"][3] = bad
                with tempfile.TemporaryDirectory() as scratch:
                    with self.assertRaisesRegex(ValueError, "Malformed row 3"):
                        self.fetch.main(["--output", str(Path(scratch) / "fv.json")])
        self.rows = {name: synthetic_upstream_rows(name) for name in self.rows}
        functions = {task: {"pairs": self.synthetic_pairs(task)} for task in self.fetch.SELECTIONS}
        self.fetch.require_expected_pairs(functions)
        with self.assertRaisesRegex(ValueError, "tasks must be exactly"):
            self.fetch.require_expected_pairs({**functions, "extra": {"pairs": []}})
        functions["capitalize"] = {"pairs": [("a", "b")]}
        with self.assertRaisesRegex(ValueError, "Malformed Function Vector pairs for task capitalize"):
            self.fetch.require_expected_pairs(functions)

    def test_broad_study_runner_checks_pair_digests_on_load(self) -> None:
        tree = ast.parse((ROOT / "experiments" / "broad_study" / "run_study.py").read_text(encoding="utf-8"))
        statements = [ast.unparse(node) for node in tree.body]

        def first(prefix: str) -> int:
            return next(i for i, text in enumerate(statements) if text.startswith(prefix))

        imports = statements.index("from experiments.broad_study.fetch_function_vectors import require_expected_pairs")
        check = next(
            index for index, node in enumerate(tree.body)
            if isinstance(node, ast.Try)
            and "require_expected_pairs(SEMANTIC_FUNCTIONS)" in ast.unparse(node)
        )
        materials = first("try:\n    materials_bytes")
        # Local inputs are validated before the heavy imports, output creation,
        # model load and run specification.
        self.assertLess(imports, check)
        self.assertLess(check, materials)
        self.assertLess(materials, first("import torch"))
        self.assertLess(first("import torch"), first("OUTPUT_DIR.mkdir"))
        self.assertLess(first("OUTPUT_DIR.mkdir"), first("os.chdir(OUTPUT_DIR)"))
        self.assertLess(first("os.chdir(OUTPUT_DIR)"), first("model = HookedTransformer"))
        self.assertLess(first("model = HookedTransformer"), first("RUN_SPECIFICATION ="))
        self.assertEqual(sum(text.startswith("OUTPUT_DIR.mkdir") for text in statements), 1)


if __name__ == "__main__":
    unittest.main()
