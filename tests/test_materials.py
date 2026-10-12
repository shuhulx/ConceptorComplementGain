from __future__ import annotations

import ast
import csv
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class MaterialTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = json.loads((ROOT / "experimental_materials.json").read_text())

    def test_complete_target_inventory(self) -> None:
        expected = {
            "sycophancy",
            "hallucination",
            "evil",
            "impolite",
            "optimistic",
            "apathetic",
            "humorous",
        }
        self.assertEqual(set(self.document["traits"]), expected)
        instruction_strings = set()
        extraction = set()
        evaluation = set()
        for name, trait in self.document["traits"].items():
            self.assertEqual(len(trait["instructions"]), 5, name)
            self.assertEqual(len(trait["extract_questions"]), 20, name)
            self.assertEqual(len(trait["eval_questions"]), 20, name)
            self.assertEqual(len(set(trait["extract_questions"])), 20, name)
            self.assertEqual(len(set(trait["eval_questions"])), 20, name)
            self.assertIn("{question}", trait["eval_prompt"])
            self.assertIn("{answer}", trait["eval_prompt"])
            for pair in trait["instructions"]:
                self.assertEqual(set(pair), {"pos", "neg"}, name)
                self.assertTrue(pair["pos"].strip() and pair["neg"].strip(), name)
                self.assertNotEqual(pair["pos"], pair["neg"], name)
                instruction_strings.update((name, pair[key]) for key in ("pos", "neg"))
            extraction.update((name, q) for q in trait["extract_questions"])
            evaluation.update((name, q) for q in trait["eval_questions"])
        # 7 targets x 5 pairs x 2 polarities; 5 pairs x 20 questions = 100 rows
        # per polarity and target.
        self.assertEqual(len(instruction_strings), 70)
        self.assertEqual((len(extraction), len(evaluation)), (140, 140))

    def test_extraction_and_evaluation_are_separate(self) -> None:
        for name, trait in self.document["traits"].items():
            self.assertEqual(
                set(trait["extract_questions"]).intersection(trait["eval_questions"]),
                set(),
                name,
            )

    def test_coherence_prompt_placeholders(self) -> None:
        prompt = self.document["coherence_prompt"]
        self.assertIn("{question}", prompt)
        self.assertIn("{answer}", prompt)

    def test_upstream_materials_keep_separate_license_terms(self) -> None:
        scope = (ROOT / "LICENSE.md").read_text()
        notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text()
        self.assertIn("Apache-2.0", scope)
        self.assertIn("CC BY 4.0", scope)
        self.assertIn("coherence_prompt", scope)
        self.assertIn("ConceptorSteering function vectors", notices)
        self.assertTrue((ROOT / "third_party/persona_vectors/LICENSE").is_file())
        self.assertIn("function_vectors.json", (ROOT / ".gitignore").read_text())

    def test_independent_judge_materials(self) -> None:
        prompts = json.loads((ROOT / "deepseek_judge_prompts.json").read_text())
        self.assertEqual(len(prompts["evaluated_traits"]), 4)
        self.assertEqual(len(prompts["evaluated_conditions"]), 4)
        self.assertEqual(set(prompts["trait_rubrics"]), set(prompts["evaluated_traits"]))
        with (ROOT / "secondary_evaluator_selection.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 20)
        self.assertEqual(len({(r["target"], r["question_index"], r["replicate"]) for r in rows}), 20)

    def test_embedded_runner_materials_match_canonical_inventory(self) -> None:
        canonical = self.document["traits"]
        protocol = self._literal_assignment(
            ROOT / "experiments" / "protocol_swap" / "source" / "runner.py",
            "PERSONA_TRAITS",
        )
        expected_protocol = {
            name: {
                "instructions": material["instructions"],
                "extract_questions": material["extract_questions"],
            }
            for name, material in canonical.items()
        }
        self.assertEqual(protocol, expected_protocol)

    @staticmethod
    def _literal_assignment(path: Path, variable_name: str):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if any(
                isinstance(target, ast.Name) and target.id == variable_name
                for target in node.targets
            ):
                return ast.literal_eval(node.value)
        raise AssertionError(f"Missing literal {variable_name} in {path}")


if __name__ == "__main__":
    unittest.main()
