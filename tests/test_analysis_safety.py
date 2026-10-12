from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPECTRAL_SCREEN = (
    ROOT / "experiments" / "spectral_screening" / "source" / "formula_screening.py"
)


class AnalysisSafetyTests(unittest.TestCase):
    def test_spectral_screen_uses_two_sided_inference(self) -> None:
        source = (
            ROOT
            / "experiments"
            / "spectral_screening"
            / "source"
            / "formula_screening.py"
        ).read_text(encoding="utf-8")
        self.assertIn("oriented_two_sided_mann_whitney", source)
        self.assertNotIn("alternative='greater'", source)
        self.assertNotIn('alternative="greater"', source)
        self.assertNotIn("formula_screening_results_corrected.json", source)
        self.assertIn("evaluate_finite_formula", source)
        self.assertIn("--extractions-sha256", source)
        self.assertIn("--run-manifest", source)
        self.assertIn("PT_SHA256", source)
        self.assertIn('ckpt.get("run_specification_sha256")', source)
        self.assertIn("Participation-ratio effective rank", source)

    def test_spectral_screen_labels_are_descriptive(self) -> None:
        source = SPECTRAL_SCREEN.read_text(encoding="utf-8")
        lowered = source.lower()
        for rejected in (
            "significant steering",
            "steering_significant",
            "adds info",
            "steering accuracy",
            "reported with bonferroni flags",
        ):
            self.assertNotIn(rejected, lowered)
        self.assertIn("'steering_nominal_p_lt_05_descriptive'", source)
        self.assertIn("nominal asymptotic Spearman p<0.05", source)
        self.assertIn("n=6, descriptive only", source)
        self.assertIn("next-token prefix", source)
        self.assertIn("'bonferroni_pass'", source)

    def _run_spectral_screen(
        self, output: Path | None, *extra: str, inputs: Path | None = None
    ) -> subprocess.CompletedProcess:
        # Heavy dependencies are made unimportable: argument validation must fail
        # before numpy/scipy/torch are imported or any input file is read, and a
        # valid run stops at the first heavy import, so nothing is ever written.
        bootstrap = (
            "import runpy, sys\n"
            "for name in ('numpy', 'scipy', 'torch', 'spectral_metrics'):\n"
            "    sys.modules[name] = None\n"
            "sys.argv = sys.argv[1:]\n"
            "runpy.run_path(sys.argv[0], run_name='__main__')\n"
        )
        with tempfile.TemporaryDirectory() as scratch:
            source = inputs if inputs is not None else Path(scratch) / "missing"
            command = [
                sys.executable, "-B", "-c", bootstrap, str(SPECTRAL_SCREEN),
                "--extractions", str(source / "extractions.pt"),
                "--extractions-sha256", "0" * 64,
                "--run-manifest", str(source / "run_manifest.json"),
                *(("--output", str(output)) if output is not None else ()),
                *extra,
            ]
            return subprocess.run(
                command, cwd=scratch, capture_output=True, text=True, check=False,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
            )

    def assert_refused(self, completed: subprocess.CompletedProcess, message: str) -> None:
        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertIn(message, completed.stderr)
        self.assertNotIn("ModuleNotFoundError", completed.stderr)
        self.assertNotIn("Loading", completed.stdout)

    def test_spectral_screen_refuses_in_repository_output_before_loading(self) -> None:
        results = SPECTRAL_SCREEN.parents[1] / "results"
        targets = [
            results / "formula_screening_results_corrected.json",
            results / "new_subdirectory" / "screen.json",
            ROOT / "plot_data" / "screen.json",
            ROOT / "screen.json",
            ROOT,
        ]
        swapped = Path(str(ROOT).swapcase())
        if swapped.exists() and os.path.samefile(swapped, ROOT):
            targets.append(swapped / "plot_data" / "screen.json")  # case-insensitive alias
        for target in targets:
            with self.subTest(target=target):
                self.assert_refused(self._run_spectral_screen(target), "inside the repository")
                if target != ROOT:
                    self.assertFalse(target.exists())
        self.assertFalse((results / "new_subdirectory").exists())

    def test_spectral_screen_refuses_symlink_aliases_into_repository(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            alias = Path(scratch) / "alias"
            alias.symlink_to(SPECTRAL_SCREEN.parents[1] / "results", target_is_directory=True)
            dangling = Path(scratch) / "dangling.json"
            dangling.symlink_to(ROOT / "plot_data" / "absent_screen.json")
            for target in (alias / "screen.json", alias / "nested" / "screen.json", dangling):
                with self.subTest(target=target):
                    self.assert_refused(
                        self._run_spectral_screen(target), "inside the repository"
                    )
            self.assertFalse((alias / "nested").exists())
            self.assertFalse((ROOT / "plot_data" / "absent_screen.json").exists())

    def test_spectral_screen_refuses_existing_and_symlink_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            existing = Path(scratch) / "existing.json"
            existing.write_text("{}", encoding="utf-8")
            outside_dangling = Path(scratch) / "outside_dangling.json"
            outside_dangling.symlink_to(Path(scratch) / "absent.json")
            for target in (existing, outside_dangling):
                with self.subTest(target=target):
                    self.assert_refused(self._run_spectral_screen(target), "refusing to overwrite")
            self.assertEqual(existing.read_text(encoding="utf-8"), "{}")
            self.assertFalse((Path(scratch) / "absent.json").exists())
            blocker = Path(scratch) / "file"
            blocker.write_text("", encoding="utf-8")
            self.assert_refused(
                self._run_spectral_screen(blocker / "screen.json"), "is not a directory"
            )

    def test_spectral_screen_requires_output(self) -> None:
        completed = self._run_spectral_screen(None)
        self.assertEqual(completed.returncode, 2)
        self.assertIn("--output", completed.stderr)
        self.assertNotIn("Loading", completed.stdout)

    def test_spectral_screen_refuses_missing_inputs_before_imports(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            target = Path(scratch) / "screen.json"
            self.assert_refused(self._run_spectral_screen(target), "input not found")
            self.assertFalse(target.exists())

    def test_spectral_screen_accepts_outside_output_then_stops_at_imports(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            inputs = Path(scratch) / "inputs"
            inputs.mkdir()
            for name in ("extractions.pt", "run_manifest.json", "consolidated.json"):
                (inputs / name).write_text("SYNTHETIC placeholder", encoding="utf-8")
            target = Path(scratch) / "new" / "screen.json"
            completed = self._run_spectral_screen(target, inputs=inputs)
            self.assertNotEqual(completed.returncode, 0)
            self.assertNotIn("refusing", completed.stderr)
            self.assertIn("ModuleNotFoundError", completed.stderr)
            self.assertFalse(target.parent.exists())


    def test_broad_study_export_schema_and_separate_outcome_panels(self) -> None:
        path = ROOT / "provenance" / "broad_study_results.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            set(document),
            {"Qwen2.5-1.5B", "Qwen2.5-7B", "Llama-3.1-8B"},
        )
        for model in document.values():
            results = model["results"]
            self.assertNotIn("key_findings", results)
            self.assertNotIn("version", results)
            for field in ("spectral", "semantic_results", "persona_table", "correlation"):
                self.assertTrue(results[field], field)

    def test_broad_study_checkpoints_are_run_bound(self) -> None:
        source = (ROOT / "experiments" / "broad_study" / "run_study.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("initialize_checkpoint_directory", source)
        self.assertGreaterEqual(source.count("require_checkpoint_identity"), 4)
        self.assertIn('"run_specification_sha256"', source)
        self.assertIn('"runner_sha256"', source)
        self.assertIn('"runtime"', source)

    def test_broad_study_reports_continuous_spectral_measurements(self) -> None:
        source = (ROOT / "experiments" / "broad_study" / "run_study.py").read_text(
            encoding="utf-8"
        ).lower()
        for rejected in (
            "conceptors fail",
            "will conceptors work",
            "the real claim",
            "binary diagnostic",
            "signal preservation ratio",
            "n=8",
        ):
            self.assertNotIn(rejected, source)
        self.assertIn("squared power ratio", source)
        self.assertIn("participation-ratio effective rank", source)

    def test_broad_study_judgments_fail_closed_and_missing_scores_are_not_zero(self) -> None:
        source = (ROOT / "experiments" / "broad_study" / "run_study.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("prompt_block_reason", source)
        self.assertIn("require_complete_evaluation", source)
        self.assertIn("judgments_are_complete", source)
        self.assertNotIn("if tvs else 0", source)
        self.assertNotIn("if tvs_h else 0", source)

    def test_checkpoint_extraction_uses_exclusive_temporaries(self) -> None:
        for relative in (
            Path("experiments/matched_control/source/runner.py"),
            Path("experiments/erasure/source/runner.py"),
        ):
            source = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn("tempfile.mkstemp", source, str(relative))
            self.assertNotIn('target.with_suffix(".pt.tmp")', source, str(relative))


def load_spectral_output_functions(repository: Path) -> dict:
    """Compile only is_inside_repository and open_new_output from the screen's AST.

    The module's top-level script (argument parsing, input loading) is never run;
    REPO_ROOT is bound to ``repository`` so tests can use a stand-in repository.
    """
    tree = ast.parse(SPECTRAL_SCREEN.read_text(encoding="utf-8"), filename=str(SPECTRAL_SCREEN))
    wanted = {"is_inside_repository", "open_new_output"}
    module = ast.Module(
        body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted],
        type_ignores=[],
    )
    namespace = {"os": os, "Path": Path, "REPO_ROOT": repository}
    exec(compile(module, str(SPECTRAL_SCREEN), "exec"), namespace)
    assert wanted <= set(namespace)
    return namespace


class SpectralScreenExclusiveWriteTests(unittest.TestCase):
    def setUp(self) -> None:
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.scratch = Path(scratch.name).resolve()
        self.repository = self.scratch / "repository"  # SYNTHETIC stand-in for the release tree
        self.repository.mkdir()
        self.functions = load_spectral_output_functions(self.repository)
        self.open_new_output = self.functions["open_new_output"]

    def test_screen_binds_repository_root_to_the_release_tree(self) -> None:
        tree = ast.parse(SPECTRAL_SCREEN.read_text(encoding="utf-8"))
        assignments = [
            node for node in tree.body
            if isinstance(node, ast.Assign)
            and [ast.unparse(target) for target in node.targets] in (["BASE_DIR"], ["REPO_ROOT"])
        ]
        namespace = {"Path": Path, "__file__": str(SPECTRAL_SCREEN)}
        exec(compile(ast.Module(body=assignments, type_ignores=[]), str(SPECTRAL_SCREEN), "exec"),
             namespace)
        self.assertEqual(namespace["REPO_ROOT"], ROOT)

    def test_new_output_is_created_and_written(self) -> None:
        target = self.scratch / "outside" / "nested" / "screen.json"
        with self.open_new_output(target) as handle:
            handle.write("{}")
        self.assertEqual(target.read_text(encoding="utf-8"), "{}")

    def test_existing_output_is_refused_unchanged(self) -> None:
        existing = self.scratch / "existing.json"
        existing.write_text("kept", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.open_new_output(existing)
        self.assertEqual(existing.read_text(encoding="utf-8"), "kept")

    def test_symlink_outputs_are_refused_without_following(self) -> None:
        dangling = self.scratch / "dangling.json"
        dangling.symlink_to(self.scratch / "absent.json")
        target = self.scratch / "target.json"
        target.write_text("kept", encoding="utf-8")
        link = self.scratch / "link.json"
        link.symlink_to(target)
        for output in (dangling, link):
            with self.subTest(output=output.name), self.assertRaises(FileExistsError):
                self.open_new_output(output)
        self.assertFalse((self.scratch / "absent.json").exists())
        self.assertEqual(target.read_text(encoding="utf-8"), "kept")

    def test_parent_replaced_after_validation_is_refused_before_creation(self) -> None:
        elsewhere = self.scratch / "elsewhere"
        elsewhere.mkdir()
        inside = self.repository / "plot_data"
        inside.mkdir()
        for destination in (inside, elsewhere):
            with self.subTest(destination=destination.name):
                parent = self.scratch / f"validated_{destination.name}"
                parent.mkdir()
                validated = parent / "screen.json"
                # Validation saw a real directory outside the repository; it is
                # then replaced by a symlink before the write.
                parent.rmdir()
                parent.symlink_to(destination, target_is_directory=True)
                with mock.patch.object(os, "open", side_effect=AssertionError("file opened")):
                    with self.assertRaisesRegex(RuntimeError, "changed after validation"):
                        self.open_new_output(validated)
                self.assertEqual(list(destination.iterdir()), [])
        self.assertEqual([path.name for path in self.repository.iterdir()], ["plot_data"])

    def test_case_alias_parent_inside_repository_is_refused(self) -> None:
        swapped = Path(str(self.repository).swapcase())
        if not swapped.exists() or not os.path.samefile(swapped, self.repository):
            self.skipTest("filesystem is case-sensitive")
        with mock.patch.object(os, "open", side_effect=AssertionError("file opened")):
            with self.assertRaisesRegex(RuntimeError, "changed after validation"):
                self.open_new_output(swapped / "screen.json")
        self.assertEqual(list(self.repository.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
