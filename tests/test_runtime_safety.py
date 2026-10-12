"""Preflight and output-safety regressions for model-run entry points.

Subprocess fixtures block PyTorch and model imports: valid preflight reaches the
first heavy import, while invalid input must fail earlier. Synthetic Function
Vector pairs use digests scoped to the test subprocess or patched module.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
BROAD_STUDY = ROOT / "experiments" / "broad_study"
PROTOCOL_RUNNER = ROOT / "experiments" / "protocol_swap" / "source" / "runner.py"
ORIGINAL_SCR = ROOT / "plot_data" / "deep_validation_results.json"
REVISION = "0" * 40
HEAVY_MODULES = ("torch", "numpy", "scipy", "tqdm", "transformer_lens", "transformers")
BROAD_STUDY_HEAVY_MODULES = HEAVY_MODULES + (
    "matplotlib", "google", "google.genai", "google.api_core", "backoff",
)
BROAD_STUDY_SOURCES = (
    "run_study.py", "checkpointing.py", "evaluation.py", "fetch_function_vectors.py", "extraction.py",
)
MATERIALS = ROOT / "experimental_materials.json"


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


fetch = load_module("_test_runtime_fetch_function_vectors", BROAD_STUDY / "fetch_function_vectors.py")
evaluation = load_module("_test_runtime_broad_study_evaluation", BROAD_STUDY / "evaluation.py")
protocol = load_module("_test_runtime_protocol_swap_runner", PROTOCOL_RUNNER)


def synthetic_functions() -> dict[str, dict[str, list[list[str]]]]:
    return {
        task: {"pairs": [[f"in-{task}-{index}", f"out-{task}-{index}"] for index in range(32)]}
        for task in fetch.SELECTIONS
    }


def synthetic_digests() -> dict[str, str]:
    return {task: fetch.pair_sha256(spec["pairs"]) for task, spec in synthetic_functions().items()}


class ProtocolSwapPreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scratch = Path(tempfile.mkdtemp())
        self.addCleanup(self._cleanup)
        self.function_vectors = self.scratch / "function_vectors.json"
        self.function_vectors.write_text(
            json.dumps({"functions": synthetic_functions()}), encoding="utf-8"
        )

    def _cleanup(self) -> None:
        for path in sorted(self.scratch.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if path.is_symlink() or path.is_file():
                path.unlink()
            else:
                path.rmdir()
        self.scratch.rmdir()

    def run_runner(self, *arguments: str, pin_synthetic: bool = True,
                   function_vectors: Path | None = None) -> subprocess.CompletedProcess:
        # The runner loads fetch_function_vectors.py by file location; synthetic
        # digests are pinned on that module instance as it is executed.
        bootstrap = (
            "import importlib.util, json, runpy, sys\n"
            f"for name in {HEAVY_MODULES!r}:\n"
            "    sys.modules[name] = None\n"
            "digests = json.loads(sys.argv[1])\n"
            "original = importlib.util.spec_from_file_location\n"
            "def pinned(name, location, *args, **kwargs):\n"
            "    specification = original(name, location, *args, **kwargs)\n"
            f"    if digests and str(location) == {str(BROAD_STUDY / 'fetch_function_vectors.py')!r}:\n"
            "        execute = specification.loader.exec_module\n"
            "        def exec_module(module):\n"
            "            execute(module)\n"
            "            module.EXPECTED_PAIR_SHA256.update(digests)\n"
            "        specification.loader.exec_module = exec_module\n"
            "    return specification\n"
            "importlib.util.spec_from_file_location = pinned\n"
            "sys.argv = sys.argv[2:]\n"
            "runpy.run_path(sys.argv[0], run_name='__main__')\n"
        )
        vectors = function_vectors if function_vectors is not None else self.function_vectors
        command = [
            sys.executable, "-B", "-c", bootstrap,
            json.dumps(synthetic_digests() if pin_synthetic else {}),
            str(PROTOCOL_RUNNER), "--function-vectors", str(vectors), *arguments,
        ]
        return subprocess.run(
            command, cwd=self.scratch, capture_output=True, text=True, check=False,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        )

    def arguments(self, output: Path | None, model: str = "qwen1.5b",
                  revision: str = REVISION, original: Path | None = ORIGINAL_SCR) -> list[str]:
        arguments = ["--model", model, "--model-revision", revision]
        if output is not None:
            arguments += ["--output", str(output)]
        if original is not None:
            arguments += ["--original-scr-json", str(original)]
        return arguments

    def assert_reaches_heavy_imports(self, completed: subprocess.CompletedProcess) -> None:
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("ModuleNotFoundError", completed.stderr)
        self.assertIn("torch", completed.stderr.strip().splitlines()[-1])
        self.assertNotIn("MODEL:", completed.stdout)

    def assert_refused(self, completed: subprocess.CompletedProcess, message: str) -> None:
        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertIn(message, completed.stderr)
        self.assertNotIn("ModuleNotFoundError", completed.stderr)
        self.assertNotIn("MODEL:", completed.stdout)

    def test_valid_new_output_outside_repository_passes_preflight(self) -> None:
        output = self.scratch / "new" / "protocol_swap_results.json"
        self.assert_reaches_heavy_imports(self.run_runner(*self.arguments(output)))
        self.assertFalse(output.parent.exists())

    def test_existing_output_with_other_models_passes_preflight_unchanged(self) -> None:
        output = self.scratch / "results.json"
        text = json.dumps({"Qwen2.5-7B": {"summary": {}}})
        output.write_text(text, encoding="utf-8")
        self.assert_reaches_heavy_imports(self.run_runner(*self.arguments(output)))
        self.assertEqual(output.read_text(encoding="utf-8"), text)

    def test_output_is_required(self) -> None:
        self.assert_refused(self.run_runner(*self.arguments(None)), "--output")

    def test_repository_outputs_are_refused(self) -> None:
        frozen = ROOT / "experiments" / "protocol_swap" / "results" / "protocol_swap_results.json"
        before = frozen.read_bytes()
        alias = self.scratch / "alias"
        alias.symlink_to(ROOT / "plot_data", target_is_directory=True)
        targets = [
            frozen,
            ROOT / "protocol_swap_results.json",
            ROOT / "experiments" / "broad_study" / "runs" / "protocol_swap_results.json",
            alias / "protocol_swap_results.json",
        ]
        swapped = Path(str(ROOT).swapcase())
        if swapped.exists() and os.path.samefile(swapped, ROOT):
            targets.append(swapped / "protocol_swap_results.json")
        for target in targets:
            with self.subTest(target=target):
                self.assert_refused(
                    self.run_runner(*self.arguments(target)), "inside the repository"
                )
        self.assertEqual(frozen.read_bytes(), before)
        self.assertFalse((ROOT / "protocol_swap_results.json").exists())
        self.assertFalse((ROOT / "plot_data" / "protocol_swap_results.json").exists())

    def test_unsafe_existing_outputs_are_refused(self) -> None:
        link_target = self.scratch / "target.json"
        link_target.write_text("{}", encoding="utf-8")
        symlink = self.scratch / "link.json"
        symlink.symlink_to(link_target)
        dangling = self.scratch / "dangling.json"
        dangling.symlink_to(self.scratch / "absent.json")
        duplicate = self.scratch / "duplicate.json"
        duplicate.write_text(json.dumps({"Qwen2.5-1.5B": {"kept": True}}), encoding="utf-8")
        listing = self.scratch / "list.json"
        listing.write_text("[]", encoding="utf-8")
        broken = self.scratch / "broken.json"
        broken.write_text("{", encoding="utf-8")
        directory = self.scratch / "directory.json"
        directory.mkdir()
        blocker = self.scratch / "file"
        blocker.write_text("", encoding="utf-8")
        cases = [
            (symlink, "symlink"),
            (dangling, "symlink"),
            (duplicate, "already contains results for Qwen2.5-1.5B"),
            (listing, "JSON object"),
            (broken, "not readable JSON"),
            (directory, "not a regular file"),
            (blocker / "out.json", "not a writable directory"),
        ]
        snapshots = {path: path.read_bytes() for path in (link_target, duplicate, listing, broken)}
        for output, message in cases:
            with self.subTest(output=output.name):
                self.assert_refused(self.run_runner(*self.arguments(output)), message)
        for path, data in snapshots.items():
            self.assertEqual(path.read_bytes(), data)
        self.assertFalse((self.scratch / "absent.json").exists())

    def test_model_revision_must_be_a_full_commit(self) -> None:
        output = self.scratch / "out.json"
        for revision in ("main", "0" * 39, "g" * 40):
            with self.subTest(revision=revision):
                self.assert_refused(
                    self.run_runner(*self.arguments(output, revision=revision)),
                    "--model-revision",
                )

    def test_function_vector_digests_are_checked_before_model_load(self) -> None:
        output = self.scratch / "out.json"
        self.assert_refused(
            self.run_runner(*self.arguments(output), pin_synthetic=False),
            "do not match the executed broad-study selection",
        )
        self.assert_refused(
            self.run_runner(*self.arguments(output),
                            function_vectors=self.scratch / "missing.json"),
            "fetch_function_vectors.py",
        )
        functions = synthetic_functions()
        functions["antonym"]["pairs"][0][1] += " "
        tampered = self.scratch / "tampered.json"
        tampered.write_text(json.dumps({"functions": functions}), encoding="utf-8")
        self.assert_refused(
            self.run_runner(*self.arguments(output), function_vectors=tampered), "task antonym"
        )
        for document in ([], {"functions": []}):
            malformed = self.scratch / "malformed.json"
            malformed.write_text(json.dumps(document), encoding="utf-8")
            with self.subTest(document=document):
                self.assert_refused(
                    self.run_runner(*self.arguments(output), function_vectors=malformed),
                    "six pinned semantic tasks",
                )

    def test_original_scr_json_is_validated_before_model_load(self) -> None:
        output = self.scratch / "out.json"
        document = json.loads(ORIGINAL_SCR.read_text(encoding="utf-8"))
        without_model = {key: value for key, value in document.items() if key != "Qwen2.5-1.5B"}
        missing_value = json.loads(json.dumps(document))
        del missing_value["Qwen2.5-1.5B"]["SCR"]["values"]["evil"]
        non_finite = json.loads(json.dumps(document))
        non_finite["Qwen2.5-1.5B"]["SCR"]["values"]["evil"] = float("nan")
        cases = {
            "absent": (None, "not a readable file"),
            "list": ([], "JSON object keyed by model"),
            "without_model": (without_model, "no SCR entry for Qwen2.5-1.5B"),
            "missing_value": (missing_value, "finite SCR for each of the 13 concepts"),
            "non_finite": (non_finite, "finite SCR for each of the 13 concepts"),
        }
        for name, (content, message) in cases.items():
            path = self.scratch / f"{name}.json"
            if content is not None:
                path.write_text(json.dumps(content), encoding="utf-8")
            with self.subTest(case=name):
                self.assert_refused(
                    self.run_runner(*self.arguments(output, original=path)), message
                )
        for model in ("qwen7b", "llama8b"):
            with self.subTest(model=model):
                self.assert_reaches_heavy_imports(
                    self.run_runner(*self.arguments(output, model=model))
                )


class ProtocolSwapSaveTests(unittest.TestCase):
    def test_runner_module_defers_heavy_imports(self) -> None:
        for name in ("torch", "np", "stats", "tqdm"):
            self.assertNotIn(name, vars(protocol))

    def test_seed_is_42_and_seeds_every_generator_after_heavy_imports(self) -> None:
        tree = ast.parse(PROTOCOL_RUNNER.read_text(encoding="utf-8"))
        seed = [
            node.value.value for node in tree.body
            if isinstance(node, ast.Assign)
            and [ast.unparse(target) for target in node.targets] == ["SEED"]
        ]
        self.assertEqual(seed, [42])
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "import_heavy_modules"
        )
        calls = {ast.unparse(node) for node in ast.walk(function) if isinstance(node, ast.Call)}
        for call in ("random.seed(SEED)", "np.random.seed(SEED)", "torch.manual_seed(SEED)"):
            with self.subTest(call=call):
                self.assertIn(call, calls)
        self.assertEqual(protocol.SEED, 42)

    def test_pair_digest_check_loads_fetch_module_by_file_location(self) -> None:
        source = PROTOCOL_RUNNER.read_text(encoding="utf-8")
        self.assertNotIn("sys.path", source)
        self.assertNotIn("from fetch_function_vectors import", source)
        self.assertNotIn("import fetch_function_vectors", source)
        functions = synthetic_functions()
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "function_vectors.json"
            path.write_text(json.dumps({"functions": functions}), encoding="utf-8")
            with mock.patch.object(protocol, "SEMANTIC_FUNCTIONS", None), \
                    mock.patch.object(protocol.importlib.util, "spec_from_file_location",
                                      wraps=importlib.util.spec_from_file_location) as locate:
                with self.assertRaisesRegex(ValueError, "do not match the executed broad-study selection"):
                    protocol.get_semantic_functions(path)
        self.assertEqual(locate.call_args.args[1], BROAD_STUDY / "fetch_function_vectors.py")

    def test_original_scr_values_load_for_each_model(self) -> None:
        for model in protocol.MODEL_CONFIGS.values():
            with self.subTest(model=model["short"]):
                values = protocol.load_original_scr(model["short"], str(ORIGINAL_SCR))
                self.assertEqual(len(values), 13)

    def test_entries_accumulate_atomically_and_are_never_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            output = Path(scratch).resolve() / "nested" / "protocol_swap_results.json"
            protocol.save_results_atomically(output, "Qwen2.5-7B", {"value": 1})
            protocol.save_results_atomically(output, "Qwen2.5-1.5B", {"value": 2})
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")),
                {"Qwen2.5-7B": {"value": 1}, "Qwen2.5-1.5B": {"value": 2}},
            )
            before = output.read_bytes()
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                protocol.save_results_atomically(output, "Qwen2.5-7B", {"value": 3})
            with mock.patch.object(protocol.os, "replace", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    protocol.save_results_atomically(output, "Llama-3.1-8B", {"value": 4})
            self.assertEqual(output.read_bytes(), before)
            # The lock inode must persist so waiting and future writers share it.
            self.assertEqual(sorted(path.name for path in output.parent.iterdir()),
                             sorted([output.name, f".{output.name}.lock"]))

    def test_save_rechecks_repository_location(self) -> None:
        target = ROOT / "plot_data" / "protocol_swap_results.json"
        with self.assertRaisesRegex(ValueError, "inside the repository"):
            protocol.save_results_atomically(target, "Qwen2.5-7B", {})
        self.assertFalse(target.exists())


class BroadStudyPreflightTests(unittest.TestCase):
    """run_study.py and its helpers are copied into a scratch tree and run there.

    The scratch tree mirrors the repository layout the runner reads:
    experiments/broad_study/function_vectors.json and experimental_materials.json at the
    scratch root. Nothing in the repository is read or written by the runner.
    """

    def setUp(self) -> None:
        self.scratch = Path(tempfile.mkdtemp()).resolve()
        self.addCleanup(shutil.rmtree, self.scratch)
        self.broad_study = self.scratch / "experiments" / "broad_study"
        self.broad_study.mkdir(parents=True)
        for name in BROAD_STUDY_SOURCES:
            shutil.copyfile(BROAD_STUDY / name, self.broad_study / name)
        shutil.copyfile(ROOT / "extract_activations.py", self.scratch / "extract_activations.py")
        self.function_vectors = self.broad_study / "function_vectors.json"
        self.materials = self.scratch / "experimental_materials.json"
        self.output = self.scratch / "run"
        self.write_json(self.function_vectors, {"functions": synthetic_functions()})
        self.write_json(self.materials, self.shipped_materials())

    @staticmethod
    def shipped_materials() -> dict:
        return json.loads(MATERIALS.read_text(encoding="utf-8"))

    @staticmethod
    def write_json(path: Path, document: object) -> None:
        path.write_text(json.dumps(document), encoding="utf-8")

    def run_runner(self, *, pin_synthetic: bool = True,
                   output_dir: bool = True) -> subprocess.CompletedProcess:
        # Pin the synthetic pair digests before executing the copied module.
        bootstrap = (
            "import json, runpy, sys\n"
            f"for name in {BROAD_STUDY_HEAVY_MODULES!r}:\n"
            "    sys.modules[name] = None\n"
            f"sys.path.insert(0, {str(self.scratch)!r})\n"
            "from experiments.broad_study import fetch_function_vectors\n"
            "fetch_function_vectors.EXPECTED_PAIR_SHA256.update(json.loads(sys.argv[1]))\n"
            "sys.argv = sys.argv[2:]\n"
            "runpy.run_path(sys.argv[0], run_name='__main__')\n"
        )
        arguments = ["--model", "Qwen/Qwen2.5-1.5B-Instruct", "--model-revision", REVISION]
        if output_dir:
            arguments += ["--output-dir", str(self.output)]
        command = [
            sys.executable, "-B", "-c", bootstrap,
            json.dumps(synthetic_digests() if pin_synthetic else {}),
            str(self.broad_study / "run_study.py"), *arguments,
        ]
        return subprocess.run(
            command, cwd=self.scratch, capture_output=True, text=True, check=False,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        )

    def assert_no_output(self, completed: subprocess.CompletedProcess) -> None:
        # Nothing is printed before the heavy imports; the model-load summary
        # ("layers=") and run directory are never reached.
        self.assertEqual(completed.stdout, "")
        self.assertNotIn("HookedTransformer", completed.stderr)
        self.assertFalse(self.output.exists())
        self.assertFalse((self.broad_study / "runs").exists())

    def assert_refused(self, completed: subprocess.CompletedProcess, *messages: str) -> None:
        self.assertEqual(completed.returncode, 2, completed.stderr)
        for message in messages:
            self.assertIn(message, completed.stderr)
        self.assertNotIn("ModuleNotFoundError", completed.stderr)
        self.assert_no_output(completed)

    def test_valid_inputs_pass_preflight_and_stop_at_first_heavy_import(self) -> None:
        for output_dir in (True, False):
            with self.subTest(output_dir=output_dir):
                completed = self.run_runner(output_dir=output_dir)
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("ModuleNotFoundError", completed.stderr)
                self.assertIn("torch", completed.stderr.strip().splitlines()[-1])
                self.assert_no_output(completed)

    def test_function_vectors_are_validated_before_heavy_imports(self) -> None:
        self.assert_refused(
            self.run_runner(pin_synthetic=False), "do not match the executed broad-study selection"
        )
        tampered = synthetic_functions()
        tampered["antonym"]["pairs"][0][1] += " "
        short = synthetic_functions()
        short["capitalize"]["pairs"].pop()
        cases = {
            "tampered": ({"functions": tampered}, ("task antonym", "do not match")),
            "non_object": ([], ("'functions' mapping",)),
            "functions_list": ({"functions": []}, ("'functions' mapping",)),
            "pair_count": ({"functions": short}, ("capitalize must contain 32 pairs",)),
        }
        for name, (document, messages) in cases.items():
            self.write_json(self.function_vectors, document)
            with self.subTest(case=name):
                self.assert_refused(self.run_runner(), str(self.function_vectors), *messages)
        self.function_vectors.write_text("{", encoding="utf-8")
        self.assert_refused(self.run_runner(), str(self.function_vectors))
        self.function_vectors.unlink()
        self.assert_refused(self.run_runner(), "Run fetch_function_vectors.py first")

    def test_experimental_materials_are_validated_before_heavy_imports(self) -> None:
        def mutated(change):
            document = self.shipped_materials()
            change(document)
            return document

        cases = {
            "non_object": ([], "'traits' mapping"),
            "traits_list": (mutated(lambda d: d.update(traits=[])), "'traits' mapping"),
            "no_coherence": (mutated(lambda d: d.pop("coherence_prompt")), "coherence_prompt"),
            "no_sycophancy": (
                mutated(lambda d: d["traits"].pop("sycophancy")), "seven persona traits"
            ),
            "extra_trait": (
                mutated(lambda d: d["traits"].update(extra=d["traits"]["evil"])),
                "seven persona traits",
            ),
            "no_rubric": (
                mutated(lambda d: d["traits"]["evil"].pop("eval_prompt")), "evaluation rubric"
            ),
            "four_instructions": (
                mutated(lambda d: d["traits"]["evil"]["instructions"].pop()), "5 instruction pairs"
            ),
            "instruction_without_neg": (
                mutated(lambda d: d["traits"]["evil"]["instructions"][0].pop("neg")),
                "'pos' and 'neg'",
            ),
            "nineteen_extract": (
                mutated(lambda d: d["traits"]["humorous"]["extract_questions"].pop()),
                "20 extraction questions",
            ),
            "twenty_one_eval": (
                mutated(lambda d: d["traits"]["humorous"]["eval_questions"].append("q")),
                "20 evaluation questions",
            ),
        }
        for name, (document, message) in cases.items():
            self.write_json(self.materials, document)
            with self.subTest(case=name):
                self.assert_refused(self.run_runner(), str(self.materials), message)
        self.materials.write_text("{", encoding="utf-8")
        self.assert_refused(self.run_runner(), str(self.materials))
        self.materials.unlink()
        self.assert_refused(self.run_runner(), str(self.materials), "No such file")

    def test_shipped_materials_and_synthetic_vectors_pass_the_validators(self) -> None:
        evaluation.require_experimental_materials(self.shipped_materials())
        functions = synthetic_functions()
        self.assertEqual(
            evaluation.require_function_vector_document({"functions": functions}), functions
        )


class FunctionVectorOutputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.calls: list[str] = []
        self.original_fetch_json = fetch.fetch_json
        self.original_digests = dict(fetch.EXPECTED_PAIR_SHA256)
        rows = {
            filename: [{"input": f"in-{task}-{i}", "output": f"out-{task}-{i}"} for i in range(40)]
            for task, (filename, _) in fetch.SELECTIONS.items()
        }

        def stub(filename: str):
            self.calls.append(filename)
            return rows[filename], "0" * 64

        fetch.fetch_json = stub
        fetch.EXPECTED_PAIR_SHA256.update(
            {
                task: fetch.pair_sha256([
                    [rows[filename][i]["input"], rows[filename][i]["output"]] for i in indices
                ])
                for task, (filename, indices) in fetch.SELECTIONS.items()
            }
        )

    def tearDown(self) -> None:
        fetch.fetch_json = self.original_fetch_json
        fetch.EXPECTED_PAIR_SHA256.clear()
        fetch.EXPECTED_PAIR_SHA256.update(self.original_digests)

    def assert_refused_before_fetch(self, output: Path, message: str) -> None:
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit) as raised:
            fetch.main(["--output", str(output)])
        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(self.calls, [])
        with self.assertRaisesRegex(ValueError, message):
            fetch.validate_output(output)

    def test_repository_paths_other_than_the_default_cache_are_refused(self) -> None:
        # Validation only: every case is refused before any fetch or write.
        cases = [
            (ROOT / "README.md", "repository path"),
            (BROAD_STUDY / "run_study.py", "repository path"),
            (ROOT / "experimental_materials.json", "repository path"),
            (ROOT / "plot_data" / "function_vectors.json", "repository path"),
            (ROOT / "provenance" / "function_vectors.json", "repository path"),
            (ROOT / "experiments" / "independent_judge" / "fv.json", "repository path"),
            (ROOT / "function_vectors.json", "repository path"),
            (BROAD_STUDY / "function_vectors_copy.json", "repository path"),
            (BROAD_STUDY / "runs" / "function_vectors.json", "repository path"),
            (ROOT / ".env", "repository path"),
            (ROOT / ".venv" / "function_vectors.json", "repository path"),
            (ROOT, "not a regular file"),
        ]
        swapped = Path(str(fetch.DEFAULT_OUTPUT).swapcase())
        if swapped.parent.exists() and os.path.samefile(swapped.parent, fetch.DEFAULT_OUTPUT.parent):
            cases.append((swapped, "repository path"))
        for output, message in cases:
            with self.subTest(output=output):
                before = output.read_bytes() if output.is_file() else None
                self.assert_refused_before_fetch(output, message)
                if before is None:
                    self.assertFalse(output.is_file())
                else:
                    self.assertEqual(output.read_bytes(), before)

    def test_default_cache_is_the_only_allowed_repository_path(self) -> None:
        self.assertEqual(fetch.validate_output(fetch.DEFAULT_OUTPUT), fetch.DEFAULT_OUTPUT)
        relative = Path("experiments") / "broad_study" / "function_vectors.json"
        self.assertEqual(fetch.DEFAULT_OUTPUT, ROOT / relative)

    def test_symlink_outputs_and_aliases_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            scratch = Path(scratch)
            target = scratch / "target.json"
            target.write_text("kept", encoding="utf-8")
            link = scratch / "link.json"
            link.symlink_to(target)
            dangling = scratch / "dangling.json"
            dangling.symlink_to(scratch / "absent.json")
            alias = scratch / "alias"
            alias.symlink_to(ROOT / "plot_data", target_is_directory=True)
            directory = scratch / "directory.json"
            directory.mkdir()
            for output, message in (
                (link, "symlink"),
                (dangling, "symlink"),
                (alias / "function_vectors.json", "repository path"),
                (directory, "not a regular file"),
            ):
                with self.subTest(output=output.name):
                    self.assert_refused_before_fetch(output, message)
            self.assertEqual(target.read_text(encoding="utf-8"), "kept")
            self.assertFalse((scratch / "absent.json").exists())
            self.assertFalse((ROOT / "plot_data" / "function_vectors.json").exists())

    def test_refetch_replaces_outside_file_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            output = Path(scratch) / "function_vectors.json"
            output.write_text("stale", encoding="utf-8")
            fetch.main(["--output", str(output)])
            document = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(set(document["functions"]), set(fetch.SELECTIONS))
            self.assertEqual(len(self.calls), 6)
            self.assertEqual([path.name for path in Path(scratch).iterdir()], [output.name])
            with mock.patch.object(fetch.os, "replace", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    fetch.main(["--output", str(output)])
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), document)
            self.assertEqual([path.name for path in Path(scratch).iterdir()], [output.name])


if __name__ == "__main__":
    unittest.main()
