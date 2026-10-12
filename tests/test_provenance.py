from __future__ import annotations

import ast
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
CLAIM_MAP = ROOT / "docs" / "RESULTS.md"
CURRENT_PAPER_TITLE = (
    "When a Conceptor Complement Becomes Gain: Operator Geometry and Behavioral "
    "Attribution in Language Models"
)
CURRENT_PAPER_AUTHORS = ("Shuhul Razdan", "R Balamurugan")
CURRENT_PAPER_VENUE = "NeurReps 2026 (Symmetry and Geometry in Neural Representations)"
TOP_LEVEL_DIRECTORIES = (
    ".github/",
    "docs/",
    "experiments/",
    "examples/",
    "eval/",
    "plot_data/",
    "provenance/",
    "scripts/",
    "tests/",
    "third_party/",
)
PATH_SUFFIXES = (".cfg", ".csv", ".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml")
BARE_NAME_SUFFIXES = (".md", ".py", ".toml", ".txt")
WORKSTATION_PATH = re.compile(r"(?:^|\s)(?:/Users/|/home/|/content/drive/|[A-Za-z]:\\)")


def documentation_files() -> list[Path]:
    return sorted(
        [ROOT / "README.md", ROOT / "LICENSE.md", ROOT / "THIRD_PARTY_NOTICES.md"]
        + list((ROOT / "docs").glob("*.md"))
        + list((ROOT / "experiments").rglob("README.md"))
        + list((ROOT / "third_party").rglob("*.md"))
    )


def documented_basenames() -> set[str]:
    names = {path.name for path in ROOT.iterdir() if path.is_file()}
    for directory in TOP_LEVEL_DIRECTORIES:
        for _, children, files in os.walk(ROOT / directory):
            children[:] = [
                name for name in children
                if name not in {".venv", ".git", "__pycache__", "runs", ".cache"}
            ]
            names.update(files)
    return names


BASENAMES = documented_basenames()


def markdown_anchors(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"^```.*?^```\s*$", "", text, flags=re.MULTILINE | re.DOTALL)
    anchors = set(re.findall(r'<a\s+(?:id|name)="([^"]+)"', text))
    counts = {}
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*#*\s*$", text, flags=re.MULTILINE):
        slug = re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
        number = counts.get(slug, 0)
        counts[slug] = number + 1
        anchors.add(f"{slug}-{number}" if number else slug)
    return anchors


def text_files() -> list[Path]:
    return [
        *documentation_files(),
        ROOT / "pyproject.toml",
        ROOT / ".github" / "workflows" / "verify.yml",
        ROOT / "requirements.txt",
    ]


def referenced_paths(text: str) -> set[str]:
    """Repository-looking paths in inline code spans and fenced code blocks."""
    fence = re.compile(r"^(```|~~~).*?$(.*?)^\1\s*$", re.MULTILINE | re.DOTALL)
    snippets = [block[1] for block in fence.findall(text)]
    without_fences = fence.sub("", text)
    for span in re.findall(r"`([^`\n]+)`", without_fences):
        if not re.search(r"\s", span):
            snippets.append(span)
    # Keep quoted arguments whole, separating them from Python calls and shell flags.
    token_pattern = re.compile(r"""(['"])([^\n]*?)\1|([^\s'"(),;]+)""")
    tokens = {
        match[2] if match[1] else match[3]
        for snippet in snippets
        for match in token_pattern.finditer(snippet)
    }
    paths = set()
    for token in tokens:
        token = token.strip("\"'(),;:")
        token = re.sub(r"(#.*|:\d+)$", "", token).rstrip(".")
        if (
            not token
            or token.startswith(("/", "-", "~", "$"))
            or "://" in token
            or any(character in token for character in "<>*{}$=?|[]")
        ):
            continue
        if "/" in token:
            if token.endswith(PATH_SUFFIXES) or token.startswith(TOP_LEVEL_DIRECTORIES):
                paths.add(token)
        elif token.endswith(BARE_NAME_SUFFIXES):
            paths.add(token)
    return paths


def is_git_ignored(relative: str) -> bool:
    if not (ROOT / ".git").exists():
        local_outputs = {"experiments/broad_study/function_vectors.json", "experiments/broad_study/runs/"}
        return relative in local_outputs and relative in (ROOT / ".gitignore").read_text()
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--no-index", relative],
        cwd=ROOT,
        check=False,
    )
    return result.returncode == 0


class ProvenanceTests(unittest.TestCase):
    def test_released_records_exclude_private_execution_metadata(self) -> None:
        excluded = {
            "run_signature", "source_receipt", "input_hashes", "provider_run_id",
            "generation_archive", "extraction_archive", "extraction_member",
            "generation_manifest_member", "historical_sources", "completed_unix",
            "first_judgment_utc", "last_judgment_utc", "peak_uncached_cost_upper_usd",
        }

        def check_json(value, path):
            if isinstance(value, dict):
                self.assertFalse(set(value) & excluded, str(path))
                for item in value.values():
                    check_json(item, path)
            elif isinstance(value, list):
                for item in value:
                    check_json(item, path)

        directories = [ROOT / "plot_data", ROOT / "provenance"]
        directories.extend((ROOT / "experiments").glob("*/results"))
        for directory in directories:
            for path in directory.rglob("*.json"):
                check_json(json.loads(path.read_text()), path.relative_to(ROOT))
            for path in directory.rglob("*.csv"):
                with path.open(newline="") as stream:
                    columns = csv.DictReader(stream).fieldnames or []
                self.assertFalse(set(columns) & excluded, str(path))

    def test_claim_map_names_existing_checks(self) -> None:
        named_checks = set(re.findall(r"`(test_[A-Za-z0-9_]+)`", CLAIM_MAP.read_text()))
        test_source = "\n".join(
            path.read_text(encoding="utf-8") for path in sorted((ROOT / "tests").glob("test_*.py"))
        )
        missing = sorted(name for name in named_checks if f"def {name}(" not in test_source)
        self.assertEqual(missing, [])

    def test_claim_map_evidence_files_exist(self) -> None:
        evidence = (
            "plot_data/operator_cells.csv",
            "provenance/broad_study_results.json",
            "plot_data/correction_cells.csv",
            "plot_data/corrected_coherence_results.json",
            "experiments/matched_control/results/combined/combined_exact_output_match.csv",
            "experiments/matched_control/results/combined/combined_not2_vs_identity2_by_view.csv",
            "plot_data/gemini_behavior_cells.csv",
            "experiments/matched_control/results/combined/combined_categorical_trait_refusals.csv",
            "deepseek_judge_prompts.json",
            "secondary_evaluator_selection.csv",
            "experiments/independent_judge/results/analysis/validation_decision.json",
            "plot_data/erasure_rank_sweep_summary.csv",
            "plot_data/paired_geometry_main.csv",
            "plot_data/paired_delta_results.json",
            "plot_data/semantic_per_concept_outcomes.csv",
            "plot_data/protocol_swap_summary.csv",
            "experiments/protocol_swap/results/protocol_swap_results.json",
            "plot_data/formula_screening_results.csv",
            "plot_data/finite_alternatives_scores.csv",
            "plot_data/deep_validation_results.json",
            "plot_data/broad_generated_response_summary.json",
            "experimental_materials.json",
        )
        missing = [relative for relative in evidence if not (ROOT / relative).is_file()]
        self.assertEqual(missing, [])

    def test_internal_markdown_links_resolve(self) -> None:
        link_pattern = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
        broken = []
        for document in documentation_files():
            relative = document.relative_to(ROOT).as_posix()
            for raw_target in link_pattern.findall(document.read_text(encoding="utf-8")):
                target = raw_target.strip()
                if target.startswith("<") and target.endswith(">"):
                    target = target[1:-1]
                else:
                    target = target.split(maxsplit=1)[0]
                if target.startswith(("http://", "https://", "mailto:")):
                    continue
                path, _, anchor = unquote(target).partition("#")
                destination = document.parent / path if path else document
                if not destination.exists():
                    broken.append(f"{relative} -> {target}")
                    continue
                if destination.is_dir():
                    destination = destination / "README.md"
                if anchor and (not destination.is_file() or anchor not in markdown_anchors(destination)):
                    broken.append(f"{relative} -> {target} (missing anchor)")
        self.assertEqual(broken, [])

    def test_documented_repository_paths_exist(self) -> None:
        # A referenced path must exist relative to the repository root or the
        # document, or be a documented local output that .gitignore excludes.
        missing = []
        for document in documentation_files():
            for token in sorted(referenced_paths(document.read_text(encoding="utf-8"))):
                if token == "eval/prompts.py" and document.relative_to(ROOT).as_posix() in {
                    "LICENSE.md", "third_party/persona_vectors/CHANGES.md",
                }:
                    continue  # Explicitly attributed upstream path, not a local file.
                candidates = [ROOT / token, document.parent / token]
                if any(candidate.exists() for candidate in candidates):
                    continue
                if "/" not in token and token in BASENAMES:
                    continue
                relative = (document.parent / token).resolve()
                if relative.is_relative_to(ROOT) and is_git_ignored(
                    relative.relative_to(ROOT).as_posix()
                ):
                    continue
                if is_git_ignored(token):
                    continue
                missing.append(f"{document.relative_to(ROOT)}: {token}")
        self.assertEqual(missing, [])

    def test_path_extraction_on_synthetic_markdown(self) -> None:
        sample = "\n".join(
            [
                "Run `scripts/verify_semantic_rebuild.py` and read `docs/`.",
                "Math such as `R = X^T X / n` and `<private>/x.zip` is ignored.",
                "```bash",
                "python3 scripts/verify_semantic_rebuild.py --out /tmp/x",
                "python -m pip install -r requirements.txt",
                "```",
            ]
        )
        self.assertEqual(
            referenced_paths(sample),
            {
                "scripts/verify_semantic_rebuild.py",
                "docs/",
                "scripts/verify_semantic_rebuild.py",
                "requirements.txt",
            },
        )

    def test_python_csv_arguments_yield_only_the_embedded_path(self) -> None:
        cases = (
            (
                "d = pd.read_csv('plot_data/aperture_baseline_21_fits.csv')",
                {"plot_data/aperture_baseline_21_fits.csv"},
            ),
            ('d=pd.read_csv("plot_data/nonexistent.csv")', {"plot_data/nonexistent.csv"}),
            (
                "d = pd.read_csv(filepath_or_buffer='plot_data/missing.csv', sep=',')",
                {"plot_data/missing.csv"},
            ),
            ('d = pd.read_csv("plot_data/missing records.csv")', {"plot_data/missing records.csv"}),
            ("d = pd.read_csv(Path('plot_data/missing.csv'))", {"plot_data/missing.csv"}),
        )
        for statement, expected in cases:
            with self.subTest(statement=statement):
                self.assertEqual(referenced_paths(f"```python\n{statement}\n```"), expected)

    def test_missing_paths_remain_visible_in_shell_fences_and_inline_code(self) -> None:
        sample = "\n".join(
            [
                "Read `plot_data/missing_inline.csv` and `docs/missing.md:12`.",
                "```bash",
                "python examples/missing.py --input 'plot_data/missing_shell.csv'",
                'python scripts/missing.py --input="plot_data/missing_option.csv"',
                "```",
                "~~~python",
                "d = pd.read_csv('plot_data/missing_python.csv')",
                "~~~",
            ]
        )
        self.assertEqual(
            referenced_paths(sample),
            {
                "plot_data/missing_inline.csv",
                "docs/missing.md",
                "examples/missing.py",
                "scripts/missing.py",
                "plot_data/missing_shell.csv",
                "plot_data/missing_option.csv",
                "plot_data/missing_python.csv",
            },
        )

    def test_urls_and_angle_bracket_placeholders_are_not_repository_paths(self) -> None:
        sample = "\n".join(
            [
                "Use `<private>/input.csv` or `https://example.org/input.csv`.",
                "```python",
                'd = pd.read_csv("https://example.org/plot_data/input.csv")',
                "d = pd.read_csv('<archive>/input.csv')",
                "```",
                "~~~bash",
                'python <repository>/examples/run.py --input="<private>/input.csv"',
                "~~~",
            ]
        )
        self.assertEqual(referenced_paths(sample), set())

    def test_shared_records_do_not_contain_workstation_paths(self) -> None:
        def check(value, path):
            if isinstance(value, dict):
                for item in value.values():
                    check(item, path)
            elif isinstance(value, list):
                for item in value:
                    check(item, path)
            elif isinstance(value, str):
                self.assertIsNone(WORKSTATION_PATH.search(value), str(path))

        directories = [ROOT / "plot_data", ROOT / "provenance"]
        directories.extend((ROOT / "experiments").glob("*/results"))
        for directory in directories:
            for path in directory.rglob("*.json"):
                check(json.loads(path.read_text()), path.relative_to(ROOT))

    def test_tracked_files_exclude_local_environments_and_credentials(self) -> None:
        if not (ROOT / ".git").exists():
            self.skipTest("Git-index check does not apply to source exports")
        completed = subprocess.run(
            ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True,
        )
        for name in completed.stdout.decode().split("\0"):
            if not name:
                continue
            path = Path(name)
            self.assertFalse(set(path.parts) & {".venv", "__pycache__", ".cache"}, name)
            self.assertNotIn(path.name, {".env", "id_rsa", "id_ed25519"}, name)

    def test_workstation_path_detection(self) -> None:
        for value in ("/Users/example/input.json", "/home/example/input.json",
                      "/content/drive/MyDrive/input.json", r"C:\Users\example\input.json"):
            with self.subTest(value=value):
                self.assertIsNotNone(WORKSTATION_PATH.search(value))
        for value in ("input.json", "results/input.json", "https://example.org/data.json"):
            self.assertIsNone(WORKSTATION_PATH.search(value))

    def test_readme_identifies_current_paper(self) -> None:
        for relative in ("README.md", "docs/REPRODUCIBILITY.md"):
            text = (ROOT / relative).read_text(encoding="utf-8")
            unquoted = re.sub(r"^\s*>\s?", "", text, flags=re.MULTILINE)
            normalized = re.sub(r"[\s*_]+", " ", unquoted)
            self.assertIn(CURRENT_PAPER_TITLE, normalized)
            if relative == "README.md":
                self.assertIn("NeurReps 2026", normalized)
                self.assertIn("Symmetry and Geometry in Neural Representations", normalized)
            else:
                self.assertIn(CURRENT_PAPER_VENUE, normalized)
            for author in CURRENT_PAPER_AUTHORS:
                self.assertIn(author, normalized)
        self.assertIn("The paper is not included", (ROOT / "docs/REPRODUCIBILITY.md").read_text())


# Import name -> distribution name where they differ beyond "_" vs "-".
IMPORT_DISTRIBUTIONS = {
    "sklearn": "scikit-learn",
    "google.genai": "google-genai",
    "google.api_core": "google-api-core",
    "yaml": "pyyaml",
    "PIL": "pillow",
    "dotenv": "python-dotenv",
}
# PyTorch advisory GHSA-63cw-57p8-fm3p affects torch <= 2.9.1; the first fixed
# release is 2.10.0. Historical runs used 2.11.0 and the verification environment
# is tested with 2.13.0. Sampled affected versions (including pre-releases):
TORCH_VERSIONS_THROUGH_2_9_1 = (
    "1.13.1", "2.0.0", "2.3.0", "2.4.1", "2.5.0", "2.5.1", "2.6.0", "2.6.0rc1", "2.7.1",
    "2.8.0", "2.9.0", "2.9.0rc1", "2.9.1",
)


def normalized_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_specifiers(path: Path) -> dict[str, str]:
    """Map normalized distribution name -> specifier text for one requirements file."""
    requirements = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].split(";", 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9._-]*)(\[[^\]]*\])?\s*(.*)", line)
        if match is None:
            raise AssertionError(f"{path}: unparsed requirement line {line!r}")
        requirements[normalized_distribution(match.group(1))] = match.group(3).strip()
    return requirements


def third_party_imports(path: Path, local_modules: set[str]) -> set[str]:
    """Normalized distributions imported (anywhere) by one program, excluding stdlib/local."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.update(f"{node.module}.{alias.name}" for alias in node.names)
    distributions = set()
    for module in modules:
        top = module.split(".")[0]
        if top in sys.stdlib_module_names or top in local_modules or top == "__future__":
            continue
        two = ".".join(module.split(".")[:2])
        name = IMPORT_DISTRIBUTIONS.get(two, IMPORT_DISTRIBUTIONS.get(top, top))
        distributions.add(normalized_distribution(name))
    return distributions


def torch_spec_admits_vulnerable_version(specifier: str) -> bool:
    from packaging.specifiers import SpecifierSet

    specifiers = SpecifierSet(specifier)
    return any(
        specifiers.contains(version, prereleases=True) for version in TORCH_VERSIONS_THROUGH_2_9_1
    )


class ExperimentRequirementTests(unittest.TestCase):
    def requirement_files(self) -> list[Path]:
        files = sorted((ROOT / "experiments").glob("*/requirements.txt"))
        self.assertGreaterEqual(len(files), 6)
        return files + [ROOT / "requirements.txt"]

    def test_no_requirement_admits_torch_through_2_9_1(self) -> None:
        for path in self.requirement_files():
            torch = requirement_specifiers(path).get("torch")
            if torch is None:
                continue
            with self.subTest(path=path.relative_to(ROOT).as_posix()):
                self.assertFalse(torch_spec_admits_vulnerable_version(torch), torch or "unpinned")

    def test_torch_rule_rejects_unpinned_and_affected_versions(self) -> None:
        for specifier in ("", ">=2.3", ">=2.5,<3", "==2.5.1", "<2.6", "~=2.4", ">=2.6",
                          "~=2.6", ">=2.9", "==2.9.0", "==2.9.1", ">=2.9.1", ">2.9.0"):
            with self.subTest(specifier=specifier):
                self.assertTrue(torch_spec_admits_vulnerable_version(specifier))
        for specifier in (">=2.10.0", ">=2.10", "==2.10.0", "==2.11.0", "==2.13.0",
                          ">=2.10.0,<3", ">2.9.1"):
            with self.subTest(specifier=specifier):
                self.assertFalse(torch_spec_admits_vulnerable_version(specifier))

    def test_runner_imports_are_declared_in_experiment_requirements(self) -> None:
        # Repository modules (e.g. broad_study/fetch_function_vectors.py, imported by the
        # protocol-swap runner for its pair-digest check) are local, not distributions.
        repository_modules = {path.stem for path in (ROOT / "experiments").rglob("*.py")}
        repository_modules.update({path.stem for path in ROOT.glob("*.py")})
        repository_modules.add("experiments")
        for requirements in sorted((ROOT / "experiments").glob("*/requirements.txt")):
            experiment = requirements.parent
            programs = sorted(
                path for path in experiment.rglob("*.py") if "__pycache__" not in path.parts
            )
            local_modules = repository_modules
            declared = set(requirement_specifiers(requirements))
            with self.subTest(experiment=experiment.name):
                self.assertTrue(programs)
                missing = {
                    f"{program.relative_to(ROOT).as_posix()}: {distribution}"
                    for program in programs
                    for distribution in third_party_imports(program, local_modules)
                    if distribution not in declared
                }
                self.assertEqual(sorted(missing), [])

    def test_import_scan_maps_names_and_skips_stdlib_and_local_modules(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            program = Path(scratch) / "runner.py"
            program.write_text(
                "import os, json\nimport numpy as np\nfrom sklearn.linear_model import Ridge\n"
                "from google import genai\nfrom google.api_core import exceptions\n"
                "import transformer_lens\nfrom helpers import thing\n"
                "def late():\n    import torch\n",
                encoding="utf-8",
            )
            self.assertEqual(
                third_party_imports(program, {"helpers", "runner"}),
                {"numpy", "scikit-learn", "google-genai", "google-api-core", "transformer-lens",
                 "torch"},
            )


if __name__ == "__main__":
    unittest.main()
