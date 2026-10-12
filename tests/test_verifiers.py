"""Record rebuilds and CI configuration; no model or provider execution."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VERIFIERS = (
    "scripts/verify_semantic_rebuild.py",
    "experiments/aperture/source/verify_aperture_records.py",
    "experiments/corrected_coherence/source/verify_corrected_coherence.py",
    "experiments/paired_geometry/source/verify_paired_geometry_records.py",
    "experiments/finite_alternatives/source/verify_finite_alternatives.py",
)


@pytest.mark.parametrize("script", VERIFIERS, ids=lambda path: Path(path).stem)
def test_record_verifier(script):
    environment = os.environ.copy()
    for name in (
        "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "GEMINI_API_KEY", "GH_TOKEN",
        "GITHUB_TOKEN", "GOOGLE_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "OPENAI_API_KEY",
    ):
        environment.pop(name, None)
    environment.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    completed = subprocess.run(
        [sys.executable, script], cwd=ROOT, env=environment, capture_output=True,
        text=True, timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_ci_is_read_only_and_runs_lint_and_tests():
    workflow = (ROOT / ".github/workflows/verify.yml").read_text(encoding="utf-8")
    uses = re.findall(r"^\s*-?\s*uses:\s*(\S+)", workflow, flags=re.MULTILINE)
    assert uses
    assert all(re.search(r"@[0-9a-f]{40}$", action) for action in uses)
    assert "permissions:\n  contents: read\n" in workflow
    assert not re.search(r"\bwrite\b|secrets\.|GITHUB_TOKEN|pull_request_target", workflow)
    assert workflow.count("persist-credentials: false") == sum(
        action.startswith("actions/checkout@") for action in uses
    )
    assert "run: python -m ruff check" in workflow
    assert "run: python -m pytest -q" in workflow
    assert "pip install -r requirements.txt" in workflow
    assert 'python-version: "3.12"' in workflow
    assert not re.search(r"continue-on-error|\|\|\s*true|--ignore|--deselect", workflow)


def test_verifier_programs_do_not_launch_model_or_network_commands():
    forbidden = (
        "run_study.py", "fetch_function_vectors.py", "runner.py", "deepseek_coherence.py",
        "coherence_rescore.py", "aperture_evidence.py", "formula_screening.py",
        "smoke_crosscheck.py", "git push", "gh ", "curl", "wget", "pip install", "--upload",
    )
    source = (ROOT / "scripts/verify_semantic_rebuild.py").read_text(encoding="utf-8")
    assert all(word not in source for word in forbidden)
    assert len(VERIFIERS) == 5
    assert all((ROOT / script).is_file() for script in VERIFIERS)
