"""Completion metadata reflects the actual per-concept analysis outcome."""

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/semantic_functions/source/per_concept_outcome_analysis.py"
INPUT = ROOT / "provenance/broad_study_results.json"
EXPECTED = ROOT / "experiments/semantic_functions/results/per_concept"


def run_analysis(source, output, *, override=False, fail_validation_write=False):
    command = [sys.executable, str(SCRIPT)]
    if fail_validation_write:
        command = [sys.executable, "-c", """
import runpy
import sys
module = runpy.run_path(sys.argv[1])
original = module['atomic_write_text']
def fail_validation(path, text):
    if path.name == 'VALIDATION.md':
        raise OSError('simulated validation artifact write failure')
    return original(path, text)
module['run'].__globals__['atomic_write_text'] = fail_validation
sys.argv = sys.argv[1:]
module['main']()
""", str(SCRIPT)]
    command += ["--input-json", str(source), "--output-dir", str(output)]
    if override:
        command += ["--expected-sha256", ""]
    environment = os.environ.copy()
    environment.update(MPLBACKEND="Agg", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=180)


def test_changed_canonical_input_has_no_success_manifest(tmp_path):
    data = json.loads(INPUT.read_text())
    data["Qwen2.5-1.5B"]["results"]["semantic_results"]["antonym"]["additive_mean"] = 0.99
    source = tmp_path / "altered.json"
    source.write_text(json.dumps(data))
    output = tmp_path / "analysis"
    completed = run_analysis(source, output, override=True)
    assert completed.returncode != 0
    assert "Canonical correlation cross-check failed" in completed.stderr
    assert not (output / "analysis_manifest.json").exists()
    assert not (output / "VALIDATION.json").exists()
    assert not (output / "VALIDATION.md").exists()
    assert "PASS:" not in completed.stdout


def test_canonical_analysis_completes_without_changing_scientific_tables(tmp_path):
    output = tmp_path / "analysis"
    completed = run_analysis(INPUT, output)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout == f"PASS: {output}\n"
    assert json.loads((output / "analysis_manifest.json").read_text())["status"] == "PASS"
    assert json.loads((output / "VALIDATION.json").read_text())["status"] == "PASS"
    assert (output / "VALIDATION.md").is_file()
    for name in ("semantic_per_concept_outcomes.csv", "persona_per_concept_outcomes.csv", "within_model_inversion_examples.csv"):
        assert (output / name).read_bytes() == (EXPECTED / name).read_bytes()
    with (output / "correlation_summary.csv").open() as stream:
        actual = list(csv.DictReader(stream))
    with (EXPECTED / "correlation_summary.csv").open() as stream:
        expected = list(csv.DictReader(stream))
    assert len(actual) == len(expected)
    for row, reference in zip(actual, expected, strict=True):
        assert row.keys() == reference.keys()
        for field, value in reference.items():
            try:
                number = float(value)
            except ValueError:
                assert row[field] == value
            else:
                assert float(row[field]) == pytest.approx(number, rel=1e-12, abs=1e-12)
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in output.iterdir()}
    repeated = run_analysis(INPUT, output)
    assert repeated.returncode != 0
    assert "Output directory already exists" in repeated.stderr
    assert "PASS:" not in repeated.stdout
    assert before == {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in output.iterdir()}


def test_failed_validation_artifact_write_has_no_success_manifest(tmp_path):
    output = tmp_path / "analysis"
    completed = run_analysis(INPUT, output, fail_validation_write=True)
    assert completed.returncode != 0
    assert "simulated validation artifact write failure" in completed.stderr
    assert not (output / "analysis_manifest.json").exists()
    assert "PASS:" not in completed.stdout
