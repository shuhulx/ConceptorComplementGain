"""Run the actual CLI preflight without loading models or contacting a provider."""

from __future__ import annotations

import builtins
import json
import runpy
import sys
from pathlib import Path

import pytest

from experiments.broad_study import fetch_function_vectors

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments/broad_study/run_study.py"


class ModelStackReached(Exception):
    pass


def run_preflight(output, monkeypatch, *, key=None):
    """Supply synthetic local task inputs and stop at the first model import."""
    pairs = [[f"input-{i}", f"output-{i}"] for i in range(32)]
    tasks = {
        name: {"pairs": pairs}
        for name in fetch_function_vectors.EXPECTED_PAIR_SHA256
    }
    monkeypatch.setattr(fetch_function_vectors, "EXPECTED_PAIR_SHA256", {
        name: fetch_function_vectors.pair_sha256(pairs) for name in tasks
    })
    task_path = RUNNER.with_name("function_vectors.json")
    original_is_file, original_read_bytes = Path.is_file, Path.read_bytes
    monkeypatch.setattr(Path, "is_file", lambda path: True if path == task_path else original_is_file(path))
    monkeypatch.setattr(Path, "read_bytes", lambda path: (
        json.dumps({"functions": tasks}).encode() if path == task_path else original_read_bytes(path)
    ))
    if key is None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("GEMINI_API_KEY", key)
    monkeypatch.setattr(sys, "argv", [
        str(RUNNER), "--model", "Qwen/Qwen2.5-1.5B-Instruct",
        "--model-revision", "a" * 40, "--output-dir", str(output),
    ])
    original_import = builtins.__import__

    def import_without_models(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "numpy", "scipy", "matplotlib", "transformer_lens", "google"}:
            raise ModelStackReached(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_models)
    runpy.run_path(str(RUNNER), run_name="__main__")


def completed_run(output):
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True)
    response = {"question": "Question?", "response": "Answer.", "trait_score": None, "coherence": None}
    scored = {"trait_score": 15.5, "coherence": 80.0}
    artifacts = {
        checkpoints / "persona_responses.json": {"responses": {"row": response}},
        checkpoints / "persona_judgments.json": {"scores": {"row": scored}},
        checkpoints / "run_manifest.json": {"run_specification_sha256": "synthetic"},
        output / "Qwen2.5-1.5B-Instruct_scores_broad_study.json": {"row": scored},
        output / "Qwen2.5-1.5B-Instruct_results_broad_study.json": {"study": "synthetic"},
    }
    for path, value in artifacts.items():
        path.write_text(json.dumps(value))


def snapshot(output):
    return {
        path.relative_to(output).as_posix(): path.read_bytes() if path.is_file() else None
        for path in output.rglob("*")
    }


@pytest.mark.parametrize("alias", [False, True])
def test_keyless_resume_refuses_before_model_import_or_writes(tmp_path, monkeypatch, capsys, alias):
    output = tmp_path / "completed"
    completed_run(output)
    before = snapshot(output)
    target = output
    if alias:
        target = tmp_path / "alias"
        target.symlink_to(output, target_is_directory=True)
    with pytest.raises(SystemExit) as error:
        run_preflight(target, monkeypatch)
    assert error.value.code == 2
    assert "persona_judgments.json" in capsys.readouterr().err
    assert snapshot(output) == before


def test_fresh_keyless_run_passes_preflight_without_creating_output(tmp_path, monkeypatch):
    output = tmp_path / "fresh"
    with pytest.raises(ModelStackReached, match="torch"):
        run_preflight(output, monkeypatch)
    assert not output.exists()


def test_keyed_resume_is_not_refused_and_preflight_preserves_artifacts(tmp_path, monkeypatch):
    output = tmp_path / "completed"
    completed_run(output)
    before = snapshot(output)
    with pytest.raises(ModelStackReached, match="torch"):
        run_preflight(output, monkeypatch, key="synthetic-test-key-not-used")
    assert snapshot(output) == before
