"""Module entry points and independent checks of the synthetic gain example."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from conceptor_tools import FittedConceptor, fit_conceptor
from examples import synthetic_complement_gain as example

ROOT = Path(__file__).resolve().parents[1]


def run_script(*arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-B", "-m", "examples.synthetic_complement_gain", *arguments],
        cwd=ROOT, capture_output=True, text=True, check=False, timeout=300,
    )


def test_script_runs_deterministically_without_paper_records() -> None:
    first, second = run_script(), run_script()
    assert first.returncode == second.returncode == 0, first.stdout + first.stderr
    assert first.stdout == second.stdout
    assert "Synthetic geometry" in first.stdout
    assert "3 of 8 directions attenuate" in first.stdout
    assert "paper record" not in first.stdout
    assert run_script("--seed", "7").stdout != first.stdout


@pytest.mark.parametrize("aperture", ["0", "-0.5", "nan", "inf", "-inf"])
def test_cli_rejects_invalid_aperture(aperture: str) -> None:
    completed = run_script(f"--aperture={aperture}")
    assert completed.returncode == 2
    assert "aperture must be finite and greater than zero" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert completed.stdout == ""


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("aperture", [0.05, 0.5])
def test_toy_fit_against_independent_dense_solve_and_gain_threshold(seed: int, aperture: float) -> None:
    train, heldout = example.synthetic_rows(seed)
    fitted = fit_conceptor(train, aperture)
    center = train.mean(0)
    z = train - center
    covariance = z.T @ z / len(train)
    c = torch.linalg.solve(
        covariance + torch.eye(train.shape[1], dtype=torch.float64) / aperture**2, covariance,
    )
    difference = fitted.complete(heldout) - fitted.control(heldout)
    torch.testing.assert_close(difference, -2 * (heldout - center) @ c, atol=1e-12, rtol=1e-12)
    eigs = torch.linalg.eigvalsh(c).flip(0)
    gains = fitted.directional_gains().represented
    torch.testing.assert_close(gains, 2 * (1 - eigs))
    assert torch.equal(gains < 1, fitted.coefficients > 0.5)
    assert bool((gains < 1).any()) == (aperture == 0.5)
    images = fitted.complete(center + fitted.basis) - center
    torch.testing.assert_close((images * fitted.basis).sum(1), gains)


def test_main_returns_nonzero_when_an_operator_is_wrong(monkeypatch) -> None:
    original = FittedConceptor.complete
    monkeypatch.setattr(FittedConceptor, "complete", lambda self, h, beta=2: original(self, h, beta) + 0.001)
    assert example.main([]) == 1
