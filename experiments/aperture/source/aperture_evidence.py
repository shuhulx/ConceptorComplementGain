#!/usr/bin/env python3
"""Numerical aperture analysis for CPU activation tensors.

The complete map has gain beta / (1 + alpha**2 * lambda) along each
covariance eigendirection. Functions compute spectra, directional gains and
held-out corrections relative to the matched control."""

from __future__ import annotations

import csv
import io
import json
import math
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPOSITORY = Path(__file__).resolve().parents[3]
FROZEN_DIRECTORY = REPOSITORY / "plot_data"
APERTURES = (0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
BASELINE_APERTURE = 0.05
BETA = 2.0
TRAITS = ("sycophancy", "hallucination", "evil", "impolite", "optimistic", "apathetic", "humorous")
MATCHED = {"sycophancy", "hallucination", "optimistic", "apathetic"}
# Zero-based generation layers (blocks.<layer>.hook_resid_pre), hidden width, display name.
MODELS = {
    "1.5b": (9, 1536, "Qwen2.5-1.5B-Instruct"),
    "7b": (9, 3584, "Qwen2.5-7B-Instruct"),
    "8b": (10, 4096, "Llama-3.1-8B-Instruct"),
}
N_ROWS, N_QUESTIONS, N_TEMPLATES, N_FOLDS, FOLD_SEED = 100, 20, 5, 5, 42
OUTPUT_NAMES = (
    "aperture_spectra_21_fits.csv",
    "aperture_correction_12_settings.csv",
    "aperture_correction_fold_evidence.csv",
    "aperture_summary.csv",
    "aperture_baseline_21_fits.csv",
    "aperture_provenance.json",
)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_tensor_tree(data: bytes) -> Any:
    """Load a checkpoint with PyTorch's restricted unpickler; arbitrary objects are refused."""
    return torch.load(io.BytesIO(data), weights_only=True, map_location="cpu")


def centered_covariance(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Center rows by their own sample mean; R = X^T X / n (divisor n, not n - 1)."""
    centered = rows - rows.mean(0)
    return centered, centered.T @ centered / rows.shape[0]


def conceptor_matrix(covariance: torch.Tensor, aperture: float) -> torch.Tensor:
    """Dense C = R (R + alpha^-2 I)^-1 for small synthetic checks."""
    identity = torch.eye(covariance.shape[0], dtype=covariance.dtype)
    return covariance @ torch.linalg.inv(covariance + aperture**-2 * identity)


def conceptor_coefficient(eigenvalue: float, aperture: float) -> float:
    scaled = aperture**2 * eigenvalue
    return scaled / (1 + scaled)


def complete_gain(eigenvalue: float, aperture: float, beta: float = BETA) -> float:
    """Centered gain of F_C along an eigendirection; below one iff alpha^2 lambda > beta - 1."""
    return beta / (1 + aperture**2 * eigenvalue)


def complete_map(h: torch.Tensor, center: torch.Tensor, conceptor: torch.Tensor, beta: float = BETA) -> torch.Tensor:
    identity = torch.eye(conceptor.shape[0], dtype=conceptor.dtype)
    return center + beta * (h - center) @ (identity - conceptor).T


def matched_control_map(h: torch.Tensor, center: torch.Tensor, beta: float = BETA) -> torch.Tensor:
    return center + beta * (h - center)


def question_folds(seed: int = FOLD_SEED) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Historical template-major question folds.

    Row r holds question r % 20 under template r // 20 (question labels tiled
    across five templates). A RandomState(seed) permutation of the 20 question
    indices is cut into five groups of four; each fold holds out every template
    of its four questions (20 rows) and fits on the other 80.
    """
    groups = np.random.RandomState(seed).permutation(N_QUESTIONS).reshape(N_FOLDS, N_QUESTIONS // N_FOLDS)
    question_labels = np.tile(np.arange(N_QUESTIONS), N_TEMPLATES)
    indices = np.arange(N_ROWS)
    folds = [indices[np.isin(question_labels, questions)] for questions in groups]
    if sorted(np.concatenate(folds).tolist()) != list(range(N_ROWS)):
        raise RuntimeError("question folds do not partition the stored rows exactly once")
    return groups, question_labels, folds


def heldout_correction_energies(
    train: torch.Tensor, test: torch.Tensor, apertures: Sequence[float]
) -> tuple[float, dict[float, float], dict[float, float]]:
    """Return sum ||z||^2 and sum ||C z||^2 for held-out z = test - train mean.

    C is fitted on the centered training rows a (n x d) without forming a d x d
    matrix: C z = a^T (a a^T + n alpha^-2 I)^-1 a z. With a a^T = U diag(e) U^T
    and coordinates k = (z a^T) U, ||C z||^2 = sum_j k_j^2 e_j / (e_j + n alpha^-2)^2.
    The second dictionary recomputes the same energies by an explicit dual solve.
    CPU float16/bfloat16/float32 inputs are promoted before centering. Inputs
    already in float64 retain the historical arithmetic and reduction order.
    """
    for name, rows in (("train", train), ("test", test)):
        if not isinstance(rows, torch.Tensor) or rows.device.type != "cpu" or rows.layout != torch.strided:
            raise ValueError(f"{name} must be a strided CPU floating tensor")
        if rows.dtype not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}:
            raise ValueError(f"{name} must be a real floating tensor")
        if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] == 0:
            raise ValueError(f"{name} must have nonempty shape [n, width]")
        if not bool(torch.isfinite(rows).all()):
            raise ValueError(f"{name} must contain only finite values")
    if train.shape[1] != test.shape[1]:
        raise ValueError("train and test must have the same positive width")
    apertures = tuple(apertures)
    for alpha in apertures:
        if isinstance(alpha, bool) or not isinstance(alpha, Real) or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("apertures must be finite and greater than zero")
        try:
            penalty = train.shape[0] / alpha**2
        except (OverflowError, ZeroDivisionError) as error:
            raise ValueError("aperture regularization is outside the float64 arithmetic range") from error
        if not math.isfinite(penalty) or penalty <= 0:
            raise ValueError("aperture regularization is outside the float64 arithmetic range")
    train, test = train.to(dtype=torch.float64), test.to(dtype=torch.float64)
    n = train.shape[0]
    center = train.mean(0)
    a, b = train - center, test - center
    gram = a @ a.T
    eig, directions = torch.linalg.eigh(gram)
    # Allow roundoff relative to the n-by-n eigensystem's scale, retaining the
    # historical absolute allowance for small inputs. This is a conservative
    # numerical guard, not a universal backward-error guarantee.
    roundoff = max(1e-8, 10 * torch.finfo(gram.dtype).eps * n * float(eig.abs().max()))
    if float(eig.min()) < -roundoff:
        raise RuntimeError(f"training Gram matrix is not positive semidefinite (min eigenvalue {float(eig.min())})")
    eig = eig.clamp_min(0)
    cross = b @ a.T
    coordinates = cross @ directions
    input_energy = float(b.square().sum())
    gram_energies, direct_energies = {}, {}
    for alpha in apertures:
        penalty = n / alpha**2
        gram_energies[alpha] = float((coordinates.square() * (eig / (eig + penalty).square())).sum())
        weights = torch.linalg.solve(gram + penalty * torch.eye(n, dtype=gram.dtype), cross.T).T
        direct_energies[alpha] = float((weights @ a).square().sum())
    return input_energy, gram_energies, direct_energies


def rms_ratio(input_energy: float, correction_energy: float, beta: float = BETA) -> float:
    """RMS(beta C z) / RMS((beta - 1) z) from energies already summed over all five folds.

    The estimator pools numerator and denominator energies before the square
    root; it is not the mean of five per-fold ratios.
    """
    return beta / abs(beta - 1) * np.sqrt(correction_energy / input_energy)


def check_records(spectra: list[dict], corrections: list[dict], foldrows: list[dict]) -> None:
    if (len(spectra), len(corrections), len(foldrows)) != (147, 84, 420):
        raise RuntimeError(
            "expected 147 spectral, 84 correction and 420 fold rows; found "
            f"{len(spectra)}, {len(corrections)} and {len(foldrows)}"
        )
    if {(r["model"], r["trait"], r["aperture"]) for r in spectra} != {(m, t, a) for m in MODELS for t in TRAITS for a in APERTURES}:
        raise RuntimeError("spectral rows do not cover every model x target x aperture exactly")
    if {(r["model"], r["trait"], r["aperture"]) for r in corrections} != {(m, t, a) for m in MODELS for t in MATCHED for a in APERTURES}:
        raise RuntimeError("correction rows do not cover every model x matched target x aperture exactly")
    for rows, field, direction in [(spectra, "c_max", 1), (spectra, "minimum_gain_beta2", -1), (corrections, "rms_correction_percent", 1)]:
        for model, trait in {(r["model"], r["trait"]) for r in rows}:
            values = [r[field] for r in sorted(rows, key=lambda r: r["aperture"]) if r["model"] == model and r["trait"] == trait]
            if not all(direction * delta > 0 for delta in np.diff(values)):
                raise RuntimeError(f"{field} is not strictly monotone in the aperture for {model} {trait}")
    for row in spectra:
        if not (0 < row["c_max"] < 1 and 0 < row["minimum_gain_beta2"] < 2):
            raise RuntimeError(f"c_max or minimum gain out of range for {row['model']} {row['trait']} at aperture {row['aperture']}")
    # ||C||_2 < 1 bounds the beta-2 ratio by 200%.
    if not all(0 < r["rms_correction_percent"] < 200 for r in corrections):
        raise RuntimeError("an RMS correction percentage lies outside (0, 200)")


def summarise(spectra: list[dict], corrections: list[dict]) -> list[dict]:
    rows = []
    for alpha in APERTURES:
        row = {"aperture": alpha}
        for prefix, traits in [("all21", set(TRAITS)), ("matched12", MATCHED)]:
            selected = [r for r in spectra if r["aperture"] == alpha and r["trait"] in traits]
            row[f"{prefix}_max_c"] = max(r["c_max"] for r in selected)
            row[f"{prefix}_min_gain"] = min(r["minimum_gain_beta2"] for r in selected)
            row[f"{prefix}_fits_with_any_attenuation"] = sum(r["has_any_attenuated_direction"] for r in selected)
        selected = [r["rms_correction_percent"] for r in corrections if r["aperture"] == alpha]
        row["heldout12_min_correction_percent"] = min(selected)
        row["heldout12_max_correction_percent"] = max(selected)
        rows.append(row)
    return rows


def prepare_output_directory(output: Path, overwrite: bool = False) -> Path:
    """Refuse any directory inside the repository and, unless overwrite is set, existing outputs.

    Paths are resolved (symlinks followed) before the check, so an alias outside
    the repository that points into it is refused as well.
    """
    output = output.resolve()
    if output == FROZEN_DIRECTORY.resolve():
        raise RuntimeError(f"Refusing to write into {FROZEN_DIRECTORY.name}/: aperture_* files there are frozen evidence.")
    repository = REPOSITORY.resolve()
    if output == repository or repository in output.parents:
        raise RuntimeError(
            f"Refusing to write into the repository ({output}); the release tree is "
            "verified by its manifest. Choose an output directory outside the "
            "repository, e.g. under the system temporary directory."
        )
    existing = [name for name in OUTPUT_NAMES if (output / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing outputs without --overwrite: {', '.join(existing)}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def write_outputs(output: Path, spectra: list[dict], corrections: list[dict], foldrows: list[dict],
                  summary: list[dict], provenance: Mapping[str, Any], overwrite: bool = False) -> None:
    mode = "w" if overwrite else "x"
    for name, rows in [("aperture_spectra_21_fits.csv", spectra), ("aperture_correction_12_settings.csv", corrections),
                       ("aperture_correction_fold_evidence.csv", foldrows), ("aperture_summary.csv", summary),
                       ("aperture_baseline_21_fits.csv", [r for r in spectra if r["aperture"] == BASELINE_APERTURE])]:
        with (output / name).open(mode, newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with (output / "aperture_provenance.json").open(mode, encoding="utf-8") as stream:
        stream.write(json.dumps(provenance, indent=2) + "\n")
