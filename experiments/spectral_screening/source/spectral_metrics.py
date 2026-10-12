"""Numerical primitives for the spectral-screening analysis."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from scipy import stats


def validate_run_binding(
    manifest: dict[str, Any],
    checkpoint_identity: object,
    *,
    required_model: str,
    current_materials_sha256: str,
) -> str:
    """Validate that a checkpoint identity is owned by one canonical run manifest."""
    specification = manifest.get("run_specification")
    stored_hash = manifest.get("run_specification_sha256")
    if not isinstance(specification, dict):
        raise RuntimeError("Run manifest is missing its run specification.")
    computed_hash = hashlib.sha256(
        json.dumps(
            specification, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()
    if stored_hash != computed_hash:
        raise RuntimeError("Run manifest contains an invalid specification hash.")
    if specification.get("model") != required_model:
        raise RuntimeError(f"Run manifest model must be {required_model}.")
    if specification.get("experimental_materials_sha256") != current_materials_sha256:
        raise RuntimeError("Run manifest does not match the tracked experimental materials.")
    if checkpoint_identity != stored_hash:
        raise RuntimeError("Activation checkpoint is not owned by the supplied run manifest.")
    return computed_hash


def _normalized_nonnegative_spectrum(values: Sequence[float]) -> np.ndarray:
    spectrum = np.asarray(values, dtype=float)
    if spectrum.ndim != 1 or spectrum.size == 0:
        raise ValueError("spectrum must be a non-empty one-dimensional sequence")
    if not np.isfinite(spectrum).all() or (spectrum < -1e-12).any():
        raise ValueError("spectrum must contain finite nonnegative values")
    spectrum = np.clip(spectrum, 0.0, None)
    total = float(spectrum.sum())
    return spectrum / total if total > 0 else spectrum


def shannon_effective_rank(values: Sequence[float]) -> float:
    """Return exp(Shannon entropy) for a nonnegative spectrum."""
    probabilities = _normalized_nonnegative_spectrum(values)
    positive = probabilities[probabilities > 0]
    if positive.size == 0:
        return 0.0
    return float(np.exp(-np.sum(positive * np.log(positive))))


def participation_ratio_effective_rank(values: Sequence[float]) -> float:
    """Return the Rényi-2/participation-ratio effective rank."""
    probabilities = _normalized_nonnegative_spectrum(values)
    squared_mass = float(np.square(probabilities).sum())
    return 1.0 / squared_mass if squared_mass > 0 else 0.0


def conceptor_eigenvalues(
    activations: torch.Tensor, aperture: float
) -> np.ndarray:
    """Return the full conceptor spectrum without forming a dense d-by-d matrix."""
    if activations.ndim != 2:
        raise ValueError(
            f"activations must have shape (samples, dimensions), got {activations.shape}"
        )
    if activations.shape[0] < 1 or activations.shape[1] < 1:
        raise ValueError("activations must be non-empty")
    if not np.isfinite(aperture) or aperture <= 0:
        raise ValueError(f"aperture must be finite and positive, got {aperture}")

    matrix = activations.detach().to(dtype=torch.float64, device="cpu")
    if not torch.isfinite(matrix).all():
        raise ValueError("activations contain non-finite values")

    centered = matrix - matrix.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    covariance_eigenvalues = singular_values.square() / centered.shape[0]
    alpha_inverse_squared = aperture**-2
    nonzero = covariance_eigenvalues / (
        covariance_eigenvalues + alpha_inverse_squared
    )

    spectrum = torch.zeros(centered.shape[1], dtype=torch.float64)
    spectrum[: nonzero.numel()] = nonzero
    return spectrum.sort(descending=True).values.numpy()


def evaluate_finite_formula(
    name: str,
    model: str,
    concept: str,
    formula: Callable[..., float],
    *arguments: object,
) -> float:
    """Evaluate one registered formula and reject missing or non-finite results."""
    try:
        value = float(formula(*arguments))
    except Exception as error:
        raise RuntimeError(
            f"Formula {name!r} failed for model={model!r}, concept={concept!r}"
        ) from error
    if not np.isfinite(value):
        raise ValueError(
            f"Formula {name!r} returned {value!r} for "
            f"model={model!r}, concept={concept!r}"
        )
    return value


def oriented_two_sided_mann_whitney(
    first: Sequence[float], second: Sequence[float]
) -> tuple[float, float, str]:
    """Return a descriptive larger-family U and a calibrated two-sided p-value."""
    first_array = np.asarray(first, dtype=float)
    second_array = np.asarray(second, dtype=float)
    if first_array.size < 2 or second_array.size < 2:
        raise ValueError("Mann-Whitney comparison requires at least two values per group")
    if not np.isfinite(first_array).all() or not np.isfinite(second_array).all():
        raise ValueError("Mann-Whitney comparison received non-finite values")

    first_u, p_two_sided = stats.mannwhitneyu(
        first_array, second_array, alternative="two-sided"
    )
    maximum_u = float(first_array.size * second_array.size)
    if first_array.mean() >= second_array.mean():
        oriented_u = float(first_u)
        direction = "sem>per"
    else:
        oriented_u = maximum_u - float(first_u)
        direction = "per>sem"
    return oriented_u, float(p_two_sided), direction


def mann_whitney_auto_method(first: Sequence[float], second: Sequence[float]) -> str:
    """Label SciPy's auto method: exact for a small untied sample, otherwise asymptotic."""
    pooled = np.concatenate((np.asarray(first, dtype=float), np.asarray(second, dtype=float)))
    untied = np.unique(pooled).size == pooled.size
    return "exact" if min(len(first), len(second)) <= 8 and untied else "asymptotic"


def is_complete_ordering(first: Sequence[float], second: Sequence[float]) -> bool:
    """Return whether either group lies entirely above the other, with no ties."""
    first_array = np.asarray(first, dtype=float)
    second_array = np.asarray(second, dtype=float)
    if first_array.size == 0 or second_array.size == 0:
        return False
    if not np.isfinite(first_array).all() or not np.isfinite(second_array).all():
        raise ValueError("Complete-ordering check received non-finite values")
    return bool(
        first_array.min() > second_array.max()
        or second_array.min() > first_array.max()
    )


def hedges_g(first: Sequence[float], second: Sequence[float]) -> tuple[float | None, str]:
    """Return signed Hedges' g, or an explicit reason it is undefined."""
    x, y = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or min(x.size, y.size) < 2:
        raise ValueError("Hedges' g requires at least two values per group")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Hedges' g requires finite values")
    df = x.size + y.size - 2
    both_constant = bool(np.all(x == x[0]) and np.all(y == y[0]))
    pooled_sd = 0.0 if both_constant else np.sqrt(
        ((x.size - 1) * x.var(ddof=1) + (y.size - 1) * y.var(ddof=1)) / df
    )
    difference = float(x[0] - y[0]) if both_constant else float(x.mean() - y.mean())
    if pooled_sd == 0:
        status = "equal_constant_groups" if difference == 0 else (
            "zero_variance_positive_difference" if difference > 0
            else "zero_variance_negative_difference"
        )
        return None, status
    value = float((1 - 3 / (4 * df - 1)) * difference / pooled_sd)
    if not np.isfinite(value):
        return None, "nonfinite_standardized_difference"
    return value, "defined"


def spectrum_summary(eigenvalues: Sequence[float]) -> dict[str, float]:
    """Compute the four summary moments from the same full spectrum."""
    spectrum = np.asarray(eigenvalues, dtype=float)
    _normalized_nonnegative_spectrum(spectrum)
    ccr = float(spectrum.mean())
    spr = float(np.square(spectrum).mean())
    return {
        "rawCCR": ccr,
        "SPR": spr,
        "SCR": spr / ccr if ccr > 0 else 0.0,
        "EffRank": participation_ratio_effective_rank(spectrum),
    }


def spectral_variance_edge_fraction(eigenvalues: Sequence[float], n_samples: int) -> float:
    """Fraction above var(c_positive)*(1+sqrt(n/d))², a descriptive heuristic.

    This preserves the historical formula, not an estimated covariance-noise edge.
    The positive-spectrum variance is used only with at least three positive values;
    otherwise its historical fallback is one. n and d come from the actual fit.
    """
    spectrum = np.asarray(eigenvalues, dtype=float)
    _normalized_nonnegative_spectrum(spectrum)
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 1:
        raise ValueError("n_samples must be a positive integer")
    positive = spectrum[spectrum > 1e-15]
    variance = float(positive.var()) if positive.size > 2 else 1.0
    threshold = variance * (1 + np.sqrt(n_samples / spectrum.size)) ** 2
    return float(np.mean(spectrum > threshold))
