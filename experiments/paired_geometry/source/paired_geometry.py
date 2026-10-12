#!/usr/bin/env python3
"""Descriptive spectral metrics for target-present and paired-difference arrays.

Rows are centered with covariance divisor n. The primary estimator follows
float32 arithmetic; float64 SVD supplies an independent cross-check."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[3]
FROZEN_DIRECTORY = REPOSITORY / "plot_data"
ALPHA = 0.05
TRAITS = ("sycophancy", "hallucination", "evil", "impolite", "optimistic", "apathetic", "humorous")
# Zero-based diagnostic layer, hidden width, display name.
MODELS = {
    "1.5b": (26, 1536, "Qwen2.5-1.5B-Instruct"),
    "7b": (26, 3584, "Qwen2.5-7B-Instruct"),
    "8b": (30, 4096, "Llama-3.1-8B-Instruct"),
}
N_ROWS = 100
METRICS = ("scr", "mu1", "mu5", "mu1_mu5", "k90", "eff_rank", "fdr")
# Agreement required of a rebuild against the frozen float32 values. Different
# BLAS/LAPACK builds round float32 eigenvalues differently; k90 must match exactly.
FROZEN_TOLERANCE = 1e-3
FLOAT64_TOLERANCE = 1e-3
OUTPUT_NAMES = ("paired_geometry_main.csv", "paired_geometry_fits.csv", "paired_geometry_provenance.json")


def load_tensor_tree(data: bytes) -> Any:
    """Load a checkpoint with PyTorch's restricted unpickler; arbitrary objects are refused."""
    import torch

    return torch.load(io.BytesIO(data), weights_only=True, map_location="cpu")


def paired_difference(present: np.ndarray, opposed: np.ndarray) -> np.ndarray:
    """Row-aligned target-present minus target-opposed rows; shapes must agree."""
    if present.shape != opposed.shape:
        raise ValueError(f"unpaired rows: {present.shape} vs {opposed.shape}")
    return present - opposed


def covariance_eigenvalues(rows: np.ndarray) -> np.ndarray:
    """Eigenvalues of R = X_c^T X_c / n in decreasing order, in the dtype of ``rows``.

    X_c is centered by its own mean. When n < d the nonzero spectrum is read
    from the Gram matrix X_c X_c^T / n. Round-off negatives are set to zero.
    """
    centered = rows - rows.mean(0)
    n, d = centered.shape
    matrix = centered @ centered.T / n if n < d else centered.T @ centered / n
    return np.sort(np.maximum(np.linalg.eigvalsh(matrix), 0))[::-1]


def covariance_eigenvalues_svd(rows: np.ndarray) -> np.ndarray:
    """Float64 cross-check: squared singular values of the centered rows over n."""
    rows = np.asarray(rows, dtype=np.float64)
    centered = rows - rows.mean(0)
    return np.linalg.svd(centered, compute_uv=False) ** 2 / rows.shape[0]


def conceptor_eigenvalues(eigenvalues: np.ndarray, aperture: float = ALPHA) -> np.ndarray:
    """c = lambda / (lambda + alpha^-2), i.e. the spectrum of C = R (R + alpha^-2 I)^-1."""
    return np.sort(eigenvalues / (eigenvalues + aperture**-2))[::-1]


def effective_rank(conceptor: np.ndarray) -> float:
    """exp of the Shannon entropy of p = c / sum(c); terms with p <= 1e-15 are dropped (0 log 0 = 0)."""
    total = np.sum(conceptor)
    p = conceptor / total if total > 1e-15 else np.ones_like(conceptor) / len(conceptor)
    p = p[p > 1e-15]
    return float(np.exp(-np.sum(p * np.log(p))))


def k90(conceptor: np.ndarray) -> int:
    """Smallest k whose k largest conceptor eigenvalues hold at least 90% of the trace."""
    total = np.sum(conceptor)
    if not total > 0:
        return len(conceptor)
    return int(np.searchsorted(np.cumsum(conceptor), 0.9 * total) + 1)


def spectral_metrics(rows: np.ndarray, aperture: float = ALPHA) -> dict:
    """Per-fit descriptive fields recorded in the frozen JSON, computed in the dtype of ``rows``."""
    raw = covariance_eigenvalues(rows)
    mu = conceptor_eigenvalues(raw, aperture)
    total = np.sum(mu)
    mu1 = float(mu[0]) if len(mu) > 0 else 0
    mu5 = float(mu[4]) if len(mu) > 4 else 0
    raw_total = np.sum(raw)
    return {
        "scr": float(np.sum(mu**2) / total) if total > 1e-15 else 0.0,
        "mu1": mu1,
        "mu5": mu5,
        "mu1_mu5": mu1 / mu5 if mu5 > 1e-12 else float("inf"),
        "k90": k90(mu),
        "eff_rank": effective_rank(mu),
        "fdr": float(raw[0] / raw_total) if raw_total > 1e-15 else 0.0,
        "n_samples": rows.shape[0],
        "n_dims": rows.shape[1],
    }


def float64_metrics(rows: np.ndarray, aperture: float = ALPHA) -> dict:
    mu = conceptor_eigenvalues(covariance_eigenvalues_svd(rows), aperture)
    return {"eff_rank": effective_rank(mu), "k90": k90(mu)}


def check_records(comparisons: list[dict]) -> None:
    if len(comparisons) != 21:
        raise RuntimeError(f"expected 21 model-target comparisons, found {len(comparisons)}")
    if {(r["model_key"], r["target"].lower()) for r in comparisons} != {(m, t) for m in MODELS for t in TRAITS}:
        raise RuntimeError("comparisons do not cover every model x target exactly")
    for row in comparisons:
        for prefix in ("positive", "paired"):
            for field in (f"{prefix}_shannon_effective_rank", f"{prefix}_k90"):
                if not 1 <= row[field] <= N_ROWS - 1:
                    raise RuntimeError(
                        f"{row['model_key']} {row['target']}: {field} = {row[field]} lies outside [1, {N_ROWS - 1}]"
                    )


def prepare_output_directory(output: Path, overwrite: bool = False) -> Path:
    """Refuse any directory inside the repository and, unless overwrite is set, existing outputs.

    Paths are resolved (symlinks followed) before the check, so an alias outside
    the repository that points into it is refused as well.
    """
    output = output.resolve()
    if output == FROZEN_DIRECTORY.resolve():
        raise RuntimeError(f"Refusing to write into {FROZEN_DIRECTORY.name}/: paired geometry files there are frozen evidence.")
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


def write_outputs(output: Path, comparisons: list[dict], fits: list[dict], provenance: Mapping[str, Any],
                  overwrite: bool = False) -> None:
    mode = "w" if overwrite else "x"
    for name, rows in [("paired_geometry_main.csv", comparisons), ("paired_geometry_fits.csv", fits)]:
        with (output / name).open(mode, newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with (output / "paired_geometry_provenance.json").open(mode, encoding="utf-8") as stream:
        stream.write(json.dumps(provenance, indent=2) + "\n")
