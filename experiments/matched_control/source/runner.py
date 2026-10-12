#!/usr/bin/env python3
"""Factorized complete/control operators, checkpoint loading and paired summaries.

Provides covariance-factor fitting, checkpoint compatibility checks and
question-cluster bootstrap summaries of paired numeric differences."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

MAX_EXTRACTION_BYTES = 4 * 1024**3
MAX_ZIP_COMPRESSION_RATIO = 200


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_extraction_file(
    source: Path,
    *,
    expected_filename: str,
    cache_root: Path = Path(".cache/not_vs_identity_extraction_cache"),
) -> Path:
    """Resolve an exact .pt checkpoint, extracting and verifying ZIP contents."""
    source = source.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if source.suffix.lower() == ".pt":
        if source.name != expected_filename:
            raise ValueError(
                f"Expected checkpoint named {expected_filename}, received {source.name}. "
                "Rename the file or provide the ZIP with the canonical member name; "
                "this guard prevents a same-width checkpoint from being silently misused."
            )
        return source
    if source.suffix.lower() != ".zip":
        raise ValueError(f"Extraction source must be .pt or .zip, got {source}")

    import zipfile

    cache_root.mkdir(parents=True, exist_ok=True)
    if cache_root.is_symlink():
        raise ValueError(f"Extraction cache must not be a symlink: {cache_root}")
    zip_hash = sha256_file(source)
    target_dir = cache_root / zip_hash[:16]
    target_dir.mkdir(parents=True, exist_ok=True)
    if target_dir.is_symlink():
        raise ValueError(f"Extraction directory must not be a symlink: {target_dir}")
    target = target_dir / expected_filename
    marker = target_dir / "source.json"
    if target.is_symlink() or marker.is_symlink():
        raise ValueError(f"Extraction cache contains a symlink: {target_dir}")

    if target.exists() and marker.exists():
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        marker_matches = (
            metadata.get("zip_sha256") == zip_hash
            and Path(str(metadata.get("member", ""))).name == expected_filename
            and metadata.get("extracted_sha256")
        )
        if marker_matches:
            current_hash = sha256_file(target)
            if current_hash == metadata["extracted_sha256"]:
                return target
            print(
                f"[warning] Cached extraction hash mismatch for {target}; re-extracting.",
                file=sys.stderr,
            )

    with zipfile.ZipFile(source) as archive:
        members = [
            name
            for name in archive.namelist()
            if Path(name).name == expected_filename and not name.endswith("/")
        ]
        if len(members) != 1:
            raise ValueError(
                f"Expected exactly one {expected_filename} in {source}, found {members}"
            )
        member = members[0]
        info = archive.getinfo(member)
        ratio = info.file_size / max(1, info.compress_size)
        if not 0 < info.file_size <= MAX_EXTRACTION_BYTES:
            raise ValueError(f"Unsafe checkpoint member size: {info.file_size}")
        if ratio > MAX_ZIP_COMPRESSION_RATIO:
            raise ValueError(f"Unsafe checkpoint compression ratio: {ratio:.1f}")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.stem}.", suffix=".pt.tmp", dir=target_dir
        )
        temporary = Path(temporary_name)
        written = 0
        try:
            with os.fdopen(descriptor, "wb") as target_handle, archive.open(
                info
            ) as source_handle:
                while chunk := source_handle.read(8 * 1024 * 1024):
                    written += len(chunk)
                    if written > info.file_size or written > MAX_EXTRACTION_BYTES:
                        raise ValueError("Checkpoint ZIP member exceeded its declared size")
                    target_handle.write(chunk)
                target_handle.flush()
                os.fsync(target_handle.fileno())
            if written != info.file_size:
                raise ValueError(
                    f"Checkpoint member size mismatch: expected {info.file_size}, got {written}"
                )
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        extracted_hash = sha256_file(target)
        atomic_write_json(
            marker,
            {
                "zip_name": source.name,
                "zip_sha256": zip_hash,
                "member": Path(member).name,
                "extracted_name": target.name,
                "extracted_sha256": extracted_hash,
            },
        )
    return target


def load_extraction_checkpoint(path: Path) -> dict[str, Any]:
    # Fail closed on unsupported PyTorch versions: these files are tensor
    # dictionaries and must never be deserialized through unrestricted pickle.
    if path.stat().st_size > MAX_EXTRACTION_BYTES:
        raise ValueError(f"Extraction checkpoint exceeds {MAX_EXTRACTION_BYTES} bytes")
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict) or "persona" not in data:
        raise ValueError(f"Unexpected checkpoint structure in {path}")
    return data


@dataclass(frozen=True)
class ConceptorFactors:
    mean_cpu: torch.Tensor
    vh_cpu: torch.Tensor
    mu_cpu: torch.Tensor
    aperture: float
    layer: int
    n_samples: int
    hidden_size: int
    centered_energy_retention: float
    mean_cosine_z_notz: float
    min_cosine_z_notz: float
    fixed_norm_match_scale: float
    trace: float
    frobenius_sq: float
    gram_crosscheck_max_abs: float
    gram_crosscheck_relative_l2: float


@torch.inference_mode()
def compute_conceptor_factors(
    positive_activations: torch.Tensor,
    *,
    aperture: float,
    layer: int,
) -> ConceptorFactors:
    """
    Compute the centered conceptor through the exact nonzero SVD factors and
    independently cross-check its action using an n x n Gram-system solve.
    """
    x = positive_activations.detach().to("cpu", dtype=torch.float32)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError(f"Expected [n,d] activations, got {tuple(x.shape)}")
    if not torch.isfinite(x).all():
        raise ValueError("Activation tensor contains NaN or infinity")
    if not aperture > 0:
        raise ValueError("aperture must be positive")

    mean = x.mean(dim=0)
    z = x - mean
    _, singular_values, vh = torch.linalg.svd(z, full_matrices=False)
    lambdas = singular_values.square() / x.shape[0]
    alpha_inv_sq = aperture ** -2
    mu = lambdas / (lambdas + alpha_inv_sq)

    coords = z @ vh.T
    c_z = (coords * mu) @ vh
    not_z = z - c_z

    z_energy = z.square().sum()
    not_energy = not_z.square().sum()
    retention = float((not_energy / z_energy.clamp_min(1e-30)).item())
    cosine = torch.nn.functional.cosine_similarity(z, not_z, dim=-1)
    norm_scale = math.sqrt(1.0 / max(retention, 1e-30))

    # Independent action check using C = Xc^T (Xc Xc^T + n*a I)^-1 Xc.
    generator = torch.Generator(device="cpu").manual_seed(1701 + int(layer))
    probes = torch.randn(3, x.shape[1], generator=generator, dtype=torch.float32)
    svd_action = ((probes @ vh.T) * mu) @ vh
    gram = z @ z.T + (x.shape[0] * alpha_inv_sq) * torch.eye(x.shape[0])
    rhs = z @ probes.T
    gram_action = torch.linalg.solve(gram, rhs).T @ z
    action_delta = svd_action - gram_action
    max_abs = float(action_delta.abs().max().item())
    relative_l2 = float(
        (action_delta.norm() / gram_action.norm().clamp_min(1e-30)).item()
    )
    if max_abs > 2e-5 and relative_l2 > 2e-5:
        raise AssertionError(
            "SVD/Gram conceptor cross-check failed: "
            f"max_abs={max_abs:.3e}, relative_l2={relative_l2:.3e}"
        )

    return ConceptorFactors(
        mean_cpu=mean.contiguous(),
        vh_cpu=vh.contiguous(),
        mu_cpu=mu.contiguous(),
        aperture=float(aperture),
        layer=int(layer),
        n_samples=int(x.shape[0]),
        hidden_size=int(x.shape[1]),
        centered_energy_retention=retention,
        mean_cosine_z_notz=float(cosine.mean().item()),
        min_cosine_z_notz=float(cosine.min().item()),
        fixed_norm_match_scale=float(norm_scale),
        trace=float(mu.sum().item()),
        frobenius_sq=float(mu.square().sum().item()),
        gram_crosscheck_max_abs=max_abs,
        gram_crosscheck_relative_l2=relative_l2,
    )


class MatchedNotIdentityHook:
    """
    Shared arithmetic for the complete map and matched control.

    Identity uses zero conceptor eigenvalues, so the exact same centering,
    low-rank multiplication, beta scaling, all-token behavior, and dtype path are
    executed for NOT and identity.
    """

    def __init__(
        self,
        factors: ConceptorFactors,
        *,
        beta: float,
        method: str,
        model_device: torch.device,
    ) -> None:
        if method not in {"not", "identity", "normmatched_not"}:
            raise ValueError(f"Unsupported method: {method}")
        self.method = method
        self.beta = float(beta)
        self.mean = factors.mean_cpu.to(model_device, dtype=torch.float32)
        self.vh = factors.vh_cpu.to(model_device, dtype=torch.float32)
        if method == "identity":
            self.mu = torch.zeros_like(
                factors.mu_cpu, device=model_device, dtype=torch.float32
            )
            self.norm_scale = 1.0
        else:
            self.mu = factors.mu_cpu.to(model_device, dtype=torch.float32)
            self.norm_scale = (
                factors.fixed_norm_match_scale if method == "normmatched_not" else 1.0
            )

    def __call__(self, h: torch.Tensor, hook: Any) -> torch.Tensor:
        h_float = h.float()
        z = h_float - self.mean
        coords = torch.einsum("bsd,rd->bsr", z, self.vh)
        c_z = torch.einsum("bsr,rd->bsd", coords * self.mu, self.vh)
        transformed = z - c_z
        result = self.mean + (self.beta * self.norm_scale) * transformed
        return result.to(dtype=h.dtype)


def percentile_interval(
    values: np.ndarray,
    *,
    confidence: float,
) -> tuple[float, float]:
    if values.size == 0:
        raise ValueError("Cannot calculate an interval from an empty array")
    alpha = 1.0 - confidence
    return (
        float(np.quantile(values, alpha / 2)),
        float(np.quantile(values, 1 - alpha / 2)),
    )


def cluster_bootstrap_question_means(
    paired_rows: Any,
    outcome_column: str,
    *,
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    question_means = (
        paired_rows.groupby("question_index", sort=True)[outcome_column]
        .mean()
        .dropna()
        .to_numpy(dtype=float)
    )
    if len(question_means) == 0:
        return np.array([], dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(question_means), size=(n_bootstrap, len(question_means)))
    return question_means[indices].mean(axis=1)


def holm_adjust(p_values: Sequence[float | None]) -> list[float | None]:
    valid: list[tuple[int, float]] = []
    for index, value in enumerate(p_values):
        if value is None:
            continue
        numeric = float(value)
        if not np.isfinite(numeric):
            continue
        if not 0.0 <= numeric <= 1.0:
            raise ValueError(f"P-value outside [0, 1] at index {index}: {numeric}")
        valid.append((index, numeric))
    adjusted: list[float | None] = [None] * len(p_values)
    if not valid:
        return adjusted
    ordered = sorted(valid, key=lambda item: item[1])
    running = 0.0
    m = len(ordered)
    for rank, (index, value) in enumerate(ordered):
        candidate = min(1.0, (m - rank) * value)
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def paired_contrast_summary(
    paired: Any,
    *,
    endpoint: str,
    column: str,
    equivalence_margin: float,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> dict[str, Any] | None:
    from scipy.stats import wilcoxon

    valid = paired.dropna(subset=[column]).copy()
    if valid.empty:
        return None
    differences = valid[column].to_numpy(dtype=float)
    question_diffs = (
        valid.groupby("question_index", sort=True)[column]
        .mean()
        .dropna()
        .to_numpy(dtype=float)
    )
    boot = cluster_bootstrap_question_means(
        valid,
        column,
        n_bootstrap=n_bootstrap,
        seed=bootstrap_seed,
    )
    ci95 = percentile_interval(boot, confidence=0.95)
    ci90 = percentile_interval(boot, confidence=0.90)
    if len(question_diffs) > 0 and np.any(question_diffs != 0):
        try:
            result = wilcoxon(
                question_diffs,
                alternative="two-sided",
                zero_method="wilcox",
            )
            wilcoxon_stat = float(result.statistic)
            wilcoxon_p = float(result.pvalue)
        except ValueError:
            wilcoxon_stat = None
            wilcoxon_p = None
    else:
        wilcoxon_stat = 0.0
        wilcoxon_p = 1.0

    # The cluster-level estimate is the primary estimate. With two complete
    # replicates per question it equals the row-level mean, but remains correct
    # if a diagnostic run has unbalanced replicates.
    mean_difference = float(np.mean(question_diffs))
    median_difference = float(np.median(question_diffs))
    within_margin = ci90[0] >= -equivalence_margin and ci90[1] <= equivalence_margin

    return {
        "endpoint": endpoint,
        "n_paired_rows": int(len(valid)),
        "n_question_clusters": int(valid["question_index"].nunique()),
        "mean_difference": mean_difference,
        "row_level_mean_difference": float(np.mean(differences)),
        "median_question_difference": median_difference,
        "ci95_low": ci95[0],
        "ci95_high": ci95[1],
        "ci90_low": ci90[0],
        "ci90_high": ci90[1],
        "equivalence_margin": equivalence_margin,
        "ci90_within_prespecified_margin": bool(within_margin),
        "margin_scope": (
            "Compatibility with a prespecified numerical margin; the margin is not calibrated "
            "to human meaningfulness or judge test-retest variation."
        ),
        "wilcoxon_question_cluster_stat": wilcoxon_stat,
        "wilcoxon_question_cluster_p_nominal": wilcoxon_p,
    }


def _add_holm_family(rows: list[dict[str, Any]], family: str) -> None:
    adjusted = holm_adjust(
        [row.get("wilcoxon_question_cluster_p_nominal") for row in rows]
    )
    for row, value in zip(rows, adjusted):
        row["wilcoxon_holm_within_family"] = value
        row["multiplicity_family"] = family


def _pair_scores(
    frame: Any,
    *,
    left_condition: str,
    right_condition: str,
    view: str,
    endpoint: str,
) -> Any:
    score_column = f"{endpoint}_score"
    pair_keys = ["sample_index", "question_index", "replicate_index", "seed"]
    selected = frame[frame["view"] == view].copy()
    left = selected[selected["condition"] == left_condition].copy()
    right = selected[selected["condition"] == right_condition].copy()
    columns = pair_keys + [
        score_column,
        "hit_max_new_tokens",
        "eos_reached",
        "generated_token_count",
        "response_text_changed_after_prefix200",
        "prefix200_has_additional_tokens",
        "prefix200_text_changed",
    ]
    paired = left[columns].merge(
        right[columns],
        on=pair_keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_left", "_right"),
    )
    paired["difference"] = paired[f"{score_column}_left"] - paired[f"{score_column}_right"]
    paired["either_1024_cap_hit"] = (
        paired["hit_max_new_tokens_left"] | paired["hit_max_new_tokens_right"]
    )
    paired["either_text_changed_after_prefix200"] = (
        paired["response_text_changed_after_prefix200_left"]
        | paired["response_text_changed_after_prefix200_right"]
    )
    paired["either_prefix200_has_continuation"] = (
        paired["prefix200_has_additional_tokens_left"]
        | paired["prefix200_has_additional_tokens_right"]
    )
    paired["either_prefix200_text_changed"] = (
        paired["prefix200_text_changed_left"] | paired["prefix200_text_changed_right"]
    )
    return paired


def _summarize_score_pair(
    paired: Any,
    *,
    endpoint: str,
    equivalence_margin: float,
    n_bootstrap: int,
    bootstrap_seed: int,
    analysis_population: str,
    view: str,
) -> dict[str, Any] | None:
    summary = paired_contrast_summary(
        paired,
        endpoint=endpoint,
        column="difference",
        equivalence_margin=equivalence_margin,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
    )
    if summary is None:
        return None
    left_scores = paired[f"{endpoint}_score_left"].dropna().to_numpy(dtype=float)
    right_scores = paired[f"{endpoint}_score_right"].dropna().to_numpy(dtype=float)
    ceiling_limited = bool(
        left_scores.size
        and right_scores.size
        and (
            (np.all(left_scores == 100.0) and np.all(right_scores == 100.0))
            or (np.all(left_scores == 0.0) and np.all(right_scores == 0.0))
        )
    )
    cap_rate = float(paired["either_1024_cap_hit"].mean()) if len(paired) else None
    prefix_continuation_rate = (
        float(paired["either_prefix200_has_continuation"].mean()) if len(paired) else None
    )
    prefix_text_changed_rate = (
        float(paired["either_prefix200_text_changed"].mean()) if len(paired) else None
    )
    categorical_refusal_pairs_excluded = int(len(paired) - summary["n_paired_rows"])
    if endpoint == "trait" and categorical_refusal_pairs_excluded:
        claim_status = "numeric_trait_contrast_conditional_on_nonrefusal"
    elif ceiling_limited:
        claim_status = "ceiling_limited"
    elif view == "prefix200":
        claim_status = "historical_bounded_prefix_estimand"
    elif analysis_population == "all_rows" and cap_rate is not None and cap_rate >= 0.25:
        claim_status = "censoring_sensitive"
    elif summary["ci90_within_prespecified_margin"]:
        claim_status = "margin_compatible"
    else:
        claim_status = "not_margin_compatible"
    return {
        **summary,
        "analysis_population": analysis_population,
        "view_estimand": (
            "historical_bounded_prefix" if view == "prefix200" else "eos_or_1024_full_response"
        ),
        "either_1024_cap_hit_rate": cap_rate if view == "full" else None,
        "either_prefix200_has_continuation_rate": prefix_continuation_rate,
        "either_prefix200_text_changed_rate": prefix_text_changed_rate,
        "ceiling_limited": ceiling_limited,
        "categorical_refusal_pairs_excluded": categorical_refusal_pairs_excluded,
        "claim_status": claim_status,
    }
