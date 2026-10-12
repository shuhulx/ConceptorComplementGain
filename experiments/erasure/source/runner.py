#!/usr/bin/env python3
"""Three-way held-out linear accessibility analysis on CPU activation arrays.

Fit an eraser on one split, train a fresh positive-versus-negative linear probe
on a second split, and evaluate recovery on an untouched third split. Includes
INLP, LEACE, conceptor and random-direction comparisons. Probe performance
measures linear accessibility, separately from generated-text behavior.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
import scipy.linalg
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVC

APERTURE = 0.05
ALPHA_INV_SQ = APERTURE ** -2
N_QUESTIONS = 20
N_TEMPLATES = 5
N_ROWS_PER_POLARITY = 100
QUESTION_FOLDS = (
    (0, 1, 15, 17),
    (3, 5, 8, 11),
    (2, 13, 16, 18),
    (4, 9, 12, 19),
    (6, 7, 10, 14),
)
TEMPLATE_FOLDS = ((0,), (1,), (2,), (3,), (4,))
TRAITS = ("sycophancy", "hallucination", "evil", "impolite", "optimistic", "apathetic", "humorous")
MODELS = {'1.5b': {'model_id': 'Qwen/Qwen2.5-1.5B-Instruct', 'layer': 9, 'hidden_size': 1536}, '7b': {'model_id': 'Qwen/Qwen2.5-7B-Instruct', 'layer': 9, 'hidden_size': 3584}, '8b': {'model_id': 'meta-llama/Llama-3.1-8B-Instruct', 'layer': 10, 'hidden_size': 4096}}
INLP_RANKS = (1, 2, 4, 8, 16, 20)
RANDOM_RANKS = (1, 20)
RANDOM_REPLICATES = 3
PROBE_C_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
PROBE_MAX_ITER = 20_000
PROBE_TOL = 1e-8
INLP_C = 1.0
INLP_MAX_ITER = 20_000
INLP_TOL = 1e-7
INLP_SOLVER = "LinearSVC-primal"
ROW_SPACE_SVD_DRIVER = "scipy-gesvd"
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260724


@dataclass(frozen=True)
class ProbeResult:
    selected_c: float
    inner_cv_log_loss: float
    test_log_loss: float
    constant_log_loss: float
    excess_log_loss: float
    balanced_accuracy: float
    roc_auc: float
    brier: float
    n_train: int
    n_test: int
    feature_rank: int
    fit_iterations: int
    convergence_warning: bool
    probability_mean: float
    probability_std: float
    max_abs_probability_minus_prior: float
    coefficient_l2: float
    effective_constant_probe: bool


@dataclass(frozen=True)
class TransformDiagnostics:
    method: str
    actual_rank_removed: int
    normalized_edit_mse_eraser_fit: float
    normalized_edit_mse_probe_fit: float
    normalized_edit_mse_test: float
    energy_retention_eraser_fit: float
    energy_retention_probe_fit: float
    energy_retention_test: float
    mean_cosine_eraser_fit: float
    mean_cosine_probe_fit: float
    mean_cosine_test: float
    mean_norm_ratio_eraser_fit: float
    mean_norm_ratio_probe_fit: float
    mean_norm_ratio_test: float
    class_mean_gap_before_eraser_fit: float
    class_mean_gap_after_eraser_fit: float
    class_mean_gap_before_probe_fit: float
    class_mean_gap_after_probe_fit: float
    class_mean_gap_before_test: float
    class_mean_gap_after_test: float
    crosscov_norm_before_eraser_fit: float
    crosscov_norm_after_eraser_fit: float
    crosscov_norm_before_probe_fit: float
    crosscov_norm_after_probe_fit: float
    crosscov_norm_before_test: float
    crosscov_norm_after_test: float
    idempotence_relative_error: float | None
    basis_orthonormality_error: float | None
    min_linear_eigenvalue: float | None
    max_linear_eigenvalue: float | None
    condition_number_linear_part: float | None
    notes: str


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buf.getvalue())


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    return default if abs(b) <= 1e-30 else float(a / b)


def deterministic_seed(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**32 - 1)


def random_control_seed(model: str, trait: str, scheme: str, fold: int, rank: int, replicate: int) -> int:
    """Return the fixed random-control seed for a supported study setting."""
    key = "/".join(map(str, (model, trait, scheme, fold, rank)))
    seeds = json.loads(Path(__file__).with_name("random_control_seeds.json").read_text())
    if key not in seeds or not _is_int(replicate) or not 0 <= replicate < RANDOM_REPLICATES:
        raise ValueError("unsupported random-control setting")
    return seeds[key][replicate]


def row_metadata() -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(100, dtype=int)
    return idx // 20, idx % 20


def _base_roles(split_scheme: str, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    template_ids, question_ids = row_metadata()
    if split_scheme == "question_grouped":
        folds = QUESTION_FOLDS
        base_groups = question_ids
    elif split_scheme == "template_grouped":
        folds = TEMPLATE_FOLDS
        base_groups = template_ids
    else:
        raise ValueError(split_scheme)
    test_groups = set(folds[fold])
    probe_groups = set(folds[(fold + 1) % 5])
    test_base = np.flatnonzero(np.isin(base_groups, sorted(test_groups)))
    probe_base = np.flatnonzero(np.isin(base_groups, sorted(probe_groups)))
    eraser_base = np.setdiff1d(np.arange(100), np.union1d(test_base, probe_base))
    # Probe hyperparameters are selected by question ID in both split schemes.
    probe_cv_groups = question_ids[probe_base]
    return eraser_base, probe_base, test_base, probe_cv_groups


def make_split_indices(split_scheme: str, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    eraser_base, probe_base, test_base, probe_cv_base = _base_roles(split_scheme, fold)
    eraser = np.concatenate([eraser_base, eraser_base + 100])
    probe = np.concatenate([probe_base, probe_base + 100])
    test = np.concatenate([test_base, test_base + 100])
    probe_groups = np.concatenate([probe_cv_base, probe_cv_base])
    if (len(eraser), len(probe), len(test)) != (120, 40, 40):
        raise AssertionError((split_scheme, fold, len(eraser), len(probe), len(test)))
    if set(eraser) & set(probe) or set(eraser) & set(test) or set(probe) & set(test):
        raise AssertionError("Role overlap")
    if set(np.concatenate([eraser, probe, test])) != set(range(200)):
        raise AssertionError("Incomplete role coverage")
    return eraser, probe, test, probe_groups


def validate_folds() -> dict[str, Any]:
    report: dict[str, Any] = {}
    for scheme in ("question_grouped", "template_grouped"):
        test_seen: list[int] = []
        probe_seen: list[int] = []
        eraser_count = np.zeros(100, dtype=int)
        rows = []
        template_ids, question_ids = row_metadata()
        role_groups = question_ids if scheme == "question_grouped" else template_ids
        for fold in range(5):
            eraser, probe, test, probe_cv = make_split_indices(scheme, fold)
            eb, pb, tb = eraser[eraser < 100], probe[probe < 100], test[test < 100]
            test_seen.extend(tb.tolist())
            probe_seen.extend(pb.tolist())
            eraser_count[eb] += 1
            rows.append({
                "fold": fold,
                "eraser_fit_n": len(eraser),
                "probe_fit_n": len(probe),
                "test_n": len(test),
                "eraser_groups": sorted(set(int(role_groups[i]) for i in eb)),
                "probe_groups": sorted(set(int(role_groups[i]) for i in pb)),
                "test_groups": sorted(set(int(role_groups[i]) for i in tb)),
                "probe_inner_question_groups": sorted(set(int(x) for x in probe_cv)),
            })
        if sorted(test_seen) != list(range(100)) or sorted(probe_seen) != list(range(100)):
            raise AssertionError(f"{scheme} does not rotate test/probe roles exactly once")
        if not np.all(eraser_count == 3):
            raise AssertionError(f"{scheme} eraser role count is not three per base row")
        report[scheme] = rows
    return report


def svd_tol(s: np.ndarray, shape: tuple[int, int]) -> float:
    return 0.0 if not len(s) else float(s[0] * max(shape) * np.finfo(s.dtype).eps)


def _finite_matrix(x: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(x)
    if array.ndim != 2 or 0 in array.shape or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a nonempty real numerical matrix")
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _binary_labels(y: np.ndarray, n_rows: int) -> np.ndarray:
    labels = np.asarray(y)
    if labels.shape != (n_rows,) or labels.dtype.kind not in "biuf":
        raise ValueError("labels must be a numerical vector matching the rows")
    if not np.isfinite(labels).all() or set(np.unique(labels)) != {0, 1}:
        raise ValueError("labels must contain both binary classes 0 and 1")
    return labels.astype(np.float64, copy=False)


def row_space_basis(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = _finite_matrix(x, "x")
    mean = x.mean(0)
    centered = x - mean
    # LAPACK gesvd is slower than gesdd but is substantially more robust for
    # the highly rank-deficient matrices produced late in the INLP path.
    _, s, vt = scipy.linalg.svd(
        centered,
        full_matrices=False,
        check_finite=False,
        lapack_driver="gesvd",
    )
    keep = s > svd_tol(s, centered.shape)
    return mean, vt[keep].T.copy()


def prepare_coordinates(x_fit: np.ndarray, x_apply: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    x_fit = _finite_matrix(x_fit, "x_fit")
    x_apply = _finite_matrix(x_apply, "x_apply")
    if x_fit.shape[1] != x_apply.shape[1]:
        raise ValueError("fitted and applied coordinates must have the same feature dimension")
    mean, basis = row_space_basis(x_fit)
    if basis.shape[1] == 0:
        return np.zeros((len(x_fit), 1)), np.zeros((len(x_apply), 1)), 0
    a = (x_fit - mean) @ basis
    b = (x_apply - mean) @ basis
    cm = a.mean(0)
    cs = a.std(0)
    cs = np.where(cs > 1e-12, cs, 1.0)
    return (a - cm) / cs, (b - cm) / cs, basis.shape[1]


def fit_logistic(x: np.ndarray, y: np.ndarray, c_value: float) -> tuple[LogisticRegression, bool]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model = LogisticRegression(
            C=float(c_value), penalty="l2", solver="liblinear", fit_intercept=True,
            max_iter=PROBE_MAX_ITER, tol=PROBE_TOL, random_state=0,
        )
        model.fit(x, y)
    warned = any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return model, warned


def fit_fresh_probe(
    x_probe: np.ndarray, y_probe: np.ndarray, groups_probe: np.ndarray,
    x_test: np.ndarray, y_test: np.ndarray,
) -> tuple[ProbeResult, np.ndarray, np.ndarray]:
    if set(np.unique(y_probe)) != {0, 1} or set(np.unique(y_test)) != {0, 1}:
        raise ValueError("Both classes required")
    unique_groups = np.unique(groups_probe)
    n_splits = min(4, len(unique_groups))
    if n_splits < 2:
        raise ValueError("Insufficient groups for probe CV")
    splitter = GroupKFold(n_splits=n_splits)
    splits = list(splitter.split(np.zeros(len(y_probe)), groups=groups_probe))
    scored: list[tuple[float, float]] = []
    for c in PROBE_C_GRID:
        losses = []
        for tr, va in splits:
            a, b, _ = prepare_coordinates(x_probe[tr], x_probe[va])
            model, _ = fit_logistic(a, y_probe[tr], c)
            losses.append(log_loss(y_probe[va], model.predict_proba(b)[:, 1], labels=[0, 1]))
        scored.append((float(np.mean(losses)), float(c)))
    best_loss, best_c = min(scored, key=lambda z: (z[0], z[1]))
    a, b, rank = prepare_coordinates(x_probe, x_test)
    model, warned = fit_logistic(a, y_probe, best_c)
    prob = model.predict_proba(b)[:, 1]
    pred = (prob >= 0.5).astype(int)
    prior = float(np.clip(y_probe.mean(), 1e-12, 1 - 1e-12))
    const = np.full(len(y_test), prior)
    test_ll = float(log_loss(y_test, prob, labels=[0, 1]))
    const_ll = float(log_loss(y_test, const, labels=[0, 1]))
    result = ProbeResult(
        selected_c=best_c,
        inner_cv_log_loss=best_loss,
        test_log_loss=test_ll,
        constant_log_loss=const_ll,
        excess_log_loss=test_ll - const_ll,
        balanced_accuracy=float(balanced_accuracy_score(y_test, pred)),
        roc_auc=float(roc_auc_score(y_test, prob)),
        brier=float(brier_score_loss(y_test, prob)),
        n_train=len(y_probe), n_test=len(y_test), feature_rank=rank,
        fit_iterations=int(np.max(model.n_iter_)), convergence_warning=warned,
        probability_mean=float(prob.mean()), probability_std=float(prob.std()),
        max_abs_probability_minus_prior=float(np.max(np.abs(prob - prior))),
        coefficient_l2=float(np.linalg.norm(model.coef_)),
        effective_constant_probe=bool(prob.std() < 1e-8 and np.linalg.norm(model.coef_) < 1e-6),
    )
    return result, prob, pred


def apply_basis_projection(x: np.ndarray, basis_rows: np.ndarray) -> np.ndarray:
    if basis_rows.size == 0:
        return x.copy()
    return x - (x @ basis_rows.T) @ basis_rows


def fit_inlp_bases(
    x: np.ndarray, y: np.ndarray, max_rank: int
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    x = _finite_matrix(x, "x")
    y = _binary_labels(y, len(x))
    current = x.copy()
    learned: list[np.ndarray] = []
    snapshots: dict[int, np.ndarray] = {}
    iteration_rows: list[dict[str, Any]] = []
    for iteration in range(1, max_rank + 1):
        mean, support = row_space_basis(current)
        if support.shape[1] == 0:
            break
        coords = (current - mean) @ support
        # Each eraser fit has 120 rows and at most 119 centered coordinates.
        # The primal optimizer solves the same L2 squared-hinge objective while
        # avoiding the dual solver's observed late-rank convergence failure.
        clf = LinearSVC(
            C=INLP_C, loss="squared_hinge", penalty="l2", dual=False,
            fit_intercept=True, max_iter=INLP_MAX_ITER, tol=INLP_TOL,
            random_state=iteration,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            clf.fit(coords, y)
        convergence_warning = any(
            issubclass(w.category, ConvergenceWarning) for w in caught
        )
        n_iter = int(np.max(np.atleast_1d(clf.n_iter_)))
        if convergence_warning or n_iter >= INLP_MAX_ITER:
            raise RuntimeError(
                f"INLP construction probe did not converge at iteration {iteration}: "
                f"n_iter={n_iter}, warning={convergence_warning}"
            )
        iteration_rows.append({
            "iteration": iteration,
            "n_iter": n_iter,
            "eraser_fit_accuracy": float(clf.score(coords, y)),
        })
        w = support @ clf.coef_.reshape(-1)
        if learned:
            prev = np.stack(learned)
            w = w - prev.T @ (prev @ w)
            w = w - prev.T @ (prev @ w)
        norm = float(np.linalg.norm(w))
        if norm <= 1e-10:
            break
        w /= norm
        learned.append(w)
        current = current - np.outer(current @ w, w)
        if iteration in INLP_RANKS:
            snapshots[iteration] = np.stack(learned).copy()
    if not learned:
        raise RuntimeError("INLP learned no direction")
    missing = [rank for rank in INLP_RANKS if rank <= max_rank and rank not in snapshots]
    if missing:
        raise RuntimeError(
            f"INLP stopped before the prespecified ranks {missing}; "
            f"actual learned rank={len(learned)}"
        )
    return snapshots, {
        "construction_iterations": iteration_rows,
        "max_fit_iterations": max(r["n_iter"] for r in iteration_rows),
    }


def fit_conceptor_not(x_pos: np.ndarray) -> tuple[Any, dict[str, float]]:
    x_pos = _finite_matrix(x_pos, "x_pos")
    mean = x_pos.mean(0)
    centered = x_pos - mean
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    keep = s > svd_tol(s, centered.shape)
    v = vt[keep].T.copy()
    lam = s[keep] ** 2 / len(centered)
    mu = lam / (lam + ALPHA_INV_SQ)
    def transform(x: np.ndarray) -> np.ndarray:
        delta = x - mean
        return x.copy() if not len(mu) else x - ((delta @ v) * mu) @ v.T
    min_eig = float(1 - np.max(mu)) if len(mu) else 1.0
    # I-C has unit gain only in directions outside the fitted covariance support.
    max_eig = 1.0 if len(mu) < x_pos.shape[1] else float(1 - np.min(mu))
    return transform, {
        "rank": 0,
        "conceptor_numerical_rank": len(mu),
        "min_eigenvalue": min_eig,
        "max_eigenvalue": max_eig,
        "condition_number": max_eig / min_eig,
        "scr": safe_divide(float(np.sum(mu**2)), float(np.sum(mu))),
        "notes": "Historical centered positive-trait NOT; invertible at finite aperture",
    }


def fit_leace_binary(x: np.ndarray, y: np.ndarray) -> tuple[Any, dict[str, Any]]:
    x = _finite_matrix(x, "x")
    y = _binary_labels(y, len(x))
    mean = x.mean(0)
    centered = x - mean
    z = y.astype(np.float64) - y.mean()
    n1 = max(len(x) - 1, 1)
    c = centered.T @ z / n1
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    keep = s > svd_tol(s, centered.shape)
    v = vt[keep].T.copy()
    eig = s[keep] ** 2 / n1
    if v.shape[1] == 0 or np.linalg.norm(c) <= 1e-30:
        raise RuntimeError("Degenerate LEACE fit")
    q = v @ ((v.T @ c) / eig)
    denom = float(c @ q)
    if denom <= 1e-20:
        raise RuntimeError(f"Degenerate LEACE denominator {denom}")
    right = q / denom
    def transform(a: np.ndarray) -> np.ndarray:
        return a - np.outer((a - mean) @ right, c)
    return transform, {
        "rank": 1, "mean": mean, "left": c, "right": right,
        "denominator": denom, "feature_rank": v.shape[1],
        "notes": "Exact empirical binary LEACE paper formula",
    }


def random_rowspace_basis(x_fit: np.ndarray, rank: int, seed: int) -> np.ndarray:
    _, support = row_space_basis(x_fit)
    if support.shape[1] < rank:
        raise RuntimeError(
            f"Cannot draw rank-{rank} control from row space of rank "
            f"{support.shape[1]}"
        )
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(
        rng.normal(size=(support.shape[1], rank)), mode="reduced"
    )
    return (support @ q).T.copy()


def class_mean_gap(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.linalg.norm(x[y == 1].mean(0) - x[y == 0].mean(0)))


def crosscov_norm(x: np.ndarray, y: np.ndarray) -> float:
    z = y.astype(float) - y.mean()
    return float(np.linalg.norm((x - x.mean(0)).T @ z / max(len(x) - 1, 1)))


def normalized_edit_mse(before: np.ndarray, after: np.ndarray, reference_mean: np.ndarray) -> float:
    return safe_divide(float(np.sum((after - before) ** 2)), float(np.sum((before - reference_mean) ** 2)))


def energy_retention(before: np.ndarray, after: np.ndarray) -> float:
    return safe_divide(float(np.sum(after**2)), float(np.sum(before**2)))


def mean_cosine(before: np.ndarray, after: np.ndarray) -> float:
    num = np.sum(before * after, axis=1)
    den = np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1)
    vals = np.divide(num, den, out=np.ones_like(num), where=den > 1e-30)
    return float(vals.mean())


def mean_norm_ratio(before: np.ndarray, after: np.ndarray) -> float:
    b = np.linalg.norm(before, axis=1)
    a = np.linalg.norm(after, axis=1)
    return float(np.divide(a, b, out=np.ones_like(a), where=b > 1e-30).mean())


def transform_diagnostics(
    method: str,
    x_e: np.ndarray, x_p: np.ndarray, x_t: np.ndarray,
    y_e: np.ndarray, y_p: np.ndarray, y_t: np.ndarray,
    xe: np.ndarray, xp: np.ndarray, xt: np.ndarray,
    rank: int, idem: float | None, orth: float | None,
    min_eig: float | None, max_eig: float | None, cond: float | None,
    notes: str,
) -> TransformDiagnostics:
    mean = x_e.mean(0)
    return TransformDiagnostics(
        method=method, actual_rank_removed=rank,
        normalized_edit_mse_eraser_fit=normalized_edit_mse(x_e, xe, mean),
        normalized_edit_mse_probe_fit=normalized_edit_mse(x_p, xp, mean),
        normalized_edit_mse_test=normalized_edit_mse(x_t, xt, mean),
        energy_retention_eraser_fit=energy_retention(x_e, xe),
        energy_retention_probe_fit=energy_retention(x_p, xp),
        energy_retention_test=energy_retention(x_t, xt),
        mean_cosine_eraser_fit=mean_cosine(x_e, xe),
        mean_cosine_probe_fit=mean_cosine(x_p, xp),
        mean_cosine_test=mean_cosine(x_t, xt),
        mean_norm_ratio_eraser_fit=mean_norm_ratio(x_e, xe),
        mean_norm_ratio_probe_fit=mean_norm_ratio(x_p, xp),
        mean_norm_ratio_test=mean_norm_ratio(x_t, xt),
        class_mean_gap_before_eraser_fit=class_mean_gap(x_e, y_e),
        class_mean_gap_after_eraser_fit=class_mean_gap(xe, y_e),
        class_mean_gap_before_probe_fit=class_mean_gap(x_p, y_p),
        class_mean_gap_after_probe_fit=class_mean_gap(xp, y_p),
        class_mean_gap_before_test=class_mean_gap(x_t, y_t),
        class_mean_gap_after_test=class_mean_gap(xt, y_t),
        crosscov_norm_before_eraser_fit=crosscov_norm(x_e, y_e),
        crosscov_norm_after_eraser_fit=crosscov_norm(xe, y_e),
        crosscov_norm_before_probe_fit=crosscov_norm(x_p, y_p),
        crosscov_norm_after_probe_fit=crosscov_norm(xp, y_p),
        crosscov_norm_before_test=crosscov_norm(x_t, y_t),
        crosscov_norm_after_test=crosscov_norm(xt, y_t),
        idempotence_relative_error=idem,
        basis_orthonormality_error=orth,
        min_linear_eigenvalue=min_eig,
        max_linear_eigenvalue=max_eig,
        condition_number_linear_part=cond,
        notes=notes,
    )


def validate_leace(transform: Any, info: Mapping[str, Any], x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    z = transform(x)
    zz = transform(z)
    return {
        "idempotence": safe_divide(float(np.linalg.norm(zz-z)), float(np.linalg.norm(z))),
        "mean_gap_ratio": safe_divide(class_mean_gap(z, y), class_mean_gap(x, y)),
        "crosscov_ratio": safe_divide(crosscov_norm(z, y), crosscov_norm(x, y)),
        "right_dot_left_error": abs(float(np.asarray(info["right"]) @ np.asarray(info["left"])) - 1.0),
    }


def validate_projection(basis: np.ndarray) -> tuple[float, float]:
    orth = float(np.max(np.abs(basis @ basis.T - np.eye(len(basis)))))
    rng = np.random.default_rng(12345)
    x = rng.normal(size=(7, basis.shape[1]))
    a = apply_basis_projection(x, basis)
    b = apply_basis_projection(a, basis)
    idem = safe_divide(float(np.linalg.norm(b-a)), float(np.linalg.norm(a)))
    return orth, idem


def build_methods(x_e: np.ndarray, y_e: np.ndarray, model: str, trait: str, scheme: str, fold: int, smoke: bool):
    methods: list[tuple[str, Any, dict[str, Any]]] = [
        ("raw", lambda x: x.copy(), {"rank": 0, "notes": "No transformation"})
    ]
    not_fn, not_info = fit_conceptor_not(x_e[y_e == 1])
    methods.append(("conceptor_not_beta1", not_fn, not_info))
    leace_fn, leace_info = fit_leace_binary(x_e, y_e)
    methods.append(("leace_paper_exact", leace_fn, leace_info))
    snapshots, inlp_fit = fit_inlp_bases(x_e, y_e, max(INLP_RANKS))
    for rank in ((1, 4, 20) if smoke else INLP_RANKS):
        basis = snapshots[rank]
        methods.append((
            f"inlp_r{rank}",
            lambda x, b=basis: apply_basis_projection(x, b),
            {
                "rank": len(basis),
                "basis": basis,
                "inlp_max_fit_iterations": inlp_fit["max_fit_iterations"],
                "notes": (
                    f"INLP fixed rank {len(basis)}; LinearSVC C={INLP_C}, "
                    "primal optimizer (dual=False)"
                ),
            },
        ))
    reps = 1 if smoke else RANDOM_REPLICATES
    for rank in RANDOM_RANKS:
        for rep in range(reps):
            seed = random_control_seed(model, trait, scheme, fold, rank, rep)
            basis = random_rowspace_basis(x_e, rank, seed)
            methods.append((
                f"random_rowspace_r{rank}_s{rep}",
                lambda x, b=basis: apply_basis_projection(x, b),
                {
                    "rank": rank,
                    "basis": basis,
                    "random_seed": seed,
                    "notes": "Frozen random orthogonal projection within eraser-fit row space",
                },
            ))
    return methods


# Validation-ledger and method-diagnostic definitions shared by run_fold and
# validate_checkpoint_payload; they match verify_erasure_outputs.py.
VALIDATION_THRESHOLD = 1e-8
LEACE_VALIDATION_CHECKS = (
    "leace_idempotence", "leace_mean_gap_ratio", "leace_crosscov_ratio", "leace_right_dot_left_error",
)
PROJECTION_VALIDATION_CHECKS = ("projection_orthonormality", "projection_idempotence")
CHECKPOINT_FINITE_METHOD_FIELDS = (
    "test_log_loss", "balanced_accuracy", "roc_auc", "normalized_edit_mse_test",
    "crosscov_norm_after_eraser_fit", "crosscov_norm_after_probe_fit", "crosscov_norm_after_test",
)


def _is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and math.isfinite(value)
    )


def expected_method_names(smoke: bool) -> list[str]:
    inlp_ranks = (1, 4, 20) if smoke else INLP_RANKS
    replicates = 1 if smoke else RANDOM_REPLICATES
    return (
        ["raw", "conceptor_not_beta1", "leace_paper_exact"]
        + [f"inlp_r{rank}" for rank in inlp_ranks]
        + [
            f"random_rowspace_r{rank}_s{rep}"
            for rank in RANDOM_RANKS
            for rep in range(replicates)
        ]
    )


def validate_checkpoint_payload(
    payload: Mapping[str, Any],
    signature: str,
    model: str,
    trait: str,
    scheme: str,
    fold: int,
    smoke: bool,
) -> None:
    """Validate one fold checkpoint's contents before it may be resumed.

    Checks prediction lineage, required validation keys, finite values and
    numerical thresholds. ``verify_erasure_outputs.py`` applies corresponding
    checks across the complete result tables.
    """
    if payload.get("run_signature") != signature:
        raise RuntimeError("Checkpoint run signature mismatch")
    expected_identity = (model, trait, scheme, int(fold))
    identity = (
        payload.get("model_key"), payload.get("trait"),
        payload.get("split_scheme"), payload.get("fold"),
    )
    if not _is_int(identity[3]) or identity != expected_identity:
        raise RuntimeError("Checkpoint cell identity mismatch")
    method_names = expected_method_names(smoke)
    expected = set(method_names)
    method_rows = payload.get("method_rows")
    prediction_rows = payload.get("prediction_rows")
    validation_rows = payload.get("validation_rows")
    if not isinstance(method_rows, list) or not isinstance(prediction_rows, list):
        raise RuntimeError("Checkpoint is missing method or prediction rows")
    if not all(isinstance(row, Mapping) for row in (*method_rows, *prediction_rows)):
        raise RuntimeError("Checkpoint rows must be objects")
    actual = {str(row.get("method")) for row in method_rows}
    if actual != expected or len(method_rows) != len(expected):
        raise RuntimeError(
            f"Checkpoint method universe mismatch: missing={expected-actual}, "
            f"extra={actual-expected}"
        )
    prediction_counts = {
        method: sum(1 for row in prediction_rows if row.get("method") == method)
        for method in expected
    }
    if set(prediction_counts.values()) != {40} or len(prediction_rows) != 40 * len(expected):
        raise RuntimeError(f"Checkpoint prediction counts mismatch: {prediction_counts}")

    eraser, probe, test, _ = make_split_indices(scheme, int(fold))
    templates, questions = row_metadata()
    all_templates = np.concatenate([templates, templates])
    all_questions = np.concatenate([questions, questions])
    role_fields = {
        "n_eraser_fit": len(eraser), "n_probe_fit": len(probe), "n_test": len(test),
        "model_id": MODELS[model]["model_id"], "layer": MODELS[model]["layer"],
    }
    for role, indices in (("eraser_fit", eraser), ("probe_fit", probe), ("test", test)):
        role_fields[f"{role}_question_ids"] = json.dumps(sorted(set(int(all_questions[i]) for i in indices)))
        role_fields[f"{role}_template_ids"] = json.dumps(sorted(set(int(all_templates[i]) for i in indices)))
    for row in method_rows:
        if (
            (row.get("model_key"), row.get("trait"), row.get("split_scheme"), row.get("fold"))
            != expected_identity
            or not _is_int(row.get("fold"))
            or any(row.get(field) != value for field, value in role_fields.items())
        ):
            raise RuntimeError(f"Checkpoint method-row identity mismatch: {row.get('method')}")
        if not all(_is_finite_number(row.get(field)) for field in CHECKPOINT_FINITE_METHOD_FIELDS):
            raise RuntimeError(f"Non-finite checkpoint method diagnostic: {row.get('method')}")

    expected_predictions = {(method, int(index)) for method in method_names for index in test}
    seen_predictions: set[tuple[str, int]] = set()
    for row in prediction_rows:
        global_index = row.get("global_index")
        key = (row.get("method"), global_index)
        if not _is_int(global_index) or key not in expected_predictions:
            raise RuntimeError(f"Checkpoint prediction outside the test role: {key}")
        if key in seen_predictions:
            raise RuntimeError(f"Duplicate checkpoint prediction: {key}")
        seen_predictions.add(key)
        base = global_index % 100
        label = 1 if global_index < 100 else 0
        probability = row.get("probability_positive")
        if (
            (row.get("model_key"), row.get("trait"), row.get("split_scheme"), row.get("fold"))
            != expected_identity
            or not _is_int(row.get("fold"))
            or row.get("layer") != MODELS[model]["layer"]
            or not _is_int(row.get("label")) or row.get("label") != label
            or row.get("polarity") != ("positive" if label == 1 else "negative")
            or row.get("sample_index_within_polarity") != base
            or row.get("template_id") != base // 20
            or row.get("question_id") != base % 20
            or row.get("prediction") not in (0, 1)
            or isinstance(row.get("prediction"), bool)
        ):
            raise RuntimeError(f"Checkpoint prediction lineage mismatch: {key}")
        if not _is_finite_number(probability) or not 0.0 <= probability <= 1.0:
            raise RuntimeError(f"Invalid checkpoint probability: {key}")
    if seen_predictions != expected_predictions:
        raise RuntimeError("Checkpoint prediction universe mismatch")

    if not isinstance(validation_rows, list) or not all(
        isinstance(row, Mapping) for row in validation_rows
    ):
        raise RuntimeError("Checkpoint validation ledger is missing")
    expected_checks = {
        ("leace_paper_exact", check) for check in LEACE_VALIDATION_CHECKS
    } | {
        (method, check)
        for method in method_names
        if method.startswith(("inlp_", "random_rowspace_"))
        for check in PROJECTION_VALIDATION_CHECKS
    }
    seen_checks: set[tuple[Any, Any]] = set()
    for row in validation_rows:
        key = (row.get("method"), row.get("check"))
        if key in seen_checks:
            raise RuntimeError(f"Duplicate checkpoint validation: {key}")
        seen_checks.add(key)
        value, threshold = row.get("value"), row.get("threshold")
        if (
            (row.get("model_key"), row.get("trait"), row.get("split_scheme"), row.get("fold"))
            != expected_identity
            or not _is_int(row.get("fold"))
        ):
            raise RuntimeError(f"Checkpoint validation identity mismatch: {key}")
        if (
            not isinstance(row.get("pass"), (bool, np.bool_))
            or not row.get("pass")
            or not _is_finite_number(value)
            or threshold != VALIDATION_THRESHOLD
            or value > threshold
        ):
            raise RuntimeError(f"Checkpoint validation failed: {key}")
    if seen_checks != expected_checks:
        raise RuntimeError(
            f"Checkpoint validation universe mismatch: "
            f"missing={expected_checks-seen_checks}, extra={seen_checks-expected_checks}"
        )


def run_fold(pos: np.ndarray, neg: np.ndarray, model: str, trait: str, scheme: str, fold: int, smoke: bool) -> dict[str, Any]:
    pos = _finite_matrix(pos, "pos")
    neg = _finite_matrix(neg, "neg")
    if len(pos) != N_ROWS_PER_POLARITY or len(neg) != N_ROWS_PER_POLARITY:
        raise ValueError(f"each polarity must have exactly {N_ROWS_PER_POLARITY} rows")
    if pos.shape[1] != neg.shape[1]:
        raise ValueError("positive and negative rows must have the same feature dimension")
    if model not in MODELS or trait not in TRAITS:
        raise ValueError("unsupported study model or trait")
    if scheme not in ("question_grouped", "template_grouped") or not _is_int(fold) or not 0 <= fold < 5:
        raise ValueError("unsupported split scheme or fold")
    x = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(N_ROWS_PER_POLARITY, int), np.zeros(N_ROWS_PER_POLARITY, int)])
    templates, questions = row_metadata()
    all_templates = np.concatenate([templates, templates])
    all_questions = np.concatenate([questions, questions])
    ei, pi, ti, probe_groups = make_split_indices(scheme, fold)
    x_e, x_p, x_t = x[ei], x[pi], x[ti]
    y_e, y_p, y_t = y[ei], y[pi], y[ti]
    for role, labels in (("eraser", y_e), ("probe", y_p), ("test", y_t)):
        if labels.sum() != len(labels)//2:
            raise AssertionError(f"Unbalanced {role}")
    method_rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    for name, fn, info in build_methods(x_e, y_e, model, trait, scheme, fold, smoke):
        started = time.time()
        xe, xp, xt = map(lambda a: np.asarray(fn(a), dtype=np.float64), (x_e, x_p, x_t))
        for before, after in ((x_e, xe), (x_p, xp), (x_t, xt)):
            if before.shape != after.shape or not np.isfinite(after).all():
                raise RuntimeError(f"Invalid transform output for {name}")
        idem = orth = None
        if name == "leace_paper_exact":
            checks = validate_leace(fn, info, x_e, y_e)
            idem = checks["idempotence"]
            for check, value in checks.items():
                threshold = VALIDATION_THRESHOLD
                validations.append({"check": f"leace_{check}", "method": name, "value": value, "threshold": threshold, "pass": value <= threshold})
        if "basis" in info:
            orth, idem = validate_projection(np.asarray(info["basis"]))
            validations.extend([
                {"check": "projection_orthonormality", "method": name, "value": orth, "threshold": VALIDATION_THRESHOLD, "pass": orth <= VALIDATION_THRESHOLD},
                {"check": "projection_idempotence", "method": name, "value": idem, "threshold": VALIDATION_THRESHOLD, "pass": idem <= VALIDATION_THRESHOLD},
            ])
        probe, prob, pred = fit_fresh_probe(xp, y_p, probe_groups, xt, y_t)
        diag = transform_diagnostics(
            name, x_e, x_p, x_t, y_e, y_p, y_t, xe, xp, xt,
            int(info.get("rank", 0)), idem, orth,
            info.get("min_eigenvalue"), info.get("max_eigenvalue"), info.get("condition_number"),
            str(info.get("notes", "")),
        )
        row = {
            "model_key": model, "model_id": MODELS[model]["model_id"], "trait": trait,
            "layer": MODELS[model]["layer"], "split_scheme": scheme, "fold": fold, "method": name,
            "n_eraser_fit": len(ei), "n_probe_fit": len(pi), "n_test": len(ti),
            "eraser_fit_question_ids": json.dumps(sorted(set(int(all_questions[i]) for i in ei))),
            "probe_fit_question_ids": json.dumps(sorted(set(int(all_questions[i]) for i in pi))),
            "test_question_ids": json.dumps(sorted(set(int(all_questions[i]) for i in ti))),
            "eraser_fit_template_ids": json.dumps(sorted(set(int(all_templates[i]) for i in ei))),
            "probe_fit_template_ids": json.dumps(sorted(set(int(all_templates[i]) for i in pi))),
            "test_template_ids": json.dumps(sorted(set(int(all_templates[i]) for i in ti))),
            "elapsed_seconds": time.time() - started,
            **asdict(probe), **asdict(diag),
        }
        if "scr" in info:
            row["conceptor_scr_eraser_positive"] = info["scr"]
        if "random_seed" in info:
            row["random_seed"] = info["random_seed"]
        if "inlp_max_fit_iterations" in info:
            row["inlp_max_fit_iterations"] = info["inlp_max_fit_iterations"]
        method_rows.append(row)
        for local, global_idx in enumerate(ti):
            base = int(global_idx % 100)
            predictions.append({
                "model_key": model, "trait": trait, "layer": MODELS[model]["layer"],
                "split_scheme": scheme, "fold": fold, "method": name,
                "global_index": int(global_idx), "label": int(y_t[local]),
                "polarity": "positive" if y_t[local] == 1 else "negative",
                "sample_index_within_polarity": base,
                "template_id": int(all_templates[global_idx]), "question_id": int(all_questions[global_idx]),
                "probability_positive": float(prob[local]), "prediction": int(pred[local]),
            })
    for row in validations:
        row.update({"model_key": model, "trait": trait, "split_scheme": scheme, "fold": fold})
    failures = [r for r in validations if not r["pass"]]
    if failures:
        raise RuntimeError(f"Validation failures: {failures}")
    return {"model_key": model, "trait": trait, "split_scheme": scheme, "fold": fold,
            "method_rows": method_rows, "prediction_rows": predictions, "validation_rows": validations}


def fast_binary_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=np.float64)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Both classes are required for binary metrics")
    clipped = np.clip(p, np.finfo(np.float64).eps, 1 - np.finfo(np.float64).eps)
    pred = (p >= 0.5).astype(int)
    prior = float(y.mean())
    prior = float(np.clip(prior, 1e-12, 1 - 1e-12))
    ll = float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped)))
    cll = float(-(y.mean() * np.log(prior) + (1 - y.mean()) * np.log1p(-prior)))
    tpr = float(np.mean(pred[y == 1] == 1))
    tnr = float(np.mean(pred[y == 0] == 0))
    order = np.argsort(p, kind="mergesort")
    sorted_p = p[order]
    ranks = np.arange(1, len(p) + 1, dtype=np.float64)
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_p) != 0) + 1]
    ends = np.r_[starts[1:], len(p)]
    for start, end in zip(starts, ends):
        ranks[start:end] = (start + 1 + end) / 2.0
    unsorted_ranks = np.empty_like(ranks)
    unsorted_ranks[order] = ranks
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    auc = (
        float(unsorted_ranks[y == 1].sum()) - n_pos * (n_pos + 1) / 2
    ) / (n_pos * n_neg)
    return {
        "log_loss": ll,
        "constant_log_loss": cll,
        "excess_log_loss": ll - cll,
        "balanced_accuracy": (tpr + tnr) / 2,
        "roc_auc": float(auc),
        "brier": float(np.mean((p - y) ** 2)),
    }


def metrics_from_predictions(rows: list[dict[str, Any]]) -> dict[str, float]:
    y = np.asarray([r["label"] for r in rows], int)
    p = np.asarray([r["probability_positive"] for r in rows], float)
    return fast_binary_metrics(y, p)


def bootstrap_metrics(rows: list[dict[str, Any]], draws: int, seed: int) -> dict[str, tuple[float, float]]:
    qids = sorted(set(int(r["question_id"]) for r in rows))
    byq = {q: [r for r in rows if int(r["question_id"]) == q] for q in qids}
    rng = np.random.default_rng(seed)
    vals = {k: [] for k in ("log_loss", "excess_log_loss", "balanced_accuracy", "roc_auc", "brier")}
    for _ in range(draws):
        sample: list[dict[str, Any]] = []
        for q in rng.choice(qids, size=len(qids), replace=True):
            sample.extend(byq[int(q)])
        m = metrics_from_predictions(sample)
        for k in vals:
            vals[k].append(m[k])
    return {k: (float(np.quantile(v, .025)), float(np.quantile(v, .975))) for k, v in vals.items()}


def paired_metric_contrast(
    treatment: list[dict[str, Any]],
    reference: list[dict[str, Any]],
    draws: int,
    seed: int,
) -> dict[str, float]:
    order = lambda row: int(row["global_index"])
    treatment = sorted(treatment, key=order)
    reference = sorted(reference, key=order)
    t_keys = [
        (int(r["global_index"]), int(r["label"]), int(r["question_id"]))
        for r in treatment
    ]
    r_keys = [
        (int(r["global_index"]), int(r["label"]), int(r["question_id"]))
        for r in reference
    ]
    if t_keys != r_keys:
        raise RuntimeError("Paired contrast rows are not aligned")
    point_t = metrics_from_predictions(treatment)
    point_r = metrics_from_predictions(reference)
    metric_names = ("log_loss", "excess_log_loss", "balanced_accuracy", "roc_auc", "brier")
    qids = sorted(set(k[2] for k in t_keys))
    t_by_q = {q: [r for r in treatment if int(r["question_id"]) == q] for q in qids}
    r_by_q = {q: [r for r in reference if int(r["question_id"]) == q] for q in qids}
    rng = np.random.default_rng(seed)
    sampled = {name: [] for name in metric_names}
    for _ in range(draws):
        chosen = rng.choice(qids, size=len(qids), replace=True)
        t_rows = [row for q in chosen for row in t_by_q[int(q)]]
        r_rows = [row for q in chosen for row in r_by_q[int(q)]]
        mt = metrics_from_predictions(t_rows)
        mr = metrics_from_predictions(r_rows)
        for name in metric_names:
            sampled[name].append(mt[name] - mr[name])
    out: dict[str, float] = {}
    for name in metric_names:
        values = np.asarray(sampled[name], dtype=np.float64)
        out[f"delta_{name}"] = float(point_t[name] - point_r[name])
        out[f"delta_{name}_ci95_low"] = float(np.quantile(values, 0.025))
        out[f"delta_{name}_ci95_high"] = float(np.quantile(values, 0.975))
    return out


def aggregate(payloads: Sequence[Mapping[str, Any]], output: Path, mode: str) -> dict[str, Any]:
    folds = [r for p in payloads for r in p["method_rows"]]
    preds = [r for p in payloads for r in p["prediction_rows"]]
    val = [r for p in payloads for r in p["validation_rows"]]
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for r in preds:
        groups.setdefault((r["model_key"], r["trait"], r["split_scheme"], r["method"]), []).append(r)
    agg: list[dict[str, Any]] = []
    diag_fields = [
        "actual_rank_removed", "normalized_edit_mse_eraser_fit", "normalized_edit_mse_probe_fit",
        "normalized_edit_mse_test", "energy_retention_eraser_fit", "energy_retention_probe_fit",
        "energy_retention_test", "mean_cosine_eraser_fit", "mean_cosine_probe_fit", "mean_cosine_test",
        "mean_norm_ratio_eraser_fit", "mean_norm_ratio_probe_fit", "mean_norm_ratio_test",
        "class_mean_gap_after_eraser_fit", "class_mean_gap_after_probe_fit", "class_mean_gap_after_test",
        "crosscov_norm_after_eraser_fit", "crosscov_norm_after_probe_fit", "crosscov_norm_after_test",
    ]
    for key, rows in sorted(groups.items()):
        model, trait, scheme, method = key
        m = metrics_from_predictions(rows)
        cis = bootstrap_metrics(rows, 500 if mode == "smoke" else BOOTSTRAP_DRAWS,
                                deterministic_seed(BOOTSTRAP_SEED, *key))
        matching = [r for r in folds if (r["model_key"], r["trait"], r["split_scheme"], r["method"]) == key]
        out = {"model_key": model, "model_id": MODELS[model]["model_id"], "trait": trait,
               "layer": MODELS[model]["layer"], "split_scheme": scheme, "method": method,
               "n_predictions": len(rows), "n_questions": len(set(r["question_id"] for r in rows)), **m}
        for metric, (lo, hi) in cis.items():
            out[f"{metric}_ci95_low"] = lo; out[f"{metric}_ci95_high"] = hi
        for field in diag_fields:
            out[f"mean_{field}"] = float(np.mean([float(r[field]) for r in matching]))
        agg.append(out)
    random_summary: list[dict[str, Any]] = []
    for model in MODELS:
        for trait in TRAITS:
            for scheme in sorted(set(r["split_scheme"] for r in agg)):
                for rank in RANDOM_RANKS:
                    subset = [r for r in agg if r["model_key"] == model and r["trait"] == trait and r["split_scheme"] == scheme and r["method"].startswith(f"random_rowspace_r{rank}_")]
                    for metric in ("log_loss", "excess_log_loss", "balanced_accuracy", "roc_auc", "brier"):
                        if subset:
                            v = np.asarray([r[metric] for r in subset])
                            random_summary.append({"model_key": model, "trait": trait, "split_scheme": scheme,
                                "rank": rank, "metric": metric, "replicates": len(v), "mean": float(v.mean()),
                                "min": float(v.min()), "max": float(v.max()), "std": float(v.std(ddof=1)) if len(v)>1 else 0.0})
    atomic_write_csv(output / "fold_results.csv", folds)
    atomic_write_csv(output / "predictions.csv", preds)
    atomic_write_csv(output / "aggregate_results.csv", agg)
    atomic_write_csv(output / "validation_ledger.csv", val)
    if random_summary:
        atomic_write_csv(output / "random_control_summary.csv", random_summary)
    contrast_rows: list[dict[str, Any]] = []
    draws = 500 if mode == "smoke" else BOOTSTRAP_DRAWS
    cells = sorted({
        (r["model_key"], r["trait"], r["split_scheme"]) for r in preds
    })
    for model, trait, scheme in cells:
        cell_rows = [
            r for r in preds
            if (r["model_key"], r["trait"], r["split_scheme"])
            == (model, trait, scheme)
        ]
        raw_rows = [r for r in cell_rows if r["method"] == "raw"]
        methods = sorted(set(r["method"] for r in cell_rows) - {"raw"})
        for method in methods:
            treatment = [r for r in cell_rows if r["method"] == method]
            contrast_rows.append({
                "model_key": model,
                "trait": trait,
                "split_scheme": scheme,
                "treatment_method": method,
                "reference_method": "raw",
                "n_predictions": len(treatment),
                **paired_metric_contrast(
                    treatment,
                    raw_rows,
                    draws,
                    deterministic_seed(
                        BOOTSTRAP_SEED, model, trait, scheme, method, "vs_raw"
                    ),
                ),
            })
    atomic_write_csv(output / "paired_contrasts_vs_raw.csv", contrast_rows)
    key = build_key_findings(agg, mode)
    atomic_write_json(output / "key_findings.json", key)
    return {
        "fold_rows": len(folds),
        "prediction_rows": len(preds),
        "aggregate_rows": len(agg),
        "validation_rows": len(val),
        "paired_contrast_rows": len(contrast_rows),
        "key_findings": key,
    }


def build_key_findings(rows: Sequence[Mapping[str, Any]], mode: str) -> dict[str, Any]:
    split = "template_grouped" if mode == "template" else "question_grouped"
    selected = [r for r in rows if r["split_scheme"] == split]
    summary: dict[str, Any] = {"mode": mode, "primary_split": split, "method_summary": {}}
    for method in ("raw", "conceptor_not_beta1", "leace_paper_exact", "inlp_r1", "inlp_r20"):
        s = [r for r in selected if r["method"] == method]
        if s:
            summary["method_summary"][method] = {
                "cells": len(s), "mean_balanced_accuracy": float(np.mean([r["balanced_accuracy"] for r in s])),
                "mean_roc_auc": float(np.mean([r["roc_auc"] for r in s])),
                "mean_log_loss": float(np.mean([r["log_loss"] for r in s])),
                "mean_excess_log_loss": float(np.mean([r["excess_log_loss"] for r in s])),
                "mean_rank_removed": float(np.mean([r["mean_actual_rank_removed"] for r in s])),
                "mean_normalized_edit_mse_test": float(np.mean([r["mean_normalized_edit_mse_test"] for r in s])),
            }
    return summary


def synthetic_verification() -> dict[str, Any]:
    rng = np.random.default_rng(20260724)
    d = 32
    direction = np.zeros(d); direction[0] = 2.5
    def sample(groups: int, per_class: int):
        xs=[]; ys=[]; gs=[]
        for g in range(groups):
            for label in (0,1):
                n = rng.normal(size=(per_class,d)); n[:,1] += rng.normal(scale=2,size=per_class)
                xs.append(n + (2*label-1)*direction); ys += [label]*per_class; gs += [g]*per_class
        x=np.concatenate(xs); y=np.asarray(ys,int); gr=np.asarray(gs,int); order=rng.permutation(len(y))
        return x[order],y[order],gr[order]
    xe,ye,_=sample(250,4); xp,yp,gp=sample(100,4); xt,yt,_=sample(100,4)
    raw,_,_=fit_fresh_probe(xp,yp,gp,xt,yt)
    lf,li=fit_leace_binary(xe,ye); lc=validate_leace(lf,li,xe,ye)
    leace,_,_=fit_fresh_probe(lf(xp),yp,gp,lf(xt),yt)
    bases,_=fit_inlp_bases(xe,ye,4)
    inlp,_,_=fit_fresh_probe(apply_basis_projection(xp,bases[4]),yp,gp,apply_basis_projection(xt,bases[4]),yt)
    rb=random_rowspace_basis(xe,1,123)
    rnd,_,_=fit_fresh_probe(apply_basis_projection(xp,rb),yp,gp,apply_basis_projection(xt,rb),yt)
    result={"eraser_fit_rows":len(ye),"probe_fit_rows":len(yp),"test_rows":len(yt),
            "raw_balanced_accuracy":raw.balanced_accuracy,"leace_balanced_accuracy":leace.balanced_accuracy,
            "inlp_r4_balanced_accuracy":inlp.balanced_accuracy,"random_rowspace_r1_balanced_accuracy":rnd.balanced_accuracy,
            "leace_eraser_mean_gap_ratio":lc["mean_gap_ratio"],"leace_eraser_crosscov_ratio":lc["crosscov_ratio"],
            "leace_idempotence":lc["idempotence"]}
    if raw.balanced_accuracy < .95 or leace.balanced_accuracy > .60 or inlp.balanced_accuracy > .60 or rnd.balanced_accuracy < .85:
        raise AssertionError(result)
    if lc["mean_gap_ratio"] > 1e-8 or lc["crosscov_ratio"] > 1e-8 or lc["idempotence"] > 1e-8:
        raise AssertionError(result)
    return result
