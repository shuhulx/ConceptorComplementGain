#!/usr/bin/env python3
"""Paired targeted-erasure versus matched-random statistics from test predictions.

Intervals resample questions independently within each model-target setting
and average settings with equal weight."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

BOOTSTRAP_SEED_NAMESPACE = "ERASURE-RANDOM-PAIRED-1.0"
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 20260725

EXPECTED_MODELS = ("1.5b", "7b", "8b")
EXPECTED_TRAITS = (
    "sycophancy",
    "hallucination",
    "evil",
    "impolite",
    "optimistic",
    "apathetic",
    "humorous",
)
EXPECTED_RANDOM_R1 = tuple(f"random_rowspace_r1_s{i}" for i in range(3))
EXPECTED_RANDOM_R20 = tuple(f"random_rowspace_r20_s{i}" for i in range(3))
CONTRASTS = {
    "leace_vs_random_r1": {
        "treatment": "leace_paper_exact",
        "random_methods": EXPECTED_RANDOM_R1,
        "expected_rank": 1,
    },
    "inlp20_vs_random_r20": {
        "treatment": "inlp_r20",
        "random_methods": EXPECTED_RANDOM_R20,
        "expected_rank": 20,
    },
}
METRICS = ("log_loss", "balanced_accuracy", "roc_auc")


def stable_seed(*parts: str) -> int:
    payload = "||".join(parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def auc_pairwise(y: np.ndarray, score: np.ndarray) -> float:
    pos = score[y == 1]
    neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("AUROC requires both labels")
    comparisons = []
    for positive in pos:
        for negative in neg:
            comparisons.append(float(positive > negative) + 0.5 * float(positive == negative))
    return float(np.mean(comparisons))


def question_metrics(group: pd.DataFrame) -> pd.Series:
    y = group["label"].to_numpy(dtype=int)
    prediction = group["prediction"].to_numpy(dtype=int)
    probability = np.clip(group["probability_positive"].to_numpy(dtype=float), 1e-12, 1 - 1e-12)
    if sorted(np.unique(y).tolist()) != [0, 1]:
        raise ValueError("Every question cluster must contain both labels")
    log_loss = -np.mean(y * np.log(probability) + (1 - y) * np.log(1 - probability))
    recalls = []
    for label in (0, 1):
        mask = y == label
        recalls.append(float(np.mean(prediction[mask] == y[mask])))
    return pd.Series(
        {
            "log_loss": float(log_loss),
            "balanced_accuracy": float(np.mean(recalls)),
            "roc_auc": auc_pairwise(y, probability),
            "n_rows": int(len(group)),
            "n_positive": int(np.sum(y == 1)),
            "n_negative": int(np.sum(y == 0)),
        }
    )


def bootstrap_equal_cells(
    frame: pd.DataFrame,
    metric: str,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float]:
    cells: list[np.ndarray] = []
    for _, group in frame.groupby(["model_key", "trait"], sort=True):
        values = group.sort_values("question_id")[metric].to_numpy(dtype=float)
        if len(values) != 20:
            raise RuntimeError(f"Expected 20 questions per cell, found {len(values)}")
        cells.append(values)
    point = float(np.mean([values.mean() for values in cells]))
    rng = np.random.default_rng(seed)
    bootstrap_cell_means = []
    for values in cells:
        indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
        bootstrap_cell_means.append(values[indices].mean(axis=1))
    draws = np.vstack(bootstrap_cell_means).mean(axis=0)
    low, high = np.quantile(draws, [0.025, 0.975])
    return point, float(low), float(high)


def bootstrap_single_cell(
    values: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float]:
    if len(values) != 20:
        raise RuntimeError(f"Expected 20 questions, found {len(values)}")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    draws = values[indices].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def build_question_table(predictions: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "model_key",
        "trait",
        "question_id",
        "method",
        "label",
        "prediction",
        "probability_positive",
        "split_scheme",
        "template_id",
        "fold",
    }
    missing = sorted(required_columns - set(predictions.columns))
    if missing:
        raise RuntimeError(f"Missing prediction columns: {missing}")
    if len(predictions) != 63_000:
        raise RuntimeError(f"Expected 63,000 prediction rows, found {len(predictions)}")
    if set(predictions["model_key"].astype(str)) != set(EXPECTED_MODELS):
        raise RuntimeError("Model universe mismatch")
    if set(predictions["trait"]) != set(EXPECTED_TRAITS):
        raise RuntimeError("Trait universe mismatch")
    split_values = predictions["split_scheme"].unique().tolist()
    if len(split_values) != 1:
        raise RuntimeError(f"Expected one split scheme, got {split_values}")
    predictions = predictions.copy()
    grouped = (
        predictions.groupby(["model_key", "trait", "question_id", "method"], sort=True)
        .apply(question_metrics, include_groups=False)
        .reset_index()
    )
    expected_groups = 3 * 7 * 20 * 15
    if len(grouped) != expected_groups:
        raise RuntimeError(f"Expected {expected_groups} question-method groups, found {len(grouped)}")
    if set(grouped["n_rows"]) != {10}:
        raise RuntimeError(f"Expected 10 rows per question-method group, got {set(grouped['n_rows'])}")
    if set(grouped["n_positive"]) != {5} or set(grouped["n_negative"]) != {5}:
        raise RuntimeError("Each question-method group must contain five rows per polarity")
    return grouped


def build_contrast_table(question_metrics_table: pd.DataFrame) -> pd.DataFrame:
    outputs = []
    index_columns = ["model_key", "trait", "question_id"]
    for contrast_name, specification in CONTRASTS.items():
        treatment_method = specification["treatment"]
        random_methods = list(specification["random_methods"])
        treatment = (
            question_metrics_table[question_metrics_table["method"] == treatment_method]
            .set_index(index_columns)[list(METRICS)]
            .rename(columns={metric: f"{metric}_treatment" for metric in METRICS})
        )
        random = (
            question_metrics_table[question_metrics_table["method"].isin(random_methods)]
            .groupby(index_columns, sort=True)[list(METRICS)]
            .mean()
            .rename(columns={metric: f"{metric}_random_mean" for metric in METRICS})
        )
        joined = treatment.join(random, how="inner")
        if len(joined) != 420:
            raise RuntimeError(
                f"{contrast_name}: expected 420 model-trait-question pairs, found {len(joined)}"
            )
        joined = joined.reset_index()
        joined["contrast"] = contrast_name
        joined["treatment_method"] = treatment_method
        joined["random_methods"] = "|".join(random_methods)
        for metric in METRICS:
            joined[f"delta_{metric}"] = (
                joined[f"{metric}_treatment"] - joined[f"{metric}_random_mean"]
            )
        outputs.append(joined)
    return pd.concat(outputs, ignore_index=True)


def summarize_contrasts(question_pairs: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    pooled_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    for contrast_name, contrast_frame in question_pairs.groupby("contrast", sort=True):
        for metric in (f"delta_{value}" for value in METRICS):
            point, low, high = bootstrap_equal_cells(
                contrast_frame,
                metric,
                n_bootstrap=N_BOOTSTRAP,
                seed=stable_seed(
                    BOOTSTRAP_SEED_NAMESPACE, contrast_name, "pooled", metric
                ),
            )
            expected_direction = "positive" if metric == "delta_log_loss" else "negative"
            cell_means = (
                contrast_frame.groupby(["model_key", "trait"], sort=True)[metric].mean()
            )
            expected_sign_count = int(
                np.sum(cell_means.to_numpy() > 0)
                if expected_direction == "positive"
                else np.sum(cell_means.to_numpy() < 0)
            )
            pooled_rows.append(
                {
                    "contrast": contrast_name,
                    "metric": metric.removeprefix("delta_"),
                    "delta_definition": "treatment_minus_mean_matched_random",
                    "expected_erasure_direction": expected_direction,
                    "n_model_trait_cells": 21,
                    "n_question_clusters": 420,
                    "point_estimate": point,
                    "ci95_low": low,
                    "ci95_high": high,
                    "cells_with_expected_sign": expected_sign_count,
                    "cells_total": 21,
                    "n_bootstrap": N_BOOTSTRAP,
                }
            )
        for model_key, model_frame in contrast_frame.groupby("model_key", sort=True):
            for metric in (f"delta_{value}" for value in METRICS):
                point, low, high = bootstrap_equal_cells(
                    model_frame,
                    metric,
                    n_bootstrap=N_BOOTSTRAP,
                    seed=stable_seed(
                        BOOTSTRAP_SEED_NAMESPACE, contrast_name, model_key, metric
                    ),
                )
                model_rows.append(
                    {
                        "contrast": contrast_name,
                        "model_key": model_key,
                        "metric": metric.removeprefix("delta_"),
                        "delta_definition": "treatment_minus_mean_matched_random",
                        "n_traits": 7,
                        "n_question_clusters": 140,
                        "point_estimate": point,
                        "ci95_low": low,
                        "ci95_high": high,
                        "n_bootstrap": N_BOOTSTRAP,
                    }
                )
        for (model_key, trait), cell_frame in contrast_frame.groupby(
            ["model_key", "trait"], sort=True
        ):
            for metric in (f"delta_{value}" for value in METRICS):
                values = cell_frame.sort_values("question_id")[metric].to_numpy(dtype=float)
                point, low, high = bootstrap_single_cell(
                    values,
                    n_bootstrap=N_BOOTSTRAP,
                    seed=stable_seed(
                        BOOTSTRAP_SEED_NAMESPACE,
                        contrast_name,
                        model_key,
                        trait,
                        metric,
                    ),
                )
                cell_rows.append(
                    {
                        "contrast": contrast_name,
                        "model_key": model_key,
                        "trait": trait,
                        "metric": metric.removeprefix("delta_"),
                        "delta_definition": "treatment_minus_mean_matched_random",
                        "n_question_clusters": 20,
                        "point_estimate": point,
                        "ci95_low": low,
                        "ci95_high": high,
                        "n_bootstrap": N_BOOTSTRAP,
                    }
                )
    return pooled_rows, model_rows, cell_rows
