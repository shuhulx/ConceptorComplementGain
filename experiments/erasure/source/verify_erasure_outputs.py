#!/usr/bin/env python3
"""Numerical and checkpoint validation for three-way erasure results."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import warnings
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

EXPECTED_LAYERS = {"1.5b": 9, "7b": 9, "8b": 10}
METRICS = (
    "log_loss",
    "constant_log_loss",
    "excess_log_loss",
    "balanced_accuracy",
    "roc_auc",
    "brier",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_bool(value: str) -> bool:
    return value.lower() in {"true", "1", "yes"}


def parse_list(value: str) -> set[int]:
    return {int(item) for item in json.loads(value)}


def close(left: Any, right: Any) -> bool:
    return math.isclose(
        float(left), float(right), rel_tol=1e-11, abs_tol=1e-11
    )


def metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    labels = np.asarray([int(row["label"]) for row in rows])
    probabilities = np.asarray(
        [float(row["probability_positive"]) for row in rows]
    )
    clipped = np.clip(
        probabilities,
        np.finfo(np.float64).eps,
        1 - np.finfo(np.float64).eps,
    )
    predictions = (probabilities >= 0.5).astype(int)
    prior = float(np.clip(labels.mean(), 1e-12, 1 - 1e-12))
    ll = float(
        -np.mean(
            labels * np.log(clipped)
            + (1 - labels) * np.log1p(-clipped)
        )
    )
    constant_ll = float(
        -(labels.mean() * np.log(prior)
          + (1 - labels.mean()) * np.log1p(-prior))
    )
    tpr = float(np.mean(predictions[labels == 1] == 1))
    tnr = float(np.mean(predictions[labels == 0] == 0))
    order = np.argsort(probabilities, kind="mergesort")
    sorted_probabilities = probabilities[order]
    ranks = np.arange(1, len(probabilities) + 1, dtype=np.float64)
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_probabilities) != 0) + 1]
    ends = np.r_[starts[1:], len(probabilities)]
    for start, end in zip(starts, ends):
        ranks[start:end] = (start + 1 + end) / 2.0
    unsorted_ranks = np.empty_like(ranks)
    unsorted_ranks[order] = ranks
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    auc = (
        float(unsorted_ranks[labels == 1].sum())
        - n_pos * (n_pos + 1) / 2
    ) / (n_pos * n_neg)
    return {
        "log_loss": ll,
        "constant_log_loss": constant_ll,
        "excess_log_loss": ll - constant_ll,
        "balanced_accuracy": (tpr + tnr) / 2,
        "roc_auc": float(auc),
        "brier": float(np.mean((probabilities - labels) ** 2)),
    }


def expected_methods(spec: Mapping[str, Any]) -> list[str]:
    smoke = spec["mode"] == "smoke"
    inlp_ranks = (1, 4, 20) if smoke else tuple(spec["inlp_ranks"])
    replicates = 1 if smoke else int(spec["random_replicates"])
    return (
        ["raw", "conceptor_not_beta1", "leace_paper_exact"]
        + [f"inlp_r{rank}" for rank in inlp_ranks]
        + [
            f"random_rowspace_r{rank}_s{rep}"
            for rank in spec["random_ranks"]
            for rep in range(replicates)
        ]
    )


def expected_cells(spec: Mapping[str, Any]) -> set[tuple[str, str, str]]:
    return set(product(spec["models"], spec["traits"], spec["split_schemes"]))


def verify_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    spec = manifest["run_spec"]
    if spec.get("inlp_solver") != "LinearSVC-primal" or spec.get("inlp_dual") is not False:
        raise RuntimeError("Wrong INLP construction solver")
    if spec.get("row_space_svd_driver") != "scipy-gesvd":
        raise RuntimeError("Wrong row-space SVD driver")
    if spec.get("layers") != EXPECTED_LAYERS:
        raise RuntimeError("Frozen intervention layers do not match")
    if (
        spec["fold_definition"].get("three_way_cycle")
        != "test=f; probe=(f+1) mod 5; eraser=remaining 3 groups"
    ):
        raise RuntimeError("Wrong three-way role cycle")
    if spec["fold_definition"].get("role_sizes") != {
        "eraser_fit": 120,
        "probe_fit": 40,
        "test": 40,
    }:
        raise RuntimeError("Wrong role sizes")
    return {
        "mode": spec["mode"],
        "expected_methods": len(expected_methods(spec)),
        "expected_cells": len(expected_cells(spec)),
    }


def verify_ledger(root: Path) -> dict[str, Any]:
    ledger = root / "SHA256SUMS.txt"
    covered: set[str] = set()
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise RuntimeError(f"Checksum mismatch: {relative}")
        covered.add(relative)
    reports = {
        "INDEPENDENT_OUTPUT_VERIFICATION.md",
        "INDEPENDENT_OUTPUT_VERIFICATION.json",
    }
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "SHA256SUMS.txt"
        and path.name not in reports
    }
    if covered != actual:
        raise RuntimeError(
            f"Checksum coverage mismatch: missing={actual-covered}, "
            f"extra={covered-actual}"
        )
    return {"entries": len(covered), "ledger_sha256": sha256_file(ledger)}


def prediction_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row["model_key"]),
        str(row["trait"]),
        str(row["split_scheme"]),
        str(row["method"]),
    )


def verify_predictions(
    root: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    spec = manifest["run_spec"]
    methods_expected = expected_methods(spec)
    groups_expected = {
        (*cell, method)
        for cell in expected_cells(spec)
        for method in methods_expected
    }
    predictions = read_csv(root / "predictions.csv")
    aggregates = read_csv(root / "aggregate_results.csv")
    aggregate_map: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in aggregates:
        key = prediction_key(row)
        if key in aggregate_map:
            raise RuntimeError(f"Duplicate aggregate: {key}")
        aggregate_map[key] = row
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    seen: set[tuple[str, str, str, str, int]] = set()
    for row in predictions:
        key4 = prediction_key(row)
        key5 = (*key4, int(row["global_index"]))
        if key5 in seen:
            raise RuntimeError(f"Duplicate prediction: {key5}")
        seen.add(key5)
        grouped[key4].append(row)
        probability = float(row["probability_positive"])
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise RuntimeError(f"Invalid probability: {key5}")
        global_index = int(row["global_index"])
        expected_label = 1 if global_index < 100 else 0
        base = global_index % 100
        if (
            int(row["label"]) != expected_label
            or int(row["sample_index_within_polarity"]) != base
            or int(row["template_id"]) != base // 20
            or int(row["question_id"]) != base % 20
        ):
            raise RuntimeError(f"Prediction lineage mismatch: {key5}")
    if set(grouped) != groups_expected:
        raise RuntimeError(
            f"Prediction group universe mismatch: "
            f"missing={groups_expected-set(grouped)}, "
            f"extra={set(grouped)-groups_expected}"
        )
    if set(aggregate_map) != groups_expected:
        raise RuntimeError("Aggregate group universe mismatch")
    expected_n = 40 if spec["mode"] == "smoke" else 200
    smoke_questions = set(spec["fold_definition"]["question_folds"][0])
    for key, rows in grouped.items():
        indices = {int(row["global_index"]) for row in rows}
        if len(rows) != expected_n or len(indices) != expected_n:
            raise RuntimeError(f"Bad test universe: {key}")
        expected_questions = smoke_questions if spec["mode"] == "smoke" else set(range(20))
        if {int(row["question_id"]) for row in rows} != expected_questions:
            raise RuntimeError(f"Bad question universe: {key}")
        calculated = metrics(rows)
        archived = aggregate_map[key]
        for name, value in calculated.items():
            if not close(value, archived[name]):
                raise RuntimeError(
                    f"Aggregate mismatch: {key}, {name}, "
                    f"{value}, {archived[name]}"
                )
    counts = manifest["counts"]
    if (
        len(predictions) != int(counts["prediction_rows"])
        or len(aggregates) != int(counts["aggregate_rows"])
    ):
        raise RuntimeError("Manifest prediction/aggregate counts are stale")
    return {
        "prediction_rows": len(predictions),
        "groups": len(grouped),
        "metric_discrepancies": 0,
    }


def verify_roles(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    spec = manifest["run_spec"]
    methods = expected_methods(spec)
    expected_keys = {
        (*cell, str(fold), method)
        for cell in expected_cells(spec)
        for fold in spec["folds"]
        for method in methods
    }
    rows = read_csv(root / "fold_results.csv")
    keys: set[tuple[str, str, str, str, str]] = set()
    signatures: dict[tuple[str, str, str, str], tuple[frozenset[int], ...]] = {}
    for row in rows:
        key = (
            row["model_key"],
            row["trait"],
            row["split_scheme"],
            row["fold"],
            row["method"],
        )
        if key in keys:
            raise RuntimeError(f"Duplicate fold result: {key}")
        keys.add(key)
        if (
            int(row["n_eraser_fit"]),
            int(row["n_probe_fit"]),
            int(row["n_test"]),
        ) != (120, 40, 40):
            raise RuntimeError(f"Bad role sizes: {key}")
        eq, pq, tq = map(
            parse_list,
            (
                row["eraser_fit_question_ids"],
                row["probe_fit_question_ids"],
                row["test_question_ids"],
            ),
        )
        et, pt, tt = map(
            parse_list,
            (
                row["eraser_fit_template_ids"],
                row["probe_fit_template_ids"],
                row["test_template_ids"],
            ),
        )
        if row["split_scheme"] == "question_grouped":
            if eq & pq or eq & tq or pq & tq or eq | pq | tq != set(range(20)):
                raise RuntimeError(f"Question leakage: {key}")
        elif row["split_scheme"] == "template_grouped":
            if et & pt or et & tt or pt & tt or et | pt | tt != set(range(5)):
                raise RuntimeError(f"Template leakage: {key}")
        else:
            raise RuntimeError(f"Unknown split: {row['split_scheme']}")
        cell_fold = key[:4]
        signature = (
            frozenset(eq),
            frozenset(pq),
            frozenset(tq),
            frozenset(et),
            frozenset(pt),
            frozenset(tt),
        )
        if cell_fold in signatures and signatures[cell_fold] != signature:
            raise RuntimeError(f"Method role mismatch: {cell_fold}")
        signatures[cell_fold] = signature
        finite_fields = (
            "test_log_loss",
            "balanced_accuracy",
            "roc_auc",
            "normalized_edit_mse_test",
            "crosscov_norm_after_eraser_fit",
            "crosscov_norm_after_probe_fit",
            "crosscov_norm_after_test",
        )
        if any(not math.isfinite(float(row[field])) for field in finite_fields):
            raise RuntimeError(f"Non-finite fold diagnostic: {key}")
        if row["method"] == "leace_paper_exact":
            before = float(row["class_mean_gap_before_eraser_fit"])
            after = float(row["class_mean_gap_after_eraser_fit"])
            if (after / before if before > 0 else 0) > 1e-8:
                raise RuntimeError(f"LEACE fit guardedness failed: {key}")
        if row["method"] == "conceptor_not_beta1":
            if (
                float(row["min_linear_eigenvalue"]) <= 0
                or not close(row["max_linear_eigenvalue"], 1.0)
            ):
                raise RuntimeError(f"NOT eigenvalue bounds failed: {key}")
        if row["method"].startswith("inlp_"):
            if int(float(row["inlp_max_fit_iterations"])) <= 0:
                raise RuntimeError(f"Missing INLP convergence diagnostic: {key}")
    if keys != expected_keys:
        raise RuntimeError(
            f"Fold-result universe mismatch: missing={expected_keys-keys}, "
            f"extra={keys-expected_keys}"
        )
    if len(rows) != int(manifest["counts"]["fold_rows"]):
        raise RuntimeError("Manifest fold count is stale")
    return {
        "fold_rows": len(rows),
        "role_signatures": len(signatures),
    }


def verify_validations(
    root: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    spec = manifest["run_spec"]
    methods = expected_methods(spec)
    basis_methods = [
        method
        for method in methods
        if method.startswith("inlp_") or method.startswith("random_rowspace_")
    ]
    expected: set[tuple[str, str, str, str, str, str]] = set()
    for cell in expected_cells(spec):
        for fold in spec["folds"]:
            for check in (
                "leace_idempotence",
                "leace_mean_gap_ratio",
                "leace_crosscov_ratio",
                "leace_right_dot_left_error",
            ):
                expected.add((*cell, str(fold), "leace_paper_exact", check))
            for method in basis_methods:
                for check in (
                    "projection_orthonormality",
                    "projection_idempotence",
                ):
                    expected.add((*cell, str(fold), method, check))
    rows = read_csv(root / "validation_ledger.csv")
    actual: set[tuple[str, str, str, str, str, str]] = set()
    for row in rows:
        key = (
            row["model_key"],
            row["trait"],
            row["split_scheme"],
            row["fold"],
            row["method"],
            row["check"],
        )
        if key in actual:
            raise RuntimeError(f"Duplicate validation: {key}")
        actual.add(key)
        try:
            threshold = float(row["threshold"])
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid validation threshold: {key}") from exc
        if not math.isfinite(threshold) or threshold != 1e-8:
            raise RuntimeError(f"Validation threshold must equal 1e-8: {key}")
        if (
            not parse_bool(row["pass"])
            or not math.isfinite(float(row["value"]))
            or float(row["value"]) > threshold + 1e-15
        ):
            raise RuntimeError(f"Failed validation: {key}")
    if actual != expected:
        raise RuntimeError(
            f"Validation universe mismatch: missing={expected-actual}, "
            f"extra={actual-expected}"
        )
    if len(rows) != int(manifest["counts"]["validation_rows"]):
        raise RuntimeError("Manifest validation count is stale")
    return {"validation_rows": len(rows), "failures": 0}


def verify_contrasts(
    root: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    spec = manifest["run_spec"]
    methods = expected_methods(spec)
    expected = {
        (*cell, method, "raw")
        for cell in expected_cells(spec)
        for method in methods
        if method != "raw"
    }
    rows = read_csv(root / "paired_contrasts_vs_raw.csv")
    aggregates = {
        prediction_key(row): row
        for row in read_csv(root / "aggregate_results.csv")
    }
    actual: set[tuple[str, str, str, str, str]] = set()
    for row in rows:
        key = (
            row["model_key"],
            row["trait"],
            row["split_scheme"],
            row["treatment_method"],
            row["reference_method"],
        )
        if key in actual:
            raise RuntimeError(f"Duplicate paired contrast: {key}")
        actual.add(key)
        treatment = aggregates[key[:4]]
        reference = aggregates[(*key[:3], key[4])]
        for metric in (
            "log_loss",
            "excess_log_loss",
            "balanced_accuracy",
            "roc_auc",
            "brier",
        ):
            expected_delta = float(treatment[metric]) - float(reference[metric])
            if not close(expected_delta, row[f"delta_{metric}"]):
                raise RuntimeError(f"Paired point contrast mismatch: {key}, {metric}")
            low = float(row[f"delta_{metric}_ci95_low"])
            high = float(row[f"delta_{metric}_ci95_high"])
            if not math.isfinite(low) or not math.isfinite(high) or low > high:
                raise RuntimeError(f"Invalid paired interval: {key}, {metric}")
    if actual != expected:
        raise RuntimeError(
            f"Paired-contrast universe mismatch: missing={expected-actual}, "
            f"extra={actual-expected}"
        )
    if len(rows) != int(manifest["counts"]["paired_contrast_rows"]):
        raise RuntimeError("Manifest paired-contrast count is stale")
    return {"paired_contrast_rows": len(rows)}


def verify_checkpoints(
    root: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    spec = manifest["run_spec"]
    methods = set(expected_methods(spec))
    expected_cells_folds = {
        (*cell, int(fold))
        for cell in expected_cells(spec)
        for fold in spec["folds"]
    }
    paths = sorted((root / "checkpoints").glob("*.json"))
    actual_cells_folds: set[tuple[str, str, str, int]] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        identity = (
            payload["model_key"],
            payload["trait"],
            payload["split_scheme"],
            int(payload["fold"]),
        )
        if identity in actual_cells_folds:
            raise RuntimeError(f"Duplicate checkpoint identity: {identity}")
        actual_cells_folds.add(identity)
        if payload.get("run_signature") != manifest["run_signature"]:
            raise RuntimeError(f"Checkpoint signature mismatch: {path.name}")
        method_rows = payload.get("method_rows", [])
        if (
            {row.get("method") for row in method_rows} != methods
            or len(method_rows) != len(methods)
        ):
            raise RuntimeError(f"Checkpoint method universe mismatch: {path.name}")
        prediction_counts = Counter(
            row.get("method") for row in payload.get("prediction_rows", [])
        )
        if set(prediction_counts) != methods or set(prediction_counts.values()) != {40}:
            raise RuntimeError(f"Checkpoint prediction universe mismatch: {path.name}")
    if actual_cells_folds != expected_cells_folds:
        raise RuntimeError(
            f"Checkpoint universe mismatch: "
            f"missing={expected_cells_folds-actual_cells_folds}, "
            f"extra={actual_cells_folds-expected_cells_folds}"
        )
    return {"checkpoints": len(paths)}


def verify_key_findings(
    root: Path, manifest: Mapping[str, Any], *, allow_legacy_template_summary: bool = False
) -> dict[str, Any]:
    findings = json.loads((root / "key_findings.json").read_text(encoding="utf-8"))
    aggregates = read_csv(root / "aggregate_results.csv")
    mode = manifest["run_spec"]["mode"]
    split = "template_grouped" if mode == "template" else "question_grouped"
    if findings != manifest["counts"]["key_findings"]:
        raise RuntimeError("Key findings disagree with the run manifest")
    if allow_legacy_template_summary and mode == "template" and findings == {
        "mode": "template", "primary_split": "question_grouped", "method_summary": {},
    }:
        warnings.warn(
            "Historical template summary is unavailable; use aggregate_results.csv. "
            "No template key findings were verified.",
            RuntimeWarning,
            stacklevel=2,
        )
        return findings
    if findings.get("mode") != mode or findings.get("primary_split") != split:
        raise RuntimeError("Key-findings mode or split disagrees with the run specification")
    primary = [row for row in aggregates if row["split_scheme"] == split]
    expected_methods_summary = (
        "raw",
        "conceptor_not_beta1",
        "leace_paper_exact",
        "inlp_r1",
        "inlp_r20",
    )
    archived = findings.get("method_summary", {})
    expected_present = {
        method
        for method in expected_methods_summary
        if any(row["method"] == method for row in primary)
    }
    if set(archived) != expected_present:
        raise RuntimeError("Key-findings method universe is stale")
    fields = {
        "mean_balanced_accuracy": "balanced_accuracy",
        "mean_roc_auc": "roc_auc",
        "mean_log_loss": "log_loss",
        "mean_excess_log_loss": "excess_log_loss",
        "mean_rank_removed": "mean_actual_rank_removed",
        "mean_normalized_edit_mse_test": "mean_normalized_edit_mse_test",
    }
    for method in expected_present:
        rows = [row for row in primary if row["method"] == method]
        if int(archived[method]["cells"]) != len(rows):
            raise RuntimeError(f"Stale key-findings cell count: {method}")
        for output_field, aggregate_field in fields.items():
            value = float(np.mean([float(row[aggregate_field]) for row in rows]))
            if not close(value, archived[method][output_field]):
                raise RuntimeError(f"Stale key finding: {method}, {output_field}")
    return findings
