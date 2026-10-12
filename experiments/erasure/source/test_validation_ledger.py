"""Complete validation-ledger thresholds, boundaries and coverage."""

import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("erasure_ledger_verifier", Path(__file__).with_name("verify_erasure_outputs.py"))
assert spec and spec.loader
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


@pytest.fixture
def ledger(tmp_path):
    specification = {
        "mode": "primary", "models": ["1.5b"], "traits": ["evil"],
        "split_schemes": ["question_grouped"], "folds": [0, 1],
        "inlp_ranks": [1, 4, 20], "random_ranks": [1, 20], "random_replicates": 3,
    }
    rows = []
    for fold in specification["folds"]:
        for method in verifier.expected_methods(specification):
            if method == "leace_paper_exact":
                checks = ("leace_idempotence", "leace_mean_gap_ratio", "leace_crosscov_ratio", "leace_right_dot_left_error")
            elif method.startswith(("inlp_", "random_rowspace_")):
                checks = ("projection_orthonormality", "projection_idempotence")
            else:
                continue
            for check in checks:
                rows.append({
                    "model_key": "1.5b", "trait": "evil", "split_scheme": "question_grouped",
                    "fold": fold, "method": method, "check": check,
                    "value": 1e-9, "threshold": 1e-8, "pass": True,
                })
    manifest = {"run_spec": specification, "counts": {"validation_rows": len(rows)}}
    return tmp_path, manifest, rows


def check_ledger(ledger):
    directory, manifest, rows = ledger
    with (directory / "validation_ledger.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    return verifier.verify_validations(directory, manifest)


def test_complete_valid_ledger(ledger):
    assert check_ledger(ledger) == {"validation_rows": 44, "failures": 0}


@pytest.mark.parametrize("threshold", ["nan", "inf", "-inf", "invalid", "", 124.0, 1e-7])
def test_invalid_or_relaxed_threshold_rejected(ledger, threshold):
    for row in ledger[2]:
        row.update(value=123.0, threshold=threshold)
    with pytest.raises(RuntimeError, match="threshold"):
        check_ledger(ledger)


@pytest.mark.parametrize("value,passes", [
    (1e-8, True), (1e-8 + 1e-15, True),
    (np.nextafter(1e-8 + 1e-15, np.inf), False), (123.0, False),
    (np.nan, False), (np.inf, False), (-np.inf, False),
])
def test_validation_error_boundary(ledger, value, passes):
    for row in ledger[2]:
        row["value"] = value
    if passes:
        assert check_ledger(ledger)["failures"] == 0
    else:
        with pytest.raises(RuntimeError, match="Failed validation"):
            check_ledger(ledger)


@pytest.mark.parametrize("mutation", ["pass_flag", "duplicate", "missing", "unexpected", "count"])
def test_other_ledger_guards_remain_active(ledger, mutation):
    _, manifest, rows = ledger
    if mutation == "pass_flag":
        rows[0]["pass"] = False
    elif mutation == "duplicate":
        rows.append(rows[0].copy())
    elif mutation == "missing":
        rows.pop()
    elif mutation == "unexpected":
        rows[0]["check"] = "unknown_check"
    else:
        manifest["counts"]["validation_rows"] += 1
    with pytest.raises(RuntimeError):
        check_ledger(ledger)
