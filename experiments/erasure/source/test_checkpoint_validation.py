"""Checkpoint acceptance and rejection using seeded Gaussian run_fold fixtures."""
from __future__ import annotations

import copy
import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


r = _load("runner_checkpoint_validation", "runner.py")
verifier = _load("verify_erasure_outputs_checkpoint_validation", "verify_erasure_outputs.py")

SIGNATURE = "synthetic-signature"
CELL = ("1.5b", "hallucination", "question_grouped", 0)


@pytest.fixture(scope="module")
def synthetic_payload():
    rng = r.np.random.default_rng(20261007)
    pos = rng.normal(size=(100, 32))
    pos[:, 0] += 2.0
    neg = rng.normal(size=(100, 32))
    payload = r.run_fold(pos, neg, *CELL, True)
    payload["run_signature"] = SIGNATURE
    # The resume path reads the JSON written by atomic_write_json.
    return json.loads(json.dumps(payload, allow_nan=False))


def validate(payload):
    r.validate_checkpoint_payload(payload, SIGNATURE, *CELL, True)


def test_synthetic_complete_payload_is_accepted(synthetic_payload):
    validate(synthetic_payload)
    assert len(synthetic_payload["prediction_rows"]) == 40 * len(r.expected_method_names(True))


def test_payload_also_passes_independent_verifier_definitions(synthetic_payload, tmp_path):
    spec = {
        "mode": "smoke", "models": [CELL[0]], "traits": [CELL[1]], "split_schemes": [CELL[2]],
        "folds": [CELL[3]], "inlp_ranks": list(r.INLP_RANKS), "random_ranks": list(r.RANDOM_RANKS),
        "random_replicates": 1,
    }
    rows = synthetic_payload["validation_rows"]
    with (tmp_path / "validation_ledger.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "cell.json").write_text(json.dumps(synthetic_payload))
    manifest = {"run_spec": spec, "run_signature": SIGNATURE, "counts": {"validation_rows": len(rows)}}
    assert verifier.verify_validations(tmp_path, manifest)["failures"] == 0
    assert verifier.verify_checkpoints(tmp_path, manifest)["checkpoints"] == 1


def _first(rows, method):
    return next(i for i, row in enumerate(rows) if row["method"] == method)


def _mutations():
    def missing_prediction_replaced_by_duplicate(p):
        rows = p["prediction_rows"]
        i = _first(rows, "raw")
        rows[i + 1] = copy.deepcopy(rows[i])  # count for "raw" stays 40

    def wrong_cell_prediction(p):
        p["prediction_rows"][0]["trait"] = "evil"

    def prediction_outside_test_role(p):
        rows = p["prediction_rows"]
        i = _first(rows, "raw")
        test = set(int(x) for x in r.make_split_indices(CELL[2], CELL[3])[2])
        outside = next(x for x in range(200) if x not in test)
        base = outside % 100
        rows[i].update(global_index=outside, label=1 if outside < 100 else 0,
                       polarity="positive" if outside < 100 else "negative",
                       sample_index_within_polarity=base, template_id=base // 20, question_id=base % 20)

    def wrong_label(p):
        p["prediction_rows"][0]["label"] = 1 - p["prediction_rows"][0]["label"]

    def wrong_question_lineage(p):
        p["prediction_rows"][0]["question_id"] = (p["prediction_rows"][0]["question_id"] + 1) % 20

    def nonfinite_probability(p):
        p["prediction_rows"][0]["probability_positive"] = float("nan")

    def out_of_range_probability(p):
        p["prediction_rows"][0]["probability_positive"] = 1.5

    def prediction_moved_to_other_method(p):
        # Counts stay 40 per method, but (inlp_r1, t0) is duplicated and
        # (raw, t0) is missing.
        rows = p["prediction_rows"]
        i_raw, i_inlp = _first(rows, "raw"), _first(rows, "inlp_r1")
        rows[i_raw]["method"] = "inlp_r1"
        rows[i_inlp + 1]["method"] = "raw"

    def wrong_cell_method_row(p):
        p["method_rows"][0]["fold"] = 1

    def wrong_role_ids_method_row(p):
        p["method_rows"][0]["test_question_ids"] = "[0, 1, 2, 3]"

    def nonfinite_method_diagnostic(p):
        p["method_rows"][0]["roc_auc"] = float("inf")

    def empty_ledger(p):
        p["validation_rows"] = []

    def missing_ledger(p):
        del p["validation_rows"]

    def partial_ledger(p):
        p["validation_rows"] = p["validation_rows"][:-1]

    def duplicate_ledger_row(p):
        p["validation_rows"][-1] = copy.deepcopy(p["validation_rows"][0])

    def wrong_fold_ledger_row(p):
        p["validation_rows"][0]["fold"] = 2

    def nonfinite_ledger_value(p):
        p["validation_rows"][0]["value"] = float("nan")

    def failing_ledger_row(p):
        p["validation_rows"][0]["pass"] = False

    def ledger_value_above_threshold_marked_pass(p):
        p["validation_rows"][0]["value"] = 1.0

    def loosened_ledger_threshold(p):
        p["validation_rows"][0].update(value=1.0, threshold=10.0)

    def truthy_non_boolean_pass(p):
        p["validation_rows"][0]["pass"] = "yes"

    def wrong_outer_cell(p):
        p["fold"] = 1

    def wrong_signature(p):
        p["run_signature"] = "other"

    return {f.__name__: f for f in (
        missing_prediction_replaced_by_duplicate, wrong_cell_prediction, prediction_outside_test_role,
        wrong_label, wrong_question_lineage, nonfinite_probability, out_of_range_probability,
        prediction_moved_to_other_method, wrong_cell_method_row, wrong_role_ids_method_row,
        nonfinite_method_diagnostic, empty_ledger, missing_ledger, partial_ledger,
        duplicate_ledger_row, wrong_fold_ledger_row, nonfinite_ledger_value, failing_ledger_row,
        ledger_value_above_threshold_marked_pass, loosened_ledger_threshold,
        truthy_non_boolean_pass, wrong_outer_cell, wrong_signature,
    )}


@pytest.mark.parametrize("name", sorted(_mutations()))
def test_invalid_payload_is_rejected(synthetic_payload, name):
    payload = copy.deepcopy(synthetic_payload)
    _mutations()[name](payload)
    with pytest.raises(RuntimeError):
        validate(payload)


def test_prediction_counts_stay_forty_in_count_preserving_mutations(synthetic_payload):
    for name in ("missing_prediction_replaced_by_duplicate", "wrong_cell_prediction",
                 "prediction_outside_test_role"):
        payload = copy.deepcopy(synthetic_payload)
        _mutations()[name](payload)
        counts = {m: sum(row["method"] == m for row in payload["prediction_rows"])
                  for m in r.expected_method_names(True)}
        assert set(counts.values()) == {40}, name


def test_consistently_missing_method_is_rejected(synthetic_payload, tmp_path):
    payload = copy.deepcopy(synthetic_payload)
    for key in ("method_rows", "prediction_rows", "validation_rows"):
        payload[key] = [row for row in payload[key] if row.get("method") != "inlp_r20"]
    with pytest.raises(RuntimeError):
        validate(payload)
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "cell.json").write_text(json.dumps(payload))
    manifest = {
        "run_signature": SIGNATURE,
        "run_spec": {
            "mode": "smoke", "models": [CELL[0]], "traits": [CELL[1]],
            "split_schemes": [CELL[2]], "folds": [CELL[3]],
            "inlp_ranks": list(r.INLP_RANKS), "random_ranks": list(r.RANDOM_RANKS),
            "random_replicates": 1,
        },
    }
    with pytest.raises((AssertionError, RuntimeError, ValueError)):
        verifier.verify_checkpoints(tmp_path, manifest)


def test_historical_placeholder_payload_is_rejected():
    # The guide's negative helper case: correct outer identity, every method,
    # 40 repeated placeholder predictions per method, empty validation ledger.
    methods = r.expected_method_names(True)
    payload = {
        "run_signature": SIGNATURE, "model_key": CELL[0], "trait": CELL[1],
        "split_scheme": CELL[2], "fold": CELL[3],
        "method_rows": [{"method": m} for m in methods],
        "prediction_rows": [{"method": m} for m in methods for _ in range(40)],
        "validation_rows": [],
    }
    with pytest.raises(RuntimeError):
        validate(payload)
