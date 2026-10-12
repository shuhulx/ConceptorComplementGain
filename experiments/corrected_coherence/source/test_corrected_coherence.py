"""Corrected-coherence estimators on synthetic fixtures and loaders on included tables.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import corrected_coherence as cc
import numpy as np
import pytest
import verify_corrected_coherence as verifier


def synthetic_table(value: float = 0.0) -> dict[tuple[str, str], list[float]]:
    return {key: [value] * cc.N_QUESTIONS for key in cc.SETTINGS}


def test_synthetic_constant_difference_has_exact_interval() -> None:
    for row in cc.summarize_questions(synthetic_table(2.5)):
        assert (row["n_pairs"], row["n_questions"]) == (40, 20)
        assert (row["mean_difference"], row["ci95_low"], row["ci95_high"]) == (2.5, 2.5, 2.5)


def test_synthetic_zero_difference_has_unit_p_and_holm() -> None:
    for row in cc.summarize_questions(synthetic_table(0.0)):
        assert row["mean_difference"] == 0.0
        assert row["wilcoxon_p_nominal"] == 1.0
        assert row["holm_12_corrected_coherence"] == 1.0


def test_synthetic_bootstrap_matches_frozen_rng_call_order() -> None:
    values = np.array([-5.0, 0.0, 2.5, 10.0, -37.5] * 4)
    seed = cc.bootstrap_seed("hallucination")
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, 20, size=(cc.BOOTSTRAP_DRAWS, 20))].mean(axis=1)
    expected = tuple(float(x) for x in np.quantile(draws, [0.025, 0.975]))
    assert cc.question_bootstrap_ci(values, seed) == expected


def test_bootstrap_seed_depends_on_trait_only() -> None:
    assert cc.bootstrap_seed("apathetic") == 3561
    assert cc.bootstrap_seed("hallucination") == 4001
    assert cc.bootstrap_seed("optimistic") == 3707
    assert cc.bootstrap_seed("sycophancy") == 3703


def test_holm_is_step_down_over_the_supplied_family() -> None:
    p_values = [0.01, 0.04, 0.03, 0.5]
    assert cc.holm(p_values) == pytest.approx([0.04, 0.09, 0.09, 0.5])
    assert cc.holm([0.004] * 12) == pytest.approx([0.048] * 12)


def test_holm_family_is_exactly_twelve_settings() -> None:
    table = synthetic_table(0.0)
    assert len(cc.summarize_questions(table)) == cc.HOLM_FAMILY_SIZE == 12
    del table["8b", "sycophancy"]
    with pytest.raises(ValueError, match="12 matched settings"):
        cc.summarize_questions(table)
    rng = np.random.default_rng(0)
    varied = {key: list(rng.choice([-10.0, -5.0, 0.0, 5.0], size=20)) for key in cc.SETTINGS}
    rows = cc.summarize_questions(varied)
    nominal = [row["wilcoxon_p_nominal"] for row in rows]
    assert [row["holm_12_corrected_coherence"] for row in rows] == cc.holm(nominal)
    assert max(r["holm_12_corrected_coherence"] for r in rows) <= min(1.0, 12 * max(nominal))


def test_question_table_loader_rejects_missing_and_duplicate_rows(tmp_path: Path) -> None:
    rows = [{"model": m, "trait": t, "question_index": q, "n_pairs": 2, "mean_difference": 0.0}
            for m, t in cc.SETTINGS for q in range(20)]

    def write(records: list[dict]) -> Path:
        path = tmp_path / "SYNTHETIC_question_table.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        return path

    assert cc.load_question_contrasts(write(rows)) == synthetic_table(0.0)
    for bad in (rows[:-1], rows[:-1] + [rows[0]], [dict(rows[0], n_pairs=1)] + rows[1:]):
        with pytest.raises(ValueError):
            cc.load_question_contrasts(write(bad))


def test_shipped_question_table_reproduces_corrected_json() -> None:
    corrected = cc.load_corrected_results()
    for row in cc.summarize_questions(cc.load_question_contrasts()):
        stored = corrected[row["model"], row["trait"]]
        for field in ("mean_difference", "ci95_low", "ci95_high", "wilcoxon_p_nominal",
                      "holm_12_corrected_coherence", "bootstrap_seed"):
            assert row[field] == pytest.approx(stored[field], abs=1e-12, rel=0)


def test_current_endpoints_are_bound_to_separate_sources() -> None:
    endpoints = cc.current_matched_endpoints()
    cc.validate_current_endpoints(endpoints)
    assert len(endpoints["target_expression"]) == len(endpoints["coherence"]) == 12
    assert {r["source"] for r in endpoints["target_expression"].values()} == {
        cc.CURRENT_TARGET_SOURCE}
    assert {r["source"] for r in endpoints["coherence"].values()} == {cc.CURRENT_COHERENCE_SOURCE}
    corrected = cc.load_corrected_results()
    for key, row in endpoints["coherence"].items():
        assert row["mean_difference"] == corrected[key]["mean_difference"]


def naive_old_csv_loader() -> dict[str, dict[tuple[str, str], dict]]:
    """Regression model: a generic loader that takes both endpoints from the old Gemini CSV."""
    endpoints = cc.current_matched_endpoints()
    with cc.GEMINI_CELLS.open(newline="") as stream:
        for row in csv.DictReader(stream):
            name = "target_expression" if row["endpoint"] == "trait" else "coherence"
            source = cc.CURRENT_TARGET_SOURCE if name == "target_expression" else (
                cc.CURRENT_COHERENCE_SOURCE)
            endpoints[name][row["model_key"], row["trait"]].update({
                "status": "current", "source": source,
                "mean_difference": float(row["mean_difference_not2_minus_identity2"]),
                "ci95_low": float(row["ci95_low"]), "ci95_high": float(row["ci95_high"])})
    return endpoints


def test_loader_pulling_both_endpoints_from_old_csv_fails() -> None:
    with pytest.raises(ValueError, match="not the corrected rescore"):
        cc.validate_current_endpoints(naive_old_csv_loader())


@pytest.mark.parametrize("name", ["target_expression", "coherence"])
@pytest.mark.parametrize("field", [
    "model", "trait", "endpoint", "status", "source", "n_pairs", "n_questions",
    "mean_difference", "ci95_low", "ci95_high", "holm_12_corrected_coherence",
])
def test_endpoint_validator_checks_every_returned_field(name, field) -> None:
    endpoints = cc.current_matched_endpoints()
    row = endpoints[name][cc.SETTINGS[0]]
    if field not in row:
        row[field] = 0.5
    else:
        value = row[field]
        row[field] = value + 1 if isinstance(value, (int, float)) else "wrong"
    with pytest.raises(ValueError):
        cc.validate_current_endpoints(endpoints)


@pytest.mark.parametrize("name", ["target_expression", "coherence"])
@pytest.mark.parametrize("value", [True, "40", 40.0, float("nan"), float("inf")])
def test_endpoint_validator_rejects_invalid_count_types(name, value) -> None:
    endpoints = cc.current_matched_endpoints()
    endpoints[name][cc.SETTINGS[0]]["n_pairs"] = value
    with pytest.raises(ValueError):
        cc.validate_current_endpoints(endpoints)


@pytest.mark.parametrize("name", ["target_expression", "coherence"])
def test_endpoint_validator_rejects_missing_fields_and_settings(name) -> None:
    endpoints = cc.current_matched_endpoints()
    del endpoints[name][cc.SETTINGS[0]]["ci95_low"]
    with pytest.raises(ValueError, match="missing or unexpected fields"):
        cc.validate_current_endpoints(endpoints)
    endpoints = cc.current_matched_endpoints()
    del endpoints[name][cc.SETTINGS[0]]
    with pytest.raises(ValueError, match="12 matched settings"):
        cc.validate_current_endpoints(endpoints)


def test_historical_coherence_is_separately_labelled() -> None:
    historical = cc.historical_gemini_coherence()
    current = cc.current_matched_endpoints()["coherence"]
    corrected = cc.load_corrected_results()
    assert len(historical) == 12
    for key, row in historical.items():
        assert row["status"] == "historical_gemini_old_rubric"
        assert row["source"] == cc.HISTORICAL_COHERENCE_SOURCE != current[key]["source"]
        assert "holm_12_corrected_coherence" not in row
        assert corrected[key]["historical_mean_difference"] == row["mean_difference"]
    assert any(historical[k]["mean_difference"] != current[k]["mean_difference"]
               for k in cc.SETTINGS)


def test_secondary_240_answer_subset_is_kept_separate() -> None:
    secondary = cc.secondary_evaluator_sensitivity()
    assert sum(int(row["selected_pair_n"]) for row in secondary.values()) == 60
    assert {row["status"] for row in secondary.values()} == {"historical_secondary_subset"}
    current = cc.current_matched_endpoints()["coherence"]
    assert sum(row["n_pairs"] for row in current.values()) == 480
    assert all(row["source"] == cc.CURRENT_COHERENCE_SOURCE for row in current.values())


def test_verifier_passes_on_shipped_data_and_fails_on_changed_estimate(monkeypatch) -> None:
    script = Path(verifier.__file__)
    completed = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "FAIL" not in completed.stdout
    original = cc.summarize_questions
    def changed(table):
        rows = original(table)
        rows[0]["mean_difference"] += 1
        return rows
    monkeypatch.setattr(cc, "summarize_questions", changed)
    assert verifier.main() == 1


@pytest.mark.parametrize("function", ["current_matched_endpoints", "validate_current_endpoints"])
def test_verifier_reports_invalid_endpoints_without_unbound_state(monkeypatch, capsys, function) -> None:
    def corrupt(*args):
        raise ValueError("corrupted endpoint data")

    monkeypatch.setattr(cc, function, corrupt)
    assert verifier.main() == 1
    output = capsys.readouterr().out
    assert "FAIL [BOUND] current endpoints" in output
    assert "FAIL [BOUND] secondary subset" in output
    assert "FAILED: corrected coherence verification" in output
