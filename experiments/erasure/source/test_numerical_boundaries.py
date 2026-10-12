"""Fitting precision, input contracts and fixed random-control seeds."""

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("erasure_numerical_boundaries", Path(__file__).with_name("runner.py"))
assert spec and spec.loader
r = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = r
spec.loader.exec_module(r)

verify_spec = importlib.util.spec_from_file_location("erasure_summary_verifier", Path(__file__).with_name("verify_erasure_outputs.py"))
assert verify_spec and verify_spec.loader
verifier = importlib.util.module_from_spec(verify_spec)
verify_spec.loader.exec_module(verifier)


@pytest.mark.parametrize("x", [
    np.array([[-20.0], [0.0], [20.0]]),
    np.array([[-20.0, 0.0], [20.0, 0.0], [0.0, -10.0], [0.0, 10.0]]),
    np.array([[-20.0, 0.0], [0.0, 0.0], [20.0, 0.0]]),
    np.ones((3, 2)),
])
def test_complement_diagnostics_match_dense_transform_spectrum(x):
    transform, info = r.fit_conceptor_not(x)
    mean = x.mean(axis=0)
    centered = x - mean
    covariance = centered.T @ centered / len(x)
    identity = np.eye(x.shape[1])
    conceptor = np.linalg.solve(covariance + r.ALPHA_INV_SQ * identity, covariance)
    complement = identity - conceptor
    expected = np.linalg.eigvalsh(complement)
    np.testing.assert_allclose(transform(mean + identity) - mean, complement, atol=1e-14, rtol=1e-14)
    assert info["min_eigenvalue"] == pytest.approx(expected.min(), abs=1e-14)
    assert info["max_eigenvalue"] == pytest.approx(expected.max(), abs=1e-14)
    assert info["condition_number"] == pytest.approx(expected.max() / expected.min(), abs=1e-14)
    if x.shape[1] == 1:
        assert info["min_eigenvalue"] == pytest.approx(0.6)
        assert info["max_eigenvalue"] == pytest.approx(0.6)
        assert info["condition_number"] == pytest.approx(1.0)


@pytest.mark.parametrize("directory,mode,split", [
    ("primary", "primary", "question_grouped"),
    ("primary", "smoke", "question_grouped"),
    ("template_sensitivity", "template", "template_grouped"),
])
def test_key_findings_summarize_selected_split(directory, mode, split, tmp_path):
    results = Path(__file__).parents[1] / "results"
    rows = []
    for study in ("primary", "template_sensitivity"):
        with (results / study / "aggregate_results.csv").open() as stream:
            for row in csv.DictReader(stream):
                rows.append({key: float(value) if key in {
                    "balanced_accuracy", "roc_auc", "log_loss", "excess_log_loss",
                    "mean_actual_rank_removed", "mean_normalized_edit_mse_test",
                } else value for key, value in row.items()})
    summary = r.build_key_findings(rows, mode)
    assert summary["primary_split"] == split
    assert set(summary["method_summary"]) == {
        "raw", "conceptor_not_beta1", "leace_paper_exact", "inlp_r1", "inlp_r20",
    }
    for method, stats in summary["method_summary"].items():
        selected = [row for row in rows if row["split_scheme"] == split and row["method"] == method]
        assert stats["cells"] == len(selected) == 21
        for output, column in {
            "mean_balanced_accuracy": "balanced_accuracy", "mean_roc_auc": "roc_auc",
            "mean_log_loss": "log_loss", "mean_excess_log_loss": "excess_log_loss",
            "mean_rank_removed": "mean_actual_rank_removed",
            "mean_normalized_edit_mse_test": "mean_normalized_edit_mse_test",
        }.items():
            assert stats[output] == pytest.approx(sum(row[column] for row in selected) / len(selected), abs=1e-14)
    if directory == "primary":
        historical = json.loads((results / directory / "key_findings.json").read_text())
        assert summary["method_summary"] == historical["method_summary"]
    (tmp_path / "key_findings.json").write_text(json.dumps(summary))
    (tmp_path / "aggregate_results.csv").write_bytes((results / directory / "aggregate_results.csv").read_bytes())
    manifest = {"run_spec": {"mode": mode}, "counts": {"key_findings": summary}}
    assert verifier.verify_key_findings(tmp_path, manifest) == summary
    summary["method_summary"]["raw"]["mean_balanced_accuracy"] += 0.1
    (tmp_path / "key_findings.json").write_text(json.dumps(summary))
    with pytest.raises(RuntimeError, match="Stale key finding"):
        verifier.verify_key_findings(tmp_path, manifest)


def test_historical_empty_template_summary_requires_explicit_opt_in():
    root = Path(__file__).parents[1] / "results" / "template_sensitivity"
    manifest = json.loads((root / "run_manifest.json").read_text())
    with pytest.raises(RuntimeError, match="mode or split"):
        verifier.verify_key_findings(root, manifest)
    with pytest.warns(RuntimeWarning, match="No template key findings were verified"):
        legacy = verifier.verify_key_findings(root, manifest, allow_legacy_template_summary=True)
    assert legacy["method_summary"] == {}


def test_rank_deficient_float32_fits_match_explicit_float64():
    rng = np.random.default_rng(123)
    x = rng.normal(size=(20, 30)).astype(np.float32)
    y = np.r_[np.zeros(10), np.ones(10)]
    heldout = rng.normal(size=(10, 30))
    _, basis = r.row_space_basis(x)
    assert basis.shape == (30, 19)
    transform, info = r.fit_leace_binary(x, y)
    reference, reference_info = r.fit_leace_binary(x.astype(np.float64), y)
    assert info["feature_rank"] == reference_info["feature_rank"] == 19
    np.testing.assert_allclose(transform(heldout), reference(heldout), rtol=1e-13, atol=1e-13)
    assert np.linalg.norm(transform(heldout)) / np.linalg.norm(heldout) < 2
    assert max(r.validate_leace(transform, info, x.astype(np.float64), y).values()) < 1e-10
    conceptor, conceptor_info = r.fit_conceptor_not(x)
    expected, expected_info = r.fit_conceptor_not(x.astype(np.float64))
    assert conceptor_info["conceptor_numerical_rank"] == expected_info["conceptor_numerical_rank"] == 19
    np.testing.assert_allclose(conceptor(heldout), expected(heldout), rtol=1e-13, atol=1e-13)
    actual_coordinates = r.prepare_coordinates(x, heldout)
    expected_coordinates = r.prepare_coordinates(x.astype(np.float64), heldout)
    for actual, expected in zip(actual_coordinates, expected_coordinates):
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    bases, _ = r.fit_inlp_bases(x, y, 2)
    expected_bases, _ = r.fit_inlp_bases(x.astype(np.float64), y, 2)
    for rank in bases:
        np.testing.assert_allclose(bases[rank], expected_bases[rank], rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_svd_tolerance_matches_computed_precision(dtype):
    singular_values = np.array([1, 1e-9], dtype=dtype)
    assert r.svd_tol(singular_values, (20, 30)) == 30 * np.finfo(dtype).eps


@pytest.mark.parametrize("x", [
    np.empty((0, 3)), np.empty((3, 0)), np.ones(3),
    np.array([[np.nan], [1]]), np.array([[np.inf], [1]]),
    np.ones((2, 3), dtype=complex), np.array([["1"], ["2"]]),
])
@pytest.mark.parametrize("function", ["row_space_basis", "fit_conceptor_not", "fit_leace_binary", "fit_inlp_bases"])
def test_fitting_rejects_invalid_activation_arrays(x, function):
    args = (x,)
    if function in ("fit_leace_binary", "fit_inlp_bases"):
        args += (np.array([0, 1]),)
    if function == "fit_inlp_bases":
        args += (1,)
    with pytest.raises(ValueError):
        getattr(r, function)(*args)


@pytest.mark.parametrize("labels", [[0], [0, 2], [0, np.nan], [[0], [1]], [1, 1]])
def test_leace_rejects_invalid_labels(labels):
    with pytest.raises(ValueError):
        r.fit_leace_binary(np.eye(2), np.asarray(labels))


@pytest.mark.parametrize("pos,neg", [
    (np.ones((120, 3)), np.ones((100, 3))),
    (np.ones((100, 3)), np.ones((120, 3))),
    (np.ones((100, 3)), np.ones((100, 4))),
    (np.empty((100, 0)), np.empty((100, 0))),
    (np.ones(100), np.ones((100, 3))),
    (np.full((100, 3), np.nan), np.ones((100, 3))),
    (np.ones((100, 3)), np.full((100, 3), np.inf)),
])
def test_run_fold_rejects_invalid_rows_before_fitting(monkeypatch, pos, neg):
    def should_not_fit(*args):
        pytest.fail("validation must precede splitting and fitting")
    monkeypatch.setattr(r, "build_methods", should_not_fit)
    with pytest.raises(ValueError):
        r.run_fold(pos, neg, "1.5b", "evil", "question_grouped", 0, True)


def test_run_fold_preserves_valid_polarity_labels(monkeypatch):
    seen = []
    def capture(x, y, *args):
        np.testing.assert_array_equal(x[:, 0] > 0, y == 1)
        assert x.dtype == np.float64
        assert len(x) == 120
        seen.append(True)
        return []
    monkeypatch.setattr(r, "build_methods", capture)
    r.run_fold(np.ones((100, 3), np.float32), -np.ones((100, 3), np.float32), "1.5b", "evil", "question_grouped", 0, True)
    assert seen == [True]


def test_seed_table_covers_study_and_matches_recorded_controls():
    table = json.loads(Path(__file__).with_name("random_control_seeds.json").read_text())
    expected = {
        f"{model}/{trait}/{scheme}/{fold}/{rank}"
        for model in r.MODELS for trait in r.TRAITS
        for scheme in ("question_grouped", "template_grouped")
        for fold in range(5) for rank in r.RANDOM_RANKS
    }
    assert set(table) == expected
    assert all(len(seeds) == 3 and all(type(seed) is int and 0 <= seed < 2**32 - 1 for seed in seeds) for seeds in table.values())
    checked = 0
    for path in (Path(__file__).parents[1] / "results").glob("*/fold_results.csv"):
        for row in csv.DictReader(path.open()):
            if not row["method"].startswith("random_rowspace_"):
                continue
            rank, replicate = row["method"].removeprefix("random_rowspace_r").split("_s")
            seed = r.random_control_seed(row["model_key"], row["trait"], row["split_scheme"], int(row["fold"]), int(rank), int(replicate))
            assert seed == int(row["random_seed"])
            checked += 1
    assert checked == 1260
