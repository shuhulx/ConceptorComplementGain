"""NumPy paired-geometry estimators on synthetic arrays."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from paired_geometry import (
    ALPHA,
    FROZEN_DIRECTORY,
    OUTPUT_NAMES,
    REPOSITORY,
    conceptor_eigenvalues,
    covariance_eigenvalues,
    covariance_eigenvalues_svd,
    effective_rank,
    float64_metrics,
    k90,
    paired_difference,
    prepare_output_directory,
    spectral_metrics,
    write_outputs,
)


def synthetic_rows(rows: int, width: int, seed: int, scale: float = 1.0) -> np.ndarray:
    generator = np.random.default_rng(seed)
    scales = scale * np.geomspace(30.0, 0.3, width)
    return 5 + generator.standard_normal((rows, width)) * scales


def biased_covariance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (x - x.mean(0)).T @ (y - y.mean(0)) / len(x)


@pytest.mark.parametrize(
    ("spectrum", "expected"),
    [
        ([0.3], 1.0),
        ([0.2, 0.2, 0.2, 0.2], 4.0),
        ([0.4, 0.4, 0.0, 0.0], 2.0),  # 0 log 0 = 0
        ([0.5, 0.25, 0.25], 2**1.5),  # H = 1.5 log 2
    ],
)
def test_shannon_effective_rank_on_hand_checkable_spectra(spectrum, expected) -> None:
    assert effective_rank(np.array(spectrum)) == pytest.approx(expected, rel=1e-12)
    # Invariant to the scale of the conceptor trace.
    assert effective_rank(7 * np.array(spectrum)) == pytest.approx(expected, rel=1e-12)


def test_effective_rank_is_not_the_participation_ratio() -> None:
    spectrum = np.array([0.5, 0.25, 0.25])
    participation = spectrum.sum() ** 2 / np.square(spectrum).sum()
    assert participation == pytest.approx(8 / 3)
    assert effective_rank(spectrum) != pytest.approx(participation)


@pytest.mark.parametrize(
    ("spectrum", "expected"),
    [
        ([9.0, 1.0], 1),  # exactly 90%: the bound is inclusive
        ([4.0, 2.0, 1.0, 1.0], 4),  # cumulative .5, .75, .875, 1
        ([5.0, 4.0, 0.5, 0.5], 2),
        ([1.0] * 10, 9),
        ([0.0, 0.0], 2),  # zero trace
    ],
)
def test_k90_is_smallest_count_reaching_ninety_percent(spectrum, expected) -> None:
    assert k90(np.array(spectrum)) == expected


def test_conceptor_eigenvalues_match_dense_conceptor() -> None:
    assert conceptor_eigenvalues(np.array([400.0, 0.0]), ALPHA) == pytest.approx([0.5, 0.0])
    rows = synthetic_rows(6, 10, seed=1)
    eigenvalues = covariance_eigenvalues(rows)
    covariance = biased_covariance(rows, rows)
    dense = covariance @ np.linalg.inv(covariance + ALPHA**-2 * np.eye(10))
    expected = np.sort(np.linalg.eigvals(dense).real)[::-1][:6]
    np.testing.assert_allclose(conceptor_eigenvalues(eigenvalues), expected, rtol=1e-9, atol=1e-12)
    assert np.all(np.diff(conceptor_eigenvalues(eigenvalues)) <= 0)


def test_covariance_centers_by_own_mean_and_divides_by_n() -> None:
    wide, tall = synthetic_rows(6, 9, seed=2), synthetic_rows(12, 4, seed=3)
    for rows in (wide, tall):
        biased = np.sort(np.linalg.eigvalsh(np.cov(rows, rowvar=False, bias=True)))[::-1]
        observed = covariance_eigenvalues(rows)
        rank = min(rows.shape[0] - 1, rows.shape[1])
        np.testing.assert_allclose(observed[:rank], biased[:rank], rtol=1e-10)
        np.testing.assert_allclose(covariance_eigenvalues_svd(rows)[:rank], biased[:rank], rtol=1e-10)
        unbiased = np.sort(np.linalg.eigvalsh(np.cov(rows, rowvar=False)))[::-1]
        assert not np.allclose(observed[:rank], unbiased[:rank])
        # Adding a constant row offset does not change the spectrum.
        np.testing.assert_allclose(covariance_eigenvalues(rows + 100.0), observed, rtol=1e-8, atol=1e-8)
    # Six centered rows span five directions: the sixth Gram eigenvalue is zero.
    assert covariance_eigenvalues(wide)[-1] == pytest.approx(0.0, abs=1e-9)


def test_paired_difference_covariance_identity() -> None:
    u = synthetic_rows(50, 5, seed=4)
    v = 0.6 * u + synthetic_rows(50, 5, seed=5)
    left = biased_covariance(u - v, u - v)
    right = biased_covariance(u, u) + biased_covariance(v, v) - biased_covariance(u, v) - biased_covariance(v, u)
    np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-9)
    # The cross-covariance terms matter: Cov(U) + Cov(V) alone is wrong here.
    assert not np.allclose(left, biased_covariance(u, u) + biased_covariance(v, v), rtol=1e-3)


def test_paired_spectrum_is_not_determined_by_target_present_rows() -> None:
    u = synthetic_rows(40, 6, seed=6)
    correlated, independent = 0.9 * u, synthetic_rows(40, 6, seed=7)
    assert not np.allclose(covariance_eigenvalues(paired_difference(u, correlated)),
                           covariance_eigenvalues(paired_difference(u, independent)))


def test_paired_difference_is_row_aligned_and_refuses_unpaired_rows() -> None:
    present, opposed = synthetic_rows(8, 3, seed=8), synthetic_rows(8, 3, seed=9)
    np.testing.assert_array_equal(paired_difference(present, opposed)[2], present[2] - opposed[2])
    with pytest.raises(ValueError, match="unpaired"):
        paired_difference(present, opposed[:7])


def test_float32_estimator_agrees_with_float64_svd() -> None:
    rows = synthetic_rows(100, 300, seed=10, scale=2.0)
    present, opposed = rows, 0.5 * rows + synthetic_rows(100, 300, seed=11, scale=2.0)
    for x in (present, paired_difference(present, opposed)):
        single = spectral_metrics(x.astype(np.float32))
        double = float64_metrics(x)
        assert 1 < single["eff_rank"] < 99 and 1 < single["k90"] < 99
        assert abs(single["eff_rank"] - double["eff_rank"]) < 1e-3
        assert single["k90"] == double["k90"]
        assert float(np.float32(single["eff_rank"])) == single["eff_rank"]


def test_spectral_metrics_fields_are_consistent() -> None:
    metrics = spectral_metrics(synthetic_rows(20, 40, seed=12))
    assert (metrics["n_samples"], metrics["n_dims"]) == (20, 40)
    assert 0 < metrics["mu5"] <= metrics["mu1"] < 1
    assert 0 < metrics["scr"] <= metrics["mu1"]
    assert metrics["mu1_mu5"] == metrics["mu1"] / metrics["mu5"]
    assert 0 < metrics["fdr"] <= 1


def frozen_digests() -> dict[str, str]:
    names = ("paired_delta_results.json", "paired_geometry_main.csv")
    return {name: hashlib.sha256((FROZEN_DIRECTORY / name).read_bytes()).hexdigest() for name in names}


def test_output_directory_refuses_frozen_evidence_and_existing_outputs(tmp_path) -> None:
    before = frozen_digests()
    for overwrite in (False, True):
        with pytest.raises(RuntimeError, match="frozen evidence"):
            prepare_output_directory(FROZEN_DIRECTORY, overwrite=overwrite)
    alias = tmp_path / "alias"
    alias.symlink_to(FROZEN_DIRECTORY, target_is_directory=True)
    with pytest.raises(RuntimeError, match="frozen evidence"):
        prepare_output_directory(alias, overwrite=True)

    output = prepare_output_directory(tmp_path / "rebuild")
    rows = [{"model_key": "synthetic", "value": 1.0}]
    write_outputs(output, rows, rows, {"status": "synthetic"})
    assert sorted(path.name for path in output.iterdir()) == sorted(OUTPUT_NAMES)
    with pytest.raises(FileExistsError):
        prepare_output_directory(output)
    with pytest.raises(FileExistsError):
        write_outputs(output, rows, rows, {"status": "synthetic"})
    assert prepare_output_directory(output, overwrite=True) == output.resolve()
    assert frozen_digests() == before


def test_output_directory_refuses_any_path_inside_the_repository(tmp_path) -> None:
    for path in (REPOSITORY, REPOSITORY / "experiments" / "paired_geometry" / "rebuild", REPOSITORY / "new_output"):
        with pytest.raises(RuntimeError, match="outside the repository"):
            prepare_output_directory(path, overwrite=True)
        assert path == REPOSITORY or not path.exists()
    alias = tmp_path / "repository_alias"
    alias.symlink_to(REPOSITORY, target_is_directory=True)
    with pytest.raises(RuntimeError, match="outside the repository"):
        prepare_output_directory(alias / "rebuild", overwrite=True)
    assert not (REPOSITORY / "rebuild").exists()
    assert REPOSITORY.resolve() not in tmp_path.resolve().parents


def test_record_checks_fire_under_python_optimize() -> None:
    # The integrity checks are explicit raises, so `python -O` must not strip them.
    code = "\n".join([
        "import sys",
        f"sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})",
        "assert False, 'assert statements must be stripped under -O'",
        "from paired_geometry import check_records",
        "try:",
        "    check_records([])",
        "except RuntimeError as error:",
        "    print('RAISED', error)",
        "else:",
        "    print('NOT RAISED')",
    ])
    completed = subprocess.run(
        [sys.executable, "-O", "-B", "-c", code], capture_output=True, text=True, check=True, timeout=300
    )
    assert completed.stdout.startswith("RAISED expected 21 model-target"), completed.stdout + completed.stderr
