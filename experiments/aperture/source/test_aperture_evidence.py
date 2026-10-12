"""Aperture formulas and tensor-loading contracts on synthetic activations."""

from __future__ import annotations

import hashlib
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from aperture_evidence import (
    FROZEN_DIRECTORY,
    OUTPUT_NAMES,
    REPOSITORY,
    centered_covariance,
    complete_gain,
    complete_map,
    conceptor_coefficient,
    conceptor_matrix,
    heldout_correction_energies,
    load_tensor_tree,
    matched_control_map,
    prepare_output_directory,
    question_folds,
    rms_ratio,
    write_outputs,
)


def synthetic_rows(rows: int, width: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    scales = torch.linspace(3.0, 0.1, width, dtype=torch.float64)
    return 5 + torch.randn(rows, width, generator=generator, dtype=torch.float64) * scales


def test_covariance_centers_by_sample_mean_and_divides_by_n() -> None:
    x = synthetic_rows(6, 4, seed=1)
    centered, covariance = centered_covariance(x)
    torch.testing.assert_close(centered.mean(0), torch.zeros(4, dtype=torch.float64), atol=1e-12, rtol=0)
    biased = np.cov(x.numpy(), rowvar=False, bias=True)
    np.testing.assert_allclose(covariance.numpy(), biased, rtol=1e-12, atol=1e-12)
    assert not np.allclose(covariance.numpy(), np.cov(x.numpy(), rowvar=False, bias=False))


def test_dense_conceptor_matches_eigenvalue_formula_including_null_directions() -> None:
    # Five rows in eight dimensions: the covariance has rank four, so four null directions.
    _centered, covariance = centered_covariance(synthetic_rows(5, 8, seed=2))
    aperture = 0.7
    eigenvalues, vectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(0)
    dense = conceptor_matrix(covariance, aperture)
    expected = torch.tensor([conceptor_coefficient(float(v), aperture) for v in eigenvalues], dtype=torch.float64)
    torch.testing.assert_close(torch.diag(vectors.T @ dense @ vectors), expected, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(dense, dense.T, atol=1e-12, rtol=0)
    assert int((eigenvalues < 1e-10).sum()) == 4
    assert all(complete_gain(float(v), aperture) == pytest.approx(2.0) for v in eigenvalues[eigenvalues < 1e-10])


@pytest.mark.parametrize("scaled", [0.5, 0.999, 1.001, 3.0])
def test_beta2_gain_falls_below_one_exactly_when_c_exceeds_half(scaled: float) -> None:
    aperture = 0.2
    eigenvalue = scaled / aperture**2
    c = conceptor_coefficient(eigenvalue, aperture)
    g = complete_gain(eigenvalue, aperture, beta=2.0)
    assert g == pytest.approx(2.0 * (1 - c))
    assert (g < 1) == (c > 0.5) == (aperture**2 * eigenvalue > 2.0 - 1)


def test_complete_minus_control_is_exactly_minus_beta_c_times_deviation() -> None:
    train = synthetic_rows(10, 5, seed=3)
    center = train.mean(0)
    conceptor = conceptor_matrix(centered_covariance(train)[1], 0.4)
    h = synthetic_rows(7, 5, seed=4)
    beta = 2.0
    difference = complete_map(h, center, conceptor, beta) - matched_control_map(h, center, beta)
    torch.testing.assert_close(difference, -beta * (h - center) @ conceptor.T, atol=1e-12, rtol=1e-12)


def test_question_folds_follow_template_major_random_state_42_convention() -> None:
    groups, labels, folds = question_folds()
    expected = np.random.RandomState(42).permutation(20).reshape(5, 4)
    np.testing.assert_array_equal(groups, expected)
    assert [sorted(group) for group in groups.tolist()] == [[0, 1, 15, 17], [3, 5, 8, 11], [2, 13, 16, 18], [4, 9, 12, 19], [6, 7, 10, 14]]
    np.testing.assert_array_equal(labels, np.arange(100) % 20)
    rows = np.arange(100)
    for group, test in zip(groups, folds):
        assert sorted(test.tolist()) == sorted(t * 20 + int(q) for t in range(5) for q in group)
        train = rows[~np.isin(rows, test)]
        assert len(test) == 20 and len(train) == 80
        assert not set(labels[train]) & set(labels[test])
    assert sorted(np.concatenate(folds).tolist()) == list(range(100))


def test_heldout_energies_match_dense_conceptor_on_fold_refitted_center() -> None:
    train, test = synthetic_rows(8, 6, seed=5), synthetic_rows(3, 6, seed=6)
    apertures = (0.05, 0.3, 2.0)
    input_energy, energies, direct = heldout_correction_energies(train, test, apertures)
    center = train.mean(0)
    z = test - center
    assert input_energy == pytest.approx(float(z.square().sum()), rel=1e-12)
    covariance = centered_covariance(train)[1]
    for alpha in apertures:
        dense = float((z @ conceptor_matrix(covariance, alpha).T).square().sum())
        assert energies[alpha] == pytest.approx(dense, rel=1e-9)
        assert direct[alpha] == pytest.approx(dense, rel=1e-9)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_heldout_precision_promotes_exact_input_values_before_centering(dtype) -> None:
    # This wide float32 Gram matrix previously failed the fixed PSD cutoff.
    x = torch.randn(100, 128, generator=torch.Generator().manual_seed(0)).to(dtype)
    before = x.clone()
    observed = heldout_correction_energies(x[:80], x[80:], (0.05, 0.5))
    expected = heldout_correction_energies(x[:80].double(), x[80:].double(), (0.05, 0.5))
    assert observed == expected
    assert torch.equal(x, before)


@pytest.mark.parametrize("bad", [
    torch.ones(3), torch.ones(2, 3, 4), torch.empty(0, 4), torch.empty(3, 0),
    torch.ones(3, 4, dtype=torch.int64), torch.ones(3, 4, dtype=torch.complex64),
    torch.full((3, 4), float("nan")), torch.full((3, 4), float("inf")),
    torch.ones(3, 4, device="meta"), torch.eye(4).to_sparse(), [[1.0]],
])
def test_heldout_rejects_malformed_inputs(bad) -> None:
    valid = torch.ones(3, 4)
    for train, test in ((bad, valid), (valid, bad)):
        with pytest.raises(ValueError):
            heldout_correction_energies(train, test, (0.5,))
    with pytest.raises(ValueError, match="same positive width"):
        heldout_correction_energies(valid, torch.ones(3, 5), (0.5,))


@pytest.mark.parametrize("alpha", [0, -1, float("nan"), float("inf"), True, "0.5"])
def test_heldout_rejects_invalid_apertures_before_linear_algebra(monkeypatch, alpha) -> None:
    def unexpected_eigh(*args, **kwargs):
        raise AssertionError("invalid aperture reached linear algebra")

    monkeypatch.setattr(torch.linalg, "eigh", unexpected_eigh)
    with pytest.raises(ValueError, match="apertures"):
        heldout_correction_energies(torch.ones(3, 4), torch.ones(1, 4), (0.5, alpha))


@pytest.mark.parametrize("alpha", [1e-160, 1e-200, 1e160])
def test_heldout_rejects_unrepresentable_regularization_before_algebra(monkeypatch, alpha):
    def unexpected_eigh(*args, **kwargs):
        raise AssertionError("unrepresentable regularization reached linear algebra")

    monkeypatch.setattr(torch.linalg, "eigh", unexpected_eigh)
    with pytest.raises(ValueError, match="regularization"):
        heldout_correction_energies(torch.ones(3, 4), torch.ones(1, 4), (alpha,))


def test_heldout_gram_guard_allows_roundoff_at_large_activation_scale():
    generator = torch.Generator().manual_seed(0)
    train = 1000 * torch.randn(80, 4096, generator=generator, dtype=torch.float64)
    test = 1000 * torch.randn(20, 4096, generator=generator, dtype=torch.float64)
    input_energy, energies, direct = heldout_correction_energies(train, test, (0.0001, 0.005))
    assert np.isfinite(input_energy)
    for aperture in energies:
        assert energies[aperture] == pytest.approx(direct[aperture], rel=1e-10)


def test_heldout_gram_guard_rejects_materially_negative_eigenvalue(monkeypatch):
    def indefinite_eigh(gram):
        return torch.tensor([-0.1, 1.0], dtype=gram.dtype), torch.eye(2, dtype=gram.dtype)

    monkeypatch.setattr(torch.linalg, "eigh", indefinite_eigh)
    with pytest.raises(RuntimeError, match="not positive semidefinite"):
        heldout_correction_energies(torch.ones(2, 4), torch.ones(1, 4), (0.05,))


def test_pooled_ratio_is_not_the_mean_of_fold_ratios() -> None:
    inputs, corrections = [1.0, 100.0], [0.25, 1.0]
    per_fold = [rms_ratio(i, c) for i, c in zip(inputs, corrections)]
    pooled = rms_ratio(sum(inputs), sum(corrections))
    assert per_fold == pytest.approx([1.0, 0.2])
    assert pooled == pytest.approx(2 * np.sqrt(1.25 / 101))
    assert pooled != pytest.approx(np.mean(per_fold))


def test_beta2_ratio_stays_below_two_even_at_large_aperture() -> None:
    train, test = synthetic_rows(12, 4, seed=7), synthetic_rows(6, 4, seed=8)
    apertures = (0.01, 1.0, 100.0, 1e4)
    input_energy, energies, _direct = heldout_correction_energies(train, test, apertures)
    ratios = [rms_ratio(input_energy, energies[alpha]) for alpha in apertures]
    assert all(0 < ratio < 2 for ratio in ratios)
    assert all(np.diff(ratios) > 0)


EXECUTED: list[str] = []


def _record_execution() -> None:
    EXECUTED.append("unsafe")


class UnsafePayload:
    def __reduce__(self):
        return (_record_execution, ())


def test_tensor_loader_refuses_arbitrary_pickled_objects(tmp_path) -> None:
    path = tmp_path / "synthetic_unsafe.pt"
    torch.save({"persona": UnsafePayload()}, path)
    with pytest.raises(pickle.UnpicklingError):
        load_tensor_tree(path.read_bytes())
    assert EXECUTED == []
    safe = tmp_path / "synthetic_safe.pt"
    torch.save({"persona": {"x": {"pos": {9: torch.ones(2, 3)}}}}, safe)
    assert load_tensor_tree(safe.read_bytes())["persona"]["x"]["pos"][9].shape == (2, 3)


def frozen_digests() -> dict[str, str]:
    return {name: hashlib.sha256((FROZEN_DIRECTORY / name).read_bytes()).hexdigest()
            for name in OUTPUT_NAMES if (FROZEN_DIRECTORY / name).exists()}


def test_output_directory_refuses_frozen_evidence_and_existing_outputs(tmp_path) -> None:
    before = frozen_digests()
    with pytest.raises(RuntimeError, match="frozen evidence"):
        prepare_output_directory(FROZEN_DIRECTORY)
    with pytest.raises(RuntimeError, match="frozen evidence"):
        prepare_output_directory(FROZEN_DIRECTORY, overwrite=True)
    alias = tmp_path / "alias"
    alias.symlink_to(FROZEN_DIRECTORY, target_is_directory=True)
    with pytest.raises(RuntimeError, match="frozen evidence"):
        prepare_output_directory(alias, overwrite=True)

    output = prepare_output_directory(tmp_path / "rebuild")
    rows = [{"aperture": 0.05, "value": 1.0}]
    write_outputs(output, rows, rows, rows, rows, {"status": "synthetic"})
    with pytest.raises(FileExistsError):
        prepare_output_directory(output)
    with pytest.raises(FileExistsError):
        write_outputs(output, rows, rows, rows, rows, {"status": "synthetic"})
    assert prepare_output_directory(output, overwrite=True) == output.resolve()
    assert frozen_digests() == before


def test_output_directory_refuses_any_path_inside_the_repository(tmp_path) -> None:
    inside = [REPOSITORY, REPOSITORY / "experiments" / "aperture" / "rebuild", REPOSITORY / "new_output"]
    for path in inside:
        with pytest.raises(RuntimeError, match="outside the repository"):
            prepare_output_directory(path, overwrite=True)
        assert path == REPOSITORY or not path.exists()
    alias = tmp_path / "repository_alias"
    alias.symlink_to(REPOSITORY, target_is_directory=True)
    with pytest.raises(RuntimeError, match="outside the repository"):
        prepare_output_directory(alias / "rebuild", overwrite=True)
    assert not (REPOSITORY / "rebuild").exists()
    assert REPOSITORY.resolve() not in tmp_path.resolve().parents


def test_integrity_checks_fire_under_python_optimize() -> None:
    # The integrity checks are explicit raises, so `python -O` must not strip them.
    code = "\n".join([
        "import sys",
        f"sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})",
        "assert False, 'assert statements must be stripped under -O'",
        "from aperture_evidence import check_records",
        "try:",
        "    check_records([], [], [])",
        "except RuntimeError as error:",
        "    print('RAISED', error)",
        "else:",
        "    print('NOT RAISED')",
    ])
    completed = subprocess.run(
        [sys.executable, "-O", "-B", "-c", code], capture_output=True, text=True, check=True, timeout=300
    )
    assert completed.stdout.startswith("RAISED expected 147 spectral"), completed.stdout + completed.stderr
