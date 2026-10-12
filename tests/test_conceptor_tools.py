"""Synthetic checks of the factorized activation-analysis API."""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import pytest
import torch

from conceptor_tools import fit_conceptor
from eval.compare_activations import compare_activations
from examples.compare_activations import synthetic_rows
from extract_activations import last_token_resid_pre

ROOT = Path(__file__).resolve().parents[1]
DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def rows(n: int, d: int, dtype=torch.float64) -> torch.Tensor:
    return (3 + torch.randn(n, d, generator=torch.Generator().manual_seed(n + d))).to(dtype)


def dense_reference(train, heldout, aperture, beta):
    x, h = train.double(), heldout.double()
    center = x.mean(0)
    z = x - center
    covariance = z.T @ z / len(x)
    # Independent primal solve; production fitting never constructs this matrix.
    conceptor = torch.linalg.solve(
        covariance + aperture**-2 * torch.eye(x.shape[1], dtype=torch.float64), covariance
    )
    deviation = h - center
    correction = -beta * (deviation @ conceptor.T)
    return center + beta * (deviation @ (torch.eye(x.shape[1]) - conceptor).T), correction


@pytest.mark.parametrize("n,d", [(5, 11), (19, 4), (1, 7)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_factorized_maps_match_independent_dense_solve(n, d, dtype):
    train, heldout = rows(n, d, dtype), rows(1, d, dtype)
    fitted = fit_conceptor(train, 0.7)
    expected, correction = dense_reference(train, heldout, 0.7, 1.8)
    torch.testing.assert_close(fitted.complete(heldout, 1.8), expected, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(fitted.correction(heldout, 1.8), correction, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(
        fitted.control(heldout, 1.8), fitted.center + 1.8 * (heldout.double() - fitted.center)
    )
    assert fitted.center.dtype == fitted.basis.dtype == fitted.coefficients.dtype == torch.float64
    assert fitted.complete(heldout).device.type == "cpu"


def test_rank_deficient_and_constant_rows():
    train = rows(8, 5)
    train[:, 3:] = train[:, :2]
    heldout = rows(6, 5)
    fitted = fit_conceptor(train, 1.2)
    expected, correction = dense_reference(train, heldout, 1.2, 2)
    torch.testing.assert_close(fitted.complete(heldout), expected, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(fitted.correction(heldout), correction, atol=1e-11, rtol=1e-11)
    constant = fit_conceptor(torch.full((4, 9), 5.0), 0.7)
    assert torch.count_nonzero(constant.coefficients) == 0
    assert torch.count_nonzero(constant.correction(rows(3, 9))) == 0
    torch.testing.assert_close(constant.complete(rows(3, 9)), constant.control(rows(3, 9)))


@pytest.mark.parametrize("shape", [(7,), (1, 7), (2, 3, 7), (2, 1, 3, 7)])
def test_leading_batch_dimensions_use_one_fixed_training_center(shape):
    train = rows(6, 7)
    heldout = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float64).reshape(shape)
    fitted = fit_conceptor(train, 0.4)
    expected, correction = dense_reference(train, heldout, 0.4, 2)
    torch.testing.assert_close(fitted.complete(heldout), expected, atol=1e-10, rtol=1e-12)
    torch.testing.assert_close(fitted.correction(heldout), correction, atol=1e-10, rtol=1e-12)
    assert fitted.complete(heldout).shape == shape


def test_common_translation_and_center_are_preserved():
    train, heldout = rows(9, 6), rows(3, 6)
    shift = torch.arange(6, dtype=torch.float64) * 3
    fitted, translated = fit_conceptor(train, 0.4), fit_conceptor(train + shift, 0.4)
    torch.testing.assert_close(fitted.complete(fitted.center), fitted.center)
    torch.testing.assert_close(fitted.control(fitted.center), fitted.center)
    for name in ("complete", "control"):
        torch.testing.assert_close(
            getattr(translated, name)(heldout + shift), getattr(fitted, name)(heldout) + shift
        )
    torch.testing.assert_close(translated.correction(heldout + shift), fitted.correction(heldout))


def test_stored_factors_and_nullspace_gain_have_economy_shapes():
    fitted = fit_conceptor(rows(5, 13), 0.6)
    assert fitted.center.shape == (13,)
    assert fitted.basis.shape == (5, 13)
    assert fitted.coefficients.shape == (5,)
    tensors = [getattr(fitted, field.name) for field in fields(fitted)
               if isinstance(getattr(fitted, field.name), torch.Tensor)]
    assert len(tensors) == 3 and all(tensor.shape != (13, 13) for tensor in tensors)
    gains = fitted.directional_gains(2.3)
    assert gains.orthogonal_nullspace_dimension == 8
    assert gains.orthogonal_nullspace_gain == 2.3
    torch.testing.assert_close(gains.represented, 2.3 * (1 - fitted.coefficients))
    # Complete an orthonormal basis only in this small test, to probe the omitted span.
    null_direction = torch.linalg.qr(fitted.basis.T, mode="complete").Q[:, -1]
    torch.testing.assert_close(
        fitted.complete(fitted.center + null_direction, 2.3) - fitted.center,
        2.3 * null_direction, atol=1e-12, rtol=1e-12,
    )
    assert fit_conceptor(rows(19, 4), 0.6).directional_gains().orthogonal_nullspace_dimension == 0


def test_inputs_factors_and_global_rng_are_not_mutated():
    train, heldout = rows(6, 11).requires_grad_(), rows(3, 11).requires_grad_()
    before_train, before_heldout, rng = train.detach().clone(), heldout.detach().clone(), torch.get_rng_state()
    fitted = fit_conceptor(train, 0.4)
    factors = [fitted.center.clone(), fitted.basis.clone(), fitted.coefficients.clone()]
    for method in (fitted.complete, fitted.control, fitted.correction):
        assert not method(heldout).requires_grad
    synthetic_rows(5)
    torch.testing.assert_close(train, before_train, rtol=0, atol=0)
    torch.testing.assert_close(heldout, before_heldout, rtol=0, atol=0)
    for actual, expected in zip((fitted.center, fitted.basis, fitted.coefficients), factors):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("invalid", [0, -1, float("nan"), float("inf"), -float("inf"), True, "0.5"])
def test_invalid_aperture_and_beta_are_rejected(invalid):
    with pytest.raises(ValueError, match="aperture"):
        fit_conceptor(rows(3, 4), invalid)
    fitted = fit_conceptor(rows(3, 4), 0.5)
    for method in (fitted.complete, fitted.control, fitted.correction, fitted.directional_gains):
        with pytest.raises(ValueError, match="beta"):
            method(invalid) if method == fitted.directional_gains else method(rows(1, 4), invalid)


@pytest.mark.parametrize("bad", [
    [[1.0]], torch.ones(3), torch.ones(2, 3, 4), torch.empty(0, 4), torch.empty(3, 0),
    torch.ones(3, 4, dtype=torch.int64), torch.ones(3, 4, dtype=torch.complex64),
    torch.full((3, 4), float("nan")), torch.full((3, 4), float("inf")),
    torch.ones(3, 4, device="meta"), torch.eye(3).to_sparse(),
])
def test_malformed_training_inputs_are_rejected(bad):
    with pytest.raises(ValueError):
        fit_conceptor(bad, 0.5)


@pytest.mark.parametrize("bad", [
    [[1.0]], torch.tensor(1.0), torch.ones(2, 5), torch.empty(0, 4), torch.empty(2, 0, 4),
    torch.ones(2, 4, dtype=torch.int32), torch.ones(2, 4, dtype=torch.complex64),
    torch.full((2, 4), float("nan")), torch.ones(2, 4, device="meta"),
])
def test_malformed_application_inputs_are_rejected(bad):
    fitted = fit_conceptor(rows(5, 4), 0.5)
    for method in (fitted.complete, fitted.control, fitted.correction):
        with pytest.raises(ValueError):
            method(bad)


@pytest.mark.parametrize("aperture", [1e-300, 1e300])
def test_extreme_finite_apertures_keep_zero_rank_coefficients_finite(aperture):
    fitted = fit_conceptor(torch.ones(2, 4), aperture)
    assert torch.equal(fitted.coefficients, torch.zeros(2, dtype=torch.float64))


def test_saturated_coefficients_are_rejected_instead_of_reporting_zero_gain():
    train = torch.tensor([[-1.0, 0], [1.0, 0]], dtype=torch.float64)
    with pytest.raises(ValueError, match="saturates float64"):
        fit_conceptor(train, 1e8)


def test_representable_subnormal_coefficients_are_not_lost_by_sigmoid_overflow():
    train = torch.tensor([[-1.0, 0], [1.0, 0]], dtype=torch.float64)
    fitted = fit_conceptor(train, 1e-155)
    assert float(fitted.coefficients[0]) == pytest.approx(1e-310, rel=1e-11, abs=0)
    correction = fitted.correction(torch.tensor([[1e155, 0]], dtype=torch.float64))
    assert float(correction[0, 0]) == pytest.approx(-2e-155, rel=1e-11, abs=0)


def test_unrepresentable_nonzero_coefficients_are_rejected():
    train = torch.tensor([[-1.0, 0], [1.0, 0]], dtype=torch.float64)
    with pytest.raises(ValueError, match="underflows float64"):
        fit_conceptor(train, 1e-200)


@pytest.mark.parametrize("scale", [1e160, 1e-160, 1e-200])
def test_rms_ratio_survives_extreme_finite_state_scales(scale):
    train = torch.tensor([[-1.0, 0], [1.0, 0]], dtype=torch.float64)
    heldout = torch.tensor([[scale, scale]], dtype=torch.float64)
    result = compare_activations(train, heldout, aperture=0.05)
    coefficient = 1 / (1 + 0.05**-2)
    assert result.correction_rms == pytest.approx(2 * coefficient * scale / 2**0.5, rel=1e-12, abs=0)
    assert result.matched_displacement_rms == pytest.approx(scale, rel=1e-12, abs=0)
    assert result.correction_ratio == pytest.approx(2 * coefficient / 2**0.5, rel=1e-12, abs=0)
    assert result.ratio_undefined_reason is None


def test_overflowing_derived_states_are_rejected():
    fitted = fit_conceptor(torch.tensor([[-1.0, 0], [1.0, 0]], dtype=torch.float64), 0.05)
    for method in (fitted.complete, fitted.control):
        with pytest.raises(ValueError, match="float64 arithmetic range"):
            method(torch.tensor([[2.0, 0]], dtype=torch.float64), 1e308)
    with pytest.raises(ValueError, match="float64 arithmetic range"):
        fitted.correction(torch.tensor([[1e308, 0]], dtype=torch.float64), 1e308)


def test_comparison_diagnostics_and_heldout_is_never_refitted():
    train, heldout = rows(5, 9, torch.float32), rows(3, 9, torch.float32)
    result = compare_activations(train, heldout, aperture=0.6, beta=2.0)
    expected, correction = dense_reference(train, heldout, 0.6, 2.0)
    torch.testing.assert_close(result.complete, expected, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(result.correction, correction, atol=1e-11, rtol=1e-11)
    assert result.correction_rms == pytest.approx(float(correction.square().mean().sqrt()))
    assert result.correction_ratio == pytest.approx(float(correction.norm() / (result.control - heldout).norm()))
    assert result.algebra_max_abs_error < 1e-12
    changed = compare_activations(train, heldout + 100, aperture=0.6)
    for name in ("center", "basis", "coefficients"):
        assert torch.equal(getattr(result.fitted, name), getattr(changed.fitted, name))
    torch.testing.assert_close(result.fitted.center, train.double().mean(0))


def test_undefined_ratio_is_explicit_for_beta_one_or_zero_displacement():
    train = rows(6, 4)
    one = compare_activations(train, rows(1, 4), beta=1)
    assert one.correction_ratio is None and "beta=1" in one.ratio_undefined_reason
    assert one.matched_displacement_rms == 0
    centered = compare_activations(train, train.mean(0))
    assert centered.correction_ratio is None and "zero" in centered.ratio_undefined_reason
    assert centered.correction_rms == 0


def test_float32_extraction_outputs_compose_without_a_model_download():
    class FakeModel:
        def run_with_cache(self, prompt, names_filter):
            name = "blocks.4.hook_resid_pre"
            assert names_filter(name)
            values = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
            return None, {name: values + float(prompt)}

    extracted = last_token_resid_pre(FakeModel(), ["1", "2", "4", "8"], layers=(4,))[4]
    assert extracted.dtype == torch.float32 and extracted.device.type == "cpu"
    result = compare_activations(extracted[:3], extracted[3:], aperture=0.5)
    expected, _ = dense_reference(extracted[:3], extracted[3:], 0.5, 2)
    torch.testing.assert_close(result.complete, expected)


def test_cli_is_deterministic_labels_wide_float32_and_explains_nullspace():
    command = [sys.executable, "-B", "-m", "examples.compare_activations"]
    first = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=300)
    second = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert first.returncode == second.returncode == 0, first.stdout + first.stderr
    assert first.stdout == second.stdout
    assert "Synthetic float32 activation comparison" in first.stdout
    assert "fit=(80, 128)" in first.stdout and "48 orthogonal nullspace directions" in first.stdout
    assert "not the full ambient spectrum" in first.stdout
    assert "Correction RMS:" in first.stdout
    assert "Algebra max error |F_C - F_0 + beta C(h-m)|:" in first.stdout
    one = subprocess.run(command + ["--beta", "1"], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert one.returncode == 0 and "undefined (beta=1" in one.stdout


@pytest.mark.parametrize("document", ["README.md", "docs/OWN_ACTIVATIONS.md"])
def test_documented_recipe_uses_supplied_training_and_heldout_tensors(document):
    text = (ROOT / document).read_text()
    recipe = re.findall(r"```python\n(.*?)```", text, re.DOTALL)[0]
    generator = torch.Generator().manual_seed(53)
    train = 2 + torch.randn(12, 8, generator=generator)
    heldout = 2 + torch.randn(5, 8, generator=generator)
    namespace = {"train": train, "heldout": heldout}
    exec(compile(recipe, document, "exec"), namespace)
    expected, _ = dense_reference(train, heldout, 0.05, 2)
    torch.testing.assert_close(namespace["result"].complete, expected)
    assert namespace["train"] is train and namespace["heldout"] is heldout
