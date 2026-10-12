"""Factorized conceptor fitting for offline CPU activation analysis.

Rows are observations and the last dimension is residual-stream width. Inputs
may be CPU float16, bfloat16, float32 or float64; all arithmetic and outputs are
float64.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import torch

_DTYPES = {torch.float16, torch.bfloat16, torch.float32, torch.float64}


def _positive_scalar(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and greater than zero")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return value


def _cpu_values(values: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(values, torch.Tensor):
        raise ValueError(f"{name} must be a CPU floating tensor")
    if values.device.type != "cpu" or values.layout != torch.strided:
        raise ValueError(f"{name} must be a strided CPU floating tensor; no device transfer is performed")
    if values.dtype not in _DTYPES:
        raise ValueError(f"{name} must have dtype float16, bfloat16, float32 or float64")
    if values.numel() == 0 or not bool(torch.isfinite(values).all()):
        raise ValueError(f"{name} must be nonempty and contain only finite values")
    return values.detach().to(dtype=torch.float64)


def _finite_result(values: torch.Tensor) -> torch.Tensor:
    if not bool(torch.isfinite(values).all()):
        raise ValueError("derived states exceed the float64 arithmetic range")
    return values


@dataclass(frozen=True)
class DirectionalGains:
    """Gains along economy-basis rows and outside their span.

    The represented gains are not necessarily the full ambient spectrum.
    Zero singular values within the basis also have gain beta.
    """

    represented: torch.Tensor
    orthogonal_nullspace_gain: float
    orthogonal_nullspace_dimension: int


@dataclass(frozen=True)
class FittedConceptor:
    """Fixed training center and factors; treat the stored tensors as read-only.

    ``basis`` has shape [min(n_fit, width), width] with orthonormal rows;
    ``coefficients`` contains c_i = lambda_i / (lambda_i + aperture^-2),
    where lambda_i = s_i^2 / n_fit. No dense covariance or operator is stored.
    """

    center: torch.Tensor
    basis: torch.Tensor
    coefficients: torch.Tensor
    aperture: float
    n_fit: int

    def _deviation(self, h: torch.Tensor) -> torch.Tensor:
        values = _cpu_values(h, "h")
        if values.ndim < 1 or values.shape[-1] != self.center.numel():
            raise ValueError(f"h must have shape [..., {self.center.numel()}]")
        return _finite_result(values - self.center)

    def _action(self, deviation: torch.Tensor) -> torch.Tensor:
        return _finite_result(((deviation @ self.basis.T) * self.coefficients) @ self.basis)

    def complete(self, h: torch.Tensor, beta: float = 2.0) -> torch.Tensor:
        """F_C(h) = m + beta (I - C)(h - m), using the training center."""
        beta = _positive_scalar(beta, "beta")
        deviation = self._deviation(h)
        return _finite_result(self.center + beta * (deviation - self._action(deviation)))

    def control(self, h: torch.Tensor, beta: float = 2.0) -> torch.Tensor:
        """F_0(h) = m + beta (h - m), sharing the same training center."""
        beta = _positive_scalar(beta, "beta")
        return _finite_result(self.center + beta * self._deviation(h))

    def correction(self, h: torch.Tensor, beta: float = 2.0) -> torch.Tensor:
        """Isolated correction -beta C(h - m); no subtraction of mapped outputs."""
        beta = _positive_scalar(beta, "beta")
        return _finite_result(-beta * self._action(self._deviation(h)))

    def directional_gains(self, beta: float = 2.0) -> DirectionalGains:
        beta = _positive_scalar(beta, "beta")
        return DirectionalGains(
            represented=beta * (1 - self.coefficients),
            orthogonal_nullspace_gain=beta,
            orthogonal_nullspace_dimension=self.center.numel() - self.basis.shape[0],
        )


def fit_conceptor(rows: torch.Tensor, aperture: float) -> FittedConceptor:
    """Fit C from nonempty [n, width] rows using R = Xc.T @ Xc / n.

    An economy SVD uses O(n*width + min(n,width)*width) storage plus SVD
    workspace. Neither fitting nor applying constructs a dense covariance or
    conceptor. The fit accepts one row (a zero centered covariance).
    """
    aperture = _positive_scalar(aperture, "aperture")
    values = _cpu_values(rows, "rows")
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("rows must have nonempty shape [n, width]")
    center = values.mean(0)
    centered = values - center
    if not bool(torch.isfinite(centered).all()):
        raise ValueError("centered rows exceed the float64 arithmetic range")
    _, singular_values, basis = torch.linalg.svd(centered, full_matrices=False)
    if not bool(torch.isfinite(singular_values).all()):
        raise ValueError("singular values exceed the float64 arithmetic range")
    # Evaluate sigmoid(log(alpha^2 * s^2 / n)) without exp of a positive
    # number; this also preserves representable subnormal coefficients.
    logits = 2 * (singular_values.log() + math.log(aperture)) - math.log(values.shape[0])
    small = torch.exp(-logits.abs())
    coefficients = torch.where(logits <= 0, small / (1 + small), 1 / (1 + small))
    if bool((coefficients == 1).any()):
        raise ValueError("aperture saturates float64 conceptor coefficients; choose a smaller aperture")
    if bool(((singular_values > 0) & (coefficients == 0)).any()):
        raise ValueError("aperture underflows float64 conceptor coefficients; choose a larger aperture")
    return FittedConceptor(center, basis, coefficients, aperture, values.shape[0])
