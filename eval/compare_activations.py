"""Common-input comparisons of complete and matched-control maps."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from conceptor_tools import FittedConceptor, fit_conceptor


@dataclass(frozen=True)
class ActivationComparison:
    fitted: FittedConceptor
    complete: torch.Tensor
    control: torch.Tensor
    correction: torch.Tensor
    correction_rms: float
    matched_displacement_rms: float
    correction_ratio: float | None
    ratio_undefined_reason: str | None
    algebra_max_abs_error: float


def _rms(values: torch.Tensor) -> float:
    if not bool(torch.isfinite(values).all()):
        raise ValueError("diagnostic states exceed the float64 arithmetic range")
    scale = float(values.abs().max())
    return 0.0 if scale == 0 else scale * float((values / scale).square().mean().sqrt())


def compare_activations(
    train: torch.Tensor, heldout: torch.Tensor, *, aperture: float = 0.05, beta: float = 2.0
) -> ActivationComparison:
    """Fit training rows and evaluate both maps at identical held-out states.

    Held-out inputs have shape [..., width], including a single vector. RMS
    pools all elements. The correction ratio divides correction RMS by the
    matched displacement RMS, RMS((beta - 1)(heldout - training mean)).
    An undefined ratio is None with an explicit reason, including at beta=1.
    """
    fitted = fit_conceptor(train, aperture)
    complete = fitted.complete(heldout, beta)
    control = fitted.control(heldout, beta)
    correction = fitted.correction(heldout, beta)
    displacement = (beta - 1) * (heldout.detach().to(dtype=torch.float64) - fitted.center)
    correction_rms = _rms(correction)
    displacement_rms = _rms(displacement)
    reason = None
    if beta == 1:
        reason = "beta=1 gives zero matched-control displacement"
    elif displacement_rms == 0:
        reason = "held-out states have zero matched-control displacement"
    ratio = None if reason is not None else correction_rms / displacement_rms
    error = float((complete - control - correction).abs().max())
    if not math.isfinite(error) or (ratio is not None and not math.isfinite(ratio)):
        raise ValueError("comparison diagnostics exceed the float64 arithmetic range")
    return ActivationComparison(
        fitted, complete, control, correction, correction_rms, displacement_rms,
        ratio, reason, error,
    )
