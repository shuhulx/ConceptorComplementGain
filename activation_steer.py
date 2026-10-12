"""Complete-map and matched-control residual-stream hooks."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

import torch

from conceptor_tools import FittedConceptor


def _positive(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and positive")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def make_matched_hooks(
    fitted: FittedConceptor, *, beta: float = 2.0, compute_dtype: torch.dtype = torch.float32,
) -> dict[str, Any]:
    """Return inference-use complete/control hooks sharing center, gain and arithmetic.

    Inputs are [batch, tokens, width], with every token transformed. Factors
    are copied once per device; states never travel to CPU. Arithmetic uses
    compute_dtype (float32 or float64) even under caller autocast, then casts
    back to the input dtype.
    Do not reuse a device cache created under inference_mode for later autograd.
    """
    beta = _positive(beta, "beta")
    if compute_dtype not in (torch.float32, torch.float64):
        raise ValueError("compute_dtype must be float32 or float64")
    if not isinstance(fitted, FittedConceptor):
        raise ValueError("fitted must be a FittedConceptor from fit_conceptor")
    tensors = (fitted.center, fitted.basis, fitted.coefficients)
    if any(not isinstance(t, torch.Tensor) or t.layout != torch.strided
           or not t.is_floating_point() or not bool(torch.isfinite(t).all()) for t in tensors):
        raise ValueError("conceptor factors must be finite floating tensors")
    center, basis, coefficients = (t.detach().clone() for t in tensors)
    if (center.ndim != 1 or center.numel() == 0 or basis.ndim != 2
            or basis.shape[1] != center.numel() or basis.shape[0] == 0
            or coefficients.shape != (basis.shape[0],)
            or bool(((coefficients < 0) | (coefficients >= 1)).any())):
        raise ValueError("invalid conceptor factor shapes or coefficients")
    cache = {}

    def mapped(h: torch.Tensor, *, complete: bool) -> torch.Tensor:
        if (not isinstance(h, torch.Tensor) or h.ndim != 3 or h.numel() == 0
                or h.shape[-1] != center.numel() or h.layout != torch.strided
                or h.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)):
            raise ValueError("hook input must be floating [batch, tokens, fitted width]")
        if not bool(torch.isfinite(h).all()):
            raise ValueError("hook input must contain finite states")
        if h.device not in cache:
            converted = tuple(t.to(device=h.device, dtype=compute_dtype) for t in
                              (center, basis, coefficients))
            if any(not bool(torch.isfinite(t).all()) for t in converted):
                raise ValueError("conceptor factors overflow compute_dtype")
            cache[h.device] = (*converted, torch.zeros_like(converted[2]))
        m, v, c, zero = cache[h.device]
        with torch.autocast(device_type=h.device.type, enabled=False):
            z = h.to(compute_dtype) - m
            correction = ((z @ v.T) * (c if complete else zero)) @ v
            result = (m + beta * (z - correction)).to(h.dtype)
        if not bool(torch.isfinite(result).all()):
            raise ValueError("mapped states overflow the selected arithmetic or input dtype")
        return result

    def complete(h, hook):
        return mapped(h, complete=True)

    def control(h, hook):
        return mapped(h, complete=False)

    return {"complete": complete, "control": control}
