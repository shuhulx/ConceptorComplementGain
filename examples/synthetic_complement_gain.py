"""Inspect directional gains in a small anisotropic synthetic fit."""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence

import torch

from eval.compare_activations import compare_activations

TOY_APERTURE = 0.5
SCALES = (4.0, 3.0, 2.5, 1.5, 1.0, 0.6, 0.3, 0.1)


def synthetic_rows(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    width = len(SCALES)
    offset = 5 + 3 * torch.randn(width, generator=generator, dtype=torch.float64)
    rotation = torch.linalg.qr(torch.randn(width, width, generator=generator, dtype=torch.float64)).Q
    noise = torch.randn(400, width, generator=generator, dtype=torch.float64)
    rows = offset + (noise * torch.tensor(SCALES, dtype=torch.float64)) @ rotation.T
    return rows[:300], rows[300:]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--aperture", type=float, default=TOY_APERTURE)
    args = parser.parse_args(argv)
    if not math.isfinite(args.aperture) or args.aperture <= 0:
        parser.error("aperture must be finite and greater than zero")
    train, heldout = synthetic_rows(args.seed)
    result = compare_activations(train, heldout, aperture=args.aperture, beta=2)
    fitted = result.fitted
    gains = fitted.directional_gains(2).represented
    images = fitted.complete(fitted.center + fitted.basis, 2) - fitted.center
    measured = (images * fitted.basis).sum(1)
    gain_error = float((measured - gains).abs().max())
    print(f"Synthetic geometry; seed={args.seed}; aperture={args.aperture:g}; beta=2")
    print(" direction   coefficient   complete gain   matched-control gain")
    for index, (coefficient, gain) in enumerate(zip(fitted.coefficients, gains)):
        print(f" {index:9d}   {float(coefficient):11.4f}   {float(gain):13.4f}   {2:20.4f}")
    print(f"{int((gains < 1).sum())} of {len(gains)} directions attenuate (gain < 1).")
    print(f"Correction / matched displacement: {result.correction_ratio:.6f}")
    print(f"Gain error: {gain_error:.3e}; map identity error: {result.algebra_max_abs_error:.3e}")
    return int(max(gain_error, result.algebra_max_abs_error) > 1e-9)


if __name__ == "__main__":
    raise SystemExit(main())
