"""Compare fitted maps on separate synthetic training and held-out rows."""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence

import torch

from eval.compare_activations import compare_activations


def synthetic_rows(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    scales = torch.linspace(3.0, 0.1, 128, dtype=torch.float32)
    rows = 5 + torch.randn(100, 128, generator=generator, dtype=torch.float32) * scales
    return rows[:80], rows[80:]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--aperture", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=2.0)
    args = parser.parse_args(argv)
    try:
        train, heldout = synthetic_rows(args.seed)
        result = compare_activations(train, heldout, aperture=args.aperture, beta=args.beta)
    except ValueError as error:
        parser.error(str(error))
    gains = result.fitted.directional_gains(args.beta)
    print("Synthetic float32 activation comparison")
    print(f"seed={args.seed}; fit={tuple(train.shape)}; heldout={tuple(heldout.shape)}")
    print(f"CPU float64 analysis; aperture={args.aperture:g}; beta={args.beta:g}")
    print(
        f"Economy basis: {len(gains.represented)} represented directions; complete gains "
        f"{float(gains.represented.min()):.6f} to {float(gains.represented.max()):.6f}."
    )
    print(
        f"Outside that basis: {gains.orthogonal_nullspace_dimension} orthogonal nullspace "
        f"directions, each with gain {gains.orthogonal_nullspace_gain:g}; "
        "represented gains alone are not the full ambient spectrum."
    )
    print(f"Matched-control gain in every direction: {args.beta:g}")
    print(f"Correction RMS: {result.correction_rms:.6f}")
    print(f"Matched displacement RMS: {result.matched_displacement_rms:.6f}")
    if result.correction_ratio is None:
        print(f"Correction / matched displacement: undefined ({result.ratio_undefined_reason})")
    else:
        print(f"Correction / matched displacement: {result.correction_ratio:.6f}")
    print(f"Algebra max error |F_C - F_0 + beta C(h-m)|: {result.algebra_max_abs_error:.3e}")
    return 0 if math.isfinite(result.algebra_max_abs_error) and result.algebra_max_abs_error < 1e-9 else 1


if __name__ == "__main__":
    raise SystemExit(main())
