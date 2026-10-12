#!/usr/bin/env python3
"""Extract prompt-final residual-input activations from a TransformerLens model."""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterable, Sequence
from typing import Any

SYNTHETIC_PROMPTS = (
    "apple:fruit, carrot:vegetable, pear:",
    "small:large, cold:hot, short:",
    "one:1, two:2, three:",
)


def extract_from_model(
    model: Any,
    prompts: Iterable[str] = SYNTHETIC_PROMPTS,
    layers: Sequence[int] = (4,),
):
    """Return CPU prompt-final rows using an already loaded compatible model.

    Returns ``{layer: [n_prompts, d_model]}``, preserving prompt order and cache
    dtype. Each row is the final prompt position from a separate forward pass.
    """
    from extract_activations import last_token_resid_pre

    return last_token_resid_pre(model, prompts, layers)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="TransformerLens-supported model name")
    parser.add_argument(
        "--layer", type=int, action="append",
        help="zero-based block index; repeat for multiple layers (default: 4)",
    )
    parser.add_argument("--device", default="cpu", help="model device (default: cpu)")
    parser.add_argument(
        "--revision", help="optional immutable 40-character Hugging Face commit SHA"
    )
    args = parser.parse_args(argv)
    layers = args.layer if args.layer is not None else [4]
    if any(layer < 0 for layer in layers) or len(set(layers)) != len(layers):
        parser.error("--layer values must be nonnegative and distinct")
    if args.revision is not None and re.fullmatch(r"[0-9a-fA-F]{40}", args.revision) is None:
        parser.error("--revision must be a full 40-character Hugging Face commit SHA")
    if __name__ == "__main__" and not __package__:
        parser.error("use python -m examples.extract_activations from the repository root")

    import torch
    from transformer_lens import HookedTransformer

    kwargs = {"device": args.device}
    if args.revision is not None:
        kwargs["revision"] = args.revision.lower()
    model = HookedTransformer.from_pretrained(args.model, **kwargs)
    model.eval()
    if max(layers) >= model.cfg.n_layers:
        parser.error(f"--layer must be below this model's {model.cfg.n_layers} blocks")
    with torch.inference_mode():
        activations = extract_from_model(model, SYNTHETIC_PROMPTS, layers)
    print("Prompt-final activations for three synthetic example prompts.")
    for layer, rows in activations.items():
        print(f"Layer {layer}: shape={tuple(rows.shape)}, dtype={rows.dtype}, device={rows.device}")
        for index, row in enumerate(rows):
            print(f"  Prompt {index + 1}: norm={row.float().norm().item():.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
