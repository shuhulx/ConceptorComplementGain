"""Prompt-final residual-stream extraction from a loaded TransformerLens model."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import torch


def _layers(layers: Sequence[int]) -> tuple[int, ...]:
    ordered = tuple(layers)
    if not ordered or any(type(layer) is not int or layer < 0 for layer in ordered):
        raise ValueError("layers must contain nonnegative integer block indices")
    if len(set(ordered)) != len(ordered):
        raise ValueError("layers must not contain duplicates")
    return ordered


def last_token_resid_pre(
    model: Any,
    prompts: Iterable[str],
    layers: Sequence[int],
    *,
    clear_cache: Callable[[], None] | None = None,
) -> dict[int, torch.Tensor]:
    """Return one prompt-final ``resid_pre`` row per prompt and layer.

    Calls ``model.run_with_cache`` separately for each string, in input order,
    selecting only ``blocks.<layer>.hook_resid_pre``. Results have shape
    ``[n_prompts, d_model]``, live on CPU, are detached and preserve cache dtype.
    Retained rows own their storage; outputs and caches are released before cleanup.
    A single prompt retains its row dimension. No chat wrapping or generation
    is added. Model tokenization (including its BOS defaults) is left intact.
    Call model.eval(); use torch.inference_mode() when gradients are unneeded.
    This helper leaves caller gradient mode unchanged.

    ``clear_cache``, when supplied, runs after each prompt's output and cache
    references have been released, before the next forward pass.
    """
    layers = _layers(layers)
    if isinstance(prompts, (str, bytes)):
        raise ValueError("prompts must be an iterable of nonempty strings, not one string")
    try:
        prompts = tuple(prompts)
    except TypeError as error:
        raise ValueError("prompts must be an iterable of nonempty strings") from error
    if not prompts or any(not isinstance(prompt, str) or not prompt for prompt in prompts):
        raise ValueError("at least one prompt is required; each must be a nonempty string")
    n_layers = getattr(getattr(model, "cfg", None), "n_layers", None)
    if n_layers is not None and max(layers) >= n_layers:
        raise ValueError("layers must be valid zero-based model block indices")
    if isinstance(model, torch.nn.Module) and any(module.training for module in model.modules()):
        raise ValueError("call model.eval() before extracting activations")
    activations = {li: [] for li in layers}
    names = [f"blocks.{li}.hook_resid_pre" for li in layers]
    for prompt in prompts:
        output, cache = model.run_with_cache(prompt, names_filter=lambda n: n in names)
        del output
        for li in layers:
            act = cache[f"blocks.{li}.hook_resid_pre"][:, -1, :].detach()
            if act.ndim != 2 or act.shape[0] != 1:
                raise ValueError("each prompt must produce a single unbatched cache row")
            activations[li].append(act.squeeze(0).to(device="cpu", copy=True))
            del act
        del cache
        if clear_cache is not None:
            clear_cache()
    if not activations[layers[0]]:
        raise ValueError("at least one prompt is required")
    return {li: torch.stack(activations[li]) for li in layers}
