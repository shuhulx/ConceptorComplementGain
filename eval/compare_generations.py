"""Compare generated outputs with the fitted matrix present or removed."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from activation_steer import _positive, make_matched_hooks
from conceptor_tools import FittedConceptor, fit_conceptor


@dataclass(frozen=True)
class GenerationPair:
    prompt: str
    seed: int
    complete: str
    control: str


def compare_generations(
    model: Any,
    train: torch.Tensor | FittedConceptor,
    prompts: Sequence[str],
    *,
    layer: int,
    seeds: Sequence[int],
    aperture: float | None = None,
    beta: float = 2.0,
    compute_dtype: torch.dtype = torch.float32,
    max_new_tokens: int = 64,
    do_sample: bool = True,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int | None = None,
    stop_at_eos: bool = True,
    prepend_bos: bool | None = None,
) -> list[GenerationPair]:
    """Generate one pair per prompt and seed, without changing caller RNG/hooks.

    model must be an eval-mode TransformerLens-compatible torch module on CPU
    or CUDA, with cfg, hook_dict, hooks() and generate(). Pre-existing hooks
    remain active. Plain prompt strings are passed unchanged, without chat
    formatting. Results are full strings returned by generate(), including
    the prompt when the model returns it.
    With prepend_bos=None, omit the keyword to inherit model tokenization defaults.

    Each arm starts from the same Torch seed; divergent trajectories need not
    consume identical random draws. Caller CPU and model CUDA RNG states are
    restored even on failure. Python/NumPy randomness in custom model code or
    existing hooks is outside this contract. Do not call concurrently on the
    same model or while another thread is using these global RNG generators.
    """
    if not isinstance(model, torch.nn.Module):
        raise ValueError("model must be a TransformerLens-compatible torch module")
    if any(module.training for module in model.modules()):
        raise ValueError("call model.eval() before comparing generations")
    if (isinstance(layer, bool) or not isinstance(layer, int) or layer < 0
            or layer >= getattr(getattr(model, "cfg", None), "n_layers", 0)):
        raise ValueError("layer must be a valid zero-based model block index")
    hook_name = f"blocks.{layer}.hook_resid_pre"
    if (hook_name not in getattr(model, "hook_dict", {})
            or not callable(getattr(model, "hooks", None))
            or not callable(getattr(model, "generate", None))):
        raise ValueError("model needs TransformerLens residual-pre hooks and generate")
    if isinstance(prompts, (str, bytes)) or not isinstance(prompts, Sequence):
        raise ValueError("prompts must be a nonempty sequence of nonempty strings")
    prompts = tuple(prompts)
    if not prompts or any(not isinstance(p, str) or not p for p in prompts):
        raise ValueError("prompts must be a nonempty sequence of nonempty strings")
    if isinstance(seeds, (str, bytes)) or not isinstance(seeds, Sequence):
        raise ValueError("seeds must be a nonempty sequence of distinct integers")
    seeds = tuple(seeds)
    if (not seeds or any(isinstance(s, bool) or not isinstance(s, int)
                         or not 0 <= s < 2**63 for s in seeds) or len(set(seeds)) != len(seeds)):
        raise ValueError("seeds must be distinct integers in [0, 2**63)")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    if any(not isinstance(v, bool) for v in (do_sample, stop_at_eos)):
        raise ValueError("do_sample and stop_at_eos must be booleans")
    if prepend_bos is not None and not isinstance(prepend_bos, bool):
        raise ValueError("prepend_bos must be None or a boolean")
    temperature = _positive(temperature, "temperature")
    top_p = _positive(top_p, "top_p")
    if top_p > 1:
        raise ValueError("top_p must be at most 1")
    if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1):
        raise ValueError("top_k must be None or a positive integer")
    devices = {p.device for p in (*model.parameters(), *model.buffers())}
    cfg_device = getattr(model.cfg, "device", None)
    if cfg_device is not None:
        devices.add(torch.device(cfg_device))
    if any(d.type not in ("cpu", "cuda") for d in devices):
        raise ValueError("paired RNG preservation supports CPU and CUDA models only")
    cuda_devices = sorted({d.index if d.index is not None else torch.cuda.current_device()
                           for d in devices if d.type == "cuda"})
    if isinstance(train, FittedConceptor):
        if aperture is not None:
            raise ValueError("omit aperture when passing an already fitted conceptor")
        fitted = train
    else:
        if aperture is None:
            raise ValueError("aperture is required when fitting training rows")
        fitted = fit_conceptor(train, aperture)
    hooks = make_matched_hooks(fitted, beta=beta, compute_dtype=compute_dtype)
    if fitted.center.numel() != getattr(model.cfg, "d_model", None):
        raise ValueError("training width must match model.cfg.d_model")
    options = dict(max_new_tokens=max_new_tokens, do_sample=do_sample,
                   stop_at_eos=stop_at_eos, return_type="str", verbose=False)
    if prepend_bos is not None:
        options["prepend_bos"] = prepend_bos
    if do_sample:
        options.update(temperature=temperature, top_p=top_p, top_k=top_k)
    pairs = []
    with torch.random.fork_rng(devices=cuda_devices), torch.inference_mode():
        for prompt in prompts:
            for seed in seeds:
                outputs = {}
                for arm, hook in hooks.items():
                    torch.random.default_generator.manual_seed(seed)
                    for device in cuda_devices:
                        torch.cuda.default_generators[device].manual_seed(seed)
                    with model.hooks(fwd_hooks=[(hook_name, hook)]):
                        output = model.generate(prompt, **options)
                    if not isinstance(output, str):
                        raise ValueError("model.generate must return a string for a string prompt")
                    outputs[arm] = output
                pairs.append(GenerationPair(prompt, seed, outputs["complete"], outputs["control"]))
    return pairs
