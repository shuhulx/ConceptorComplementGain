"""Activation extraction for an already loaded TransformerLens-like model.

Prompt-final extraction returns CPU rows in prompt order and the cache dtype.
Persona extraction returns CPU float32 generation-event averages in template,
question and polarity order. Prompt-final rows are keyed by layer; persona rows
are keyed by polarity, then layer. Each tensor has shape [rows, width]; the two
extraction modes measure different activation constructions.
Caller gradient mode and RNG state are not reset by these helpers.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

from extract_activations import _layers, last_token_resid_pre  # noqa: F401


def persona_generation_resid_pre(
    model: Any,
    trait_name: str,
    trait_data: Mapping[str, Any],
    layers: Sequence[int],
    *,
    n_instruction_pairs: int = 5,
    n_extraction_questions: int = 20,
    progress: Any = None,
    clear_cache: Callable[[], None] | None = None,
) -> dict[str, dict[int, torch.Tensor]]:
    """Extract in template-major, question-major, positive-then-negative order.

    Each row averages the final-position state at *every hook event* observed
    during a generation (including initial prompt prefill), with at most 50
    new tokens, temperature .7 and sampling enabled. Early EOS therefore
    produces fewer events. The hook detaches to CPU float32 and returns the
    original tensor unchanged. Results are ``{pos/neg: {layer: [rows, width]}}``.

    The model's hook context manages cleanup on success and failure. Exceptions
    propagate; no retry, reseeding, batching or gradient-mode changes occur.
    """
    layers = _layers(layers)
    extractions = {"pos": {li: [] for li in layers}, "neg": {li: [] for li in layers}}
    pairs_to_use = trait_data["instructions"][:n_instruction_pairs]
    questions = trait_data["extract_questions"][:n_extraction_questions]
    for pair in pairs_to_use:
        for question in questions:
            for polarity in ["pos", "neg"]:
                messages = [
                    {
                        "role": "system",
                        "content": (
                            f"You are a {trait_name if polarity == 'pos' else 'helpful'} "
                            f"assistant. {pair[polarity]}"
                        ),
                    },
                    {"role": "user", "content": question},
                ]
                prompt = model.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                captured = {li: [] for li in layers}

                def make_hook(li, storage=captured):
                    def hook_fn(h, hook):
                        storage[li].append(
                            h[:, -1, :].detach().to(device="cpu", dtype=torch.float32, copy=True)
                        )
                        return h
                    return hook_fn

                fwd_hooks = [(f"blocks.{li}.hook_resid_pre", make_hook(li)) for li in layers]
                tokens = model.to_tokens(prompt)
                with model.hooks(fwd_hooks=fwd_hooks):
                    model.generate(
                        tokens, max_new_tokens=50, temperature=0.7,
                        do_sample=True, verbose=False,
                    )
                for li in layers:
                    if captured[li]:
                        avg_act = torch.stack(captured[li]).mean(dim=0).squeeze(0)
                        extractions[polarity][li].append(avg_act)
                del captured, fwd_hooks, make_hook, tokens
                if clear_cache is not None:
                    clear_cache()
                if progress is not None:
                    progress.update(1)
    if progress is not None:
        progress.close()
    return {
        pol: {li: torch.stack(extractions[pol][li]) for li in layers}
        for pol in ["pos", "neg"]
    }
