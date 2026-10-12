"""Optional real-model API tests with random CPU weights and a local tokenizer."""

import ast
import os
import socket
from importlib.metadata import version
from pathlib import Path

import pytest
import torch

from conceptor_tools import fit_conceptor
from eval.compare_generations import compare_generations
from extract_activations import last_token_resid_pre

required = os.environ.get("REQUIRE_TRANSFORMERLENS") == "1"
if required:
    import transformer_lens as tl
else:
    tl = pytest.importorskip("transformer_lens", reason="optional own-model dependencies not installed")
if version("transformer-lens") != "3.5.1":
    if required:
        raise RuntimeError("integration tests require the optional TransformerLens 3.5.1 stack")
    pytest.skip("integration tests use the optional TransformerLens 3.5.1 stack", allow_module_level=True)


@pytest.fixture
def model(monkeypatch):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("WANDB_MODE", "disabled")

    def no_network(*args, **kwargs):
        raise AssertionError("integration test attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    words = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "small", "large", "cold", "hot",
             "left", "right", "up", "down", ":", ",", "near", "far"]
    tokenizer = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(single="[BOS] $A", special_tokens=[("[BOS]", 2)])
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]",
        bos_token="[BOS]", eos_token="[EOS]", name_or_path="local-test-vocabulary",
        add_bos_token=True,
    )
    before = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(871)
        cfg = tl.HookedTransformerConfig(
            n_layers=2, d_model=16, n_ctx=32, d_head=4, n_heads=4, d_mlp=32,
            d_vocab=len(words), act_fn="relu", normalization_type="LN",
            seed=None, device="cpu", default_prepend_bos=True,
        )
        result = tl.HookedTransformer(cfg, tokenizer=tokenizer).eval()
    assert torch.equal(torch.get_rng_state(), before)
    return result


HOOK_NAME = "blocks.1.hook_resid_pre"
PROMPTS = ["small : large", "left : right"]
OPTIONS = dict(layer=1, seeds=(12, 19), max_new_tokens=4, stop_at_eos=False)


@pytest.mark.parametrize("prompt", ["small : large", "small       :    large"])
def test_broad_study_completion_uses_actual_token_boundary(model, prompt):
    source = Path(__file__).resolve().parents[1] / "experiments/broad_study/run_study.py"
    tree = ast.parse(source.read_text())
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SteerableModel")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    tokens = model.to_tokens(prompt)
    assert tokens[0, 0].item() == model.tokenizer.bos_token_id
    with torch.inference_mode():
        output = model.generate(tokens, max_new_tokens=3, do_sample=False,
                                temperature=0.7, top_p=0.95, verbose=False,
                                return_type="tokens")
        assert torch.equal(output[:, :tokens.shape[1]], tokens)
        expected = model.to_string(output[0, tokens.shape[1]:]).strip()
        actual = namespace["SteerableModel"](model).generate(prompt, max_new_tokens=3, do_sample=False)
    assert actual == expected


@pytest.mark.parametrize("default", [True, False])
def test_real_prompt_final_extraction_matches_cache(model, default):
    model.cfg.default_prepend_bos = default
    with torch.inference_mode():
        actual = last_token_resid_pre(model, PROMPTS, (0, 1))
        expected = {0: [], 1: []}
        for prompt in PROMPTS:
            _, cache = model.run_with_cache(prompt)
            for layer in expected:
                expected[layer].append(cache[f"blocks.{layer}.hook_resid_pre"][0, -1].cpu())
    for layer in expected:
        torch.testing.assert_close(actual[layer], torch.stack(expected[layer]), rtol=0, atol=0)
        assert actual[layer].shape == (len(PROMPTS), model.cfg.d_model)
        assert actual[layer].device.type == "cpu"


def test_real_zero_map_pairing_and_lifecycle(model):
    parameters = {k: v.detach().clone() for k, v in model.named_parameters()}
    observations = []

    def observe(h, hook):
        observations.append(h.shape)
        return h

    with model.hooks(fwd_hooks=[(HOOK_NAME, observe)]):
        old_hooks = list(model.hook_dict[HOOK_NAME].fwd_hooks)
        rng = torch.get_rng_state().clone()
        zero = fit_conceptor(torch.ones(3, 16), 0.05)
        pairs = compare_generations(model, zero, PROMPTS, **OPTIONS)
        assert len(pairs) == 4 and all(p.complete == p.control for p in pairs)
        assert torch.equal(torch.get_rng_state(), rng)
        assert model.hook_dict[HOOK_NAME].fwd_hooks == old_hooks
        assert pairs == compare_generations(model, zero, PROMPTS, **OPTIONS)
        assert torch.equal(torch.get_rng_state(), rng)
        assert any(shape[1] > 1 for shape in observations)
        assert any(shape[1] == 1 for shape in observations)
    assert not model.hook_dict[HOOK_NAME].fwd_hooks
    assert not any(module.training for module in model.modules())
    assert torch.is_grad_enabled()
    for key, parameter in model.named_parameters():
        assert torch.equal(parameter, parameters[key])


def test_real_generations_match_independent_dense_maps(model):
    rows = torch.randn(8, 16, generator=torch.Generator().manual_seed(52)) + 0.4
    fitted = fit_conceptor(rows, 0.3)
    actual = compare_generations(model, fitted, PROMPTS[:1], **OPTIONS)
    center = rows.double().mean(0)
    x = rows.double() - center
    covariance = x.T @ x / len(rows)
    conceptor = torch.linalg.solve(
        covariance + torch.eye(16, dtype=torch.float64) / 0.3**2, covariance,
    )

    def independent_hook(complete):
        def transform(h, hook):
            z = h.float() - center.float()
            corrected = z - z @ conceptor.float().T if complete else z
            return (center.float() + 2 * corrected).to(h.dtype)
        return transform

    for pair in actual:
        for complete, expected in ((True, pair.complete), (False, pair.control)):
            with torch.random.fork_rng(devices=[]), torch.inference_mode():
                torch.random.default_generator.manual_seed(pair.seed)
                with model.hooks(fwd_hooks=[(HOOK_NAME, independent_hook(complete))]):
                    generated = model.generate(
                        pair.prompt, max_new_tokens=4, do_sample=True, stop_at_eos=False,
                        prepend_bos=True, return_type="str", verbose=False,
                        temperature=0.7, top_p=0.95, top_k=None,
                    )
            assert generated == expected


def test_real_hook_failure_restores_state(model):
    calls = 0
    parameters = {k: v.detach().clone() for k, v in model.named_parameters()}

    def fail(h, hook):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("deliberate integration failure")
        return h

    with model.hooks(fwd_hooks=[(HOOK_NAME, fail)]):
        old_hooks = list(model.hook_dict[HOOK_NAME].fwd_hooks)
        rng = torch.get_rng_state().clone()
        with pytest.raises(RuntimeError, match="deliberate integration failure"):
            compare_generations(model, torch.ones(3, 16), PROMPTS[:1], aperture=0.05, **OPTIONS)
        assert model.hook_dict[HOOK_NAME].fwd_hooks == old_hooks
        assert torch.equal(torch.get_rng_state(), rng)
    assert not model.hook_dict[HOOK_NAME].fwd_hooks
    assert not any(module.training for module in model.modules())
    assert torch.is_grad_enabled()
    for key, parameter in model.named_parameters():
        assert torch.equal(parameter, parameters[key])


@pytest.mark.parametrize("default", [True, False])
@pytest.mark.parametrize("override", [None, True, False])
def test_real_bos_defaults_and_explicit_overrides(model, default, override):
    model.cfg.default_prepend_bos = default
    prompt = PROMPTS[0]
    token_options = {} if override is None else {"prepend_bos": override}
    expected_tokens = model.to_tokens(prompt, **token_options).shape[1]
    lengths = []

    def observe(h, hook):
        lengths.append(h.shape[1])
        return h

    fitted = fit_conceptor(torch.ones(3, 16), 0.05)
    with model.hooks(fwd_hooks=[(HOOK_NAME, observe)]):
        pair, = compare_generations(
            model, fitted, [prompt], layer=1, seeds=(12,), max_new_tokens=3,
            stop_at_eos=False, prepend_bos=override,
        )
    assert lengths[0] == expected_tokens

    def dense_control(h, hook):
        return 1 + 2 * (h - 1)

    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        torch.random.default_generator.manual_seed(12)
        with model.hooks(fwd_hooks=[(HOOK_NAME, dense_control)]):
            expected = model.generate(
                prompt, max_new_tokens=3, do_sample=True, stop_at_eos=False,
                return_type="str", verbose=False, temperature=0.7,
                top_p=0.95, top_k=None, **token_options,
            )
    assert pair.complete == pair.control == expected
