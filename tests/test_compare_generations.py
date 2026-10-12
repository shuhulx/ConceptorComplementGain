"""Dense-map, RNG and hook-lifecycle tests using a small fake model."""

from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from activation_steer import make_matched_hooks
from conceptor_tools import fit_conceptor
from eval.compare_generations import compare_generations


def training(n=7, width=4):
    return 3 + torch.randn(n, width, generator=torch.Generator().manual_seed(12)).double()


class FakeHookedModel(torch.nn.Module):
    """Minimal TransformerLens hook scopes plus stochastic generation for tests."""

    def __init__(self, device="cpu"):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4, device=device))
        self.cfg = SimpleNamespace(n_layers=2, d_model=4, device=device)
        self.hook_dict = {"blocks.1.hook_resid_pre": object()}
        self.active = []
        self.calls = []
        self.fail_on = None
        self.eval()

    @contextmanager
    def hooks(self, fwd_hooks):
        previous = list(self.active)
        self.active.extend(fwd_hooks)
        try:
            yield self
        finally:
            self.active[:] = previous

    def generate(self, prompt, **options):
        cpu_draw = torch.rand(3)
        device_draw = torch.rand(1, device=self.weight.device)
        self.calls.append((prompt, options.copy(), cpu_draw.clone(), device_draw.clone()))
        h = torch.ones(1, 3, 4, device=self.weight.device) * len(prompt)
        for _, hook in self.active:
            h = hook(h, None)
        if len(self.calls) == self.fail_on:
            raise RuntimeError("generation failed")
        return f"{prompt} {h.sum().item():.6f} {cpu_draw.tolist()} {device_draw.tolist()}"


@pytest.mark.parametrize("n,width", [(1, 4), (3, 8), (11, 4)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_both_all_token_maps_match_independent_dense_solve(n, width, dtype):
    rows = training(n, width)
    rows[:, -1] = rows[:, 0]  # Include rank-deficient fits.
    fitted = fit_conceptor(rows, 0.6)
    before = [v.clone() for v in (fitted.center, fitted.basis, fitted.coefficients)]
    h = torch.arange(6 * width).reshape(2, 3, width).to(dtype) / 10
    original = h.clone()
    m = rows.mean(0)
    x = rows - m
    r = x.T @ x / n
    identity = torch.eye(width, dtype=torch.float64)
    c = torch.linalg.solve(r + identity / 0.6**2, r)
    hooks = make_matched_hooks(fitted, beta=1.7, compute_dtype=torch.float64)
    for arm, matrix in (("complete", identity - c), ("control", identity)):
        actual = hooks[arm](h, None)
        expected = (m + 1.7 * ((h.double() - m) @ matrix.T)).to(dtype)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        assert actual.shape == h.shape and actual.dtype == h.dtype and actual.device == h.device
        assert actual.data_ptr() != h.data_ptr()
    torch.testing.assert_close(h, original, rtol=0, atol=0)
    for v, old in zip((fitted.center, fitted.basis, fitted.coefficients), before):
        torch.testing.assert_close(v, old, rtol=0, atol=0)


def test_float32_default_and_factor_snapshot():
    fitted = fit_conceptor(training(), 0.2)
    hooks = make_matched_hooks(fitted)
    h = training(6).reshape(2, 3, 4).half()
    expected = fitted.complete(h, 2).half()
    torch.testing.assert_close(hooks["complete"](h, None), expected)
    cached = hooks["complete"](h, None)
    fitted.center.add_(100)
    fitted.coefficients.zero_()
    torch.testing.assert_close(hooks["complete"](h, None), cached, rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
)])
@pytest.mark.parametrize("compute_dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("input_dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_hooks_preserve_compute_dtype_under_autocast(device, compute_dtype, input_dtype):
    generator = torch.Generator().manual_seed(192)
    fitted = fit_conceptor(torch.randn(100, 64, generator=generator), 0.5)
    h = torch.randn(2, 4, 64, generator=generator).to(device=device, dtype=input_dtype)
    hooks = make_matched_hooks(fitted, compute_dtype=compute_dtype)
    reference = make_matched_hooks(fitted, compute_dtype=compute_dtype)
    with torch.autocast(device_type=device, enabled=False):
        expected = {arm: hook(h, None) for arm, hook in reference.items()}
    enabled_before = torch.is_autocast_enabled(device)
    dtype_before = torch.get_autocast_dtype(device)
    autocast_dtype = torch.bfloat16 if device == "cpu" else torch.float16
    with torch.autocast(device_type=device, dtype=autocast_dtype):
        for _ in range(2):  # First use populates the cache; the second reuses it.
            for arm, hook in hooks.items():
                actual = hook(h, None)
                assert actual.shape == h.shape and actual.dtype == input_dtype
                assert actual.device == h.device
                torch.testing.assert_close(actual, expected[arm], rtol=0, atol=0)
                assert torch.is_autocast_enabled(device)
                assert torch.get_autocast_dtype(device) == autocast_dtype
        overflow = torch.full_like(h, torch.finfo(input_dtype).max)
        with pytest.raises(ValueError, match="overflow"):
            hooks["control"](overflow, None)
        assert torch.is_autocast_enabled(device)
        assert torch.get_autocast_dtype(device) == autocast_dtype
    assert torch.is_autocast_enabled(device) == enabled_before
    assert torch.get_autocast_dtype(device) == dtype_before


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_zero_conceptor_arms_are_bitwise_equal(dtype):
    hooks = make_matched_hooks(fit_conceptor(torch.full((3, 4), 2.0), 0.1))
    h = training(6).reshape(2, 3, 4).to(dtype)
    assert torch.equal(hooks["complete"](h, None), hooks["control"](h, None))


@pytest.mark.parametrize("fail_on", [None, 1, 2, 3, 4])
def test_pair_settings_rng_and_hook_cleanup(fail_on):
    model = FakeHookedModel()
    calls = []

    def existing(h, hook):
        calls.append(h.shape)
        return h + 0.1

    model.active.append(("blocks.1.hook_resid_pre", existing))
    before_rng = torch.get_rng_state().clone()
    before_weight = model.weight.detach().clone()
    model.fail_on = fail_on
    arguments = dict(layer=1, seeds=(42, 43), aperture=0.1)
    if fail_on is not None:
        with pytest.raises(RuntimeError, match="generation failed"):
            compare_generations(model, training(), ["hello", "next"], **arguments)
    else:
        result = compare_generations(model, training(), ["hello", "next"], **arguments)
        assert [(p.prompt, p.seed) for p in result] == [
            ("hello", 42), ("hello", 43), ("next", 42), ("next", 43)]
        for start in range(0, 8, 2):
            a, b = model.calls[start:start + 2]
            assert a[:2] == b[:2]
            torch.testing.assert_close(a[2], b[2], rtol=0, atol=0)
            torch.testing.assert_close(a[3], b[3], rtol=0, atol=0)
        assert not torch.equal(model.calls[0][2], model.calls[2][2])
        assert compare_generations(model, training(), ["hello", "next"], **arguments) == result
    assert calls
    assert model.active == [("blocks.1.hook_resid_pre", existing)]
    assert not model.training and torch.is_grad_enabled()
    torch.testing.assert_close(torch.get_rng_state(), before_rng, rtol=0, atol=0)
    torch.testing.assert_close(model.weight, before_weight, rtol=0, atol=0)


def test_prefit_greedy_and_existing_inference_context():
    model = FakeHookedModel()
    fitted = fit_conceptor(training(), 0.1)
    factors = [t.clone() for t in (fitted.center, fitted.basis, fitted.coefficients)]
    with torch.inference_mode():
        compare_generations(model, fitted, ["hello"], layer=1, seeds=(0,), do_sample=False)
        assert torch.is_inference_mode_enabled()
    assert not torch.is_inference_mode_enabled()
    assert "temperature" not in model.calls[0][1]
    assert model.calls[0][1]["do_sample"] is False
    for old, current in zip(factors, (fitted.center, fitted.basis, fitted.coefficients)):
        torch.testing.assert_close(old, current, rtol=0, atol=0)


@pytest.mark.parametrize("override", [
    {"layer": -1}, {"layer": True}, {"layer": 2}, {"seeds": ()}, {"seeds": (2, 2)},
    {"seeds": (True,)}, {"seeds": (-1,)}, {"seeds": (2**63,)}, {"prompts": "hello"},
    {"prompts": []}, {"prompts": [""]}, {"max_new_tokens": 0}, {"max_new_tokens": True},
    {"do_sample": 1}, {"temperature": 0}, {"temperature": float("nan")}, {"top_p": 2},
    {"top_k": 0}, {"top_k": True}, {"beta": 0}, {"compute_dtype": torch.float16},
    {"aperture": None}, {"prepend_bos": 1}, {"prepend_bos": "false"},
])
def test_invalid_arguments_before_generation(override):
    model = FakeHookedModel()
    args = dict(prompts=["hello"], layer=1, seeds=(42,), aperture=0.1)
    args.update(override)
    with pytest.raises(ValueError):
        compare_generations(model, training(), **args)
    assert not model.calls and not model.active


def test_invalid_model_width_training_mode_and_prefit_aperture():
    model = FakeHookedModel()
    args = dict(prompts=["hello"], layer=1, seeds=(42,))
    model.train()
    with pytest.raises(ValueError, match="eval"):
        compare_generations(model, training(), aperture=0.1, **args)
    assert model.training
    model.eval()
    with pytest.raises(ValueError, match="width"):
        compare_generations(model, training(width=3), aperture=0.1, **args)
    with pytest.raises(ValueError, match="omit aperture"):
        compare_generations(model, fit_conceptor(training(), 0.1), aperture=0.1, **args)
    model.cfg.device = "mps"
    with pytest.raises(ValueError, match="CPU and CUDA"):
        compare_generations(model, training(), aperture=0.1, **args)
    assert not model.calls


@pytest.mark.parametrize("shape,dtype", [((2, 4), torch.float32), ((1, 2, 3), torch.float32),
                                        ((1, 2, 4), torch.int64), ((1, 0, 4), torch.float32)])
def test_invalid_hook_inputs(shape, dtype):
    hook = make_matched_hooks(fit_conceptor(training(), 0.1))["complete"]
    with pytest.raises(ValueError, match="hook input"):
        hook(torch.ones(shape, dtype=dtype), None)


@pytest.mark.parametrize("field,value", [
    ("center", torch.zeros(3)), ("center", torch.full((4,), float("nan"))),
    ("basis", torch.zeros(4)), ("coefficients", torch.ones(4)),
    ("coefficients", -torch.ones(4)),
])
def test_invalid_factors(field, value):
    fitted = replace(fit_conceptor(training(), 0.1), **{field: value})
    with pytest.raises(ValueError, match="factor"):
        make_matched_hooks(fitted)


def test_nonfinite_states_and_cast_overflow_raise():
    hook = make_matched_hooks(fit_conceptor(training(), 0.1))["control"]
    with pytest.raises(ValueError, match="finite states"):
        hook(torch.full((1, 2, 4), float("nan")), None)
    with pytest.raises(ValueError, match="overflow"):
        hook(torch.full((1, 2, 4), 60000, dtype=torch.float16), None)


def test_generation_options_are_forwarded_without_extra_hook_arguments():
    model = FakeHookedModel()
    compare_generations(model, training(), ["prompt"], layer=1, seeds=(7,), aperture=0.1,
                        max_new_tokens=12, top_k=5, top_p=0.8, temperature=0.9,
                        stop_at_eos=False, prepend_bos=False)
    assert model.calls[0][1] == dict(max_new_tokens=12, do_sample=True, top_k=5,
                                   top_p=0.8, temperature=0.9, stop_at_eos=False,
                                   prepend_bos=False, return_type="str", verbose=False)
    assert model.calls[0][1] == model.calls[1][1]


@pytest.mark.parametrize("default", [True, False])
@pytest.mark.parametrize("override", [None, True, False])
def test_bos_uses_model_default_unless_explicitly_overridden(default, override):
    class TokenizingModel(FakeHookedModel):
        def generate(self, prompt, **options):
            actual = options.get("prepend_bos", default)
            assert actual is (default if override is None else override)
            if override is None:
                assert "prepend_bos" not in options
            return super().generate(prompt, **options)

    model = TokenizingModel()
    compare_generations(model, training(), ["prompt"], layer=1, seeds=(7,),
                        aperture=0.1, prepend_bos=override)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware not available")
@pytest.mark.parametrize("fail_on", [None, 2])
def test_cuda_rng_and_dtype_preserved(fail_on):
    model = FakeHookedModel("cuda:0")
    before = torch.cuda.get_rng_state_all()
    cpu = torch.get_rng_state().clone()
    model.fail_on = fail_on
    args = dict(layer=1, seeds=(42,), aperture=0.1)
    if fail_on:
        with pytest.raises(RuntimeError):
            compare_generations(model, training(), ["hello"], **args)
    else:
        compare_generations(model, training(), ["hello"], **args)
    for old, current in zip(before, torch.cuda.get_rng_state_all()):
        torch.testing.assert_close(old, current, rtol=0, atol=0)
    torch.testing.assert_close(cpu, torch.get_rng_state(), rtol=0, atol=0)
    h = torch.ones(2, 3, 4, device="cuda:0", dtype=torch.float16)
    hook = make_matched_hooks(fit_conceptor(training(), 0.1))["complete"]
    actual = hook(h, None)
    assert actual.device == h.device and actual.dtype == h.dtype and actual.shape == h.shape
