"""Synthetic characterization of broad-study extraction; no model or network is used.

The reference below preserves the pre-refactor extraction code. The fake model
executes real cache selections and hook callbacks over nonconstant tensors,
including a multi-position prefill, single-position decode events and early EOS.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import os
import random
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch.multiprocessing.reductions import StorageWeakRef

from experiments.broad_study import extraction
from experiments.broad_study.checkpointing import initialize_checkpoint_directory, sha256_file

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments" / "broad_study" / "run_study.py"

# Kept independently of the maintained implementation. These are the original
# sampling, formatting, collection and pooling operations, with only layout
# shortened. Compiling functions avoids executing the notebook-style runner.
REFERENCE_SOURCE = '''
def create_random_icl_string(pairs, num_examples):
    sampled = random.sample(pairs, min(num_examples, len(pairs)))
    prefix = ", ".join(f"{inp}:{out}" for inp, out in sampled[:-1])
    return f"{prefix}, {sampled[-1][0]}:"

def get_icl_prompts(pairs, n_prompts, n_examples):
    return [create_random_icl_string(pairs, n_examples) for _ in range(n_prompts)]

def extract_semantic_icl(func_name, func_data, layers):
    icl_prompts = get_icl_prompts(func_data["pairs"], NUM_ICL_PROMPTS, NUM_ICL_EXAMPLES)
    activations = {li: [] for li in layers}
    names = [f"blocks.{li}.hook_resid_pre" for li in layers]
    for prompt in tqdm(icl_prompts, desc=f"  {func_name}", leave=False):
        _, cache = model.run_with_cache(prompt, names_filter=lambda n: n in names)
        for li in layers:
            act = cache[f"blocks.{li}.hook_resid_pre"][:, -1, :].detach().cpu()
            activations[li].append(act.squeeze(0))
        del cache; torch.cuda.empty_cache()
    result = {li: torch.stack(activations[li]) for li in layers}
    for li in layers:
        s = result[li].shape
        assert s[0] == NUM_ICL_PROMPTS
        assert s[1] == HIDDEN_SIZE
    return result

def extract_seed_semantic(seed_idx, func_data):
    icl_prompts = get_icl_prompts(func_data["pairs"], NUM_ICL_PROMPTS, NUM_ICL_EXAMPLES)
    seed_acts = {li: [] for li in ANALYSIS_LAYERS}
    names = [f"blocks.{li}.hook_resid_pre" for li in ANALYSIS_LAYERS]
    for prompt in tqdm(icl_prompts, desc=f"    S{seed_idx+1} extract", leave=False):
        _, cache = model.run_with_cache(prompt, names_filter=lambda n: n in names)
        for li in ANALYSIS_LAYERS:
            act = cache[f"blocks.{li}.hook_resid_pre"][:, -1, :].detach().cpu()
            seed_acts[li].append(act.squeeze(0))
        del cache; torch.cuda.empty_cache()
    seed_acts = {li: torch.stack(seed_acts[li]) for li in ANALYSIS_LAYERS}
    return seed_acts

def extract_persona_activations(trait_name, trait_data, layers):
    extractions = {"pos": {li: [] for li in layers}, "neg": {li: [] for li in layers}}
    pairs_to_use = trait_data["instructions"][:N_INSTRUCTION_PAIRS]
    questions = trait_data["extract_questions"][:N_EXTRACTION_QUESTIONS]
    total_iters = len(pairs_to_use) * len(questions) * 2
    pbar = tqdm(total=total_iters, desc=f"  {trait_name}", leave=False)
    for pair_idx, pair in enumerate(pairs_to_use):
        for q in questions:
            for polarity in ["pos", "neg"]:
                messages = [
                    {"role": "system", "content": f"You are a {trait_name if polarity=='pos' else 'helpful'} assistant. {pair[polarity]}"},
                    {"role": "user", "content": q},
                ]
                prompt = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                captured = {li: [] for li in layers}
                def make_hook(li, storage=captured):
                    def hook_fn(h, hook):
                        storage[li].append(h[:, -1, :].detach().cpu().float())
                        return h
                    return hook_fn
                fwd_hooks = [(f"blocks.{li}.hook_resid_pre", make_hook(li)) for li in layers]
                tokens = model.to_tokens(prompt)
                with model.hooks(fwd_hooks=fwd_hooks):
                    model.generate(tokens, max_new_tokens=50, temperature=0.7,
                                   do_sample=True, verbose=False)
                for li in layers:
                    if captured[li]:
                        avg_act = torch.stack(captured[li]).mean(dim=0).squeeze(0)
                        extractions[polarity][li].append(avg_act)
                del captured; torch.cuda.empty_cache()
                pbar.update(1)
    pbar.close()
    result = {pol: {li: torch.stack(extractions[pol][li]) for li in layers} for pol in ["pos", "neg"]}
    expected_n = N_INSTRUCTION_PAIRS * N_EXTRACTION_QUESTIONS
    for pol in ["pos", "neg"]:
        for li in layers:
            s = result[pol][li].shape
            assert s[0] == expected_n
            assert s[1] == HIDDEN_SIZE
    return result
'''


class Progress:
    def __init__(self, iterable=None, **kwargs):
        self.iterable = iterable
        self.updates = 0
        self.closed = False

    def __iter__(self):
        return iter(self.iterable)

    def update(self, count):
        self.updates += count

    def close(self):
        self.closed = True


class FakeModel:
    """TransformerLens-like cache and hook execution on synthetic CPU tensors."""

    def __init__(self, dtype=torch.float64, fail_call=None):
        self.dtype = dtype
        self.fail_call = fail_call
        self.events = []
        self.active_hooks = []
        self.generation_index = 0
        self.cache_index = 0
        self.tokenizer = SimpleNamespace(apply_chat_template=self.apply_chat_template)

    def apply_chat_template(self, messages, **kwargs):
        self.events.append(("template", copy.deepcopy(messages), kwargs))
        return "|".join(message["content"] for message in messages) + "|assistant:"

    def to_tokens(self, prompt):
        self.events.append(("tokens", prompt))
        return prompt

    def state(self, index, layer, step, positions):
        # Different positions, dimensions, layers and steps detect averaging
        # the entire sequence, using the wrong hook or dropping the prefill.
        values = torch.arange(positions * 3, dtype=self.dtype).reshape(1, positions, 3)
        return (values + 100 * index + 10 * layer + step / 3).requires_grad_()

    def run_with_cache(self, prompt, names_filter):
        names = [f"blocks.{li}.hook_resid_pre" for li in range(4)]
        names += ["blocks.2.hook_resid_post", "hook_embed"]
        selected = [name for name in names if names_filter(name)]
        self.events.append(("cache", prompt, selected, torch.is_grad_enabled()))
        index = self.cache_index
        self.cache_index += 1
        random.random()
        torch.rand(1)
        cache = {
            name: self.state(index, int(name.split(".")[1]), 0, 4)
            for name in selected
        }
        return None, cache

    @contextmanager
    def hooks(self, fwd_hooks):
        previous = self.active_hooks
        self.active_hooks = list(fwd_hooks)
        self.events.append(("enter", [name for name, _ in fwd_hooks]))
        try:
            yield
        finally:
            self.active_hooks = previous
            self.events.append(("exit",))

    def generate(self, tokens, **kwargs):
        index = self.generation_index
        self.generation_index += 1
        self.events.append(("generate", tokens, kwargs, torch.is_grad_enabled()))
        # One event represents immediate EOS; longer cases have prefill and
        # decode forwards. A generation uses the caller's current RNG stream.
        count = (3, 1, 2, 4)[index % 4]
        for step in range(count):
            self.events.append(("rng", random.random(), float(torch.rand(1))))
            for name, hook in self.active_hooks:
                layer = int(name.split(".")[1])
                h = self.state(index, layer, step, 4 if step == 0 else 1)
                before = h.detach().clone()
                returned = hook(h, SimpleNamespace(name=name))
                self.events.append(("hook", name, step, returned is h))
                torch.testing.assert_close(h, before)
            if index == self.fail_call:
                raise RuntimeError("synthetic generation failure")
        return tokens


def namespace(model, **overrides):
    result = {
        "model": model, "torch": torch, "random": random, "tqdm": Progress,
        "extraction": extraction,
        "NUM_ICL_PROMPTS": 3, "NUM_ICL_EXAMPLES": 2, "HIDDEN_SIZE": 3,
        "N_INSTRUCTION_PAIRS": 2, "N_EXTRACTION_QUESTIONS": 2,
    }
    result.update(overrides)
    return result


def functions(model, *, reference=False, **overrides):
    bindings = namespace(model, **overrides)
    if reference:
        tree = ast.parse(REFERENCE_SOURCE)
    else:
        tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
        names = {"create_random_icl_string", "get_icl_prompts", "extract_semantic_icl",
                 "extract_persona_activations"}
        tree.body = [node for node in tree.body
                     if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(tree, "<extraction-characterization>", "exec"), bindings)
    return bindings


PAIRS = {"pairs": [["a", "A"], ["b", "B"], ["c", "C"]]}
TRAIT = {
    "instructions": [{"pos": "present one", "neg": "opposed one"},
                     {"pos": "present two", "neg": "opposed two"},
                     {"pos": "unused", "neg": "unused"}],
    "extract_questions": ["question one?", "question two?", "unused?"],
}


def run_seeded(function, *args):
    # Test isolation only: the implementation must neither reset nor add draws.
    random.seed(1729)
    torch.manual_seed(1729)
    with mock.patch.object(torch.cuda, "empty_cache") as clear:
        result = function(*args)
    return result, random.getstate(), torch.random.get_rng_state(), clear.call_count


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
@pytest.mark.parametrize("n_prompts", [1, 3])
def test_semantic_matches_reference_cache_rows_dtype_and_rng(dtype, n_prompts):
    original, current = FakeModel(dtype), FakeModel(dtype)
    expected = run_seeded(functions(original, reference=True, NUM_ICL_PROMPTS=n_prompts)[
        "extract_semantic_icl"], "synthetic", PAIRS, [2, 0])
    actual = run_seeded(functions(current, NUM_ICL_PROMPTS=n_prompts)[
        "extract_semantic_icl"], "synthetic", PAIRS, [2, 0])
    assert current.events == original.events
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])
    assert actual[3] == expected[3] == n_prompts
    assert list(actual[0]) == [2, 0]
    for layer, rows in actual[0].items():
        torch.testing.assert_close(rows, expected[0][layer], rtol=0, atol=0)
        torch.testing.assert_close(rows[0], current.state(0, layer, 0, 4)[0, -1, :])
        assert rows.shape == (n_prompts, 3)
        assert rows.dtype == dtype and rows.device.type == "cpu"
        assert not rows.requires_grad


def test_semantic_single_example_keeps_original_leading_comma():
    bindings = functions(FakeModel(), NUM_ICL_EXAMPLES=1)
    prompt = bindings["create_random_icl_string"]([["a", "A"]], 1)
    assert prompt == ", a:"


def seed_extraction_function(model, *, reference=False, **overrides):
    bindings = functions(model, reference=reference, ANALYSIS_LAYERS=[2, 0], **overrides)
    if reference:
        return bindings["extract_seed_semantic"]
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    seed_loop = next(node for node in ast.walk(tree)
                     if isinstance(node, ast.For) and isinstance(node.target, ast.Name)
                     and node.target.id == "seed_idx")

    def assigns(node, name):
        return isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        )

    start = next(index for index, node in enumerate(seed_loop.body) if assigns(node, "icl_prompts"))
    stop = next(index for index, node in enumerate(seed_loop.body)
                if assigns(node, "seed_conceptors"))
    function = ast.parse("def extracted(seed_idx, func_data):\n    pass\n").body[0]
    function.body = seed_loop.body[start:stop] + [ast.Return(
        value=ast.Name(id="seed_acts", ctx=ast.Load())
    )]
    fragment = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(fragment, "<seed-extraction-characterization>", "exec"), bindings)
    return bindings["extracted"]


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
@pytest.mark.parametrize("n_prompts", [1, 3])
def test_per_seed_semantic_loop_matches_reference_prompt_order_dtype_and_rng(dtype, n_prompts):
    original, current = FakeModel(dtype), FakeModel(dtype)
    expected = run_seeded(seed_extraction_function(
        original, reference=True, NUM_ICL_PROMPTS=n_prompts
    ), 1, PAIRS)
    actual = run_seeded(seed_extraction_function(current, NUM_ICL_PROMPTS=n_prompts), 1, PAIRS)
    assert current.events == original.events
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])
    assert actual[3] == expected[3] == n_prompts
    assert list(actual[0]) == [2, 0]
    for layer, rows in actual[0].items():
        torch.testing.assert_close(rows, expected[0][layer], rtol=0, atol=0)
        assert rows.shape == (n_prompts, 3) and rows.dtype == dtype


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_persona_matches_reference_event_order_pooling_and_rng(dtype):
    original, current = FakeModel(dtype), FakeModel(dtype)
    expected = run_seeded(functions(original, reference=True)[
        "extract_persona_activations"], "synthetic", TRAIT, [2, 0])
    actual = run_seeded(functions(current)[
        "extract_persona_activations"], "synthetic", TRAIT, [2, 0])
    assert current.events == original.events
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])
    assert actual[3] == expected[3] == 8
    for polarity in ("pos", "neg"):
        assert list(actual[0][polarity]) == [2, 0]
        for layer, rows in actual[0][polarity].items():
            torch.testing.assert_close(rows, expected[0][polarity][layer], rtol=0, atol=0)
            assert rows.shape == (4, 3)
            assert rows.dtype == torch.float32 and rows.device.type == "cpu"
            assert not rows.requires_grad
    # Independent numeric expectations: positive row zero averages 3 forwards;
    # negative row zero has only prefill (immediate EOS). Prefill has 4 positions.
    positive = torch.stack([current.state(0, 2, s, 4 if s == 0 else 1)[0, -1].float()
                            for s in range(3)]).mean(0)
    negative = current.state(1, 2, 0, 4)[0, -1].float()
    torch.testing.assert_close(actual[0]["pos"][2][0], positive, rtol=0, atol=0)
    torch.testing.assert_close(actual[0]["neg"][2][0], negative, rtol=0, atol=0)
    templates = [event[1] for event in current.events if event[0] == "template"]
    assert [m[1]["content"] for m in templates] == [
        "question one?", "question one?", "question two?", "question two?",
        "question one?", "question one?", "question two?", "question two?",
    ]
    assert templates[0][0]["content"] == "You are a synthetic assistant. present one"
    assert templates[1][0]["content"] == "You are a helpful assistant. opposed one"
    assert current.active_hooks == []


def test_persona_generation_failure_propagates_and_unwinds_hooks():
    logs = []
    for reference in (True, False):
        model = FakeModel(fail_call=1)
        with pytest.raises(RuntimeError, match="synthetic generation failure"):
            run_seeded(functions(model, reference=reference)[
                "extract_persona_activations"], "synthetic", TRAIT, [2, 0])
        assert model.active_hooks == []
        assert model.events[-1] == ("exit",)
        logs.append(model.events)
    assert logs[0] == logs[1]


def test_generic_prompt_helper_accepts_generator_keeps_one_row_and_caller_grad_mode():
    model = FakeModel(torch.float16)
    with torch.no_grad(), mock.patch.object(torch.cuda, "empty_cache") as clear:
        rows = extraction.last_token_resid_pre(model, (p for p in ["synthetic"]), [2, 0])
    assert list(rows) == [2, 0]
    assert len(model.events) == 1
    assert model.events[0] == (
        "cache", "synthetic", ["blocks.0.hook_resid_pre", "blocks.2.hook_resid_pre"], False
    )
    assert clear.call_count == 0
    for layer, values in rows.items():
        assert values.shape == (1, 3) and values.dtype == torch.float16
        torch.testing.assert_close(values[0], model.state(0, layer, 0, 4)[0, -1])


@pytest.mark.parametrize("retained", ["cache", "output"])
@pytest.mark.parametrize("grad_enabled", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_generic_helper_releases_sequence_storage_before_cleanup_and_next_forward(
    retained, grad_enabled, dtype,
):
    references = {"cache": [], "output": []}
    snapshots = []
    prompts = ["long", "short"]
    layers = [2, 0]

    def snapshot(stage):
        assert torch.is_grad_enabled() == grad_enabled
        snapshots.append((stage, {
            kind: [reference.expired() for reference in refs]
            for kind, refs in references.items()
        }))

    def states(prompt, layer):
        length = 64 if prompt == "long" else 1
        return torch.arange(length * 4, dtype=dtype).reshape(1, length, 4) + layer * 300

    class Model:
        def run_with_cache(self, prompt, names_filter):
            snapshot("forward")
            cache = {
                f"blocks.{layer}.hook_resid_pre": states(prompt, layer).requires_grad_()
                for layer in layers if names_filter(f"blocks.{layer}.hook_resid_pre")
            }
            output = torch.ones(1, 64, 8, dtype=dtype, requires_grad=True)
            # Observe the allocation, not just the original Tensor wrapper.
            references["cache"].extend(StorageWeakRef(t.untyped_storage()) for t in cache.values())
            references["output"].append(StorageWeakRef(output.untyped_storage()))
            return output, cache

    with torch.set_grad_enabled(grad_enabled):
        rows = extraction.last_token_resid_pre(
            Model(), iter(prompts), layers, clear_cache=lambda: snapshot("cleanup"),
        )
        assert torch.is_grad_enabled() == grad_enabled

    assert [stage for stage, _ in snapshots] == ["forward", "cleanup", "forward", "cleanup"]
    assert len(references["cache"]) == 4 and len(references["output"]) == 2
    for stage, released in snapshots:
        assert all(released[retained]), f"{retained} storage retained at {stage}: {released}"
    assert list(rows) == layers
    for layer, values in rows.items():
        expected = torch.stack([states(prompt, layer)[0, -1] for prompt in prompts])
        torch.testing.assert_close(values, expected, rtol=0, atol=0)
        assert values.shape == (2, 4) and values.dtype == dtype
        assert values.device.type == "cpu" and not values.requires_grad


@pytest.mark.parametrize("layers", [[], [-1], [True], [0, 0], [1.5]])
def test_generic_helper_rejects_invalid_layers_before_forward(layers):
    model = FakeModel()
    with pytest.raises(ValueError, match="layers"):
        extraction.last_token_resid_pre(model, ["synthetic"], layers)
    assert model.events == []


def test_generic_helper_requires_prompts_and_unbatched_cache():
    model = FakeModel()
    with pytest.raises(ValueError, match="at least one prompt"):
        extraction.last_token_resid_pre(model, [], [0])
    assert model.events == []

    class BatchedModel(FakeModel):
        def run_with_cache(self, prompt, names_filter):
            _, cache = super().run_with_cache(prompt, names_filter)
            return None, {name: tensor.repeat(2, 1, 1) for name, tensor in cache.items()}

    with pytest.raises(ValueError, match="single unbatched cache row"):
        extraction.last_token_resid_pre(BatchedModel(), ["synthetic"], [0])


def test_persona_single_row_and_failure_restore_existing_hook_context():
    model = FakeModel(torch.float16)
    rows = extraction.persona_generation_resid_pre(
        model, "synthetic", TRAIT, [0], n_instruction_pairs=1, n_extraction_questions=1
    )
    assert rows["pos"][0].shape == rows["neg"][0].shape == (1, 3)
    assert rows["pos"][0].dtype == rows["neg"][0].dtype == torch.float32
    existing = [("unrelated", object())]
    model = FakeModel(fail_call=0)
    model.active_hooks = existing
    with pytest.raises(RuntimeError, match="synthetic generation failure"):
        extraction.persona_generation_resid_pre(
            model, "synthetic", TRAIT, [0], n_instruction_pairs=1, n_extraction_questions=1
        )
    assert model.active_hooks is existing


def protocol_runner():
    path = ROOT / "experiments/protocol_swap/source/runner.py"
    spec = importlib.util.spec_from_file_location("protocol_extraction_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.torch = torch
    module.tqdm = Progress
    return module


@pytest.mark.parametrize("protocol", ["persona", "semantic_chat"])
@pytest.mark.parametrize("grad_enabled", [True, False])
def test_generation_extraction_does_not_retain_event_storage(protocol, grad_enabled):
    references = []
    snapshots = []

    class Model(FakeModel):
        def generate(self, tokens, **kwargs):
            assert torch.is_grad_enabled() == grad_enabled
            for step, length in enumerate([64, 1, 1]):
                for name, hook in self.active_hooks:
                    layer = int(name.split(".")[1])
                    state = torch.full((1, length, 3), float(layer + step), requires_grad=True)
                    references.append(StorageWeakRef(state.untyped_storage()))
                    assert hook(state, SimpleNamespace(name=name)) is state
                    del state
                    snapshots.append([ref.expired() for ref in references])
            return tokens

    model = Model()
    with torch.set_grad_enabled(grad_enabled):
        if protocol == "persona":
            result = extraction.persona_generation_resid_pre(
                model, "synthetic", TRAIT, [2, 0],
                n_instruction_pairs=1, n_extraction_questions=1,
            )
            groups = [result["pos"], result["neg"]]
        else:
            runner = protocol_runner()
            runner.NUM_ICL_PROMPTS = 2
            runner.create_random_icl_string = lambda pairs, count: "synthetic prompt"
            groups = [runner.extract_semantic_as_chat(model, "synthetic", PAIRS, [2, 0])]
        assert torch.is_grad_enabled() == grad_enabled

    assert snapshots and all(all(snapshot) for snapshot in snapshots)
    for group in groups:
        assert list(group) == [2, 0]
        for layer, rows in group.items():
            torch.testing.assert_close(rows, torch.full_like(rows, float(layer + 1)), rtol=0, atol=0)
            assert rows.dtype == torch.float32 and not rows.requires_grad


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_protocol_prompt_final_releases_output_and_cache_before_cleanup(dtype):
    runner = protocol_runner()
    runner.N_INSTRUCTION_PAIRS = 1
    runner.N_EXTRACTION_QUESTIONS = 2
    references = []
    snapshots = []

    class Model(FakeModel):
        def run_with_cache(self, prompt, names_filter):
            snapshots.append([ref.expired() for ref in references])
            cache = {
                f"blocks.{layer}.hook_resid_pre":
                    torch.full((1, 64, 3), float(layer), dtype=dtype, requires_grad=True)
                for layer in [2, 0] if names_filter(f"blocks.{layer}.hook_resid_pre")
            }
            output = torch.ones(1, 64, 8, requires_grad=True)
            references.extend(StorageWeakRef(t.untyped_storage()) for t in [output, *cache.values()])
            return output, cache

    with mock.patch.object(torch.cuda, "empty_cache", side_effect=lambda: snapshots.append(
        [ref.expired() for ref in references]
    )):
        result = runner.extract_persona_as_prompt_final(Model(), "synthetic", TRAIT, [2, 0])
    assert len(snapshots) == 4 and all(all(snapshot) for snapshot in snapshots)
    assert list(result) == [2, 0]
    for layer, rows in result.items():
        assert rows.shape == (2, 3) and rows.dtype == torch.float32
        torch.testing.assert_close(rows, torch.full_like(rows, float(layer)), rtol=0, atol=0)


def test_runner_import_follows_preflight_and_checkpoint_identity_binds_helper(tmp_path):
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    import_line = next(node.lineno for node in tree.body if isinstance(node, ast.ImportFrom)
                       and any(alias.name == "extraction" for alias in node.names))
    for function_name in ("require_expected_pairs", "require_experimental_materials"):
        line = next(node.lineno for node in ast.walk(tree)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == function_name)
        assert line < import_line
    specification = next(
        node.value for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "RUN_SPECIFICATION"
                for target in node.targets)
    )
    hash_expression = next(value for key, value in zip(specification.keys, specification.values)
                           if isinstance(key, ast.Constant) and key.value == "extraction_sha256")
    helper_hash = eval(compile(ast.Expression(hash_expression), "<helper-hash>", "eval"), {
        "sha256_file": sha256_file, "SCRIPT_DIR": RUNNER.parent,
    })
    assert helper_hash == hashlib.sha256(Path(extraction.__file__).read_bytes()).hexdigest()
    public_expression = next(value for key, value in zip(specification.keys, specification.values)
                             if isinstance(key, ast.Constant)
                             and key.value == "reusable_extraction_sha256")
    public_hash = eval(compile(ast.Expression(public_expression), "<public-hash>", "eval"), {
        "sha256_file": sha256_file, "REPO_ROOT": ROOT,
    })
    assert public_hash == hashlib.sha256((ROOT / "extract_activations.py").read_bytes()).hexdigest()
    bound = {"extraction_sha256": helper_hash, "reusable_extraction_sha256": public_hash}
    initialize_checkpoint_directory(tmp_path, bound, ("extractions.pt",))
    changed = {**bound, "reusable_extraction_sha256": "0" * 64}
    with pytest.raises(RuntimeError, match="Checkpoint manifest mismatch"):
        initialize_checkpoint_directory(tmp_path, changed, ("extractions.pt",))


EXAMPLE_PATH = ROOT / "examples" / "extract_activations.py"


def load_example():
    spec = importlib.util.spec_from_file_location("_test_activation_example", EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_example_without_model_stack(tmp_path, *arguments, import_only=False):
    bootstrap = (
        "import runpy, sys\n"
        "for name in ('torch', 'numpy', 'transformer_lens', 'transformers', 'huggingface_hub'):\n"
        "    sys.modules[name] = None\n"
        "sys.argv = sys.argv[1:]\n"
        f"runpy.run_path(sys.argv[0], run_name={'<import-only>' if import_only else '__main__'!r})\n"
    )
    return subprocess.run(
        [sys.executable, "-B", "-c", bootstrap, str(EXAMPLE_PATH), *arguments],
        cwd=tmp_path, capture_output=True, text=True, check=False, timeout=30,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
    )


def test_example_help_and_import_need_no_heavy_dependencies_or_output_files(tmp_path):
    imported = run_example_without_model_stack(tmp_path, import_only=True)
    assert imported.returncode == 0, imported.stderr
    assert imported.stdout == imported.stderr == ""
    help_result = run_example_without_model_stack(tmp_path, "--help")
    assert help_result.returncode == 0, help_result.stderr
    for flag in ("--model", "--layer", "--revision", "--device"):
        assert flag in help_result.stdout
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("arguments", [
    ("--model", "synthetic", "--layer", "-1"),
    ("--model", "synthetic", "--layer", "0", "--layer", "0"),
    ("--model", "synthetic", "--revision", "main"),
])
def test_example_invalid_flags_fail_before_model_loading(tmp_path, arguments):
    result = run_example_without_model_stack(tmp_path, *arguments)
    assert result.returncode == 2
    assert "ModuleNotFoundError" not in result.stderr
    assert not list(tmp_path.iterdir())


def test_direct_extraction_script_gives_module_command_before_loading(tmp_path):
    result = run_example_without_model_stack(tmp_path, "--model", "gpt2")
    assert result.returncode == 2
    assert "python -m examples.extract_activations" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr
    assert not list(tmp_path.iterdir())


def test_example_callable_uses_loaded_model_and_cli_prints_real_synthetic_norms(monkeypatch, capsys):
    example = load_example()
    loaded = FakeModel(torch.float64)
    values = example.extract_from_model(loaded, ["inline synthetic input"], [2, 0])
    assert values[2].shape == (1, 3)
    assert loaded.events[0][0:2] == ("cache", "inline synthetic input")

    class OwnModel(FakeModel):
        cfg = SimpleNamespace(n_layers=4)

        def run_with_cache(self, *args, **kwargs):
            assert torch.is_inference_mode_enabled()
            return super().run_with_cache(*args, **kwargs)

        def eval(self):
            self.events.append(("eval",))
            return self

    own = OwnModel()
    loads = []

    def load(model_name, **kwargs):
        loads.append((model_name, kwargs))
        return own

    monkeypatch.setitem(sys.modules, "transformer_lens", SimpleNamespace(
        HookedTransformer=SimpleNamespace(from_pretrained=load)
    ))
    assert example.main([
        "--model", "synthetic", "--layer", "2", "--layer", "0",
        "--revision", "A" * 40,
    ]) == 0
    assert loads == [("synthetic", {"device": "cpu", "revision": "a" * 40})]
    assert own.events[0] == ("eval",)
    assert [event[1] for event in own.events if event[0] == "cache"] == list(
        example.SYNTHETIC_PROMPTS
    )
    output = capsys.readouterr().out
    assert "Prompt-final activations for three synthetic example prompts." in output
    for layer in (2, 0):
        assert f"Layer {layer}: shape=(3, 3), dtype=torch.float64, device=cpu" in output
        norm = own.state(0, layer, 0, 4)[0, -1].float().norm().item()
        assert f"Prompt 1: norm={norm:.6f}" in output
