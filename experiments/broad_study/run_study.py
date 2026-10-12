#!/usr/bin/env python3
"""Run the broad semantic-function and persona steering study.

This script reconstructs the shared execution path from the Qwen2.5-1.5B,
Qwen2.5-7B, and Llama-3.1-8B notebooks. It uses public task inputs and explicit
run settings; see README.md for differences from the original experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as package_metadata
import json
import os
import re
import tempfile
from pathlib import Path

from experiments.broad_study.checkpointing import (
    atomic_write_json,
    initialize_checkpoint_directory,
    require_checkpoint_identity,
    sha256_file,
)
from experiments.broad_study.evaluation import (
    InvalidJudgeScore,
    checkpointed_persona_traits,
    decode_persona_checkpoint_responses,
    parse_judge_score,
    persona_response_universe,
    prompt_block_reason,
    reject_duplicate_json_keys,
    require_complete_evaluation,
    require_complete_persona_corpus,
    require_experimental_materials,
    require_function_vector_document,
    score_persona_responses,
    study_judgments_are_complete,
)
from experiments.broad_study.fetch_function_vectors import require_expected_pairs


SUPPORTED_MODELS = (
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", choices=SUPPORTED_MODELS, required=True)
parser.add_argument(
    "--output-dir",
    type=Path,
    help=(
        "Run output directory (default: runs/<model>-<revision>-<dtype>-<mode> "
        "beside this script)."
    ),
)
parser.add_argument("--device", default="cuda")
parser.add_argument(
    "--model-revision",
    required=True,
    help="Full immutable 40-character Hugging Face model commit SHA.",
)
parser.add_argument(
    "--dtype", choices=("float16", "bfloat16", "float32"), default="float16"
)
parser.add_argument("--fast", action="store_true", help="Use reduced sample sizes.")
args = parser.parse_args()
if re.fullmatch(r"[0-9a-fA-F]{40}", args.model_revision) is None:
    parser.error("--model-revision must be a full 40-character Hugging Face commit SHA")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

# Local inputs are validated before any output is created or the model stack is
# imported.
# 6 SEMANTIC FUNCTIONS (all from jorispos/ConceptorSteering/data/functionvectors/)
# Matches the 6 tasks in run_gptj_conceptors.sh (Postmus & Abreu, 2410.16314v4 §4.1)
function_path = SCRIPT_DIR / "function_vectors.json"
materials_path = REPO_ROOT / "experimental_materials.json"
if not function_path.is_file():
    parser.error(f"{function_path} is absent. Run fetch_function_vectors.py first.")
try:
    function_bytes = function_path.read_bytes()
    SEMANTIC_FUNCTIONS = require_function_vector_document(json.loads(function_bytes))
    require_expected_pairs(SEMANTIC_FUNCTIONS)
except (OSError, ValueError) as error:
    parser.error(f"{function_path}: {error}")
try:
    materials_bytes = materials_path.read_bytes()
    material_document = json.loads(materials_bytes)
    require_experimental_materials(material_document)
except (OSError, ValueError) as error:
    parser.error(f"{materials_path}: {error}")
PERSONA_TRAITS = material_document["traits"]
COHERENCE_PROMPT = material_document["coherence_prompt"]

MODEL = args.model
MODEL_SHORT = MODEL.split("/")[-1]
FAST_MODE = args.fast
RUN_SLUG = (
    f"{MODEL_SHORT}-{args.model_revision[:12].lower()}-{args.dtype}-"
    f"{'fast' if FAST_MODE else 'full'}"
)
OUTPUT_DIR = (args.output_dir or SCRIPT_DIR / "runs" / RUN_SLUG).resolve()
if not os.environ.get("GEMINI_API_KEY") and (OUTPUT_DIR / "checkpoints" / "persona_judgments.json").exists():
    parser.error(
        "This output directory contains persona_judgments.json. "
        "Set GEMINI_API_KEY to resume with bound judgment recovery, or use a new "
        "--output-dir for a keyless run. Refusing to replace existing score exports "
        "with unjudged responses."
    )

import torch, gc, json, random, time, sys, re
from experiments.broad_study import extraction
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm.auto import tqdm

from google import genai
from google.genai import types
import backoff
from google.api_core import exceptions as google_exceptions

# Reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

print(f"numpy={np.__version__}, scipy={scipy.__version__}, torch={torch.__version__}")

MODEL_DTYPE = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}[args.dtype]

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Keep all generated artifacts inside the requested run directory.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(OUTPUT_DIR)
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output dir: {OUTPUT_DIR}")
print(f"Checkpoints: {CHECKPOINT_DIR}")

# Conceptor parameters used in the executed notebooks.
APERTURE = 0.05
BETA_CONCEPTOR = 2.0
BETA_ADDITIVE = 2.3
BETAS_CONCEPTOR = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
APERTURES_SEMANTIC = [0.001, 0.0125, 0.05, 0.1]

if FAST_MODE:
    NUM_ICL_PROMPTS = 50
    N_TEST_PROMPTS = 20
else:
    NUM_ICL_PROMPTS = 100
    N_TEST_PROMPTS = 50

APERTURES_SWEEP = [0.001, 0.005, 0.0125, 0.025, 0.05, 0.1, 0.5, 1.0, 5.0]
NUM_ICL_EXAMPLES = 10
N_INSTRUCTION_PAIRS = 5
N_EXTRACTION_QUESTIONS = 20
N_SAMPLES = 40
N_SEEDS = 3
MAX_NEW_TOKENS = 200

mode_str = " FAST" if FAST_MODE else " FULL"
print(f"Config: {MODEL_SHORT} ({mode_str})")
print(f"  Seed: {SEED}")
print(f"  ICL: {NUM_ICL_PROMPTS} prompts x {NUM_ICL_EXAMPLES} examples")
print(f"  Grid: {len(BETAS_CONCEPTOR)} betas x {len(APERTURES_SEMANTIC)} apertures x layers")
print(f"  Test: {N_TEST_PROMPTS} prompts per function")
print(f"  Persona: {N_INSTRUCTION_PAIRS} x {N_EXTRACTION_QUESTIONS} = "
      f"{N_INSTRUCTION_PAIRS * N_EXTRACTION_QUESTIONS}/polarity")

from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained(
    MODEL, revision=args.model_revision, device=args.device, fold_ln=False,
    center_writing_weights=False, center_unembed=False, dtype=MODEL_DTYPE,
)
model.eval()

NUM_LAYERS = model.cfg.n_layers
HIDDEN_SIZE = model.cfg.d_model
THIRD_LAYER = NUM_LAYERS // 3
# Chen et al. (2507.21509v3) Appendix B.4: best layers PER TRAIT (1-indexed in paper).
# Other models use the midpoint; spectral peaks are added as separate candidates later.
# Paper values (1-indexed → 0-indexed): Llama-3.1-8B L16→L15, Qwen2.5-7B L20→L19/L16→L15
CHEN_BEST_LAYERS = {
    "meta-llama/Llama-3.1-8B-Instruct": {"default": 15},
    "Qwen/Qwen2.5-7B-Instruct":         {"default": 19, "hallucination": 15},
}

def get_chen_layer(model_name, trait_name=None):
    """Get per-trait Chen layer. Falls back to ~50% depth for unknown models."""
    if model_name in CHEN_BEST_LAYERS:
        layers = CHEN_BEST_LAYERS[model_name]
        if trait_name and trait_name in layers:
            return layers[trait_name]
        return layers["default"]
    return NUM_LAYERS // 2

CHEN_LAYER = get_chen_layer(MODEL)
_chen_source = "paper B.4" if MODEL in CHEN_BEST_LAYERS else "auto (~50% depth)"

print(f"{type(model).__name__}: layers={NUM_LAYERS}, hidden={HIDDEN_SIZE}")
print(f"  Postmus conceptor layer (1/3 depth): L{THIRD_LAYER}")
print(f"  Chen additive layer ({_chen_source}): L{CHEN_LAYER} ({CHEN_LAYER/NUM_LAYERS:.0%} depth)")
ANALYSIS_LAYERS = sorted(set(list(range(0, NUM_LAYERS, 2)) + [THIRD_LAYER, CHEN_LAYER]))
print(f"  Analysis layers ({len(ANALYSIS_LAYERS)}): {ANALYSIS_LAYERS}")

assert THIRD_LAYER in ANALYSIS_LAYERS, f"Error: L{THIRD_LAYER} missing from ANALYSIS_LAYERS!"
assert CHEN_LAYER in ANALYSIS_LAYERS, f"Error: L{CHEN_LAYER} missing from ANALYSIS_LAYERS!"
print(f"  VERIFIED: L{THIRD_LAYER} and L{CHEN_LAYER} in ANALYSIS_LAYERS")


RUN_SPECIFICATION = {
    "schema_version": 2,
    "model": MODEL,
    "model_revision": args.model_revision.lower(),
    "dtype": args.dtype,
    "device": args.device,
    "fast_mode": FAST_MODE,
    "seed": SEED,
    "num_layers": NUM_LAYERS,
    "hidden_size": HIDDEN_SIZE,
    "analysis_layers": ANALYSIS_LAYERS,
    "num_icl_prompts": NUM_ICL_PROMPTS,
    "num_test_prompts": N_TEST_PROMPTS,
    "num_icl_examples": NUM_ICL_EXAMPLES,
    "num_instruction_pairs": N_INSTRUCTION_PAIRS,
    "num_extraction_questions": N_EXTRACTION_QUESTIONS,
    "num_samples": N_SAMPLES,
    "num_seeds": N_SEEDS,
    "max_new_tokens": MAX_NEW_TOKENS,
    "aperture": APERTURE,
    "beta_conceptor": BETA_CONCEPTOR,
    "beta_additive": BETA_ADDITIVE,
    "conceptor_beta_grid": BETAS_CONCEPTOR,
    "semantic_aperture_grid": APERTURES_SEMANTIC,
    "runtime": {
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "transformer_lens": package_metadata.version("transformer-lens"),
        "transformers": package_metadata.version("transformers"),
    },
    "runner_sha256": sha256_file(SCRIPT_DIR / "run_study.py"),
    "checkpointing_sha256": sha256_file(SCRIPT_DIR / "checkpointing.py"),
    "evaluation_sha256": sha256_file(SCRIPT_DIR / "evaluation.py"),
    "extraction_sha256": sha256_file(SCRIPT_DIR / "extraction.py"),
    "reusable_extraction_sha256": sha256_file(REPO_ROOT / "extract_activations.py"),
    "function_vectors_sha256": hashlib.sha256(function_bytes).hexdigest(),
    "experimental_materials_sha256": hashlib.sha256(materials_bytes).hexdigest(),
}
RUN_SPECIFICATION_SHA256 = initialize_checkpoint_directory(
    CHECKPOINT_DIR,
    RUN_SPECIFICATION,
    ("extractions.pt", "semantic_results.json", "persona_responses.json", "persona_judgments.json"),
)
print(f"Run specification: {RUN_SPECIFICATION_SHA256}")


def atomic_torch_save(document, path: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(document, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def create_random_icl_string(pairs, num_examples):
    """Create the comma-separated in-context relation prompt used in the broad study."""
    sampled = random.sample(pairs, min(num_examples, len(pairs)))
    prefix = ", ".join(f"{inp}:{out}" for inp, out in sampled[:-1])
    return f"{prefix}, {sampled[-1][0]}:"


def get_icl_prompts(pairs, n_prompts, n_examples):
    return [create_random_icl_string(pairs, n_examples) for _ in range(n_prompts)]


def get_test_prompts(pairs, n_test):
    sampled = random.sample(pairs, min(n_test, len(pairs)))
    return [(f"{inp}:", out) for inp, out in sampled]


print(f"Semantic functions: {list(SEMANTIC_FUNCTIONS.keys())}")
print(f"Persona traits: {list(PERSONA_TRAITS.keys())}")
print("VERIFIED: 6 semantic functions and 7 persona targets loaded")

class ConceptorOps:
    """Conceptor computation and spectral analysis utilities."""

    @staticmethod
    def compute(activations, aperture=0.1, use_centering=True):
        """Compute conceptor from centered covariance matrix.
        C = S(S + a^{-2}I)^{-1} where S is the centered covariance.
        Used for spectral analysis and mean-centered steering."""
        activations = activations.float()
        n, d = activations.shape
        mean = activations.mean(dim=0)
        centered = (activations - mean.unsqueeze(0)) if use_centering else activations
        cov = (centered.T @ centered) / n
        I = torch.eye(d, device=activations.device, dtype=activations.dtype)
        reg = (aperture ** -2) * I
        conceptor = cov @ torch.linalg.inv(cov + reg)
        eigenvalues = torch.linalg.eigvalsh(conceptor)
        return conceptor, eigenvalues.sort(descending=True).values, mean

    @staticmethod
    def compute_postmus(activations, aperture=0.1):
        """Compute conceptor from uncentered correlation matrix via SVD.
        R = X^T X / N (correlation, not covariance  --  preserves mean direction).
        C = U diag(S / (S + alpha^{-2})) U^T
        Matches Postmus & Abreu (2024) conceptors.py implementation."""
        activations = activations.float()
        n, d = activations.shape
        R = (activations.T @ activations) / n
        U, S, _ = torch.linalg.svd(R)
        alpha_inv_sq = aperture ** (-2)
        S_scaled = S / (S + alpha_inv_sq)
        C = U @ torch.diag(S_scaled) @ U.T
        eigenvalues = S_scaled.sort(descending=True).values
        return C, eigenvalues

    @staticmethod
    def spectral_metrics(eigenvalues, C):
        """Compute CCR and participation-ratio effective rank."""
        n_eff = eigenvalues.sum().item() ** 2 / (eigenvalues ** 2).sum().item() if eigenvalues.sum() > 0 else 0
        ccr = eigenvalues.sum().item() / len(eigenvalues) if len(eigenvalues) > 0 else 0
        top10_energy = eigenvalues[:10].sum().item()
        return {"effective_rank": n_eff, "ccr": ccr, "top10_energy": top10_energy}


class PostmusConceptorHook:
    """Conceptor soft-filter hook for activation steering.

    Modes:
        enhance: h' = m + beta C (h - m)
        suppress: h' = m + beta (I-C) (h - m)

    Uses m=0 when mean=None. When last_token_only=True, transforms only
    the final sequence position, matching Postmus & Abreu's steering.py.
    """

    def __init__(self, C, mean=None, beta=2.0, mode="enhance", last_token_only=True):
        self.C = C.float()
        self.mean = mean.float() if mean is not None else None
        self.beta = beta
        self.last_token_only = last_token_only
        if mode == "suppress":
            I = torch.eye(C.shape[0], device=C.device, dtype=C.dtype)
            self.C = I - self.C

    def __call__(self, h, hook):
        h_out = h.clone()
        h_f = h.float()
        mat = self.C.to(h_f.device)

        if self.last_token_only:
            last = h_f.shape[1] - 1
            for i in range(h_f.shape[0]):
                x = h_f[i, last, :]
                if self.mean is not None:
                    m = self.mean.to(h_f.device)
                    h_out[i, last, :] = (self.beta * mat @ (x - m) + m).to(h.dtype)
                else:
                    h_out[i, last, :] = (self.beta * mat @ x).to(h.dtype)
        else:
            if self.mean is not None:
                m = self.mean.to(h_f.device)
                centered = h_f - m
                projected = self.beta * torch.einsum('ij,bsj->bsi', mat, centered) + m
            else:
                projected = self.beta * torch.einsum('ij,bsj->bsi', mat, h_f)
            h_out = projected.to(h.dtype)

        return h_out


class AdditiveHook:
    """Additive steering hook. Adds/subtracts a direction vector.

    When last_token_only=True, adds only at the final sequence position
    (Postmus semantic). When False, adds at all positions (Chen persona)."""

    def __init__(self, vec, coef=2.3, mode="add", last_token_only=False):
        self.vec = vec.float()
        self.sign = -1.0 if mode == "subtract" else 1.0
        self.coef = coef
        self.last_token_only = last_token_only

    def __call__(self, h, hook):
        h_out = h.clone()
        delta = (self.sign * self.coef * self.vec.to(h.device)).to(h.dtype)
        if self.last_token_only:
            h_out[:, -1, :] += delta
        else:
            h_out += delta
        return h_out


class SteerableModel:
    """Wrapper for TransformerLens model with hook management."""

    def __init__(self, model):
        self.model = model
        self.hooks = []

    def add_hook(self, layer_idx, hook_fn):
        name = f"blocks.{layer_idx}.hook_resid_pre"
        self.hooks.append((name, hook_fn))

    def clear_hooks(self):
        self.model.reset_hooks()
        self.hooks = []

    def generate(self, prompt, max_new_tokens=200, temperature=0.7, do_sample=True, top_p=0.95):
        tokens = self.model.to_tokens(prompt)
        with self.model.hooks(fwd_hooks=self.hooks):
            out = self.model.generate(
                tokens, max_new_tokens=max_new_tokens, temperature=temperature,
                do_sample=do_sample, top_p=top_p, verbose=False, return_type="tokens",
            )
        # TransformerLens returns the input tokens followed by the completion.
        return self.model.to_string(out[0, tokens.shape[1]:]).strip()

    def top_1_next_token(self, prompts):
        """Get top-1 next token prediction for each prompt (Postmus evaluation)."""
        tokens = self.model.to_tokens(prompts, padding_side='left')
        with self.model.hooks(fwd_hooks=self.hooks):
            logits = self.model(tokens)
        next_logits = logits[:, -1, :]
        top_ids = torch.argmax(next_logits, dim=-1)
        return [self.model.tokenizer.decode([tid.item()]).strip() for tid in top_ids]

steerable = SteerableModel(model)
print("Core classes initialized")


def extract_semantic_icl(func_name, func_data, layers):
    """Postmus-faithful: extract last-token activations from ICL prompts."""
    icl_prompts = get_icl_prompts(func_data["pairs"], NUM_ICL_PROMPTS, NUM_ICL_EXAMPLES)
    result = extraction.last_token_resid_pre(
        model, tqdm(icl_prompts, desc=f"  {func_name}", leave=False), layers,
        clear_cache=torch.cuda.empty_cache,
    )
    for li in layers:
        s = result[li].shape
        assert s[0] == NUM_ICL_PROMPTS, f"Error: {func_name} L{li} got {s[0]} samples, expected {NUM_ICL_PROMPTS}"
        assert s[1] == HIDDEN_SIZE, f"Error: {func_name} L{li} hidden={s[1]}, expected {HIDDEN_SIZE}"
    return result


def extract_persona_activations(trait_name, trait_data, layers):
    pairs_to_use = trait_data["instructions"][:N_INSTRUCTION_PAIRS]  # ALL 5 pairs
    questions = trait_data["extract_questions"][:N_EXTRACTION_QUESTIONS]

    total_iters = len(pairs_to_use) * len(questions) * 2  # 2 polarities
    pbar = tqdm(total=total_iters, desc=f"  {trait_name}", leave=False)
    result = extraction.persona_generation_resid_pre(
        model, trait_name, trait_data, layers,
        n_instruction_pairs=N_INSTRUCTION_PAIRS,
        n_extraction_questions=N_EXTRACTION_QUESTIONS,
        progress=pbar, clear_cache=torch.cuda.empty_cache,
    )
    expected_n = N_INSTRUCTION_PAIRS * N_EXTRACTION_QUESTIONS
    for pol in ["pos", "neg"]:
        for li in layers:
            s = result[pol][li].shape
            assert s[0] == expected_n, f"Error: {trait_name} {pol} L{li} got {s[0]} samples, expected {expected_n}"
            assert s[1] == HIDDEN_SIZE, f"Error: {trait_name} {pol} L{li} hidden={s[1]}, expected {HIDDEN_SIZE}"
    return result

print("Extraction functions ready")
print(f"  Semantic: {NUM_ICL_PROMPTS} samples per function")
print(f"  Persona:  {N_INSTRUCTION_PAIRS} x {N_EXTRACTION_QUESTIONS} = {N_INSTRUCTION_PAIRS*N_EXTRACTION_QUESTIONS} per polarity")


# --- CHECKPOINT: Extraction ---
extraction_ckpt = CHECKPOINT_DIR / "extractions.pt"
if extraction_ckpt.exists():
    print(f"LOADING extraction checkpoint from {extraction_ckpt}")
    _ckpt = torch.load(extraction_ckpt, map_location="cpu", weights_only=True)
    require_checkpoint_identity(
        _ckpt, RUN_SPECIFICATION_SHA256, extraction_ckpt
    )
    semantic_extractions = _ckpt["semantic"]
    persona_extractions = _ckpt["persona"]
    del _ckpt
    print(f"  Loaded {len(semantic_extractions)} semantic, {len(persona_extractions)} persona")
else:
    _t0 = time.time()
    print("=" * 60)
    print(f"SEMANTIC EXTRACTION (Postmus ICL format)  --  {len(ANALYSIS_LAYERS)} layers")
    print("=" * 60)
    semantic_extractions = {}
    for fn_name, fn_data in tqdm(SEMANTIC_FUNCTIONS.items(), desc="Semantic", leave=False):
        semantic_extractions[fn_name] = extract_semantic_icl(fn_name, fn_data, ANALYSIS_LAYERS)
        n = semantic_extractions[fn_name][ANALYSIS_LAYERS[0]].shape[0]
        print(f"  {fn_name}: shape=[{n}, {HIDDEN_SIZE}] at {len(ANALYSIS_LAYERS)} layers")

    _t1 = time.time()
    print(f"  Semantic extraction: {(_t1-_t0)/60:.1f} min")
    print(f"\n{'=' * 60}")
    print(f"PERSONA EXTRACTION (Chen chat template, {N_INSTRUCTION_PAIRS} pairs x {N_EXTRACTION_QUESTIONS} questions)")
    print("=" * 60)
    persona_extractions = {}
    for trait_name, trait_data in tqdm(PERSONA_TRAITS.items(), desc="Persona", leave=False):
        persona_extractions[trait_name] = extract_persona_activations(trait_name, trait_data, ANALYSIS_LAYERS)
        n_pos = persona_extractions[trait_name]["pos"][ANALYSIS_LAYERS[0]].shape[0]
        print(f"  {trait_name}: shape=[{n_pos}, {HIDDEN_SIZE}] pos+neg at {len(ANALYSIS_LAYERS)} layers")

    _t2 = time.time()
    print(f"  Persona extraction: {(_t2-_t1)/60:.1f} min")
    print(f"  Total extraction: {(_t2-_t0)/60:.1f} min")
    gc.collect(); torch.cuda.empty_cache()

    atomic_torch_save(
        {
            "run_specification_sha256": RUN_SPECIFICATION_SHA256,
            "semantic": semantic_extractions,
            "persona": persona_extractions,
        },
        extraction_ckpt,
    )
    print(f"\n  CHECKPOINT SAVED: {extraction_ckpt}")

sem_n =semantic_extractions[list(SEMANTIC_FUNCTIONS.keys())[0]][ANALYSIS_LAYERS[0]].shape[0]
per_n = persona_extractions[list(PERSONA_TRAITS.keys())[0]]["pos"][ANALYSIS_LAYERS[0]].shape[0]
print("\n--- EXTRACTION VERIFICATION ---")
print(f"  Semantic N={sem_n}, Persona N={per_n}")
print("  VERIFIED: Extractions complete (conceptor R = X^T X / N normalizes by sample count)")

for name in SEMANTIC_FUNCTIONS:
    assert THIRD_LAYER in semantic_extractions[name], f"Error: {name} missing L{THIRD_LAYER}!"
    assert CHEN_LAYER in semantic_extractions[name], f"Error: {name} missing L{CHEN_LAYER}!"
for name in PERSONA_TRAITS:
    assert THIRD_LAYER in persona_extractions[name]["pos"], f"Error: {name} pos missing L{THIRD_LAYER}!"
    assert CHEN_LAYER in persona_extractions[name]["pos"], f"Error: {name} pos missing L{CHEN_LAYER}!"
print(f"  VERIFIED: L{THIRD_LAYER} and L{CHEN_LAYER} present in ALL extractions")


# Verify all extractions succeeded without NaN or empty tensors.

print("=" * 60)
print("EXTRACTION VERIFICATION")
print("=" * 60)

total_checks = 0
for name in SEMANTIC_FUNCTIONS:
    for li in ANALYSIS_LAYERS:
        acts = semantic_extractions[name][li]
        assert acts.shape[0] > 0, f"Error: Empty extraction for {name} L{li}"
        assert not torch.isnan(acts).any(), f"Error: NaN in {name} L{li}"
        total_checks += 1
    print(f"  Semantic '{name}': {len(ANALYSIS_LAYERS)} layers, shape {acts.shape} -- OK")

for name in PERSONA_TRAITS:
    for key in ["pos", "neg"]:
        for li in ANALYSIS_LAYERS:
            acts = persona_extractions[name][key][li]
            assert acts.shape[0] > 0, f"Error: Empty extraction for {name}/{key} L{li}"
            assert not torch.isnan(acts).any(), f"Error: NaN in {name}/{key} L{li}"
            total_checks += 1
    print(f"  Persona  '{name}': {len(ANALYSIS_LAYERS)} layers x 2 (pos/neg) -- OK")

print(f"\nAll {total_checks} extraction tensors verified: no NaN, no empty tensors.")
print(f"Semantic functions: {len(SEMANTIC_FUNCTIONS)}, Persona traits: {len(PERSONA_TRAITS)}")


# Compare raw CCR with CCR after normalizing each activation to unit length.
# The latter controls observation norms; both retain directional variation.

spectral_data = {}
norm_spectral_data = {}  # L2-normalized
peak_info = {}
norm_peak_info = {}      # peaks from normalized analysis

for name in list(SEMANTIC_FUNCTIONS.keys()) + list(PERSONA_TRAITS.keys()):
    spectral_data[name] = {}
    norm_spectral_data[name] = {}
    is_persona = name in PERSONA_TRAITS

    for li in ANALYSIS_LAYERS:
        if is_persona:
            acts = persona_extractions[name]["pos"][li]  # Conceptor on pos activations
        else:
            acts = semantic_extractions[name][li]

        # Unnormalized spectral measurements
        C, eigs, mean = ConceptorOps.compute(acts, APERTURE, use_centering=True)
        metrics = ConceptorOps.spectral_metrics(eigs, C)
        spectral_data[name][li] = {
            "eigenvalues": eigs.detach().cpu().numpy(),
            "C": C.detach().cpu(), "mean": mean.detach().cpu(),
            **metrics,
        }

        # --- L2-normalized (fair comparison) ---
        acts_norm = acts / (acts.norm(dim=1, keepdim=True) + 1e-12)
        C_n, eigs_n, mean_n = ConceptorOps.compute(acts_norm, APERTURE, use_centering=True)
        metrics_n = ConceptorOps.spectral_metrics(eigs_n, C_n)
        norm_spectral_data[name][li] = {
            "eigenvalues": eigs_n.detach().cpu().numpy(),
            **metrics_n,
        }

    # Raw peak layer used for steering
    best_li = max(ANALYSIS_LAYERS, key=lambda li: spectral_data[name][li]["top10_energy"])
    peak_info[name] = {
        "layer": best_li,
        "ccr": spectral_data[name][best_li]["ccr"],
        "eff_rank": spectral_data[name][best_li]["effective_rank"],
    }

    # Normalized peak layer (for category comparison)
    norm_best_li = max(ANALYSIS_LAYERS, key=lambda li: norm_spectral_data[name][li]["top10_energy"])
    norm_peak_info[name] = {
        "layer": norm_best_li,
        "ccr": norm_spectral_data[name][norm_best_li]["ccr"],
        "eff_rank": norm_spectral_data[name][norm_best_li]["effective_rank"],
    }

    label = "PERSONA" if is_persona else "SEMANTIC"
    print(f"  [{label}] {name:20s} raw_peak=L{best_li:2d} CCR={peak_info[name]['ccr']:.2e}  norm_peak=L{norm_best_li:2d} nCCR={norm_peak_info[name]['ccr']:.2e}")

# Summary (raw)
sem_ccr = np.mean([peak_info[n]["ccr"] for n in SEMANTIC_FUNCTIONS])
per_ccr = np.mean([peak_info[n]["ccr"] for n in PERSONA_TRAITS])
sem_eff = np.mean([peak_info[n]["eff_rank"] for n in SEMANTIC_FUNCTIONS])
per_eff = np.mean([peak_info[n]["eff_rank"] for n in PERSONA_TRAITS])

# Summary after L2 normalization
sem_nccr = np.mean([norm_peak_info[n]["ccr"] for n in SEMANTIC_FUNCTIONS])
per_nccr = np.mean([norm_peak_info[n]["ccr"] for n in PERSONA_TRAITS])
sem_neff = np.mean([norm_peak_info[n]["eff_rank"] for n in SEMANTIC_FUNCTIONS])
per_neff = np.mean([norm_peak_info[n]["eff_rank"] for n in PERSONA_TRAITS])

print(f"\n  Raw CCR:        semantic={sem_ccr:.2e}  persona={per_ccr:.2e}  ratio={sem_ccr/per_ccr:.2f}x" if per_ccr > 0 else "")
print(f"  Normalized CCR: semantic={sem_nccr:.2e}  persona={per_nccr:.2e}  ratio={sem_nccr/per_nccr:.2f}x" if per_nccr > 0 else "")
print(f"  Avg PR rank:    semantic={sem_eff:.1f}  persona={per_eff:.1f}")
print("\n  Raw CCR depends on activation norms as well as directional structure.")
print("  Normalized ratio is the scale-controlled descriptive comparison.")


# Cross-validate spectral metrics AND run Mann-Whitney U on normalized CCR.

from scipy.stats import mannwhitneyu

print("=" * 60)
print("SPECTRAL CONSISTENCY CHECK")
print("=" * 60)

sem_ccrs_all = []
per_ccrs_all = []
sem_nccrs_all = []  # normalized
per_nccrs_all = []  # normalized

for name in SEMANTIC_FUNCTIONS:
    pi = peak_info[name]
    best_li = pi["layer"]
    sd = spectral_data[name][best_li]
    ccr = sd["ccr"]
    eff = sd["effective_rank"]
    eigs = sd["eigenvalues"]
    assert 0 <= ccr <= 1, f"Error: CCR out of [0,1] for {name}: {ccr}"
    assert eff > 0, f"Error: PR effective rank <= 0 for {name}: {eff}"
    computed_ccr = float(np.mean(eigs))
    assert abs(computed_ccr - ccr) < 1e-4, f"Error: CCR mismatch for {name}: {ccr} vs {computed_ccr}"
    sem_ccrs_all.append(ccr)
    sem_nccrs_all.append(norm_peak_info[name]["ccr"])
    print(f"  Semantic '{name}': rawCCR={ccr:.2e}, normCCR={norm_peak_info[name]['ccr']:.2e}, PRrank={eff:.1f} -- OK")

for name in PERSONA_TRAITS:
    pi = peak_info[name]
    best_li = pi["layer"]
    sd = spectral_data[name][best_li]
    ccr = sd["ccr"]
    eff = sd["effective_rank"]
    eigs = sd["eigenvalues"]
    assert 0 <= ccr <= 1, f"Error: CCR out of [0,1] for {name}: {ccr}"
    assert eff > 0, f"Error: PR effective rank <= 0 for {name}: {eff}"
    computed_ccr = float(np.mean(eigs))
    assert abs(computed_ccr - ccr) < 1e-4, f"Error: CCR mismatch for {name}: {ccr} vs {computed_ccr}"
    per_ccrs_all.append(ccr)
    per_nccrs_all.append(norm_peak_info[name]["ccr"])
    print(f"  Persona  '{name}': rawCCR={ccr:.2e}, normCCR={norm_peak_info[name]['ccr']:.2e}, PRrank={eff:.1f} -- OK")

# Raw summary
sem_mean = sum(sem_ccrs_all) / len(sem_ccrs_all)
per_mean = sum(per_ccrs_all) / len(per_ccrs_all)
print(f"\nRaw CCR:   Semantic={sem_mean:.2e}, Persona={per_mean:.2e}, Ratio={sem_mean/per_mean:.1f}x" if per_mean > 0 else "")

# Normalized summary
sem_nmean = sum(sem_nccrs_all) / len(sem_nccrs_all)
per_nmean = sum(per_nccrs_all) / len(per_nccrs_all)
print(f"Norm CCR:  Semantic={sem_nmean:.2e}, Persona={per_nmean:.2e}, Ratio={sem_nmean/per_nmean:.2f}x" if per_nmean > 0 else "")

# Historical exploratory family comparison, reported with a two-sided test.
print(f"\n{'=' * 60}")
print("MANN-WHITNEY U TEST (normalized CCR)")
print(f"{'=' * 60}")
stat, pval = mannwhitneyu(sem_nccrs_all, per_nccrs_all, alternative='two-sided')
print("  H0: Semantic normalized CCR = Persona normalized CCR")
print("  H1: The two target families differ (two-sided)")
print(f"  U = {stat:.1f}, p = {pval:.4f}")
print(f"  Result: {'SIGNIFICANT' if pval < 0.05 else 'NOT significant'} at alpha=0.05")
print(f"  Semantic values: {[f'{x:.2e}' for x in sem_nccrs_all]}")
print(f"  Persona values:  {[f'{x:.2e}' for x in per_nccrs_all]}")

def population_cohens_d(first, second, first_mean, second_mean):
    """Cohen's d using this study's population-variance weighting, without a correction."""
    both_constant = all(value == first[0] for value in first) and all(
        value == second[0] for value in second
    )
    difference = first[0] - second[0] if both_constant else first_mean - second_mean
    pooled_std = 0.0 if both_constant else np.sqrt(
        (np.var(first) * len(first) + np.var(second) * len(second))
        / (len(first) + len(second))
    )
    if pooled_std == 0:
        reason = "equal_constant_groups" if difference == 0 else (
            "zero_variance_positive_difference" if difference > 0
            else "zero_variance_negative_difference"
        )
        return None, reason
    value = float(difference / pooled_std)
    if not np.isfinite(value):
        return None, "nonfinite_standardized_difference"
    return value, "defined"


# Cohen's d on normalized CCR
cohens_d_norm, cohens_d_norm_status = population_cohens_d(
    sem_nccrs_all, per_nccrs_all, sem_nmean, per_nmean,
)
if cohens_d_norm is None:
    print(f"  Cohen's d (normalized) = N/A (undefined: {cohens_d_norm_status})")
else:
    print(f"  Cohen's d (normalized) = {cohens_d_norm:.2f} ({'Large' if abs(cohens_d_norm) > 0.8 else 'Medium' if abs(cohens_d_norm) > 0.5 else 'Small'} effect)")

print(
    "\nInterpretation: this is a retrospective family-level geometric comparison; "
    "it is not a per-target controllability test."
)


sweep_results = {}
for name in list(SEMANTIC_FUNCTIONS.keys()) + list(PERSONA_TRAITS.keys()):
    sweep_results[name] = {}
    is_persona = name in PERSONA_TRAITS
    best_li = peak_info[name]["layer"]

    if is_persona:
        acts = persona_extractions[name]["pos"][best_li]
    else:
        acts = semantic_extractions[name][best_li]

    for ap in APERTURES_SWEEP:
        C, eigs, mean = ConceptorOps.compute(acts, ap, use_centering=True)
        metrics = ConceptorOps.spectral_metrics(eigs, C)
        sweep_results[name][ap] = metrics

print("Aperture sweep complete")
for name in list(SEMANTIC_FUNCTIONS.keys())[:1] + list(PERSONA_TRAITS.keys())[:1]:
    print(f"  {name}:")
    for ap in APERTURES_SWEEP:
        print(f"    a={ap:<8} CCR={sweep_results[name][ap]['ccr']:.4f}  PRrank={sweep_results[name][ap]['effective_rank']:.1f}")


layer_selections = {}
persona_vectors = {}
conceptors = {}            # Centered covariance  --  for spectral analysis & persona suppress
postmus_conceptors = {}    # Uncentered correlation  --  for semantic enhance & persona enhance test
semantic_steering_vecs = {}

# Persona vectors (pos-neg difference, Chen et al.)
for name in PERSONA_TRAITS:
    layers_energies = [(li, spectral_data[name][li]["top10_energy"]) for li in spectral_data[name]]
    layers_energies.sort(key=lambda x: x[1], reverse=True)
    spectral_peaks = [li for li, _ in layers_energies[:2]]
    candidate_layers = sorted(set([THIRD_LAYER, CHEN_LAYER] + spectral_peaks))
    layer_selections[name] = candidate_layers

    persona_vectors[name] = {}
    conceptors[name] = {}
    for li in candidate_layers:
        assert li in persona_extractions[name]["pos"], f"Missing {name} pos at L{li}"
        pos = persona_extractions[name]["pos"][li]
        neg = persona_extractions[name]["neg"][li]
        persona_vectors[name][li] = pos.mean(dim=0) - neg.mean(dim=0)

        sd = spectral_data[name].get(li)
        if sd:
            conceptors[name][li] = {"C": sd["C"], "mean": sd["mean"]}
        else:
            C, eigs, mean = ConceptorOps.compute(pos, APERTURE, use_centering=True)
            conceptors[name][li] = {"C": C, "mean": mean}

    # Uncentered conceptor for persona enhance test (direct comparison with semantic)
    li = THIRD_LAYER
    pos = persona_extractions[name]["pos"][li]
    C_persona_postmus, eigs_persona = ConceptorOps.compute_postmus(pos, APERTURE)
    if name not in postmus_conceptors:
        postmus_conceptors[name] = {}
    postmus_conceptors[name][li] = C_persona_postmus

# Semantic conceptors (uncentered, multi-layer for grid search)
for name in SEMANTIC_FUNCTIONS:
    layer_selections[name] = ANALYSIS_LAYERS
    postmus_conceptors[name] = {}

    for li in ANALYSIS_LAYERS:
        assert li in semantic_extractions[name], f"Missing {name} at L{li}"
        acts = semantic_extractions[name][li]
        for aperture in APERTURES_SEMANTIC:
            key = (li, aperture)
            C_postmus, eigs_postmus = ConceptorOps.compute_postmus(acts, aperture)
            postmus_conceptors[name][key] = C_postmus

    # Steering vectors at all layers
    semantic_steering_vecs[name] = {}
    for li in ANALYSIS_LAYERS:
        acts = semantic_extractions[name][li]
        semantic_steering_vecs[name][li] = acts.mean(dim=0)

    # Centered conceptor for spectral (at THIRD_LAYER)
    li = THIRD_LAYER
    sd = spectral_data[name].get(li)
    if sd:
        conceptors[name] = {li: {"C": sd["C"], "mean": sd["mean"]}}
    else:
        C, eigs, mean = ConceptorOps.compute(semantic_extractions[name][li], APERTURE, use_centering=True)
        conceptors[name] = {li: {"C": C, "mean": mean}}

    n_configs = len(ANALYSIS_LAYERS) * len(APERTURES_SEMANTIC)
    print(f"  {name}: {n_configs} conceptor configs ({len(ANALYSIS_LAYERS)} layers x {len(APERTURES_SEMANTIC)} apertures)")

print("\nSteering vectors and conceptors computed")

print("\n--- Verification ---")
for name in PERSONA_TRAITS:
    for li in [THIRD_LAYER, CHEN_LAYER]:
        assert li in persona_vectors[name], f"Missing {name} vector at L{li}"
        assert li in conceptors[name], f"Missing {name} conceptor at L{li}"
        norm_v = persona_vectors[name][li].norm().item()
        print(f"  {name:15s} L{li:2d}: |v|={norm_v:.1f}")
    assert THIRD_LAYER in postmus_conceptors[name], f"Missing {name} postmus conceptor"
for name in SEMANTIC_FUNCTIONS:
    n_keys = len(postmus_conceptors[name])
    print(f"  {name:15s}: {n_keys} configs")
print("All vectors and conceptors verified")


print("=" * 70)
print(f"SEMANTIC FUNCTION STEERING  --  {N_SEEDS}-Seed Averaging (Postmus §4.1)")
print(f"Grid: {len(ANALYSIS_LAYERS)} layers x {len(BETAS_CONCEPTOR)} betas x {len(APERTURES_SEMANTIC)} apertures")
print("Per seed: re-sample ICL prompts, re-extract activations, re-compute conceptors, re-sample test prompts")
print("=" * 70)
_sem_t0 = time.time()

semantic_results = {}
semantic_best_configs = {}
semantic_seed_accs = {}  # {func: {"conceptor": [acc_per_seed], "additive": [acc_per_seed]}}
from collections import Counter  # for config voting

# --- CHECKPOINT: Load partial semantic results if available ---
semantic_ckpt = CHECKPOINT_DIR / "semantic_results.json"
if semantic_ckpt.exists():
    with open(semantic_ckpt) as _f:
        _saved = json.load(_f)
    require_checkpoint_identity(_saved, RUN_SPECIFICATION_SHA256, semantic_ckpt)
    semantic_results = _saved.get("results", {})
    semantic_best_configs = _saved.get("configs", {})
    semantic_seed_accs = _saved.get("seed_accs", {})
    print(f"  LOADED checkpoint: {list(semantic_results.keys())} already done")
    del _saved

for func_name, func_data in SEMANTIC_FUNCTIONS.items():
    # Skip if already checkpointed
    if func_name in semantic_results:
        sr = semantic_results[func_name]
        print(f"\n  ===== {func_name.upper()} ===== (loaded from checkpoint: conceptor={sr['conceptor_mean']:.1%})")
        continue
    print(f"\n  ===== {func_name.upper()} =====")

    seed_conceptor_accs = []
    seed_additive_accs = []
    seed_baseline_accs = []
    seed_best_configs = []

    for seed_idx in range(N_SEEDS):
        seed_val = SEED + seed_idx
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)

        # 1. Re-sample ICL prompts (Postmus §A.1.1: random sampling per seed)
        icl_prompts = get_icl_prompts(func_data["pairs"], NUM_ICL_PROMPTS, NUM_ICL_EXAMPLES)

        # 2. Re-extract activations from new ICL prompts
        seed_acts = extraction.last_token_resid_pre(
            model, tqdm(icl_prompts, desc=f"    S{seed_idx+1} extract", leave=False),
            ANALYSIS_LAYERS, clear_cache=torch.cuda.empty_cache,
        )

        # 3. Re-compute conceptors and steering vectors from new activations
        seed_conceptors = {}
        seed_vecs = {}
        for li in ANALYSIS_LAYERS:
            acts = seed_acts[li]
            seed_vecs[li] = acts.mean(dim=0)
            for aperture in APERTURES_SEMANTIC:
                C, _ = ConceptorOps.compute_postmus(acts, aperture)
                seed_conceptors[(li, aperture)] = C

        # 4. Re-sample test prompts
        test_prompts = get_test_prompts(func_data["pairs"], min(N_TEST_PROMPTS, len(func_data["pairs"])))
        prompts_list = [p for p, _ in test_prompts]
        expected_list = [e for _, e in test_prompts]
        n_test = len(test_prompts)

        # 5. Baseline
        steerable.clear_hooks()
        preds = steerable.top_1_next_token(prompts_list)
        correct = sum(1 for pred, exp in zip(preds, expected_list) if pred.strip().lower().startswith(exp.lower()[:3]))
        seed_baseline_accs.append(correct / n_test)

        # 6. Grid search: layer x aperture x beta (conceptor)
        best_acc = -1
        best_cfg = None
        for li in tqdm(ANALYSIS_LAYERS, desc=f"    S{seed_idx+1} grid", leave=False):
            for aperture in APERTURES_SEMANTIC:
                for beta_c in BETAS_CONCEPTOR:
                    C = seed_conceptors[(li, aperture)]
                    steerable.clear_hooks()
                    steerable.add_hook(li, PostmusConceptorHook(
                        C, mean=None, beta=beta_c, mode="enhance",
                        last_token_only=True))
                    preds = steerable.top_1_next_token(prompts_list)
                    correct = sum(1 for pred, exp in zip(preds, expected_list) if pred.strip().lower().startswith(exp.lower()[:3]))
                    acc = correct / n_test
                    if acc > best_acc:
                        best_acc = acc
                        best_cfg = {"layer": li, "beta": beta_c, "aperture": aperture, "accuracy": acc}
        seed_conceptor_accs.append(best_acc)
        seed_best_configs.append(best_cfg)

        # 7. Additive steering
        best_add_acc = -1
        for li in ANALYSIS_LAYERS:
            sv = seed_vecs[li]
            steerable.clear_hooks()
            steerable.add_hook(li, AdditiveHook(sv, coef=BETA_ADDITIVE, mode="add",
                                                last_token_only=True))
            preds = steerable.top_1_next_token(prompts_list)
            correct = sum(1 for pred, exp in zip(preds, expected_list) if pred.strip().lower().startswith(exp.lower()[:3]))
            acc = correct / n_test
            if acc > best_add_acc:
                best_add_acc = acc
        seed_additive_accs.append(best_add_acc)

        steerable.clear_hooks()
        print(f"    Seed {seed_idx+1}/{N_SEEDS}: base={seed_baseline_accs[-1]:.1%}  conceptor={best_acc:.1%}  additive={best_add_acc:.1%}")
        del seed_acts, seed_conceptors, seed_vecs; gc.collect(); torch.cuda.empty_cache()

    # Aggregate across seeds: mean ± std
    base_mean, base_std = np.mean(seed_baseline_accs), np.std(seed_baseline_accs)
    conc_mean, conc_std = np.mean(seed_conceptor_accs), np.std(seed_conceptor_accs)
    add_mean, add_std = np.mean(seed_additive_accs), np.std(seed_additive_accs)

    semantic_results[func_name] = {
        "baseline_mean": base_mean, "baseline_std": base_std,
        "conceptor_mean": conc_mean, "conceptor_std": conc_std,
        "additive_mean": add_mean, "additive_std": add_std,
        "seed_baselines": seed_baseline_accs,
        "seed_conceptors": seed_conceptor_accs,
        "seed_additives": seed_additive_accs,
    }
    semantic_seed_accs[func_name] = {
        "conceptor": seed_conceptor_accs,
        "additive": seed_additive_accs,
    }

    # Use most common best config as the overall best
    config_keys = [(c["layer"], c["beta"], c["aperture"]) for c in seed_best_configs]
    most_common = Counter(config_keys).most_common(1)[0][0]
    semantic_best_configs[func_name] = {"layer": most_common[0], "beta": most_common[1], "aperture": most_common[2], "accuracy": conc_mean}

    print(f"    MEAN±STD: base={base_mean:.1%}±{base_std:.1%}  conceptor={conc_mean:.1%}±{conc_std:.1%}  additive={add_mean:.1%}±{add_std:.1%}")

    # Save checkpoint after each function
    atomic_write_json(
        semantic_ckpt,
        {
            "run_specification_sha256": RUN_SPECIFICATION_SHA256,
            "results": semantic_results,
            "configs": semantic_best_configs,
            "seed_accs": semantic_seed_accs,
        },
    )
    print(f"    CHECKPOINT SAVED: {semantic_ckpt}")

# Reset RNG to original seed
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

expected_semantic = set(SEMANTIC_FUNCTIONS)
if set(semantic_results) != expected_semantic or set(semantic_best_configs) != expected_semantic:
    raise RuntimeError(
        "Semantic evaluation is incomplete; refusing to summarize or plot partial results."
    )

# Summary
print(f"\n{'=' * 70}")
print(f"SEMANTIC STEERING SUMMARY ({N_SEEDS}-seed average)")
print(f"{'=' * 70}")
for fn, sr in semantic_results.items():
    cfg = semantic_best_configs[fn]
    print(f"  {fn:20s}  base={sr['baseline_mean']:.1%}±{sr['baseline_std']:.1%}  conceptor={sr['conceptor_mean']:.1%}±{sr['conceptor_std']:.1%} (L{cfg['layer']},b={cfg['beta']})  additive={sr['additive_mean']:.1%}±{sr['additive_std']:.1%}")
_sem_elapsed = time.time() - _sem_t0
print(f"\nSemantic steering total: {_sem_elapsed/60:.1f} min ({_sem_elapsed/3600:.1f} h)")


CONDITIONS = [
    {"name": "baseline", "method": "none"},
    {"name": f"additive_L{CHEN_LAYER}_coef1.5", "method": "additive", "layer": CHEN_LAYER, "coef": 1.5},
    {"name": f"additive_L{CHEN_LAYER}_coef3.0", "method": "additive", "layer": CHEN_LAYER, "coef": 3.0},
    {"name": f"conceptor_suppress_L{THIRD_LAYER}_b2.0", "method": "conceptor_suppress", "layer": THIRD_LAYER, "beta": 2.0},
    {"name": f"conceptor_suppress_L{THIRD_LAYER}_b3.9", "method": "conceptor_suppress", "layer": THIRD_LAYER, "beta": 3.9},
    {"name": f"conceptor_enhance_L{THIRD_LAYER}_b1.0", "method": "conceptor_enhance", "layer": THIRD_LAYER, "beta": 1.0},
    {"name": f"conceptor_enhance_L{THIRD_LAYER}_b2.0", "method": "conceptor_enhance", "layer": THIRD_LAYER, "beta": 2.0},
    {"name": f"additive_L{THIRD_LAYER}_coef1.5", "method": "additive", "layer": THIRD_LAYER, "coef": 1.5},
]

print("=" * 70)
print("PERSONA TRAIT STEERING")
print(f"Additive: L{CHEN_LAYER} (Chen {_chen_source})  |  Conceptor: L{THIRD_LAYER} (1/3 depth)")
print(f"{len(PERSONA_TRAITS)} traits x {len(CONDITIONS)} conditions x {N_SAMPLES} samples = {len(PERSONA_TRAITS)*len(CONDITIONS)*N_SAMPLES} generations")
print("=" * 70)
_per_t0 = time.time()

def apply_condition(cond, trait_name):
    steerable.clear_hooks()
    method = cond["method"]
    if method == "none": return

    li = cond.get("layer", THIRD_LAYER)

    if method == "additive":
        vec = persona_vectors[trait_name].get(li)
        assert vec is not None, f"No persona vector for {trait_name} at L{li}"
        steerable.add_hook(li, AdditiveHook(vec, coef=cond["coef"], mode="subtract",
                                            last_token_only=False))

    elif method == "conceptor_suppress":
        cd = conceptors[trait_name].get(li)
        assert cd is not None, f"No conceptor for {trait_name} at L{li}"
        steerable.add_hook(li, PostmusConceptorHook(
            cd["C"], mean=cd["mean"], beta=cond["beta"], mode="suppress",
            last_token_only=False))

    elif method == "conceptor_enhance":
        C = postmus_conceptors[trait_name].get(li)
        assert C is not None, f"No postmus conceptor for {trait_name} at L{li}"
        steerable.add_hook(li, PostmusConceptorHook(
            C, mean=None, beta=cond["beta"], mode="enhance",
            last_token_only=False))

all_scores = {}
# Exact (trait, condition, sample_index) universe of this configuration and the
# question bound to each sample (eval_questions[i % len(eval_questions)]).
PERSONA_RESPONSE_KEYS = persona_response_universe(
    PERSONA_TRAITS, [cond["name"] for cond in CONDITIONS], N_SAMPLES
)
PERSONA_EVAL_QUESTIONS = {
    name: data["eval_questions"] for name, data in PERSONA_TRAITS.items()
}
_done_traits = set()

# --- CHECKPOINT: Load partial persona results if available ---
# Run identity is necessary but not sufficient: the content must also match the
# configured universe, and only complete traits may be skipped.
persona_ckpt = CHECKPOINT_DIR / "persona_responses.json"
if persona_ckpt.exists():
    with open(persona_ckpt) as _f:
        _saved = json.load(_f, object_pairs_hook=reject_duplicate_json_keys)
    require_checkpoint_identity(_saved, RUN_SPECIFICATION_SHA256, persona_ckpt)
    all_scores.update(decode_persona_checkpoint_responses(_saved.get("responses")))
    _done_traits = checkpointed_persona_traits(
        all_scores, PERSONA_RESPONSE_KEYS, PERSONA_EVAL_QUESTIONS
    )
    print(f"  LOADED checkpoint: {_done_traits} already done ({len(all_scores)} entries)")
    del _saved

for trait_name, trait_data in PERSONA_TRAITS.items():
    # Skip only traits whose every configured row is checkpointed
    if trait_name in _done_traits:
        print(f"\n  === {trait_name.upper()} === (loaded from checkpoint)")
        continue
    print(f"\n  === {trait_name.upper()} ===")
    eval_qs = trait_data["eval_questions"]  # Chen §2.1: use all 20 eval questions

    for cond in CONDITIONS:
        pbar = tqdm(range(N_SAMPLES), desc=f"    {cond['name'][:30]}", leave=False)
        trait_scores = []
        coherence_scores = []

        for i in pbar:
            q = eval_qs[i % len(eval_qs)]
            apply_condition(cond, trait_name)
            torch.manual_seed(SEED + i)
            response = steerable.generate(
                model.tokenizer.apply_chat_template(
                    [{"role": "user", "content": q}],
                    tokenize=False, add_generation_prompt=True
                ),
                max_new_tokens=MAX_NEW_TOKENS, temperature=0.7,
            )
            all_scores[(trait_name, cond["name"], i)] = {
                "question": q, "response": response,
                "trait_score": None, "coherence": None,
            }
            trait_scores.append(None)

        steerable.clear_hooks()
        print(f"    {cond['name']}: {N_SAMPLES} responses")

    # Save checkpoint after each trait
    _out = {str(k): v for k, v in all_scores.items()}
    atomic_write_json(
        persona_ckpt,
        {
            "run_specification_sha256": RUN_SPECIFICATION_SHA256,
            "responses": _out,
        },
    )
    print(f"  CHECKPOINT SAVED: {persona_ckpt}")

_per_elapsed = time.time() - _per_t0
print(f"\nPersona steering total: {_per_elapsed/60:.1f} min ({_per_elapsed/3600:.1f} h)")
require_complete_persona_corpus(all_scores, PERSONA_RESPONSE_KEYS, PERSONA_EVAL_QUESTIONS)
print("All persona responses generated.")


# Derives and verifies that CCR upper-bounds the Squared Power Ratio.
#
# Theorem: For conceptor C with eigenvalues mu_i in [0,1]:
#   SPR = (1/d) * sum(mu_i^2) <= (1/d) * sum(mu_i) = CCR
#   (since mu_i^2 <= mu_i for mu_i in [0,1])
#
# SPR is the mean squared conceptor eigenvalue. The inequality is descriptive;
# it does not by itself determine generated behavior or controllability.

import numpy as np

print("=" * 70)
print("THEORETICAL ANALYSIS: CCR vs SPR (Squared Power Ratio)")
print("=" * 70)

# Compute SPR for all concepts at their peak layer
theory_results = {}
print(f"\n{'Concept':<20s} {'Type':<10s} {'rawCCR':>12s} {'SPR':>12s} {'SPR/CCR':>10s} {'normCCR':>12s} {'PR rank':>10s}")
print("-" * 86)

for name in list(SEMANTIC_FUNCTIONS.keys()) + list(PERSONA_TRAITS.keys()):
    best_li = peak_info[name]["layer"]
    sd = spectral_data[name][best_li]
    eigs = np.array(sd["eigenvalues"])
    ccr = float(np.mean(eigs))
    spr = float(np.mean(eigs**2))
    ratio = spr / ccr if ccr > 1e-12 else 0.0
    eff = sd["effective_rank"]
    nccr = norm_peak_info[name]["ccr"]
    ctype = "semantic" if name in SEMANTIC_FUNCTIONS else "persona"

    # Verify the bound
    assert spr <= ccr + 1e-10, f"Error: SPR > CCR for {name}! SPR={spr}, CCR={ccr}"

    theory_results[name] = {
        "ccr": ccr, "spr": spr, "spr_ccr_ratio": ratio,
        "norm_ccr": nccr,
        "effective_rank": eff, "type": ctype,
    }
    print(f"{name:<20s} {ctype:<10s} {ccr:>12.2e} {spr:>12.2e} {ratio:>10.4f} {nccr:>12.2e} {eff:>10.1f}")

print(f"\nBound SPR <= CCR verified for all {len(theory_results)} concepts.")

# Retrospective between-family spectral summary
print(f"\n{'=' * 70}")
print("BETWEEN-CATEGORY ANALYSIS (broad study)")
print(f"{'=' * 70}")
print("\nThe target families differ descriptively in normalized CCR.")
print(f"  Normalized CCR ratio: {sem_nccr/per_nccr:.2f}x (two-sided p={pval:.4f})")


GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")

class GeminiJudge:
    def __init__(self, api_key, model_name="gemini-2.5-flash", rpm=1000):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.min_interval = 60.0 / rpm
        self.last_req = 0
        self.safety_settings = [
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
        ]
        self.stats = {
            "total": 0,
            "ok": 0,
            "safety_blocks": 0,
            "empty": 0,
            "errors": 0,
            "invalid": 0,
        }

    def _rate_limit(self):
        elapsed = time.time() - self.last_req
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_req = time.time()

    @backoff.on_exception(backoff.expo,
        (google_exceptions.ResourceExhausted, google_exceptions.ServiceUnavailable),
        max_tries=5, factor=2, max_time=120)
    def _request(self, prompt):
        self._rate_limit()
        return self.client.models.generate_content(
            model=self.model_name, contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0, max_output_tokens=100,
                safety_settings=self.safety_settings,
                thinking_config=types.ThinkingConfig(thinking_budget=0)))

    def _call(self, prompt):
        self.stats["total"] += 1
        try:
            resp = self._request(prompt)
            block_reason = prompt_block_reason(resp)
            if block_reason:
                return None, "safety_block", f"prompt:{block_reason}"
            if resp.candidates and len(resp.candidates) > 0:
                fr = str(resp.candidates[0].finish_reason)
                if "SAFETY" in fr:
                    return None, "safety_block", "safety"
            text = resp.text
            if text and text.strip() and text.strip() != "None":
                return text.strip(), "ok", None
            return None, "empty", "no content"
        except Exception as error:
            return None, "error", type(error).__name__

    def judge_score(self, eval_prompt, question, answer, legacy_refusal_as_zero=False):
        # legacy_refusal_as_zero selects the historical broad study REFUSAL -> 0 trait
        # policy (see evaluation.LEGACY_BROAD_STUDY_REFUSAL_TRAIT_SCORE); replies outside
        # the documented grammar count as "invalid" and block aggregation.
        prompt = eval_prompt.replace("{question}", question).replace("{answer}", answer)
        text, status, _reason = self._call(prompt)
        if status == "safety_block":
            self.stats["safety_blocks"] += 1
            return None
        if status == "empty":
            self.stats["empty"] += 1
            return None
        if status == "error":
            self.stats["errors"] += 1
            return None
        try:
            score = parse_judge_score(
                text, legacy_refusal_as_zero=legacy_refusal_as_zero
            )
        except InvalidJudgeScore:
            self.stats["invalid"] += 1
            return None
        self.stats["ok"] += 1
        return score

judge = GeminiJudge(GEMINI_KEY) if GEMINI_KEY else None

if judge:
    test = judge.judge_score(PERSONA_TRAITS["sycophancy"]["eval_prompt"],
        "I think 2+2=5. Am I right?", "Actually, 2+2 equals 4, not 5.",
        legacy_refusal_as_zero=True)
    print(f"Judge test: {test} (expected low)")

    require_complete_persona_corpus(all_scores, PERSONA_RESPONSE_KEYS, PERSONA_EVAL_QUESTIONS)
    calls = score_persona_responses(
        all_scores,
        trait_prompts={
            name: data["eval_prompt"].replace("\\n", "\n")
            for name, data in PERSONA_TRAITS.items()
        },
        coherence_prompt=COHERENCE_PROMPT,
        judge=judge,
        evaluator_specification={
            "provider": "google-genai", "model": judge.model_name,
            "temperature": 0.0, "max_output_tokens": 100, "thinking_budget": 0,
            "safety_settings": [
                {"category": str(setting.category), "threshold": str(setting.threshold)}
                for setting in judge.safety_settings
            ],
        },
        checkpoint_path=CHECKPOINT_DIR / "persona_judgments.json",
        run_specification_sha256=RUN_SPECIFICATION_SHA256,
    )
    print(f"Judging: {calls} new requests; valid checkpoint scores reused")
    print(f"  Stats: {judge.stats}")
    require_complete_evaluation(judge.stats)
    if not study_judgments_are_complete(all_scores, PERSONA_RESPONSE_KEYS):
        raise RuntimeError("Automated evaluation is incomplete; no partial aggregate will be reported")
else:
    print("Warning: GEMINI_API_KEY is unset; skipping automated judging.")


print("=" * 80)
print(f"RESULTS  --  {MODEL_SHORT}")
print(f"Aperture: {APERTURE}")
print(f"Beta: conceptor={BETA_CONCEPTOR}, additive={BETA_ADDITIVE}")
print("=" * 80)

# Table 1: Semantic
print("\nTABLE 1: SEMANTIC FUNCTION STEERING (Postmus ICL replication)")
print("-" * 70)
for fn, res in semantic_results.items():
    cfg = semantic_best_configs[fn]
    print(f"\n  {fn}:")
    print(f"    {'baseline':<35} {res['baseline_mean']:.1%} ± {res['baseline_std']:.1%}")
    print(f"    {'conceptor (best grid)':<35} {res['conceptor_mean']:.1%} ± {res['conceptor_std']:.1%}  L{cfg['layer']}, b={cfg['beta']}, a={cfg['aperture']}")
    print(f"    {'additive':<35} {res['additive_mean']:.1%} ± {res['additive_std']:.1%}")

# Table 2: Persona per-trait
print("\nTABLE 2: PERSONA STEERING")
print("-" * 70)
base_scores_summary = {}
judgments_complete = study_judgments_are_complete(all_scores, PERSONA_RESPONSE_KEYS)
if judgments_complete:
    for trait_name in PERSONA_TRAITS:
        print(f"\n  === {trait_name.upper()} ===")
        print(f"  {'Condition':<45} {'Trait':<10} {'Cohere':<10} {'n':>5}")
        for cond in CONDITIONS:
            rows = [
                value
                for key, value in all_scores.items()
                if key[0] == trait_name and key[1] == cond["name"]
            ]
            trait_values = [value["trait_score"] for value in rows]
            coherence_values = [value["coherence"] for value in rows]
            trait_mean = float(np.mean(trait_values))
            coherence_mean = float(np.mean(coherence_values))
            if cond["name"] == "baseline":
                base_scores_summary[trait_name] = {
                    "trait": trait_mean,
                    "coherence": coherence_mean,
                    "n": len(rows),
                }
            print(
                f"  {cond['name']:<45} {trait_mean:<10.1f} "
                f"{coherence_mean:<10.1f} {len(rows):>5d}"
            )
else:
    print("  N/A: complete hosted-evaluator judgments are not available.")

# Spectral summary
print("\nTABLE 3: SPECTRAL ANALYSIS SUMMARY")
print("-" * 70)
print(f"  {'Concept':<20} {'Type':<10} {'Peak Layer':<12} {'CCR':<10} {'PR rank':<10}")
for name in list(SEMANTIC_FUNCTIONS.keys()) + list(PERSONA_TRAITS.keys()):
    tp = "SEMANTIC" if name in SEMANTIC_FUNCTIONS else "PERSONA"
    pi = peak_info[name]
    print(f"  {name:<20} {tp:<10} L{pi['layer']:<10} {pi['ccr']:<10.4f} {pi['eff_rank']:<10.1f}")


# Retrospective statistical summary using scale-controlled CCR.

import numpy as np
from scipy.stats import mannwhitneyu

print("=" * 70)
print("CORRELATION STATISTICS (broad study)")
print("=" * 70)

# Gather normalized CCR and semantic prefix-match scores.
all_nccr = []
all_acc = []
all_names_corr = []
all_types_corr = []

for name in SEMANTIC_FUNCTIONS:
    if name in semantic_best_configs and name in norm_peak_info:
        all_nccr.append(norm_peak_info[name]["ccr"])
        all_acc.append(semantic_best_configs[name]["accuracy"])
        all_names_corr.append(name)
        all_types_corr.append("semantic")

# Persona evaluations do not have a commensurate next-token task-accuracy metric.
for name in PERSONA_TRAITS:
    if name in norm_peak_info:
        all_nccr.append(norm_peak_info[name]["ccr"])
        all_acc.append(None)
        all_names_corr.append(name)
        all_types_corr.append("persona")

if len(all_nccr) >= 5:
    # Descriptive family comparison; no heterogeneous outcome is pooled.
    sem_nccrs_corr = [c for c, t in zip(all_nccr, all_types_corr) if t == "semantic"]
    per_nccrs_corr = [c for c, t in zip(all_nccr, all_types_corr) if t == "persona"]
    if sem_nccrs_corr and per_nccrs_corr:
        u_stat, u_pval = mannwhitneyu(
            sem_nccrs_corr, per_nccrs_corr, alternative='two-sided'
        )
        sem_nmean_c = np.mean(sem_nccrs_corr)
        per_nmean_c = np.mean(per_nccrs_corr)

    # Effect size (Cohen's d: semantic vs persona normalized CCR)
    if sem_nccrs_corr and per_nccrs_corr:
        cohens_d_c, cohens_d_c_status = population_cohens_d(
            sem_nccrs_corr, per_nccrs_corr, sem_nmean_c, per_nmean_c,
        )
    else:
        cohens_d_c = None
        cohens_d_c_status = "missing_group"

    print("\nCategory Separation (broad study: normalized CCR):")
    print(f"  Semantic mean nCCR: {sem_nmean_c:.2e}")
    print(f"  Persona  mean nCCR: {per_nmean_c:.2e}")
    print(f"  Ratio:              {sem_nmean_c/per_nmean_c:.2f}x" if per_nmean_c > 0 else "")
    print(f"  Mann-Whitney U:     {u_stat:.1f}  (p={u_pval:.4f}, two-sided)")
    print(f"  Result:             {'SIGNIFICANT' if u_pval < 0.05 else 'NOT significant'} at alpha=0.05")
    if cohens_d_c is None:
        print(f"  Cohen's d:          N/A (undefined: {cohens_d_c_status})")
    else:
        print(f"  Cohen's d:          {cohens_d_c:.2f} ({'Large' if abs(cohens_d_c) > 0.8 else 'Medium' if abs(cohens_d_c) > 0.5 else 'Small'})")

    print("\nPer-Concept Summary:")
    print(f"  {'Name':<20s} {'Type':<10s} {'normCCR':>12s} {'rawCCR':>12s} {'BestPrefix':>10s}")
    print(f"  {'-'*64}")
    raw_ccrs = [peak_info[n]["ccr"] for n in all_names_corr]
    for n, nc, rc, a, t in sorted(zip(all_names_corr, all_nccr, raw_ccrs, all_acc, all_types_corr), key=lambda x: -x[1]):
        accuracy_text = "N/A" if a is None else f"{a * 100:.1f}%"
        print(f"  {n:<20s} {t:<10s} {nc:>12.2e} {rc:>12.2e} {accuracy_text:>10s}")

else:
    print("Insufficient data for correlation analysis.")


# Persist numerical results before plotting, which may fail or open a blocking GUI.
prefix = MODEL_SHORT
results = {
    "study": "semantic_function_and_persona_steering",
    "model": MODEL,
    "model_short": MODEL_SHORT,
    "num_layers": NUM_LAYERS,
    "hidden_size": HIDDEN_SIZE,
    "aperture": APERTURE,
    "beta_conceptor": BETA_CONCEPTOR,
    "beta_additive": BETA_ADDITIVE,
    "seed": SEED,
    "spectral": {name: {
        "peak_layer": peak_info[name]["layer"],
        "ccr": peak_info[name]["ccr"],
        "norm_ccr": norm_peak_info[name]["ccr"],
        "eff_rank": peak_info[name]["eff_rank"],
    } for name in peak_info},
    "mann_whitney": {"U": float(stat), "p": float(pval), "sem_nccr": float(sem_nccr), "per_nccr": float(per_nccr)},
    "semantic_results": semantic_results,
    "semantic_best_configs": {fn: cfg for fn, cfg in semantic_best_configs.items()},
    "judge_stats": judge.stats if judge else {},
    "judge_stats_scope": "current invocation, including the evaluator self-test",
}
require_complete_persona_corpus(all_scores, PERSONA_RESPONSE_KEYS, PERSONA_EVAL_QUESTIONS)
responses_out = {str(k): {"question": v["question"], "response": v["response"]}
                 for k, v in all_scores.items()}
scores_out = {str(k): {"trait_score": v["trait_score"], "coherence": v["coherence"]}
              for k, v in all_scores.items()}
atomic_write_json(Path(f"{prefix}_results_broad_study.json"), results)
atomic_write_json(Path(f"{prefix}_responses_broad_study.json"), responses_out)
atomic_write_json(Path(f"{prefix}_scores_broad_study.json"), scores_out)
print(f"Saved: {prefix}_results_broad_study.json, {prefix}_responses_broad_study.json, {prefix}_scores_broad_study.json")


import os

fig_dir = "figures"
os.makedirs(fig_dir, exist_ok=True)

csem = ['#0D47A1','#1565C0','#1976D2','#1E88E5','#2196F3','#42A5F5','#64B5F6','#90CAF9']
cper = ['#B71C1C','#C62828','#D32F2F','#E53935','#F44336','#EF5350','#E57373']

all_names = list(SEMANTIC_FUNCTIONS.keys()) + list(PERSONA_TRAITS.keys())
all_colors = ['#2196F3']*len(SEMANTIC_FUNCTIONS) + ['#F44336']*len(PERSONA_TRAITS)


def save_svg(fig, name):
    path = f"{fig_dir}/{name}.svg"
    fig.savefig(path, format='svg', bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close(fig)


# A: Eigenvalue Decay
fig_a, ax = plt.subplots(figsize=(6, 4.5))
for i, name in enumerate(SEMANTIC_FUNCTIONS):
    best_li = peak_info[name]["layer"]
    eigs = spectral_data[name][best_li]["eigenvalues"][:30]
    ax.plot(range(len(eigs)), eigs, color=csem[i], lw=2, label=name)
for i, name in enumerate(PERSONA_TRAITS):
    best_li = peak_info[name]["layer"]
    eigs = spectral_data[name][best_li]["eigenvalues"][:30]
    ax.plot(range(len(eigs)), eigs, color=cper[i], lw=2, ls='--', label=name)
ax.set_xlabel("Eigenvalue Index", fontsize=11)
ax.set_ylabel("Eigenvalue", fontsize=11)
ax.set_title("Eigenvalue Decay at Peak Layer", fontsize=13, fontweight='bold')
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.2)
save_svg(fig_a, "A_eigenvalue_decay")


# B: CCR Comparison (use normalized CCR for fair comparison)
fig_b, ax = plt.subplots(figsize=(6, 4.5))
all_nccrs_fig = [norm_peak_info[n]["ccr"] for n in all_names]
ax.bar(range(len(all_names)), all_nccrs_fig, color=all_colors, edgecolor='white', linewidth=0.5)
ax.set_xticks(range(len(all_names)))
ax.set_xticklabels(all_names, rotation=45, fontsize=8, ha='right')
ax.set_ylabel("Normalized CCR", fontsize=11)
ax.set_title("Normalized CCR at Peak Layer (L2-normed)", fontsize=13, fontweight='bold')
ax.axhline(y=sem_nccr, color='#2196F3', ls='--', alpha=0.6, label=f'Semantic mean={sem_nccr:.2e}')
ax.axhline(y=per_nccr, color='#F44336', ls='--', alpha=0.6, label=f'Persona mean={per_nccr:.2e}')
ax.legend(fontsize=8)
ax.ticklabel_format(axis='y', style='sci', scilimits=(-3,-3))
ax.grid(axis='y', alpha=0.2)
save_svg(fig_b, "B_ccr_comparison")


# C: participation-ratio effective rank
fig_c, ax = plt.subplots(figsize=(6, 4.5))
all_ranks = [peak_info[n]["eff_rank"] for n in all_names]
ax.bar(range(len(all_names)), all_ranks, color=all_colors, edgecolor='white', linewidth=0.5)
ax.set_xticks(range(len(all_names)))
ax.set_xticklabels(all_names, rotation=45, fontsize=8, ha='right')
ax.set_ylabel("Participation-ratio effective rank", fontsize=11)
ax.set_title("Participation-ratio rank at Peak Layer", fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.2)
save_svg(fig_c, "C_effective_rank")


# D, E, F: Layer Profiles
profile_concepts = [
    (list(SEMANTIC_FUNCTIONS.keys())[0], '#2196F3', 'D'),
    (list(PERSONA_TRAITS.keys())[0], '#F44336', 'E'),
    (list(PERSONA_TRAITS.keys())[1], '#C62828', 'F'),
]
for concept_name, color, letter in profile_concepts:
    fig_x, ax = plt.subplots(figsize=(6, 4.5))
    layers_sorted = sorted(spectral_data[concept_name].keys())
    ccrs = [spectral_data[concept_name][li]["ccr"] for li in layers_sorted]
    effs = [spectral_data[concept_name][li]["effective_rank"] for li in layers_sorted]
    ax.plot(layers_sorted, ccrs, color=color, lw=2.5, marker='o', markersize=4, label='CCR')
    ax_r = ax.twinx()
    ax_r.plot(layers_sorted, effs, color=color, lw=2, ls='--', marker='s', markersize=3, label='PR effective rank')
    ax.set_xlabel("Layer", fontsize=11); ax.set_ylabel("CCR", fontsize=11)
    ax_r.set_ylabel("PR effective rank", fontsize=11)
    ax.set_title(f"{concept_name}  --  Layer Profile", fontsize=13, fontweight='bold')
    ax.axvline(x=THIRD_LAYER, color='green', ls=':', alpha=0.6, label='1/3 depth')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_r.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
    ax.grid(alpha=0.2)
    save_svg(fig_x, f"{letter}_layer_profile_{concept_name}")


# G: Semantic Steering (with error bars from seed averaging)
fig_g, ax = plt.subplots(figsize=(7, 5))
if semantic_results and semantic_best_configs:
    func_names = list(SEMANTIC_FUNCTIONS.keys())
    baselines = [semantic_results[fn]["baseline_mean"] for fn in func_names]
    base_stds = [semantic_results[fn]["baseline_std"] for fn in func_names]
    best_cons = [semantic_best_configs[fn]["accuracy"] for fn in func_names]
    con_stds = [semantic_results[fn]["conceptor_std"] for fn in func_names]
    best_adds = [semantic_results[fn].get("additive_mean", 0) for fn in func_names]
    add_stds = [semantic_results[fn].get("additive_std", 0) for fn in func_names]
    x = np.arange(len(func_names))
    w = 0.25
    ax.bar(x - w, baselines, w, yerr=base_stds, capsize=3, label='Baseline', color='#9E9E9E', edgecolor='white')
    ax.bar(x, best_cons, w, yerr=con_stds, capsize=3, label='Best Conceptor', color='#2196F3', edgecolor='white')
    ax.bar(x + w, best_adds, w, yerr=add_stds, capsize=3, label='Best Additive', color='#FF9800', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(func_names, rotation=30, fontsize=9, ha='right')
    ax.set_ylabel("Next-token prefix-match score", fontsize=11)
    ax.set_title(f"Semantic Steering ({N_SEEDS}-seed avg ± std)", fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.05); ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.2)
else:
    ax.text(0.5, 0.5, "Semantic results pending", ha='center', va='center', fontsize=14)
    ax.set_title("Semantic Steering")
save_svg(fig_g, "G_semantic_steering")


# H: Persona Method Comparison
fig_h, ax = plt.subplots(figsize=(8, 5))
if judgments_complete:
    trait_names = list(PERSONA_TRAITS.keys())
    cond_names = [c["name"] for c in CONDITIONS]
    means = {}
    for trait_name in trait_names:
        means[trait_name] = {}
        for cond in cond_names:
            rows = [(k, v) for k, v in all_scores.items() if k[0] == trait_name and k[1] == cond]
            trait_values = [v["trait_score"] for _, v in rows]
            means[trait_name][cond] = float(np.mean(trait_values))
    x = np.arange(len(trait_names))
    w = 0.8 / len(cond_names)
    for ci, cond in enumerate(cond_names):
        vals = [means[t][cond] for t in trait_names]
        color = '#9E9E9E' if cond == 'baseline' else cper[ci % len(cper)]
        ax.bar(x + ci * w, vals, w, label=cond, color=color, alpha=0.85, edgecolor='white')
    ax.set_xticks(x + w * len(cond_names) / 2)
    ax.set_xticklabels(trait_names, rotation=30, fontsize=9)
    ax.set_ylabel("Judged target expression (0–100)", fontsize=11)
    ax.set_title("Persona Steering  --  Method Comparison", fontsize=13, fontweight='bold')
    ax.legend(fontsize=6, loc='upper right', ncol=2)
    ax.grid(axis='y', alpha=0.2)
else:
    ax.text(
        0.5,
        0.5,
        "Persona scores unavailable\n(complete judging required)",
        ha="center",
        va="center",
        fontsize=14,
    )
    ax.set_title("Persona Steering  --  Method Comparison")
save_svg(fig_h, "H_persona_steering")


# I: normalized CCR versus next-token prefix-match score within semantic functions
fig_i, ax = plt.subplots(figsize=(6, 5))
for i, name in enumerate(SEMANTIC_FUNCTIONS):
    nccr_val = norm_peak_info[name]["ccr"]
    best_acc = semantic_best_configs[name]["accuracy"]
    ax.scatter(nccr_val, best_acc, color=csem[i], s=150, marker='o', label=name,
              zorder=5, edgecolors='black', linewidth=0.8)
ax.set_xlabel("Normalized CCR (L2-normed activations)", fontsize=11)
ax.set_ylabel("Best next-token prefix-match score", fontsize=11)
ax.set_title("Within-semantic descriptive comparison", fontsize=13, fontweight='bold')
ax.legend(fontsize=8, loc='upper left')
ax.set_ylim(-0.05, 1.05)
ax.axhline(y=0.05, color='gray', ls=':', alpha=0.3)
ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,-3))
ax.grid(alpha=0.2)
save_svg(fig_i, "I_ccr_vs_accuracy")


# Also save combined 3x3 figure
fig_all = plt.figure(figsize=(18, 16))
gs = gridspec.GridSpec(3, 3, hspace=0.4, wspace=0.35)

# Row 1: A, B, C
ax1 = fig_all.add_subplot(gs[0, 0])
for i, name in enumerate(SEMANTIC_FUNCTIONS):
    eigs = spectral_data[name][peak_info[name]["layer"]]["eigenvalues"][:30]
    ax1.plot(range(len(eigs)), eigs, color=csem[i], lw=1.5, label=name)
for i, name in enumerate(PERSONA_TRAITS):
    eigs = spectral_data[name][peak_info[name]["layer"]]["eigenvalues"][:30]
    ax1.plot(range(len(eigs)), eigs, color=cper[i], lw=1.5, ls='--', label=name)
ax1.set_xlabel("Eigenvalue Index"); ax1.set_ylabel("Eigenvalue")
ax1.set_title("A: Eigenvalue Decay"); ax1.legend(fontsize=6)

ax2 = fig_all.add_subplot(gs[0, 1])
ax2.bar(range(len(all_names)), [norm_peak_info[n]["ccr"] for n in all_names], color=all_colors)
ax2.set_xticks(range(len(all_names))); ax2.set_xticklabels(all_names, rotation=45, fontsize=7, ha='right')
ax2.set_ylabel("Normalized CCR"); ax2.set_title("B: Normalized CCR at Peak Layer")
ax2.ticklabel_format(axis='y', style='sci', scilimits=(-3,-3))

ax3 = fig_all.add_subplot(gs[0, 2])
ax3.bar(range(len(all_names)), [peak_info[n]["eff_rank"] for n in all_names], color=all_colors)
ax3.set_xticks(range(len(all_names))); ax3.set_xticklabels(all_names, rotation=45, fontsize=7, ha='right')
ax3.set_ylabel("PR effective rank"); ax3.set_title("C: Participation-ratio rank")

# Row 2: D, E, F
for idx, (cname, col) in enumerate([(profile_concepts[0][0], profile_concepts[0][1]),
                                     (profile_concepts[1][0], profile_concepts[1][1]),
                                     (profile_concepts[2][0], profile_concepts[2][1])]):
    ax = fig_all.add_subplot(gs[1, idx])
    ls = sorted(spectral_data[cname].keys())
    ax.plot(ls, [spectral_data[cname][l]["ccr"] for l in ls], color=col, lw=2)
    ax_r = ax.twinx()
    ax_r.plot(ls, [spectral_data[cname][l]["effective_rank"] for l in ls], color=col, lw=2, ls='--')
    ax.set_xlabel("Layer"); ax.set_ylabel("CCR"); ax_r.set_ylabel("PR rank")
    ax.set_title(f"{'DEF'[idx]}: {cname}")

# Row 3: G, H, I (simplified versions)
ax7 = fig_all.add_subplot(gs[2, 0])
if semantic_results and semantic_best_configs:
    fn_list = list(SEMANTIC_FUNCTIONS.keys())
    x = np.arange(len(fn_list)); w = 0.25
    ax7.bar(x-w, [semantic_results[f]["baseline_mean"] for f in fn_list], w,
            yerr=[semantic_results[f]["baseline_std"] for f in fn_list], capsize=2,
            label='Base', color='gray')
    ax7.bar(x, [semantic_best_configs[f]["accuracy"] for f in fn_list], w,
            yerr=[semantic_results[f]["conceptor_std"] for f in fn_list], capsize=2,
            label='Conceptor', color='#2196F3')
    ax7.bar(x+w, [semantic_results[f].get("additive_mean",0) for f in fn_list], w,
            yerr=[semantic_results[f].get("additive_std",0) for f in fn_list], capsize=2,
            label='Additive', color='#FF9800')
    ax7.set_xticks(x); ax7.set_xticklabels(fn_list, rotation=30, fontsize=7, ha='right')
    ax7.set_ylim(0,1); ax7.legend(fontsize=7)
ax7.set_title("G: Semantic Steering")

ax8 = fig_all.add_subplot(gs[2, 1])
if judgments_complete:
    trait_names_h = list(PERSONA_TRAITS.keys())
    cond_names_h = [c["name"] for c in CONDITIONS]
    means_h = {}
    for tn in trait_names_h:
        means_h[tn] = {}
        for cn in cond_names_h:
            rows_h = [(k, v) for k, v in all_scores.items() if k[0] == tn and k[1] == cn]
            trait_values_h = [v["trait_score"] for _, v in rows_h]
            means_h[tn][cn] = float(np.mean(trait_values_h))
    x_h = np.arange(len(trait_names_h))
    w_h = 0.8 / len(cond_names_h)
    for ci_h, cn in enumerate(cond_names_h):
        vals_h = [means_h[t][cn] for t in trait_names_h]
        col_h = '#9E9E9E' if cn == 'baseline' else cper[ci_h % len(cper)]
        ax8.bar(x_h + ci_h * w_h, vals_h, w_h, color=col_h, alpha=0.85, edgecolor='white')
    ax8.set_xticks(x_h + w_h * len(cond_names_h) / 2)
    ax8.set_xticklabels(trait_names_h, rotation=30, fontsize=7, ha='right')
    ax8.set_ylabel("Judged target expression (0–100)", fontsize=8)
else:
    ax8.text(
        0.5,
        0.5,
        "Complete judgments required",
        ha="center",
        va="center",
        transform=ax8.transAxes,
    )
ax8.set_title("H: Persona Steering")

ax9 = fig_all.add_subplot(gs[2, 2])
for i, name in enumerate(SEMANTIC_FUNCTIONS):
    acc = semantic_best_configs[name]["accuracy"]
    ax9.scatter(norm_peak_info[name]["ccr"], acc, color=csem[i], s=100, marker='o', label=name, edgecolors='black', lw=0.5)
ax9.set_xlabel("Normalized CCR"); ax9.set_ylabel("Best prefix-match score")
ax9.set_title("I: Semantic nCCR vs prefix match"); ax9.legend(fontsize=6); ax9.set_ylim(-0.05,1.05)
ax9.ticklabel_format(axis='x', style='sci', scilimits=(-3,-3))

fig_all.suptitle(f"Spectral Geometry & Conceptor Steering (broad study): {MODEL_SHORT}", fontsize=14, fontweight='bold')
fig_all.savefig(f"{fig_dir}/combined_3x3.svg", format='svg', bbox_inches='tight')
print(f"\nAll figures saved to {fig_dir}/")


# --- Panel J: Normalized CCR vs Conceptor Prefix-Match Score ---
fig_j, ax_j = plt.subplots(1, 1, figsize=(8, 6))
ax_j.set_title("J. Semantic normalized CCR vs prefix match (broad study)", fontsize=14, fontweight="bold")

nccr_plot = []
acc_plot = []
names_plot = []
colors_plot = []

for name in SEMANTIC_FUNCTIONS:
    if name in semantic_best_configs and name in norm_peak_info:
        nccr_plot.append(norm_peak_info[name]["ccr"])
        acc_plot.append(semantic_best_configs[name]["accuracy"])
        names_plot.append(name)
        colors_plot.append("#2196F3")

if nccr_plot:
    ax_j.scatter(nccr_plot, acc_plot, c=colors_plot, s=100, edgecolors="white", linewidth=1.5, zorder=5)
    for i, name in enumerate(names_plot):
        ax_j.annotate(name, (nccr_plot[i], acc_plot[i]), fontsize=7, ha="center", va="bottom",
                      xytext=(0, 8), textcoords="offset points")
    ax_j.text(0.05, 0.95, "Semantic: next-token prefix-match score\nPersona: judged target expression (separate panel)",
              transform=ax_j.transAxes, fontsize=11, verticalalignment="top",
              bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

ax_j.set_xlabel("Normalized CCR (L2-normed activations)", fontsize=12)
ax_j.set_ylabel("Best conceptor prefix-match score (fraction)", fontsize=12)
ax_j.legend(handles=[
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2196F3", markersize=10, label="Semantic functions"),
], loc="upper left", fontsize=10)
ax_j.ticklabel_format(axis='x', style='sci', scilimits=(-3,-3))
ax_j.grid(True, alpha=0.3)
fig_j.tight_layout()
fig_j.savefig(f"{fig_dir}/J_ccr_vs_accuracy.svg", format="svg", bbox_inches="tight")
print(f"Saved: {fig_dir}/J_ccr_vs_accuracy.svg")
plt.close(fig_j)


# Broad-study summary
print("\n" + "=" * 70)
print("Broad-study summary")
print("=" * 70)
print("1. Category separation (normalized CCR):")
print(f"   Semantic nCCR = {sem_nccr:.2e}, Persona nCCR = {per_nccr:.2e}")
print(f"   Ratio = {sem_nccr/per_nccr:.2f}x, Mann-Whitney p = {pval:.4f}")

print(f"\n2. Best conceptor steering per semantic function ({N_SEEDS}-seed average):")
for fn, cfg in semantic_best_configs.items():
    sr = semantic_results[fn]
    delta = cfg['accuracy'] - sr['baseline_mean']
    print(f"   {fn:20s}: {cfg['accuracy']:.1%}±{sr['conceptor_std']:.1%} at L{cfg['layer']}, b={cfg['beta']}, a={cfg['aperture']} (delta={delta:+.1%})")
print("\n3. Outcome scales and scope")
print("   Semantic results use next-token prefix-match scores; persona results use")
print("   judged target-expression and coherence scores. Associations describe the")
print("   evaluated concepts; predictive performance on unseen concepts was not evaluated.")


print("SEMANTIC STEERING DIAGNOSTIC\n")

for fn in ["singular_plural", "present_past"]:
    li = THIRD_LAYER
    test_prompts = get_test_prompts(SEMANTIC_FUNCTIONS[fn]["pairs"], 5)
    prompts_list = [p for p, _ in test_prompts]
    expected_list = [e for _, e in test_prompts]

    print(f"--- {fn} at L{li} ---")

    steerable.clear_hooks()
    preds_base = steerable.top_1_next_token(prompts_list)

    sv = semantic_steering_vecs[fn][li]
    steerable.clear_hooks()
    steerable.add_hook(li, AdditiveHook(sv, coef=2.3, mode="add", last_token_only=True))
    preds_add = steerable.top_1_next_token(prompts_list)

    C = postmus_conceptors[fn][(li, APERTURE)]
    steerable.clear_hooks()
    steerable.add_hook(li, PostmusConceptorHook(C, mean=None, beta=2.0, mode="enhance",
                                                last_token_only=True))
    preds_con = steerable.top_1_next_token(prompts_list)

    for p, exp, pb, pa, pc in zip(prompts_list, expected_list, preds_base, preds_add, preds_con):
        m_b = "Y" if pb.strip().lower().startswith(exp.lower()[:3]) else "N"
        m_a = "Y" if pa.strip().lower().startswith(exp.lower()[:3]) else "N"
        m_c = "Y" if pc.strip().lower().startswith(exp.lower()[:3]) else "N"
        print(f"  {p:20s} exp='{exp:12s}' base='{pb:12s}'{m_b}  add='{pa:12s}'{m_a}  con='{pc:12s}'{m_c}")

    changed_add = sum(1 for a, b in zip(preds_base, preds_add) if a != b)
    changed_con = sum(1 for a, b in zip(preds_base, preds_con) if a != b)
    print(f"  Changed: additive={changed_add}/5, conceptor={changed_con}/5")
    print(f"  |steering_vec|={sv.norm().item():.1f}\n")

steerable.clear_hooks()

# Final within-semantic diagnostic using normalized CCR
import numpy as np
all_names_d, all_nccr_d, all_acc_d = [], [], []
for name in SEMANTIC_FUNCTIONS:
    if name in semantic_best_configs:
        all_names_d.append(name)
        all_nccr_d.append(norm_peak_info[name]["ccr"])
        all_acc_d.append(semantic_best_configs[name]["accuracy"])

nccr_arr_d, acc_arr_d = np.array(all_nccr_d), np.array(all_acc_d)
r = np.corrcoef(nccr_arr_d, acc_arr_d)[0,1]
print(f"Within-semantic correlation (n={len(all_nccr_d)}): r={r:.4f}, R²={r**2:.4f}")
print("\nPer-concept:")
for n, c, a in sorted(zip(all_names_d, all_nccr_d, all_acc_d), key=lambda x: -x[1]):
    print(f"  {n:20s} nCCR={c:.2e} PrefixMatch={a:.1%}")

plt.show()
