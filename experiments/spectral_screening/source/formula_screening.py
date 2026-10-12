"""
Formula Screening: Comprehensive spectral measure evaluation
============================================================
Tests 48 concentration/rank/separation measures from multiple fields
(physics, information theory, economics, signal processing, RMT, statistics)
from either historical summaries (13 measures, three models) or a fresh
checkpoint (48 measures, one model). These inputs are never combined.

For each formula:
  - two-sided Mann-Whitney U (semantic vs persona separation)
  - Hedges' g effect size
  - Spearman correlation with SCR (redundancy check)
  - Spearman correlation with the Function-task conceptor next-token prefix
    score (n=6; descriptive only, not full-answer accuracy)
  - Cross-model consistency (JSON formulas only)

The larger observed family determines only the descriptive orientation of U.
All p-values are two-sided and descriptive. The bonferroni_pass field compares
each p-value with .05/48. A no-tie complete 6-versus-7 ordering gives the
exact p = 2/C(13,6) ~ .0011655 > .05/48; tied samples may use other test behavior.

The output must be a new file outside the repository; arguments are validated
before any input is read.
"""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

# ---- Config ----
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[2]
parser = argparse.ArgumentParser(
    description="Screen fresh activation spectra or reanalyse historical summary measures separately."
)
parser.add_argument("--mode", choices=("checkpoint", "historical-summaries"), default="checkpoint")
parser.add_argument(
    "--consolidated-json",
    type=Path,
    help="Historical consolidated summaries; allowed only in historical-summaries mode.",
)
parser.add_argument(
    "--extractions",
    type=Path,
    help="User-provided broad-study activation-spectrum checkpoint.",
)
parser.add_argument(
    "--extractions-sha256",
    help="Expected SHA-256 of your extraction checkpoint.",
)
parser.add_argument(
    "--run-manifest",
    type=Path,
    help="run_manifest.json that owns your broad-study extraction checkpoint.",
)
parser.add_argument(
    "--output",
    type=Path,
    required=True,
    help=(
        "New JSON file outside the repository for the corrected two-sided output. "
        "Paths inside the repository (resolved through symlinks) and existing "
        "files are refused; the frozen results/ directory is never written."
    ),
)
args = parser.parse_args()
if args.extractions_sha256 is not None and re.fullmatch(r"[0-9a-fA-F]{64}", args.extractions_sha256) is None:
    parser.error("--extractions-sha256 must contain exactly 64 hexadecimal characters")


def is_inside_repository(resolved: Path) -> bool:
    """True if a symlink-resolved path is the repository or below it (case/alias safe)."""
    repository = REPO_ROOT.resolve()
    if resolved == repository or repository in resolved.parents:
        return True
    repository_stat = repository.stat()
    for candidate in (resolved, *resolved.parents):
        try:
            if os.path.samestat(candidate.stat(), repository_stat):
                return True
        except OSError:
            continue
    return False


def refuse_unsafe_output(output: Path) -> Path:
    """Return the resolved output path, refusing repository paths and existing files."""
    output = output.expanduser()
    resolved = output.resolve()
    if is_inside_repository(resolved):
        parser.error(
            f"refusing to write --output inside the repository ({resolved}); the release "
            "results must remain unchanged. Choose a path outside the repository."
        )
    if output.is_symlink() or resolved.is_symlink() or os.path.lexists(resolved):
        parser.error(f"refusing to overwrite existing --output {resolved}")
    existing_parent = next(parent for parent in resolved.parents if os.path.lexists(parent))
    if not existing_parent.is_dir():
        parser.error(f"--output parent {existing_parent} is not a directory")
    return resolved


def open_new_output(resolved: Path):
    """Create the output exclusively, re-checking the location at write time."""
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved.parent.resolve() != resolved.parent or is_inside_repository(resolved.parent):
        raise RuntimeError(f"--output parent changed after validation: {resolved.parent}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    return os.fdopen(os.open(resolved, flags, 0o644), "w", encoding="utf-8")


OUTPUT_PATH = refuse_unsafe_output(args.output)
if args.mode == "checkpoint":
    if args.consolidated_json is not None:
        parser.error("checkpoint mode cannot combine spectra with --consolidated-json")
    if any(value is None for value in (args.extractions, args.extractions_sha256, args.run_manifest)):
        parser.error("checkpoint mode requires --extractions, --extractions-sha256 and --run-manifest")
    input_paths = (("--extractions", args.extractions), ("--run-manifest", args.run_manifest))
else:
    if any(value is not None for value in (args.extractions, args.extractions_sha256, args.run_manifest)):
        parser.error("historical-summaries mode does not accept checkpoint inputs")
    args.consolidated_json = args.consolidated_json or REPO_ROOT / "provenance" / "broad_study_results.json"
    input_paths = (("--consolidated-json", args.consolidated_json),)
for label, path in input_paths:
    if not path.is_file():
        parser.error(f"{label} input not found: {path}")

# Heavy imports follow argument validation so that unsafe arguments fail fast.
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy import stats  # noqa: E402
from spectral_metrics import (  # noqa: E402
    conceptor_eigenvalues,
    evaluate_finite_formula,
    hedges_g,
    is_complete_ordering,
    mann_whitney_auto_method,
    oriented_two_sided_mann_whitney,
    spectral_variance_edge_fraction,
    spectrum_summary,
    validate_run_binding,
)

JSON_PATH = args.consolidated_json
PT_PATH = args.extractions
MANIFEST_PATH = args.run_manifest
APERTURE = 0.05

SEMANTIC = ['antonym', 'country_capital', 'present_past', 'english_french', 'singular_plural', 'capitalize']
PERSONA = ['sycophancy', 'hallucination', 'evil', 'impolite', 'optimistic', 'apathetic', 'humorous']
ALL_CONCEPTS = SEMANTIC + PERSONA
N_CONCEPTS = len(ALL_CONCEPTS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

# ---- Load JSON ----
input_hashes = {}
if args.mode == "historical-summaries":
    print("Loading historical summary data...")
    input_hashes['consolidated_json'] = sha256_file(JSON_PATH)
    with JSON_PATH.open() as f:
        jdata = json.load(f)
    MODELS = list(jdata.keys())
else:
    jdata = {}
    MODELS = ['Qwen2.5-1.5B']
print(f"  Models: {MODELS}")

def get_theory(model, concept):
    if args.mode == "checkpoint":
        return spectrum_summary(eig_spectra[concept])
    return jdata[model]['results']['theory'][concept]

def get_steering_outcome(model, concept):
    """Function task: conceptor next-token prefix score; assistant target: max |trait change|/100."""
    r = jdata[model]['results']
    if concept in SEMANTIC:
        return r['semantic_results'][concept]['conceptor_mean']
    else:
        pt = r['persona_table'][concept]
        bl_trait = pt.get('baseline', {}).get('trait', 0)
        best_delta = 0
        for k, v in pt.items():
            if k == 'baseline' or not isinstance(v, dict):
                continue
            t = v.get('trait', 0)
            if isinstance(t, (int, float)):
                delta = abs(t - bl_trait)
                if delta > best_delta:
                    best_delta = delta
        return best_delta / 100.0

# ---- Load .pt and compute eigenvalue spectra ----
peak_layers_pt = {}
eig_spectra = {}
fit_sample_counts = {}
run_specification_sha256 = None
if args.mode == "checkpoint":
    print("Loading activation checkpoint...")
    PT_SHA256 = sha256_file(PT_PATH)
    if PT_SHA256 != args.extractions_sha256.lower():
        raise RuntimeError("Activation checkpoint hash mismatch")
    input_hashes.update(extractions=PT_SHA256, run_manifest=sha256_file(MANIFEST_PATH))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    current_materials_sha256 = sha256_file(REPO_ROOT / "experimental_materials.json")
    ckpt = torch.load(PT_PATH, map_location="cpu", weights_only=True)
    run_specification_sha256 = validate_run_binding(
        manifest,
        ckpt.get("run_specification_sha256"),
        required_model="Qwen/Qwen2.5-1.5B-Instruct",
        current_materials_sha256=current_materials_sha256,
    )
    sem_ext, per_ext = ckpt["semantic"], ckpt["persona"]
    if set(sem_ext) != set(SEMANTIC) or set(per_ext) != set(PERSONA):
        raise RuntimeError("Activation checkpoint target inventory does not match the fixed 6+7 screen.")
    analysis_layers = sorted(sem_ext[SEMANTIC[0]])
    if not analysis_layers:
        raise RuntimeError("Activation checkpoint contains no analysis layers")
    for name in ALL_CONCEPTS:
        layers = per_ext[name]["pos"] if name in PERSONA else sem_ext[name]
        if set(layers) != set(analysis_layers):
            raise RuntimeError(f"Inconsistent analysis layers for {name}")
        best_energy = -1
        for li in analysis_layers:
            acts = layers[li]
            eigs = conceptor_eigenvalues(acts, APERTURE)
            top10 = eigs[:10].sum()
            if top10 > best_energy:
                best_energy = top10
                peak_layers_pt[name] = li
                eig_spectra[name] = eigs
                fit_sample_counts[name] = acts.shape[0]
        print(f"  {name:20s} L{peak_layers_pt[name]} n={fit_sample_counts[name]}")
    del ckpt, sem_ext, per_ext, layers, acts

# =====================================================================
# FORMULA DEFINITIONS
# =====================================================================
# Each formula: (name, source_field, requires_pt, function, description)
# function signature: f(eigs, ccr, spr, scr, effrank) -> float
# For PT formulas: eigs is the full spectrum (numpy array)
# For JSON formulas: eigs may be None for non-1.5B models

def safe_log(x):
    return np.log(x) if x > 1e-30 else -70.0

formulas = []

# ---- GROUP A: JSON-derivable (functions of summary stats) ----

formulas.append(("CCR", "conceptor", False,
    lambda e, ccr, spr, scr, er: ccr,
    "Mean eigenvalue (Jaeger quota)"))

formulas.append(("SPR", "conceptor", False,
    lambda e, ccr, spr, scr, er: spr,
    "Mean squared eigenvalue"))

formulas.append(("SCR", "conceptor", False,
    lambda e, ccr, spr, scr, er: scr,
    "Spectral concentration ratio SPR/CCR"))

formulas.append(("EffRank", "conceptor", False,
    lambda e, ccr, spr, scr, er: er,
    "Participation-ratio effective rank"))

formulas.append(("log_SCR", "transform", False,
    lambda e, ccr, spr, scr, er: safe_log(scr),
    "Log of SCR"))

formulas.append(("Leakage", "conceptor", False,
    lambda e, ccr, spr, scr, er: 1.0 - scr,
    "1 - SCR"))

formulas.append(("PR_analogue", "physics", False,
    lambda e, ccr, spr, scr, er: ccr / scr if scr > 1e-12 else 0,
    "CCR/SCR = CCR^2/SPR (participation ratio analogue)"))

formulas.append(("CCR_x_EffRank", "derived", False,
    lambda e, ccr, spr, scr, er: ccr * er,
    "Product CCR * participation-ratio effective rank"))

formulas.append(("SCR_x_EffRank", "derived", False,
    lambda e, ccr, spr, scr, er: scr * er,
    "Product SCR * participation-ratio effective rank"))

formulas.append(("inv_EffRank", "derived", False,
    lambda e, ccr, spr, scr, er: 1.0 / er if er > 0 else 0,
    "Inverse participation-ratio effective rank"))

formulas.append(("SCR_over_EffRank", "derived", False,
    lambda e, ccr, spr, scr, er: scr / er if er > 0 else 0,
    "SCR / participation-ratio effective rank"))

formulas.append(("log_CCR", "transform", False,
    lambda e, ccr, spr, scr, er: safe_log(ccr),
    "Log of CCR"))

formulas.append(("CCR_sq_over_SPR", "physics", False,
    lambda e, ccr, spr, scr, er: (ccr**2) / spr if spr > 1e-15 else 0,
    "CCR^2/SPR = 1/IPR_analogue"))

# ---- GROUP B: Full-spectrum formulas (require .pt, 1.5B only) ----

def _shannon_entropy(eigs):
    """Shannon entropy of normalized eigenvalue distribution."""
    eigs_pos = eigs[eigs > 1e-15]
    if len(eigs_pos) == 0:
        return 0.0
    p = eigs_pos / eigs_pos.sum()
    return -np.sum(p * np.log(p))

def _renyi_entropy(eigs, q):
    """Renyi entropy of order q."""
    eigs_pos = eigs[eigs > 1e-15]
    if len(eigs_pos) == 0:
        return 0.0
    p = eigs_pos / eigs_pos.sum()
    if abs(q - 1.0) < 1e-6:
        return _shannon_entropy(eigs)
    return np.log(np.sum(p**q)) / (1.0 - q)

def _gini_coefficient(eigs):
    """Gini coefficient of eigenvalue distribution."""
    eigs_sorted = np.sort(eigs[eigs > 1e-15])
    n = len(eigs_sorted)
    if n <= 1 or eigs_sorted.sum() < 1e-15:
        return 0.0
    index = np.arange(1, n + 1)
    return (2.0 * np.sum(index * eigs_sorted) / (n * eigs_sorted.sum())) - (n + 1) / n

def _theil_index(eigs):
    """Theil index (generalized entropy GE(1))."""
    eigs_pos = eigs[eigs > 1e-15]
    n = len(eigs_pos)
    if n <= 1:
        return 0.0
    mu = eigs_pos.mean()
    if mu < 1e-15:
        return 0.0
    return np.mean((eigs_pos / mu) * np.log(eigs_pos / mu))

def _atkinson_index(eigs, epsilon=1.0):
    """Atkinson inequality index."""
    eigs_pos = eigs[eigs > 1e-15]
    if len(eigs_pos) <= 1:
        return 0.0
    mu = eigs_pos.mean()
    if mu < 1e-15:
        return 0.0
    if abs(epsilon - 1.0) < 1e-6:
        geo_mean = np.exp(np.mean(np.log(eigs_pos)))
        return 1.0 - geo_mean / mu
    else:
        return 1.0 - (np.mean(eigs_pos**(1.0 - epsilon)))**(1.0 / (1.0 - epsilon)) / mu

def _spectral_flatness(eigs):
    """Spectral flatness = geometric_mean / arithmetic_mean."""
    eigs_pos = eigs[eigs > 1e-15]
    if len(eigs_pos) == 0:
        return 0.0
    geo = np.exp(np.mean(np.log(eigs_pos)))
    arith = eigs_pos.mean()
    return geo / arith if arith > 1e-15 else 0.0

def _power_law_exponent(eigs):
    """Fit power law mu_i ~ i^(-alpha) to top eigenvalues."""
    eigs_pos = eigs[eigs > 1e-6]
    n = min(len(eigs_pos), 50)
    if n < 5:
        return 0.0
    x = np.log(np.arange(1, n + 1))
    y = np.log(eigs_pos[:n])
    slope, _, _, _, _ = stats.linregress(x, y)
    return -slope

def _exp_decay_rate(eigs):
    """Fit exponential decay mu_i ~ exp(-rate*i)."""
    eigs_pos = eigs[eigs > 1e-6]
    n = min(len(eigs_pos), 50)
    if n < 5:
        return 0.0
    x = np.arange(n)
    y = np.log(eigs_pos[:n])
    slope, _, _, _, _ = stats.linregress(x, y)
    return -slope

# Register full-spectrum formulas
formulas.append(("Shannon_H", "info_theory", True,
    lambda e, *_: _shannon_entropy(e),
    "Shannon entropy of normalized eigenvalues"))

formulas.append(("Renyi_0.5", "info_theory", True,
    lambda e, *_: _renyi_entropy(e, 0.5),
    "Renyi entropy order 0.5"))

formulas.append(("Renyi_2", "info_theory", True,
    lambda e, *_: _renyi_entropy(e, 2.0),
    "Renyi entropy order 2 (collision entropy)"))

formulas.append(("Renyi_inf", "info_theory", True,
    lambda e, *_: -safe_log(e[0] / e[e > 1e-15].sum()) if e[e > 1e-15].sum() > 0 else 0,
    "Min-entropy = -log(max p_i)"))

formulas.append(("KL_from_uniform", "info_theory", True,
    lambda e, *_: (lambda p: np.sum(p * np.log(p * len(p))))(
        (lambda x: x/x.sum())(e[e > 1e-15])) if e[e > 1e-15].sum() > 0 else 0,
    "KL divergence from uniform distribution"))

formulas.append(("Gini", "economics", True,
    lambda e, *_: _gini_coefficient(e),
    "Gini coefficient of eigenvalue inequality"))

formulas.append(("Theil", "economics", True,
    lambda e, *_: _theil_index(e),
    "Theil index (GE(1))"))

formulas.append(("Atkinson_05", "economics", True,
    lambda e, *_: _atkinson_index(e, 0.5),
    "Atkinson index epsilon=0.5"))

formulas.append(("Atkinson_1", "economics", True,
    lambda e, *_: _atkinson_index(e, 1.0),
    "Atkinson index epsilon=1 (log-sensitivity)"))

formulas.append(("Atkinson_2", "economics", True,
    lambda e, *_: _atkinson_index(e, 2.0),
    "Atkinson index epsilon=2 (tail-sensitive)"))

formulas.append(("HHI", "economics", True,
    lambda e, *_: (lambda p: np.sum(p**2))((lambda x: x/x.sum())(e[e > 1e-15])) if e[e > 1e-15].sum() > 0 else 0,
    "Herfindahl-Hirschman index (= quantum purity)"))

formulas.append(("Spectral_Flatness", "signal_proc", True,
    lambda e, *_: _spectral_flatness(e),
    "Geometric/arithmetic mean ratio"))

formulas.append(("Spectral_Crest", "signal_proc", True,
    lambda e, *_: e[0] / e.mean() if e.mean() > 1e-15 else 0,
    "Max eigenvalue / mean eigenvalue"))

formulas.append(("Top1_Energy", "signal_proc", True,
    lambda e, *_: e[0] / e.sum() if e.sum() > 1e-15 else 0,
    "Fraction of energy in top eigenvalue"))

formulas.append(("Top5_Energy", "signal_proc", True,
    lambda e, *_: e[:5].sum() / e.sum() if e.sum() > 1e-15 else 0,
    "Fraction of energy in top 5 eigenvalues"))

formulas.append(("Top10_Energy", "signal_proc", True,
    lambda e, *_: e[:10].sum() / e.sum() if e.sum() > 1e-15 else 0,
    "Fraction of energy in top 10 eigenvalues"))

formulas.append(("Top20_Energy", "signal_proc", True,
    lambda e, *_: e[:20].sum() / e.sum() if e.sum() > 1e-15 else 0,
    "Fraction of energy in top 20 eigenvalues"))

formulas.append(("Spectral_Gap", "network_sci", True,
    lambda e, *_: float(e[0] - e[1]) if len(e) > 1 else 0,
    "Gap between 1st and 2nd eigenvalue"))

formulas.append(("Gap_Ratio", "network_sci", True,
    lambda e, *_: float((e[0] - e[1]) / e[0]) if e[0] > 1e-15 else 0,
    "Relative gap (mu1-mu2)/mu1"))

formulas.append(("Stable_Rank", "optimization", True,
    lambda e, *_: e.sum() / e[0] if e[0] > 1e-15 else 0,
    "Trace-to-peak effective rank (nuclear/operator norm, not matrix stable rank)"))

formulas.append(("Condition_Number", "statistics", True,
    lambda e, *_: (lambda ep: ep[0]/ep[-1] if ep[-1] > 1e-15 else ep[0]/1e-15)(e[e > 1e-10]),
    "Max/min eigenvalue ratio (positive only)"))

formulas.append(("CV_eigenvalues", "statistics", True,
    lambda e, *_: (lambda ep: ep.std()/ep.mean() if ep.mean() > 1e-15 else 0)(e[e > 1e-15]),
    "Coefficient of variation of positive eigenvalues"))

formulas.append(("Skewness", "statistics", True,
    lambda e, *_: float(stats.skew(e[e > 1e-15])) if np.sum(e > 1e-15) > 3 else 0,
    "Skewness of eigenvalue distribution"))

formulas.append(("Kurtosis", "statistics", True,
    lambda e, *_: float(stats.kurtosis(e[e > 1e-15])) if np.sum(e > 1e-15) > 4 else 0,
    "Excess kurtosis of eigenvalue distribution"))

formulas.append(("MP_Fraction", "RMT", True,
    lambda e, *arguments: spectral_variance_edge_fraction(e, arguments[-1]),
    "Fraction above spectral-variance heuristic var(c_positive)*(1+sqrt(n/d))^2; not a validated noise edge"))

formulas.append(("N_above_0.5", "threshold", True,
    lambda e, *_: np.sum(e > 0.5),
    "Number of eigenvalues > 0.5"))

formulas.append(("N_above_0.1", "threshold", True,
    lambda e, *_: np.sum(e > 0.1),
    "Number of eigenvalues > 0.1"))

formulas.append(("N_above_0.01", "threshold", True,
    lambda e, *_: np.sum(e > 0.01),
    "Number of eigenvalues > 0.01"))

formulas.append(("Pct90_over_median", "statistics", True,
    lambda e, *_: (lambda ep: np.percentile(ep, 90)/np.median(ep) if np.median(ep) > 1e-15 else 0)(e[e > 1e-15]),
    "90th percentile / median of positive eigenvalues"))

formulas.append(("IQR", "statistics", True,
    lambda e, *_: (lambda ep: np.percentile(ep, 75) - np.percentile(ep, 25))(e[e > 1e-15]) if np.sum(e > 1e-15) > 4 else 0,
    "Interquartile range of positive eigenvalues"))

formulas.append(("Power_Law_Exp", "RMT", True,
    lambda e, *_: _power_law_exponent(e),
    "Power law decay exponent"))

formulas.append(("Exp_Decay_Rate", "signal_proc", True,
    lambda e, *_: _exp_decay_rate(e),
    "Exponential decay rate"))

formulas.append(("Tail_Weight", "statistics", True,
    lambda e, *_: (lambda ep: ep[len(ep)//2:].sum()/ep.sum() if ep.sum() > 1e-15 else 0)(e[e > 1e-15]),
    "Fraction of energy in bottom half of positive eigenvalues"))

formulas.append(("Entropy_Deficit", "info_theory", True,
    lambda e, *_: (lambda ep: np.log(len(ep)) - _shannon_entropy(e))(e[e > 1e-15]) if np.sum(e > 1e-15) > 0 else 0,
    "log(N) - H: how far from uniform"))

formulas.append(("Nuclear_Schatten_Ratio", "optimization", True,
    lambda e, *_: e.sum() / (np.sum(e**2)**0.5) if np.sum(e**2) > 1e-15 else 0,
    "Schatten-1 / Schatten-2 norm ratio"))

N_FORMULAS = len(formulas)
if args.mode == "historical-summaries":
    formulas = [formula for formula in formulas if not formula[2]]
print(f"\nTotal formulas registered: {N_FORMULAS}")
print(f"  Measures available in {args.mode} mode: {len(formulas)}")
print(f"  JSON-derivable: {sum(1 for f in formulas if not f[2])}")
print(f"  PT-only: {sum(1 for f in formulas if f[2])}")
print(f"  Bonferroni alpha: {0.05/N_FORMULAS:.6f}")

# =====================================================================
# COMPUTE ALL FORMULA VALUES
# =====================================================================
print(f"\n{'='*80}")
print("COMPUTING FORMULA VALUES")
print(f"{'='*80}")

# Summary formulas use either the historical tables or the new checkpoint's moments.
# Full-spectrum formulas are available only in checkpoint mode.
results = {}

for fname, field, needs_pt, func, desc in formulas:
    results[fname] = {
        'field': field,
        'needs_pt': needs_pt,
        'desc': desc,
        'values': {},       # {model: {concept: value}}
        'sem_vals': {},     # {model: [values for semantic]}
        'per_vals': {},     # {model: [values for persona]}
    }

    models_to_test = ['Qwen2.5-1.5B'] if needs_pt else MODELS

    for model in models_to_test:
        results[fname]['values'][model] = {}
        results[fname]['sem_vals'][model] = []
        results[fname]['per_vals'][model] = []

        for concept in ALL_CONCEPTS:
            theory = get_theory(model, concept)
            ccr = theory['rawCCR']
            spr = theory['SPR']
            scr = theory['SCR']
            er = theory['EffRank']

            eigs = eig_spectra.get(concept) if (needs_pt and model == 'Qwen2.5-1.5B') else None
            if needs_pt and eigs is None:
                continue

            arguments = (eigs, ccr, spr, scr, er)
            if fname == 'MP_Fraction':
                arguments += (fit_sample_counts[concept],)
            val = evaluate_finite_formula(fname, model, concept, func, *arguments)

            results[fname]['values'][model][concept] = val
            if concept in SEMANTIC:
                results[fname]['sem_vals'][model].append(val)
            else:
                results[fname]['per_vals'][model].append(val)

# =====================================================================
# STATISTICAL TESTS
# =====================================================================
print(f"\n{'='*80}")
print("STATISTICAL ANALYSIS")
print(f"{'='*80}")

def format_effect(value):
    return f"{value:+.2f}" if value is not None else "N/A"

# Get SCR values for redundancy check
scr_vals_15b = [results['SCR']['values']['Qwen2.5-1.5B'][c] for c in ALL_CONCEPTS]

# Function-task next-token prefix scores for the descriptive correlation
steering_sem = {}
if args.mode == "historical-summaries":
    for model in MODELS:
        steering_sem[model] = [get_steering_outcome(model, c) for c in SEMANTIC]

all_results = []

header = f"{'Formula':25s} {'Field':12s} {'PT?':4s} | {'U':>4s} {'p':>10s} {'g':>7s} {'Bonf':>5s} | {'rho_SCR':>8s} {'rho_steer':>10s} | {'XM_rho':>7s} {'Sep':>4s}"
print(f"\n{header}")
print("-" * len(header))

for fname, field, needs_pt, func, desc in formulas:
    r = results[fname]
    model_15 = 'Qwen2.5-1.5B'

    # --- Separation test (1.5B) ---
    sv = r['sem_vals'].get(model_15, [])
    pv = r['per_vals'].get(model_15, [])

    if len(sv) < 2 or len(pv) < 2:
        continue

    sv_arr = np.array(sv, dtype=float)
    pv_arr = np.array(pv, dtype=float)

    u_stat, u_p, direction = oriented_two_sided_mann_whitney(sv_arr, pv_arr)

    g_val, g_status = hedges_g(sv_arr, pv_arr)
    bonf_pass = "YES" if u_p < 0.05 / N_FORMULAS else "NO"

    # --- Redundancy check: Spearman with SCR ---
    formula_vals_15b = [r['values'][model_15].get(c, float('nan')) for c in ALL_CONCEPTS]
    if not np.isfinite(formula_vals_15b).all():
        raise ValueError(f"Non-finite redundancy inputs for {fname}")
    rho_scr_raw, _ = stats.spearmanr(formula_vals_15b, scr_vals_15b)
    rho_scr = float(rho_scr_raw) if np.isfinite(rho_scr_raw) else None
    redundant = None if rho_scr is None else abs(rho_scr) > 0.95

    # --- Steering correlation (within semantic, 1.5B) ---
    formula_sem_15b = r['sem_vals'].get(model_15, [])
    steer_15b = steering_sem.get(model_15, [])
    if len(formula_sem_15b) == 6 and len(steer_15b) == 6:
        rho_steer, p_steer = stats.spearmanr(formula_sem_15b, steer_15b)
    else:
        rho_steer, p_steer = float('nan'), float('nan')

    # --- Cross-model consistency ---
    if not needs_pt and len(r['sem_vals']) >= 3:
        cross_rhos = []
        model_pairs = [(MODELS[i], MODELS[j]) for i in range(len(MODELS)) for j in range(i+1, len(MODELS))]
        for m1, m2 in model_pairs:
            vals1 = [r['values'][m1].get(c, float('nan')) for c in ALL_CONCEPTS]
            vals2 = [r['values'][m2].get(c, float('nan')) for c in ALL_CONCEPTS]
            if not any(np.isnan(v) for v in vals1 + vals2):
                rho_xm, _ = stats.spearmanr(vals1, vals2)
                cross_rhos.append(rho_xm)
        xm_rho = np.mean(cross_rhos) if cross_rhos else float('nan')

        # Check U=42 on all models
        all_u42 = True
        for model in MODELS:
            sv_m = np.array(r['sem_vals'].get(model, []), dtype=float)
            pv_m = np.array(r['per_vals'].get(model, []), dtype=float)
            if len(sv_m) < 2 or len(pv_m) < 2:
                all_u42 = False
                break
            if not is_complete_ordering(sv_m, pv_m):
                all_u42 = False
                break
        sep_str = f"{len(MODELS)}/{len(MODELS)}" if all_u42 else f"< {len(MODELS)}"
    else:
        xm_rho = float('nan')
        all_u42 = is_complete_ordering(sv_arr, pv_arr)
        sep_str = "1/1" if u_stat == 42 else "< 1"

    # Print row
    pt_str = "PT" if args.mode == "checkpoint" else "JSON"
    bonf_mark = bonf_pass
    rho_scr_str = f"{rho_scr:+.3f}" if rho_scr is not None else "  N/A"
    rho_steer_str = f"{rho_steer:+.3f}" if not np.isnan(rho_steer) else "  N/A"
    xm_str = f"{xm_rho:.3f}" if not np.isnan(xm_rho) else "  N/A"

    row = f"{fname:25s} {field:12s} {pt_str:4s} | {u_stat:4.0f} {u_p:10.6f} {format_effect(g_val):>7s} {bonf_mark:>5s} | {rho_scr_str:>8s} {rho_steer_str:>10s} | {xm_str:>7s} {sep_str:>4s}"
    print(row)

    all_results.append({
        'name': fname,
        'field': field,
        'needs_pt': needs_pt,
        'desc': desc,
        'direction': direction,
        'U': float(u_stat),
        'p': float(u_p),
        'p_method': 'two-sided',
        'mann_whitney_method': mann_whitney_auto_method(sv_arr, pv_arr),
        'hedges_g': g_val,
        'hedges_g_status': g_status,
        'bonferroni_pass': bonf_pass == "YES",
        'rho_scr': rho_scr,
        'redundant_with_scr': redundant,
        'rho_steering': float(rho_steer) if not np.isnan(rho_steer) else None,
        'p_steering': float(p_steer) if not np.isnan(p_steer) else None,
        'cross_model_rho': float(xm_rho) if not np.isnan(xm_rho) else None,
        'all_models_u42': bool(all_u42) if len(MODELS) >= 3 and not needs_pt else None,
        'sep_str': sep_str,
        'values_1.5B': {c: float(r['values'][model_15].get(c, float('nan'))) for c in ALL_CONCEPTS},
    })

# =====================================================================
# ANALYSIS SUMMARY
# =====================================================================
print(f"\n{'='*80}")
print("SUMMARY ANALYSIS")
print(f"{'='*80}")

# 1. Which formulas achieve U=42 on 1.5B?
u42_formulas = [r for r in all_results if r['U'] == 42]
print(f"\n1. Formulas achieving U=42 on 1.5B: {len(u42_formulas)}/{len(all_results)}")
for r in u42_formulas:
    red = " [REDUNDANT w/ SCR]" if r['redundant_with_scr'] else ""
    rho_str = f"{r['rho_scr']:+.3f}" if r['rho_scr'] is not None else "N/A"
    print(f"   {r['name']:25s} g={format_effect(r['hedges_g'])} rho_SCR={rho_str:>7}{red}")

# 2. Which achieve U=42 on ALL models?
u42_all = [r for r in all_results if r.get('all_models_u42') == True]
print(f"\n2. Formulas with complete ordering across all {len(MODELS)} models: "
      f"{len(u42_all) if len(MODELS) >= 3 else 'N/A (fewer than three models)'}")
for r in u42_all:
    red = " [REDUNDANT]" if r['redundant_with_scr'] else ""
    rho_str = f"{r['rho_scr']:+.3f}" if r['rho_scr'] is not None else "N/A"
    print(f"   {r['name']:25s} g={format_effect(r['hedges_g'])} rho_SCR={rho_str:>7}{red}")

# 3. Which are NOT redundant with SCR?
non_redundant_u42 = [
    r for r in u42_formulas if r['redundant_with_scr'] is False
]
print(f"\n3. U=42 AND not redundant with SCR (|rho| < 0.95): {len(non_redundant_u42)}")
for r in non_redundant_u42:
    rho_str = f"{r['rho_scr']:+.3f}" if r['rho_scr'] is not None else "N/A"
    print(f"   {r['name']:25s} rho_SCR={rho_str:>7}  field={r['field']}")

# 4. Prefix-score association (descriptive only)
steer_nominal = [r for r in all_results if r['p_steering'] is not None and r['p_steering'] < 0.05]
print(
    "\n4. Formulas with nominal asymptotic Spearman p<0.05 against the Function-task "
    f"prefix score (n=6, descriptive only): {len(steer_nominal)}"
)
for r in steer_nominal:
    print(f"   {r['name']:25s} rho_steer={r['rho_steering']:+.3f} p={r['p_steering']:.4f}")

# 5. Best candidates (U=42 + not redundant + theoretical grounding)
print("\n5. TOP CANDIDATES (U=42, not redundant, cross-model or PT-rich):")
candidates = [
    r
    for r in all_results
    if r['U'] == 42 and r['redundant_with_scr'] is False
]
candidates.sort(key=lambda x: abs(x['rho_scr']) if x['rho_scr'] is not None else 1.0)
for r in candidates[:10]:
    steer = f"rho_steer={r['rho_steering']:+.3f}" if r['rho_steering'] is not None else "steer=N/A"
    xm = f"XM={r['cross_model_rho']:.3f}" if r['cross_model_rho'] is not None else "XM=N/A"
    rho_str = f"{r['rho_scr']:+.3f}" if r['rho_scr'] is not None else "N/A"
    print(f"   {r['name']:25s} rho_SCR={rho_str:>7} {steer:>18s} {xm:>10s} [{r['field']}]")

# 6. Redundancy matrix for top formulas
print("\n6. REDUNDANCY MATRIX (Spearman between top U=42 formulas on 1.5B):")
top_names = [r['name'] for r in u42_formulas[:15]]
if len(top_names) > 1:
    print(f"   {'':25s}", end="")
    for n in top_names[:8]:
        print(f" {n[:7]:>7s}", end="")
    print()
    for i, n1 in enumerate(top_names[:8]):
        v1 = [all_results[[r['name'] for r in all_results].index(n1)]['values_1.5B'][c] for c in ALL_CONCEPTS]
        print(f"   {n1:25s}", end="")
        for j, n2 in enumerate(top_names[:8]):
            v2 = [all_results[[r['name'] for r in all_results].index(n2)]['values_1.5B'][c] for c in ALL_CONCEPTS]
            rho, _ = stats.spearmanr(v1, v2)
            print(f" {rho:+.3f}", end="")
        print()

# =====================================================================
# RESIDUAL ANALYSIS
# =====================================================================
print(f"\n{'='*80}")
print("RESIDUAL ANALYSIS: ordering of residuals after linear regression on SCR (descriptive)")
print(f"{'='*80}")

for r in non_redundant_u42[:10]:
    fname = r['name']
    fvals = np.array([r['values_1.5B'][c] for c in ALL_CONCEPTS])
    scr_arr = np.array(scr_vals_15b)
    labels = np.array([0]*6 + [1]*7)  # 0=semantic, 1=persona

    # Regress formula on SCR
    slope, intercept, _, _, _ = stats.linregress(scr_arr, fvals)
    residuals = fvals - (slope * scr_arr + intercept)

    # Test if residuals still separate
    res_sem = residuals[:6]
    res_per = residuals[6:]

    if np.std(res_sem) > 1e-15 or np.std(res_per) > 1e-15:
        u_res, p_res = stats.mannwhitneyu(res_sem, res_per, alternative='two-sided')
        print(f"  {fname:25s}: residual U={u_res}/42  two-sided p={p_res:.4f}")
    else:
        print(f"  {fname:25s}: constant residuals")

# =====================================================================
# SAVE RESULTS
# =====================================================================
output = {
    'config': {
        'mode': args.mode,
        'n_formulas': len(formulas),
        'bonferroni_family_size': N_FORMULAS,
        'n_concepts': N_CONCEPTS,
        'n_semantic': len(SEMANTIC),
        'n_persona': len(PERSONA),
        'bonferroni_alpha': float(0.05 / N_FORMULAS),
        'mann_whitney_alternative': 'two-sided',
        'u_orientation': 'larger observed family mean; descriptive only',
        'models': MODELS,
        'aperture': APERTURE,
        'input_sha256': input_hashes,
        'run_specification_sha256': run_specification_sha256,
        'peak_layers': peak_layers_pt,
        'fit_sample_counts': fit_sample_counts,
        'fit_dimensions': {name: len(spectrum) for name, spectrum in eig_spectra.items()},
        'steering_associations_available': args.mode == 'historical-summaries',
    },
    'formulas': all_results,
    'summary': {
        'u42_on_1.5B': len(u42_formulas),
        'u42_on_all_models': len(u42_all) if len(MODELS) >= 3 else None,
        'non_redundant_u42': len(non_redundant_u42),
        'steering_nominal_p_lt_05_descriptive': len(steer_nominal) if steering_sem else None,
    }
}

out_path = OUTPUT_PATH
with open_new_output(out_path) as f:
    json.dump(output, f, indent=2, allow_nan=False)
print(f"\nResults saved to {out_path}")

# Also print per-concept values for top candidates
print(f"\n{'='*80}")
print("PER-CONCEPT VALUES FOR TOP NON-REDUNDANT U=42 FORMULAS")
print(f"{'='*80}")

for r in non_redundant_u42[:8]:
    print(f"\n--- {r['name']} ({r['desc']}) ---")
    rho_s = f"{r['rho_scr']:+.3f}" if r['rho_scr'] is not None else "N/A"
    print(f"  Direction: {r['direction']}, g={format_effect(r['hedges_g'])}, rho_SCR={rho_s}")
    print(f"  {'Concept':20s} {'Value':>10s} {'SCR':>8s} {'Type':>6s}")
    for c in ALL_CONCEPTS:
        v = r['values_1.5B'].get(c, float('nan'))
        scr_v = results['SCR']['values']['Qwen2.5-1.5B'].get(c, float('nan'))
        cat = "SEM" if c in SEMANTIC else "PER"
        print(f"  {c:20s} {v:10.4f} {scr_v:8.4f} {cat:>6s}")
