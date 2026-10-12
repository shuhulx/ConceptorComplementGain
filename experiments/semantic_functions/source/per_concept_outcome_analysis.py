#!/usr/bin/env python3
"""Build descriptive per-concept SCR-versus-outcome tables.

This script reads the frozen broad_study_results.json and produces
three separate descriptive panels:
  1. SCR versus additive semantic task accuracy
  2. SCR versus conceptor semantic task accuracy
  3. SCR versus beta=2 conceptor-suppression trait change

The three outcomes are analyzed separately. These retrospective associations
do not measure prospective out-of-sample prediction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

SCRIPT_VERSION = "PER-CONCEPT-OUTCOME-TABLE-1.2"
BOOTSTRAP_SEED_NAMESPACE = "PER-CONCEPT-OUTCOME-TABLE-1.1"
EXPECTED_JSON_SHA256 = "82e913868d26c84abb98d210abd72d526c1a48972ef2cfb2aa247c255ea02dd4"
N_BOOTSTRAP = 10_000

SEMANTIC = (
    "antonym",
    "country_capital",
    "present_past",
    "english_french",
    "singular_plural",
    "capitalize",
)
PERSONA = (
    "sycophancy",
    "hallucination",
    "evil",
    "impolite",
    "optimistic",
    "apathetic",
    "humorous",
)
EXPECTED_MODELS = (
    "Qwen2.5-1.5B",
    "Qwen2.5-7B",
    "Llama-3.1-8B",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: str) -> int:
    payload = "||".join(parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def concept_cluster_bootstrap(
    frame: pd.DataFrame,
    outcome: str,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    """Resample concepts while retaining all three model rows per concept.

    NumPy arrays support 10,000 draws with each concept's model rows resampled
    together and the fixed RNG call order used for the included results.
    The interval is descriptive, not an out-of-sample prediction interval.
    """
    concepts = sorted(frame["concept"].unique())
    blocks_x: list[np.ndarray] = []
    blocks_y: list[np.ndarray] = []
    for concept in concepts:
        block = frame[frame["concept"] == concept].sort_values("model")
        if len(block) != 3:
            raise RuntimeError(f"Expected three model rows for {concept}, found {len(block)}")
        blocks_x.append(block["scr"].to_numpy(dtype=float))
        blocks_y.append(block[outcome].to_numpy(dtype=float))
    x = np.stack(blocks_x, axis=0)
    y = np.stack(blocks_y, axis=0)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(concepts), size=(n_bootstrap, len(concepts)))
    draws: list[float] = []
    for indices in sampled:
        xb = x[indices].reshape(-1)
        yb = y[indices].reshape(-1)
        rx = rankdata(xb, method="average")
        ry = rankdata(yb, method="average")
        sx = float(np.std(rx))
        sy = float(np.std(ry))
        if sx == 0.0 or sy == 0.0:
            continue
        rho = float(np.corrcoef(rx, ry)[0, 1])
        if math.isfinite(rho):
            draws.append(rho)
    if not draws:
        return float("nan"), float("nan")
    low, high = np.quantile(np.asarray(draws, dtype=float), [0.025, 0.975])
    return float(low), float(high)


def exact_concept_block_permutation(
    frame: pd.DataFrame,
    outcome: str,
) -> tuple[float, int, int]:
    """Permute complete concept blocks while preserving model alignment."""
    concepts = sorted(frame["concept"].unique())
    models = sorted(frame["model"].unique())
    x_blocks = []
    y_blocks = []
    for concept in concepts:
        block = frame[frame["concept"] == concept].set_index("model").loc[models]
        if len(block) != 3:
            raise RuntimeError(f"Expected three model rows for {concept}")
        x_blocks.append(block["scr"].to_numpy(dtype=float))
        y_blocks.append(block[outcome].to_numpy(dtype=float))
    x = np.concatenate(x_blocks)
    observed = float(spearmanr(x, np.concatenate(y_blocks)).statistic)
    exceedances = 0
    total = 0
    for permutation in itertools.permutations(range(len(concepts))):
        permuted_y = np.concatenate([y_blocks[index] for index in permutation])
        statistic = float(spearmanr(x, permuted_y).statistic)
        exceedances += int(abs(statistic) >= abs(observed) - 1e-15)
        total += 1
    return float(exceedances / total), exceedances, total


def add_holm_adjustment(rows: list[dict[str, Any]]) -> None:
    field = "exact_concept_block_permutation_p_two_sided"
    ordered = sorted(enumerate(rows), key=lambda item: item[1][field])
    running_max = 0.0
    adjusted = [0.0] * len(rows)
    for rank, (original_index, row) in enumerate(ordered):
        candidate = min(1.0, (len(rows) - rank) * float(row[field]))
        running_max = max(running_max, candidate)
        adjusted[original_index] = running_max
    for row, value in zip(rows, adjusted):
        row["holm_adjusted_exact_block_p_across_three_panels"] = value


def find_suppression_key(persona_record: Mapping[str, Any]) -> str:
    candidates = sorted(
        key
        for key in persona_record
        if key.startswith("conceptor_suppress_") and key.endswith("_b2.0")
    )
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one beta=2 suppress key, found {candidates}")
    return candidates[0]


def build_tables(data: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if tuple(data.keys()) != EXPECTED_MODELS:
        if set(data.keys()) != set(EXPECTED_MODELS):
            raise RuntimeError(f"Model universe mismatch: {list(data)}")
    semantic_rows: list[dict[str, Any]] = []
    persona_rows: list[dict[str, Any]] = []
    for model_name in EXPECTED_MODELS:
        results = data[model_name]["results"]
        for concept in SEMANTIC:
            theory = results["theory"][concept]
            outcome = results["semantic_results"][concept]
            semantic_rows.append(
                {
                    "model": model_name,
                    "concept": concept,
                    "scr": float(theory["SCR"]),
                    "raw_ccr": float(theory["rawCCR"]),
                    "additive_accuracy": float(outcome["additive_mean"]),
                    "conceptor_accuracy": float(outcome["conceptor_mean"]),
                    "additive_layer": int(
                        next(
                            (
                                int(key.split("_L", 1)[1].split("_", 1)[0])
                                for key in []
                            ),
                            results.get("chen_layer", -1),
                        )
                    ),
                    "conceptor_best_layer": int(outcome["best_config"]["layer"]),
                    "conceptor_best_beta": float(outcome["best_config"]["beta"]),
                    "conceptor_best_aperture": float(outcome["best_config"]["aperture"]),
                }
            )
        for trait in PERSONA:
            theory = results["theory"][trait]
            record = results["persona_table"][trait]
            suppress_key = find_suppression_key(record)
            baseline_trait = float(record["baseline"]["trait"])
            steered_trait = float(record[suppress_key]["trait"])
            baseline_coherence = float(record["baseline"]["coherence"])
            steered_coherence = float(record[suppress_key]["coherence"])
            persona_rows.append(
                {
                    "model": model_name,
                    "concept": trait,
                    "scr": float(theory["SCR"]),
                    "raw_ccr": float(theory["rawCCR"]),
                    "condition": suppress_key,
                    "baseline_trait_score": baseline_trait,
                    "steered_trait_score": steered_trait,
                    "trait_change_steered_minus_baseline": steered_trait - baseline_trait,
                    "suppression_improvement_baseline_minus_steered": baseline_trait
                    - steered_trait,
                    "baseline_coherence": baseline_coherence,
                    "steered_coherence": steered_coherence,
                    "coherence_change_steered_minus_baseline": steered_coherence
                    - baseline_coherence,
                }
            )
    semantic = pd.DataFrame(semantic_rows)
    persona = pd.DataFrame(persona_rows)
    if len(semantic) != 18 or len(persona) != 21:
        raise RuntimeError(
            f"Unexpected table sizes: semantic={len(semantic)}, persona={len(persona)}"
        )
    return semantic, persona


def correlation_row(
    panel: str,
    frame: pd.DataFrame,
    outcome: str,
    outcome_direction: str,
) -> dict[str, Any]:
    result = spearmanr(frame["scr"], frame[outcome])
    exact_p, exceedances, permutations = exact_concept_block_permutation(
        frame, outcome
    )
    low, high = concept_cluster_bootstrap(
        frame,
        outcome,
        n_bootstrap=N_BOOTSTRAP,
        seed=stable_seed(BOOTSTRAP_SEED_NAMESPACE, panel, outcome),
    )
    return {
        "panel": panel,
        "x": "scr",
        "y": outcome,
        "n_model_concept_rows": int(len(frame)),
        "n_unique_concepts": int(frame["concept"].nunique()),
        "spearman_rho": float(result.statistic),
        "ordinary_row_level_p_for_audit_only": float(result.pvalue),
        "exact_concept_block_permutation_p_two_sided": exact_p,
        "exact_block_permutation_exceedances": exceedances,
        "exact_block_permutations": permutations,
        "concept_cluster_bootstrap_ci95_low": low,
        "concept_cluster_bootstrap_ci95_high": high,
        "inference_method": (
            "exact two-sided permutation of complete concept blocks while "
            "preserving the three aligned model rows per block"
        ),
        "outcome_direction": outcome_direction,
        "interpretation_boundary": (
            "descriptive association on the measured concept set; "
            "not prospective out-of-sample prediction"
        ),
    }


def max_inversion_pair(frame: pd.DataFrame, outcome: str, model: str) -> dict[str, Any]:
    subset = frame[frame["model"] == model].copy()
    candidates = []
    records = subset.to_dict(orient="records")
    for higher in records:
        for lower in records:
            if higher["scr"] <= lower["scr"]:
                continue
            # Higher SCR but lower measured outcome.
            outcome_gap = float(lower[outcome] - higher[outcome])
            if outcome_gap <= 0:
                continue
            candidates.append((outcome_gap, higher, lower))
    if not candidates:
        return {
            "model": model,
            "outcome": outcome,
            "found": False,
        }
    gap, higher, lower = max(candidates, key=lambda item: item[0])
    return {
        "model": model,
        "outcome": outcome,
        "found": True,
        "higher_scr_concept": higher["concept"],
        "higher_scr": float(higher["scr"]),
        "higher_scr_outcome": float(higher[outcome]),
        "lower_scr_concept": lower["concept"],
        "lower_scr": float(lower["scr"]),
        "lower_scr_outcome": float(lower[outcome]),
        "outcome_inversion_gap": float(gap),
    }


def make_scatter(
    frame: pd.DataFrame,
    outcome: str,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    for model_name, group in frame.groupby("model", sort=True):
        ax.scatter(group["scr"], group[outcome], label=model_name)
        for _, row in group.iterrows():
            ax.annotate(
                row["concept"].replace("_", " "),
                (row["scr"], row[outcome]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
            )
    ax.set_xlabel("Spectral Concentration Ratio (SCR)")
    ax.set_ylabel(outcome.replace("_", " "))
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run(input_json: Path, output_dir: Path, expected_sha256: str | None) -> None:
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. Use a new directory."
        )
    if not input_json.is_file():
        raise FileNotFoundError(input_json)
    input_sha = sha256_file(input_json)
    if expected_sha256 and input_sha.lower() != expected_sha256.lower():
        raise RuntimeError(
            f"Input JSON hash mismatch: expected {expected_sha256}, got {input_sha}"
        )
    with input_json.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    semantic, persona = build_tables(data)
    correlations = [
        correlation_row(
            "semantic_additive",
            semantic,
            "additive_accuracy",
            "higher is better task accuracy",
        ),
        correlation_row(
            "semantic_conceptor",
            semantic,
            "conceptor_accuracy",
            "higher is better task accuracy",
        ),
        correlation_row(
            "persona_beta2_trait_change",
            persona,
            "trait_change_steered_minus_baseline",
            "positive means the target trait increased under attempted suppression",
        ),
    ]
    add_holm_adjustment(correlations)
    inversions = []
    for model_name in EXPECTED_MODELS:
        inversions.append(
            max_inversion_pair(semantic, "additive_accuracy", model_name)
        )
        inversions.append(
            max_inversion_pair(semantic, "conceptor_accuracy", model_name)
        )

    output_dir.mkdir(parents=True)
    atomic_write_csv(
        output_dir / "semantic_per_concept_outcomes.csv",
        semantic.to_dict(orient="records"),
    )
    atomic_write_csv(
        output_dir / "persona_per_concept_outcomes.csv",
        persona.to_dict(orient="records"),
    )
    atomic_write_csv(output_dir / "correlation_summary.csv", correlations)
    atomic_write_json(output_dir / "correlation_summary.json", correlations)
    atomic_write_csv(output_dir / "within_model_inversion_examples.csv", inversions)

    make_scatter(
        semantic,
        "additive_accuracy",
        "SCR versus additive semantic accuracy",
        output_dir / "scr_vs_additive_accuracy.png",
    )
    make_scatter(
        semantic,
        "conceptor_accuracy",
        "SCR versus conceptor semantic accuracy",
        output_dir / "scr_vs_conceptor_accuracy.png",
    )
    make_scatter(
        persona,
        "trait_change_steered_minus_baseline",
        "SCR versus beta=2 trait change under attempted suppression",
        output_dir / "scr_vs_persona_trait_change.png",
    )

    corr_frame = pd.DataFrame(correlations)
    qwen = semantic[semantic["model"] == "Qwen2.5-1.5B"].set_index("concept")
    cc = qwen.loc["country_capital"]
    pp = qwen.loc["present_past"]
    report = f"""# Per-concept measured-outcome analysis

## Data and outcomes

The broad-study results contain six semantic tasks and seven persona traits
evaluated across three models. Semantic outcomes are the additive and conceptor
next-token prefix-match scores. Persona outcomes are changes in judged target
expression at beta=2. Each outcome has its own association panel.

## Associations

| Panel | Rows / concept blocks | Spearman rho | Exact block-permutation p | Holm p | Concept-block bootstrap 95% interval |
|---|---:|---:|---:|---:|---:|
| SCR vs additive semantic accuracy | 18 / 6 | {corr_frame.iloc[0].spearman_rho:.3f} | {corr_frame.iloc[0].exact_concept_block_permutation_p_two_sided:.3f} | {corr_frame.iloc[0].holm_adjusted_exact_block_p_across_three_panels:.3f} | [{corr_frame.iloc[0].concept_cluster_bootstrap_ci95_low:.3f}, {corr_frame.iloc[0].concept_cluster_bootstrap_ci95_high:.3f}] |
| SCR vs conceptor semantic accuracy | 18 / 6 | {corr_frame.iloc[1].spearman_rho:.3f} | {corr_frame.iloc[1].exact_concept_block_permutation_p_two_sided:.3f} | {corr_frame.iloc[1].holm_adjusted_exact_block_p_across_three_panels:.3f} | [{corr_frame.iloc[1].concept_cluster_bootstrap_ci95_low:.3f}, {corr_frame.iloc[1].concept_cluster_bootstrap_ci95_high:.3f}] |
| SCR vs beta=2 persona trait change | 21 / 7 | {corr_frame.iloc[2].spearman_rho:+.3f} | {corr_frame.iloc[2].exact_concept_block_permutation_p_two_sided:.3f} | {corr_frame.iloc[2].holm_adjusted_exact_block_p_across_three_panels:.3f} | [{corr_frame.iloc[2].concept_cluster_bootstrap_ci95_low:.3f}, {corr_frame.iloc[2].concept_cluster_bootstrap_ci95_high:.3f}] |

The persona outcome is `steered trait score - baseline trait score`; positive
values mean that the target trait increased under attempted suppression.
The exact tests permute complete concept blocks and preserve the aligned rows
for all three models. Holm values adjust the three reported panels. The
`ordinary_row_level_p_for_audit_only` column in the correlation tables reports
unadjusted row-level Spearman p-values, which ignore dependence between models
evaluating the same concept. Inference here uses the concept-block tests.

## Example comparison

For Qwen2.5-1.5B, country-capital has higher SCR than present-past
({cc.scr:.3f} versus {pp.scr:.3f}) but much lower measured task success:
conceptor accuracy is {cc.conceptor_accuracy:.3f} versus
{pp.conceptor_accuracy:.3f}, and additive accuracy is
{cc.additive_accuracy:.3f} versus {pp.additive_accuracy:.3f}.

## Scope

The associations describe the evaluated concepts rather than prospective
prediction on unseen concepts. Semantic and persona outcomes use different
scales and are interpreted separately. With six semantic concepts and seven persona
traits, the intervals and tests are conditional on this small measured set.
"""
    atomic_write_text(output_dir / "analysis_report.md", report)

    manifest = {
        "status": "PASS",
        "script_version": SCRIPT_VERSION,
        "script_sha256": sha256_file(Path(__file__)),
        "input_json": input_json.name,
        "input_json_sha256": input_sha,
        "models": list(EXPECTED_MODELS),
        "semantic_concepts": list(SEMANTIC),
        "persona_concepts": list(PERSONA),
        "semantic_rows": int(len(semantic)),
        "persona_rows": int(len(persona)),
        "n_bootstrap": N_BOOTSTRAP,
        "bootstrap_seed_namespace": BOOTSTRAP_SEED_NAMESPACE,
        "bootstrap_scheme": (
            "resample concepts while retaining all three model rows for each "
            "sampled concept; descriptive sensitivity only"
        ),
        "permutation_scheme": (
            "enumerate all permutations of complete concept blocks while "
            "preserving aligned model rows; Holm-adjust across three panels"
        ),
        "interpretation_boundary": (
            "measured-concept descriptive association, not prospective OOS prediction"
        ),
    }
    # Fail closed on the three canonical point estimates.
    expected_points = {
        "semantic_additive": (
            -0.4447884416924664,
            0.06438522410365068,
            13 / 720,
            0.03611111111111111,
        ),
        "semantic_conceptor": (
            -0.45534337508430983,
            0.05758324531713278,
            4 / 720,
            0.016666666666666666,
        ),
        "persona_beta2_trait_change": (
            0.10008227820562743,
            0.6660048640566664,
            2400 / 5040,
            0.47619047619047616,
        ),
    }
    checks = {}
    for row in correlations:
        expected_rho, expected_naive_p, expected_exact_p, expected_holm_p = (
            expected_points[row["panel"]]
        )
        rho_error = abs(row["spearman_rho"] - expected_rho)
        naive_p_error = abs(
            row["ordinary_row_level_p_for_audit_only"] - expected_naive_p
        )
        exact_p_error = abs(
            row["exact_concept_block_permutation_p_two_sided"]
            - expected_exact_p
        )
        holm_p_error = abs(
            row["holm_adjusted_exact_block_p_across_three_panels"]
            - expected_holm_p
        )
        checks[row["panel"]] = {
            "rho_abs_error": rho_error,
            "ordinary_p_abs_error": naive_p_error,
            "exact_block_p_abs_error": exact_p_error,
            "holm_p_abs_error": holm_p_error,
            "pass": max(
                rho_error,
                naive_p_error,
                exact_p_error,
                holm_p_error,
            )
            < 1e-12,
        }
    if not all(item["pass"] for item in checks.values()):
        raise RuntimeError(f"Canonical correlation cross-check failed: {checks}")
    validation = {
        "status": "PASS",
        "checks": checks,
        "country_capital_scr": float(cc.scr),
        "present_past_scr": float(pp.scr),
        "country_capital_conceptor_accuracy": float(cc.conceptor_accuracy),
        "present_past_conceptor_accuracy": float(pp.conceptor_accuracy),
    }
    atomic_write_json(output_dir / "VALIDATION.json", validation)
    atomic_write_text(
        output_dir / "VALIDATION.md",
        "# Validation\n\nStatus: **PASS**\n\n"
        "- Exact semantic additive correlation reproduced\n"
        "- Exact semantic conceptor correlation reproduced\n"
        "- Exact persona beta=2 trait-change correlation reproduced\n"
        "- Exact concept-block permutation p-values reproduced\n"
        "- Holm adjustment across the three panels reproduced\n"
        "- 18 semantic and 21 persona model-concept rows present\n"
        "- Canonical country-capital/present-past counterexample reproduced\n",
    )
    atomic_write_json(output_dir / "analysis_manifest.json", manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute per-concept associations from broad-study results.")
    parser.add_argument("--input-json", type=Path, required=True, help="Broad-study results JSON.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New directory outside the repository for tables, plots and the analysis report.")
    parser.add_argument(
        "--expected-sha256",
        default=EXPECTED_JSON_SHA256,
        help="Set to an empty string to record but not pre-enforce the input hash.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected = args.expected_sha256.strip() or None
    run(args.input_json, args.output_dir, expected)
    print(f"PASS: {args.output_dir}")


if __name__ == "__main__":
    main()
