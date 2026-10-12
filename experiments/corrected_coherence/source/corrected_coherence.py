"""Offline analysis of the current matched-panel endpoints.

Current endpoints for the 3 models x 4 matched targets:

* target expression: historical Gemini scores, ``endpoint == "trait"`` rows of
  ``plot_data/gemini_behavior_cells.csv`` only (original scores, not rescored);
* coherence: the corrected DeepSeek coherence-only rescore in
  ``plot_data/corrected_coherence_results.json``.

The ``endpoint == "coherence"`` Gemini rows are historical (old rubric) and are
reachable only through :func:`historical_gemini_coherence`. The earlier
240-answer secondary-evaluator subset is reachable only through
:func:`secondary_evaluator_sensitivity`.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[3]
PLOT_DATA = ROOT / "plot_data"
GEMINI_CELLS = PLOT_DATA / "gemini_behavior_cells.csv"
CORRECTED_RESULTS = PLOT_DATA / "corrected_coherence_results.json"
CORRECTED_AUDIT = PLOT_DATA / "corrected_coherence_audit.json"
QUESTION_CONTRASTS = PLOT_DATA / "corrected_coherence_question_contrasts.csv"
SECONDARY_SENSITIVITY = PLOT_DATA / "secondary_matched_sensitivity.csv"

MODELS = ("1.5b", "7b", "8b")
TRAITS = ("apathetic", "hallucination", "optimistic", "sycophancy")
SETTINGS = tuple((model, trait) for model in MODELS for trait in TRAITS)
N_QUESTIONS = 20
PAIRS_PER_QUESTION = 2
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_BASE_SEED = 991
HOLM_FAMILY_SIZE = 12

CURRENT_TARGET_SOURCE = "gemini_behavior_cells.csv[endpoint=trait]"
CURRENT_COHERENCE_SOURCE = "corrected_coherence_results.json[contrasts]"
HISTORICAL_COHERENCE_SOURCE = "gemini_behavior_cells.csv[endpoint=coherence]"


def bootstrap_seed(trait: str) -> int:
    """Seed used by the frozen estimator; it depends on the target, not the model."""
    return BOOTSTRAP_BASE_SEED + sum(map(ord, f"{trait}|full|coherence"))


def question_bootstrap_ci(
    values: Sequence[float], seed: int, draws: int = BOOTSTRAP_DRAWS
) -> tuple[float, float]:
    """Percentile 95% interval over question-cluster resamples (frozen RNG call order)."""
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = array[rng.integers(0, len(array), size=(draws, len(array)))].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def wilcoxon_p(values: Sequence[float]) -> float:
    """Two-sided Wilcoxon signed-rank p; 1.0 when every question difference is zero."""
    array = np.asarray(values, dtype=float)
    if not np.any(array):
        return 1.0
    return float(wilcoxon(array, alternative="two-sided", zero_method="wilcox").pvalue)


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjustment over exactly the supplied family, in input order."""
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def summarize_questions(table: dict[tuple[str, str], list[float]]) -> list[dict]:
    """Recompute the 12 corrected contrasts from question-level paired means."""
    if set(table) != set(SETTINGS):
        raise ValueError("Question table must cover exactly the 12 matched settings")
    if len(SETTINGS) != HOLM_FAMILY_SIZE:
        raise ValueError("Corrected coherence Holm family must contain 12 tests")
    result = []
    for model, trait in SETTINGS:
        values = np.asarray(table[model, trait], dtype=float)
        if values.shape != (N_QUESTIONS,):
            raise ValueError(f"{model}/{trait} must have {N_QUESTIONS} question means")
        seed = bootstrap_seed(trait)
        low, high = question_bootstrap_ci(values, seed)
        result.append({"model": model, "trait": trait,
                       "n_pairs": N_QUESTIONS * PAIRS_PER_QUESTION, "n_questions": N_QUESTIONS,
                       "mean_difference": float(values.mean()), "ci95_low": low, "ci95_high": high,
                       "wilcoxon_p_nominal": wilcoxon_p(values), "bootstrap_seed": seed})
    for row, adjusted in zip(result, holm([r["wilcoxon_p_nominal"] for r in result])):
        row["holm_12_corrected_coherence"] = adjusted
    return result


def load_question_contrasts(path: Path = QUESTION_CONTRASTS) -> dict[tuple[str, str], list[float]]:
    """Load the 240-row question table keyed by (model, trait), ordered by question."""
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    cells: dict[tuple[str, str], dict[int, float]] = {}
    for row in rows:
        if int(row["n_pairs"]) != PAIRS_PER_QUESTION:
            raise ValueError("Every question must contribute two paired replicates")
        value = float(row["mean_difference"])
        if not math.isfinite(value) or abs(value) > 100:
            raise ValueError("Question mean difference must be finite and within [-100, 100]")
        questions = cells.setdefault((row["model"], row["trait"]), {})
        index = int(row["question_index"])
        if index in questions:
            raise ValueError(f"Duplicate question row: {row['model']}/{row['trait']}/{index}")
        questions[index] = value
    if len(rows) != len(SETTINGS) * N_QUESTIONS or set(cells) != set(SETTINGS):
        raise ValueError("Question table must contain 12 settings x 20 questions")
    for key, questions in cells.items():
        if set(questions) != set(range(N_QUESTIONS)):
            raise ValueError(f"Incomplete question coverage for {key}")
    return {key: [cells[key][q] for q in range(N_QUESTIONS)] for key in SETTINGS}


def load_corrected_results(path: Path = CORRECTED_RESULTS) -> dict[tuple[str, str], dict]:
    """Corrected coherence contrasts keyed by (model, trait)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    keyed = {}
    for row in data["contrasts"]:
        key = (row["model"], row["trait"])
        if key in keyed:
            raise ValueError(f"Duplicate corrected contrast: {key}")
        keyed[key] = row
    if set(keyed) != set(SETTINGS):
        raise ValueError("Corrected results must contain exactly the 12 matched settings")
    return keyed


def _gemini_rows(path: Path, endpoint: str) -> dict[tuple[str, str], dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["endpoint"] == endpoint]
    keyed = {}
    for row in rows:
        key = (row["model_key"], row["trait"])
        if key in keyed:
            raise ValueError(f"Duplicate Gemini {endpoint} row: {key}")
        keyed[key] = row
    if set(keyed) != set(SETTINGS):
        raise ValueError(f"Gemini {endpoint} rows must cover exactly the 12 matched settings")
    return keyed


def current_matched_endpoints(
    gemini_csv: Path = GEMINI_CELLS, corrected_json: Path = CORRECTED_RESULTS
) -> dict[str, dict[tuple[str, str], dict]]:
    """Return the two current endpoints, each keyed by (model, trait).

    Target expression comes only from Gemini ``endpoint == "trait"`` rows;
    coherence comes only from the corrected rescore JSON.
    """
    target = {
        key: {"model": key[0], "trait": key[1], "endpoint": "target_expression",
              "status": "current", "source": CURRENT_TARGET_SOURCE,
              "n_pairs": int(row["n_paired_rows"]), "n_questions": int(row["n_question_clusters"]),
              "mean_difference": float(row["mean_difference_not2_minus_identity2"]),
              "ci95_low": float(row["ci95_low"]), "ci95_high": float(row["ci95_high"])}
        for key, row in _gemini_rows(gemini_csv, "trait").items()
    }
    coherence = {
        key: {"model": key[0], "trait": key[1], "endpoint": "coherence",
              "status": "current", "source": CURRENT_COHERENCE_SOURCE,
              "n_pairs": row["n_pairs"], "n_questions": row["n_questions"],
              "mean_difference": row["mean_difference"],
              "ci95_low": row["ci95_low"], "ci95_high": row["ci95_high"],
              "holm_12_corrected_coherence": row["holm_12_corrected_coherence"]}
        for key, row in load_corrected_results(corrected_json).items()
    }
    return {"target_expression": target, "coherence": coherence}


def historical_gemini_coherence(gemini_csv: Path = GEMINI_CELLS) -> dict[tuple[str, str], dict]:
    """Historical Gemini 2.5 Flash coherence (old rubric); not a current endpoint."""
    return {
        key: {"model": key[0], "trait": key[1], "endpoint": "coherence",
              "status": "historical_gemini_old_rubric", "source": HISTORICAL_COHERENCE_SOURCE,
              "mean_difference": float(row["mean_difference_not2_minus_identity2"]),
              "wilcoxon_holm_historical_24_test_family": float(row["wilcoxon_holm_within_family"])}
        for key, row in _gemini_rows(gemini_csv, "coherence").items()
    }


def validate_current_endpoints(
    endpoints: dict[str, dict[tuple[str, str], dict]],
    gemini_csv: Path = GEMINI_CELLS,
    corrected_json: Path = CORRECTED_RESULTS,
) -> None:
    """Fail if either current endpoint is not bound to its required source rows."""
    if set(endpoints) != {"target_expression", "coherence"}:
        raise ValueError("Exactly two current matched endpoints are expected")
    expected = current_matched_endpoints(gemini_csv, corrected_json)
    for name in expected:
        if set(endpoints[name]) != set(SETTINGS):
            raise ValueError(f"{name} must contain exactly the 12 matched settings")
        for key, row in endpoints[name].items():
            reference = expected[name][key]
            if set(row) != set(reference):
                raise ValueError(f"{name} {key} has missing or unexpected fields")
            for field, value in reference.items():
                actual = row[field]
                if type(actual) is not type(value) or actual != value:
                    source = "corrected rescore" if name == "coherence" else "Gemini trait row"
                    raise ValueError(f"{name} {key} {field} is not the {source}")


def secondary_evaluator_sensitivity(path: Path = SECONDARY_SENSITIVITY) -> dict[tuple[str, str], dict]:
    """Earlier 240-answer / 60-pair DeepSeek sensitivity subset (historical, not prospective)."""
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    keyed = {(row["model_key"], row["trait"]): dict(row, status="historical_secondary_subset")
             for row in rows}
    if len(rows) != len(SETTINGS) or set(keyed) != set(SETTINGS):
        raise ValueError("Secondary sensitivity table must contain the 12 matched settings once")
    return keyed
