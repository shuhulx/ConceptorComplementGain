#!/usr/bin/env python3
"""Check paired-difference geometry records and recompute descriptive summaries.

Reads only the released ``plot_data/paired_delta_results.json`` and
``plot_data/paired_geometry_main.csv``. It checks their structure, the
relations between recorded per-fit fields, the descriptive group means, the
derivation of the CSV from the JSON and the finding that paired
differences have higher Shannon effective rank and k90 in all 21 comparisons.
The checks start from recorded per-fit metrics, not extraction tensors.
Wilcoxon, Mann-Whitney and Cohen's d fields fall outside this verifier's scope.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PLOT_DATA = Path(__file__).resolve().parents[3] / "plot_data"
TRAITS = ("sycophancy", "hallucination", "evil", "impolite", "optimistic", "apathetic", "humorous")
SEMANTIC = ("antonym", "country_capital", "present_past", "english_french", "singular_plural", "capitalize")
# Model: (diagnostic layer, generation layer, steer layer, hidden width, display name).
MODELS = {
    "1.5b": (26, 9, 14, 1536, "Qwen2.5-1.5B-Instruct"),
    "7b": (26, 9, 19, 3584, "Qwen2.5-7B-Instruct"),
    "8b": (30, 10, 15, 4096, "Llama-3.1-8B-Instruct"),
}
# Stored extraction layers: every even layer, plus the generation and steer layers.
TRAJECTORY_LAYERS = {
    model: sorted(set(range(0, diagnostic + 1, 2)) | {generation, steer})
    for model, (diagnostic, generation, steer, _width, _name) in MODELS.items()
}
N_ROWS = 100
CSV_FIELDS = ["model_key", "model", "target", "positive_shannon_effective_rank", "paired_shannon_effective_rank",
              "positive_k90", "paired_k90"]
FIT_FIELDS = {"scr", "mu1", "mu5", "mu1_mu5", "k90", "eff_rank", "fdr", "n_samples", "n_dims"}
FLOAT32_FIELDS = ("scr", "mu1", "mu5", "eff_rank", "fdr")
MEAN_FIELDS = {"scr": "scr", "fdr": "fdr", "mu15": "mu1_mu5", "erank": "eff_rank"}
RTOL = 1e-12


class Checks:
    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.failures: list[str] = []

    def __call__(self, group: str, condition: bool, detail: str) -> None:
        self.counts[group] = self.counts.get(group, 0) + 1
        if not condition:
            self.failures.append(f"{group}: {detail}")

    def close(self, group: str, observed: float, expected: float, detail: str, rtol: float = RTOL) -> None:
        self(group, bool(np.isclose(observed, expected, rtol=rtol, atol=0.0)), f"{detail}: {observed!r} != {expected!r}")


def layer_keys(model: str) -> tuple[str, str]:
    diagnostic, _generation, steer, _width, _name = MODELS[model]
    return f"L{steer}_steer", f"L{diagnostic}_peak"


def check_fit(check: Checks, fit: dict, width: int, label: str) -> None:
    check("fit_fields", set(fit) == FIT_FIELDS, f"{label} fields {sorted(fit)}")
    if set(fit) != FIT_FIELDS:
        return
    check("fit_fields", (fit["n_samples"], fit["n_dims"]) == (N_ROWS, width), f"{label} shape")
    # n centered rows span at most n - 1 directions.
    check("fit_bounds", isinstance(fit["k90"], int) and 1 <= fit["k90"] <= N_ROWS - 1, f"{label} k90 {fit['k90']}")
    check("fit_bounds", 1 <= fit["eff_rank"] <= N_ROWS - 1, f"{label} eff_rank {fit['eff_rank']}")
    # Conceptor eigenvalues lie in [0, 1); sum c^2 / sum c cannot exceed the largest c.
    check("fit_bounds", 0 < fit["mu5"] <= fit["mu1"] < 1 and 0 < fit["scr"] <= fit["mu1"] and 0 < fit["fdr"] <= 1, label)
    check.close("fit_mu1_mu5_ratio", fit["mu1_mu5"], fit["mu1"] / fit["mu5"], label)
    # The historical estimator ran in float32: recorded values must be float32-representable.
    check("fit_float32_values", all(float(np.float32(fit[f])) == fit[f] for f in FLOAT32_FIELDS), label)


def check_results(check: Checks, results: dict) -> None:
    check("json_structure", set(results) == set(MODELS), f"models {sorted(results)}")
    for model, (_diagnostic, _generation, _steer, width, _name) in MODELS.items():
        entry = results.get(model, {})
        # Diagnostic ("peak") layers 26/26/30, not the generation layers 9/9/10.
        check("layer_roles", set(entry) == {*layer_keys(model), "all_layer_trajectory"}, f"{model} keys {sorted(entry)}")
        for key in layer_keys(model):
            layer = entry.get(key, {})
            check("json_structure", set(layer) == {*SEMANTIC, *TRAITS, "group_stats"}, f"{model}/{key}")
            for concept in SEMANTIC:
                fits = layer.get(concept, {})
                # Semantic tasks have no target-opposed rows, hence no paired difference.
                check("json_structure", fits.get("delta", 0) is None, f"{model}/{key}/{concept} delta")
                check_fit(check, fits.get("pos", {}), width, f"{model}/{key}/{concept}/pos")
            for trait in TRAITS:
                for side in ("pos", "delta"):
                    check_fit(check, layer.get(trait, {}).get(side) or {}, width, f"{model}/{key}/{trait}/{side}")
            check_group_means(check, layer, f"{model}/{key}")
        check_trajectory(check, entry, model)


def check_group_means(check: Checks, layer: dict, label: str) -> None:
    stats = layer.get("group_stats", {})
    for name, field in MEAN_FIELDS.items():
        for side, suffix in (("pos", "pos"), ("delta", "del")):
            values = [layer.get(t, {}).get(side, {}).get(field, np.nan) for t in TRAITS]
            check.close("group_descriptive_means", stats.get(f"per_{name}_{suffix}_mean", np.nan), float(np.mean(values)),
                        f"{label} per_{name}_{suffix}_mean")


def check_trajectory(check: Checks, entry: dict, model: str) -> None:
    rows = entry.get("all_layer_trajectory", [])
    check("trajectory_layers", [r.get("layer") for r in rows] == TRAJECTORY_LAYERS[model], f"{model} layers")
    by_layer = {r.get("layer"): r for r in rows}
    for row in rows:
        check.close("trajectory_lift", row["delta_lift"], row["per_del"] - row["per_pos"], f"{model}/L{row['layer']}")
    for key in layer_keys(model):
        layer, row = entry.get(key, {}), by_layer.get(int(key[1:].split("_")[0]), {})
        stats = layer.get("group_stats", {})
        semantic = float(np.mean([layer.get(c, {}).get("pos", {}).get("scr", np.nan) for c in SEMANTIC]))
        for observed, expected, name in [(row.get("per_pos", np.nan), stats.get("per_scr_pos_mean", np.nan), "per_pos"),
                                         (row.get("per_del", np.nan), stats.get("per_scr_del_mean", np.nan), "per_del"),
                                         (row.get("sem_pos", np.nan), semantic, "sem_pos")]:
            check.close("trajectory_matches_layer_fits", observed, expected, f"{model}/{key} {name}")


def check_table(check: Checks, rows: list[dict], header: list[str], results: dict) -> list[dict]:
    check("csv_structure", header == CSV_FIELDS, f"header {header}")
    expected_keys = [(m, MODELS[m][4], t.capitalize()) for m in MODELS for t in TRAITS]
    check("csv_structure", [(r.get("model_key"), r.get("model"), r.get("target")) for r in rows] == expected_keys,
          f"{len(rows)} rows in model/target order")
    comparisons = []
    for row in rows:
        model, trait = row.get("model_key"), str(row.get("target")).lower()
        label = f"{model}/{trait}"
        recorded = results.get(model, {}).get(layer_keys(model)[1] if model in MODELS else "", {}).get(trait)
        if not recorded:
            check("csv_derives_from_json", False, f"{label} missing in JSON")
            continue
        values = {}
        for prefix, side in (("positive", "pos"), ("paired", "delta")):
            rank, k = float(row[f"{prefix}_shannon_effective_rank"]), float(row[f"{prefix}_k90"])
            check("csv_derives_from_json", rank == recorded[side]["eff_rank"], f"{label} {prefix} effective rank")
            check("csv_derives_from_json", k == recorded[side]["k90"] and k.is_integer(), f"{label} {prefix} k90")
            values[prefix] = (rank, k)
        comparisons.append({"label": label, **values})
    return comparisons


def check_claim(check: Checks, comparisons: list[dict]) -> None:
    check("manuscript_21_comparisons", len(comparisons) == 21, f"{len(comparisons)} comparisons")
    for row in comparisons:
        check("manuscript_paired_rank_higher", row["paired"][0] > row["positive"][0], row["label"])
        check("manuscript_paired_k90_higher", row["paired"][1] > row["positive"][1], row["label"])


def run(plot_data: Path) -> Checks:
    check = Checks()
    results = json.loads((plot_data / "paired_delta_results.json").read_text(encoding="utf-8"))
    with (plot_data / "paired_geometry_main.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        header = list(reader.fieldnames or [])
    check_results(check, results)
    check_claim(check, check_table(check, rows, header, results))
    return check


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plot-data", type=Path, default=PLOT_DATA,
                        help="directory holding the paired geometry records (default: repository plot_data/)")
    args = parser.parse_args(argv)
    print("Paired-geometry consistency checks from recorded per-fit metrics")
    try:
        check = run(args.plot_data)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"FAIL: could not read paired geometry records: {error!r}")
        return 1
    for group, count in check.counts.items():
        failed = sum(failure.startswith(group + ":") for failure in check.failures)
        print(f"  {'FAIL' if failed else 'PASS'} {group}: {count - failed}/{count}")
    for failure in check.failures[:20]:
        print(f"  - {failure}")
    total = sum(check.counts.values())
    if check.failures:
        print(f"FAIL: {len(check.failures)} of {total} checks failed")
        return 1
    print(f"PASS: {total} checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
