#!/usr/bin/env python3
"""Recompute aperture formulas, pooled ratios and summaries from result tables.

Reads ``plot_data/aperture_*``, ``operator_cells.csv`` and
``correction_cells.csv`` to check coverage and agreement between tables.
These checks start from recorded spectral values and corrections, rather than
the original extraction tensors.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PLOT_DATA = Path(__file__).resolve().parents[3] / "plot_data"
APERTURES = (0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
BETA = 2.0
TRAITS = ("sycophancy", "hallucination", "evil", "impolite", "optimistic", "apathetic", "humorous")
MATCHED = ("sycophancy", "hallucination", "optimistic", "apathetic")
MODELS = {
    "1.5b": (9, "Qwen2.5-1.5B-Instruct"),
    "7b": (9, "Qwen2.5-7B-Instruct"),
    "8b": (10, "Llama-3.1-8B-Instruct"),
}
N_QUESTIONS, N_TEMPLATES, N_FOLDS, FOLD_SEED = 20, 5, 5, 42
# Manuscript aperture table at printed precision:
# alpha: (max c, min g, attenuating /21, /12, RMS min %, RMS max %), decimals 6, 6, -, -, 3, 3.
PAPER_TABLE = {
    0.025: (0.048751, 1.902497, 0, 0, 0.023, 2.089),
    0.05: (0.170124, 1.659751, 0, 0, 0.093, 7.815),
    0.1: (0.450549, 1.098901, 0, 0, 0.370, 24.861),
    0.2: (0.766355, 0.467290, 4, 2, 1.459, 55.470),
    0.5: (0.953488, 0.093023, 14, 8, 8.247, 93.568),
    1.0: (0.987952, 0.024096, 14, 8, 24.895, 115.562),
    2.0: (0.996960, 0.006079, 21, 12, 52.998, 129.875),
}
# Held-out correction ranges at six-decimal precision.
QUOTED_RANGES = {0.05: (0.092979, 7.814755), 0.2: (1.459120, 55.469812), 2.0: (52.998220, 129.874515)}
RTOL = 1e-13


class Checks:
    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.failures: list[str] = []

    def __call__(self, group: str, condition: bool, detail: str) -> None:
        self.counts[group] = self.counts.get(group, 0) + 1
        if not condition:
            self.failures.append(f"{group}: {detail}")

    def close(self, group: str, observed: float, expected: float, detail: str, rtol: float = RTOL, atol: float = 0.0) -> None:
        self(group, bool(np.isclose(observed, expected, rtol=rtol, atol=atol)), f"{detail}: {observed!r} != {expected!r}")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def flag(value: str) -> bool:
    if value not in ("True", "False"):
        raise ValueError(f"not a boolean: {value!r}")
    return value == "True"


def expected_groups() -> list[list[int]]:
    permutation = np.random.RandomState(FOLD_SEED).permutation(N_QUESTIONS)
    return [sorted(int(q) for q in group) for group in permutation.reshape(N_FOLDS, N_QUESTIONS // N_FOLDS)]


def check_spectra(check: Checks, spectra: list[dict], baseline: list[dict]) -> dict:
    keys = [(r["model"], r["trait"], float(r["aperture"])) for r in spectra]
    check("spectra_coverage", len(spectra) == 147 and len(set(keys)) == 147, f"{len(spectra)} rows")
    check("spectra_coverage", set(keys) == {(m, t, a) for m in MODELS for t in TRAITS for a in APERTURES}, "model/trait/aperture grid")
    by_fit: dict[tuple, list[dict]] = {}
    for row in spectra:
        model, trait, alpha = row["model"], row["trait"], float(row["aperture"])
        label = f"{model}/{trait}/{alpha:g}"
        layer, name = MODELS.get(model, (None, None))
        check("spectra_fields", (int(row["layer"]), row["model_name"], int(row["n_fit"]), float(row["beta"])) == (layer, name, 100, BETA), label)
        check("spectra_fields", flag(row["in_matched_generation"]) == (trait in MATCHED), label)
        lam, c, g = float(row["lambda_max"]), float(row["c_max"]), float(row["minimum_gain_beta2"])
        scaled = alpha**2 * lam
        check.close("spectra_formula_c", c, scaled / (1 + scaled), label)
        check.close("spectra_formula_g", g, BETA / (1 + scaled), label)
        check.close("spectra_formula_g", g, BETA * (1 - c), label, rtol=1e-12)
        check.close("spectra_formula_unit_gain_aperture", float(row["first_unit_gain_aperture_beta2"]), 1 / np.sqrt(lam), label)
        attenuated = flag(row["has_any_attenuated_direction"])
        # At beta 2: g < 1  <=>  c > 0.5  <=>  alpha^2 lambda > beta - 1  <=>  alpha > 1/sqrt(lambda).
        check("spectra_attenuation_flag", attenuated == (g < 1) == (c > 0.5) == (scaled > BETA - 1)
              == (alpha > float(row["first_unit_gain_aperture_beta2"])), label)
        check("spectra_bounds", 0 < c < 1 and 0 < g < BETA and lam > 0, label)
        by_fit.setdefault((model, trait), []).append(row)
    for (model, trait), rows in by_fit.items():
        rows.sort(key=lambda r: float(r["aperture"]))
        label = f"{model}/{trait}"
        check("spectra_fit_constant", len({(r["lambda_max"], r["first_unit_gain_aperture_beta2"]) for r in rows}) == 1, label)
        check("spectra_monotone", bool(np.all(np.diff([float(r["c_max"]) for r in rows]) > 0)), f"{label} c_max")
        check("spectra_monotone", bool(np.all(np.diff([float(r["minimum_gain_beta2"]) for r in rows]) < 0)), f"{label} gain")
    expected_baseline = [r for r in spectra if float(r["aperture"]) == 0.05]
    check("baseline_equals_spectra_alpha_0.05", baseline == expected_baseline, f"{len(baseline)} baseline rows")
    return by_fit


def check_folds(check: Checks, foldrows: list[dict], groups: list[list[int]]) -> dict:
    keys = [(r["model"], r["trait"], float(r["aperture"]), int(r["fold"])) for r in foldrows]
    grid = {(m, t, a, f) for m in MODELS for t in MATCHED for a in APERTURES for f in range(N_FOLDS)}
    check("fold_coverage", len(foldrows) == 420 and len(set(keys)) == 420 and set(keys) == grid, f"{len(foldrows)} rows")
    check("fold_questions", groups == expected_groups(), "provenance folds differ from RandomState(42) 5x4 groups")
    check("fold_questions", sorted(q for group in groups for q in group) == list(range(N_QUESTIONS))
          and all(len(group) == 4 for group in groups), "question groups do not partition 0..19")
    # Template-major rows: row index = template * 20 + question; each fold holds out 20 of 100 rows.
    heldout = [{t * N_QUESTIONS + q for t in range(N_TEMPLATES) for q in group} for group in groups]
    check("fold_questions", all(len(rows) == 20 for rows in heldout)
          and sorted(i for rows in heldout for i in rows) == list(range(100)), "template-major row partition")
    series: dict[tuple, list[dict]] = {}
    for row in foldrows:
        model, trait, fold = row["model"], row["trait"], int(row["fold"])
        label = f"{model}/{trait}/{float(row['aperture']):g}/fold{fold}"
        check("fold_fields", (int(row["layer"]), int(row["train_rows"]), int(row["test_rows"])) == (MODELS[model][0], 80, 20), label)
        check("fold_questions", row["heldout_question_indices"] == ";".join(map(str, groups[fold])), label)
        energy, correction = float(row["input_energy"]), float(row["correction_energy"])
        # ||C z|| < ||z|| for a finite-aperture conceptor.
        check("fold_energy_bounds", 0 < correction < energy, label)
        series.setdefault((model, trait, fold), []).append(row)
    for (model, trait, fold), rows in series.items():
        rows.sort(key=lambda r: float(r["aperture"]))
        label = f"{model}/{trait}/fold{fold}"
        check("fold_input_energy_constant", len({r["input_energy"] for r in rows}) == 1, label)
        check("fold_energy_monotone", bool(np.all(np.diff([float(r["correction_energy"]) for r in rows]) > 0)), label)
    return series


def check_corrections(check: Checks, corrections: list[dict], foldrows: list[dict]) -> None:
    keys = [(r["model"], r["trait"], float(r["aperture"])) for r in corrections]
    check("correction_coverage", len(corrections) == 84 and len(set(keys)) == 84
          and set(keys) == {(m, t, a) for m in MODELS for t in MATCHED for a in APERTURES}, f"{len(corrections)} rows")
    folds_by_setting: dict[tuple, list[dict]] = {}
    for row in foldrows:
        folds_by_setting.setdefault((row["model"], row["trait"], float(row["aperture"])), []).append(row)
    by_setting: dict[tuple, list[dict]] = {}
    for row in corrections:
        model, trait, alpha = row["model"], row["trait"], float(row["aperture"])
        label = f"{model}/{trait}/{alpha:g}"
        check("correction_fields", (int(row["layer"]), row["model_name"], float(row["beta"]), int(row["question_folds"]),
                                    int(row["heldout_rows"])) == (*MODELS[model], BETA, 5, 100), label)
        folds = sorted(folds_by_setting.get((model, trait, alpha), []), key=lambda r: int(r["fold"]))
        check("correction_fold_count", len(folds) == 5, label)
        # Pool energies over folds (fold order), then take the square root.
        denominator = numerator = 0.0
        for fold in folds:
            denominator += float(fold["input_energy"])
            numerator += float(fold["correction_energy"])
        check.close("correction_pooled_sums", float(row["input_energy_sum"]), denominator, label)
        check.close("correction_pooled_sums", float(row["correction_energy_sum"]), numerator, label)
        percent = float(row["rms_correction_percent"])
        check.close("correction_pooled_ratio", percent, 100 * BETA / abs(BETA - 1) * np.sqrt(numerator / denominator), label)
        check("correction_200_percent_bound", 0 < percent < 200, label)
        by_setting.setdefault((model, trait), []).append(row)
    for (model, trait), rows in by_setting.items():
        rows.sort(key=lambda r: float(r["aperture"]))
        check("correction_monotone", bool(np.all(np.diff([float(r["rms_correction_percent"]) for r in rows]) > 0)), f"{model}/{trait}")


def check_summary(check: Checks, summary: list[dict], spectra: list[dict], corrections: list[dict]) -> None:
    check("summary_rows", [float(r["aperture"]) for r in summary] == list(APERTURES), "aperture order")
    for row in summary:
        alpha = float(row["aperture"])
        label = f"alpha={alpha:g}"
        for prefix, traits in [("all21", TRAITS), ("matched12", MATCHED)]:
            selected = [r for r in spectra if float(r["aperture"]) == alpha and r["trait"] in traits]
            check("summary_recomputed", float(row[f"{prefix}_max_c"]) == max(float(r["c_max"]) for r in selected), f"{label} {prefix} max c")
            check("summary_recomputed", float(row[f"{prefix}_min_gain"]) == min(float(r["minimum_gain_beta2"]) for r in selected), f"{label} {prefix} min gain")
            check("summary_recomputed", int(row[f"{prefix}_fits_with_any_attenuation"])
                  == sum(flag(r["has_any_attenuated_direction"]) for r in selected), f"{label} {prefix} count")
        selected = [float(r["rms_correction_percent"]) for r in corrections if float(r["aperture"]) == alpha]
        check("summary_recomputed", float(row["heldout12_min_correction_percent"]) == min(selected), f"{label} min ratio")
        check("summary_recomputed", float(row["heldout12_max_correction_percent"]) == max(selected), f"{label} max ratio")
        max_c, min_g, all21, matched12, low, high = PAPER_TABLE[alpha]
        observed = (float(row["all21_max_c"]), float(row["all21_min_gain"]), float(row["heldout12_min_correction_percent"]),
                    float(row["heldout12_max_correction_percent"]))
        for value, printed, decimals in zip(observed, (max_c, min_g, low, high), (6, 6, 3, 3)):
            check("summary_manuscript_table", abs(value - printed) <= 0.5 * 10**-decimals + 1e-12, f"{label}: {value} vs {printed}")
        check("summary_manuscript_table", (int(row["all21_fits_with_any_attenuation"]), int(row["matched12_fits_with_any_attenuation"]))
              == (all21, matched12), f"{label} attenuation counts")
        if alpha in QUOTED_RANGES:
            for value, quoted in zip(observed[2:], QUOTED_RANGES[alpha]):
                check("summary_quoted_ranges", abs(value - quoted) <= 5e-7, f"{label}: {value} vs {quoted}")
    baseline = [r for r in spectra if float(r["aperture"]) == 0.05]
    largest = max(baseline, key=lambda r: float(r["c_max"]))
    check("summary_quoted_ranges", (largest["model"], largest["trait"]) == ("1.5b", "evil"), "alpha=.05 maximum is not Qwen-1.5B Evil")


def check_historical(check: Checks, spectra: list[dict], corrections: list[dict], operator: list[dict], correction_cells: list[dict]) -> None:
    baseline = {(r["model"], r["trait"]): r for r in spectra if float(r["aperture"]) == 0.05}
    check("historical_operator_cells", len(operator) == 21 and {(r["model_key"], r["trait"]) for r in operator} == set(baseline), "21 fits")
    for row in operator:
        key = (row["model_key"], row["trait"])
        if key not in baseline:
            continue
        new, label = baseline[key], "/".join(key)
        check("historical_operator_cells", (int(row["layer"]), int(row["n_samples"]), float(row["aperture"])) == (int(new["layer"]), 100, 0.05), label)
        check.close("historical_operator_cells", float(new["c_max"]), float(row["c_max"]), label, rtol=0, atol=1e-12)
        check.close("historical_operator_cells", float(new["minimum_gain_beta2"]), float(row["min_beta2_eigendirectional_gain"]), label, rtol=0, atol=1e-12)
    pooled = {(r["model"], r["trait"]): r for r in corrections if float(r["aperture"]) == 0.05}
    check("historical_correction_cells", len(correction_cells) == 12 and {(r["model_key"], r["trait"]) for r in correction_cells} == set(pooled), "12 settings")
    for row in correction_cells:
        key = (row["model_key"], row["trait"])
        if key not in pooled:
            continue
        label = "/".join(key)
        check("historical_correction_cells", (int(row["layer"]), float(row["beta"])) == (int(pooled[key]["layer"]), BETA), label)
        check.close("historical_correction_cells", float(pooled[key]["rms_correction_percent"]) / 100, float(row["rms_specific_over_shared"]),
                    label, rtol=0, atol=1e-14)


def check_provenance(check: Checks, provenance: dict) -> None:
    expected = {"beta": BETA, "apertures": list(APERTURES), "spectral_rows": 147, "correction_rows": 84, "fold_rows": 420, "question_fold_seed": FOLD_SEED}
    for key, value in expected.items():
        check("provenance", provenance.get(key) == value, key)
    sources = {s.get("model"): s for s in provenance.get("sources", [])}
    for model, (layer, _name) in MODELS.items():
        source = sources.get(model, {})
        check("provenance", (source.get("generation_layer"), source.get("hook_name"))
              == (layer, f"blocks.{layer}.hook_resid_pre"), model)


def run(plot_data: Path) -> Checks:
    check = Checks()
    spectra = read_csv(plot_data / "aperture_spectra_21_fits.csv")
    baseline = read_csv(plot_data / "aperture_baseline_21_fits.csv")
    corrections = read_csv(plot_data / "aperture_correction_12_settings.csv")
    foldrows = read_csv(plot_data / "aperture_correction_fold_evidence.csv")
    summary = read_csv(plot_data / "aperture_summary.csv")
    provenance = json.loads((plot_data / "aperture_provenance.json").read_text(encoding="utf-8"))
    groups = provenance.get("heldout_question_indices_by_fold", [])
    check_provenance(check, provenance)
    check_spectra(check, spectra, baseline)
    check_folds(check, foldrows, groups)
    check_corrections(check, corrections, foldrows)
    check_summary(check, summary, spectra, corrections)
    check_historical(check, spectra, corrections, read_csv(plot_data / "operator_cells.csv"), read_csv(plot_data / "correction_cells.csv"))
    return check


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plot-data", type=Path, default=PLOT_DATA, help="directory holding the aperture records (default: repository plot_data/)")
    args = parser.parse_args(argv)
    print("Aperture formulas and consistency checks from recorded spectra and corrections")
    try:
        check = run(args.plot_data)
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as error:
        print(f"FAIL: could not read aperture records: {error!r}")
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
