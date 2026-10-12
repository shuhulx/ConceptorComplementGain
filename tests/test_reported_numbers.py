from __future__ import annotations

import csv
import importlib.util
import json
import math
import unittest
from pathlib import Path

import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "plot_data"


def csv_rows(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


class ReportedNumberTests(unittest.TestCase):
    def test_operator_scope_and_expansion(self) -> None:
        # c_max is the largest conceptor eigenvalue c (not a covariance
        # eigenvalue); the complete-map gain along that direction is beta*(1-c).
        rows = csv_rows("operator_cells.csv")
        self.assertEqual(len(rows), 21)
        self.assertEqual(
            {(r["model_key"], r["layer"]) for r in rows},
            {("1.5b", "9"), ("7b", "9"), ("8b", "10")},
        )
        self.assertEqual(len({(r["model_key"], r["trait"]) for r in rows}), 21)
        self.assertEqual(len({r["trait"] for r in rows}), 7)
        self.assertEqual({r["aperture"] for r in rows}, {"0.05"})
        self.assertEqual({r["n_samples"] for r in rows}, {"100"})
        for row in rows:
            c_max = float(row["c_max"])
            gain = float(row["min_beta2_eigendirectional_gain"])
            self.assertTrue(0 <= c_max < 0.5, row["trait"])
            self.assertAlmostEqual(gain, 2.0 * (1.0 - c_max), places=12)
        cmax = max(float(r["c_max"]) for r in rows)
        min_gain = min(float(r["min_beta2_eigendirectional_gain"]) for r in rows)
        self.assertAlmostEqual(cmax, 0.17012441499087702, places=14)
        self.assertAlmostEqual(min_gain, 1.659751170018246, places=14)
        self.assertEqual(round(cmax, 6), 0.170124)
        self.assertEqual(round(min_gain, 6), 1.659751)
        self.assertTrue(all(float(r["min_beta2_eigendirectional_gain"]) > 1 for r in rows))
        extremum = min(rows, key=lambda r: float(r["min_beta2_eigendirectional_gain"]))
        self.assertEqual((extremum["model_key"], extremum["trait"]), ("1.5b", "evil"))
        matched_traits = {r["trait"] for r in csv_rows("gemini_behavior_cells.csv")}
        self.assertNotIn(extremum["trait"], matched_traits)

    def test_matched_correction_range(self) -> None:
        rows = csv_rows("correction_cells.csv")
        self.assertEqual(len(rows), 12)
        percentages = [100 * float(r["rms_specific_over_shared"]) for r in rows]
        self.assertEqual(round(min(percentages), 3), 0.093)
        self.assertEqual(round(max(percentages), 3), 7.815)

    def test_paired_geometry_is_above_equality(self) -> None:
        rows = csv_rows("paired_geometry_main.csv")
        self.assertEqual(len(rows), 21)
        full = json.loads((DATA / "paired_delta_results.json").read_text())
        peak_layer = {"1.5b": "L26_peak", "7b": "L26_peak", "8b": "L30_peak"}
        for row in rows:
            source = full[row["model_key"]][peak_layer[row["model_key"]]][
                row["target"].lower()
            ]
            self.assertAlmostEqual(
                float(row["positive_shannon_effective_rank"]),
                float(source["pos"]["eff_rank"]),
                places=14,
            )
            self.assertAlmostEqual(
                float(row["paired_shannon_effective_rank"]),
                float(source["delta"]["eff_rank"]),
                places=14,
            )
            self.assertEqual(float(row["positive_k90"]), float(source["pos"]["k90"]))
            self.assertEqual(float(row["paired_k90"]), float(source["delta"]["k90"]))
            self.assertGreater(
                float(row["paired_shannon_effective_rank"]),
                float(row["positive_shannon_effective_rank"]),
            )
            self.assertGreater(float(row["paired_k90"]), float(row["positive_k90"]))

    def test_semantic_outcome_count(self) -> None:
        rows = csv_rows("semantic_per_concept_outcomes.csv")
        self.assertEqual(len(rows), 18)
        wins_or_ties = sum(
            float(row["conceptor_accuracy"]) >= float(row["additive_accuracy"])
            for row in rows
        )
        self.assertEqual(wins_or_ties, 14)

    def test_erasure_scope_and_soft_complement_identity(self) -> None:
        rows = csv_rows("erasure_rank_sweep_summary.csv")
        self.assertEqual(len(rows), 22)
        self.assertEqual({r["split"] for r in rows}, {"question_grouped", "template_grouped"})
        source_directories = {
            "question_grouped": "primary",
            "template_grouped": "template_sensitivity",
        }
        source_columns = {
            "mean_balanced_accuracy": "balanced_accuracy",
            "mean_roc_auc": "roc_auc",
            "mean_rank_removed": "mean_actual_rank_removed",
            "mean_normalized_edit_mse_test": "mean_normalized_edit_mse_test",
            "mean_energy_retention_test": "mean_energy_retention_test",
        }
        for split in {r["split"] for r in rows}:
            by_method = {r["method"]: r for r in rows if r["split"] == split}
            aggregate_path = (
                ROOT
                / "experiments"
                / "erasure"
                / "results"
                / source_directories[split]
                / "aggregate_results.csv"
            )
            with aggregate_path.open(newline="") as handle:
                aggregate = list(csv.DictReader(handle))
            for method, reported in by_method.items():
                if method.startswith("random_rowspace_r1_"):
                    selected = [
                        row
                        for row in aggregate
                        if row["method"].startswith("random_rowspace_r1_s")
                    ]
                elif method.startswith("random_rowspace_r20_"):
                    selected = [
                        row
                        for row in aggregate
                        if row["method"].startswith("random_rowspace_r20_s")
                    ]
                else:
                    selected = [row for row in aggregate if row["method"] == method]
                self.assertEqual(len(selected), int(reported["n_rows"]))
                for reported_column, source_column in source_columns.items():
                    source_mean = sum(float(row[source_column]) for row in selected) / len(
                        selected
                    )
                    self.assertAlmostEqual(
                        float(reported[reported_column]), source_mean, places=14
                    )
            raw = float(by_method["raw"]["mean_balanced_accuracy"])
            complement = float(
                by_method["conceptor_not_beta1"]["mean_balanced_accuracy"]
            )
            self.assertLessEqual(abs(raw - complement), 0.001)
            for method in ("leace_paper_exact", "inlp_r20"):
                self.assertLess(
                    float(by_method[method]["mean_balanced_accuracy"]), raw
                )

    def test_spectral_screen_scope(self) -> None:
        rows = csv_rows("formula_screening_results.csv")
        self.assertEqual(len(rows), 48)
        full = json.loads(
            (
                ROOT
                / "experiments"
                / "spectral_screening"
                / "results"
                / "formula_screening_results.json"
            ).read_text()
        )
        self.assertEqual(full["config"]["n_formulas"], 48)
        self.assertEqual(full["summary"]["u42_on_all_models"], 8)
        by_name = {formula["name"]: formula for formula in full["formulas"]}
        self.assertEqual(len(by_name), 48)
        numeric_columns = {
            "mann_whitney_u_qwen_1_5b": "U",
            "mann_whitney_p_qwen_1_5b": "p",
            "hedges_g_qwen_1_5b": "hedges_g",
            "spearman_with_scr_qwen_1_5b": "rho_scr",
            "spearman_with_semantic_conceptor_accuracy_qwen_1_5b": "rho_steering",
        }
        for row in rows:
            source = by_name[row["measure"]]
            self.assertEqual(row["field"], source["field"])
            self.assertEqual(row["description"], source["desc"])
            self.assertEqual(
                row["perfect_separation_all_three_core_models"],
                str(bool(source.get("all_models_u42", False))),
            )
            for reported_column, source_column in numeric_columns.items():
                self.assertAlmostEqual(
                    float(row[reported_column]), float(source[source_column]), places=14
                )
        self.assertEqual(
            sum(r["perfect_separation_all_three_core_models"] == "True" for r in rows),
            8,
        )

        semantic = {
            "antonym",
            "country_capital",
            "present_past",
            "english_french",
            "singular_plural",
            "capitalize",
        }
        complete_orderings = 0
        corrected_passes = 0
        for formula in full["formulas"]:
            values = formula["values_1.5B"]
            semantic_values = [value for name, value in values.items() if name in semantic]
            persona_values = [value for name, value in values.items() if name not in semantic]
            self.assertEqual((len(semantic_values), len(persona_values)), (6, 7))
            two_sided_p = stats.mannwhitneyu(
                semantic_values, persona_values, alternative="two-sided"
            ).pvalue
            corrected_passes += two_sided_p < 0.05 / 48
            complete_orderings += (
                min(semantic_values) > max(persona_values)
                or min(persona_values) > max(semantic_values)
            )
        self.assertEqual(corrected_passes, 0)
        self.assertEqual(complete_orderings, 21)
        self.assertEqual(
            sum(bool(formula.get("all_models_u42")) for formula in full["formulas"]),
            8,
        )

    def test_protocol_summary_matches_full_results(self) -> None:
        rows = csv_rows("protocol_swap_summary.csv")
        self.assertEqual(len(rows), 3)
        full = json.loads(
            (
                ROOT
                / "experiments"
                / "protocol_swap"
                / "results"
                / "protocol_swap_results.json"
            ).read_text()
        )
        by_model = {row["model"]: row for row in rows}
        self.assertEqual(set(by_model), set(full))
        for model, source in full.items():
            row = by_model[model]
            summary = source["summary"]
            matched = summary["protocol_matched"]
            comparisons = {
                "fully_swapped_semantic_mean": summary["semantic_scr_mean_swapped"],
                "fully_swapped_persona_mean": summary["persona_scr_mean_swapped"],
                "fully_swapped_p": summary["mann_whitney_p"],
                "prompt_final_semantic_mean": matched["prompt_final"]["semantic_mean"],
                "prompt_final_persona_mean": matched["prompt_final"]["persona_mean"],
                "prompt_final_p": matched["prompt_final"]["mann_whitney_p"],
                "chat_semantic_mean": matched["chat_generation"]["semantic_mean"],
                "chat_persona_mean": matched["chat_generation"]["persona_mean"],
                "chat_p": matched["chat_generation"]["mann_whitney_p"],
            }
            self.assertEqual(
                row["fully_swapped_mann_whitney_u"], summary["mann_whitney_U"]
            )
            for column, expected in comparisons.items():
                self.assertAlmostEqual(float(row[column]), float(expected), places=14)

    def test_generated_response_grid_has_336_means(self) -> None:
        document = json.loads((DATA / "broad_generated_response_summary.json").read_text())
        self.assertEqual(len(document), 3)
        means = 0
        for model in document.values():
            self.assertEqual(len(model["persona_table"]), 7)
            for conditions in model["persona_table"].values():
                self.assertEqual(len(conditions), 8)
                means += sum(len(endpoints) for endpoints in conditions.values())
        self.assertEqual(means, 336)

    def test_matched_behavior_cells(self) -> None:
        # Source consistency only: the Gemini coherence rows are historical; see
        # test_current_matched_endpoints_are_separated for the current endpoints.
        rows = csv_rows("gemini_behavior_cells.csv")
        self.assertEqual(len(rows), 24)
        self.assertEqual({r["endpoint"] for r in rows}, {"trait", "coherence"})

        combined_path = (
            ROOT
            / "experiments"
            / "matched_control"
            / "results"
            / "combined"
            / "combined_not2_vs_identity2_by_view.csv"
        )
        with combined_path.open(newline="") as handle:
            combined_rows = [
                row
                for row in csv.DictReader(handle)
                if row["view"] == "full"
                and row["analysis_population"] == "all_rows"
            ]
        by_key = {
            (row["model_key"], row["trait"], row["endpoint"]): row
            for row in combined_rows
        }
        self.assertEqual(len(by_key), 24)

        float_columns = {
            "mean_difference_not2_minus_identity2": "mean_difference",
            "ci95_low": "ci95_low",
            "ci95_high": "ci95_high",
            "ci90_low": "ci90_low",
            "ci90_high": "ci90_high",
            "wilcoxon_p_nominal": "wilcoxon_question_cluster_p_nominal",
            "wilcoxon_holm_within_family": "wilcoxon_holm_within_family",
            "cap_hit_rate": "either_1024_cap_hit_rate",
        }
        for row in rows:
            key = (row["model_key"], row["trait"], row["endpoint"])
            self.assertIn(key, by_key)
            source = by_key[key]
            self.assertEqual(row["n_paired_rows"], source["n_paired_rows"])
            self.assertEqual(
                row["n_question_clusters"], source["n_question_clusters"]
            )
            self.assertEqual(
                row["ci90_within_uncalibrated_margin_10"],
                source["ci90_within_prespecified_margin"],
            )
            for reported_column, source_column in float_columns.items():
                self.assertAlmostEqual(
                    float(row[reported_column]), float(source[source_column]), places=14
                )

        self.assertLessEqual(
            max(abs(float(r["mean_difference_not2_minus_identity2"])) for r in rows),
            5.25,
        )

    def test_target_expression_range_and_intervals(self) -> None:
        # Paper: target-expression means range from -4.75 to 5.25 points and all
        # twelve 95% intervals include zero. Rows are bound by endpoint, not order.
        trait_rows = [r for r in csv_rows("gemini_behavior_cells.csv") if r["endpoint"] == "trait"]
        self.assertEqual(len(trait_rows), 12)
        means = [float(r["mean_difference_not2_minus_identity2"]) for r in trait_rows]
        self.assertEqual(min(means), -4.75)
        self.assertEqual(max(means), 5.25)
        for row in trait_rows:
            key = (row["model_key"], row["trait"])
            self.assertLessEqual(float(row["ci95_low"]), 0.0, key)
            self.assertGreaterEqual(float(row["ci95_high"]), 0.0, key)

    def test_secondary_evaluator_agreement(self) -> None:
        # Paper: target expression rho=.891, absolute-agreement ICC=.900, MAD
        # 10.29 (n=227); coherence .644/.737/6.81 (n=240); refusal 240/240.
        path = (
            ROOT
            / "experiments"
            / "independent_judge"
            / "results"
            / "analysis"
            / "provider_agreement.csv"
        )
        with path.open(newline="") as handle:
            rows = {
                (row["endpoint"], row["scope"]): row for row in csv.DictReader(handle)
            }
        expected = {
            "target_trait": (227, 0.891, 0.900, 10.29),
            "coherence": (240, 0.644, 0.737, 6.81),
        }
        for endpoint, (n, spearman, icc, mad) in expected.items():
            row = rows[endpoint, "all"]
            self.assertEqual(int(row["n"]), n, endpoint)
            self.assertEqual(round(float(row["spearman"]), 3), spearman, endpoint)
            self.assertEqual(round(float(row["icc_a1_absolute_single"]), 3), icc, endpoint)
            self.assertEqual(round(float(row["mean_absolute_difference"]), 2), mad, endpoint)
        refusal = rows["full_refusal", "all"]
        self.assertEqual(int(refusal["n"]), 240)
        self.assertEqual(float(refusal["mean_absolute_difference"]), 0.0)

    def test_erasure_headline_means(self) -> None:
        # Paper: averaged over three models and seven targets under
        # question-grouped splits, balanced accuracy/AUROC are raw .893/.928,
        # finite complement .893/.928, rank-1 LEACE .663/.730 vs random rank 1
        # .893/.927, rank-20 INLP .669/.752 vs random rank 20 .891/.927.
        # Recomputed from the shipped per-cell aggregate (random baselines pool
        # their three seeds over the 21 cells).
        path = ROOT / "experiments" / "erasure" / "results" / "primary" / "aggregate_results.csv"
        with path.open(newline="") as handle:
            aggregate = [
                row for row in csv.DictReader(handle) if row["split_scheme"] == "question_grouped"
            ]
        groups = {
            "raw": (lambda m: m == "raw", 21, (0.893, 0.928)),
            "complement": (lambda m: m == "conceptor_not_beta1", 21, (0.893, 0.928)),
            "leace": (lambda m: m == "leace_paper_exact", 21, (0.663, 0.730)),
            "random_r1": (lambda m: m.startswith("random_rowspace_r1_s"), 63, (0.893, 0.927)),
            "inlp_r20": (lambda m: m == "inlp_r20", 21, (0.669, 0.752)),
            "random_r20": (lambda m: m.startswith("random_rowspace_r20_s"), 63, (0.891, 0.927)),
        }
        traits = {row["trait"] for row in aggregate}
        self.assertEqual(len(traits), 7)
        cells = {(model, trait) for model in ("1.5b", "7b", "8b") for trait in traits}
        for label, (select, count, (accuracy, auroc)) in groups.items():
            selected = [row for row in aggregate if select(row["method"])]
            self.assertEqual(len(selected), count, label)
            for method in {row["method"] for row in selected}:
                covered = [(r["model_key"], r["trait"]) for r in selected if r["method"] == method]
                self.assertEqual(sorted(covered), sorted(cells), method)
            mean_accuracy = sum(float(r["balanced_accuracy"]) for r in selected) / count
            mean_auroc = sum(float(r["roc_auc"]) for r in selected) / count
            self.assertEqual(round(mean_accuracy, 3), accuracy, label)
            self.assertEqual(round(mean_auroc, 3), auroc, label)

    def test_current_matched_endpoints_are_separated(self) -> None:
        # Current matched endpoints: target expression = the twelve Gemini
        # endpoint=="trait" rows; coherence = the twelve corrected contrasts.
        # Gemini endpoint=="coherence" rows are historical only.
        gemini = csv_rows("gemini_behavior_cells.csv")
        trait_rows = {
            (r["model_key"], r["trait"]): r for r in gemini if r["endpoint"] == "trait"
        }
        historical_coherence = {
            (r["model_key"], r["trait"]): r for r in gemini if r["endpoint"] == "coherence"
        }
        self.assertEqual(len(trait_rows), 12)
        self.assertEqual(set(historical_coherence), set(trait_rows))
        self.assertEqual(
            {trait for _, trait in trait_rows},
            {"sycophancy", "hallucination", "optimistic", "apathetic"},
        )

        corrected = json.loads((DATA / "corrected_coherence_results.json").read_text())
        contrasts = {(c["model"], c["trait"]): c for c in corrected["contrasts"]}
        self.assertEqual(len(corrected["contrasts"]), 12)
        self.assertEqual(set(contrasts), set(trait_rows))
        self.assertEqual(sum(c["n_pairs"] for c in contrasts.values()), 480)
        self.assertEqual({c["n_questions"] for c in contrasts.values()}, {20})
        changed = 0
        for key, contrast in contrasts.items():
            self.assertNotIn("wilcoxon_holm_within_family", contrast)
            self.assertTrue(0 <= contrast["holm_12_corrected_coherence"] <= 1)
            self.assertLessEqual(contrast["ci95_low"], contrast["mean_difference"])
            self.assertLessEqual(contrast["mean_difference"], contrast["ci95_high"])
            historical = float(
                historical_coherence[key]["mean_difference_not2_minus_identity2"]
            )
            self.assertAlmostEqual(contrast["historical_mean_difference"], historical, places=12)
            self.assertAlmostEqual(
                contrast["change_in_contrast"],
                contrast["mean_difference"] - historical,
                places=12,
            )
            changed += contrast["mean_difference"] != historical
        self.assertGreater(changed, 0)


ERASURE_RESULTS = ROOT / "experiments" / "erasure" / "results"
ERASURE_RUNS = ("primary", "template_sensitivity")
# Aggregate fold-mean columns are means of five fold values; float summation
# order may differ from the original run, hence a relative tolerance of 1e-12.
FOLD_MEAN_RELATIVE_TOLERANCE = 1e-12
FOLD_MEAN_ABSOLUTE_TOLERANCE = 1e-15


def _load_erasure_verifier():
    path = ROOT / "experiments" / "erasure" / "source" / "verify_erasure_outputs.py"
    specification = importlib.util.spec_from_file_location("_reported_erasure_verifier", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class ProcessedErasureOutputTests(unittest.TestCase):
    """Recompute the shipped processed erasure outputs from their own row-level files.

    Covers fold and aggregate metrics, paired contrasts, split roles, recorded
    validation checks and summary findings. Resume-checkpoint checks require
    the corresponding checkpoint files.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.verifier = _load_erasure_verifier()

    def test_aggregate_fold_means_recompute_from_fold_results(self) -> None:
        for run in ERASURE_RUNS:
            root = ERASURE_RESULTS / run
            folds: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
            with (root / "fold_results.csv").open(newline="") as handle:
                for row in csv.DictReader(handle):
                    key = (row["model_key"], row["trait"], row["split_scheme"], row["method"])
                    folds.setdefault(key, []).append(row)
            with (root / "aggregate_results.csv").open(newline="") as handle:
                aggregate = list(csv.DictReader(handle))
            self.assertEqual(len(aggregate), 315, run)
            self.assertEqual(len(folds), len(aggregate), run)
            columns = [column for column in aggregate[0] if column.startswith("mean_")]
            self.assertIn("mean_actual_rank_removed", columns)
            self.assertIn("mean_normalized_edit_mse_test", columns)
            for row in aggregate:
                key = (row["model_key"], row["trait"], row["split_scheme"], row["method"])
                source = folds[key]
                self.assertEqual(sorted(int(r["fold"]) for r in source), [0, 1, 2, 3, 4], key)
                for column in columns:
                    values = [float(r[column.removeprefix("mean_")]) for r in source]
                    self.assertTrue(
                        math.isclose(
                            sum(values) / len(values),
                            float(row[column]),
                            rel_tol=FOLD_MEAN_RELATIVE_TOLERANCE,
                            abs_tol=FOLD_MEAN_ABSOLUTE_TOLERANCE,
                        ),
                        (run, key, column),
                    )

    def test_aggregates_contrasts_roles_and_key_findings_recompute(self) -> None:
        # verify_predictions recomputes every aggregate point metric from
        # predictions.csv (tolerance 1e-11 relative and absolute, the verifier's own).
        for run in ERASURE_RUNS:
            root = ERASURE_RESULTS / run
            manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
            self.verifier.verify_manifest(manifest)
            self.assertEqual(self.verifier.verify_predictions(root, manifest)["groups"], 315)
            self.assertEqual(self.verifier.verify_roles(root, manifest)["fold_rows"], 1575)
            self.assertEqual(self.verifier.verify_validations(root, manifest)["failures"], 0)
            self.assertEqual(
                self.verifier.verify_contrasts(root, manifest)["paired_contrast_rows"], 294
            )
            if run == "template_sensitivity":
                with pytest.warns(RuntimeWarning, match="No template key findings were verified"):
                    self.verifier.verify_key_findings(root, manifest, allow_legacy_template_summary=True)
            else:
                self.verifier.verify_key_findings(root, manifest)


if __name__ == "__main__":
    unittest.main()
