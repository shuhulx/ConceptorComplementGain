from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"


class PublicEvidenceTests(unittest.TestCase):
    def test_generation_inventory_from_aggregates(self) -> None:
        response_rows = 0
        for model in ("1.5b", "7b", "8b"):
            path = (
                EXPERIMENTS
                / "matched_control"
                / "results"
                / model
                / "analysis_data_quality.json"
            )
            document = json.loads(path.read_text())
            response_rows += int(document["actual_response_rows"])
        self.assertEqual(response_rows, 1920)

        exact_path = (
            EXPERIMENTS
            / "matched_control"
            / "results"
            / "combined"
            / "combined_exact_output_match.csv"
        )
        with exact_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 12)
        pair_count = sum(int(row["n_pairs"]) for row in rows)
        exact_count = sum(
            int(row["n_pairs"]) * float(row["full_token_exact_rate"])
            for row in rows
        )
        self.assertEqual(pair_count, 480)
        self.assertEqual(exact_count, 195)

    def test_matched_refusal_totals(self) -> None:
        path = (
            EXPERIMENTS
            / "matched_control"
            / "results"
            / "combined"
            / "combined_categorical_trait_refusals.csv"
        )
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        totals = {
            condition: sum(
                int(row["categorical_trait_refusals"])
                for row in rows
                if row["view"] == "full" and row["condition"] == condition
            )
            for condition in ("not_b2", "identity_b2")
        }
        self.assertEqual(totals, {"not_b2": 11, "identity_b2": 12})

    def test_matched_pair_refusal_and_censoring_counts(self) -> None:
        combined = EXPERIMENTS / "matched_control" / "results" / "combined"

        def rows(name: str) -> list[dict[str, str]]:
            with (combined / name).open(newline="") as handle:
                return list(csv.DictReader(handle))

        exact = rows("combined_exact_output_match.csv")
        identical = round(
            sum(int(r["n_pairs"]) * float(r["full_token_exact_rate"]) for r in exact)
        )
        self.assertEqual((480 - identical, identical), (285, 195))

        # Pair-level refusal overlap, derived from the per-condition totals and
        # the number of trait pairs excluded because either arm refused.
        refusals = rows("combined_categorical_trait_refusals.csv")
        complete_refusals = sum(
            int(r["categorical_trait_refusals"])
            for r in refusals
            if r["view"] == "full" and r["condition"] == "not_b2"
        )
        control_refusals = sum(
            int(r["categorical_trait_refusals"])
            for r in refusals
            if r["view"] == "full" and r["condition"] == "identity_b2"
        )
        contrasts = [
            r
            for r in rows("combined_not2_vs_identity2_by_view.csv")
            if r["view"] == "full" and r["analysis_population"] == "all_rows"
        ]
        trait = [r for r in contrasts if r["endpoint"] == "trait"]
        coherence = [r for r in contrasts if r["endpoint"] == "coherence"]
        self.assertEqual((len(trait), len(coherence)), (12, 12))
        either = sum(int(r["categorical_refusal_pairs_excluded"]) for r in trait)
        both = complete_refusals + control_refusals - either
        self.assertEqual(
            (complete_refusals, control_refusals, either), (11, 12, 13)
        )
        self.assertEqual(
            (both, complete_refusals - both, control_refusals - both, 480 - either),
            (10, 1, 2, 467),
        )
        self.assertEqual(sum(int(r["n_paired_rows"]) for r in trait), 467)
        self.assertEqual(sum(int(r["n_paired_rows"]) for r in coherence), 480)

        # 1,024-token cap hits: four responses across all arms; the three in the
        # beta-2 arms fall in two Qwen-1.5B Hallucination complete/control pairs.
        censoring = rows("combined_condition_censoring.csv")
        self.assertEqual(sum(int(r["n"]) for r in censoring), 1920)
        hits = {
            (r["model_key"], r["trait"], r["condition"]): round(
                int(r["n"]) * float(r["cap_hit_rate_1024"])
            )
            for r in censoring
        }
        self.assertEqual(sum(hits.values()), 4)
        beta2_hits = {
            key: count
            for key, count in hits.items()
            if count and key[2] in {"not_b2", "identity_b2"}
        }
        self.assertEqual(
            beta2_hits,
            {
                ("1.5b", "hallucination", "not_b2"): 1,
                ("1.5b", "hallucination", "identity_b2"): 2,
            },
        )
        capped_pairs = {
            (r["model_key"], r["trait"]): round(
                int(r["n_pairs"]) * float(r["either_1024_cap_hit_rate"])
            )
            for r in exact
        }
        self.assertEqual(
            {key: count for key, count in capped_pairs.items() if count},
            {("1.5b", "hallucination"): 2},
        )

    def test_independent_judge_aggregate_inventory(self) -> None:
        path = (
            EXPERIMENTS
            / "independent_judge"
            / "results"
            / "analysis"
            / "validation_decision.json"
        )
        document = json.loads(path.read_text())
        self.assertEqual(document["accepted_judgments"], 240)
        self.assertEqual(document["status"], "SCREENING_CHALLENGE")

    def test_response_bearing_artifacts_are_absent(self) -> None:
        forbidden_names = {
            "responses.jsonl",
            "judged_dual_view.jsonl",
            "dual_view_scores.csv",
            "deepseek_judgments.jsonl",
            "judgments_merged.csv",
            "identical_near_text_consistency_pairs.csv",
        }
        present = {
            path.name
            for path in EXPERIMENTS.rglob("*")
            if path.is_file() and path.name in forbidden_names
        }
        self.assertEqual(present, set())


if __name__ == "__main__":
    unittest.main()
