# Results and source files

At aperture $\alpha=.05$ and gain $\beta=2$, all 21 complete maps across
three models and seven targets expand every direction at the generation
layers. Across the twelve matched model–target settings, held-out matrix
correction RMS is 0.093%–7.815% of the matched-control displacement.

Of 480 complete/control response pairs, 285 differ in text. Text differences
do not measure successful target suppression. On the four-target panel,
retaining the matrix did not establish a consistent additional
target-suppression benefit: all twelve target-expression and all twelve
corrected-coherence 95% intervals include zero. This does not establish
equivalence. The aperture sweep measures geometry, while the separate
LEACE/INLP study measures fresh-probe accessibility, not generation after erasure.

Run `python -m pytest -q` to check the processed records. The five standalone
rebuild commands and source limitations are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
Paths below are relative to the repository root.

| Result | Records | Check |
|---|---|---|
| All 21 complete maps expand every direction at aperture .05 and gain 2 | `plot_data/aperture_baseline_21_fits.csv`, `plot_data/operator_cells.csv` | `experiments/aperture/source/verify_aperture_records.py` |
| At aperture .05 and gain 2, held-out matrix correction is 0.093%–7.815% of control displacement | `plot_data/aperture_correction_fold_evidence.csv`, `plot_data/aperture_correction_12_settings.csv` | Aperture verifier pools fold energies before taking square roots |
| Geometric aperture sweep | `plot_data/aperture_spectra_21_fits.csv`, `plot_data/aperture_summary.csv` | Aperture verifier; this is not a behavioral sweep |
| 1,920 generated answers; 480 beta-2 complete/control pairs, 195 identical and 285 different | `experiments/matched_control/results/combined/combined_exact_output_match.csv` and per-model quality records | `test_generation_inventory_from_aggregates`, `test_matched_pair_refusal_and_censoring_counts` |
| Target-expression contrasts | Trait rows of `plot_data/gemini_behavior_cells.csv` | `test_matched_behavior_cells`, `test_current_matched_endpoints_are_separated` |
| Corrected coherence: 960 answers, 12 contrasts; means −2.125 to 1.875 | `plot_data/corrected_coherence_question_contrasts.csv`, `plot_data/corrected_coherence_results.json` | `experiments/corrected_coherence/source/verify_corrected_coherence.py` recomputes intervals and Holm-12 values |
| Coherence saturation and identical-input disagreements | `plot_data/corrected_coherence_audit.json` | Corrected-coherence verifier checks summary counts; per-answer scores are not included |
| Earlier 240-answer secondary-evaluator sensitivity | `plot_data/secondary_matched_sensitivity.csv`, `secondary_evaluator_selection.csv`, `deepseek_judge_prompts.json` | `test_independent_judge_aggregate_inventory` |
| Fresh-probe accessibility under LEACE, INLP and random controls | `experiments/erasure/results/`, `plot_data/erasure_rank_sweep_summary.csv` | `tests/test_reported_numbers.py` recomputes metrics from held-out predictions |
| Paired-difference geometry, layers 26/26/30 | `plot_data/paired_geometry_main.csv`, `plot_data/paired_delta_results.json` | `experiments/paired_geometry/source/verify_paired_geometry_records.py` |
| Purification, Fisher and GEVD historical aggregate comparisons | `plot_data/finite_alternatives_scores.csv` | `experiments/finite_alternatives/source/verify_finite_alternatives.py` |
| Function-task, persona and inversion tables | `provenance/broad_study_results.json`, `plot_data/semantic_per_concept_outcomes.csv` | `scripts/verify_semantic_rebuild.py` |
| Construction sensitivity of SCR | `plot_data/protocol_swap_summary.csv`, `experiments/protocol_swap/results/protocol_swap_results.json` | `test_protocol_summary_matches_full_results` |
| 48-measure spectral screen | `plot_data/formula_screening_results.csv` | `test_spectral_screen_scope`; this CSV retains historical one-sided p-values; new analyses use two-sided tests ([conventions](../experiments/spectral_screening/README.md#statistical-conventions-in-the-analysis-code)) |
| Broad seven-target, 200-token study | `plot_data/broad_generated_response_summary.json`, `plot_data/motivating_behavior_summary.csv` | `test_generated_response_grid_has_336_means` |
| Extraction instructions and evaluation questions | `experimental_materials.json` | `test_complete_target_inventory`, `test_embedded_runner_materials_match_canonical_inventory` |

The finite-alternative table combines eight historical outputs. Its joint score
uses the original Gemini-rubric coherence endpoint; the matched panel's
corrected coherence uses a different judge and coherence-only rubric. Their
scores describe different evaluations. The material inventory checks prompt
definitions; the original tensors lack per-row prompt identifiers needed to
verify prompt-to-tensor execution.
