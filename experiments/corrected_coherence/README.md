# Corrected matched-panel coherence

Computes coherence-only paired contrasts for the matched complete-map versus
matched-control panel from included question-level differences. The panel covers three models
(Qwen2.5-1.5B, Qwen2.5-7B, Llama-3.1-8B) and four targets (apathetic,
hallucination, optimistic, sycophancy) at beta 2.

Run `python experiments/corrected_coherence/source/verify_corrected_coherence.py`
from the repository root to recompute the contrasts on CPU. The estimator
`summarize_questions` in `source/corrected_coherence.py` accepts a dictionary
mapping all twelve `(model, trait)` settings to twenty question-level mean
differences each and returns a list of summary-statistic dictionaries.

## Current versus historical records

| Record | Status | Shipped data | Accessor in `source/corrected_coherence.py` |
|---|---|---|---|
| Gemini 2.5 Flash target expression, `endpoint == "trait"` rows | **current** matched target-expression endpoint (original scores, not rescored) | `plot_data/gemini_behavior_cells.csv` | `current_matched_endpoints()["target_expression"]` |
| Corrected DeepSeek coherence-only rescore of all 960 beta-2 answers (480 pairs) | **current** matched coherence endpoint | `plot_data/corrected_coherence_results.json`, `plot_data/corrected_coherence_audit.json`, `plot_data/corrected_coherence_question_contrasts.csv` | `current_matched_endpoints()["coherence"]` |
| Gemini 2.5 Flash coherence, `endpoint == "coherence"` rows | historical only; its old rubric's 100 anchor required "no hallucinations" | `plot_data/gemini_behavior_cells.csv` | `historical_gemini_coherence()` |
| Earlier DeepSeek-V4-Pro sensitivity subset: 240 answers, 60 complete/control pairs | historical evaluator sensitivity; [selection limits](../../docs/REPRODUCIBILITY.md#interpreting-the-results) | `plot_data/secondary_matched_sensitivity.csv` | `secondary_evaluator_sensitivity()` |

The two current endpoints come from different files and are bound by
(model, trait), never by row order. In the Gemini CSV,
`ci90_within_uncalibrated_margin_10` and `wilcoxon_holm_within_family` are
historical outputs. The latter belongs to the old 24-tests-per-model family.
Neither field is equivalence evidence.

## Corrected rescore

The rescore uses `deepseek-v4-pro`, temperature 0, thinking
disabled and at most 32 output tokens. Each request contained one user message
with the exact question, the full saved answer and the coherence-only rubric. It
carried no model, target, condition or previous-score metadata. No text was
regenerated and no target-expression score changed. All 960 calls succeeded on
their first attempt, and refusals and length-capped answers were retained.

The exact corrected rubric is `corrected_coherence_rubric` in
[`deepseek_judge_prompts.json`](../../deepseek_judge_prompts.json). The
estimator is `summarize_questions` in `source/corrected_coherence.py`:

1. For each question, average the two paired replicate differences
   (complete map minus matched control).
2. Weight the 20 questions equally to get the setting mean.
3. Compute a 95% percentile interval from 20,000 question-cluster bootstrap
   resamples. The seed is `991 + sum(map(ord, f"{trait}|full|coherence"))`.
4. Run a two-sided Wilcoxon signed-rank test (`zero_method="wilcox"`), with
   p = 1 when every question difference is zero.
5. Apply Holm adjustment over one 12-test family that spans all three models.
   This family is separate from the historical 24-test family.

Results: every interval includes zero, and every Holm-adjusted p equals 1.
Means range from -2.125 to +1.875.

## Question-level table

`plot_data/corrected_coherence_question_contrasts.csv` has one row per (model,
trait, question): `model`, `trait`, `question_index`, `n_pairs` (2), and
`mean_difference`. The last is the question-level paired mean that the
estimator bootstraps. The verifier checks this table's structure and recomputes
the reported contrasts. Raw responses, individual scores and provider records
are not included; the offline analysis uses these question-level means.

## Verification

Offline (CPU), from the repository root:

```bash
python experiments/corrected_coherence/source/verify_corrected_coherence.py
python -m pytest -q experiments/corrected_coherence/source
```

The command recomputes the 12 means, 95% bootstrap intervals, Wilcoxon p-values
and Holm-12 adjustment from the included question table. It also checks
bootstrap seeds, seven zero-width intervals and `change_in_contrast`, matching
recorded values within 1e-12. Input identifiers and historical means are checked
against the recorded metadata and Gemini coherence rows.

Saturation, identical-answer disagreements, cap sensitivity and arm means are
checked against saved summaries, not recomputed from individual judgments.
Those summaries report 872/960 scores at 100, seven all-100 settings, 195
identical question/answer pairs (two with score differences of 10 and 20), and
two length-capped Qwen-1.5B hallucination pairs whose removal changes 1.875 to
1.375. Recomputing these requires the original responses and score export.
The command labels these distinctions and exits nonzero on a failed check.

## Limits

- Neither the corrected judge nor the historical judge has human validation.
- Saturation is strong. The judge gave 100 to 872 of 960 answers, and seven
  settings are at ceiling in both arms, so their intervals have zero width.
- Identical inputs were not always scored identically at temperature 0. The
  intervals are conditional on one-shot recorded judgments and do not cover
  repeated-judge variability.
- The judge and the rubric changed together, so differences from historical
  Gemini coherence do not identify the effect of the rubric correction alone.
- Null-including intervals and Holm-adjusted p = 1 do not establish
  equivalence. No equivalence margin is claimed.
