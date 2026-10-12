# Fresh-probe linear accessibility

This study asks whether a newly fitted linear probe can recover target-present
versus target-opposed polarity after an activation edit. Erasers and fresh
probes are fitted on separate activation subsets and evaluated on held-out
rows. The outcome is linear accessibility, rather than generation after
erasure or absence of all target information.

## Using the numerical functions

`source/runner.py` provides eraser, fresh-probe, split, metric and checkpoint
validation functions. `fit_leace_binary(x, y)` accepts a finite NumPy matrix
`x` of shape `[n, width]` and binary labels `y` of shape `[n]`; it returns a
transform callable and fit diagnostics. `fit_conceptor_not(x_pos)` accepts
target-present rows and returns a centered-complement transform and spectral
diagnostics. Apply either transform to rows with the same feature width.

Fitting promotes real-valued inputs to float64 before centering and SVD;
invalid or nonfinite arrays are rejected. The explicit seeds in
`source/random_control_seeds.json` specify the study's random-control draws.

`run_fold(pos, neg, model, trait, scheme, fold, smoke)` implements the study's
three-way split described below. Each polarity must have exactly 100 rows
with matching, nonempty feature dimensions and finite values in template-major
order. It returns method, prediction and validation rows. Use a different split
for arrays with different grouping.

## Design

Each of the three models and seven targets has 200 rows: five instruction
pairs, twenty questions and two polarities. The residual-stream layers are
zero-based blocks 9, 9 and 10 for Qwen2.5-1.5B-Instruct, Qwen2.5-7B-Instruct
and Llama-3.1-8B-Instruct, respectively.

Five folds rotate disjoint roles: 120 balanced rows fit the eraser, 40 select
and fit a fresh L2-regularized logistic probe, and 40 are the untouched test
set. The primary split groups questions across all templates; the sensitivity
split holds out instruction-pair templates. Probe regularization is selected
by question-grouped cross-validation within the probe-fit rows in both schemes.
Grouping assumes the recorded row order `row = template_id * 20 + question_id`
within each polarity. Saved tensors lack per-row prompt identifiers, so that
assignment cannot be independently checked from the tensor alone.

The full sweep includes raw activations; the centered finite conceptor
complement at aperture .05 and beta 1; binary rank-1 LEACE; and INLP at ranks
1, 2, 4, 8, 16 and 20. Random controls remove rank 1 or 20, with three draws
within the eraser-fit empirical row space per fold. They match removed rank
and row-space support, not edit magnitude. Balanced accuracy, AUROC and log
loss measure accessibility; normalized edit MSE measures distortion.

For test states `before`, edited states `after` and the eraser-fit mean `mean`,
normalized edit MSE is `sum((after - before)**2) / sum((before - mean)**2)`.
Energy retention is the uncentered ratio `sum(after**2) / sum(before**2)`.
Both are computed within each fold before averaging. The numerical helpers
return zero when the denominator's magnitude is at most `1e-30`.

## Check the shipped records

From the repository root, with the [CPU requirements](../../docs/REPRODUCIBILITY.md):

```bash
python -m pytest -q experiments/erasure/source tests/test_reported_numbers.py
```

The source tests use synthetic fixtures to check formulas, split isolation,
numerical transforms and checkpoint validation. `tests/test_reported_numbers.py`
also recomputes aggregate point metrics from the shipped held-out predictions,
fold means and paired point contrasts, and checks role coverage and validation
ledgers. The complete offline command is `python -m pytest -q`.

`results/primary/` and `results/template_sensitivity/` contain predictions,
fold and aggregate results, validation rows and experiment settings and counts.
For template-split findings, use `aggregate_results.csv`: the original
`key_findings.json` and its run-manifest copy are empty and labeled
`question_grouped`. The current summary function selects `template_grouped`
when called with `mode="template"`. `verify_key_findings` rejects the old empty
summary by default; record checks allow it with a warning and check the template
numerical tables separately.
`results/paired_secondary/` contains the direct LEACE/random-1 and
INLP-20/random-20 question-cluster contrasts. The CPU tests above do not recompute
this secondary analysis. Its question-cluster analysis functions are in
`source/paired_eraser_random_analysis.py`.

The exact executed runner is unavailable, and the original tensors are not
included. The functions here support new activation analyses, while the shipped
predictions support recalculation of the reported point metrics.
