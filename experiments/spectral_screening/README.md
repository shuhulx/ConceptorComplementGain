# Spectral-screening analysis

Compares 48 spectral summaries between six
Function tasks and seven assistant targets. Its current use is descriptive.

- 21 measures completely order the two families for Qwen2.5-1.5B-Instruct.
- 8 of the 13 measures available on all three models do so on every model.

Many of these measures are monotone transforms or algebraic combinations of the
same spectral moments. Several statistics capture a concentration difference
between these fitted samples under the original activation construction;
the result is not unique to SCR.

## Recorded results and inference

`results/formula_screening_results.json` and
`plot_data/formula_screening_results.csv` contain the recorded outputs. Their one-sided
Mann-Whitney comparisons were oriented after seeing which family had the larger
mean, so the recorded p-values and Bonferroni markers describe that analysis
rather than calibrated significance (see
[`../../docs/REPRODUCIBILITY.md`](../../docs/REPRODUCIBILITY.md)).

## Statistical conventions in the analysis code

These conventions apply to new analyses. The original tables retain their
recorded values, effect sizes, labels and ordering counts.

1. **Two-sided inference.** `source/formula_screening.py` uses a two-sided
   Mann-Whitney test. The larger-family orientation is used only as a
   descriptive orientation of U. For a complete six-versus-seven ordering with
   no ties, the exact p-value is `0.0005827506` one-sided and `0.0011655012`
   two-sided. This no-tie exact value exceeds the Bonferroni threshold
   `0.05/48 = 0.0010416667`. With ties, SciPy's automatic method uses an
   asymptotic calculation that can fall below this threshold; new output rows
   identify the method in `mann_whitney_method`. The frozen one-sided indicators
   are not corrected two-sided results. The complete-ordering counts above are
   rank statements.
2. **Effective rank.** The `EffRank` field imported from the broad-study
   summaries is the participation-ratio (Renyi-2) effective rank,
   `(sum_i c_i)^2 / sum_i c_i^2`. It is not the Shannon effective rank. The
   formula-screen row, and the product/ratio rows that reuse it, refer to
   participation-ratio effective rank. The separate paired-difference geometry
   uses Shannon effective rank.

3. **Steering associations.** `steering_nominal_p_lt_05_descriptive` counts
   nominal steering-association p-values below .05. This descriptive summary
   does not establish incremental information or corrected significance.

4. **Degenerate effects.** Hedges' g is undefined when both groups have zero
   variance. New outputs use `null` and `hedges_g_status`, distinguishing equal
   constants from a positive or negative difference.

5. **Spectral labels.** The retained `Stable_Rank` key measures trace divided by
   the largest eigenvalue, not the usual squared-Frobenius/operator stable rank.
   `MP_Fraction` is a descriptive threshold heuristic,
   `var(c_positive) * (1 + sqrt(n/d))^2`; it is not a validated covariance-noise
   edge. Here `n` is the actual fit's sample count and `d` its activation width.
   The historical variance fallback of one for fewer than three positive
   eigenvalues is retained.
   The descriptive name is **fraction above spectral-variance threshold
   (heuristic)**, stored under the original label "Fraction above
   Marchenko-Pastur edge" and the `MP_Fraction` key.

The corrected output has no default location. `--output` is required and must
name a new file outside the repository; paths inside the repository (symlinks
resolved) and existing files are refused, so a rerun cannot overwrite the
frozen result.

## Analyse a new extraction checkpoint

Computing all 48 measures requires a broad-study extraction checkpoint and its
`run_manifest.json`, supplied by the user from a [broad-study run](../broad_study/README.md).
The checkpoint includes the six Function tasks and seven persona targets for
Qwen2.5-1.5B-Instruct. This mode computes all 48 measures from that checkpoint
alone, including the four summary moments and SCR correlations. It does not
load historical summaries or behavioral outcomes. Steering associations and
cross-model comparisons are therefore unavailable, not zero. The analysis:

- validates the manifest's model, source/material identity, and payload
  identity;
- requires the checkpoint's expected SHA-256 on the command line;
- checks the exact 6+7 target inventory;
- selects each target's peak layer by top-ten conceptor eigenvalue mass and
  records its sample count and dimension;
- records all input hashes in its output.

Sample counts and peak-layer selection affect the descriptive family comparison;
these tests are not confirmatory evidence for a prespecified population effect.

```bash
python experiments/spectral_screening/source/formula_screening.py \
  --extractions "<your extractions.pt>" \
  --extractions-sha256 "<64-character-sha256>" \
  --run-manifest "<your run_manifest.json>" \
  --output "<new path outside the repository>/formula_screening_results_corrected.json"
```

Install `experiments/spectral_screening/requirements.txt` first
(`torch>=2.10.0`). See [verification scope](../../docs/REPRODUCIBILITY.md) for
the tested environment and available checks.

## Reanalyse the included summary tables

This separate mode evaluates the 13 summary-derived measures using the included
three-model tables and their next-token prefix scores. It needs no activation
checkpoint. Full-spectrum measures cannot be reconstructed from these summaries.

```bash
python experiments/spectral_screening/source/formula_screening.py \
  --mode historical-summaries \
  --output "<new path outside the repository>/summary_screen.json"
```

Both modes retain the original 48-measure Bonferroni comparison family, a
conservative choice for the 13-measure historical subset. They
must not be combined to imply a same-run replication of the original full
screen. `--consolidated-json` is accepted only in `historical-summaries` mode and
requires the consolidated three-model schema, not the broad runner's new-run
export schema.

The checkpoint-free numerical primitives (NumPy, SciPy and PyTorch) are in
`source/spectral_metrics.py`. Their synthetic tests run under
`python -m pytest -q`:

```bash
python -m pytest -q experiments/spectral_screening/source/test_spectral_metrics.py
```

`plot_data/deep_validation_results.json` is a separate 13-concept spectral
diagnostic used by the protocol-swap comparison, not an output of this screen.
