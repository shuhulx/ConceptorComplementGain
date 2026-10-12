# Paired-difference conceptor geometry (descriptive)

Compares the covariance spectra of target-present activations and paired
target-present minus target-opposed differences (paper Sec. 6 and App. D).
At diagnostic layers 26
(Qwen2.5-1.5B), 26 (Qwen2.5-7B) and 30 (Llama-3.1-8B), paired-difference
conceptors have higher Shannon effective rank and higher `k90` than
target-present conceptors in all 21 model-target comparisons. The result is
descriptive: it measures geometric differences at these layers, without a
significance test or a generated-behavior evaluation of a contrast-based
intervention. The matched generation panel uses different layers, 9/9/10.

## Use with paired activations

The functions in `source/paired_geometry.py` accept NumPy activation arrays
with observations in rows. `paired_difference(pos, neg)` requires aligned
`[n, width]` arrays and returns their row-wise differences. `spectral_metrics(rows)`
returns a dictionary including SCR, Shannon effective rank, `k90`, sample
count and feature width. `float64_metrics(rows)` returns effective rank and
`k90` from an independent float64 SVD calculation. Both metric functions use
aperture .05 by default.

## Estimator

- Each checkpoint holds 100 target-present (`pos`) and 100 target-opposed
  (`neg`) rows per layer, paired by stored row index. Original tensors have
  no prompt identifiers for independently checking this pairing.
- Fits: the 100 `pos` rows, and `pos[i] - neg[i]`. Rows are centered,
  `R = X^T X / n`, and the nonzero spectrum is read from the Gram matrix.
- Conceptor eigenvalues at `alpha = .05`: `c = lambda / (lambda + alpha^-2)`;
  `p_i = c_i / sum_j c_j`.
- Shannon effective rank `exp(-sum_i p_i log p_i)` (terms with
  `p_i <= 1e-15` dropped); `k90` is the smallest `k` with
  `sum_{i<=k} p_i >= .9`.
- Primary path in float32, as historically; an independent float64 SVD path
  cross-checks it.

## Files

| Path | Role |
| --- | --- |
| `plot_data/paired_delta_results.json`, `plot_data/paired_geometry_main.csv` | Recorded results (the CSV is the 21-row table derived from the JSON) |
| `experiments/paired_geometry/source/verify_paired_geometry_records.py` | Offline record verifier (NumPy only) |
| `experiments/paired_geometry/source/paired_geometry.py` | NumPy spectral functions for user-provided arrays; optional safe PyTorch loader |
| `experiments/paired_geometry/source/test_paired_geometry_metrics.py`, `experiments/paired_geometry/source/test_paired_geometry_torch.py`, `experiments/paired_geometry/source/test_paired_geometry_records.py` | Synthetic estimator, loader-safety and record tests |

## Commands

Offline check (CPU), from the repository root:

```bash
python experiments/paired_geometry/source/verify_paired_geometry_records.py
python -m pytest -q experiments/paired_geometry/source
```

The verifier checks the JSON structure and all 120 recorded fits (bounds,
ratios, float32-representability), recomputes group means and trajectory
entries, checks that the CSV equals the JSON, and checks the 21/21 statement.

The verifier uses the supplied tables. Refitting requires the original
extraction tensors, which are not included.
