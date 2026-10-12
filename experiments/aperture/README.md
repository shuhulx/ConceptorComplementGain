# Generation-layer aperture sweep (geometric)

Computes how aperture changes directional gains and the matrix-specific
correction on held-out activations. The seven-aperture study uses
`alpha in {.025, .05, .1, .2, .5, 1, 2}` with `beta = 2`
(paper Sec. 3-4 and App. A).

## Use with activation tensors

`source/aperture_evidence.py` provides `complete_gain(eigenvalue, aperture, beta)`
and `heldout_correction_energies(train, test, apertures)`. The latter accepts
finite CPU floating tensors `[n_train, width]` and `[n_test, width]` with
nonempty dimensions. It returns held-out centered input energy and two
dictionaries of conceptor-correction energies, keyed by aperture: one computed
spectrally and one by a dual solve. Float16, bfloat16 and float32 inputs are
promoted to float64 before centering.

For complete/control comparisons on individual held-out rows, use the
[factorized activation API](../../docs/OWN_ACTIVATIONS.md). Dense helpers in
this directory provide small reference calculations.

## What is computed

`F_C(h) = m + beta (I - C)(h - m)` is the complete map,
`F_0(h) = m + beta (h - m)` the matched control, and
`F_C(h) - F_0(h) = -beta C (h - m)`.

- Each fit uses the 100 target-present rows at `blocks.<layer>.hook_resid_pre`
  (zero-based layers 9, 9, 10), centered by their own mean, with
  `R = X^T X / n` and `C = R (R + alpha^-2 I)^-1`. Per covariance eigenvalue
  `lambda`: `c = alpha^2 lambda / (1 + alpha^2 lambda)`,
  `g = beta / (1 + alpha^2 lambda)`; a direction is attenuated iff
  `alpha^2 lambda > beta - 1` (`c > 0.5` at `beta = 2`).
- **Full-data spectra:** 21 fits x 7 apertures = 147 rows (largest `c`,
  minimum gain). Covariance-null directions keep gain 2.
- **Held-out correction** (12 matched settings): folds from a NumPy
  `RandomState(42)` permutation of the 20 questions, five groups of four; each
  fold refits center and `C` on 80 rows and evaluates 20 held-out-question
  rows. The ratio `sqrt(sum ||beta C (h - m)||^2 / sum ||(beta - 1)(h - m)||^2)`
  pools energies over folds before the square root (84 pooled rows, 420 fold
  rows). It can exceed 100% and is below 200% at `beta = 2`. Rows are assigned
  to questions by the executed template-major convention (row `r` is question
  `r % 20`). Original tensors have no row identifiers for independently
  checking this mapping.

## Files

| Path | Role |
| --- | --- |
| `plot_data/aperture_spectra_21_fits.csv`, `plot_data/aperture_baseline_21_fits.csv`, `plot_data/aperture_correction_12_settings.csv`, `plot_data/aperture_correction_fold_evidence.csv`, `plot_data/aperture_summary.csv`, `plot_data/aperture_provenance.json` | Spectra, fold energies, pooled corrections and study settings |
| `experiments/aperture/source/verify_aperture_records.py` | Offline record verifier (NumPy only) |
| `experiments/aperture/source/aperture_evidence.py` | Directional gains and held-out correction functions for CPU tensors |
| `experiments/aperture/source/test_aperture_evidence.py`, `experiments/aperture/source/test_aperture_records.py` | Synthetic estimator tests; verifier tests on released and corrupted records |

## Commands

Offline check (CPU), from the repository root:

```bash
python experiments/aperture/source/verify_aperture_records.py
python -m pytest -q experiments/aperture/source
```

The verifier recomputes `c`, `g` and attenuation flags for all 147 rows and
all 84 pooled ratios from the 420 fold rows, checks fold coverage, summary
rows and the 200% bound, and checks agreement with
`plot_data/operator_cells.csv` and `plot_data/correction_cells.csv` at
alpha = .05.

Verification uses the recorded spectra and fold energies; refitting requires
the original extraction tensors, which are not included. The sweep measures
geometry at extraction states. Generated behavior at larger apertures and
autoregressive state/energy trajectories were not measured.
