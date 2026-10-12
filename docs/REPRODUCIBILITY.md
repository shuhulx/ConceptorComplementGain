# Reproducibility

This repository accompanies **When a Conceptor Complement Becomes Gain:
Operator Geometry and Behavioral Attribution in Language Models**, by
Shuhul Razdan and R Balamurugan, NeurReps 2026 (Symmetry and Geometry in Neural
Representations). The paper is not included.

## CPU tests and processed results

Install the [root requirements](../requirements.txt) using the [README](../README.md).
CI uses Python 3.12 on Ubuntu; local tests also run on macOS. Native Windows is untested.

```bash
python -m ruff check
python -m pytest -q
```

These commands need no model weights, GPU or provider credentials. Pytest also
runs five record-check programs in separate subprocesses:

```bash
python scripts/verify_semantic_rebuild.py
python experiments/aperture/source/verify_aperture_records.py
python experiments/corrected_coherence/source/verify_corrected_coherence.py
python experiments/paired_geometry/source/verify_paired_geometry_records.py
python experiments/finite_alternatives/source/verify_finite_alternatives.py
```

The checks have three scopes:

- **Recalculation:** gains and attenuation flags from eigenvalues; pooled
  correction ratios from 420 fold-energy rows; corrected-coherence means,
  question-cluster bootstrap intervals, Wilcoxon tests and Holm-12 adjustments;
  semantic tables from the broad-study JSON; and finite-alternative joint scores
  from aggregate means. Semantic CSV rebuilds match exactly; correlation fields
  use a `1e-12` tolerance.
- **Consistency:** counts, endpoint definitions, material inventories and
  agreement between files. Erasure tests also recompute point metrics from
  included held-out predictions.
- **Recorded diagnostics:** coherence saturation, identical-input judge
  disagreement and cap sensitivity. These checks compare aggregate summaries;
  recalculation would require the undistributed responses and per-answer scores.

Synthetic tests cover fitting, extraction, erasure, checkpoint validation and
estimators. The separate TransformerLens CI job uses a tiny random CPU model
and local tokenizer, and fails if the required integration dependencies are
unavailable. Together these checks exercise the code and processed analyses.
[RESULTS.md](RESULTS.md) locates the records, and
[OWN_ACTIVATIONS.md](OWN_ACTIVATIONS.md) describes new activation and model
comparisons. Original input availability is described below.

## Interpreting the results

The matched panel contains 1,920 answers over four conditions. Its beta-2
complete/control comparison has 960 answers, or 480 pairs. Target expression
uses the original Gemini 2.5 Flash trait scores in
`plot_data/gemini_behavior_cells.csv`. Current coherence uses the corrected
DeepSeek-V4-Pro coherence-only rescore in
`plot_data/corrected_coherence_results.json`. The earlier 240-answer,
60-pair secondary evaluation is a separate sensitivity check: selection took
place after the original evaluation was complete, used a seed derived from those records, and is not a
prospective or random sample of the current corrected panel.

The four-target comparison was exploratory. Judgments have no human
validation, and corrected coherence is saturated (872/960 scores are 100).
Intervals containing zero do not establish equivalence. Larger-aperture
matched behavior and autoregressive telemetry were not run. Held-out
correction ratios describe extraction states, not generation-time trajectories.
LEACE/INLP results measure fresh linear-probe accessibility, with no generation
after erasure.

The records distinguish several endpoints and statistical conventions:

- Gemini coherence used a rubric whose top anchor required “no hallucinations.”
  Finite-alternative and broad-study coherence retain that historical rubric.
- Margin and `claim_status` fields use an uncalibrated margin of 10; they are
  not equivalence evidence. `ceiling_limited` denotes all-zero or all-100 arms.
- Historical within-model/table Holm families and global Holm columns differ
  from the current twelve-test corrected-coherence family.
- CCR, SPR and SCR are author-defined labels for descriptive trace summaries,
  not per-target controllability predictors. Broad-study `EffRank` means
  participation-ratio (Rényi-2) effective rank, not the
  Shannon effective rank used for paired geometry.
- Broad-study `mann_whitney` and spectral-screen one-sided p-values use post hoc
  directions, so their significance flags are not calibrated inference. With no
  ties, complete six-versus-seven ordering gives exact two-sided p = .0011655,
  above .05/48. Tied measures use SciPy's tie-corrected asymptotic test.
- `plot_data/deep_validation_results.json` contains the 13-concept spectral
  diagnostic and original SCR values, not finite-alternative results.
  The finite-alternative joint score `S = (b - t)a/100` is computed from
  `plot_data/finite_alternatives_scores.csv`.
- Broad-study aperture sweeps use analysis layers, not the current generation-layer
  sweep at 9/9/10. Its persona tables are from the broad 200-token study,
  which has no matched control.

Short labels: `not_b2`/`not2` is the complete map at gain 2,
`identity_b2`/`identity2` the matched control, `not_b1`/`not1` the
gain-1 complement, and `baseline` the unsteered arm. `prefix200` judges
the first 200 tokens; `full` judges the whole answer, capped at 1,024 tokens.
`1.5b`, `7b` and `8b` denote the two Qwen2.5 models and Llama-3.1-8B.
“Semantic” denotes the Function Vector tasks.

## Experiment protocols and recorded environments

The numerical measurements in `provenance/broad_study_results.json` came
from the original notebooks. The [broad-study runner](../experiments/broad_study/)
reconstructs that protocol. Semantic seeds 42, 43 and
44 each refit both methods from 100 resampled ICL prompts (nine example pairs
plus a query). Each seed evaluates the same 32 task items in a different order.
Scores average per-seed maxima: conceptors search layer, aperture and gain;
addition searches layer at fixed coefficient 2.3. This is an unequal,
retrospective search with a case-insensitive next-token prefix scorer, not
full-answer exact match. The broad 200-token study and the 1,920-answer matched
panel are separate experiments.

The aperture, paired-geometry and erasure modules expose numerical functions for
arrays and recorded tables. The exact source executed for the erasure results
is unavailable, which limits exact reproduction of those fits.
The corrected-coherence estimator recomputes contrasts from the shipped
question-level table, without a provider call.

Saved tensors lack per-row prompt identifiers, so they cannot independently
confirm which prompt produced each row. The original broad-study extraction runs
also lack complete package locks and some model revisions; protocol-swap
results record no model revision. The matched panel's recorded environment is
in `provenance/historical_environment.json`.

## Inputs not distributed

Raw generated responses, per-answer judgments, provider logs, extraction tensors,
model weights and run checkpoints are not included.

Fetch Function Vector pairs locally with
`python experiments/broad_study/fetch_function_vectors.py`; it writes to the
Git-ignored cache or an external path. See
[third-party notices](../THIRD_PARTY_NOTICES.md) for dataset sources and terms.

The CPU analyses above use shipped records. New activation comparisons accept
user-supplied arrays, while model studies require the inputs described below.
The omitted original inputs prevent a complete historical rerun.

## Model and provider runs

Install optional runner requirements in separate environments; they are not
combined lockfiles. Validation covers the CPU checks described above, not full
GPU or provider reruns.
The model helpers use the TransformerLens `HookedTransformer` API. Their
[optional requirements](../examples/requirements.txt) pin the tested 3.5.1 stack,
including datasets 4.3.0 for compatibility with huggingface-hub 1.23.0.
The optional stack requires PyTorch >=2.10.0; CPU tests use 2.13.0 and historical matched runs
used 2.11.0. Gated model weights require the appropriate Hugging Face access.
Keep credentials in environment variables or hidden prompts, never in source.
Do not commit credentials, responses, provider logs, tensors or checkpoints.

| Runner | Requirements |
|---|---|
| [Broad study](../experiments/broad_study/) | CUDA GPU, licensed weights pinned by required `--model-revision`, locally fetched Function Vector text. Optional hosted judging reads `GEMINI_API_KEY`; without it, judging is skipped. |
| [Protocol swap](../experiments/protocol_swap/) | CUDA GPU, pinned `--model-revision`, locally fetched pair text, external `--output`; use `--original-scr-json plot_data/deep_validation_results.json` for the protocol-matched summary. |
| `examples/extract_activations.py` | TransformerLens and a supported model. The CLI loads weights; imports, `--help` and `extract_from_model(model, ...)` do not. `--revision` optionally pins a full Hub commit. |

The broad-study runner defaults to ignored `experiments/broad_study/runs/`.
Protocol-swap outputs must be outside the repository and may accumulate
distinct model entries; spectral-screen outputs must be new external files.
Input and checkpoint validation checks compatibility before resuming a run.
