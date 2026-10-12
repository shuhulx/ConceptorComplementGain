# Matched complete-map versus matrix-removed control

Compares the complete map `F_C(h) = m + beta (I - C)(h - m)` with
`F_0(h) = m + beta (h - m)`, keeping the center, gain, layer and sampling
conditions fixed. At a common input, their difference is `-beta C(h - m)`.

## Use with new models or scores

For new model comparisons, use the
[paired-generation API](../../docs/OWN_ACTIVATIONS.md#comparing-model-outputs)
with a loaded model, training activations and held-out prompts.

`source/runner.py` provides factorized operators, checkpoint-loading utilities
and paired statistical summaries. `paired_contrast_summary` accepts a DataFrame
containing `question_index` and a numeric difference column, plus keyword
arguments `endpoint`, `column`, `equivalence_margin`, `n_bootstrap` and
`bootstrap_seed`. It returns sample counts, an equal-question mean difference,
bootstrap confidence intervals and a Wilcoxon signed-rank test, or `None` if
there are no valid differences. Its margin fields describe numerical
compatibility, not evidence of behavioral equivalence. Corrected coherence
uses the separate estimator linked below.

## Design

This directory holds numerical helpers and aggregate results for the matched
generation study. The design has:

- three models and four targets (Sycophancy, Hallucination, Optimistic,
  Apathetic);
- twenty evaluation questions and two paired sampling replicates;
- four arms: baseline, complement at beta 1, complement at beta 2, and the
  matched matrix-removed control at beta 2;
- alpha .05, layers 9/9/10, temperature .7, top-p .95, and at most 1,024 new
  tokens.

That gives 1,920 answers and 480 complete/control pairs. The two beta-2 arms
use the same question, replicate, and seed. In file names and condition
columns, `not_b2` (`not2`) is the complete map at beta 2, `identity_b2`
(`identity2`) the matched control, `not_b1` (`not1`) the complement at beta 1,
and `baseline` the unsteered arm.

## Recorded execution environment

The generation environment of this 1,024-token panel (model revisions,
software versions, hardware, decoding, hooks and historical Gemini judge
settings) is recorded in the
[historical environment record](../../provenance/historical_environment.json),
including the exact Qwen2.5-1.5B-Instruct, Qwen2.5-7B-Instruct and
Llama-3.1-8B-Instruct model revisions.

The runs used float16 and torch 2.11.0+cu128. Original responses and per-answer
judgments are not distributed; the included tables support aggregate checks
rather than regeneration of the historical panel.

The instruction and question sets are in
[`experimental_materials.json`](../../experimental_materials.json), with
[Persona Vectors attribution](../../third_party/persona_vectors/CHANGES.md).

## Results and their status

The per-model directories under `results/` and `results/combined/` were
computed from Gemini 2.5 Flash judgments.

- **Target-expression rows** (`endpoint == "trait"`) are the source of the
  current matched target-expression endpoint. The paper's twelve contrasts
  correspond to the trait rows of `plot_data/gemini_behavior_cells.csv`.
- **Coherence rows** use the historical Gemini rubric, whose top anchor
  required "no hallucinations". They are **historical** and are not the current
  matched coherence endpoint. The current endpoint is the corrected
  DeepSeek-V4-Pro rescore of all 960 beta-2 answers in
  [`../corrected_coherence/`](../corrected_coherence/).
- Older margin and multiplicity fields are described in
  [reproducibility](../../docs/REPRODUCIBILITY.md); they do not establish equivalence.
- The offline count tests read these tables: exact output match (195 of 480
  pairs identical), refusals (11 complete map, 12 control), cap censoring (four
  answers at 1,024 tokens, two affected matched pairs) and data quality.
