# Experiment index

The central question is what the fitted conceptor matrix contributes beyond
the target-specific center and gain. The studies below distinguish operator
geometry, generated behavior and linear accessibility. Study guides specify
protocols, commands and required inputs; result-only directories contain recorded
outputs. [Results](../docs/RESULTS.md) maps the findings to their source tables.

For a new experiment on your own model, start with the
[extraction and paired-generation APIs](../docs/OWN_ACTIVATIONS.md#comparing-model-outputs).
For activation-only work, the aperture, paired-geometry and erasure guides
describe functions accepting your own tensors or arrays.

For CPU analysis of the included records, see the
[recomputation commands](../docs/REPRODUCIBILITY.md#cpu-tests-and-processed-results).
Model studies have separate [environment and input requirements](../docs/REPRODUCIBILITY.md#model-and-provider-runs).

## Main comparisons

| Study | Research question | Available offline | Additional execution requirements |
|---|---|---|---|
| [Matched control](matched_control/) | What changes when only the fitted matrix is removed? | Aggregates and operator tests | New comparisons use the public generation API with a loaded model, training states and prompts |
| [Aperture](aperture/) | When does the complete map attenuate? | Spectra, held-out correction records, numerical functions and verifier | Own CPU activation arrays for new geometric calculations |
| [Corrected coherence](corrected_coherence/) | How do the same saved responses compare under a coherence-only rubric? | 960-answer rescore summaries and question-level recomputation | No additional inputs for shipped analyses; original responses and scores are not distributed |
| [Erasure](erasure/) | Can a fresh linear probe recover polarity after an edit? | Eraser/probe functions, predictions and metric checks | Own activation arrays and labels; study-specific fold functions require the documented row grouping |

Target expression retains the original trait judgments; corrected coherence
is a separate endpoint. Null-including intervals do not establish equivalence.
The aperture and erasure studies answer geometric and accessibility questions,
not whether a new intervention suppresses generated behavior.

## Supporting and retrospective analyses

| Study | Research question | Available offline | Additional execution requirements |
|---|---|---|---|
| [Paired geometry](paired_geometry/) | How does fitting paired differences change the spectrum relative to target-present rows? | Descriptive results at layers 26/26/30; record verifier and tests | Own paired arrays for the metric functions; original tensors are not distributed |
| [Independent judge](independent_judge/) | How sensitive were selected comparisons to a secondary evaluator? | Earlier 240-answer sensitivity results and inventory tests | Original evaluator runner, raw responses and judgments are not included |
| [Finite alternatives](finite_alternatives/) | How do purification, Fisher and GEVD directions compare with raw mean-difference directions in the recorded settings? | Historical aggregates, record verifier and tests; historical-rubric coherence | None for shipped-table verification; original generation runner not included |
| [Function-task tables](semantic_functions/) | What outcomes were recorded in the broad-study Function-task, persona and inversion analyses? | Retrospective tables rebuilt by `scripts/verify_semantic_rebuild.py` | None for the table rebuild; this does not rerun the models |
| [Protocol swap](protocol_swap/) | Does the family-level Spectral Concentration Ratio (SCR) ordering depend on activation construction? | Results and summary-consistency test | Weights, CUDA GPU and fetched Function Vector text; see guide for differences from the recorded execution |
| [Spectral screen](spectral_screening/) | Is the family-level ordering shared by other spectral summaries? | Descriptive records, two-sided analysis and primitive/scope tests | Included summaries, or a user broad-study checkpoint and manifest (separate modes) |
| [Broad study](broad_study/) | What did the seven-target study and Function-task searches measure? | Notebook-protocol reconstruction, checkpointing and evaluation tests | Weights, CUDA GPU and fetched Function Vector text; see guide for historical execution limits |

## Additional records

**`independent_judge/`** summarizes an earlier DeepSeek-V4-Pro evaluation of
240 answers (60 complete/control pairs) with historical multi-endpoint rubrics
(`deepseek_judge_prompts.json`, `secondary_evaluator_selection.csv`). It is an
evaluator-sensitivity check only (selection limits in
[reproducibility](../docs/REPRODUCIBILITY.md#interpreting-the-results)), and its margin fields
do not establish equivalence.

`semantic_functions/` holds Function-task and persona tables that
`scripts/verify_semantic_rebuild.py` rebuilds from the consolidated broad-study JSON.
The spectral screen's historical one-sided significance flags are not calibrated
inference; the [guide](spectral_screening/) explains the two-sided analysis.
