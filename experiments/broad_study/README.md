# Broad study

Evaluates seven assistant targets under eight steering conditions with a
200-token generation limit, alongside Function-task parameter searches and
spectral summaries. Results are in `provenance/broad_study_results.json`.

`run_study.py` is a reconstruction of the three-model notebook protocol,
rather than the original executed notebook.
The [matched generation study](../matched_control/) uses a separate protocol.

## Protocol

The Function-task searches use seeds 42, 43 and 44. Each seed fits on 100
freshly sampled ICL prompts (nine completed pairs plus an unanswered query)
and evaluates all 32 task inputs in a seed-dependent order, not independent
test subsets. Each seed independently maximizes the conceptor search over
layer, aperture and gain, and the additive search over layer at coefficient
2.3. Reported means average the three per-seed maxima; standard deviations,
where present, use the population divisor. The matched generation protocol
is separate: question `q=0..19`, replicate `r=0,1`, seed `42+q+20*r`, reset
for each matched condition.

## Runner conventions

- Persona materials are loaded from `experimental_materials.json`. Function
  Vector pairs must be fetched from the pinned upstream revision.
- Function-task scores check whether the predicted next token starts with the
  first three characters of the expected answer, after case normalization and
  stripping prediction whitespace. They are not full-answer accuracy or persona
  outcomes. Family-level CCR comparisons are descriptive, not per-target
  controllability predictions.
- Checkpoints are bound to model revision, dtype, mode, layers, parameters,
  package versions, source and material identity; incompatible or partial
  checkpoints are rejected.
- Judge responses must be a single unsigned decimal in [0, 100]; a coherence
  `REFUSAL` is invalid. The trait `REFUSAL -> 0` mapping follows the broad
  study's `LEGACY_BROAD_STUDY_REFUSAL_TRAIT_SCORE` policy. The matched analysis
  handles refusals separately.
- Each successful judgment is saved to `checkpoints/persona_judgments.json`,
  bound to the run, exact responses, rubrics and evaluator settings. Resume
  retries missing or invalid scores without repeating successful judgments.
  Incomplete judging still blocks persona aggregates. Reported evaluator call
  counts cover the current invocation, including its self-test.
- The `effective_rank` key is kept for compatibility and labelled as the
  participation-ratio (Renyi-2) effective rank.

## Reuse the extraction helpers

`extract_activations.py` accepts an already loaded eval-mode TransformerLens-like
model and extracts one prompt-final residual row per prompt and layer. It
leaves gradient mode unchanged and does not reset the caller's RNG state.

```python
from extract_activations import last_token_resid_pre

# model is already loaded and in eval mode.
states = last_token_resid_pre(model, ["A short synthetic prompt."], layers=[4])
```

`states[4]` is a detached CPU tensor of shape `[1, d_model]`, with the cache's
dtype. Each string is run separately and its last prompt-token `resid_pre`
state is returned. Format chat prompts with the model's chat template before
passing them to this function.

`persona_generation_resid_pre` in `experiments/broad_study/extraction.py` preserves the persona runner's template-major,
question-major, positive-then-negative loop. Each row averages final-position
hook events while `generate` runs, including prompt prefill, with up to 50 new
tokens, temperature .7 and sampling enabled. It returns CPU float32 tensors.
This generated-event average and the prompt-final helper are different
activation constructions.

The optional `python -m examples.extract_activations --model ...` command loads
weights and prints prompt-final rows' shapes and norms.
See the [activation guide](../../docs/OWN_ACTIVATIONS.md).
Tests check prompt order, hook selection, pooling, dtype and cleanup. Run
specifications bind checkpoints to the extraction implementation. Original
tensors have no prompt-to-row identifiers for independently checking their
row mapping.

## Run the study

Needs `experiments/broad_study/requirements.txt` (`torch>=2.10.0`),
licensed weights at a full Hugging Face commit, a CUDA GPU (default
`--device cuda`, float16) and network access for the fetch. From the
repository root:

```bash
python experiments/broad_study/fetch_function_vectors.py
python -m experiments.broad_study.run_study \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --model-revision "<40-hex-commit>"
```

This runner retains dense conceptor matrices across targets, layers and
apertures, so host memory can be substantial in addition to model VRAM.
Its memory requirements differ from the reusable low-rank helpers; no peak
memory benchmark is provided.

The fetch writes the Git-ignored local cache
`experiments/broad_study/function_vectors.json` from the pinned upstream commit. Any
other output must lie outside the repository (symlinks refused), and nothing is
written unless each task's pairs match the executed broad-study
selection; `run_study.py` repeats the check when loading. Run outputs go to the
Git-ignored `experiments/broad_study/runs/` unless `--output-dir` is given. Set `GEMINI_API_KEY` only to use the historical
judge; without it, judge calls are skipped.

A keyless run cannot reuse a directory containing `checkpoints/persona_judgments.json`:
it stops before loading the model or writing outputs, preserving completed score
exports. To resume that run, set `GEMINI_API_KEY` and keep the same configuration;
bound scores are recovered and only missing judgments are requested. For an
unjudged run, choose a new `--output-dir` instead.

Model weights are pinned with `--model-revision`; the tokenizer is loaded by
TransformerLens and its revision forwarding was not verified. Some historical
weight revisions and the full historical environment were not recorded, so new
runs may differ from the recorded results. See
[environments and verification scope](../../docs/REPRODUCIBILITY.md) for the
tested environments and available checks.
