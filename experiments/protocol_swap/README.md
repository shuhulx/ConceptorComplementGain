# Protocol-swap diagnostic

Tests whether the family-level SCR ordering between Function tasks and
assistant targets depends on how activations are collected: prompt-final
states, chat-generation averages, or the fully swapped construction. The
result is construction-sensitive. Function-task mean SCR exceeds
assistant-target mean SCR in all six pooling-matched comparisons, while the
full swap reverses the ordering in all three models. This does not isolate
target abstraction, answer space or the downstream metric, and SCR is not a
per-concept controllability predictor. Recorded one-sided exact Mann-Whitney
values are descriptive.

Control A averages the final-position state at every forward hook event during
generation, including prompt prefill. It is not an average restricted to newly
generated answer tokens.

## Results

- `results/protocol_swap_results.json`: recorded three-model output.
- `plot_data/protocol_swap_summary.csv`: summary checked against it by
  `test_protocol_summary_matches_full_results`.
- `plot_data/deep_validation_results.json`: the 13-concept spectral diagnostic
  supplying the original SCR values used in the protocol-matched comparison.

## Running with model access

Needs network access, the packages in
`experiments/protocol_swap/requirements.txt` (`torch>=2.10.0`), licensed
weights at a full Hugging Face commit, a CUDA GPU (float16) and the locally
fetched Function Vector text. From the repository root:

```bash
python experiments/broad_study/fetch_function_vectors.py
python experiments/protocol_swap/source/runner.py \
  --model qwen7b \
  --model-revision "<40-hex-commit>" \
  --output "<new path outside the repository>/protocol_swap_results.json" \
  --original-scr-json plot_data/deep_validation_results.json
```

The runner checks the output path, the revision format, the
`function_vectors.json` pair digests and the original-SCR file before any model
is loaded. `--output` must lie outside the repository. Per-model results
accumulate in that file; an existing model entry is never overwritten. The
read–merge–write transaction is protected by a process lock on a local POSIX
filesystem (Linux/macOS). A sibling
`.<output-filename>.lock` file remains beside the results; do not remove it
while writers may be running. Historical executions recorded no model-weight
commit, and the current runner adds input/output validation, so reruns may
differ from the recorded results.
The original-SCR comparison requires the explicit `--original-scr-json` option
shown above; omitting it does not automatically load the repository's reference.

The fetch command downloads Function Vector pairs from the pinned upstream
revision. See [dataset source and licensing status](../../third_party/function_vectors/README.md).
