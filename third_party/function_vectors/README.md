# Function Vector task data

Fetch the six Function-task selections used in the experiments from
`jorispos/ConceptorSteering` at commit
`5e134153f4eeefa631bda39788f1f996bbf052a1`. The download script selects and
validates the pairs used in the experiments.

To use the pairs locally, run from the repository root:

```bash
python experiments/broad_study/fetch_function_vectors.py
```

This writes the Git-ignored cache `experiments/broad_study/function_vectors.json`
after validating each task's pairs against the recorded selection digest.
Custom output paths must lie outside the repository; symlinks are refused.
Dataset attribution and licensing status are in the
[third-party notices](../../THIRD_PARTY_NOTICES.md).
