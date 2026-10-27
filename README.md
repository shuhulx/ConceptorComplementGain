# When a Conceptor Complement Becomes Gain: Operator Geometry and Behavioral Attribution in Language Models

[![arXiv:2610.16325](https://img.shields.io/badge/arXiv-2610.16325-b31b1b.svg)](https://arxiv.org/abs/2610.16325)
[![Code: Apache 2.0](https://img.shields.io/badge/code-Apache%202.0-blue)](LICENSE)

**Shuhul Razdan and R Balamurugan** · NeurReps 2026, a NeurIPS workshop.

Code and processed results for our study of scaled conceptor complements in
language models. We analyze how scaling changes their geometry and compare
the complete intervention with a control that removes the fitted conceptor
matrix while retaining the same center and gain. This separates the matrix's
contribution from the rest of the intervention.

The code provides centered, factorized conceptor fitting, directional gains,
held-out activation comparisons and paired complete/control generation.
Study-specific analyses cover aperture, paired-difference geometry, linear
erasure and spectral measurements. The repository also includes processed
results, extraction instructions and evaluation prompts.

## Installation

Use Python 3.12. Clone the repository and create a virtual environment:

```bash
git clone https://github.com/shuhulx/ConceptorComplementGain.git
cd ConceptorComplementGain
python3.12 -m venv .venv
source .venv/bin/activate
```

For CPU use on Linux:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0
python -m pip install -r requirements.txt
```

On macOS, replace the PyTorch installation command with
`python -m pip install torch==2.13.0`. For CUDA, install a compatible CUDA build
of PyTorch using the [official installation selector](https://pytorch.org/get-started/locally/)
instead of the CPU wheel, then install `requirements.txt`. The optional model
helpers require PyTorch >=2.10.0 and `examples/requirements.txt`.
Study-specific model runs have separate environments listed in their guides.
Run the commands below from the repository root.

## First example

```bash
python -m examples.synthetic_complement_gain
```

This CPU example fits an anisotropic synthetic dataset and prints each
direction's complete-map and matched-control gain, the number of attenuated
directions and the held-out correction ratio. It also checks the operator
identity numerically.

## Compare your own activations

Supply target-present training observations `train` with shape `[n, d]` and
independent `heldout` states `[n_test, d]`, both CPU tensors from the same layer
and extraction protocol. Choose aperture separately from the evaluation split.

```python
from eval.compare_activations import compare_activations

result = compare_activations(train, heldout, aperture=0.05, beta=2.0)
complete, control = result.complete, result.control
print(result.correction_rms, result.correction_ratio)
```

`result` contains the fitted conceptor, transformed tensors, common-input
correction and RMS diagnostics. `correction_ratio` is a
fraction, or `None` for zero control displacement. The
[activation guide](docs/OWN_ACTIVATIONS.md) explains fitting, directional gains
and interpretation.

## Extract activations and compare generated responses

Install the optional model dependencies in your CPU or CUDA environment:

```bash
python -m pip install -r examples/requirements.txt
```

Supply a loaded TransformerLens model on CPU or CUDA, a valid zero-based
block index `layer`, target-present `train_prompts` and held-out `eval_prompts`.
Both prompt collections are lists of strings formatted with the model's chat
template when applicable. Extract one prompt-final training row per prompt,
then compare both interventions using the same sampling seeds:

```python
import torch

from extract_activations import last_token_resid_pre
from eval.compare_generations import compare_generations

model.eval()
with torch.inference_mode():
    train = last_token_resid_pre(model, train_prompts, [layer])[layer]
pairs = compare_generations(
    model, train, eval_prompts, layer=layer, seeds=(42, 43),
    aperture=0.05, beta=2.0, max_new_tokens=1024,
)
```

`train` has shape `[len(train_prompts), model.cfg.d_model]` on CPU.
`pairs` is a list of `GenerationPair(prompt, seed, complete, control)` records,
one per prompt and seed. Complete/control strings include the prompt; use the
records in your saving and scoring workflow, distinguishing prompt text from
the generated continuation.
Both arms intervene on all tokens at `blocks.<layer>.hook_resid_pre`.
See the [generation guide](docs/OWN_ACTIVATIONS.md#comparing-model-outputs)
for extraction, paired decoding and numerical considerations.

## Specialist analyses

- [Aperture](experiments/aperture/): directional gains and held-out correction
  energies across apertures, using CPU activation tensors.
- [Paired geometry](experiments/paired_geometry/): spectra, Shannon effective
  rank and 90%-mass dimension for row-aligned NumPy activation arrays.
- [Erasure](experiments/erasure/): conceptor, LEACE and INLP helpers with
  separate eraser-fit, fresh-probe and test splits.
- [Spectral screening](experiments/spectral_screening/): spectral primitives
  and the study-specific comparison of Function tasks with assistant targets.

The [experiment index](experiments/README.md) covers the remaining studies,
including matched controls, coherence scoring and the broad-study runner.
Each guide states its input requirements, outputs and protocol assumptions.

## Recompute included analyses

These CPU commands use the included results:

```bash
python scripts/verify_semantic_rebuild.py
python experiments/aperture/source/verify_aperture_records.py
```

The first rebuilds Function-task and persona tables in a temporary directory
and compares them with the included tables. The second recomputes gains and
correction ratios from recorded spectra and fold energies. Both print their
checks to the terminal. See the
[full command list](docs/REPRODUCIBILITY.md#cpu-tests-and-processed-results)
and [results guide](docs/RESULTS.md).

These commands recompute processed results. Original responses, per-answer
judgments and extraction tensors are not included; see the
[reproducibility guide](docs/REPRODUCIBILITY.md#inputs-not-distributed)
for input availability and new model runs.

## Documentation

- [Activation and generation guide](docs/OWN_ACTIVATIONS.md): APIs, equations and numerical conventions.
- [Paper experiments](experiments/README.md): study-specific execution instructions.
- [Results](docs/RESULTS.md): findings and source tables.
- [Reproducibility](docs/REPRODUCIBILITY.md): environments, inputs and verification.

## Citation

```bibtex
@InProceedings{razdan26,
  title = {{When a Conceptor Complement Becomes Gain:
            Operator Geometry and Behavioral Attribution in Language Models}},
  author = {Razdan, Shuhul and Balamurugan, R},
  booktitle = {Symmetry and Geometry in Neural Representations (NeurReps 2026)},
  year = {2026}
}
```

Code: [Apache-2.0](LICENSE). Project-authored documentation and processed data:
[CC BY 4.0](LICENSE-DATA). See [license scope](LICENSE.md) and
[third-party notices](THIRD_PARTY_NOTICES.md) for upstream materials.
