# Using your own activations

Fit on training observations, then compare both maps on held-out states.
For an executable starting point after the [CPU installation](../README.md#installation):

```bash
python -m examples.synthetic_complement_gain
```

This synthetic example prints directional gains and held-out correction
diagnostics. To use your own data, supply `train` and
`heldout` CPU tensors from one layer and extraction protocol:

```python
from eval.compare_activations import compare_activations

result = compare_activations(train, heldout, aperture=0.05, beta=2.0)
print("correction RMS:", result.correction_rms)
print("correction / matched displacement:", result.correction_ratio)
gains = result.fitted.directional_gains(beta=2.0)
print("basis gains:", gains.represented)
print("remaining dimensions:", gains.orthogonal_nullspace_dimension)
print("gain in those dimensions:", gains.orthogonal_nullspace_gain)
```

Split related questions/templates together. Fit the center and conceptor only
on training rows, and choose aperture on a separate validation split.

| Input | Shape | Meaning |
|---|---|---|
| `train` | `[n, width]` | Sampled activations used to fit the center and covariance |
| `heldout` | `[..., width]` | States evaluated without refitting; a single `[width]` vector is allowed |

Use sampled activations from a single layer, rather than saved mean-difference
vectors. A single training row has zero centered covariance; multiple
observations are needed to estimate the geometry.

## Comparing model outputs

Use an already loaded, eval-mode TransformerLens model. The optional model
stack, separate from CPU analysis dependencies, is installed with:

```bash
python -m pip install -r examples/requirements.txt
```

The code below assumes `model`, a zero-based block index `layer`,
target-present `train_prompts`, and independent `eval_prompts`, both lists of
strings. Format both sets using the same model chat template and tokenization
convention. Extraction returns one prompt-final residual row per prompt and
layer in prompt order; `train` below has shape `[len(train_prompts), d_model]`.

```python
import torch
from conceptor_tools import fit_conceptor
from extract_activations import last_token_resid_pre
from eval.compare_generations import compare_generations

model.eval()
with torch.inference_mode():
    train = last_token_resid_pre(model, train_prompts, [layer])[layer]
fitted = fit_conceptor(train, aperture=0.05)
pairs = compare_generations(
    model, fitted, eval_prompts, layer=layer, seeds=(42, 43),
    beta=2.0, max_new_tokens=1024, temperature=0.7, top_p=0.95,
)
```

`pairs` contains a `GenerationPair(prompt, seed, complete, control)` per prompt
and seed, ordered by prompt and then seed. The complete/control strings include
the prompt. Use these records for downstream saving and scoring, separating
the generated continuation from prompt text consistently in both arms. Define
the evaluation set and rubric in advance, score both arms without condition
labels, and report target expression, coherence and uncertainty separately.
Model loading and revision, weight processing, device and dtype remain the
caller's choices. Pin and record them for an experiment. The broad-study runner uses
`fold_ln=False`, `center_writing_weights=False` and `center_unembed=False`;
TransformerLens default weight processing is not the same representation.

The reusable entry points are:

| Module | Function and return value |
|---|---|
| [extract_activations.py](../extract_activations.py) | `last_token_resid_pre(model, prompts, layers)` → CPU `[n_prompts, width]` tensors keyed by layer |
| [conceptor_tools.py](../conceptor_tools.py) | `fit_conceptor(train, aperture)` → fitted center, economy basis and conceptor coefficients |
| [activation_steer.py](../activation_steer.py) | `make_matched_hooks(fitted, beta=...)` → `complete` and `control` hook callables |
| [eval/compare_activations.py](../eval/compare_activations.py) | `compare_activations(train, heldout, ...)` → transformed states, correction and RMS diagnostics |
| [eval/compare_generations.py](../eval/compare_generations.py) | `compare_generations(model, fitted, prompts, layer=..., seeds=...)` → list of paired-generation records |

Extraction selects the final token at `blocks.<layer>.hook_resid_pre`;
intervention transforms **all tokens** at that residual-input hook, including
the prompt and decoding calls. Both inherit model BOS defaults and use the
formatted strings supplied by the caller. Paired RNG preservation supports
CPU and CUDA, not MPS.

Each pair starts from the same Torch seed, but the two generations can diverge
as decoding progresses. A small correction at a common activation does not
establish a small behavioral effect.

For persona studies, target-present rows, positive-minus-negative rows and
saved mean-difference vectors are different inputs. Choose the construction
before fitting. The paper's [broad-study protocol](../experiments/broad_study/) averages
generation-event states; [matched generation](../experiments/matched_control/)
uses its own paired seeds and extraction fits. Record the chosen extraction
construction, layer, prompts, seeds, aperture, gain and decoding settings
alongside your outputs.

## Interpreting the measurements

For training observations $x_j\in\mathbb{R}^d$, let $X_c$ contain rows
$x_j-m$. The training mean, covariance and conceptor are

$$
m=\frac{1}{n}\sum_{j=1}^{n}x_j,\qquad
R=\frac{X_c^\top X_c}{n},\qquad
C=R(R+\alpha^{-2}I)^{-1}.
$$

The covariance uses divisor $n$, not $n-1$. Aperture $\alpha>0$ controls
the filter coefficients. Fitting uses an economy SVD of $X_c$; for singular
value $s_i$, covariance eigenvalue $\lambda_i=s_i^2/n$ and gain $\beta>0$,

$$
c_i=\frac{\lambda_i}{\lambda_i+\alpha^{-2}},\qquad
g_i=\beta(1-c_i)=\frac{\beta}{1+\alpha^2\lambda_i}.
$$

The complete intervention and matrix-removed control use the same fitted
mean and gain:

$$
F_C(h)=m+\beta(I-C)(h-m),\qquad F_0(h)=m+\beta(h-m).
$$

At a common input, $F_C(h)-F_0(h)=-\beta C(h-m)$, the correction measured
here. The control retains centering and gain; it is not an unsteered baseline.
Scaling can make the complete map expand even though the complement alone
never expands. This identity does not determine differences between evolving
generation trajectories.

Correction RMS pools all held-out elements. Its ratio to
`RMS((beta - 1)(h - m))` can exceed one. At `beta=1`, or zero matched
displacement, `correction_ratio` is `None`; `ratio_undefined_reason` explains why.
`algebra_max_abs_error` checks the complete-minus-control identity.

Complete gain is `beta * (1 - c_i)`. Below one means attenuation around the
training center. Directions outside the economy basis have gain `beta`, as
do zero singular values inside it; represented gains alone may omit part of
the ambient spectrum. These quantities describe geometry, not behavioral quality.

## Precision and model state

CPU fitting accepts finite strided float16, bfloat16, float32 and float64 tensors;
arithmetic and outputs are float64. Promotion cannot recover lost extraction
precision. Treat stored tensor factors as read-only: center `[width]`, basis
`[min(n,width), width]`, and coefficients. No dense covariance is formed;
training copies and SVD workspace still require memory. Overflow or unrepresentable
nonzero coefficients raise an error instead of silently discarding a correction.

Model hooks copy factors once per device, compute in float32 by default, and
cast back to activation dtype; `compute_dtype=torch.float64` is available.
Hook arithmetic disables autocast locally to preserve the requested computation
dtype, then restores the caller's autocast context. Small corrections can
still disappear at output precision.
Nonfinite-state checks synchronize
the device. Float32 coefficients can also round to one; use float64 hook arithmetic
when preserving a small complement gain matters. Hooks are intended for
inference, not reuse of an inference-mode cache inside autograd.

Call `model.eval()` before extraction/generation. Extraction leaves gradient mode
unchanged; use `torch.inference_mode()` as above. Generation restores
caller CPU/model-CUDA RNG states and removes its temporary hooks, including on
failure. Existing hooks remain active. Model mode and parameters are unchanged.
Custom Python/NumPy randomness and concurrent use of global RNG/model state are
outside this contract. `prepend_bos=None` inherits model defaults by omitting the
option; an explicit boolean overrides it in generation.

The optional offline integration test uses a random tiny TransformerLens model
and local tokenizer, without downloading weights:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  python -m pytest -q tests/test_transformerlens_integration.py
```
