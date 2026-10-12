# Third-party notices

## Persona Vectors

The seven persona instruction, extraction and evaluation assets and the
historical coherence judge prompt are derived from
[safety-research/persona_vectors](https://github.com/safety-research/persona_vectors)
at commit `b8e0f044fe2410a6fad579f38324f03f13b4e917`, licensed Apache-2.0.
The upstream license and copyright notice are in
`third_party/persona_vectors/LICENSE`. Instruction pairs, evaluator rubrics and the
coherence prompt match upstream exactly; the Humorous question lists and one
Hallucination evaluation question are project-specific. Details:
[attribution and changes](third_party/persona_vectors/CHANGES.md).

## ConceptorSteering function vectors

The six Function-task pair selections come from
[jorispos/ConceptorSteering](https://github.com/jorispos/ConceptorSteering) at
commit `5e134153f4eeefa631bda39788f1f996bbf052a1`, which has no license file.
The pair text is therefore not committed. `experiments/broad_study/fetch_function_vectors.py`
specifies the pinned URL and revision, row indices and per-task SHA-256
digests, and validates fetched pairs against the study selections. See the
[fetch instructions](third_party/function_vectors/README.md).

## Model weights and hosted evaluators

No model weights, raw model outputs or raw hosted-service outputs are included
or relicensed. Qwen, Llama, Gemini and DeepSeek assets and services remain
subject to their providers' terms.
