# Persona Vectors attribution and changes

Source: `safety-research/persona_vectors`  
Pinned commit: `b8e0f044fe2410a6fad579f38324f03f13b4e917`  
License: Apache-2.0 (copied beside this file)

`experimental_materials.json` combines the seven study targets into a single
schema. Against the pinned upstream revision:

- all seven five-pair instruction sets match;
- all seven evaluator rubrics match;
- six extraction-question lists match; all twenty Humorous extraction questions
  are project-specific;
- five evaluation-question lists match; all twenty Humorous evaluation questions
  are project-specific, and one of twenty Hallucination evaluation questions is
  modified;
- the shared historical coherence judge prompt (`coherence_prompt`) is exactly
  the `Prompts['coherence_0_100']` string of upstream `eval/prompts.py` at the pinned
  commit.

Persona instruction and extraction text is also embedded in the protocol-swap
runner.

Licensing: all upstream text, including the coherence prompt, remains under the
upstream Apache-2.0 license and notice. The project-specific questions and the
consolidated schema are licensed CC BY 4.0. The surrounding original runner
code is Apache-2.0. The corrected coherence-only rubric in
`deepseek_judge_prompts.json` (`corrected_coherence_rubric`) is
project-authored and is not upstream text.
