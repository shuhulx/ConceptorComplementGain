# License scope

| Material | License |
|---|---|
| Original source code | Apache-2.0 ([`LICENSE`](LICENSE)) |
| Project-authored documentation, processed data (`plot_data/`, experiment result tables) and provenance records | CC BY 4.0 ([`LICENSE-DATA`](LICENSE-DATA)) |
| Persona Vectors text in `experimental_materials.json` and embedded in the protocol-swap runner, including the historical coherence judge prompt | Apache-2.0, upstream notice preserved in [`third_party/persona_vectors/`](third_party/persona_vectors/) |
| Project-specific questions and the consolidated materials schema | CC BY 4.0 |

Notes:

- The historical coherence judge prompt (`coherence_prompt` in
  `experimental_materials.json`) is
  exactly the upstream Persona Vectors `Prompts['coherence_0_100']` in `eval/prompts.py`
  at commit `b8e0f044fe2410a6fad579f38324f03f13b4e917`, so the upstream
  Apache-2.0 license applies. The CC BY 4.0 grant does not cover it.
- Corrected-coherence aggregates in `plot_data/` are project-derived statistics
  with no response text, per-answer judgments or provider identifiers; they are
  CC BY 4.0 to the extent of the project's rights.
- No raw model outputs, hosted-evaluator judgments or provider logs are
  included, and no license is granted over any model or provider output.
- Function Vector pairs are downloaded separately; see their source and licensing status in
  [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

Third-party material remains governed by its upstream terms.
