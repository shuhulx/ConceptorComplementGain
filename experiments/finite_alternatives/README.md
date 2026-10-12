# Finite alternatives: purification, Fisher and GEVD (historical aggregates)

Compares alternative static steering directions with the raw mean-difference
direction at the same coefficient. The record
`plot_data/finite_alternatives_scores.csv` holds 77 fixed comparisons of the raw
mean-difference direction with one alternative direction at the same coefficient:

| Study | Models | Compared conditions | Rows |
| --- | --- | --- | --- |
| Rank-10 purification | Qwen2.5-1.5B, Qwen2.5-7B, Llama-3.1-8B | `raw_c1.5` vs `purified_c1.5`; `raw_c3.0` vs `purified_c3.0` | 42 |
| Fisher (OAS shrinkage) | Qwen2.5-1.5B, Qwen2.5-7B, Llama-3.1-8B | `raw_c3.0` vs `fisher_c3.0` | 21 |
| GEVD top-1 | Qwen2.5-7B, Llama-3.1-8B | `raw_c3.0` vs `gevd_top1_c3.0` | 14 |

Each row contains aggregate means: baseline and steered target
expression and steered coherence over n = 20 questions (greedy decoding, one
response per question), at steering layers 14/19/15 (1.5B/7B/8B). Coherence is
the historical Gemini-rubric score; this study was not rescored with the
corrected coherence evaluator. Each study is a separate run, so baseline
and raw means differ between studies. The purification panel uses rank-10
purification at coefficients 1.5 and 3.0 across the three models and seven
targets shown above.

With baseline mean `b`, steered target mean `t` and steered coherence mean `a`,
the joint descriptive score is `S = (b - t) * a / 100`. An alternative wins when
its S is strictly greater than the raw S at the same coefficient, ties on exact
float equality and otherwise loses. Every tie, in all three studies, is a row
whose baseline, raw and alternative target means are all 0, so both scores are 0.

| Study | Win | Tie | Loss | Total |
| --- | --- | --- | --- | --- |
| Purification | 12 | 16 | 14 | 42 |
| Fisher | 2 | 9 | 10 | 21 |
| GEVD | 1 | 5 | 8 | 14 |

The GEVD win is Qwen2.5-7B-Instruct Apathetic, S -20.04 to -17.70; it stays negative.

Purification both wins and loses within every model. These win/tie/loss
counts summarize the included aggregate means, not uncertainty in individual
responses.

## Commands

From the repository root (standard library and pytest only):

```bash
python experiments/finite_alternatives/source/verify_finite_alternatives.py
python -m pytest -q experiments/finite_alternatives/source
```

The verifier checks the exact header, the 77-row study/model/trait/coefficient
grid and order, layers, n = 20, finite values in [0, 100], canonical float
serialization and the exact counts above. Tests run on CPU using the shipped
CSV, modified copies in a temporary directory and synthetic records.

## Limits

- Original generation inputs, per-answer scores, response text and judge
  records are not distributed; the verifier checks the recorded aggregate means.
- The counts compare these fixed settings, rather than ranking other
  coefficients, purification variants or multidirectional methods.
