#!/usr/bin/env python3
"""Offline record check for the fixed finite-alternative comparisons.

Reads only the released ``plot_data/finite_alternatives_scores.csv`` (standard
library only). It validates the schema and the fixed study/model/layer/
coefficient/condition grid and recomputes the joint score S for every raw and
alternative condition, then checks each study's win/tie/loss counts. Fisher
improves 2 of 21 comparisons; GEVD improves 1 of 14, with a negative score after
the improvement; purification both improves and worsens scores in every model.
The inputs are aggregate means evaluated with the original coherence rubric.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from finite_alternatives import (
    MODELS,
    N_QUESTIONS,
    RECORDS,
    STUDIES,
    TRAITS,
    RecordError,
    compare,
    read_records,
    summarize,
)

# Exact (win, tie, loss) counts per study from the included aggregate means.
EXPECTED_COUNTS = {"purification": (12, 16, 14), "fisher": (2, 9, 10), "gevd": (1, 5, 8)}
EXPECTED_TOTALS = {"purification": 42, "fisher": 21, "gevd": 14}
# Paper: the single GEVD improvement, (model, trait, coefficient, raw S, GEVD S) at 2 decimals.
GEVD_EXCEPTION = ("Qwen2.5-7B-Instruct", "apathetic", 3.0, -20.04, -17.70)


class Checks:
    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.failures: list[str] = []

    def __call__(self, group: str, condition: bool, detail: str) -> None:
        self.counts[group] = self.counts.get(group, 0) + 1
        if not condition:
            self.failures.append(f"{group}: {detail}")


def run(path: Path) -> tuple[Checks, dict]:
    check = Checks()
    records = read_records(path)
    comparisons = [compare(record) for record in records]
    summary = summarize(records)
    for study, (_prefix, models, coefficients) in STUDIES.items():
        rows = [c for c in comparisons if c["study"] == study]
        check("grid", len(rows) == EXPECTED_TOTALS[study] == len(models) * len(TRAITS) * len(coefficients),
              f"{study}: {len(rows)} rows")
        observed = tuple(summary[study][outcome] for outcome in ("win", "tie", "loss"))
        check("outcome_counts", observed == EXPECTED_COUNTS[study],
              f"{study}: win/tie/loss {observed} != {EXPECTED_COUNTS[study]}")
    for row in comparisons:
        label = f"{row['study']}/{row['model']}/{row['trait']}/c{row['coefficient']!r}"
        check("layers", row["layer"] == MODELS[row["model"]], label)
        check("sample_size", row["n"] == N_QUESTIONS, label)
        if row["outcome"] == "tie":
            # Every recorded tie is a steered mean equal to the baseline (both scores zero).
            check("exact_ties", row["alternative_score"] == row["raw_score"] == 0.0, label)
    # Method-specific comparison patterns.
    purification = [c for c in comparisons if c["study"] == "purification"]
    by_model = {m: {c["outcome"] for c in purification if c["model"] == m} for m in MODELS}
    check("manuscript_purification_varies", all({"win", "loss"} <= outcomes for outcomes in by_model.values()),
          "purification should both win and lose within every model")
    fisher = [c for c in comparisons if c["study"] == "fisher"]
    fisher_wins = [c for c in fisher if c["outcome"] == "win"]
    check("manuscript_fisher_2_of_21", (len(fisher_wins), len(fisher)) == (2, 21),
          f"{len(fisher_wins)} Fisher wins of {len(fisher)}")
    gevd = [c for c in comparisons if c["study"] == "gevd"]
    gevd_wins = [c for c in gevd if c["outcome"] == "win"]
    model, trait, coefficient, raw, alternative = GEVD_EXCEPTION
    check("manuscript_gevd_exception", (len(gevd_wins), len(gevd)) == (1, 14),
          f"{len(gevd_wins)} GEVD wins of {len(gevd)}")
    for win in gevd_wins:
        check("manuscript_gevd_exception", (win["model"], win["trait"], win["coefficient"]) == (model, trait, coefficient),
              f"GEVD win at {win['model']}/{win['trait']}")
        check("manuscript_gevd_exception", (round(win["raw_score"], 2), round(win["alternative_score"], 2)) == (raw, alternative),
              f"S {win['raw_score']:.4f} -> {win['alternative_score']:.4f} != {raw} -> {alternative}")
        check("manuscript_gevd_exception", win["alternative_score"] < 0, "GEVD exception score is not negative")
    return check, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--records", type=Path, default=RECORDS,
                        help="finite-alternative CSV (default: repository plot_data/finite_alternatives_scores.csv)")
    args = parser.parse_args(argv)
    print("Finite-alternative score and count checks (original-rubric aggregate means)")
    try:
        check, summary = run(args.records)
    except (OSError, RecordError) as error:
        print(f"FAIL: invalid finite-alternative record: {error}")
        return 1
    for group, count in check.counts.items():
        failed = sum(failure.startswith(group + ":") for failure in check.failures)
        print(f"  {'FAIL' if failed else 'PASS'} {group}: {count - failed}/{count}")
    for failure in check.failures[:20]:
        print(f"  - {failure}")
    for study, counts in summary.items():
        print(f"  {study}: {counts['win']} win / {counts['tie']} tie / {counts['loss']} loss of {sum(counts.values())}")
    total = sum(check.counts.values())
    if check.failures:
        print(f"FAIL: {len(check.failures)} of {total} checks failed")
        return 1
    print(f"PASS: {total} checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
