#!/usr/bin/env python3
"""Recompute per-concept summaries and compare them with the included results."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments" / "semantic_functions" / "source" / "per_concept_outcome_analysis.py"
INPUT = ROOT / "provenance" / "broad_study_results.json"
EXPECTED = ROOT / "experiments" / "semantic_functions" / "results" / "per_concept"
EXACT_FILES = (
    "semantic_per_concept_outcomes.csv",
    "persona_per_concept_outcomes.csv",
    "within_model_inversion_examples.csv",
)


def compare_correlation_summary(rebuilt: Path) -> float:
    expected_path = EXPECTED / "correlation_summary.csv"
    rebuilt_path = rebuilt / "correlation_summary.csv"
    with expected_path.open(newline="", encoding="utf-8") as handle:
        expected_rows = list(csv.DictReader(handle))
    with rebuilt_path.open(newline="", encoding="utf-8") as handle:
        rebuilt_rows = list(csv.DictReader(handle))
    if len(expected_rows) != len(rebuilt_rows):
        raise SystemExit("Correlation-summary row count changed")
    maximum_difference = 0.0
    for row_number, (expected, actual) in enumerate(
        zip(expected_rows, rebuilt_rows, strict=True), start=1
    ):
        if expected.keys() != actual.keys():
            raise SystemExit(f"Correlation-summary schema changed at row {row_number}")
        for field, expected_value in expected.items():
            actual_value = actual[field]
            try:
                left = float(expected_value)
                right = float(actual_value)
            except ValueError:
                if expected_value != actual_value:
                    raise SystemExit(
                        f"Correlation-summary text changed at row {row_number}, {field}"
                    )
                continue
            difference = abs(left - right)
            maximum_difference = max(maximum_difference, difference)
            if not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12):
                raise SystemExit(
                    f"Correlation-summary value changed at row {row_number}, {field}: "
                    f"{left} != {right}"
                )
    return maximum_difference


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="conceptor-semantic-audit-") as name:
        output = Path(name) / "rebuilt"
        subprocess.run(
            [
                sys.executable,
                str(SOURCE),
                "--input-json",
                str(INPUT),
                "--output-dir",
                str(output),
            ],
            cwd=ROOT,
            check=True,
        )
        if (output / "SHA256SUMS.txt").exists():
            raise SystemExit("Semantic rebuild unexpectedly emitted a checksum ledger")
        manifest = json.loads((output / "analysis_manifest.json").read_text())
        if manifest["input_json"] != INPUT.name:
            raise SystemExit("Analysis manifest must identify its input without a workstation path")
        if not (output / "analysis_report.md").is_file():
            raise SystemExit("Missing per-concept analysis report")
        for filename in EXACT_FILES:
            if (EXPECTED / filename).read_bytes() != (output / filename).read_bytes():
                raise SystemExit(f"Semantic evidence is not byte-identical: {filename}")
            print(f"byte-identical: {filename}")
        maximum_difference = compare_correlation_summary(output)
        print(
            "numerically equivalent: correlation_summary.csv "
            f"(max absolute difference={maximum_difference:.3g})"
        )


if __name__ == "__main__":
    main()
