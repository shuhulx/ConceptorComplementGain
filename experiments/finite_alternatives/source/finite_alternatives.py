#!/usr/bin/env python3
"""Fixed finite-alternative comparisons: record schema, joint score and outcome counts.

Standard library only. The released record ``plot_data/finite_alternatives_scores.csv``
holds one row per fixed comparison of a raw mean-difference direction with one
alternative direction (rank-10 purification, Fisher, or GEVD top-1) at the same
coefficient. Each row carries only historical aggregate means over n = 20
questions: baseline target expression, steered target expression and steered
historical-rubric coherence. No per-answer score or response text is stored.

The joint descriptive score is ``S = (b - t) * a / 100`` with baseline mean ``b``,
steered target mean ``t`` and steered historical coherence mean ``a``. An
alternative "wins" when its S is strictly greater than the raw S, "ties" on
exact float equality and otherwise "loses".
"""

from __future__ import annotations

import csv
import io
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RECORDS = REPO_ROOT / "plot_data" / "finite_alternatives_scores.csv"
FIELDS = (
    "study", "model", "layer", "trait", "coefficient", "raw_condition", "alternative_condition", "n",
    "baseline_trait_mean", "raw_trait_mean", "raw_historical_coherence_mean",
    "alternative_trait_mean", "alternative_historical_coherence_mean",
)
MEAN_FIELDS = FIELDS[8:]
TRAITS = ("sycophancy", "hallucination", "evil", "impolite", "optimistic", "apathetic", "humorous")
# Model: steering layer of the finite-alternative runs.
MODELS = {"Qwen2.5-1.5B-Instruct": 14, "Qwen2.5-7B-Instruct": 19, "Llama-3.1-8B-Instruct": 15}
N_QUESTIONS = 20
# Study: (alternative condition prefix, models, coefficients) for the included comparisons.
STUDIES = {
    "purification": ("purified", tuple(MODELS), (1.5, 3.0)),
    "fisher": ("fisher", tuple(MODELS), (3.0,)),
    "gevd": ("gevd_top1", ("Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct"), (3.0,)),
}
OUTCOMES = ("win", "tie", "loss")


class RecordError(ValueError):
    """The finite-alternative record does not match the released contract."""


def expected_settings() -> list[tuple[str, str, str, float]]:
    """(study, model, trait, coefficient) in released row order."""
    return [
        (study, model, trait, coefficient)
        for study, (_prefix, models, coefficients) in STUDIES.items()
        for model in models
        for trait in TRAITS
        for coefficient in coefficients
    ]


def conditions(study: str, coefficient: float) -> tuple[str, str]:
    """Source condition names (raw, alternative) compared at one coefficient."""
    return f"raw_c{coefficient!r}", f"{STUDIES[study][0]}_c{coefficient!r}"


def _canonical_float(text: str, label: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise RecordError(f"{label}: not a number: {text!r}") from None
    if not math.isfinite(value) or repr(value) != text:
        raise RecordError(f"{label}: not a finite canonical float: {text!r}")
    return value


def _canonical_int(text: str, label: str) -> int:
    if not text.isdigit() or str(int(text)) != text:
        raise RecordError(f"{label}: not a canonical integer: {text!r}")
    return int(text)


def parse_row(row: dict, index: int) -> dict:
    label = f"row {index + 1}"
    if None in row or any(value is None for value in row.values()):
        raise RecordError(f"{label}: wrong number of fields")
    study, model, trait = row["study"], row["model"], row["trait"]
    if study not in STUDIES:
        raise RecordError(f"{label}: unexpected study {study!r}")
    if model not in STUDIES[study][1]:
        raise RecordError(f"{label}: unexpected model {model!r} for {study}")
    if trait not in TRAITS:
        raise RecordError(f"{label}: unexpected trait {trait!r}")
    label = f"{label} ({study}/{model}/{trait})"
    layer = _canonical_int(row["layer"], f"{label} layer")
    if layer != MODELS[model]:
        raise RecordError(f"{label}: layer {layer} != {MODELS[model]}")
    coefficient = _canonical_float(row["coefficient"], f"{label} coefficient")
    if coefficient not in STUDIES[study][2]:
        raise RecordError(f"{label}: unexpected coefficient {coefficient!r}")
    if (row["raw_condition"], row["alternative_condition"]) != conditions(study, coefficient):
        raise RecordError(f"{label}: unexpected conditions "
                          f"{row['raw_condition']!r} vs {row['alternative_condition']!r}")
    n = _canonical_int(row["n"], f"{label} n")
    if n != N_QUESTIONS:
        raise RecordError(f"{label}: n {n} != {N_QUESTIONS}")
    record = {"study": study, "model": model, "layer": layer, "trait": trait,
              "coefficient": coefficient, "raw_condition": row["raw_condition"],
              "alternative_condition": row["alternative_condition"], "n": n}
    for field in MEAN_FIELDS:
        value = _canonical_float(row[field], f"{label} {field}")
        if not 0.0 <= value <= 100.0:
            raise RecordError(f"{label}: {field} {value!r} outside [0, 100]")
        record[field] = value
    return record


def parse_records(data: bytes) -> list[dict]:
    """Parse and validate the released CSV bytes against the full contract."""
    if data.startswith(b"\xef\xbb\xbf") or b"\r" in data:
        raise RecordError("record must be UTF-8 without BOM and use LF line endings")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RecordError(f"record is not UTF-8: {error}") from None
    reader = csv.DictReader(io.StringIO(text, newline=""))
    header = list(reader.fieldnames or [])
    if header != list(FIELDS):
        extra = [f for f in header if f not in FIELDS]
        missing = [f for f in FIELDS if f not in header]
        raise RecordError(f"header mismatch (extra {extra}, missing {missing}, or wrong order)")
    records = [parse_row(row, index) for index, row in enumerate(reader)]
    keys = [(r["study"], r["model"], r["trait"], r["coefficient"]) for r in records]
    expected = expected_settings()
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise RecordError(f"duplicate comparisons: {duplicates[:3]}")
    present = set(keys)
    missing = [key for key in expected if key not in present]
    if missing:
        raise RecordError(f"{len(records)} rows; missing comparisons: {missing[:3]}")
    if keys != expected:
        raise RecordError("rows are not in released order (study, model, trait, coefficient)")
    if render_csv(records) != data:
        raise RecordError("record is not in canonical serialized form")
    return records


def read_records(path: Path = RECORDS) -> list[dict]:
    return parse_records(Path(path).read_bytes())


def render_csv(records: list[dict]) -> bytes:
    """Serialize records; floats use repr so that every value round-trips exactly."""
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(FIELDS)
    for record in records:
        writer.writerow([
            repr(float(record[field])) if field == "coefficient" or field in MEAN_FIELDS
            else str(record[field])
            for field in FIELDS
        ])
    return stream.getvalue().encode("utf-8")


def joint_score(baseline: float, steered_trait: float, steered_coherence: float) -> float:
    """S = (b - t) * a / 100, evaluated in this order."""
    return (baseline - steered_trait) * steered_coherence / 100


def classify(raw_score: float, alternative_score: float) -> str:
    """Win: strictly higher S. Tie: exact float equality. Otherwise loss."""
    if alternative_score > raw_score:
        return "win"
    if alternative_score == raw_score:
        return "tie"
    return "loss"


def compare(record: dict) -> dict:
    raw = joint_score(record["baseline_trait_mean"], record["raw_trait_mean"],
                      record["raw_historical_coherence_mean"])
    alternative = joint_score(record["baseline_trait_mean"], record["alternative_trait_mean"],
                              record["alternative_historical_coherence_mean"])
    return {**record, "raw_score": raw, "alternative_score": alternative,
            "outcome": classify(raw, alternative)}


def summarize(records: list[dict]) -> dict[str, dict[str, int]]:
    """Per-study win/tie/loss counts in study order."""
    summary = {study: dict.fromkeys(OUTCOMES, 0) for study in STUDIES}
    for comparison in map(compare, records):
        summary[comparison["study"]][comparison["outcome"]] += 1
    return summary
