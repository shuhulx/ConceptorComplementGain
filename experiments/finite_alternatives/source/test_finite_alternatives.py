"""Tests for the finite-alternative record and verifier (shipped aggregate CSV plus synthetic inputs)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import verify_finite_alternatives as verify
from finite_alternatives import (
    FIELDS,
    MODELS,
    RECORDS,
    RecordError,
    classify,
    conditions,
    expected_settings,
    joint_score,
    parse_records,
    read_records,
    render_csv,
    summarize,
)


def shipped_rows() -> list[dict]:
    with RECORDS.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, rows: list[dict], fields=FIELDS) -> Path:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def find(rows: list[dict], study: str, model: str, trait: str) -> dict:
    return next(r for r in rows if (r["study"], r["model"], r["trait"]) == (study, model, trait))


def run_verifier(path: Path, capsys) -> tuple[int, str]:
    code = verify.main(["--records", str(path)])
    return code, capsys.readouterr().out


def synthetic_records() -> list[dict]:
    records = []
    for study, model, trait, coefficient in expected_settings():
        raw, alternative = conditions(study, coefficient)
        records.append({"study": study, "model": model, "layer": MODELS[model], "trait": trait,
                        "coefficient": coefficient, "raw_condition": raw, "alternative_condition": alternative,
                        "n": 20, "baseline_trait_mean": 50.0, "raw_trait_mean": 30.0,
                        "raw_historical_coherence_mean": 80.0, "alternative_trait_mean": 30.0,
                        "alternative_historical_coherence_mean": 80.0})
    return records


# Released record and verifier.

def test_shipped_record_passes(capsys) -> None:
    code, out = run_verifier(RECORDS, capsys)
    assert code == 0, out
    assert "PASS:" in out and "fisher: 2 win / 9 tie / 10 loss of 21" in out


def test_shipped_record_counts_and_gevd_exception() -> None:
    records = read_records()
    assert len(records) == 77
    assert summarize(records) == {"purification": {"win": 12, "tie": 16, "loss": 14},
                                  "fisher": {"win": 2, "tie": 9, "loss": 10},
                                  "gevd": {"win": 1, "tie": 5, "loss": 8}}
    assert verify.EXPECTED_COUNTS == {study: tuple(counts.values()) for study, counts in summarize(records).items()}
    assert verify.EXPECTED_TOTALS == {study: sum(counts.values()) for study, counts in summarize(records).items()}
    row = next(r for r in records if (r["study"], r["model"], r["trait"]) == ("gevd", "Qwen2.5-7B-Instruct", "apathetic"))
    raw = joint_score(row["baseline_trait_mean"], row["raw_trait_mean"], row["raw_historical_coherence_mean"])
    gevd = joint_score(row["baseline_trait_mean"], row["alternative_trait_mean"],
                       row["alternative_historical_coherence_mean"])
    assert (round(raw, 2), round(gevd, 2)) == (-20.04, -17.7) and gevd < 0


def test_extra_text_column_rejected(tmp_path, capsys) -> None:
    rows = [{**r, "response_text": "synthetic text"} for r in shipped_rows()]
    code, out = run_verifier(write_rows(tmp_path / "x.csv", rows, (*FIELDS, "response_text")), capsys)
    assert code == 1 and "header mismatch" in out and "response_text" in out


def test_missing_and_renamed_columns_rejected(tmp_path, capsys) -> None:
    rows = [{k: v for k, v in r.items() if k != "n"} for r in shipped_rows()]
    code, out = run_verifier(write_rows(tmp_path / "a.csv", rows, [f for f in FIELDS if f != "n"]), capsys)
    assert code == 1 and "header mismatch" in out
    rows = [{("coherence_mean" if k == "raw_historical_coherence_mean" else k): v for k, v in r.items()}
            for r in shipped_rows()]
    renamed = [("coherence_mean" if f == "raw_historical_coherence_mean" else f) for f in FIELDS]
    code, out = run_verifier(write_rows(tmp_path / "b.csv", rows, renamed), capsys)
    assert code == 1 and "header mismatch" in out


def test_missing_duplicate_and_reordered_rows_rejected(tmp_path, capsys) -> None:
    rows = shipped_rows()
    code, out = run_verifier(write_rows(tmp_path / "missing.csv", rows[:10] + rows[11:]), capsys)
    assert code == 1 and "missing comparisons" in out
    code, out = run_verifier(write_rows(tmp_path / "duplicate.csv", rows + [rows[0]]), capsys)
    assert code == 1 and "duplicate" in out
    code, out = run_verifier(write_rows(tmp_path / "replaced.csv", [rows[1]] + rows[1:]), capsys)
    assert code == 1 and "duplicate" in out
    code, out = run_verifier(write_rows(tmp_path / "order.csv", [rows[1], rows[0]] + rows[2:]), capsys)
    assert code == 1 and "released order" in out


@pytest.mark.parametrize(("field", "value", "message"), [
    ("n", "19", "n 19 != 20"),
    ("n", "20.0", "canonical integer"),
    ("layer", "9", "layer 9 != 14"),
    ("baseline_trait_mean", "nan", "finite canonical float"),
    ("raw_trait_mean", "inf", "finite canonical float"),
    ("raw_historical_coherence_mean", "100.5", "outside [0, 100]"),
    ("alternative_trait_mean", "-0.5", "outside [0, 100]"),
    ("alternative_historical_coherence_mean", "9.1e1", "finite canonical float"),
    ("alternative_trait_mean", "", "not a number"),
])
def test_invalid_values_rejected(tmp_path, capsys, field, value, message) -> None:
    rows = shipped_rows()
    rows[0][field] = value
    code, out = run_verifier(write_rows(tmp_path / "x.csv", rows), capsys)
    assert code == 1 and message in out


@pytest.mark.parametrize(("coefficient", "alternative", "message"), [
    ("5.0", "fisher_c5.0", "unexpected coefficient"),
    ("3.0", "fisher_pur_c3.0", "unexpected conditions"),
    ("1.5", "fisher_c1.5", "unexpected coefficient"),
])
def test_unexpected_fisher_condition_rejected(tmp_path, capsys, coefficient, alternative, message) -> None:
    rows = shipped_rows()
    row = find(rows, "fisher", "Qwen2.5-7B-Instruct", "apathetic")
    row.update(coefficient=coefficient, raw_condition=f"raw_c{coefficient}", alternative_condition=alternative)
    code, out = run_verifier(write_rows(tmp_path / "x.csv", rows), capsys)
    assert code == 1 and message in out


def test_gevd_variant_and_model_substitution_rejected(tmp_path, capsys) -> None:
    rows = shipped_rows()
    find(rows, "gevd", "Qwen2.5-7B-Instruct", "evil")["alternative_condition"] = "gevd_top5_c3.0"
    code, out = run_verifier(write_rows(tmp_path / "variant.csv", rows), capsys)
    assert code == 1 and "unexpected conditions" in out
    rows = shipped_rows()
    find(rows, "gevd", "Qwen2.5-7B-Instruct", "evil").update(model="Qwen2.5-1.5B-Instruct", layer="14")
    code, out = run_verifier(write_rows(tmp_path / "model.csv", rows), capsys)
    assert code == 1 and "unexpected model" in out


def test_non_canonical_bytes_rejected(tmp_path, capsys) -> None:
    data = RECORDS.read_bytes()
    (tmp_path / "crlf.csv").write_bytes(data.replace(b"\n", b"\r\n"))
    code, out = run_verifier(tmp_path / "crlf.csv", capsys)
    assert code == 1 and "LF line endings" in out
    (tmp_path / "quoted.csv").write_bytes(data.replace(b"gevd,", b'"gevd",', 1))
    code, out = run_verifier(tmp_path / "quoted.csv", capsys)
    assert code == 1 and "canonical serialized form" in out


def test_altered_value_flips_a_count(tmp_path, capsys) -> None:
    rows = shipped_rows()
    find(rows, "fisher", "Qwen2.5-1.5B-Instruct", "evil")["alternative_trait_mean"] = "5.0"
    code, out = run_verifier(write_rows(tmp_path / "x.csv", rows), capsys)
    assert code == 1 and "outcome_counts: fisher: win/tie/loss (2, 8, 11)" in out
    assert "PASS manuscript_fisher_2_of_21" in out


def test_paper_claims_fail_on_synthetic_all_ties(tmp_path, capsys) -> None:
    path = tmp_path / "ties.csv"
    path.write_bytes(render_csv(synthetic_records()))
    code, out = run_verifier(path, capsys)
    assert code == 1 and "PASS grid" in out
    for group in ("manuscript_purification_varies", "manuscript_fisher_2_of_21", "manuscript_gevd_exception"):
        assert f"FAIL {group}" in out


def test_gevd_exception_is_checked(tmp_path, capsys) -> None:
    rows = shipped_rows()
    find(rows, "gevd", "Qwen2.5-7B-Instruct", "apathetic")["alternative_historical_coherence_mean"] = "61.0"
    code, out = run_verifier(write_rows(tmp_path / "value.csv", rows), capsys)
    assert code == 1 and "FAIL manuscript_gevd_exception" in out and "PASS outcome_counts" in out
    rows = shipped_rows()
    find(rows, "gevd", "Qwen2.5-7B-Instruct", "apathetic").update(alternative_trait_mean="84.0",
                                                               alternative_historical_coherence_mean="24.0")
    code, out = run_verifier(write_rows(tmp_path / "tie.csv", rows), capsys)
    assert code == 1 and "gevd: win/tie/loss (0, 6, 8)" in out and "0 GEVD wins" in out


def test_unreadable_record_fails_cleanly(tmp_path, capsys) -> None:
    code, out = run_verifier(tmp_path / "absent.csv", capsys)
    assert code == 1 and out.splitlines()[-1].startswith("FAIL: invalid finite-alternative record")


# Module behavior on synthetic records.

def test_tie_is_exact_float_equality() -> None:
    assert classify(0.3, 0.1 + 0.2) == "win"
    assert classify(0.1 + 0.2, 0.3) == "loss"
    assert classify(-17.7, -17.7) == "tie"
    # Both scores are zero when the steered mean equals the baseline, whatever the coherence.
    assert classify(joint_score(0.0, 0.0, 80.0), joint_score(0.0, 0.0, 20.0)) == "tie"
    assert joint_score(0.5, 84.0, 24.0) == (0.5 - 84.0) * 24.0 / 100


def test_synthetic_records_round_trip_and_summarize() -> None:
    records = synthetic_records()
    records[0]["alternative_trait_mean"] = 0.1 + 0.2
    data = render_csv(records)
    assert parse_records(data) == records
    summary = summarize(records)
    assert summary["purification"] == {"win": 1, "tie": 41, "loss": 0}
    assert summary["fisher"] == {"win": 0, "tie": 21, "loss": 0} and summary["gevd"]["tie"] == 14
    with pytest.raises(RecordError, match="missing comparisons"):
        parse_records(render_csv(records[:-1]))
