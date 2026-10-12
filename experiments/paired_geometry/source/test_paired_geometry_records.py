"""Tests for the offline paired geometry record checker (numpy only, no tensors)."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from verify_paired_geometry_records import PLOT_DATA, main

RECORDS = ("paired_delta_results.json", "paired_geometry_main.csv")


def copy_records(destination: Path) -> Path:
    destination.mkdir()
    for name in RECORDS:
        shutil.copyfile(PLOT_DATA / name, destination / name)
    return destination


def edit_csv(path: Path, row_index: int, field: str, value: str) -> None:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rows[row_index][field] = value
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def edit_json(path: Path, edit) -> None:
    results = json.loads(path.read_text(encoding="utf-8"))
    edit(results)
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")


def test_released_records_pass(capsys) -> None:
    assert main([]) == 0
    output = capsys.readouterr().out
    assert "from recorded per-fit metrics" in output
    assert "PASS manuscript_paired_rank_higher: 21/21" in output
    assert "PASS manuscript_paired_k90_higher: 21/21" in output
    assert output.rstrip().splitlines()[-1].startswith("PASS")


def test_csv_value_not_in_json_is_detected(tmp_path, capsys) -> None:
    records = copy_records(tmp_path / "records")
    edit_csv(records / "paired_geometry_main.csv", 4, "paired_shannon_effective_rank", "55.66")
    assert main(["--plot-data", str(records)]) == 1
    assert "csv_derives_from_json" in capsys.readouterr().out


def test_paired_not_higher_is_detected(tmp_path, capsys) -> None:
    records = copy_records(tmp_path / "records")

    def lower_paired_k90(results: dict) -> None:
        results["8b"]["L30_peak"]["humorous"]["delta"]["k90"] = 50

    edit_json(records / "paired_delta_results.json", lower_paired_k90)
    edit_csv(records / "paired_geometry_main.csv", 20, "paired_k90", "50.0")
    assert main(["--plot-data", str(records)]) == 1
    output = capsys.readouterr().out
    assert "FAIL manuscript_paired_k90_higher: 20/21" in output
    assert "PASS csv_derives_from_json" in output


def test_altered_fit_breaks_recorded_ratio(tmp_path, capsys) -> None:
    records = copy_records(tmp_path / "records")

    def alter_mu1(results: dict) -> None:
        results["7b"]["L26_peak"]["evil"]["pos"]["mu1"] = 0.5

    edit_json(records / "paired_delta_results.json", alter_mu1)
    assert main(["--plot-data", str(records)]) == 1
    output = capsys.readouterr().out
    assert "FAIL fit_mu1_mu5_ratio" in output


def test_wrong_diagnostic_layer_is_detected(tmp_path, capsys) -> None:
    records = copy_records(tmp_path / "records")

    def rename_layer(results: dict) -> None:
        results["8b"]["L10_peak"] = results["8b"].pop("L30_peak")

    edit_json(records / "paired_delta_results.json", rename_layer)
    assert main(["--plot-data", str(records)]) == 1
    assert "FAIL layer_roles" in capsys.readouterr().out


def test_missing_record_fails(tmp_path, capsys) -> None:
    records = copy_records(tmp_path / "records")
    (records / "paired_geometry_main.csv").unlink()
    assert main(["--plot-data", str(records)]) == 1
    assert "FAIL" in capsys.readouterr().out
