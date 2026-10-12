"""Tests for the offline aperture record checker (numpy only, no tensors)."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

from verify_aperture_records import PLOT_DATA, main

RECORDS = (
    "aperture_baseline_21_fits.csv",
    "aperture_spectra_21_fits.csv",
    "aperture_correction_12_settings.csv",
    "aperture_correction_fold_evidence.csv",
    "aperture_summary.csv",
    "aperture_provenance.json",
    "operator_cells.csv",
    "correction_cells.csv",
)


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


def test_released_records_pass(capsys) -> None:
    assert main([]) == 0
    output = capsys.readouterr().out
    assert "from recorded spectra and corrections" in output
    assert output.rstrip().splitlines()[-1].startswith("PASS")


def test_altered_gain_is_detected(tmp_path, capsys) -> None:
    records = copy_records(tmp_path / "records")
    edit_csv(records / "aperture_spectra_21_fits.csv", 3, "minimum_gain_beta2", "0.99")
    assert main(["--plot-data", str(records)]) == 1
    assert "spectra_formula_g" in capsys.readouterr().out


def test_altered_fold_energy_breaks_pooled_ratio(tmp_path, capsys) -> None:
    records = copy_records(tmp_path / "records")
    edit_csv(records / "aperture_correction_fold_evidence.csv", 10, "correction_energy", "1.0")
    assert main(["--plot-data", str(records)]) == 1
    assert "correction_pooled" in capsys.readouterr().out


def test_missing_record_fails(tmp_path, capsys) -> None:
    records = copy_records(tmp_path / "records")
    (records / "aperture_summary.csv").unlink()
    assert main(["--plot-data", str(records)]) == 1
    assert "FAIL" in capsys.readouterr().out
