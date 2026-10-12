from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification and specification.loader
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


erasure = load_module("erasure_runner_extraction_test", Path(__file__).with_name("runner.py"))
matched = load_module(
    "matched_runner_extraction_test",
    SOURCE / "matched_control" / "source" / "runner.py",
)


def write_checkpoint_zip(path: Path, member: str) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, b"small checkpoint fixture")


def test_matched_extraction_uses_exclusive_temporary(tmp_path: Path) -> None:
    expected = "fixture.pt"
    archive = tmp_path / "fixture.zip"
    cache = tmp_path / "cache"
    write_checkpoint_zip(archive, expected)

    extracted = matched.prepare_extraction_file(
        archive, expected_filename=expected, cache_root=cache
    )

    assert extracted.read_bytes() == b"small checkpoint fixture"
    assert not list(cache.rglob("*.tmp"))


def test_extraction_metadata_is_portable_and_cache_still_validates(tmp_path: Path) -> None:
    archive = tmp_path / "fixture.zip"
    write_checkpoint_zip(archive, "local-layout/fixture.pt")
    kwargs = dict(expected_filename="fixture.pt", cache_root=tmp_path / "cache")
    extracted = matched.prepare_extraction_file(archive, **kwargs)
    marker = json.loads(extracted.with_name("source.json").read_text())
    assert marker == {
        "zip_name": archive.name,
        "zip_sha256": matched.sha256_file(archive),
        "member": "fixture.pt",
        "extracted_name": "fixture.pt",
        "extracted_sha256": matched.sha256_file(extracted),
    }
    assert str(tmp_path) not in json.dumps(marker)
    before = extracted.stat().st_mtime_ns
    assert matched.prepare_extraction_file(archive, **kwargs) == extracted
    assert extracted.stat().st_mtime_ns == before
    extracted.write_bytes(b"changed")
    assert matched.prepare_extraction_file(archive, **kwargs).read_bytes() == b"small checkpoint fixture"


def test_matched_extraction_rejects_symlink_cache(tmp_path: Path) -> None:
    expected = "fixture.pt"
    archive = tmp_path / "fixture.zip"
    destination = tmp_path / "destination"
    destination.mkdir()
    cache = tmp_path / "cache"
    cache.symlink_to(destination, target_is_directory=True)
    write_checkpoint_zip(archive, expected)

    with pytest.raises(ValueError, match="must not be a symlink"):
        matched.prepare_extraction_file(
            archive, expected_filename=expected, cache_root=cache
        )


def test_matched_extraction_rejects_symlink_target(tmp_path: Path) -> None:
    expected = "fixture.pt"
    archive = tmp_path / "fixture.zip"
    write_checkpoint_zip(archive, expected)
    zip_hash = matched.sha256_file(archive)
    target_directory = tmp_path / "cache" / zip_hash[:16]
    target_directory.mkdir(parents=True)
    destination = tmp_path / "destination.pt"
    destination.write_bytes(b"do not overwrite")
    (target_directory / expected).symlink_to(destination)

    with pytest.raises(ValueError, match="contains a symlink"):
        matched.prepare_extraction_file(
            archive, expected_filename=expected, cache_root=tmp_path / "cache"
        )


def test_matched_metadata_write_ignores_legacy_temp_symlink(tmp_path: Path) -> None:
    expected = "fixture.pt"
    archive = tmp_path / "fixture.zip"
    write_checkpoint_zip(archive, expected)
    zip_hash = matched.sha256_file(archive)
    target_directory = tmp_path / "cache" / zip_hash[:16]
    target_directory.mkdir(parents=True)
    destination = tmp_path / "destination.txt"
    destination.write_text("do not overwrite", encoding="utf-8")
    (target_directory / "source.json.tmp").symlink_to(destination)

    extracted = matched.prepare_extraction_file(
        archive, expected_filename=expected, cache_root=tmp_path / "cache"
    )

    assert extracted.read_bytes() == b"small checkpoint fixture"
    assert destination.read_text(encoding="utf-8") == "do not overwrite"
    assert (target_directory / "source.json").is_file()
