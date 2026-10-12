"""Checkpoint identity and atomic-write helpers for the broad study rerun path."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

MANIFEST_NAME = "run_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(document: dict[str, Any]) -> str:
    encoded = json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def initialize_checkpoint_directory(
    checkpoint_directory: Path,
    run_specification: dict[str, Any],
    checkpoint_names: tuple[str, ...],
) -> str:
    """Create or validate the manifest that owns every resumable checkpoint."""
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    expected_hash = canonical_json_hash(run_specification)
    manifest_path = checkpoint_directory / MANIFEST_NAME

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual_specification = manifest.get("run_specification")
        actual_hash = manifest.get("run_specification_sha256")
        if actual_specification != run_specification or actual_hash != expected_hash:
            raise RuntimeError(
                f"Checkpoint manifest mismatch in {checkpoint_directory}. "
                "Use a new --output-dir rather than combining different runs."
            )
        return expected_hash

    existing = [
        name for name in checkpoint_names if (checkpoint_directory / name).exists()
    ]
    if existing:
        raise RuntimeError(
            f"Unbound checkpoints found in {checkpoint_directory}: {existing}. "
            "Move them to a historical directory or use a new --output-dir."
        )

    atomic_write_json(
        manifest_path,
        {
            "run_specification_sha256": expected_hash,
            "run_specification": run_specification,
        },
    )
    return expected_hash


def require_checkpoint_identity(
    payload: dict[str, Any], expected_hash: str, checkpoint_path: Path
) -> None:
    actual_hash = payload.get("run_specification_sha256")
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} belongs to run {actual_hash!r}, "
            f"expected {expected_hash!r}. Use a new --output-dir."
        )
