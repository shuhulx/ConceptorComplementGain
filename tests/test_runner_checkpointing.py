from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "broad_study"))

from checkpointing import (  # noqa: E402
    MANIFEST_NAME,
    canonical_json_hash,
    initialize_checkpoint_directory,
    require_checkpoint_identity,
)


class RunnerCheckpointingTests(unittest.TestCase):
    def test_manifest_binds_and_revalidates_a_run(self) -> None:
        specification = {"model": "example", "revision": "a" * 40, "fast": False}
        with tempfile.TemporaryDirectory() as temporary_name:
            checkpoint_directory = Path(temporary_name)
            expected = canonical_json_hash(specification)
            actual = initialize_checkpoint_directory(
                checkpoint_directory, specification, ("extractions.pt",)
            )
            self.assertEqual(actual, expected)
            self.assertEqual(
                initialize_checkpoint_directory(
                    checkpoint_directory, specification, ("extractions.pt",)
                ),
                expected,
            )
            manifest = json.loads(
                (checkpoint_directory / MANIFEST_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["run_specification"], specification)

    def test_manifest_rejects_cross_run_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            checkpoint_directory = Path(temporary_name)
            initialize_checkpoint_directory(
                checkpoint_directory, {"revision": "a" * 40}, ("extractions.pt",)
            )
            with self.assertRaisesRegex(RuntimeError, "manifest mismatch"):
                initialize_checkpoint_directory(
                    checkpoint_directory, {"revision": "b" * 40}, ("extractions.pt",)
                )

    def test_unbound_checkpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            checkpoint_directory = Path(temporary_name)
            (checkpoint_directory / "extractions.pt").write_bytes(b"historical")
            with self.assertRaisesRegex(RuntimeError, "Unbound checkpoints"):
                initialize_checkpoint_directory(
                    checkpoint_directory, {"revision": "a" * 40}, ("extractions.pt",)
                )

    def test_checkpoint_payload_requires_the_same_identity(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "belongs to run"):
            require_checkpoint_identity(
                {"run_specification_sha256": "old"},
                "new",
                Path("checkpoint.json"),
            )


if __name__ == "__main__":
    unittest.main()
