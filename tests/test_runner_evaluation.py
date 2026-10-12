from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "broad_study"))

from evaluation import (  # noqa: E402
    judgments_are_complete,
    prompt_block_reason,
    require_complete_evaluation,
)


class RunnerEvaluationTests(unittest.TestCase):
    def test_prompt_level_safety_block_is_detected(self) -> None:
        response = SimpleNamespace(
            prompt_feedback=SimpleNamespace(block_reason="BlockReason.SAFETY")
        )
        self.assertEqual(prompt_block_reason(response), "BlockReason.SAFETY")
        unspecified = SimpleNamespace(
            prompt_feedback=SimpleNamespace(
                block_reason="BlockReason.BLOCK_REASON_UNSPECIFIED"
            )
        )
        self.assertIsNone(prompt_block_reason(unspecified))
        self.assertIsNone(prompt_block_reason(SimpleNamespace(prompt_feedback=None)))

    def test_every_unscored_category_fails_closed(self) -> None:
        require_complete_evaluation(
            {"safety_blocks": 0, "errors": 0, "empty": 0}
        )
        for category in ("safety_blocks", "errors", "empty"):
            stats = {"safety_blocks": 0, "errors": 0, "empty": 0}
            stats[category] = 1
            with self.assertRaisesRegex(RuntimeError, category):
                require_complete_evaluation(stats)

    def test_aggregate_requires_both_scores_on_every_row(self) -> None:
        complete = {"row": {"trait_score": 50.0, "coherence": 80.0}}
        missing = {"row": {"trait_score": None, "coherence": 80.0}}
        self.assertTrue(judgments_are_complete(complete))
        self.assertFalse(judgments_are_complete(missing))
        self.assertFalse(judgments_are_complete({}))


if __name__ == "__main__":
    unittest.main()
