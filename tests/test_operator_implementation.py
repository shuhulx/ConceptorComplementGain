"""Check conceptor implementations against reference algebra on synthetic CPU tensors."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


matched = load_module(
    "_test_matched_runner", ROOT / "experiments" / "matched_control" / "source" / "runner.py"
)
erasure = load_module(
    "_test_erasure_runner", ROOT / "experiments" / "erasure" / "source" / "runner.py"
)
spectral = load_module(
    "_test_spectral_metrics",
    ROOT / "experiments" / "spectral_screening" / "source" / "spectral_metrics.py",
)


def reference_conceptor(rows: np.ndarray, aperture: float, divisor: int | None = None):
    mean = rows.mean(axis=0)
    centered = rows - mean
    covariance = centered.T @ centered / (len(rows) if divisor is None else divisor)
    conceptor = covariance @ np.linalg.inv(covariance + aperture**-2 * np.eye(rows.shape[1]))
    return mean, conceptor


class OperatorImplementationTests(unittest.TestCase):
    def test_matched_statistics_keep_equal_question_weight(self) -> None:
        paired = pd.DataFrame({"question_index": [0, 0, 1], "difference": [2., 2., 8.]})
        summary = matched.paired_contrast_summary(
            paired, endpoint="trait", column="difference", equivalence_margin=10,
            n_bootstrap=1000, bootstrap_seed=42,
        )
        self.assertEqual(summary["n_paired_rows"], 3)
        self.assertEqual(summary["n_question_clusters"], 2)
        self.assertEqual(summary["mean_difference"], 5.)
        self.assertEqual(summary["row_level_mean_difference"], 4.)
        self.assertEqual((summary["ci95_low"], summary["ci95_high"]), (2., 8.))
        self.assertEqual(matched.holm_adjust([0.01, 0.04, None, 0.03]), [0.03, 0.06, None, 0.06])

    def setUp(self) -> None:
        generator = np.random.default_rng(1701)
        # Synthetic rows whose leading variances straddle aperture^-2 = 400 at
        # aperture .05, so the n versus n-1 divisor changes the conceptor visibly.
        scales = np.array([30.0, 22.0, 15.0, 4.0, 1.0, 0.2])
        self.rows = (generator.normal(size=(10, 6)) * scales + 5.0).astype(np.float64)
        self.states = generator.normal(size=(2, 3, 6)) * 20.0 + 5.0
        self.aperture = 0.05

    def test_covariance_centering_and_divisor(self) -> None:
        mean, conceptor = reference_conceptor(self.rows, self.aperture)
        _, sample_divisor = reference_conceptor(
            self.rows, self.aperture, divisor=len(self.rows) - 1
        )
        self.assertGreater(np.abs(conceptor - sample_divisor).max(), 1e-2)
        expected_spectrum = np.sort(np.linalg.eigvalsh(conceptor))[::-1]

        factors = matched.compute_conceptor_factors(
            torch.tensor(self.rows, dtype=torch.float32), aperture=self.aperture, layer=9
        )
        np.testing.assert_allclose(factors.mean_cpu.numpy(), mean, rtol=1e-5, atol=1e-4)
        dense = (factors.vh_cpu.T * factors.mu_cpu) @ factors.vh_cpu
        np.testing.assert_allclose(dense.numpy(), conceptor, atol=2e-4)
        self.assertEqual(factors.n_samples, len(self.rows))

        spectrum = spectral.conceptor_eigenvalues(torch.tensor(self.rows), self.aperture)
        np.testing.assert_allclose(spectrum, expected_spectrum, atol=1e-10)
        # Centering: a common shift of all rows leaves the conceptor unchanged.
        shifted = spectral.conceptor_eigenvalues(torch.tensor(self.rows + 50.0), self.aperture)
        np.testing.assert_allclose(shifted, expected_spectrum, atol=1e-9)

        erasure_aperture = erasure.ALPHA_INV_SQ**-0.5
        erasure_mean, erasure_conceptor = reference_conceptor(self.rows, erasure_aperture)
        transform, details = erasure.fit_conceptor_not(self.rows)
        for h in self.states.reshape(-1, 6):
            np.testing.assert_allclose(
                transform(h[None, :])[0],
                h - erasure_conceptor @ (h - erasure_mean),
                atol=1e-8,
            )
        self.assertEqual(details["conceptor_numerical_rank"], 6)

    def test_matched_runner_hook_matches_reference(self) -> None:
        mean, conceptor = reference_conceptor(self.rows, self.aperture)
        factors = matched.compute_conceptor_factors(
            torch.tensor(self.rows, dtype=torch.float32), aperture=self.aperture, layer=9
        )
        h = torch.tensor(self.states, dtype=torch.float32)
        beta = 2.0
        complete = matched.MatchedNotIdentityHook(
            factors, beta=beta, method="not", model_device=torch.device("cpu")
        )(h, None).double().numpy()
        control = matched.MatchedNotIdentityHook(
            factors, beta=beta, method="identity", model_device=torch.device("cpu")
        )(h, None).double().numpy()
        centered = self.states - mean
        np.testing.assert_allclose(
            control, mean + beta * centered, rtol=1e-5, atol=1e-3
        )
        np.testing.assert_allclose(
            complete,
            mean + beta * (centered - centered @ conceptor.T),
            rtol=1e-5,
            atol=1e-3,
        )
        # Exact component identity at a common input: F_C(h) - F_0(h) = -beta C (h - m).
        np.testing.assert_allclose(
            complete - control, -beta * centered @ conceptor.T, rtol=1e-4, atol=1e-3
        )
        # Directional gains of the complete map are beta * (1 - c).
        gains = np.linalg.eigvalsh(beta * (np.eye(6) - conceptor))
        expected = np.sort(beta * (1 - factors.mu_cpu.double().numpy()))
        np.testing.assert_allclose(np.sort(gains), expected, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
