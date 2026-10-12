"""Reference identities for the complete target-centered map on synthetic arrays."""

from __future__ import annotations

import unittest

import numpy as np


def reference_conceptor(rows: np.ndarray, aperture: float) -> tuple[np.ndarray, np.ndarray]:
    """Target-present mean and C = R (R + aperture^-2 I)^-1 with R = Xc^T Xc / n."""
    mean = rows.mean(axis=0)
    centered = rows - mean
    covariance = centered.T @ centered / rows.shape[0]
    identity = np.eye(rows.shape[1])
    return mean, covariance @ np.linalg.inv(covariance + aperture**-2 * identity)


def complete_map(h, mean, conceptor, beta):
    return mean + beta * (np.eye(len(mean)) - conceptor) @ (h - mean)


def control_map(h, mean, beta):
    return mean + beta * (h - mean)


class OperatorAlgebraTests(unittest.TestCase):
    def setUp(self) -> None:
        generator = np.random.default_rng(20261007)
        scales = np.array([40.0, 25.0, 18.0, 6.0, 2.0, 0.5, 0.1, 0.0])
        self.rows = generator.normal(size=(30, 8)) * scales + generator.normal(size=8)
        self.states = generator.normal(size=(5, 8)) * 10.0

    def test_complete_map_component_identity(self) -> None:
        for aperture in (0.025, 0.05, 0.2, 2.0):
            mean, conceptor = reference_conceptor(self.rows, aperture)
            for beta in (1.0, 2.0):
                for h in self.states:
                    difference = complete_map(h, mean, conceptor, beta) - control_map(
                        h, mean, beta
                    )
                    np.testing.assert_allclose(
                        difference, -beta * conceptor @ (h - mean), rtol=0, atol=1e-10
                    )
        # The matched control is target-conditioned: it keeps the target mean and
        # gain, so it is not the identity map.
        mean, _ = reference_conceptor(self.rows, 0.05)
        np.testing.assert_allclose(control_map(mean, mean, 2.0), mean)
        self.assertFalse(np.allclose(control_map(self.states[0], mean, 2.0), self.states[0]))

    def test_directional_gain_and_attenuation_threshold(self) -> None:
        mean = self.rows.mean(axis=0)
        centered = self.rows - mean
        covariance = centered.T @ centered / len(centered)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        beta = 2.0
        for aperture in (0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0):
            _, conceptor = reference_conceptor(self.rows, aperture)
            for lam, direction in zip(eigenvalues, eigenvectors.T, strict=True):
                lam = max(lam, 0.0)
                c = aperture**2 * lam / (1 + aperture**2 * lam)
                np.testing.assert_allclose(conceptor @ direction, c * direction, atol=1e-9)
                gain = direction @ (
                    complete_map(mean + direction, mean, conceptor, beta) - mean
                )
                self.assertAlmostEqual(gain, beta * (1 - c), places=9)
                self.assertAlmostEqual(gain, beta / (1 + aperture**2 * lam), places=9)
                # At beta=2 a direction is attenuated exactly when c > 1/2,
                # i.e. aperture^2 * lambda > 1.
                if abs(aperture**2 * lam - 1) > 1e-9:
                    self.assertEqual(gain < 1, c > 0.5)
                    self.assertEqual(gain < 1, aperture**2 * lam > 1)
        # Covariance-null directions keep the full gain beta at every aperture.
        null_direction = eigenvectors[:, 0]
        self.assertLess(eigenvalues[0], 1e-12)
        _, conceptor = reference_conceptor(self.rows, 2.0)
        gain = null_direction @ (complete_map(mean + null_direction, mean, conceptor, 2.0) - mean)
        self.assertAlmostEqual(gain, 2.0, places=9)

    def test_small_aperture_expands_every_direction(self) -> None:
        mean, conceptor = reference_conceptor(self.rows, 0.01)
        gains = np.linalg.eigvalsh(2.0 * (np.eye(len(mean)) - conceptor))
        self.assertTrue(np.all(gains > 1))
        _, conceptor = reference_conceptor(self.rows, 2.0)
        gains = np.linalg.eigvalsh(2.0 * (np.eye(len(mean)) - conceptor))
        self.assertTrue(np.any(gains < 1))


if __name__ == "__main__":
    unittest.main()
