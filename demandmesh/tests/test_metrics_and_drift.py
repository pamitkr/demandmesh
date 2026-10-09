import unittest

import numpy as np
import pandas as pd

from demandmesh.drift import build_reference_profile, compute_drift_report
from demandmesh.metrics import interval_coverage, pinball_loss, point_metrics, wape
from demandmesh.model_training import conformal_interval_adjustment


class MetricTests(unittest.TestCase):
    def test_wape_and_bias(self):
        y, yhat = np.array([10.0, 0.0, 30.0]), np.array([12.0, 1.0, 27.0])
        self.assertAlmostEqual(wape(y, yhat), 6 / 40)
        self.assertAlmostEqual(point_metrics(y, yhat)["bias"], 0.0)

    def test_pinball_is_asymmetric(self):
        y = np.array([10.0])
        self.assertAlmostEqual(pinball_loss(y, np.array([8.0]), 0.9), 1.8)  # under-forecast
        self.assertAlmostEqual(pinball_loss(y, np.array([12.0]), 0.9), 0.2)  # over-forecast

    def test_conformal_adjustment_restores_coverage(self):
        rng = np.random.default_rng(0)
        y = rng.normal(0, 1, 5_000)
        lower, upper = np.full_like(y, -0.5), np.full_like(y, 0.5)  # too narrow
        q = conformal_interval_adjustment(y[:2_500], lower[:2_500], upper[:2_500], 0.8)
        coverage = interval_coverage(y[2_500:], lower[2_500:] - q, upper[2_500:] + q)
        self.assertAlmostEqual(coverage, 0.8, delta=0.03)


class DriftTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.reference = pd.DataFrame({"orders": rng.poisson(10, 20_000).astype(float)})
        self.profile = build_reference_profile(self.reference, ("orders",))

    def test_same_distribution_is_stable(self):
        rng = np.random.default_rng(2)
        current = pd.DataFrame({"orders": rng.poisson(10, 5_000).astype(float)})
        self.assertEqual(compute_drift_report(current, self.profile)["overall_status"], "stable")

    def test_shifted_distribution_alerts(self):
        rng = np.random.default_rng(3)
        current = pd.DataFrame({"orders": rng.poisson(16, 5_000).astype(float)})
        report = compute_drift_report(current, self.profile)
        self.assertEqual(report["overall_status"], "alert")
        self.assertGreater(report["features"]["orders"]["psi"], 0.25)


if __name__ == "__main__":
    unittest.main()
