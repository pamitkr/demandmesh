import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from demandmesh.exceptions import ForecastRequestError, ModelArtifactError
from demandmesh.forecaster import DemandForecaster
from demandmesh.model_training import ARTIFACT_FILE, LATEST_POINTER
from demandmesh.planner import plan_city_dispatch
from demandmesh.route_optimizer import RoutingConfig
from support import tiny_environment


class ForecasterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = tiny_environment()
        cls.forecaster = DemandForecaster.from_settings(cls.settings)

    def test_forecast_shape_and_quantile_order(self):
        fc = self.forecaster.forecast(horizon=7, cities=["SRT"])
        n_series = len(self.forecaster.series.query("city == 'SRT'"))
        self.assertEqual(len(fc), n_series * 7)
        self.assertTrue(np.all(fc["p10"] <= fc["p50"]))
        self.assertTrue(np.all(fc["p50"] <= fc["p90"]))
        self.assertTrue((fc["p10"] >= 0).all())
        self.assertEqual(sorted(fc["horizon_day"].unique()), list(range(1, 8)))

    def test_filters_and_historical_origin(self):
        fc = self.forecaster.forecast(
            horizon=3, as_of="2025-12-01", zone_ids=["LDH-Z01"], categories=["IND_MACH"]
        )
        self.assertEqual(len(fc), 3)
        self.assertEqual(str(fc["date"].min().date()), "2025-12-02")

    def test_invalid_requests_raise_domain_errors(self):
        with self.assertRaises(ForecastRequestError):
            self.forecaster.forecast(horizon=8)
        with self.assertRaises(ForecastRequestError):
            self.forecaster.forecast(zone_ids=["NOPE-Z99"])
        with self.assertRaises(ForecastRequestError):
            self.forecaster.forecast(as_of="2030-01-01")

    def test_tampered_artifact_is_rejected(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            models = tmp / "models"
            shutil.copytree(self.settings.model_dir, models)
            version = (models / LATEST_POINTER).read_text().strip()
            with (models / version / ARTIFACT_FILE).open("ab") as fh:
                fh.write(b"tampered")
            with self.assertRaises(ModelArtifactError):
                DemandForecaster(models, self.settings.history_path, self.settings.zones_path)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_drift_report_structure(self):
        report = self.forecaster.drift_report(window_days=28)
        self.assertIn(report["overall_status"], {"stable", "warn", "alert"})
        self.assertIn("orders", report["features"])

    def test_dispatch_plan_end_to_end(self):
        plan = plan_city_dispatch(self.forecaster, "SRT", routing_config=RoutingConfig(time_limit_s=0.5))
        self.assertGreater(plan["demand"]["drops"], 0)
        self.assertEqual(plan["optimised_plan"]["unserved_stop_ids"], [])
        served = sum(r["load_kg"] for r in plan["optimised_plan"]["routes"])
        self.assertAlmostEqual(served, plan["demand"]["planned_freight_kg"], delta=1.0)
        self.assertLessEqual(
            plan["optimised_plan"]["total_cost_inr"], plan["baseline_plan"]["total_cost_inr"] + 1e-6
        )

    def test_dispatch_rejects_out_of_horizon_date(self):
        with self.assertRaises(ForecastRequestError):
            plan_city_dispatch(self.forecaster, "SRT", delivery_date="2026-06-01")


if __name__ == "__main__":
    unittest.main()
