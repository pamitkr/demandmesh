"""HTTP contract tests (need ``fastapi`` + ``httpx`` from requirements-dev.txt)."""

import importlib.util
import unittest

HAS_FASTAPI = all(importlib.util.find_spec(m) for m in ("fastapi", "httpx"))


@unittest.skipUnless(HAS_FASTAPI, "fastapi/httpx not installed")
class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient

        from demandmesh.app import create_app
        from support import tiny_environment

        cls.client_cm = TestClient(create_app(settings=tiny_environment()))
        cls.client = cls.client_cm.__enter__()  # runs the lifespan (model loading)

    @classmethod
    def tearDownClass(cls):
        cls.client_cm.__exit__(None, None, None)

    def test_health_and_readiness(self):
        self.assertEqual(self.client.get("/health").json()["status"], "ok")
        ready = self.client.get("/ready")
        self.assertEqual(ready.status_code, 200)
        self.assertIn("X-Request-ID", ready.headers)

    def test_forecast_endpoint(self):
        r = self.client.post("/v1/forecast", json={"zone_ids": ["SRT-Z01"], "horizon_days": 3})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(len(body["points"]), body["n_series"] * 3)
        p = body["points"][0]
        self.assertLessEqual(p["p10"], p["p50"])
        self.assertLessEqual(p["p50"], p["p90"])

    def test_forecast_unknown_zone_is_422_with_request_id(self):
        r = self.client.post(
            "/v1/forecast", json={"zone_ids": ["XXX-Z01"]}, headers={"x-request-id": "abc123"}
        )
        self.assertEqual(r.status_code, 422)
        self.assertEqual(r.json()["error"]["request_id"], "abc123")

    def test_route_optimisation_endpoint(self):
        payload = {
            "depot": {"lat": 21.18, "lon": 72.96},
            "stops": [
                {"stop_id": "A", "lat": 21.20, "lon": 72.90, "demand_kg": 400},
                {"stop_id": "B", "lat": 21.15, "lon": 72.85, "demand_kg": 600},
                {"stop_id": "C", "lat": 21.10, "lon": 72.88, "demand_kg": 300},
            ],
            "vehicles": [{"vehicle_id": "V1"}, {"vehicle_id": "V2"}],
            "options": {"time_limit_s": 0.5},
        }
        r = self.client.post("/v1/routes/optimize", json=payload)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["optimised"]["status"], "feasible")

    def test_route_validation_rejects_bad_coordinates(self):
        payload = {"depot": {"lat": 0, "lon": 0}, "stops": [], "vehicles": []}
        self.assertEqual(self.client.post("/v1/routes/optimize", json=payload).status_code, 422)

    def test_dispatch_plan_and_monitoring(self):
        r = self.client.post("/v1/dispatch/plan", json={"city": "LDH", "options": {"time_limit_s": 0.5}})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertIn("savings_vs_baseline", r.json())
        drift = self.client.get("/v1/monitoring/drift", params={"window_days": 14})
        self.assertEqual(drift.status_code, 200)
        self.assertEqual(self.client.get("/v1/model").status_code, 200)


if __name__ == "__main__":
    unittest.main()
