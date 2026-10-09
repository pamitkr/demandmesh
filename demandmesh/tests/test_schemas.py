import unittest

from pydantic import ValidationError

from demandmesh.schemas import DispatchRequest, ForecastRequest, RouteRequest

DEPOT = {"lat": 21.18, "lon": 72.96}
STOP = {"stop_id": "S1", "lat": 21.2, "lon": 72.9, "demand_kg": 300}
VEHICLE = {"vehicle_id": "V1"}


class SchemaTests(unittest.TestCase):
    def test_valid_route_request_converts_to_domain(self):
        req = RouteRequest(depot=DEPOT, stops=[STOP], vehicles=[VEHICLE])
        self.assertEqual(req.stops[0].to_domain().demand_kg, 300)
        self.assertEqual(req.options.to_domain().time_limit_s, 2.0)

    def test_coordinates_outside_india_rejected(self):
        with self.assertRaises(ValidationError):
            RouteRequest(depot={"lat": 51.5, "lon": -0.12}, stops=[STOP], vehicles=[VEHICLE])

    def test_inverted_time_window_rejected(self):
        bad = {**STOP, "tw_start_min": 900, "tw_end_min": 600}
        with self.assertRaises(ValidationError):
            RouteRequest(depot=DEPOT, stops=[bad], vehicles=[VEHICLE])

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValidationError):
            RouteRequest(depot=DEPOT, stops=[STOP, STOP], vehicles=[VEHICLE])

    def test_half_specified_start_position_rejected(self):
        with self.assertRaises(ValidationError):
            RouteRequest(depot=DEPOT, stops=[STOP], vehicles=[{"vehicle_id": "V1", "start_lat": 21.1}])

    def test_unknown_fields_rejected(self):
        with self.assertRaises(ValidationError):
            ForecastRequest(horizon_days=3, horizn=4)

    def test_dispatch_defaults(self):
        req = DispatchRequest(city="SRT")
        self.assertEqual(req.service_level, "p90")
        self.assertAlmostEqual(req.attach_rate, 0.2)


if __name__ == "__main__":
    unittest.main()
