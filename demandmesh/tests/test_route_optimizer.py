import itertools
import unittest

import numpy as np

from demandmesh.exceptions import RoutingError
from demandmesh.route_optimizer import (
    Depot,
    RouteOptimizer,
    RoutingConfig,
    Stop,
    Vehicle,
    haversine_matrix,
    nearest_neighbour_baseline,
)

DEPOT = Depot(lat=21.18, lon=72.96)  # Surat hub
FAST = RoutingConfig(time_limit_s=30, max_iterations=400, seed=11)


def _clock(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def _random_stops(n: int, seed: int, demand=(150, 700)) -> list[Stop]:
    rng = np.random.default_rng(seed)
    return [
        Stop(
            f"S{i:02d}",
            21.18 + rng.uniform(-0.12, 0.12),
            72.96 + rng.uniform(-0.12, 0.12),
            float(rng.uniform(*demand)),
            tw_start_min=540,
            tw_end_min=780 if i % 4 == 0 else 1140,
        )
        for i in range(n)
    ]


class FeasibilityTests(unittest.TestCase):
    def assertFeasible(self, result, stops, vehicles):  # noqa: N802
        by_id = {s.stop_id: s for s in stops}
        caps = {v.vehicle_id: v.capacity_kg for v in vehicles}
        visited = [v.stop_id for r in result.routes for v in r.stops]
        self.assertEqual(len(visited), len(set(visited)), "a stop was visited twice")
        self.assertEqual(set(visited) | set(result.unserved_stop_ids), set(by_id))
        for route in result.routes:
            self.assertLessEqual(route.load_kg, caps[route.vehicle_id] + 1e-6)
            for visit in route.stops:
                stop = by_id[visit.stop_id]
                start = _clock(visit.service_start)
                self.assertGreaterEqual(start, stop.tw_start_min - 1)
                self.assertLessEqual(start, stop.tw_end_min + 1)

    def test_all_stops_served_within_capacity_and_windows(self):
        stops = _random_stops(20, seed=1)
        fleet = [Vehicle(f"V{i}", capacity_kg=2500) for i in range(5)]
        result = RouteOptimizer(FAST).solve(DEPOT, stops, fleet)
        self.assertEqual(result.status, "feasible")
        self.assertFeasible(result, stops, fleet)

    def test_capacity_forces_two_vehicles(self):
        stops = [Stop("A", 21.20, 72.97, 1500), Stop("B", 21.21, 72.98, 1500)]
        fleet = [Vehicle("V1", capacity_kg=2000), Vehicle("V2", capacity_kg=2000)]
        result = RouteOptimizer(FAST).solve(DEPOT, stops, fleet)
        self.assertEqual(result.vehicles_used, 2)
        self.assertFeasible(result, stops, fleet)

    def test_unreachable_stop_is_reported_not_fatal(self):
        stops = [
            Stop("NEAR", 21.19, 72.97, 100),
            Stop("FAR", 22.30, 73.20, 100, tw_start_min=540, tw_end_min=545),
        ]
        result = RouteOptimizer(FAST).solve(DEPOT, stops, [Vehicle("V1")])
        self.assertEqual(result.status, "partial")
        self.assertEqual(result.unserved_stop_ids, ["FAR"])

    def test_dynamic_replanning_from_current_position(self):
        """A truck that is mid-route at 14:00 is re-planned from where it is."""
        here = (21.10, 72.88)
        stops = [Stop("X", 21.11, 72.89, 200), Stop("Y", 21.15, 72.92, 200)]
        truck = Vehicle("V1", start_lat=here[0], start_lon=here[1], available_from_min=14 * 60)
        result = RouteOptimizer(FAST).solve(DEPOT, stops, [truck])
        route = result.routes[0]
        self.assertGreaterEqual(_clock(route.stops[0].service_start), 14 * 60)
        self.assertEqual(route.stops[0].stop_id, "X")  # nearest to the current position
        self.assertFeasible(result, stops, [truck])

    def test_matches_brute_force_optimum_on_small_instance(self):
        stops = _random_stops(6, seed=5, demand=(50, 100))
        stops = [Stop(s.stop_id, s.lat, s.lon, s.demand_kg) for s in stops]  # wide windows
        cfg = RoutingConfig(time_limit_s=30, max_iterations=300, seed=3)
        result = RouteOptimizer(cfg).solve(DEPOT, stops, [Vehicle("V1", capacity_kg=5000)])
        coords = np.array([[DEPOT.lat, DEPOT.lon]] + [[s.lat, s.lon] for s in stops])
        dist = haversine_matrix(coords[:, 0], coords[:, 1]) * cfg.circuity_factor
        best = min(
            dist[0, p[0]] + sum(dist[a, b] for a, b in itertools.pairwise(p)) + dist[p[-1], 0]
            for p in itertools.permutations(range(1, len(stops) + 1))
        )
        self.assertAlmostEqual(result.total_distance_km, best, delta=0.05)

    def test_never_worse_than_nearest_neighbour(self):
        stops = _random_stops(25, seed=9)
        fleet = [Vehicle(f"V{i}") for i in range(6)]
        result = RouteOptimizer(FAST).solve(DEPOT, stops, fleet)
        baseline = nearest_neighbour_baseline(DEPOT, stops, fleet, FAST)
        self.assertLessEqual(result.total_cost_inr, baseline["total_cost_inr"] + 1e-6)

    def test_deterministic_for_a_seed(self):
        stops = _random_stops(15, seed=2)
        fleet = [Vehicle(f"V{i}") for i in range(4)]
        a = RouteOptimizer(FAST).solve(DEPOT, stops, fleet)
        b = RouteOptimizer(FAST).solve(DEPOT, stops, fleet)
        self.assertEqual(a.total_cost_inr, b.total_cost_inr)


class ValidationTests(unittest.TestCase):
    def test_oversized_stop_raises(self):
        with self.assertRaises(RoutingError):
            RouteOptimizer(FAST).solve(DEPOT, [Stop("BIG", 21.2, 72.9, 9000)], [Vehicle("V1")])

    def test_duplicate_ids_raise(self):
        stops = [Stop("A", 21.2, 72.9, 10), Stop("A", 21.3, 72.9, 10)]
        with self.assertRaises(RoutingError):
            RouteOptimizer(FAST).solve(DEPOT, stops, [Vehicle("V1")])

    def test_empty_stop_list_returns_empty_plan(self):
        result = RouteOptimizer(FAST).solve(DEPOT, [], [Vehicle("V1")])
        self.assertEqual(result.status, "empty")
        self.assertEqual(result.vehicles_used, 0)


if __name__ == "__main__":
    unittest.main()
