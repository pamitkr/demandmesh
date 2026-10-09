"""Dynamic vehicle routing: CVRPTW solved with Adaptive Large Neighbourhood Search.

Problem
-------
Capacitated Vehicle Routing Problem with Time Windows (CVRPTW) and

* a **heterogeneous fleet** (capacity, cost per km, fixed daily cost per vehicle);
* **dynamic re-planning** - each vehicle may start from its *current* GPS
  position at its *current* time, so the same solver re-optimises mid-day when
  new orders arrive or a truck breaks down;
* **soft coverage** - a stop that no vehicle can feasibly reach is reported as
  unserved (with a penalty in the objective) instead of failing the whole plan.

    minimise  sum_k [F_k * used_k + c_k * dist_k]  +  P * |unserved|
    s.t.      load_k <= Q_k,  a_i <= service_start_i <= b_i,  return_k <= shift_end_k

Algorithm
---------
ALNS (Ropke & Pisinger, 2006): repeatedly *destroy* part of the solution
(random / worst-cost / Shaw-related / whole-route removal) and *repair* it
(greedy or regret-2 cheapest feasible insertion). Operators that produce better
solutions are chosen more often (adaptive roulette weights); worse solutions
are sometimes accepted under a simulated-annealing schedule to escape local
optima. The search is anytime: it returns the best plan found within the time
budget. With an iteration budget it is fully deterministic for a given seed; a
wall-clock limit may stop at different iterations on different machines.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .exceptions import RoutingError

EARTH_RADIUS_KM = 6371.0088


# --------------------------------------------------------------------------- #
# Public data model                                                           #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Depot:
    lat: float
    lon: float
    name: str = "hub"
    open_min: int = 6 * 60
    close_min: int = 23 * 60


@dataclass(frozen=True)
class Stop:
    stop_id: str
    lat: float
    lon: float
    demand_kg: float
    service_min: float = 20.0
    tw_start_min: float = 9 * 60
    tw_end_min: float = 19 * 60


@dataclass(frozen=True)
class Vehicle:
    vehicle_id: str
    capacity_kg: float = 4000.0
    cost_per_km_inr: float = 30.0
    fixed_cost_inr: float = 1800.0
    start_lat: float | None = None  # current position for mid-day re-planning
    start_lon: float | None = None
    available_from_min: float = 8 * 60
    shift_end_min: float = 21 * 60


@dataclass(frozen=True)
class RoutingConfig:
    avg_speed_kmph: float = 20.0
    circuity_factor: float = 1.4
    co2_kg_per_km: float = 0.45
    unserved_penalty_inr: float = 25_000.0
    time_limit_s: float = 2.0
    max_iterations: int = 25_000
    seed: int = 42
    min_destroy_fraction: float = 0.10
    max_destroy_fraction: float = 0.35
    segment_length: int = 50
    reaction_factor: float = 0.2
    start_temperature_pct: float = 0.03
    end_temperature_ratio: float = 0.002


@dataclass
class StopVisit:
    stop_id: str
    sequence: int
    arrival: str
    service_start: str
    departure: str
    wait_min: float
    load_after_kg: float
    leg_km: float


@dataclass
class VehicleRoute:
    vehicle_id: str
    stops: list[StopVisit]
    distance_km: float
    start_time: str
    end_time: str
    duration_min: float
    load_kg: float
    utilisation: float
    cost_inr: float
    co2_kg: float


@dataclass
class RoutingResult:
    status: str
    routes: list[VehicleRoute]
    unserved_stop_ids: list[str]
    vehicles_used: int
    total_distance_km: float
    total_cost_inr: float
    total_co2_kg: float
    served_demand_kg: float
    iterations: int
    solve_time_s: float
    operator_weights: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Geometry                                                                    #
# --------------------------------------------------------------------------- #
def haversine_matrix(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Great-circle distance matrix in km."""
    lat = np.radians(np.asarray(lats, float))
    lon = np.radians(np.asarray(lons, float))
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def fmt_clock(minutes: float) -> str:
    minutes = int(round(minutes))
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


# --------------------------------------------------------------------------- #
# Internal instance + solution                                                #
# --------------------------------------------------------------------------- #
class _Instance:
    """Validated, array-backed problem instance. Node 0 = depot, 1..V = vehicle
    start positions, V+1.. = stops (stop index s lives at node ``offset + s``)."""

    def __init__(self, depot: Depot, stops: list[Stop], vehicles: list[Vehicle], cfg: RoutingConfig) -> None:
        self._validate(depot, stops, vehicles, cfg)
        self.depot, self.stops, self.vehicles, self.cfg = depot, stops, vehicles, cfg
        self.n, self.v = len(stops), len(vehicles)
        self.offset = 1 + self.v

        lats = (
            [depot.lat]
            + [veh.start_lat if veh.start_lat is not None else depot.lat for veh in vehicles]
            + [s.lat for s in stops]
        )
        lons = (
            [depot.lon]
            + [veh.start_lon if veh.start_lon is not None else depot.lon for veh in vehicles]
            + [s.lon for s in stops]
        )
        dist = haversine_matrix(np.array(lats), np.array(lons)) * cfg.circuity_factor
        self.dist: list[list[float]] = dist.tolist()
        self.max_dist = float(dist.max()) or 1.0
        self.time: list[list[float]] = (dist / cfg.avg_speed_kmph * 60.0).tolist()

        self.demand = [float(s.demand_kg) for s in stops]
        self.service = [float(s.service_min) for s in stops]
        self.tw_start = [float(s.tw_start_min) for s in stops]
        self.tw_end = [float(s.tw_end_min) for s in stops]
        self.capacity = [float(veh.capacity_kg) for veh in vehicles]
        self.cpk = [float(veh.cost_per_km_inr) for veh in vehicles]
        self.fixed = [float(veh.fixed_cost_inr) for veh in vehicles]
        self.start_node = list(range(1, self.v + 1))
        self.avail = [
            max(float(veh.available_from_min), float(depot.open_min))
            if veh.start_lat is None
            else float(veh.available_from_min)
            for veh in vehicles
        ]
        self.shift_end = [min(float(veh.shift_end_min), float(depot.close_min)) for veh in vehicles]

    @staticmethod
    def _validate(depot: Depot, stops: list[Stop], vehicles: list[Vehicle], cfg: RoutingConfig):
        if not vehicles:
            raise RoutingError("At least one vehicle is required")
        if cfg.avg_speed_kmph <= 0 or cfg.circuity_factor < 1:
            raise RoutingError("avg_speed_kmph must be > 0 and circuity_factor >= 1")
        if len({v.vehicle_id for v in vehicles}) != len(vehicles):
            raise RoutingError("vehicle_id values must be unique")
        if len({s.stop_id for s in stops}) != len(stops):
            raise RoutingError("stop_id values must be unique")
        max_cap = max(v.capacity_kg for v in vehicles)
        for veh in vehicles:
            if veh.capacity_kg <= 0:
                raise RoutingError(f"{veh.vehicle_id}: capacity must be positive")
            if (veh.start_lat is None) != (veh.start_lon is None):
                raise RoutingError(f"{veh.vehicle_id}: give both start_lat and start_lon or neither")
            if veh.available_from_min > veh.shift_end_min:
                raise RoutingError(f"{veh.vehicle_id}: available_from is after shift_end")
        for s in stops:
            if s.demand_kg < 0:
                raise RoutingError(f"{s.stop_id}: demand cannot be negative")
            if s.demand_kg > max_cap:
                raise RoutingError(
                    f"{s.stop_id}: demand {s.demand_kg:.0f} kg exceeds the largest vehicle "
                    f"({max_cap:.0f} kg); split it into several drops"
                )
            if s.tw_start_min > s.tw_end_min:
                raise RoutingError(f"{s.stop_id}: time window start is after its end")
        for lat, lon in [(depot.lat, depot.lon)] + [(s.lat, s.lon) for s in stops]:
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                raise RoutingError(f"Invalid coordinate ({lat}, {lon})")

    def route_cost(self, k: int, seq: list[int]) -> float | None:
        """Cost of vehicle ``k`` serving ``seq`` in order, or None if infeasible."""
        if not seq:
            return 0.0
        load = 0.0
        for s in seq:
            load += self.demand[s]
        if load > self.capacity[k] + 1e-9:
            return None
        dist_m, time_m, off = self.dist, self.time, self.offset
        prev, t, dist = self.start_node[k], self.avail[k], 0.0
        for s in seq:
            node = off + s
            dist += dist_m[prev][node]
            t += time_m[prev][node]
            if t < self.tw_start[s]:
                t = self.tw_start[s]
            elif t > self.tw_end[s]:
                return None
            t += self.service[s]
            prev = node
        dist += dist_m[prev][0]
        t += time_m[prev][0]
        if t > self.shift_end[k]:
            return None
        return dist * self.cpk[k] + self.fixed[k]


class _Solution:
    __slots__ = ("routes", "costs", "loads", "unassigned")

    def __init__(self, routes, costs, loads, unassigned):
        self.routes: list[list[int]] = routes
        self.costs: list[float] = costs
        self.loads: list[float] = loads
        self.unassigned: set[int] = unassigned

    @classmethod
    def empty(cls, inst: _Instance) -> _Solution:
        return cls([[] for _ in range(inst.v)], [0.0] * inst.v, [0.0] * inst.v, set(range(inst.n)))

    def copy(self) -> _Solution:
        return _Solution([r[:] for r in self.routes], self.costs[:], self.loads[:], set(self.unassigned))

    def objective(self, penalty: float) -> float:
        return sum(self.costs) + penalty * len(self.unassigned)


# --------------------------------------------------------------------------- #
# Solver                                                                      #
# --------------------------------------------------------------------------- #
class RouteOptimizer:
    """ALNS solver for the CVRPTW. Thread-safe: no state is kept between calls."""

    DESTROY_OPS = ("random", "worst", "shaw", "route")
    REPAIR_OPS = ("greedy", "regret2")
    SCORE_BEST, SCORE_IMPROVED, SCORE_ACCEPTED = 33.0, 9.0, 13.0

    def __init__(self, config: RoutingConfig | None = None) -> None:
        self.config = config or RoutingConfig()

    def solve(self, depot: Depot, stops: list[Stop], vehicles: list[Vehicle]) -> RoutingResult:
        started = time.perf_counter()
        cfg = self.config
        inst = _Instance(depot, list(stops), list(vehicles), cfg)
        if inst.n == 0:
            return self._build_result(inst, _Solution.empty(inst), 0, started, {})

        rng = np.random.default_rng(cfg.seed)
        penalty = cfg.unserved_penalty_inr
        current = _Solution.empty(inst)
        self._repair(inst, current, rng, regret=True)
        best = current.copy()
        best_obj = cur_obj = current.objective(penalty)

        temperature = max(cfg.start_temperature_pct * cur_obj / math.log(2), 1e-6)
        cooling = cfg.end_temperature_ratio ** (1.0 / max(cfg.max_iterations, 1))
        w_destroy = np.ones(len(self.DESTROY_OPS))
        w_repair = np.ones(len(self.REPAIR_OPS))
        s_destroy, u_destroy = np.zeros_like(w_destroy), np.zeros_like(w_destroy)
        s_repair, u_repair = np.zeros_like(w_repair), np.zeros_like(w_repair)
        q_min = max(1, int(cfg.min_destroy_fraction * inst.n))
        q_max = max(q_min, int(math.ceil(cfg.max_destroy_fraction * inst.n)))
        deadline = started + cfg.time_limit_s

        iteration = 0
        while iteration < cfg.max_iterations and time.perf_counter() < deadline:
            iteration += 1
            d_op = int(rng.choice(len(w_destroy), p=w_destroy / w_destroy.sum()))
            r_op = int(rng.choice(len(w_repair), p=w_repair / w_repair.sum()))
            candidate = current.copy()
            q = int(rng.integers(q_min, q_max + 1))
            self._destroy(inst, candidate, self.DESTROY_OPS[d_op], q, rng)
            self._repair(inst, candidate, rng, regret=self.REPAIR_OPS[r_op] == "regret2")
            cand_obj = candidate.objective(penalty)

            score = 0.0
            if cand_obj < best_obj - 1e-9:
                best, best_obj = candidate.copy(), cand_obj
                current, cur_obj, score = candidate, cand_obj, self.SCORE_BEST
            elif cand_obj < cur_obj - 1e-9:
                current, cur_obj, score = candidate, cand_obj, self.SCORE_IMPROVED
            elif cand_obj > cur_obj + 1e-9 and rng.random() < math.exp(-(cand_obj - cur_obj) / temperature):
                current, cur_obj, score = candidate, cand_obj, self.SCORE_ACCEPTED

            s_destroy[d_op] += score
            u_destroy[d_op] += 1
            s_repair[r_op] += score
            u_repair[r_op] += 1
            temperature *= cooling
            if iteration % cfg.segment_length == 0:
                self._update_weights(w_destroy, s_destroy, u_destroy, cfg.reaction_factor)
                self._update_weights(w_repair, s_repair, u_repair, cfg.reaction_factor)

        weights = {
            f"destroy:{n}": round(float(w), 3) for n, w in zip(self.DESTROY_OPS, w_destroy, strict=True)
        }
        weights.update(
            {f"repair:{n}": round(float(w), 3) for n, w in zip(self.REPAIR_OPS, w_repair, strict=True)}
        )
        return self._build_result(inst, best, iteration, started, weights)

    # ---- adaptive weights --------------------------------------------------- #
    @staticmethod
    def _update_weights(weights, scores, uses, rho) -> None:
        used = uses > 0
        weights[used] = (1 - rho) * weights[used] + rho * scores[used] / uses[used]
        weights[:] = np.maximum(weights, 0.05)
        scores[:] = 0.0
        uses[:] = 0.0

    # ---- destroy ------------------------------------------------------------ #
    def _destroy(self, inst: _Instance, sol: _Solution, op: str, q: int, rng) -> None:
        assigned = [(k, s) for k, route in enumerate(sol.routes) for s in route]
        if not assigned:
            return
        q = min(q, len(assigned))
        if op == "random":
            picks = rng.choice(len(assigned), size=q, replace=False)
            victims = [assigned[i][1] for i in picks]
        elif op == "worst":
            gains = []
            for k, route in enumerate(sol.routes):
                for pos, s in enumerate(route):
                    reduced = inst.route_cost(k, route[:pos] + route[pos + 1 :])
                    gains.append((sol.costs[k] - (reduced or 0.0), s))
            gains.sort(reverse=True)
            victims = self._biased_pick([s for _, s in gains], q, rng, power=3.0)
        elif op == "shaw":
            seed = assigned[int(rng.integers(len(assigned)))][1]
            max_d = inst.max_dist
            span = max(max(inst.tw_end) - min(inst.tw_start), 1.0)
            max_q = max(max(inst.demand), 1.0)
            off = inst.offset
            related = sorted(
                (s for _, s in assigned),
                key=lambda s: (
                    inst.dist[off + seed][off + s] / max_d
                    + abs(inst.tw_start[seed] - inst.tw_start[s]) / span
                    + abs(inst.demand[seed] - inst.demand[s]) / max_q
                ),
            )
            victims = self._biased_pick(related, q, rng, power=6.0)
        else:  # "route": empty a whole route - the move that frees a vehicle
            non_empty = [k for k, r in enumerate(sol.routes) if r]
            k = non_empty[int(rng.integers(len(non_empty)))]
            victims = list(sol.routes[k])
        self._remove(inst, sol, set(victims))

    @staticmethod
    def _biased_pick(ranked: list[int], q: int, rng, power: float) -> list[int]:
        pool, chosen = list(ranked), []
        while pool and len(chosen) < q:
            idx = int(len(pool) * rng.random() ** power)
            chosen.append(pool.pop(idx))
        return chosen

    @staticmethod
    def _remove(inst: _Instance, sol: _Solution, victims: set[int]) -> None:
        for k, route in enumerate(sol.routes):
            if any(s in victims for s in route):
                kept = [s for s in route if s not in victims]
                cost = inst.route_cost(k, kept)
                if cost is None:  # cannot happen (removal keeps feasibility); be defensive
                    victims |= set(kept)
                    kept, cost = [], 0.0
                sol.routes[k] = kept
                sol.costs[k] = cost
                sol.loads[k] = sum(inst.demand[s] for s in kept)
        sol.unassigned |= victims

    # ---- repair ------------------------------------------------------------- #
    @staticmethod
    def _best_insertion(inst: _Instance, sol: _Solution, s: int, k: int):
        if sol.loads[k] + inst.demand[s] > inst.capacity[k] + 1e-9:
            return None
        route, base, best = sol.routes[k], sol.costs[k], None
        for pos in range(len(route) + 1):
            cost = inst.route_cost(k, route[:pos] + [s] + route[pos:])
            if cost is not None and (best is None or cost - base < best[0]):
                best = (cost - base, k, pos)
        return best

    def _repair(self, inst: _Instance, sol: _Solution, rng, regret: bool) -> None:
        pending = sorted(sol.unassigned)
        rng.shuffle(pending)
        options = {s: [self._best_insertion(inst, sol, s, k) for k in range(inst.v)] for s in pending}
        while pending:
            chosen, chosen_key = None, None
            for s in pending:
                feasible = sorted(o for o in options[s] if o is not None)
                if not feasible:
                    continue
                if regret:
                    gap = feasible[1][0] - feasible[0][0] if len(feasible) > 1 else 1e12
                    key = (-gap, feasible[0][0])
                else:
                    key = (feasible[0][0], 0.0)
                if chosen_key is None or key < chosen_key:
                    chosen, chosen_key = (s, feasible[0]), key
            if chosen is None:
                break
            s, (delta, k, pos) = chosen
            sol.routes[k].insert(pos, s)
            sol.costs[k] += delta
            sol.loads[k] += inst.demand[s]
            pending.remove(s)
            del options[s]
            for other in pending:
                options[other][k] = self._best_insertion(inst, sol, other, k)
        sol.unassigned = set(pending)

    # ---- reporting ---------------------------------------------------------- #
    def _build_result(
        self, inst: _Instance, sol: _Solution, iterations: int, started: float, weights: dict[str, float]
    ) -> RoutingResult:
        cfg, routes = self.config, []
        for k, seq in enumerate(sol.routes):
            if not seq:
                continue
            veh, off = inst.vehicles[k], inst.offset
            first = off + seq[0]
            arrival_first = inst.avail[k] + inst.time[inst.start_node[k]][first]
            # leave later instead of idling at the first stop (does not change any later time)
            depart = inst.avail[k] + max(0.0, inst.tw_start[seq[0]] - arrival_first)
            t, prev, load, dist, visits = depart, inst.start_node[k], 0.0, 0.0, []
            for i, s in enumerate(seq, start=1):
                node = off + s
                leg = inst.dist[prev][node]
                arrival = t + inst.time[prev][node]
                start = max(arrival, inst.tw_start[s])
                load += inst.demand[s]
                dist += leg
                visits.append(
                    StopVisit(
                        stop_id=inst.stops[s].stop_id,
                        sequence=i,
                        arrival=fmt_clock(arrival),
                        service_start=fmt_clock(start),
                        departure=fmt_clock(start + inst.service[s]),
                        wait_min=round(start - arrival, 1),
                        load_after_kg=round(load, 1),
                        leg_km=round(leg, 2),
                    )
                )
                t, prev = start + inst.service[s], node
            dist += inst.dist[prev][0]
            end = t + inst.time[prev][0]
            routes.append(
                VehicleRoute(
                    vehicle_id=veh.vehicle_id,
                    stops=visits,
                    distance_km=round(dist, 2),
                    start_time=fmt_clock(depart),
                    end_time=fmt_clock(end),
                    duration_min=round(end - depart, 1),
                    load_kg=round(load, 1),
                    utilisation=round(load / inst.capacity[k], 3),
                    cost_inr=round(dist * inst.cpk[k] + inst.fixed[k], 2),
                    co2_kg=round(dist * cfg.co2_kg_per_km, 2),
                )
            )
        unserved = sorted(inst.stops[s].stop_id for s in sol.unassigned)
        total_km = sum(r.distance_km for r in routes)
        status = "empty" if inst.n == 0 else ("partial" if unserved else "feasible")
        return RoutingResult(
            status=status,
            routes=routes,
            unserved_stop_ids=unserved,
            vehicles_used=len(routes),
            total_distance_km=round(total_km, 2),
            total_cost_inr=round(sum(r.cost_inr for r in routes), 2),
            total_co2_kg=round(total_km * cfg.co2_kg_per_km, 2),
            served_demand_kg=round(sum(r.load_kg for r in routes), 1),
            iterations=iterations,
            solve_time_s=round(time.perf_counter() - started, 3),
            operator_weights=weights,
        )


# --------------------------------------------------------------------------- #
# Baseline used to quantify savings                                           #
# --------------------------------------------------------------------------- #
def nearest_neighbour_baseline(
    depot: Depot, stops: list[Stop], vehicles: list[Vehicle], config: RoutingConfig | None = None
) -> dict:
    """Dispatcher's rule of thumb: fill a truck by always driving to the nearest
    feasible stop (capacity + time windows respected), return, send the next truck.
    Uses the largest vehicle type, departing from the hub, with an unlimited fleet."""
    cfg = config or RoutingConfig()
    template = max(vehicles, key=lambda v: v.capacity_kg)
    hub_vehicle = Vehicle(
        vehicle_id="baseline",
        capacity_kg=template.capacity_kg,
        cost_per_km_inr=template.cost_per_km_inr,
        fixed_cost_inr=template.fixed_cost_inr,
        available_from_min=template.available_from_min,
        shift_end_min=template.shift_end_min,
    )
    inst = _Instance(depot, list(stops), [hub_vehicle], cfg)
    remaining, trips, total_km, unserved = set(range(inst.n)), 0, 0.0, []
    while remaining:
        t, cur, load, served, km = inst.avail[0], 0, 0.0, 0, 0.0
        while True:
            best = None
            for s in sorted(remaining):
                if load + inst.demand[s] > inst.capacity[0] + 1e-9:
                    continue
                node = inst.offset + s
                start = max(t + inst.time[cur][node], inst.tw_start[s])
                if start > inst.tw_end[s] or start + inst.service[s] + inst.time[node][0] > inst.shift_end[0]:
                    continue
                if best is None or inst.dist[cur][node] < best[0]:
                    best = (inst.dist[cur][node], s, start)
            if best is None:
                break
            leg, s, start = best
            km += leg
            load += inst.demand[s]
            t, cur = start + inst.service[s], inst.offset + s
            remaining.discard(s)
            served += 1
        if served == 0:
            unserved = sorted(inst.stops[s].stop_id for s in remaining)
            break
        km += inst.dist[cur][0]
        total_km += km
        trips += 1
    return {
        "method": "nearest-neighbour dispatch (capacity + time-window aware)",
        "vehicles_used": trips,
        "total_distance_km": round(total_km, 2),
        "total_cost_inr": round(
            total_km * hub_vehicle.cost_per_km_inr + trips * hub_vehicle.fixed_cost_inr, 2
        ),
        "total_co2_kg": round(total_km * cfg.co2_kg_per_km, 2),
        "unserved_stop_ids": unserved,
    }
