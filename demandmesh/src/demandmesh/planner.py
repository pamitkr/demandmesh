"""Forecast-to-dispatch planner: the step that turns predictions into rupees.

1. Forecast next-day orders for every zone x category of a city (P50 or P90).
2. Convert orders into freight weight with category consignment weights, and
   keep the share that ships through the platform's logistics partners
   (``attach_rate`` - e.g. buyers who book transport via the marketplace).
3. Split each zone's load into delivery drops (buyer premises inside the zone)
   no larger than ``max_drop_kg``; each drop gets a deterministic location near
   the zone centroid and a receiving window (some buyers accept only mornings).
4. Size the fleet, solve the CVRPTW with ALNS, and benchmark the plan against
   the nearest-neighbour dispatch rule a human dispatcher would use.

Choosing P90 instead of P50 is the newsvendor decision: provision capacity at
the quantile Cu / (Cu + Co), where Cu is the cost of being short of a truck
(spot hire at a premium, missed SLA) and Co the cost of an idle one. Series-level
P90s are combined with risk pooling, because the sum of P90s is far above the
P90 of the sum (see ``reports/metrics.md`` for the measured effect).
"""

from __future__ import annotations

import hashlib
import math
from datetime import date

import numpy as np
import pandas as pd

from .config import CATEGORIES, CITIES, FLEET
from .exceptions import ForecastRequestError, RoutingError
from .forecaster import DemandForecaster
from .route_optimizer import (
    Depot,
    RouteOptimizer,
    RoutingConfig,
    Stop,
    Vehicle,
    nearest_neighbour_baseline,
)

SERVICE_LEVELS = {"p50", "p90"}
DROP_SCATTER_KM = 1.5  # buyer premises spread around the zone centroid


def _stable_unit(*parts: str) -> float:
    """Deterministic pseudo-random number in [0, 1) derived from the inputs."""
    digest = hashlib.sha256("|".join(parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def build_drops(zone_loads: pd.DataFrame, max_drop_kg: float, delivery_day: str) -> list[Stop]:
    """Split zone freight into drops with locations and receiving windows."""
    stops: list[Stop] = []
    for row in zone_loads.itertuples(index=False):
        if row.load_kg < 1.0:
            continue
        n_drops = max(1, math.ceil(row.load_kg / max_drop_kg))
        for i in range(1, n_drops + 1):
            key = (row.zone_id, str(i), delivery_day)
            radius = DROP_SCATTER_KM * math.sqrt(_stable_unit(*key, "r"))
            angle = 2 * math.pi * _stable_unit(*key, "a")
            lat = row.lat + (radius / 111.0) * math.cos(angle)
            lon = row.lon + (radius / (111.0 * math.cos(math.radians(row.lat)))) * math.sin(angle)
            morning_only = _stable_unit(*key, "tw") < 0.25
            stops.append(
                Stop(
                    stop_id=f"{row.zone_id}-D{i:02d}",
                    lat=round(lat, 6),
                    lon=round(lon, 6),
                    demand_kg=round(row.load_kg / n_drops, 1),
                    service_min=FLEET.service_min_per_drop,
                    tw_start_min=FLEET.drop_tw_start_min,
                    tw_end_min=FLEET.morning_window_end_min if morning_only else FLEET.drop_tw_end_min,
                )
            )
    return stops


def default_routing_config(time_limit_s: float = 2.0) -> RoutingConfig:
    """Routing parameters derived from the fleet assumptions in ``config.py``."""
    return RoutingConfig(
        avg_speed_kmph=FLEET.avg_speed_kmph,
        circuity_factor=FLEET.circuity_factor,
        co2_kg_per_km=FLEET.co2_kg_per_km,
        time_limit_s=time_limit_s,
    )


def default_fleet(city: str, total_kg: float, n_vehicles: int | None = None) -> list[Vehicle]:
    """Enough identical LCVs for the load at 80% fill, plus one spare."""
    count = n_vehicles or max(1, math.ceil(total_kg / (FLEET.capacity_kg * 0.8)) + 1)
    return [
        Vehicle(
            vehicle_id=f"{city}-LCV-{i:02d}",
            capacity_kg=FLEET.capacity_kg,
            cost_per_km_inr=FLEET.cost_per_km_inr,
            fixed_cost_inr=FLEET.fixed_cost_inr,
            available_from_min=FLEET.shift_start_min,
            shift_end_min=FLEET.shift_end_min,
        )
        for i in range(1, count + 1)
    ]


def plan_city_dispatch(
    forecaster: DemandForecaster,
    city: str,
    delivery_date: date | str | None = None,
    service_level: str = "p90",
    vehicles: list[Vehicle] | None = None,
    max_drop_kg: float = 400.0,
    routing_config: RoutingConfig | None = None,
    attach_rate: float = FLEET.logistics_attach_rate,
) -> dict:
    """Forecast -> freight -> optimised routes -> savings versus the baseline."""
    if city not in CITIES:
        raise ForecastRequestError(f"Unknown city {city!r}; expected one of {sorted(CITIES)}")
    if service_level not in SERVICE_LEVELS:
        raise ForecastRequestError(f"service_level must be one of {sorted(SERVICE_LEVELS)}")
    if max_drop_kg <= 0:
        raise RoutingError("max_drop_kg must be positive")
    if not 0 < attach_rate <= 1:
        raise RoutingError("attach_rate must be in (0, 1]")

    as_of = forecaster.last_history_date
    target = pd.Timestamp(delivery_date) if delivery_date else as_of + pd.Timedelta(days=1)
    horizon = int((target - as_of).days)
    if not 1 <= horizon <= forecaster.max_horizon:
        raise ForecastRequestError(
            f"delivery_date must be 1-{forecaster.max_horizon} days after {as_of.date()} (last observed day)"
        )

    fc = forecaster.forecast(horizon=horizon, as_of=as_of, cities=[city])
    day = fc.loc[fc["date"] == target].copy()
    day["kg_per_order"] = day["category"].map(lambda c: CATEGORIES[c].kg_per_order)
    day["kg_p50"] = day["p50"] * day["kg_per_order"] * attach_rate
    day["kg_p90"] = day["p90"] * day["kg_per_order"] * attach_rate
    day["orders_dev2"] = (day["p90"] - day["p50"]) ** 2
    day["kg_dev2"] = (day["kg_p90"] - day["kg_p50"]) ** 2
    zone_summary = day.groupby(["zone_id", "zone_name"], as_index=False).agg(
        orders_p50=("p50", "sum"),
        orders_dev2=("orders_dev2", "sum"),
        kg_p50=("kg_p50", "sum"),
        kg_dev2=("kg_dev2", "sum"),
    )
    # Quantiles do not add up: summing category P90s would over-book trucks.
    # Risk pooling (independent errors): P90(sum) ~= sum(P50) + sqrt(sum((P90 - P50)^2)).
    zone_summary["orders_p90"] = zone_summary["orders_p50"] + np.sqrt(zone_summary["orders_dev2"])
    zone_summary["kg_p90"] = zone_summary["kg_p50"] + np.sqrt(zone_summary["kg_dev2"])
    zone_summary = zone_summary.drop(columns=["orders_dev2", "kg_dev2"]).round(1)
    zone_loads = zone_summary.merge(forecaster.zones[["zone_id", "lat", "lon"]], on="zone_id")
    zone_loads["load_kg"] = zone_loads[f"kg_{service_level}"]

    stops = build_drops(zone_loads, max_drop_kg, str(target.date()))
    total_kg = float(sum(s.demand_kg for s in stops))
    fleet = vehicles or default_fleet(city, total_kg)
    if max(v.capacity_kg for v in fleet) < max_drop_kg:
        raise RoutingError("max_drop_kg exceeds the largest vehicle capacity")

    spec = CITIES[city]
    depot = Depot(lat=spec.hub_lat, lon=spec.hub_lon, name=f"{spec.name} consolidation hub")
    cfg = routing_config or default_routing_config()
    result = RouteOptimizer(cfg).solve(depot, stops, fleet)
    baseline = nearest_neighbour_baseline(depot, stops, fleet, cfg)

    def pct(new: float, old: float) -> float | None:
        return round(1 - new / old, 4) if old > 0 else None

    return {
        "city": city,
        "city_name": spec.name,
        "forecast_origin": str(as_of.date()),
        "delivery_date": str(target.date()),
        "service_level": service_level,
        "model_version": forecaster.version,
        "demand": {
            "orders_p50": round(float(day["p50"].sum()), 1),
            "orders_p90_pooled": round(float(day["p50"].sum() + np.sqrt(day["orders_dev2"].sum())), 1),
            "logistics_attach_rate": attach_rate,
            "planned_freight_kg": round(total_kg, 1),
            "drops": len(stops),
            "by_zone": zone_summary.to_dict(orient="records"),
        },
        "fleet_available": len(fleet),
        "stops": [s.__dict__ for s in stops],
        "depot": depot.__dict__,
        "optimised_plan": result.to_dict(),
        "baseline_plan": baseline,
        "savings_vs_baseline": {
            "distance_pct": pct(result.total_distance_km, baseline["total_distance_km"]),
            "cost_pct": pct(result.total_cost_inr, baseline["total_cost_inr"]),
            "vehicles_saved": baseline["vehicles_used"] - result.vehicles_used,
            "co2_kg_saved": round(baseline["total_co2_kg"] - result.total_co2_kg, 2),
            "comparable": not result.unserved_stop_ids and not baseline["unserved_stop_ids"],
        },
        "assumptions": {
            "cost_per_km_inr": FLEET.cost_per_km_inr,
            "fixed_cost_per_vehicle_inr": FLEET.fixed_cost_inr,
            "avg_speed_kmph": cfg.avg_speed_kmph,
            "circuity_factor": cfg.circuity_factor,
            "max_drop_kg": max_drop_kg,
            "drop_locations": "synthetic buyer premises within 1.5 km of the zone centroid",
        },
    }


def summarise_plan(plan: dict) -> pd.DataFrame:
    """One row per vehicle - handy for notebooks and reports."""
    rows = [
        {
            "vehicle": r["vehicle_id"],
            "drops": len(r["stops"]),
            "load_kg": r["load_kg"],
            "utilisation": r["utilisation"],
            "km": r["distance_km"],
            "start": r["start_time"],
            "end": r["end_time"],
            "cost_inr": r["cost_inr"],
        }
        for r in plan["optimised_plan"]["routes"]
    ]
    return pd.DataFrame(rows)


__all__ = [
    "plan_city_dispatch",
    "build_drops",
    "default_fleet",
    "default_routing_config",
    "summarise_plan",
]
