"""Pydantic v2 request/response contracts for the HTTP API.

Validation happens at the edge: coordinates must lie inside India's bounding
box, time windows must be ordered, ids must be unique - so the domain layer only
ever sees well-formed inputs. ``extra="forbid"`` rejects misspelt fields instead
of silently ignoring them.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import FLEET
from .route_optimizer import Depot, RoutingConfig, Stop, Vehicle

LAT_RANGE = (6.0, 37.5)  # India incl. islands and Ladakh
LON_RANGE = (68.0, 97.5)
DAY_MIN = 24 * 60


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- #
# Forecasting                                                                 #
# --------------------------------------------------------------------------- #
class ForecastRequest(_Request):
    zone_ids: list[str] | None = Field(default=None, max_length=200, examples=[["SRT-Z01"]])
    categories: list[str] | None = Field(default=None, max_length=20, examples=[["TEXTILE"]])
    cities: list[str] | None = Field(default=None, max_length=20, examples=[["SRT"]])
    horizon_days: int = Field(default=7, ge=1, le=28, description="1..model horizon (7)")
    as_of: date | None = Field(default=None, description="Forecast origin; default = last day")


class ForecastPoint(BaseModel):
    date: date
    city: str
    zone_id: str
    zone_name: str
    category: str
    horizon_day: int
    p10: float
    p50: float
    p90: float


class ForecastResponse(BaseModel):
    model_version: str
    as_of: date
    horizon_days: int
    n_series: int
    points: list[ForecastPoint]


# --------------------------------------------------------------------------- #
# Routing                                                                     #
# --------------------------------------------------------------------------- #
class _GeoPoint(_Request):
    lat: float = Field(ge=LAT_RANGE[0], le=LAT_RANGE[1])
    lon: float = Field(ge=LON_RANGE[0], le=LON_RANGE[1])


class DepotIn(_GeoPoint):
    name: str = Field(default="hub", max_length=120)
    open_min: int = Field(default=6 * 60, ge=0, le=DAY_MIN)
    close_min: int = Field(default=23 * 60, ge=0, le=DAY_MIN)

    @model_validator(mode="after")
    def _ordered(self) -> DepotIn:
        if self.open_min > self.close_min:
            raise ValueError("open_min must be <= close_min")
        return self

    def to_domain(self) -> Depot:
        return Depot(self.lat, self.lon, self.name, self.open_min, self.close_min)


class StopIn(_GeoPoint):
    stop_id: str = Field(min_length=1, max_length=64)
    demand_kg: float = Field(ge=0, le=40_000)
    service_min: float = Field(default=FLEET.service_min_per_drop, ge=0, le=240)
    tw_start_min: float = Field(default=FLEET.drop_tw_start_min, ge=0, le=DAY_MIN)
    tw_end_min: float = Field(default=FLEET.drop_tw_end_min, ge=0, le=DAY_MIN)

    @model_validator(mode="after")
    def _window(self) -> StopIn:
        if self.tw_start_min > self.tw_end_min:
            raise ValueError("tw_start_min must be <= tw_end_min")
        return self

    def to_domain(self) -> Stop:
        return Stop(
            self.stop_id,
            self.lat,
            self.lon,
            self.demand_kg,
            self.service_min,
            self.tw_start_min,
            self.tw_end_min,
        )


class VehicleIn(_Request):
    vehicle_id: str = Field(min_length=1, max_length=64)
    capacity_kg: float = Field(default=FLEET.capacity_kg, gt=0, le=40_000)
    cost_per_km_inr: float = Field(default=FLEET.cost_per_km_inr, ge=0, le=1_000)
    fixed_cost_inr: float = Field(default=FLEET.fixed_cost_inr, ge=0, le=100_000)
    start_lat: float | None = Field(
        default=None, ge=LAT_RANGE[0], le=LAT_RANGE[1], description="Current position (mid-day re-planning)"
    )
    start_lon: float | None = Field(default=None, ge=LON_RANGE[0], le=LON_RANGE[1])
    available_from_min: float = Field(default=FLEET.shift_start_min, ge=0, le=DAY_MIN)
    shift_end_min: float = Field(default=FLEET.shift_end_min, ge=0, le=DAY_MIN)

    @model_validator(mode="after")
    def _consistent(self) -> VehicleIn:
        if (self.start_lat is None) != (self.start_lon is None):
            raise ValueError("start_lat and start_lon must be given together")
        if self.available_from_min > self.shift_end_min:
            raise ValueError("available_from_min must be <= shift_end_min")
        return self

    def to_domain(self) -> Vehicle:
        return Vehicle(
            self.vehicle_id,
            self.capacity_kg,
            self.cost_per_km_inr,
            self.fixed_cost_inr,
            self.start_lat,
            self.start_lon,
            self.available_from_min,
            self.shift_end_min,
        )


class RoutingOptions(_Request):
    avg_speed_kmph: float = Field(default=FLEET.avg_speed_kmph, gt=3, le=80)
    circuity_factor: float = Field(default=FLEET.circuity_factor, ge=1.0, le=3.0)
    time_limit_s: float = Field(default=2.0, gt=0, le=30)
    seed: int = Field(default=42, ge=0)

    def to_domain(self) -> RoutingConfig:
        return RoutingConfig(
            avg_speed_kmph=self.avg_speed_kmph,
            circuity_factor=self.circuity_factor,
            co2_kg_per_km=FLEET.co2_kg_per_km,
            time_limit_s=self.time_limit_s,
            seed=self.seed,
        )


def _unique(ids: list[str], what: str) -> None:
    if len(set(ids)) != len(ids):
        raise ValueError(f"{what} ids must be unique")


class RouteRequest(_Request):
    depot: DepotIn
    stops: list[StopIn] = Field(min_length=1, max_length=500)
    vehicles: list[VehicleIn] = Field(min_length=1, max_length=200)
    options: RoutingOptions = Field(default_factory=RoutingOptions)

    @model_validator(mode="after")
    def _ids(self) -> RouteRequest:
        _unique([s.stop_id for s in self.stops], "stop")
        _unique([v.vehicle_id for v in self.vehicles], "vehicle")
        return self


class StopVisitOut(BaseModel):
    stop_id: str
    sequence: int
    arrival: str
    service_start: str
    departure: str
    wait_min: float
    load_after_kg: float
    leg_km: float


class VehicleRouteOut(BaseModel):
    vehicle_id: str
    stops: list[StopVisitOut]
    distance_km: float
    start_time: str
    end_time: str
    duration_min: float
    load_kg: float
    utilisation: float
    cost_inr: float
    co2_kg: float


class RoutingResultOut(BaseModel):
    status: str
    routes: list[VehicleRouteOut]
    unserved_stop_ids: list[str]
    vehicles_used: int
    total_distance_km: float
    total_cost_inr: float
    total_co2_kg: float
    served_demand_kg: float
    iterations: int
    solve_time_s: float
    operator_weights: dict[str, float]


class RouteResponse(BaseModel):
    optimised: RoutingResultOut
    baseline: dict
    distance_saving_pct: float | None


# --------------------------------------------------------------------------- #
# Forecast-driven dispatch                                                    #
# --------------------------------------------------------------------------- #
class DispatchRequest(_Request):
    city: str = Field(min_length=3, max_length=3, examples=["SRT"])
    delivery_date: date | None = Field(default=None, description="Default: day after history")
    service_level: Literal["p50", "p90"] = "p90"
    attach_rate: float = Field(default=FLEET.logistics_attach_rate, gt=0, le=1)
    max_drop_kg: float = Field(default=400.0, gt=0, le=40_000)
    vehicles: list[VehicleIn] | None = Field(default=None, max_length=200)
    options: RoutingOptions | None = None

    @model_validator(mode="after")
    def _ids(self) -> DispatchRequest:
        if self.vehicles:
            _unique([v.vehicle_id for v in self.vehicles], "vehicle")
        return self
