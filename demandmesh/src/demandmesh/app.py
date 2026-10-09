"""FastAPI service: forecasting, route optimisation and forecast-driven dispatch.

Run locally:   uvicorn demandmesh.app:app --reload
Interactive docs at /docs (Swagger UI) and /redoc.

Operational notes
-----------------
* Endpoints are plain ``def`` functions, so FastAPI runs them in its worker
  thread-pool: the CPU-bound solver never blocks the event loop.
* ``/health`` is a liveness probe; ``/ready`` is a readiness probe that fails
  until a model is loaded. Routing works even without a model (graceful
  degradation) because it does not depend on forecasts.
* Every response carries ``X-Request-ID`` and ``X-Process-Time-Ms``; errors
  return a structured body with the same request id and never leak stack traces.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse

from . import __version__
from .config import Settings, get_settings
from .exceptions import (
    DataValidationError,
    DemandMeshError,
    ForecastRequestError,
    ModelNotReadyError,
    RoutingError,
)
from .forecaster import DemandForecaster
from .logging_utils import get_logger
from .planner import default_routing_config, plan_city_dispatch
from .route_optimizer import RouteOptimizer, nearest_neighbour_baseline
from .schemas import (
    DispatchRequest,
    ForecastRequest,
    ForecastResponse,
    RouteRequest,
    RouteResponse,
)

logger = get_logger("demandmesh.api")


def _error(request: Request, status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "code": code,
                "message": message,
                "request_id": getattr(request.state, "request_id", None),
            }
        },
    )


def create_app(settings: Settings | None = None, forecaster: DemandForecaster | None = None) -> FastAPI:
    """Application factory (tests inject settings or a pre-built forecaster)."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.forecaster = forecaster
        if app.state.forecaster is None:
            try:
                app.state.forecaster = DemandForecaster.from_settings(settings)
            except DemandMeshError as exc:
                logger.error("Starting without a forecasting model: %s", exc)
        yield

    app = FastAPI(
        title="DemandMesh API",
        version=__version__,
        description=(
            "Hyper-local B2B demand forecasting (P10/P50/P90) and dynamic route "
            "optimisation (CVRPTW via ALNS) for Indian industrial clusters."
        ),
        lifespan=lifespan,
    )

    # ---- middleware --------------------------------------------------------- #
    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        request.state.request_id = request_id
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
        logger.info(
            "method=%s path=%s status=%d ms=%.1f request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            request_id,
        )
        return response

    # ---- error mapping ------------------------------------------------------ #
    @app.exception_handler(ForecastRequestError)
    async def _forecast_error(request: Request, exc: ForecastRequestError):
        return _error(request, 422, "invalid_forecast_request", str(exc))

    @app.exception_handler(RoutingError)
    async def _routing_error(request: Request, exc: RoutingError):
        return _error(request, 422, "invalid_routing_request", str(exc))

    @app.exception_handler(ModelNotReadyError)
    async def _not_ready(request: Request, exc: ModelNotReadyError):
        return _error(request, 503, "model_not_ready", str(exc))

    @app.exception_handler(DataValidationError)
    async def _data_error(request: Request, exc: DataValidationError):
        logger.error("Data error: %s", exc)
        return _error(request, 500, "data_error", "Server-side data is invalid")

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        logger.exception("Unhandled error request_id=%s", getattr(request.state, "request_id", None))
        return _error(request, 500, "internal_error", "Unexpected server error")

    def get_forecaster(request: Request) -> DemandForecaster:
        forecaster_ = getattr(request.app.state, "forecaster", None)
        if forecaster_ is None:
            raise ModelNotReadyError("No model loaded - run `python -m demandmesh.pipeline` first")
        return forecaster_

    # ---- operational endpoints --------------------------------------------- #
    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {"service": "DemandMesh", "docs": "/docs", "health": "/health"}

    @app.get("/health", tags=["ops"])
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/ready", tags=["ops"])
    def ready(request: Request):
        forecaster_ = getattr(request.app.state, "forecaster", None)
        if forecaster_ is None:
            return JSONResponse(status_code=503, content={"status": "not_ready"})
        return {"status": "ready", "model_version": forecaster_.version}

    @app.get("/v1/model", tags=["model"])
    def model_info(request: Request) -> dict[str, Any]:
        meta = get_forecaster(request).metadata
        keys = (
            "version",
            "trained_at_utc",
            "algorithm",
            "quantiles",
            "target_transform",
            "feature_spec",
            "data",
            "holdout_metrics",
            "feature_importance",
            "trees_per_quantile",
        )
        return {k: meta.get(k) for k in keys}

    @app.get("/v1/catalog", tags=["model"])
    def catalog(request: Request) -> dict[str, Any]:
        return get_forecaster(request).catalog()

    # ---- forecasting -------------------------------------------------------- #
    @app.post("/v1/forecast", response_model=ForecastResponse, tags=["forecast"])
    def forecast(body: ForecastRequest, request: Request) -> ForecastResponse:
        forecaster_ = get_forecaster(request)
        frame = forecaster_.forecast(
            horizon=body.horizon_days,
            as_of=body.as_of,
            zone_ids=body.zone_ids,
            categories=body.categories,
            cities=body.cities,
        )
        origin = body.as_of or forecaster_.last_history_date.date()
        records = frame.assign(date=frame["date"].dt.date).to_dict(orient="records")
        return ForecastResponse(
            model_version=forecaster_.version,
            as_of=origin,
            horizon_days=body.horizon_days,
            n_series=int(frame[["zone_id", "category"]].drop_duplicates().shape[0]),
            points=records,
        )

    # ---- routing ------------------------------------------------------------ #
    @app.post("/v1/routes/optimize", response_model=RouteResponse, tags=["routing"])
    def optimize_routes(body: RouteRequest) -> RouteResponse:
        depot = body.depot.to_domain()
        stops = [s.to_domain() for s in body.stops]
        vehicles = [v.to_domain() for v in body.vehicles]
        cfg = body.options.to_domain()
        result = RouteOptimizer(cfg).solve(depot, stops, vehicles)
        baseline = nearest_neighbour_baseline(depot, stops, vehicles, cfg)
        saving = None
        if baseline["total_distance_km"] > 0 and not result.unserved_stop_ids:
            saving = round(1 - result.total_distance_km / baseline["total_distance_km"], 4)
        return RouteResponse(optimised=result.to_dict(), baseline=baseline, distance_saving_pct=saving)

    @app.post("/v1/dispatch/plan", tags=["routing"])
    def dispatch_plan(body: DispatchRequest, request: Request) -> dict[str, Any]:
        return plan_city_dispatch(
            get_forecaster(request),
            city=body.city,
            delivery_date=body.delivery_date,
            service_level=body.service_level,
            vehicles=[v.to_domain() for v in body.vehicles] if body.vehicles else None,
            max_drop_kg=body.max_drop_kg,
            routing_config=(
                body.options.to_domain()
                if body.options
                else default_routing_config(settings.routing_time_limit_s)
            ),
            attach_rate=body.attach_rate,
        )

    # ---- monitoring --------------------------------------------------------- #
    @app.get("/v1/monitoring/drift", tags=["monitoring"])
    def drift(request: Request, window_days: int = Query(28, ge=7, le=180)) -> dict[str, Any]:
        return get_forecaster(request).drift_report(window_days)

    return app


app = create_app()
