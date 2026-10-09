"""Online inference: load the registered model and forecast any zone x category.

The processed history file plays the role of a feature store. Forecasts are
built with exactly the same ``build_feature_frame`` used in training, on a
window that covers the longest lag, so serving features are identical to
training features by construction.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ALL_ZONES, Settings
from .data_preprocessing import (
    SERIES_KEYS,
    FeatureSpec,
    _complete_daily_grid,
    build_feature_frame,
    load_history,
    load_zones,
    to_model_frame,
)
from .drift import compute_drift_report
from .exceptions import ForecastRequestError
from .logging_utils import get_logger
from .model_training import load_artifact, predict_quantiles, resolve_model_dir

logger = get_logger(__name__)


class DemandForecaster:
    """Thread-safe (read-only after construction) quantile demand forecaster."""

    def __init__(self, model_root: Path, history_path: Path, zones_path: Path) -> None:
        self.model_dir = resolve_model_dir(model_root)
        bundle, self.metadata, self.reference_profile = load_artifact(self.model_dir)
        self.models = bundle["models"]
        self.interval_adjustment = float(bundle.get("interval_adjustment", 0.0))
        self.target_transform = str(bundle.get("target_transform", "none"))
        self.spec = FeatureSpec.from_dict(self.metadata["feature_spec"])
        self.columns: list[str] = self.metadata["feature_columns"]
        self.levels: dict[str, list[str]] = self.metadata["category_levels"]
        self.history = load_history(history_path)
        self.zones = load_zones(zones_path)
        self.series = self.history[["city", *SERIES_KEYS]].drop_duplicates().reset_index(drop=True)
        logger.info(
            "Loaded model %s | history %s -> %s | %d series",
            self.version,
            self.first_history_date.date(),
            self.last_history_date.date(),
            len(self.series),
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> DemandForecaster:
        return cls(settings.model_dir, settings.history_path, settings.zones_path)

    # ---- properties --------------------------------------------------------- #
    @property
    def version(self) -> str:
        return str(self.metadata["version"])

    @property
    def max_horizon(self) -> int:
        return self.spec.horizon

    @property
    def first_history_date(self) -> pd.Timestamp:
        return self.history["date"].min()

    @property
    def last_history_date(self) -> pd.Timestamp:
        return self.history["date"].max()

    def catalog(self) -> dict:
        zones = self.zones.merge(self.series[["zone_id"]].drop_duplicates(), on="zone_id")
        return {
            "cities": sorted(self.series["city"].unique().tolist()),
            "categories": sorted(self.series["category"].unique().tolist()),
            "zones": zones.sort_values("zone_id").to_dict(orient="records"),
            "history_end": str(self.last_history_date.date()),
            "max_horizon_days": self.max_horizon,
        }

    # ---- forecasting -------------------------------------------------------- #
    def forecast(
        self,
        horizon: int = 7,
        as_of: date | str | None = None,
        zone_ids: list[str] | None = None,
        categories: list[str] | None = None,
        cities: list[str] | None = None,
    ) -> pd.DataFrame:
        """Return P10/P50/P90 daily orders for ``as_of + 1 .. as_of + horizon``."""
        if not 1 <= horizon <= self.max_horizon:
            raise ForecastRequestError(f"horizon must be between 1 and {self.max_horizon} days")
        origin = pd.Timestamp(as_of) if as_of is not None else self.last_history_date
        earliest = self.first_history_date + pd.Timedelta(days=self.spec.required_history_days)
        if origin > self.last_history_date:
            raise ForecastRequestError(
                f"as_of {origin.date()} is after the last observed day {self.last_history_date.date()}"
            )
        if origin < earliest:
            raise ForecastRequestError(f"as_of must be on or after {earliest.date()}")

        selected = self._select_series(zone_ids, categories, cities)
        # Features are computed for whole cities so zone / city aggregates are complete.
        city_set = set(selected["city"])
        window_start = origin - pd.Timedelta(days=self.spec.required_history_days)
        hist = self.history.loc[
            self.history["city"].isin(city_set)
            & (self.history["date"] > window_start)
            & (self.history["date"] <= origin)
        ]
        city_series = self.series.loc[self.series["city"].isin(city_set)]
        future_dates = pd.date_range(origin + pd.Timedelta(days=1), periods=horizon, freq="D")
        future = city_series.merge(pd.DataFrame({"date": future_dates}), how="cross")
        future[["enquiries", "orders"]] = np.nan

        frame = pd.concat([hist, future], ignore_index=True)
        # series that went quiet before ``origin`` get explicit NaN days -> contiguous
        frame = _complete_daily_grid(frame, report={}).drop(columns=["_was_missing"])
        features = build_feature_frame(frame, self.zones, self.spec, drop_warmup=False)
        rows = features.loc[features["date"] > origin].merge(
            selected[SERIES_KEYS], on=SERIES_KEYS, how="inner"
        )
        preds = predict_quantiles(
            self.models,
            to_model_frame(rows, self.columns, self.levels),
            self.interval_adjustment,
            self.target_transform,
        )
        out = rows[["date", "city", "zone_id", "category"]].copy()
        out["zone_name"] = out["zone_id"].map(lambda z: ALL_ZONES[z].name if z in ALL_ZONES else z)
        out["horizon_day"] = (out["date"] - origin).dt.days
        out[["p10", "p50", "p90"]] = np.round(preds, 2)
        return out.sort_values(["date", "zone_id", "category"]).reset_index(drop=True)

    def _select_series(
        self, zone_ids: list[str] | None, categories: list[str] | None, cities: list[str] | None
    ) -> pd.DataFrame:
        selected = self.series
        for column, values in (("zone_id", zone_ids), ("category", categories), ("city", cities)):
            if values:
                unknown = sorted(set(values) - set(self.series[column]))
                if unknown:
                    raise ForecastRequestError(f"Unknown {column}: {unknown}")
                selected = selected.loc[selected[column].isin(values)]
        if selected.empty:
            raise ForecastRequestError("No series match the requested filters")
        return selected

    # ---- monitoring --------------------------------------------------------- #
    def drift_report(self, window_days: int = 28) -> dict:
        """PSI of the most recent ``window_days`` against the training profile."""
        if not 7 <= window_days <= 180:
            raise ForecastRequestError("window_days must be between 7 and 180")
        last = self.last_history_date
        start = last - pd.Timedelta(days=window_days + self.spec.required_history_days)
        frame = self.history.loc[self.history["date"] > start]
        features = build_feature_frame(frame, self.zones, self.spec, drop_warmup=False)
        current = features.loc[features["date"] > last - pd.Timedelta(days=window_days)]
        report = compute_drift_report(current, self.reference_profile)
        report.update(
            model_version=self.version,
            window=[str((last - pd.Timedelta(days=window_days - 1)).date()), str(last.date())],
        )
        return report
