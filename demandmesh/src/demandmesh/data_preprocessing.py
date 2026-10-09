"""Data validation, cleaning and leakage-safe feature engineering.

Design principles
-----------------
1. **One feature function for training and serving.** ``build_feature_frame`` is
   called by the trainer and by the online forecaster, so there is no
   training/serving skew by construction.
2. **Horizon-safe features (direct multi-horizon strategy).** Every feature that
   uses the target or enquiries is shifted by at least the forecast horizon *H*.
   One model therefore forecasts any day in ``t+1 .. t+H`` from data available at
   ``t`` without recursive error compounding, and a unit test proves that
   perturbing the future never changes a feature.
3. **Hierarchical signals.** Zone totals and city x category totals are added so
   sparse hyper-local series borrow strength from their neighbours.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .calendar_india import CALENDAR_FEATURES, calendar_features
from .config import ALL_ZONES, CATEGORIES, CITIES, Settings, get_settings
from .exceptions import DataValidationError
from .logging_utils import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS: tuple[str, ...] = ("date", "city", "zone_id", "category", "enquiries", "orders")
SERIES_KEYS: list[str] = ["zone_id", "category"]
TARGET = "orders"
CATEGORICAL_FEATURES: tuple[str, ...] = ("city", "zone_id", "category")


# --------------------------------------------------------------------------- #
# Feature specification (persisted with every model)                          #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FeatureSpec:
    """Defines the feature set; stored in model metadata to guarantee parity."""

    horizon: int = 7
    lags: tuple[int, ...] = (7, 14, 21, 28, 35, 364)
    rolling_windows: tuple[int, ...] = (7, 14, 28, 56)
    enquiry_windows: tuple[int, ...] = (3, 7, 14, 28)

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        if any(lag < self.horizon for lag in self.lags):
            raise ValueError(f"All lags must be >= horizon ({self.horizon}) to avoid leakage")
        if any(w < 2 for w in self.rolling_windows + self.enquiry_windows):
            raise ValueError("Rolling windows must be >= 2")

    @property
    def required_history_days(self) -> int:
        """Days of history needed so every feature of a future date is defined."""
        longest_window = max(self.rolling_windows + self.enquiry_windows)
        return max(max(self.lags), self.horizon + longest_window) + self.horizon

    def to_dict(self) -> dict:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, data: dict) -> FeatureSpec:
        return cls(
            horizon=int(data["horizon"]),
            lags=tuple(int(x) for x in data["lags"]),
            rolling_windows=tuple(int(x) for x in data["rolling_windows"]),
            enquiry_windows=tuple(int(x) for x in data["enquiry_windows"]),
        )


def feature_columns(spec: FeatureSpec) -> list[str]:
    """Ordered list of model input columns for a given spec."""
    cols: list[str] = list(CATEGORICAL_FEATURES) + ["lat", "lon"] + list(CALENDAR_FEATURES)
    cols += [f"orders_lag_{k}" for k in spec.lags]
    cols += ["orders_same_dow_mean_4w"]
    for w in spec.rolling_windows:
        cols += [f"orders_roll_mean_{w}", f"orders_roll_std_{w}"]
    cols += ["orders_ewm_14", "orders_momentum_7_28"]
    cols += [f"enq_roll_mean_{w}" for w in spec.enquiry_windows]
    cols += ["enq_momentum", "enq_to_order_conv_28"]
    cols += ["zone_total_roll_mean_7", "city_cat_roll_mean_7", "series_share_of_zone_28"]
    return cols


def category_levels() -> dict[str, list[str]]:
    """Fixed categorical vocabularies - unknown values become missing at inference."""
    return {
        "city": sorted(CITIES),
        "zone_id": sorted(ALL_ZONES),
        "category": sorted(CATEGORIES),
    }


# --------------------------------------------------------------------------- #
# Loading and validation                                                      #
# --------------------------------------------------------------------------- #
def load_raw_data(path: Path | str) -> pd.DataFrame:
    """Read the raw demand extract; raise ``DataValidationError`` on bad input."""
    path = Path(path)
    if not path.exists():
        raise DataValidationError(f"Raw data file not found: {path}")
    try:
        df = pd.read_csv(path, dtype={"city": str, "zone_id": str, "category": str})
    except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
        raise DataValidationError(f"Could not parse {path}: {exc}") from exc
    validate_schema(df)
    return df


def load_zones(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise DataValidationError(f"Zone master not found: {path}")
    zones = pd.read_csv(path, dtype={"zone_id": str, "city": str, "zone_name": str})
    missing = {"zone_id", "city", "lat", "lon"} - set(zones.columns)
    if missing:
        raise DataValidationError(f"Zone master missing columns: {sorted(missing)}")
    if zones["zone_id"].duplicated().any():
        raise DataValidationError("Zone master has duplicate zone_id values")
    return zones


def validate_schema(df: pd.DataFrame) -> None:
    """Structural checks that must hold before any transformation."""
    if df.empty:
        raise DataValidationError("Input data is empty")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")


# --------------------------------------------------------------------------- #
# Cleaning                                                                    #
# --------------------------------------------------------------------------- #
def clean_data(df: pd.DataFrame, outlier_iqr_k: float = 5.0) -> tuple[pd.DataFrame, dict]:
    """Repair the raw extract and return ``(clean_df, data_quality_report)``.

    Steps: type coercion -> drop unusable rows -> de-duplicate (last write wins,
    i.e. late-arriving corrections) -> clip negative reversals -> complete the
    daily grid per series -> impute ingestion gaps causally (same weekday last
    week) -> cap bulk-order spikes with a per-series robust IQR fence.
    """
    validate_schema(df)
    report: dict[str, int | str] = {"rows_in": int(len(df))}
    out = df.loc[:, list(REQUIRED_COLUMNS)].copy()

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for col in ("city", "zone_id", "category"):
        out[col] = out[col].astype("string").str.strip()
    for col in ("enquiries", "orders"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    unusable = out["date"].isna() | out["zone_id"].isna() | out["category"].isna()
    report["rows_dropped_unusable"] = int(unusable.sum())
    out = out.loc[~unusable]

    dupes = out.duplicated(subset=["date", *SERIES_KEYS], keep="last")
    report["duplicate_rows_removed"] = int(dupes.sum())
    out = out.loc[~dupes]

    negatives = (out["orders"] < 0) | (out["enquiries"] < 0)
    report["negative_values_clipped"] = int(negatives.sum())
    out[["orders", "enquiries"]] = out[["orders", "enquiries"]].clip(lower=0)

    out = _complete_daily_grid(out, report)
    out = _impute_gaps(out, report)
    out = _cap_outliers(out, outlier_iqr_k, report)

    out = out.sort_values([*SERIES_KEYS, "date"]).reset_index(drop=True)
    for col in ("city", "zone_id", "category"):
        out[col] = out[col].astype(str)
    out = out.drop(columns=["_was_missing"])
    report.update(
        rows_out=int(len(out)),
        series=int(out.groupby(SERIES_KEYS).ngroups),
        date_min=str(out["date"].min().date()),
        date_max=str(out["date"].max().date()),
    )
    check_contiguity(out)
    logger.info("Data quality report: %s", report)
    return out, report


def _complete_daily_grid(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """Reindex every series to a gap-free daily calendar between its first/last date."""
    bounds = (
        df.groupby(SERIES_KEYS, observed=True)
        .agg(first=("date", "min"), last=("date", "max"), city=("city", "first"))
        .reset_index()
    )
    lengths = ((bounds["last"] - bounds["first"]).dt.days + 1).to_numpy()
    grid = bounds.loc[bounds.index.repeat(lengths)].reset_index(drop=True)
    starts = np.repeat(np.cumsum(lengths) - lengths, lengths)
    offsets = np.arange(lengths.sum()) - starts
    grid["date"] = grid["first"] + pd.to_timedelta(offsets, unit="D")
    grid = grid.drop(columns=["first", "last"])
    merged = grid.merge(
        df.drop(columns=["city"]), on=[*SERIES_KEYS, "date"], how="left", validate="one_to_one"
    )
    merged["_was_missing"] = merged["orders"].isna()
    report["missing_days_filled"] = int(merged["_was_missing"].sum())
    return merged


def _impute_gaps(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """Causal imputation: value from the same weekday one week earlier, else 0."""
    df = df.sort_values([*SERIES_KEYS, "date"]).reset_index(drop=True)
    grouped = df.groupby(SERIES_KEYS, observed=True, sort=False)
    for col in ("orders", "enquiries"):
        prev_week = grouped[col].shift(7)
        df[col] = df[col].fillna(prev_week).fillna(0.0).astype(float)
    report["imputed_values"] = int(df["_was_missing"].sum())
    return df


def _cap_outliers(df: pd.DataFrame, k: float, report: dict) -> pd.DataFrame:
    """Cap bulk-order spikes at Q3 + k * max(IQR, 1) per series (robust to zeros)."""
    grouped = df.groupby(SERIES_KEYS, observed=True, sort=False)
    capped_total = 0
    for col in ("orders", "enquiries"):
        q1 = grouped[col].transform(lambda s: s.quantile(0.25))
        q3 = grouped[col].transform(lambda s: s.quantile(0.75))
        fence = q3 + k * np.maximum(q3 - q1, 1.0)
        over = df[col] > fence
        capped_total += int(over.sum())
        df[col] = np.where(over, np.floor(fence), df[col])
    report["outliers_capped"] = capped_total
    return df


def check_contiguity(df: pd.DataFrame) -> None:
    """Every series must be a gap-free daily sequence (row shift == day shift)."""
    gaps = df.groupby(SERIES_KEYS, observed=True, sort=False)["date"].diff().dt.days
    bad = gaps.dropna().ne(1)
    if bad.any():
        raise DataValidationError(f"{int(bad.sum())} non-contiguous date steps found; run clean_data first")


# --------------------------------------------------------------------------- #
# Feature engineering                                                         #
# --------------------------------------------------------------------------- #
def _rolling(grouped, window: int, how: str) -> pd.Series:
    """Per-series rolling statistic on an already horizon-shifted column."""
    min_periods = max(2, window // 2)
    if how == "mean":
        return grouped.transform(lambda s: s.rolling(window, min_periods=min_periods).mean())
    return grouped.transform(lambda s: s.rolling(window, min_periods=min_periods).std())


def build_feature_frame(
    history: pd.DataFrame,
    zones: pd.DataFrame,
    spec: FeatureSpec,
    drop_warmup: bool = True,
) -> pd.DataFrame:
    """Build model features for every row of ``history``.

    ``history`` may contain future rows whose ``orders``/``enquiries`` are NaN
    (inference); horizon-safe shifting guarantees those NaNs never leak into the
    features of any row within the horizon.

    Args:
        history: cleaned, contiguous frame with REQUIRED_COLUMNS.
        zones: zone master with ``zone_id, lat, lon``.
        spec: feature specification.
        drop_warmup: drop rows lacking the longest lag (training only).
    """
    validate_schema(history)
    h = spec.horizon
    df = history.sort_values([*SERIES_KEYS, "date"]).reset_index(drop=True).copy()
    check_contiguity(df)

    df = df.merge(zones[["zone_id", "lat", "lon"]], on="zone_id", how="left", validate="many_to_one")
    df = pd.concat([df, calendar_features(df["date"])], axis=1)

    by_series = df.groupby(SERIES_KEYS, observed=True, sort=False)
    for k in spec.lags:
        df[f"orders_lag_{k}"] = by_series[TARGET].shift(k)
    weekly_lags = [by_series[TARGET].shift(7 * i) for i in range(1, 5) if 7 * i >= h]
    df["orders_same_dow_mean_4w"] = pd.concat(weekly_lags, axis=1).mean(axis=1)

    df["_orders_safe"] = by_series[TARGET].shift(h)
    df["_enq_safe"] = by_series["enquiries"].shift(h)
    safe = df.groupby(SERIES_KEYS, observed=True, sort=False)
    for w in spec.rolling_windows:
        df[f"orders_roll_mean_{w}"] = _rolling(safe["_orders_safe"], w, "mean")
        df[f"orders_roll_std_{w}"] = _rolling(safe["_orders_safe"], w, "std")
    df["orders_ewm_14"] = safe["_orders_safe"].transform(
        lambda s: s.ewm(span=14, adjust=False, ignore_na=True).mean()
    )
    df["orders_momentum_7_28"] = df["orders_roll_mean_7"] / (df["orders_roll_mean_28"] + 1.0)
    for w in spec.enquiry_windows:
        df[f"enq_roll_mean_{w}"] = _rolling(safe["_enq_safe"], w, "mean")
    short_w, long_w = min(spec.enquiry_windows), max(spec.enquiry_windows)
    df["enq_momentum"] = df[f"enq_roll_mean_{short_w}"] / (df[f"enq_roll_mean_{long_w}"] + 1.0)
    orders_28 = safe["_orders_safe"].transform(lambda s: s.rolling(28, min_periods=14).sum())
    enq_28 = safe["_enq_safe"].transform(lambda s: s.rolling(28, min_periods=14).sum())
    df["enq_to_order_conv_28"] = orders_28 / (enq_28 + 1.0)

    df = _add_hierarchical_features(df, h)
    df["series_share_of_zone_28"] = df["orders_roll_mean_28"] / (df["_zone_roll_mean_28"] + 1e-6)
    df = df.drop(columns=["_orders_safe", "_enq_safe", "_zone_roll_mean_28"])

    if drop_warmup:
        df = df.loc[df[f"orders_lag_{max(spec.lags)}"].notna()].reset_index(drop=True)
    return df


def _add_hierarchical_features(df: pd.DataFrame, h: int) -> pd.DataFrame:
    """Zone-level and city x category-level demand signals (horizon-safe)."""

    def level_signal(keys: list[str], windows: dict[str, int]) -> pd.DataFrame:
        agg = (
            df.groupby([*keys, "date"], observed=True)[TARGET]
            .sum(min_count=1)
            .reset_index()
            .sort_values([*keys, "date"])
        )
        agg["_safe"] = agg.groupby(keys, observed=True, sort=False)[TARGET].shift(h)
        grouped = agg.groupby(keys, observed=True, sort=False)["_safe"]
        for col, w in windows.items():
            agg[col] = grouped.transform(lambda s, w=w: s.rolling(w, min_periods=max(2, w // 2)).mean())
        return agg.drop(columns=[TARGET, "_safe"])

    zone = level_signal(["zone_id"], {"zone_total_roll_mean_7": 7, "_zone_roll_mean_28": 28})
    city_cat = level_signal(["city", "category"], {"city_cat_roll_mean_7": 7})
    df = df.merge(zone, on=["zone_id", "date"], how="left", validate="many_to_one")
    df = df.merge(city_cat, on=["city", "category", "date"], how="left", validate="many_to_one")
    return df


def to_model_frame(df: pd.DataFrame, columns: list[str], levels: dict[str, list[str]]) -> pd.DataFrame:
    """Select model columns and pin categorical vocabularies (train == serve)."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise DataValidationError(f"Feature frame missing columns: {missing}")
    X = df.loc[:, columns].copy()
    for col, vocab in levels.items():
        X[col] = pd.Categorical(X[col].astype(str), categories=vocab)
    numeric = [c for c in columns if c not in levels]
    X[numeric] = X[numeric].astype(float)
    return X


# --------------------------------------------------------------------------- #
# Persistence helpers + CLI                                                   #
# --------------------------------------------------------------------------- #
def save_history(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False, compression="gzip")


def load_history(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise DataValidationError(f"Processed history not found: {path}. Run preprocessing.")
    df = pd.read_csv(path, dtype={"city": str, "zone_id": str, "category": str})
    df["date"] = pd.to_datetime(df["date"])
    validate_schema(df)
    return df


def run_preprocessing(settings: Settings) -> pd.DataFrame:
    """Raw extract -> cleaned, contiguous history + data-quality report."""
    settings.ensure_dirs()
    raw = load_raw_data(settings.raw_demand_path)
    clean, report = clean_data(raw)
    save_history(clean, settings.history_path)
    settings.data_quality_report_path.write_text(json.dumps(report, indent=2))
    logger.info("Wrote cleaned history to %s", settings.history_path)
    return clean


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description="Clean raw demand data").parse_args(argv)
    run_preprocessing(get_settings())


if __name__ == "__main__":
    main()
