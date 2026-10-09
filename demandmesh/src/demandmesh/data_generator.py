"""Synthetic B2B marketplace demand simulator for Indian industrial clusters.

Real marketplace data is proprietary, so the pipeline ships with a simulator
whose structure mirrors what a B2B marketplace actually observes:

* **Hyper-local series** - daily orders per (industrial zone x product category).
* **Buyer enquiries as a leading indicator** - enquiries (RFQs) react to a local
  demand shock immediately; orders convert ~8 days later.
* **Indian seasonality** - B2B weekly cycle (Sunday shutdown), pre-Diwali
  ramp-up and Diwali-week shutdown, national holidays, fiscal-year-end (March)
  procurement push, quarter-end pushes, monsoon slowdown in construction and a
  kharif bump in agri-inputs.
* **Cluster specialisation** - e.g. Surat textiles, Ludhiana machinery.
* **Realistic data-quality defects** - missing records, duplicate rows, bulk-order
  spikes and negative reversals, which the preprocessing layer must repair.

Every random draw goes through one seeded generator, so a run is reproducible.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable

import numpy as np
import pandas as pd

from .calendar_india import calendar_features
from .config import CATEGORIES, CITIES, Settings, get_settings
from .logging_utils import get_logger

logger = get_logger(__name__)

# Monday..Sunday multipliers for B2B trade (Saturday half-day, Sunday largely shut).
WEEKLY_PROFILE = np.array([1.06, 1.10, 1.08, 1.05, 1.01, 0.86, 0.38])
BASE_ORDERS_PER_DAY = 5.0
ENQUIRY_TO_ORDER_LEAD_DAYS = 8
DEFAULT_START = "2023-10-01"
DEFAULT_END = "2026-09-30"


def _ar1(rng: np.random.Generator, n: int, phi: float, sigma: float) -> np.ndarray:
    """Stationary AR(1) process - models persistent local demand shocks."""
    eps = rng.normal(0.0, sigma, n)
    out = np.empty(n)
    out[0] = eps[0] / np.sqrt(1.0 - phi**2)
    for t in range(1, n):
        out[t] = phi * out[t - 1] + eps[t]
    return out


def _lagged(values: np.ndarray, lag: int) -> np.ndarray:
    out = np.zeros_like(values)
    out[lag:] = values[:-lag]
    return out


def generate_dataset(
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    seed: int = 42,
    cities: Iterable[str] | None = None,
    categories: Iterable[str] | None = None,
    inject_defects: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate daily enquiries and orders.

    Returns:
        demand: columns ``date, city, zone_id, category, enquiries, orders``
        zones:  columns ``zone_id, city, zone_name, lat, lon``
    """
    city_codes = list(cities) if cities is not None else list(CITIES)
    cat_codes = list(categories) if categories is not None else list(CATEGORIES)
    unknown = (set(city_codes) - set(CITIES)) | (set(cat_codes) - set(CATEGORIES))
    if unknown:
        raise ValueError(f"Unknown city/category codes: {sorted(unknown)}")

    dates = pd.date_range(start, end, freq="D")
    if len(dates) < 30:
        raise ValueError("Simulation window must cover at least 30 days")
    n = len(dates)
    rng = np.random.default_rng(seed)
    cal = calendar_features(dates)
    t = np.arange(n, dtype=float)
    doy = dates.dayofyear.to_numpy(dtype=float)

    # Calendar effects shared by all series -------------------------------- #
    weekly = WEEKLY_PROFILE[cal["dow"].to_numpy()]
    to_diwali = cal["days_to_diwali"].to_numpy()
    festive_ramp = np.where((to_diwali >= 3) & (to_diwali <= 21), (21.0 - to_diwali) / 18.0, 0.0)
    shutdown = (
        np.where(cal["is_diwali_shutdown"].to_numpy() == 1, 0.30, 1.0)
        * np.where(cal["is_national_holiday"].to_numpy() == 1, 0.45, 1.0)
        * np.where(cal["is_festival_day"].to_numpy() == 1, 0.70, 1.0)
    )
    quarter_push = np.where(cal["is_quarter_end_window"].to_numpy() == 1, 1.04, 1.0)
    monsoon = cal["is_monsoon"].to_numpy() == 1
    fy_end = cal["is_fy_end_fortnight"].to_numpy() == 1

    frames: list[pd.DataFrame] = []
    zone_rows: list[dict] = []
    for city_code in city_codes:
        city = CITIES[city_code]
        for zone in city.zones:
            zone_rows.append(
                {
                    "zone_id": zone.zone_id,
                    "city": city_code,
                    "zone_name": zone.name,
                    "lat": zone.lat,
                    "lon": zone.lon,
                }
            )
            zone_density = rng.lognormal(0.0, 0.35)
            for cat_code in cat_codes:
                spec = CATEGORIES[cat_code]
                base = (
                    BASE_ORDERS_PER_DAY
                    * city.demand_scale
                    * zone_density
                    * city.affinity(cat_code)
                    * rng.lognormal(0.0, 0.20)
                )
                growth = rng.uniform(0.06, 0.28)  # annual growth of online B2B adoption
                amplitude = rng.uniform(0.08, 0.18)
                intensity = (
                    base
                    * (1.0 + growth) ** (t / 365.0)
                    * weekly
                    * (1.0 + amplitude * np.cos(2 * np.pi * (doy - spec.peak_doy) / 365.25))
                    * np.where(monsoon, spec.monsoon_multiplier, 1.0)
                    * (1.0 + spec.festive_uplift * festive_ramp)
                    * shutdown
                    * np.where(fy_end, spec.fy_end_uplift, 1.0)
                    * quarter_push
                )
                shock = _ar1(rng, n, phi=0.80, sigma=0.15)
                enquiry_ratio = rng.uniform(2.5, 4.0)  # buyers enquire with several suppliers
                enquiries = rng.poisson(intensity * enquiry_ratio * np.exp(shock))
                conversion_noise = np.exp(rng.normal(0.0, 0.05, n))
                orders = rng.poisson(
                    intensity * np.exp(_lagged(shock, ENQUIRY_TO_ORDER_LEAD_DAYS)) * conversion_noise
                )
                frames.append(
                    pd.DataFrame(
                        {
                            "date": dates,
                            "city": city_code,
                            "zone_id": zone.zone_id,
                            "category": cat_code,
                            "enquiries": enquiries.astype(np.int64),
                            "orders": orders.astype(np.int64),
                        }
                    )
                )

    demand = pd.concat(frames, ignore_index=True)
    if inject_defects:
        demand = _inject_defects(demand, rng)
    zones = pd.DataFrame(zone_rows)
    logger.info(
        "Simulated %d rows | %d series | %s -> %s",
        len(demand),
        demand.groupby(["zone_id", "category"]).ngroups,
        dates[0].date(),
        dates[-1].date(),
    )
    return demand, zones


def _inject_defects(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Introduce the defects a real ingestion pipeline sees, at realistic rates."""
    df = df.copy()
    n = len(df)
    spike_idx = rng.choice(n, size=max(1, int(0.001 * n)), replace=False)
    df.loc[spike_idx, "orders"] = df.loc[spike_idx, "orders"] * rng.integers(6, 13, size=len(spike_idx)) + 20
    neg_idx = rng.choice(n, size=max(1, int(0.0005 * n)), replace=False)
    df.loc[neg_idx, "orders"] = -df.loc[neg_idx, "orders"].abs() - 1
    dup_idx = rng.choice(n, size=max(1, int(0.002 * n)), replace=False)
    duplicates = df.loc[dup_idx]
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[rng.choice(n, size=max(1, int(0.004 * n)), replace=False)] = False
    out = pd.concat([df.loc[keep_mask], duplicates], ignore_index=True)
    return out.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)


def write_dataset(settings: Settings, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate and persist the raw demand file and the zone master."""
    settings.ensure_dirs()
    demand, zones = generate_dataset(seed=settings.random_seed, **kwargs)
    out = demand.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(settings.raw_demand_path, index=False, compression="gzip")
    zones.to_csv(settings.zones_path, index=False)
    logger.info("Wrote %s and %s", settings.raw_demand_path, settings.zones_path)
    return demand, zones


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic B2B demand data")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--cities", default=None, help="Comma-separated city codes")
    args = parser.parse_args(argv)
    cities = args.cities.split(",") if args.cities else None
    write_dataset(get_settings(), start=args.start, end=args.end, cities=cities)


if __name__ == "__main__":
    main()
