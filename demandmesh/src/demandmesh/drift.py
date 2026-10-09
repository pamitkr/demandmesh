"""Data-drift monitoring with the Population Stability Index (PSI).

    PSI = sum_i (a_i - e_i) * ln(a_i / e_i)

where ``e_i`` / ``a_i`` are the shares of the reference (training) and the
current window falling in decile bin ``i`` of the reference distribution.
Industry rule of thumb: < 0.10 stable, 0.10-0.25 investigate, > 0.25 retrain.

Only behavioural features are monitored. Pure calendar features "drift" by
design (a 28-day window covers one month) and would raise false alarms.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONITORED_FEATURES: tuple[str, ...] = (
    "orders",
    "orders_lag_7",
    "orders_roll_mean_28",
    "enq_roll_mean_7",
    "enq_to_order_conv_28",
    "zone_total_roll_mean_7",
)
PSI_WARN = 0.10
PSI_ALERT = 0.25
_EPS = 1e-4


def build_reference_profile(
    df: pd.DataFrame, columns: tuple[str, ...] = MONITORED_FEATURES, n_bins: int = 10
) -> dict[str, dict]:
    """Quantile bin edges + bin shares of each monitored column."""
    profile: dict[str, dict] = {}
    quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
    for col in columns:
        if col not in df.columns:
            continue
        values = df[col].dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue
        edges = np.unique(np.quantile(values, quantiles))
        shares = _bin_shares(values, edges)
        profile[col] = {"edges": edges.tolist(), "shares": shares.tolist(), "n": int(values.size)}
    return profile


def _bin_shares(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    buckets = np.searchsorted(edges, values, side="right")
    counts = np.bincount(buckets, minlength=len(edges) + 1).astype(float)
    return counts / max(counts.sum(), 1.0)


def population_stability_index(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(np.asarray(expected, float), _EPS, None)
    a = np.clip(np.asarray(actual, float), _EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def compute_drift_report(current: pd.DataFrame, profile: dict[str, dict]) -> dict:
    """Compare a current window with the training profile, feature by feature."""
    features: dict[str, dict] = {}
    worst = 0.0
    for col, ref in profile.items():
        if col not in current.columns:
            continue
        values = current[col].dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue
        psi = population_stability_index(
            np.asarray(ref["shares"]), _bin_shares(values, np.asarray(ref["edges"]))
        )
        worst = max(worst, psi)
        features[col] = {"psi": round(psi, 4), "status": _status(psi), "n_current": int(values.size)}
    return {
        "overall_status": _status(worst),
        "max_psi": round(worst, 4),
        "thresholds": {"warn": PSI_WARN, "alert": PSI_ALERT},
        "features": features,
    }


def _status(psi: float) -> str:
    if psi >= PSI_ALERT:
        return "alert"
    if psi >= PSI_WARN:
        return "warn"
    return "stable"


def main(argv: list[str] | None = None) -> None:
    """Demo: ``python -m demandmesh.drift --shock 1.4`` inflates the last
    ``--window`` days of enquiries and orders and shows the monitor reacting."""
    import argparse
    import json

    from .config import get_settings
    from .forecaster import DemandForecaster

    parser = argparse.ArgumentParser(description="Population Stability Index drift check")
    parser.add_argument("--window", type=int, default=28)
    parser.add_argument(
        "--shock", type=float, default=1.0, help="multiply recent demand by this factor to simulate drift"
    )
    args = parser.parse_args(argv)

    forecaster = DemandForecaster.from_settings(get_settings())
    if args.shock != 1.0:
        history = forecaster.history.copy()
        recent = history["date"] > history["date"].max() - pd.Timedelta(days=args.window)
        history.loc[recent, ["orders", "enquiries"]] *= args.shock
        forecaster.history = history
    print(json.dumps(forecaster.drift_report(args.window), indent=2))


if __name__ == "__main__":
    main()
