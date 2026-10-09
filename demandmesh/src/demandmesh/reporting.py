"""Static evaluation report: figures + markdown summary committed with the repo.

Produces, under ``reports/``:
    figures/forecast_vs_actual.png   P10-P90 band, P50 and actuals on the holdout
    figures/model_vs_baselines.png   WAPE of the model vs naive forecasts
    figures/feature_importance.png   permutation importance (median model)
    figures/route_plan.png           an optimised city dispatch plan
    metrics.md                       every number quoted in the README
    routing_benchmark.json           ALNS vs nearest-neighbour for all cities
"""

from __future__ import annotations

import json
import math
import numbers
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FormatStrFormatter, MaxNLocator  # noqa: E402

from .config import CATEGORIES, CITIES, FLEET  # noqa: E402
from .logging_utils import get_logger  # noqa: E402
from .model_training import PREDICTIONS_FILE  # noqa: E402

logger = get_logger(__name__)

# Validated categorical palette (light surface) + neutral tokens
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
NEUTRAL, BAND = "#a9a7a1", "#b7d3f6"


def _style(ax: plt.Axes, title: str, subtitle: str | None = None) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(color=GRID, linewidth=1, axis="y")
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold", pad=18)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK_2, fontsize=9)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_forecast_vs_actual(pred: pd.DataFrame, path: Path) -> None:
    volume = pred.groupby(["zone_id", "category"])["orders"].sum().sort_values()
    picks = [volume.index[-1], volume.index[len(volume) // 2]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=False)
    for ax, (zone, cat) in zip(axes, picks, strict=True):
        s = pred.loc[(pred["zone_id"] == zone) & (pred["category"] == cat)].sort_values("date")
        ax.fill_between(
            s["date"], s["p10"], s["p90"], color=BAND, alpha=0.6, linewidth=0, label="P10-P90 interval"
        )
        ax.plot(s["date"], s["p50"], color=SERIES[0], linewidth=2, label="P50 forecast")
        ax.plot(
            s["date"],
            s["orders"],
            color=INK_2,
            linewidth=1.5,
            marker="o",
            markersize=4,
            label="Actual orders",
        )
        _style(ax, f"{zone} · {CATEGORIES[cat].label}", "Daily orders, 28-day holdout")
        ax.tick_params(axis="x", rotation=30)
        ax.set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")
    _save(fig, path)


def plot_model_vs_baselines(metrics: dict, path: Path) -> None:
    rows = [(k.replace("_", " "), v["wape"]) for k, v in metrics["baselines"].items()]
    rows.append(("DemandMesh P50", metrics["model_p50"]["wape"]))
    rows.sort(key=lambda r: r[1], reverse=True)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    colors = [SERIES[0] if name.startswith("DemandMesh") else NEUTRAL for name, _ in rows]
    ax.barh([r[0] for r in rows], [r[1] * 100 for r in rows], color=colors, height=0.55)
    for i, (_, value) in enumerate(rows):
        ax.text(value * 100 + 0.5, i, f"{value * 100:.1f}%", va="center", color=INK, fontsize=9)
    _style(ax, "Forecast error (WAPE) on the 28-day holdout", "Lower is better")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID)
    ax.set_xlabel("WAPE (%)", color=INK_2, fontsize=9)
    _save(fig, path)


def plot_feature_importance(importance: list[dict], path: Path, top: int = 12) -> None:
    rows = importance[:top][::-1]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.barh([r["feature"] for r in rows], [r["mae_increase"] for r in rows], color=SERIES[0], height=0.6)
    _style(ax, "What drives the forecast", "Increase in MAE when the feature is shuffled")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID)
    _save(fig, path)


def plot_route_plan(plan: dict, path: Path) -> None:
    stops = {s["stop_id"]: s for s in plan["stops"]}
    depot = plan["depot"]
    routes = plan["optimised_plan"]["routes"]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for i, route in enumerate(routes):
        color = SERIES[i % len(SERIES)]
        lats = [depot["lat"]] + [stops[v["stop_id"]]["lat"] for v in route["stops"]] + [depot["lat"]]
        lons = [depot["lon"]] + [stops[v["stop_id"]]["lon"] for v in route["stops"]] + [depot["lon"]]
        ax.plot(
            lons,
            lats,
            color=color,
            linewidth=2,
            solid_capstyle="round",
            label=f"{route['vehicle_id']} · {route['load_kg']:.0f} kg · {route['distance_km']:.0f} km",
        )
        ax.scatter(lons[1:-1], lats[1:-1], s=46, color=color, edgecolors=SURFACE, linewidths=2, zorder=3)
        # label the route at its stop farthest from the hub (least cluttered spot)
        far = max(
            range(1, len(lats) - 1),
            key=lambda j: (lats[j] - depot["lat"]) ** 2 + (lons[j] - depot["lon"]) ** 2,
        )
        ax.annotate(
            route["vehicle_id"].split("-")[-1],
            (lons[far], lats[far]),
            xytext=(6, 6),
            textcoords="offset points",
            color=INK_2,
            fontsize=8,
        )
    ax.scatter([depot["lon"]], [depot["lat"]], marker="s", s=90, color=INK, zorder=4, label="Hub")
    saving = plan["savings_vs_baseline"]["distance_pct"]
    _style(
        ax,
        f"Optimised dispatch · {plan['city_name']} · {plan['delivery_date']}",
        f"{plan['demand']['drops']} drops, {plan['demand']['planned_freight_kg']:.0f} kg at "
        f"{plan['service_level'].upper()} · {saving:.0%} less distance than nearest-neighbour",
    )
    ax.grid(color=GRID)
    ax.set_aspect(1 / math.cos(math.radians(depot["lat"])))  # undistorted local map
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_xlabel("Longitude", color=INK_2, fontsize=9)
    ax.set_ylabel("Latitude", color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="best")
    _save(fig, path)


def service_level_backtest(pred: pd.DataFrame) -> pd.DataFrame:
    """Book city-day capacity three ways on the holdout and score it on actuals.

    * P50 sum       - sum of series medians
    * naive P90 sum - sum of series P90s (quantiles do not add: over-books)
    * pooled P90    - sum(P50) + sqrt(sum((P90 - P50)^2)), i.e. risk pooling
    """
    df = pred.copy()
    kg = df["category"].map(lambda c: CATEGORIES[c].kg_per_order) * FLEET.logistics_attach_rate
    df["actual"], df["m"], df["u"] = df["orders"] * kg, df["p50"] * kg, df["p90"] * kg
    df["d2"] = (df["u"] - df["m"]) ** 2
    g = df.groupby(["city", "date"]).agg(
        actual=("actual", "sum"), m=("m", "sum"), u=("u", "sum"), d2=("d2", "sum")
    )
    strategies = {
        "P50 sum": g["m"],
        "naive P90 sum": g["u"],
        "pooled P90 (used)": g["m"] + g["d2"] ** 0.5,
    }
    total = g["actual"].sum()
    rows = []
    for name, plan in strategies.items():
        rows.append(
            {
                "capacity_booked_at": name,
                "city_days_short_pct": round(100 * float((g["actual"] > plan).mean()), 1),
                "unplanned_freight_pct": round(
                    100 * float((g["actual"] - plan).clip(lower=0).sum() / total), 1
                ),
                "idle_capacity_pct": round(100 * float((plan - g["actual"]).clip(lower=0).sum() / total), 1),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    model_dir: Path,
    metadata: dict,
    plans: dict[str, dict],
    reports_dir: Path,
    showcase_city: str,
) -> Path:
    figures = reports_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(model_dir / PREDICTIONS_FILE, parse_dates=["date"])
    metrics = metadata["holdout_metrics"]

    plot_forecast_vs_actual(pred, figures / "forecast_vs_actual.png")
    plot_model_vs_baselines(metrics, figures / "model_vs_baselines.png")
    plot_feature_importance(metadata["feature_importance"], figures / "feature_importance.png")
    plot_route_plan(plans[showcase_city], figures / "route_plan.png")
    backtest = service_level_backtest(pred)

    bench = []
    for city, plan in plans.items():
        opt, base = plan["optimised_plan"], plan["baseline_plan"]
        bench.append(
            {
                "city": CITIES[city].name,
                "drops": plan["demand"]["drops"],
                "freight_kg": plan["demand"]["planned_freight_kg"],
                "alns_vehicles": opt["vehicles_used"],
                "alns_km": opt["total_distance_km"],
                "alns_cost_inr": opt["total_cost_inr"],
                "nn_vehicles": base["vehicles_used"],
                "nn_km": base["total_distance_km"],
                "nn_cost_inr": base["total_cost_inr"],
                "km_saved_pct": round(100 * (plan["savings_vs_baseline"]["distance_pct"] or 0), 1),
                "cost_saved_pct": round(100 * (plan["savings_vs_baseline"]["cost_pct"] or 0), 1),
                "unserved": len(opt["unserved_stop_ids"]),
            }
        )
    bench_df = pd.DataFrame(bench)
    (reports_dir / "routing_benchmark.json").write_text(json.dumps(bench, indent=2))
    totals = bench_df[["alns_km", "nn_km", "alns_cost_inr", "nn_cost_inr"]].sum()

    m, cal = metrics, metrics["interval_calibration"]
    lines = [
        "# DemandMesh - evaluation report",
        "",
        f"Model `{metadata['version']}` · data {metadata['data']['history_start']} → "
        f"{metadata['data']['history_end']} · {metadata['data']['series']} series · "
        f"holdout {m['test_window'][0]} → {m['test_window'][1]}.",
        "",
        "All data is synthetic (see `src/demandmesh/data_generator.py`); numbers show the "
        "method works end-to-end, not what a real deployment would achieve.",
        "",
        "## Forecast accuracy (28-day holdout, 7-day-ahead features)",
        "",
        "| Method | WAPE | MAE | Bias |",
        "|---|---|---|---|",
        f"| **DemandMesh P50** | **{m['model_p50']['wape']:.1%}** | {m['model_p50']['mae']:.2f} | "
        f"{m['model_p50']['bias']:+.1%} |",
    ]
    lines += [
        f"| {k.replace('_', ' ')} | {v['wape']:.1%} | {v['mae']:.2f} | {v['bias']:+.1%} |"
        for k, v in m["baselines"].items()
    ]
    lines += [
        "",
        f"Improvement over the best baseline: **{m['wape_improvement_vs_best_baseline']:.1%}** lower WAPE.",
        "",
        "| Aggregation level | WAPE |",
        "|---|---|",
    ]
    lines += [f"| {k.replace('_', ' ')} | {v:.1%} |" for k, v in m["aggregation_levels_wape"].items()]
    lines += [
        "",
        "## Uncertainty calibration",
        "",
        f"P10-P90 nominal coverage {cal['nominal_coverage']:.0%} · raw model "
        f"{cal['raw_coverage']:.1%} · after conformal calibration **{cal['conformal_coverage']:.1%}** "
        f"(interval widened by {cal['conformal_adjustment_orders']:.2f} orders).",
        "",
        "## Capacity booking: P50 vs P90 (city-day freight, holdout)",
        "",
        "Shares are of actual freight. Idle capacity costs money; unplanned freight means "
        "spot-hiring trucks at a premium.",
        "",
        _md_table(backtest),
        "",
        "## Routing: ALNS vs nearest-neighbour dispatch (next-day plan, P90)",
        "",
        _md_table(bench_df),
        "",
        f"Network total: {totals['alns_km']:.0f} km vs {totals['nn_km']:.0f} km "
        f"(**{1 - totals['alns_km'] / totals['nn_km']:.1%} less distance**), "
        f"₹{totals['alns_cost_inr']:,.0f} vs ₹{totals['nn_cost_inr']:,.0f} per day "
        f"(**{1 - totals['alns_cost_inr'] / totals['nn_cost_inr']:.1%} lower cost**).",
        "",
        "## Top drivers (permutation importance)",
        "",
        "| Feature | MAE increase |",
        "|---|---|",
    ]
    lines += [f"| {f['feature']} | {f['mae_increase']:.3f} |" for f in metadata["feature_importance"][:10]]
    path = reports_dir / "metrics.md"
    path.write_text("\n".join(lines) + "\n")
    logger.info("Wrote report to %s", path)
    return path


def _md_table(df: pd.DataFrame) -> str:
    """Minimal GitHub-flavoured markdown table (no optional dependencies)."""

    def fmt(v) -> str:
        if isinstance(v, numbers.Integral):
            return f"{int(v):,}"
        if isinstance(v, numbers.Real):
            return f"{float(v):,.1f}"
        return str(v)

    header = "| " + " | ".join(df.columns) + " |"
    rule = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, rule, *body])
