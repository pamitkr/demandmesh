"""One-command reproducible pipeline (works on Windows, macOS and Linux).

python -m demandmesh.pipeline            # generate -> clean -> train -> report
python -m demandmesh.pipeline --tune     # + randomised hyper-parameter search
python -m demandmesh.pipeline --skip-generate   # retrain on existing raw data
"""

from __future__ import annotations

import argparse
import json
import time

from .config import CITIES, get_settings
from .data_generator import write_dataset
from .data_preprocessing import load_zones, run_preprocessing
from .forecaster import DemandForecaster
from .logging_utils import configure_logging, get_logger
from .model_training import train
from .planner import plan_city_dispatch

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="DemandMesh end-to-end pipeline")
    parser.add_argument("--skip-generate", action="store_true", help="reuse data/raw")
    parser.add_argument("--tune", action="store_true", help="hyper-parameter search")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--skip-report", action="store_true")
    parser.add_argument("--showcase-city", default="PUN", choices=sorted(CITIES))
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    settings = get_settings()
    settings.ensure_dirs()
    started = time.perf_counter()

    if not args.skip_generate:
        logger.info("[1/4] Generating synthetic marketplace data")
        write_dataset(settings)
    logger.info("[2/4] Validating and cleaning")
    history = run_preprocessing(settings)
    logger.info("[3/4] Training quantile forecaster")
    result = train(
        history,
        load_zones(settings.zones_path),
        settings.model_dir,
        tune=args.tune,
        n_trials=args.n_trials,
        seed=settings.random_seed,
    )

    if not args.skip_report:
        from .reporting import write_report  # matplotlib only needed here

        logger.info("[4/4] Planning next-day dispatch for every city + writing report")
        forecaster = DemandForecaster.from_settings(settings)
        plans = {city: plan_city_dispatch(forecaster, city) for city in CITIES}
        (settings.reports_dir / "sample_dispatch_plan.json").write_text(
            json.dumps(plans[args.showcase_city], indent=2)
        )
        write_report(result.model_dir, result.metadata, plans, settings.reports_dir, args.showcase_city)

    m = result.metadata["holdout_metrics"]
    logger.info(
        "Done in %.0fs | model %s | holdout WAPE %.1f%% (best baseline %.1f%%) | P10-P90 coverage %.1f%%",
        time.perf_counter() - started,
        result.version,
        100 * m["model_p50"]["wape"],
        100 * min(b["wape"] for b in m["baselines"].values()),
        100 * m["p10_p90_coverage"],
    )


if __name__ == "__main__":
    main()
