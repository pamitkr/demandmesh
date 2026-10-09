"""Train, evaluate and register the quantile demand forecaster.

Model
-----
One histogram gradient-boosted tree ensemble per quantile (P10 / P50 / P90),
trained with the pinball loss. Histogram GBDT (the LightGBM algorithm, as
implemented in scikit-learn) handles categorical zones/categories and missing
lags natively, trains on hundreds of thousands of rows in seconds, and is the
dominant model family in forecasting competitions on tabular retail data (M5).

Validation protocol (no look-ahead anywhere)
--------------------------------------------
    |------------- train -------------|-- valid (28d) --|-- test (28d) --|
    * Early stopping uses the *time-ordered* validation block, never a random split.
    * Hyper-parameter search (optional) also scores on the validation block.
    * The test block is touched exactly once, for the reported metrics.
    * The deployed model is then refit on all data with the selected
      hyper-parameters and iteration counts ("evaluate, then refit").

Every metric is reported against three naive baselines, because a model that
does not beat seasonal-naive is not worth deploying.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import ParameterSampler

from . import __version__
from .data_preprocessing import (
    SERIES_KEYS,
    TARGET,
    FeatureSpec,
    build_feature_frame,
    category_levels,
    feature_columns,
    load_history,
    load_zones,
    to_model_frame,
)
from .drift import build_reference_profile
from .exceptions import ModelArtifactError
from .logging_utils import get_logger
from .metrics import interval_coverage, pinball_loss, point_metrics, wape

logger = get_logger(__name__)

QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)
ARTIFACT_FILE = "forecaster.joblib"
METADATA_FILE = "metadata.json"
PROFILE_FILE = "reference_profile.json"
PREDICTIONS_FILE = "test_predictions.csv.gz"
LATEST_POINTER = "LATEST"

DEFAULT_PARAMS: dict[str, float | int] = {
    "learning_rate": 0.05,
    "max_iter": 1500,
    "max_leaf_nodes": 63,
    "min_samples_leaf": 40,
    "l2_regularization": 1.0,
    "max_features": 0.8,
    "max_bins": 255,
}

SEARCH_SPACE: dict[str, list] = {
    "learning_rate": [0.03, 0.05, 0.08],
    "max_leaf_nodes": [31, 63, 127],
    "min_samples_leaf": [20, 40, 80, 160],
    "l2_regularization": [0.0, 0.5, 1.0, 3.0],
    "max_features": [0.6, 0.8, 1.0],
}


TARGET_TRANSFORMS = ("level_ratio", "none")


def level_scale(X: pd.DataFrame) -> np.ndarray:
    """Recent demand level of each row (horizon-safe): 28-day mean, else 4-week
    same-weekday mean, plus 1. Dividing the target by it makes the model
    scale-free: one model serves a 2-order/day zone and a 200-order/day zone, and
    growth no longer pushes lags outside the range the trees have seen."""
    level = X["orders_roll_mean_28"].astype(float)
    level = level.fillna(X["orders_same_dow_mean_4w"].astype(float)).fillna(0.0)
    return level.to_numpy() + 1.0


@dataclass
class TrainingResult:
    version: str
    model_dir: Path
    metadata: dict


# --------------------------------------------------------------------------- #
# Core modelling helpers                                                      #
# --------------------------------------------------------------------------- #
def split_by_time(
    df: pd.DataFrame, test_days: int, valid_days: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train / validation / test split."""
    last = df["date"].max()
    test_start = last - pd.Timedelta(days=test_days - 1)
    valid_start = test_start - pd.Timedelta(days=valid_days)
    train = df.loc[df["date"] < valid_start]
    valid = df.loc[(df["date"] >= valid_start) & (df["date"] < test_start)]
    test = df.loc[df["date"] >= test_start]
    if train.empty or valid.empty or test.empty:
        raise ValueError("Not enough history for the requested train/valid/test split")
    return train, valid, test


def make_estimator(
    alpha: float, params: dict, seed: int, early_stopping: bool = True
) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="quantile",
        quantile=alpha,
        categorical_features="from_dtype",
        early_stopping=early_stopping,
        n_iter_no_change=50,
        scoring="loss",
        random_state=seed,
        **params,
    )


def fit_quantile_models(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    params: dict,
    quantiles: tuple[float, ...],
    seed: int,
    target_transform: str = "level_ratio",
) -> tuple[dict[float, HistGradientBoostingRegressor], dict[float, int]]:
    """Fit one model per quantile with early stopping on the time-ordered validation block.

    With ``level_ratio`` the model learns y / level with sample weight = level.
    Pinball loss is positively homogeneous, so this is *exactly* the volume-
    weighted pinball loss in original units, while the trees see scale-free inputs.
    """
    models, iterations = {}, {}
    y_tr, w_tr, y_va, w_va = _transform_target(X_train, y_train, X_valid, y_valid, target_transform)
    for alpha in quantiles:
        start = time.perf_counter()
        model = make_estimator(alpha, params, seed)
        model.fit(X_train, y_tr, sample_weight=w_tr, X_val=X_valid, y_val=y_va, sample_weight_val=w_va)
        models[alpha], iterations[alpha] = model, int(model.n_iter_)
        logger.info("Fitted q=%.2f | %d trees | %.1fs", alpha, model.n_iter_, time.perf_counter() - start)
    return models, iterations


def _transform_target(X_train, y_train, X_valid, y_valid, target_transform):
    if target_transform not in TARGET_TRANSFORMS:
        raise ValueError(f"target_transform must be one of {TARGET_TRANSFORMS}")
    if target_transform == "none":
        return y_train, None, y_valid, None
    s_tr, s_va = level_scale(X_train), level_scale(X_valid)
    return y_train / s_tr, s_tr, y_valid / s_va, s_va


def predict_quantiles(
    models: dict[float, HistGradientBoostingRegressor],
    X: pd.DataFrame,
    interval_adjustment: float = 0.0,
    target_transform: str = "level_ratio",
) -> np.ndarray:
    """Predict all quantiles -> (n, q) array in order units, non-negative and non-crossing.

    ``interval_adjustment`` is the conformal correction (see
    :func:`conformal_interval_adjustment`): the outer quantiles are widened (or
    narrowed, if negative) by it; the median is left untouched.
    """
    alphas = sorted(models)
    preds = np.column_stack([models[a].predict(X) for a in alphas])
    if target_transform == "level_ratio":
        preds = preds * level_scale(X)[:, None]
    preds[:, 0] -= interval_adjustment
    preds[:, -1] += interval_adjustment
    preds = np.clip(preds, 0.0, None)
    return np.sort(preds, axis=1)  # enforce P10 <= P50 <= P90 (quantile crossing fix)


def conformal_interval_adjustment(
    y: np.ndarray, lower: np.ndarray, upper: np.ndarray, coverage: float
) -> float:
    """Conformalized Quantile Regression (Romano et al., 2019).

    Conformity score E_i = max(lower_i - y_i, y_i - upper_i) on a held-out
    calibration set; the finite-sample-corrected ``coverage`` quantile of E is
    the amount by which to widen the raw interval so that it attains the
    nominal coverage under exchangeability.
    """
    scores = np.maximum(lower - y, y - upper)
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * coverage) / n)
    return float(np.quantile(scores, level, method="higher"))


def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    n_trials: int,
    seed: int,
    target_transform: str = "level_ratio",
) -> tuple[dict, list[dict]]:
    """Randomised search scored by validation pinball loss of the median model."""
    trials: list[dict] = []
    best_params, best_score = dict(DEFAULT_PARAMS), np.inf
    sampler = ParameterSampler(SEARCH_SPACE, n_iter=n_trials, random_state=seed)
    for i, candidate in enumerate(sampler, start=1):
        params = {**DEFAULT_PARAMS, **candidate}
        models, _ = fit_quantile_models(
            X_train, y_train, X_valid, y_valid, params, (0.5,), seed, target_transform
        )
        pred = predict_quantiles(models, X_valid, target_transform=target_transform)[:, 0]
        score = pinball_loss(y_valid, pred, 0.5)
        trials.append({"trial": i, "params": candidate, "valid_pinball_p50": round(score, 5)})
        logger.info("Trial %d/%d pinball=%.4f %s", i, n_trials, score, candidate)
        if score < best_score:
            best_score, best_params = score, params
    return best_params, trials


def baseline_predictions(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Naive forecasts every model must beat (all horizon-safe)."""
    return {
        "seasonal_naive_7d": df["orders_lag_7"].fillna(0.0).to_numpy(),
        "same_weekday_mean_4w": df["orders_same_dow_mean_4w"].fillna(0.0).to_numpy(),
        "moving_average_28d": df["orders_roll_mean_28"].fillna(0.0).to_numpy(),
    }


def evaluate(test: pd.DataFrame, preds: np.ndarray, quantiles: tuple[float, ...]) -> dict:
    """Holdout metrics: point accuracy, quantile calibration, segments, baselines."""
    y = test[TARGET].to_numpy(dtype=float)
    p10, p50, p90 = preds[:, 0], preds[:, quantiles.index(0.5)], preds[:, -1]
    frame = test[["date", "city", "zone_id", "category"]].assign(y=y, p50=p50)

    zone_day = frame.groupby(["zone_id", "date"], observed=True)[["y", "p50"]].sum()
    city_day = frame.groupby(["city", "date"], observed=True)[["y", "p50"]].sum()
    baselines = {name: point_metrics(y, pred) for name, pred in baseline_predictions(test).items()}
    best_baseline = min(baselines.values(), key=lambda m: m["wape"])["wape"]
    model_point = point_metrics(y, p50)
    return {
        "test_window": [str(test["date"].min().date()), str(test["date"].max().date())],
        "rows": int(len(test)),
        "model_p50": model_point,
        "wape_improvement_vs_best_baseline": round(1 - model_point["wape"] / best_baseline, 4),
        "baselines": baselines,
        "pinball": {
            f"p{int(a * 100)}": round(pinball_loss(y, preds[:, i], a), 4) for i, a in enumerate(quantiles)
        },
        "p10_p90_coverage": round(interval_coverage(y, p10, p90), 4),
        "aggregation_levels_wape": {
            "series_day": model_point["wape"],
            "zone_day": round(wape(zone_day["y"], zone_day["p50"]), 4),
            "city_day": round(wape(city_day["y"], city_day["p50"]), 4),
        },
        "wape_by_category": {
            k: round(wape(g["y"], g["p50"]), 4) for k, g in frame.groupby("category", observed=True)
        },
        "wape_by_city": {
            k: round(wape(g["y"], g["p50"]), 4) for k, g in frame.groupby("city", observed=True)
        },
    }


class _OrderUnitsModel(RegressorMixin, BaseEstimator):
    """Adapter so permutation importance scores the median model in order units."""

    def __init__(self, model: HistGradientBoostingRegressor, target_transform: str) -> None:
        self.model = model
        self.target_transform = target_transform

    def fit(self, X, y):  # pragma: no cover - required by the sklearn scorer protocol
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pred = self.model.predict(X)
        return pred * level_scale(X) if self.target_transform == "level_ratio" else pred


def compute_feature_importance(
    model: HistGradientBoostingRegressor,
    X: pd.DataFrame,
    y: np.ndarray,
    seed: int,
    target_transform: str = "level_ratio",
    top_k: int = 20,
) -> list[dict]:
    """Model-agnostic permutation importance (increase in MAE when a feature is shuffled)."""
    sample = min(len(X), 15_000)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=sample, replace=False)
    result = permutation_importance(
        _OrderUnitsModel(model, target_transform),
        X.iloc[idx],
        y[idx],
        scoring="neg_mean_absolute_error",
        n_repeats=3,
        random_state=seed,
    )
    ranked = sorted(
        zip(X.columns, result.importances_mean, result.importances_std, strict=True),
        key=lambda r: r[1],
        reverse=True,
    )
    return [
        {"feature": f, "mae_increase": round(float(m), 5), "std": round(float(s), 5)}
        for f, m, s in ranked[:top_k]
    ]


# --------------------------------------------------------------------------- #
# Registry                                                                    #
# --------------------------------------------------------------------------- #
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_model_dir(model_root: Path) -> Path:
    """Return the directory of the version referenced by ``models/LATEST``."""
    pointer = model_root / LATEST_POINTER
    if not pointer.exists():
        raise ModelArtifactError(f"No trained model found ({pointer} missing). Run training.")
    version_dir = model_root / pointer.read_text().strip()
    if not (version_dir / ARTIFACT_FILE).exists():
        raise ModelArtifactError(f"Model artefact missing in {version_dir}")
    return version_dir


def load_artifact(version_dir: Path) -> tuple[dict, dict, dict]:
    """Load models + metadata + drift profile after verifying the checksum.

    joblib files are pickles; the SHA-256 recorded at training time is checked
    before deserialising so a tampered or truncated file is rejected.
    """
    meta_path = version_dir / METADATA_FILE
    if not meta_path.exists():
        raise ModelArtifactError(f"Metadata missing in {version_dir}")
    metadata = json.loads(meta_path.read_text())
    artifact_path = version_dir / ARTIFACT_FILE
    if _sha256(artifact_path) != metadata.get("artifact_sha256"):
        raise ModelArtifactError(f"Checksum mismatch for {artifact_path}; refusing to load")
    if metadata.get("sklearn_version") != sklearn.__version__:
        logger.warning(
            "Model trained with scikit-learn %s, running %s",
            metadata.get("sklearn_version"),
            sklearn.__version__,
        )
    bundle = joblib.load(artifact_path)
    profile_path = version_dir / PROFILE_FILE
    profile = json.loads(profile_path.read_text()) if profile_path.exists() else {}
    return bundle, metadata, profile


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #
def train(
    history: pd.DataFrame,
    zones: pd.DataFrame,
    model_root: Path,
    *,
    spec: FeatureSpec | None = None,
    params: dict | None = None,
    tune: bool = False,
    n_trials: int = 10,
    refit_full: bool = True,
    test_days: int = 28,
    valid_days: int = 28,
    seed: int = 42,
    target_transform: str = "level_ratio",
) -> TrainingResult:
    """End-to-end training run; returns the registered model version."""
    spec = spec or FeatureSpec()
    params = {**DEFAULT_PARAMS, **(params or {})}
    started = time.perf_counter()

    features = build_feature_frame(history, zones, spec, drop_warmup=True)
    features = features.loc[features[TARGET].notna()].reset_index(drop=True)
    columns, levels = feature_columns(spec), category_levels()
    train_df, valid_df, test_df = split_by_time(features, test_days, valid_days)
    logger.info(
        "Rows train=%d valid=%d test=%d | %d features",
        len(train_df),
        len(valid_df),
        len(test_df),
        len(columns),
    )

    X_tr, y_tr = to_model_frame(train_df, columns, levels), train_df[TARGET].to_numpy(float)
    X_va, y_va = to_model_frame(valid_df, columns, levels), valid_df[TARGET].to_numpy(float)
    X_te = to_model_frame(test_df, columns, levels)

    trials: list[dict] = []
    if tune:
        params, trials = tune_hyperparameters(X_tr, y_tr, X_va, y_va, n_trials, seed, target_transform)

    tt = target_transform
    models, iterations = fit_quantile_models(X_tr, y_tr, X_va, y_va, params, QUANTILES, seed, tt)
    valid_raw = predict_quantiles(models, X_va, target_transform=tt)
    nominal = QUANTILES[-1] - QUANTILES[0]
    adjustment = conformal_interval_adjustment(y_va, valid_raw[:, 0], valid_raw[:, -1], nominal)
    raw_test_preds = predict_quantiles(models, X_te, target_transform=tt)
    test_preds = predict_quantiles(models, X_te, adjustment, tt)
    metrics = evaluate(test_df, test_preds, QUANTILES)
    metrics["interval_calibration"] = {
        "nominal_coverage": round(nominal, 2),
        "raw_coverage": round(
            interval_coverage(test_df[TARGET].to_numpy(float), raw_test_preds[:, 0], raw_test_preds[:, -1]), 4
        ),
        "conformal_coverage": metrics["p10_p90_coverage"],
        "conformal_adjustment_orders": round(adjustment, 4),
    }
    importance = compute_feature_importance(models[0.5], X_te, test_df[TARGET].to_numpy(float), seed, tt)
    logger.info("Holdout metrics: %s", json.dumps(metrics["model_p50"]))

    deployed = models
    if refit_full:
        X_all = to_model_frame(features, columns, levels)
        y_all = features[TARGET].to_numpy(float)
        y_fit, w_fit = y_all, None
        if tt == "level_ratio":
            w_fit = level_scale(X_all)
            y_fit = y_all / w_fit
        deployed = {}
        for alpha in QUANTILES:
            n_trees = max(50, int(round(iterations[alpha] * 1.1)))
            model = make_estimator(alpha, {**params, "max_iter": n_trees}, seed, early_stopping=False)
            deployed[alpha] = model.fit(X_all, y_fit, sample_weight=w_fit)
        logger.info("Refit deployed models on %d rows", len(features))

    version = datetime.now(UTC).strftime("v%Y%m%d-%H%M%S")
    version_dir = model_root / version
    version_dir.mkdir(parents=True, exist_ok=False)
    bundle = {
        "models": deployed,
        "quantiles": list(QUANTILES),
        "interval_adjustment": adjustment,
        "target_transform": tt,
    }
    joblib.dump(bundle, version_dir / ARTIFACT_FILE, compress=3)

    predictions = test_df[["date", "city", "zone_id", "category", TARGET]].copy()
    predictions[["p10", "p50", "p90"]] = test_preds
    for name, pred in baseline_predictions(test_df).items():
        predictions[name] = pred
    predictions.to_csv(version_dir / PREDICTIONS_FILE, index=False, compression="gzip")

    # drift reference = the most recent 180 days the model learned from
    recent_train = train_df.loc[train_df["date"] > train_df["date"].max() - pd.Timedelta(days=180)]
    profile = build_reference_profile(recent_train)
    (version_dir / PROFILE_FILE).write_text(json.dumps(profile))

    metadata = {
        "version": version,
        "package_version": __version__,
        "trained_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "algorithm": "HistGradientBoostingRegressor (quantile loss), one model per quantile",
        "quantiles": list(QUANTILES),
        "interval_adjustment": round(adjustment, 6),
        "target_transform": tt,
        "feature_spec": spec.to_dict(),
        "feature_columns": columns,
        "category_levels": levels,
        "hyperparameters": params,
        "trees_per_quantile": {str(a): iterations[a] for a in QUANTILES},
        "deployed_model": "refit on full history" if refit_full else "holdout model",
        "data": {
            "history_start": str(history["date"].min().date()),
            "history_end": str(history["date"].max().date()),
            "series": int(history.groupby(SERIES_KEYS).ngroups),
            "training_rows": int(len(features)),
        },
        "holdout_metrics": metrics,
        "feature_importance": importance,
        "tuning_trials": trials,
        "training_seconds": round(time.perf_counter() - started, 1),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "artifact_sha256": _sha256(version_dir / ARTIFACT_FILE),
    }
    (version_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2))
    (model_root / LATEST_POINTER).write_text(version)
    logger.info("Registered model %s in %s (%.1fs)", version, version_dir, metadata["training_seconds"])
    return TrainingResult(version=version, model_dir=version_dir, metadata=metadata)


def main(argv: list[str] | None = None) -> None:
    from .config import get_settings

    parser = argparse.ArgumentParser(description="Train the DemandMesh forecaster")
    parser.add_argument("--tune", action="store_true", help="randomised hyper-parameter search")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--no-refit", action="store_true", help="deploy the holdout model")
    args = parser.parse_args(argv)

    settings = get_settings()
    settings.ensure_dirs()
    result = train(
        load_history(settings.history_path),
        load_zones(settings.zones_path),
        settings.model_dir,
        tune=args.tune,
        n_trials=args.n_trials,
        refit_full=not args.no_refit,
        seed=settings.random_seed,
    )
    print(json.dumps(result.metadata["holdout_metrics"], indent=2))


if __name__ == "__main__":
    main()
