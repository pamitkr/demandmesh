"""Forecast accuracy metrics used for model selection, reporting and monitoring.

WAPE is the headline metric: it is scale-free, robust to the many zero days in
B2B data (Sundays, holidays) where MAPE explodes, and it weights errors by
volume - which is what freight cost actually depends on.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-9


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted absolute percentage error: sum|y - yhat| / sum|y|."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.abs(y_true - y_pred).sum() / max(np.abs(y_true).sum(), _EPS))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Relative forecast bias: positive = over-forecast."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float((y_pred.sum() - y_true.sum()) / max(np.abs(y_true).sum(), _EPS))


def pinball_loss(y_true: np.ndarray, q_pred: np.ndarray, alpha: float) -> float:
    """Quantile (pinball) loss - the proper scoring rule for a quantile forecast."""
    diff = np.asarray(y_true, float) - np.asarray(q_pred, float)
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1.0) * diff)))


def interval_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Share of actuals inside [lower, upper] - should match the nominal level."""
    y_true = np.asarray(y_true, float)
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "wape": round(wape(y_true, y_pred), 4),
        "mae": round(float(np.mean(np.abs(y_true - y_pred))), 4),
        "rmse": round(float(np.sqrt(np.mean((y_true - y_pred) ** 2))), 4),
        "bias": round(bias(y_true, y_pred), 4),
    }
