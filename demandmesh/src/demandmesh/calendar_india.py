"""Indian business calendar: festivals, national holidays, fiscal year and monsoon.

Everything here is a *known-in-advance* covariate, so it is legitimate to use
for future dates at inference time (unlike lags of the target).

Lunar-calendar festivals move every year; their dates are listed explicitly
(per the Indian civil calendar - observance can differ by a day regionally).
Extend the tuples each year; when a date falls beyond the last listed festival
the distance features saturate at their cap instead of failing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DIWALI = ("2022-10-24", "2023-11-12", "2024-11-01", "2025-10-20", "2026-11-08")
NAVRATRI_START = ("2022-09-26", "2023-10-15", "2024-10-03", "2025-09-22", "2026-10-11")
DUSSEHRA = ("2022-10-05", "2023-10-24", "2024-10-12", "2025-10-02", "2026-10-20")
HOLI = ("2023-03-08", "2024-03-25", "2025-03-14", "2026-03-04")
EID_AL_FITR = ("2023-04-22", "2024-04-11", "2025-03-31", "2026-03-21")
GANESH_CHATURTHI = ("2022-08-31", "2023-09-19", "2024-09-07", "2025-08-27", "2026-09-14")
NATIONAL_HOLIDAYS_MD = ((1, 26), (8, 15), (10, 2))  # Republic, Independence, Gandhi Jayanti
MAKAR_SANKRANTI_MD = (1, 14)

FESTIVAL_DISTANCE_CAP = 45  # days; beyond this a festival has no measurable pull

CALENDAR_FEATURES: tuple[str, ...] = (
    "dow",
    "dom",
    "month",
    "week_of_year",
    "doy_sin1",
    "doy_cos1",
    "doy_sin2",
    "doy_cos2",
    "is_saturday",
    "is_sunday",
    "is_month_end_window",
    "is_quarter_end_window",
    "is_fy_end_fortnight",
    "is_monsoon",
    "days_to_diwali",
    "days_since_diwali",
    "is_diwali_shutdown",
    "days_to_festival",
    "is_festival_day",
    "is_national_holiday",
)


def _to_days(values: pd.Series | pd.DatetimeIndex | np.ndarray) -> np.ndarray:
    return np.asarray(pd.to_datetime(values)).astype("datetime64[D]")


def _sorted_days(dates: tuple[str, ...]) -> np.ndarray:
    return np.sort(np.array(dates, dtype="datetime64[D]"))


def days_until_next(dates: np.ndarray, events: np.ndarray, cap: int) -> np.ndarray:
    """Days from each date to the next event on/after it, capped at ``cap``."""
    idx = np.searchsorted(events, dates, side="left")
    out = np.full(dates.shape, float(cap))
    ok = idx < len(events)
    out[ok] = (events[idx[ok]] - dates[ok]).astype("timedelta64[D]").astype(float)
    return np.minimum(out, cap)


def days_since_last(dates: np.ndarray, events: np.ndarray, cap: int) -> np.ndarray:
    """Days since the most recent event on/before each date, capped at ``cap``."""
    idx = np.searchsorted(events, dates, side="right") - 1
    out = np.full(dates.shape, float(cap))
    ok = idx >= 0
    out[ok] = (dates[ok] - events[idx[ok]]).astype("timedelta64[D]").astype(float)
    return np.minimum(out, cap)


def _all_festival_days(years: np.ndarray) -> np.ndarray:
    lunar = DIWALI + NAVRATRI_START + DUSSEHRA + HOLI + EID_AL_FITR + GANESH_CHATURTHI
    fixed = [f"{y}-{MAKAR_SANKRANTI_MD[0]:02d}-{MAKAR_SANKRANTI_MD[1]:02d}" for y in years]
    return _sorted_days(tuple(lunar) + tuple(fixed))


def calendar_features(dates: pd.Series | pd.DatetimeIndex) -> pd.DataFrame:
    """Vectorised calendar covariates for an array of dates (index is positional)."""
    ts = pd.DatetimeIndex(pd.to_datetime(dates))
    days = _to_days(ts)
    doy = ts.dayofyear.to_numpy(dtype=float)
    dom = ts.day.to_numpy()
    month = ts.month.to_numpy()
    dim = ts.days_in_month.to_numpy()
    dow = ts.dayofweek.to_numpy()

    diwali = _sorted_days(DIWALI)
    years = np.unique(ts.year.to_numpy()) if len(ts) else np.array([], dtype=int)
    festivals = _all_festival_days(np.concatenate([years - 1, years, years + 1]))
    to_diwali = days_until_next(days, diwali, FESTIVAL_DISTANCE_CAP)
    since_diwali = days_since_last(days, diwali, FESTIVAL_DISTANCE_CAP)
    to_festival = days_until_next(days, festivals, FESTIVAL_DISTANCE_CAP)

    md = list(zip(month, dom, strict=True))
    national = np.array([(m, d) in NATIONAL_HOLIDAYS_MD for m, d in md], dtype=np.int8)

    two_pi = 2.0 * np.pi
    return pd.DataFrame(
        {
            "dow": dow,
            "dom": dom,
            "month": month,
            "week_of_year": ts.isocalendar().week.to_numpy(dtype=int),
            "doy_sin1": np.sin(two_pi * doy / 365.25),
            "doy_cos1": np.cos(two_pi * doy / 365.25),
            "doy_sin2": np.sin(2 * two_pi * doy / 365.25),
            "doy_cos2": np.cos(2 * two_pi * doy / 365.25),
            "is_saturday": (dow == 5).astype(np.int8),
            "is_sunday": (dow == 6).astype(np.int8),
            "is_month_end_window": (dom >= dim - 2).astype(np.int8),
            "is_quarter_end_window": (np.isin(month, (3, 6, 9, 12)) & (dom >= dim - 6)).astype(np.int8),
            "is_fy_end_fortnight": ((month == 3) & (dom >= 16)).astype(np.int8),
            "is_monsoon": (((month == 6) & (dom >= 15)) | np.isin(month, (7, 8, 9))).astype(np.int8),
            "days_to_diwali": to_diwali,
            "days_since_diwali": since_diwali,
            "is_diwali_shutdown": ((to_diwali <= 1) | (since_diwali <= 2)).astype(np.int8),
            "days_to_festival": to_festival,
            "is_festival_day": (to_festival == 0).astype(np.int8),
            "is_national_holiday": national,
        }
    )
