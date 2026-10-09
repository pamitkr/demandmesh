import unittest

import numpy as np
import pandas as pd

from demandmesh.calendar_india import calendar_features
from demandmesh.data_generator import generate_dataset
from demandmesh.data_preprocessing import (
    FeatureSpec,
    build_feature_frame,
    clean_data,
    feature_columns,
    validate_schema,
)
from demandmesh.exceptions import DataValidationError


def _toy_frame() -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=10, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "city": "SRT",
            "zone_id": "SRT-Z01",
            "category": "TEXTILE",
            "enquiries": np.arange(10) * 3,
            "orders": np.arange(10, dtype=float),
        }
    )


class SchemaTests(unittest.TestCase):
    def test_missing_column_is_rejected(self):
        with self.assertRaises(DataValidationError):
            validate_schema(_toy_frame().drop(columns=["orders"]))

    def test_empty_frame_is_rejected(self):
        with self.assertRaises(DataValidationError):
            validate_schema(_toy_frame().iloc[0:0])

    def test_feature_spec_refuses_leaky_lags(self):
        with self.assertRaises(ValueError):
            FeatureSpec(horizon=7, lags=(1, 7))


class CleaningTests(unittest.TestCase):
    def test_duplicates_negatives_and_gaps_are_repaired(self):
        df = _toy_frame()
        df.loc[3, "orders"] = -5  # reversal
        df = df.drop(index=5)  # ingestion gap
        df = pd.concat([df, df.iloc[[0]].assign(orders=4.0)])  # late correction wins
        clean, report = clean_data(df)

        self.assertEqual(report["duplicate_rows_removed"], 1)
        self.assertEqual(report["negative_values_clipped"], 1)
        self.assertEqual(report["missing_days_filled"], 1)
        self.assertEqual(len(clean), 10)
        self.assertTrue((clean["orders"] >= 0).all())
        self.assertEqual(clean.loc[clean["date"] == "2025-01-01", "orders"].item(), 4.0)
        self.assertTrue(clean["date"].diff().dropna().dt.days.eq(1).all())


class LeakageTests(unittest.TestCase):
    def test_future_values_never_change_features_within_horizon(self):
        """Changing anything after the cutoff must not alter a feature on or before
        cutoff + horizon - the guarantee behind the direct multi-horizon design."""
        demand, zones = generate_dataset(
            start="2024-01-01", end="2025-03-31", seed=3, cities=["LDH"], inject_defects=False
        )
        history, _ = clean_data(demand)
        spec = FeatureSpec()
        cutoff = pd.Timestamp("2025-03-10")
        tampered = history.copy()
        future = tampered["date"] > cutoff
        tampered.loc[future, ["orders", "enquiries"]] = 10_000.0

        cols = [c for c in feature_columns(spec) if c not in ("city", "zone_id", "category")]
        window = lambda f: f.loc[f["date"] <= cutoff + pd.Timedelta(days=spec.horizon)]  # noqa: E731
        a = window(build_feature_frame(history, zones, spec, drop_warmup=False))
        b = window(build_feature_frame(tampered, zones, spec, drop_warmup=False))
        pd.testing.assert_frame_equal(a[cols].reset_index(drop=True), b[cols].reset_index(drop=True))


class CalendarTests(unittest.TestCase):
    def test_known_indian_calendar_dates(self):
        dates = pd.to_datetime(
            ["2024-10-30", "2024-11-01", "2025-08-15", "2025-03-20", "2025-07-10", "2025-12-28"]
        )
        cal = calendar_features(dates)
        self.assertEqual(cal.loc[0, "days_to_diwali"], 2)
        self.assertEqual(cal.loc[1, "is_diwali_shutdown"], 1)
        self.assertEqual(cal.loc[2, "is_national_holiday"], 1)
        self.assertEqual(cal.loc[3, "is_fy_end_fortnight"], 1)
        self.assertEqual(cal.loc[4, "is_monsoon"], 1)
        self.assertEqual(cal.loc[5, "is_sunday"], 1)


if __name__ == "__main__":
    unittest.main()
