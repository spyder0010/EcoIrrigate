"""Numerical and chronological regression checks for the forecast comparison."""
import unittest

import numpy as np
import pandas as pd

from continuous_scale_bounds import score_witness, verify_kernel, verify_margin
from noaa_residual_baseline import make_frame_from_raw
from scale_sampling_sensitivity import check_deviation_function


class ScaleComparisonTests(unittest.TestCase):
    def test_kernel_matches_dense_reference(self):
        verify_kernel()

    def test_coarse_grid_contains_off_grid_scores(self):
        verify_margin()

    def test_resampling_deviation_covers_joint_maps(self):
        check_deviation_function()

    def test_feasible_nonlinear_map_reverses_rank(self):
        model, persistence, observed = map(np.array, ([0.65], [0.18], [0.42]))
        edges = np.array([0., 0.5, 1.])
        transformed = np.array([0., 0.375, 1.])
        slopes = np.diff(transformed) / np.diff(edges)
        self.assertTrue(np.all(np.abs(slopes - 1) <= 0.25))
        identity_gap = score_witness(model, persistence, observed, edges, edges)
        changed_gap = score_witness(model, persistence, observed, edges, transformed)
        self.assertLess(identity_gap, 0)
        self.assertGreater(changed_gap, 0)
        self.assertAlmostEqual(changed_gap, 0.2475**2 - 0.18**2)


class ForecastOriginTests(unittest.TestCase):
    def test_gap_invalidates_history_and_shifted_target(self):
        times = pd.date_range("2025-12-29", periods=83, freq="h").delete(30)
        raw = pd.DataFrame({
            "timestamp_utc": times,
            "SOIL_MOISTURE_5": np.linspace(0.15, 0.25, len(times)),
            "T_HR_AVG": 20., "RH_HR_AVG": 50., "SOLARAD": 100., "P_CALC": 0.,
        })
        result = make_frame_from_raw(raw, "test_station")
        self.assertTrue(result.loc[29, "valid_history"])
        self.assertEqual(result.loc[29, "target_time_1"] - result.loc[29, "origin_time"],
                         pd.Timedelta(hours=2))
        self.assertFalse(result.loc[30, "valid_history"])
        self.assertFalse(result.loc[53, "valid_history"])
        self.assertTrue(result.loc[54, "valid_history"])
        at_boundary = result.origin_time.eq(pd.Timestamp("2025-12-31 23:00"))
        self.assertEqual(result.loc[at_boundary, "target_time_1"].dt.year.item(), 2026)


if __name__ == "__main__":
    unittest.main()
