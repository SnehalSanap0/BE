"""Regression tests for publication-critical split and metric behavior."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from dataset.split_assigner import _grouped_assign
from evaluation.metrics import compute_metrics


class GroupedSplitTests(unittest.TestCase):
    def test_groups_never_cross_splits(self) -> None:
        groups = pd.Series(["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"])
        assignments = _grouped_assign(groups, 0.6, 0.2, 0.2, random_seed=42)
        frame = pd.DataFrame({"group": groups, "split": assignments})
        self.assertTrue((frame.groupby("group")["split"].nunique() == 1).all())

    def test_grouped_assignment_is_reproducible(self) -> None:
        groups = pd.Series([f"g{index // 2}" for index in range(20)])
        first = _grouped_assign(groups, 0.7, 0.15, 0.15, random_seed=1337)
        second = _grouped_assign(groups, 0.7, 0.15, 0.15, random_seed=1337)
        pd.testing.assert_series_equal(first, second)

    def test_invalid_ratios_fail(self) -> None:
        with self.assertRaises(ValueError):
            _grouped_assign(pd.Series(["a", "a"]), 0.8, 0.2, 0.2, random_seed=42)


class MetricTests(unittest.TestCase):
    def test_balanced_metrics_are_reported(self) -> None:
        result = compute_metrics(
            np.array([0, 0, 1, 1]),
            np.array([0, 1, 1, 1]),
            np.array([0.1, 0.6, 0.7, 0.9]),
        )
        self.assertAlmostEqual(result["balanced_accuracy"], 0.75)
        self.assertIn("mcc", result)


if __name__ == "__main__":
    unittest.main()
