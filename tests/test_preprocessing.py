"""
Unit tests for C-MAPSS data loading, integrity checking, and utility functions.
"""

import pytest
import numpy as np
import torch
from pathlib import Path

from src.utils import set_seed, get_device, compute_nasa_score, compute_regression_metrics
from src.data_loader import CMAPSSDataLoader, SENSOR_COLS, IDENTIFIER_COLS, SETTING_COLS


def test_seed_reproducibility():
    """Verifies that set_seed ensures reproducible random numbers."""
    set_seed(42)
    a1 = np.random.randn(5)
    t1 = torch.randn(5)

    set_seed(42)
    a2 = np.random.randn(5)
    t2 = torch.randn(5)

    np.testing.assert_allclose(a1, a2)
    assert torch.allclose(t1, t2)


def test_get_device():
    """Verifies MPS hardware acceleration detection and device fallback."""
    device = get_device("mps")
    assert isinstance(device, torch.device)
    assert device.type in ["mps", "cpu"]


def test_nasa_asymmetric_scoring():
    """Verifies mathematical correctness of NASA PHM 2008 asymmetric penalty function."""
    # When error is zero: score should be zero
    y_true = np.array([50.0, 50.0])
    y_pred_zero = np.array([50.0, 50.0])
    assert compute_nasa_score(y_true, y_pred_zero) == pytest.approx(0.0, abs=1e-6)

    # Test early prediction (d = -10): d < 0 -> exp(-(-10)/13) - 1 = exp(10/13) - 1 = 1.158
    y_pred_early = np.array([40.0])
    y_true_single = np.array([50.0])
    score_early = compute_nasa_score(y_true_single, y_pred_early)
    expected_early = np.exp(10.0 / 13.0) - 1.0
    assert score_early == pytest.approx(expected_early, rel=1e-4)

    # Test late prediction (d = +10): d >= 0 -> exp(10/10) - 1 = exp(1) - 1 = 1.718
    y_pred_late = np.array([60.0])
    score_late = compute_nasa_score(y_true_single, y_pred_late)
    expected_late = np.exp(10.0 / 10.0) - 1.0
    assert score_late == pytest.approx(expected_late, rel=1e-4)

    # Late predictions must be penalized more heavily than early predictions for identical absolute error
    assert score_late > score_early


def test_regression_metrics_dictionary():
    """Verifies metrics computation dictionary contents."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([11.0, 19.0, 30.0])
    metrics = compute_regression_metrics(y_true, y_pred)
    assert "RMSE" in metrics
    assert "MAE" in metrics
    assert "R2" in metrics
    assert "NASA_SCORE" in metrics
    assert metrics["MAE"] == pytest.approx(2.0 / 3.0)


def test_cmapss_data_loader_fd001():
    """Verifies loading of FD001 dataset with ground-truth capped RUL."""
    loader = CMAPSSDataLoader()
    train_df = loader.load_train_df(subset="FD001", max_rul=125)

    assert train_df["engine_id"].nunique() == 100
    assert "target_rul" in train_df.columns
    assert "raw_rul" in train_df.columns
    assert train_df["target_rul"].max() <= 125
    assert train_df["target_rul"].min() == 0

    test_df, test_rul = loader.load_test_df(subset="FD001", max_rul=125)
    assert test_df["engine_id"].nunique() == 100
    assert len(test_rul) == 100
