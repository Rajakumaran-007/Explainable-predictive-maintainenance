"""
Unit tests for Uncertainty Quantification: Monte Carlo Dropout and Conformal Prediction.
"""

import numpy as np
import pytest
import torch

from src.models import HybridCNNLSTM
from src.uncertainty import MCDropoutEstimator, SplitConformalCalibrator


@pytest.fixture
def dummy_hybrid_model():
    """Instantiates a lightweight Hybrid CNN-LSTM model."""
    model = HybridCNNLSTM(num_features=14, lstm_hidden_dim=32, dropout_prob=0.3)
    return model


def test_mc_dropout_estimator_shapes_and_positivity(dummy_hybrid_model):
    """Verifies that MCDropoutEstimator generates correct shapes and positive variances."""
    estimator = MCDropoutEstimator(dummy_hybrid_model, device=torch.device("cpu"))
    X_dummy = np.random.randn(10, 50, 14).astype(np.float32)

    mean_pred, std_pred, all_samples = estimator.predict_with_uncertainty(X_dummy, num_passes=15)

    assert mean_pred.shape == (10,)
    assert std_pred.shape == (10,)
    assert all_samples.shape == (15, 10)
    # Standard deviations must be non-negative
    assert np.all(std_pred >= 0.0)
    # With active dropout, standard deviations should be strictly positive
    assert np.mean(std_pred) > 0.0


def test_conformal_calibrator_fit_and_predict():
    """Verifies standard split conformal calibrator interval construction and coverage."""
    calibrator = SplitConformalCalibrator(alpha=0.10, adaptive=False)

    # Synthetic calibration set
    np.random.seed(42)
    y_calib_true = np.linspace(10, 100, 100)
    y_calib_pred = y_calib_true + np.random.normal(0, 5.0, size=100)

    calibrator.fit(y_calib_true, y_calib_pred)
    assert calibrator.is_calibrated
    assert calibrator.q_hat > 0

    # Predict on test set
    y_test_true = np.linspace(15, 95, 50)
    y_test_pred = y_test_true + np.random.normal(0, 5.0, size=50)

    lower, upper = calibrator.predict_intervals(y_test_pred)
    assert lower.shape == (50,)
    assert upper.shape == (50,)
    assert np.all(lower <= upper)

    eval_res = calibrator.evaluate_coverage(y_test_true, lower, upper)
    # Nominal 90% coverage: empirical should be near or above 85%
    assert eval_res["empirical_coverage_pct"] >= 80.0
    assert eval_res["mean_interval_width"] > 0


def test_adaptive_conformal_calibrator():
    """Verifies locally-adaptive conformal prediction with variance scaling."""
    calibrator = SplitConformalCalibrator(alpha=0.10, adaptive=True)

    np.random.seed(42)
    n = 100
    y_calib_true = np.random.uniform(10, 100, size=n)
    # Heteroscedastic noise proportional to standard deviation
    y_calib_std = np.random.uniform(1.0, 5.0, size=n)
    y_calib_pred = y_calib_true + np.random.normal(0, y_calib_std)

    calibrator.fit(y_calib_true, y_calib_pred, y_calib_std=y_calib_std)
    assert calibrator.q_hat > 0

    y_test_pred = np.array([50.0, 50.0])
    # First sample has low uncertainty, second has high uncertainty
    y_test_std = np.array([1.0, 10.0])

    lower, upper = calibrator.predict_intervals(y_test_pred, y_std=y_test_std)
    widths = upper - lower

    # Adaptive property: interval width for high-uncertainty sample must be wider
    assert widths[1] > widths[0]
