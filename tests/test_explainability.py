"""
Unit tests for Explainable AI (XAI) Integrated Gradients attribution module.
"""

import numpy as np
import pytest
import torch

from src.explainability import SENSOR_PHYSICAL_NAMES, IntegratedGradientsExplainer
from src.models import HybridCNNLSTM


@pytest.fixture
def dummy_hybrid_model():
    """Instantiates a lightweight Hybrid CNN-LSTM model."""
    model = HybridCNNLSTM(num_features=14, lstm_hidden_dim=32, dropout_prob=0.2)
    model.eval()
    return model


def test_integrated_gradients_attribution_shapes(dummy_hybrid_model):
    """Verifies that Integrated Gradients attributions preserve input tensor dimensions."""
    features = [f"s_{i}" for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]]
    explainer = IntegratedGradientsExplainer(dummy_hybrid_model, feature_names=features, device=torch.device("cpu"))

    X_sample = np.random.randn(4, 50, 14).astype(np.float32)
    attributions = explainer.attribute(X_sample, n_steps=10)

    assert attributions.shape == (4, 50, 14)
    # Attribution values should not be completely zero
    assert np.max(np.abs(attributions)) > 0.0


def test_global_importance_aggregation(dummy_hybrid_model):
    """Verifies that fleet-wide global importance correctly sums and ranks features."""
    features = [f"s_{i}" for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]]
    explainer = IntegratedGradientsExplainer(dummy_hybrid_model, feature_names=features, device=torch.device("cpu"))

    X_sample = np.random.randn(5, 50, 14).astype(np.float32)
    attributions = explainer.attribute(X_sample, n_steps=10)
    global_imp = explainer.compute_global_importance(attributions)

    assert len(global_imp) == 14
    assert sum(global_imp.values()) == pytest.approx(100.0, rel=1e-3)
    # Check descending sort order
    values = list(global_imp.values())
    assert all(values[i] >= values[i + 1] for i in range(len(values) - 1))


def test_local_importance_and_temporal_saliency(dummy_hybrid_model):
    """Verifies single-engine local signed attributions and temporal saliency."""
    features = [f"s_{i}" for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]]
    explainer = IntegratedGradientsExplainer(dummy_hybrid_model, feature_names=features, device=torch.device("cpu"))

    X_single = np.random.randn(1, 50, 14).astype(np.float32)
    attr = explainer.attribute(X_single, n_steps=10)

    local_imp = explainer.compute_local_importance(attr[0])
    assert len(local_imp) == 14

    temporal_sal = explainer.compute_temporal_saliency(attr)
    assert temporal_sal.shape == (50,)
    assert np.all(temporal_sal >= 0.0)
