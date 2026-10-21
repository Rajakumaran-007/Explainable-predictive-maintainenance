"""
Unit tests for Deep Learning architectures: parameter verification, forward pass shapes,
MPS hardware execution, and Monte Carlo Dropout functionality.
"""

import pytest
import torch

from src.models import CNN1DBaseline, HybridCNNLSTM, LSTMBaseline, MLPBaseline, count_parameters
from src.utils import get_device


@pytest.fixture
def dummy_input():
    """Generates synthetic multivariate window tensor (Batch=8, W=50, D=14)."""
    return torch.randn(8, 50, 14, dtype=torch.float32)


def test_mlp_baseline_forward(dummy_input):
    """Verifies MLP baseline forward pass and output shape."""
    model = MLPBaseline(window_size=50, num_features=14)
    out = model(dummy_input)
    assert out.shape == (8, 1)
    assert count_parameters(model) > 0


def test_lstm_baseline_forward(dummy_input):
    """Verifies LSTM baseline forward pass and output shape."""
    model = LSTMBaseline(num_features=14, hidden_dim=64, num_layers=2)
    out = model(dummy_input)
    assert out.shape == (8, 1)
    assert count_parameters(model) > 0


def test_cnn1d_baseline_forward(dummy_input):
    """Verifies 1D-CNN baseline forward pass and output shape."""
    model = CNN1DBaseline(num_features=14)
    out = model(dummy_input)
    assert out.shape == (8, 1)
    assert count_parameters(model) > 0


def test_hybrid_cnn_lstm_forward(dummy_input):
    """Verifies Proposed Hybrid 1D-CNN + LSTM forward pass and output shape."""
    model = HybridCNNLSTM(num_features=14, lstm_hidden_dim=64)
    out = model(dummy_input)
    assert out.shape == (8, 1)
    params = count_parameters(model)
    # Total parameters should be compact (< 150k params) for edge efficiency
    assert 10_000 < params < 200_000


def test_mc_dropout_stochasticity(dummy_input):
    """Verifies that force_dropout=True produces stochastic variance during eval mode."""
    model = HybridCNNLSTM(num_features=14, dropout_prob=0.3)
    model.eval()

    # In standard eval mode without force_dropout, two passes should be identical
    out1_det = model(dummy_input, force_dropout=False)
    out2_det = model(dummy_input, force_dropout=False)
    torch.testing.assert_close(out1_det, out2_det)

    # In eval mode with force_dropout=True, multiple passes must produce stochastic variations
    out1_stoch = model(dummy_input, force_dropout=True)
    out2_stoch = model(dummy_input, force_dropout=True)
    assert not torch.allclose(out1_stoch, out2_stoch)


def test_mps_device_execution(dummy_input):
    """Verifies model execution on Apple Silicon MPS device if available."""
    device = get_device("mps")
    model = HybridCNNLSTM(num_features=14).to(device)
    x = dummy_input.to(device)
    out = model(x)
    assert out.device.type == device.type
    assert out.shape == (8, 1)
