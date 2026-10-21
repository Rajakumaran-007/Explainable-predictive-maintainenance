"""
Deep Learning Architectures for Remaining Useful Life (RUL) Prediction.

Implements:
1. MLPBaseline: Multi-layer perceptron flattening temporal windows.
2. LSTMBaseline: Pure recurrent multi-layer LSTM network.
3. CNN1DBaseline: Pure temporal 1D convolutional network with global pooling.
4. HybridCNNLSTM: Proposed hybrid architecture uniting 1D temporal convolutions
   with recurrent sequence modeling and Monte Carlo Dropout for epistemic uncertainty.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBaseline(nn.Module):
    """Multi-Layer Perceptron baseline flattening temporal sequence windows."""

    def __init__(
        self,
        window_size: int = 50,
        num_features: int = 14,
        hidden_dims: List[int] = [128, 64, 32],
        dropout_prob: float = 0.2,
    ):
        super().__init__()
        self.window_size = window_size
        self.num_features = num_features
        input_dim = window_size * num_features

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_prob))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        Args:
            x: Tensor of shape (Batch, window_size, num_features).
        Returns:
            Scalar RUL predictions of shape (Batch, 1).
        """
        # Flatten temporal window: (Batch, window_size * num_features)
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        return self.network(x_flat)


class LSTMBaseline(nn.Module):
    """Pure Recurrent LSTM Baseline for long-range temporal sequence modeling."""

    def __init__(
        self,
        num_features: int = 14,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout_prob: float = 0.2,
        fc_hidden_dims: List[int] = [32],
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0.0,
        )

        fc_layers = []
        prev_dim = hidden_dim
        for h_dim in fc_hidden_dims:
            fc_layers.append(nn.Linear(prev_dim, h_dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(p=dropout_prob))
            prev_dim = h_dim

        fc_layers.append(nn.Linear(prev_dim, 1))
        self.head = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        Args:
            x: Tensor of shape (Batch, window_size, num_features).
        Returns:
            Scalar RUL predictions of shape (Batch, 1).
        """
        # out: (Batch, window_size, hidden_dim)
        lstm_out, _ = self.lstm(x)
        # Take the final temporal hidden state representation
        last_hidden = lstm_out[:, -1, :]
        return self.head(last_hidden)


class CNN1DBaseline(nn.Module):
    """Pure Temporal 1D-Convolutional Baseline with Adaptive Pooling."""

    def __init__(
        self,
        num_features: int = 14,
        conv_filters: List[int] = [32, 64, 128],
        kernel_sizes: List[int] = [5, 3, 3],
        dropout_prob: float = 0.2,
        fc_hidden_dims: List[int] = [64, 32],
    ):
        super().__init__()
        assert len(conv_filters) == len(kernel_sizes), "Filter and kernel lists must match length."

        conv_layers = []
        in_channels = num_features
        for out_channels, k_size in zip(conv_filters, kernel_sizes):
            padding = k_size // 2  # 'same' padding
            conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=k_size, padding=padding))
            conv_layers.append(nn.BatchNorm1d(out_channels))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.Dropout(p=dropout_prob))
            in_channels = out_channels

        self.conv_net = nn.Sequential(*conv_layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        fc_layers = []
        prev_dim = conv_filters[-1]
        for h_dim in fc_hidden_dims:
            fc_layers.append(nn.Linear(prev_dim, h_dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(p=dropout_prob))
            prev_dim = h_dim

        fc_layers.append(nn.Linear(prev_dim, 1))
        self.head = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        Args:
            x: Tensor of shape (Batch, window_size, num_features).
        Returns:
            Scalar RUL predictions of shape (Batch, 1).
        """
        # Conv1D expects (Batch, Channels=num_features, Length=window_size)
        x_trans = x.transpose(1, 2)
        features = self.conv_net(x_trans)
        pooled = self.global_pool(features).squeeze(-1)
        return self.head(pooled)


class HybridCNNLSTM(nn.Module):
    """Proposed Hybrid 1D-CNN + LSTM Deep Neural Network for RUL Prediction.

    Synergistically combines multi-scale temporal 1D convolutions (for extracting
    localized sensor cross-correlations) with multi-layer LSTM recurrent units
    (for capturing non-linear cumulative degradation dynamics over extended cycles).

    Equipped with Monte Carlo Dropout for epistemic predictive uncertainty sampling.
    """

    def __init__(
        self,
        num_features: int = 14,
        conv_filters: List[int] = [32, 64],
        kernel_sizes: List[int] = [5, 3],
        lstm_hidden_dim: int = 64,
        lstm_num_layers: int = 2,
        dropout_prob: float = 0.2,
        fc_hidden_dims: List[int] = [64, 32],
    ):
        super().__init__()
        self.num_features = num_features
        self.dropout_prob = dropout_prob

        # Temporal 1D Convolutional Feature Extractor
        conv_layers = []
        in_channels = num_features
        for out_channels, k_size in zip(conv_filters, kernel_sizes):
            padding = k_size // 2  # 'same' padding keeps temporal sequence length W
            conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=k_size, padding=padding))
            conv_layers.append(nn.BatchNorm1d(out_channels))
            conv_layers.append(nn.ReLU())
            in_channels = out_channels

        self.conv_blocks = nn.Sequential(*conv_layers)

        # Recurrent Sequence Modeling
        self.lstm = nn.LSTM(
            input_size=conv_filters[-1],
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout_prob if lstm_num_layers > 1 else 0.0,
        )

        # Monte Carlo Dropout Layer
        self.dropout = nn.Dropout(p=dropout_prob)

        # Fully Connected Regression Head
        fc_layers = []
        prev_dim = lstm_hidden_dim
        for h_dim in fc_hidden_dims:
            fc_layers.append(nn.Linear(prev_dim, h_dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(p=dropout_prob))
            prev_dim = h_dim

        fc_layers.append(nn.Linear(prev_dim, 1))
        self.head = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor, force_dropout: bool = False) -> torch.Tensor:
        """Forward pass.
        Args:
            x: Tensor of shape (Batch, window_size, num_features).
            force_dropout: If True, activates dropout even during eval mode (MC Dropout).
        Returns:
            Scalar RUL predictions of shape (Batch, 1).
        """
        # 1. 1D Convolutions: Transpose to (Batch, Channels=num_features, Length=W)
        x_trans = x.transpose(1, 2)
        conv_feats = self.conv_blocks(x_trans)  # (Batch, 64, W)

        # 2. Transpose back to (Batch, Sequence=W, Features=64) for LSTM
        lstm_in = conv_feats.transpose(1, 2)
        lstm_out, _ = self.lstm(lstm_in)  # (Batch, W, lstm_hidden_dim)

        # 3. Extract final temporal cycle representation
        final_state = lstm_out[:, -1, :]

        # 4. Epistemic Uncertainty Dropout
        if force_dropout:
            final_state = F.dropout(final_state, p=self.dropout_prob, training=True)
        else:
            final_state = self.dropout(final_state)

        # 5. Regression Head
        return self.head(final_state)


def count_parameters(model: nn.Module) -> int:
    """Computes total number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
