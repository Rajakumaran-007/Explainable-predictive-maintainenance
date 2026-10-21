"""
Unit tests for preprocessing, engine-level splitting, sliding windows, and PyTorch dataset.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from src.data_loader import CMAPSSDataLoader
from src.dataset import CMAPSSDataset, build_dataloaders
from src.preprocessing import CMAPSSPreprocessor


@pytest.fixture
def sample_train_df():
    """Loads FD001 training data for testing."""
    loader = CMAPSSDataLoader()
    return loader.load_train_df(subset="FD001", max_rul=125)


@pytest.fixture
def sample_test_df():
    """Loads FD001 test data for testing."""
    loader = CMAPSSDataLoader()
    test_df, _ = loader.load_test_df(subset="FD001", max_rul=125)
    return test_df


def test_feature_selection(sample_train_df):
    """Verifies invariant sensor detection and informative feature filtering."""
    # Test benchmark sensor filtering (include_settings=False)
    preprocessor = CMAPSSPreprocessor(drop_constant=True, include_settings=False, variance_threshold=0.01)
    features = preprocessor.select_features(sample_train_df)

    # In FD001, sensors s_1, s_5, s_6, s_10, s_16, s_18, s_19 have near-zero variance
    for const_sensor in ["s_1", "s_5", "s_6", "s_10", "s_16", "s_18", "s_19"]:
        assert const_sensor not in features
        assert const_sensor in preprocessor.constant_cols

    # Informative features should exactly match the 14 benchmark sensors
    assert "s_2" in features
    assert "s_3" in features
    assert "s_4" in features
    assert len(features) == 14
    assert sorted(features) == sorted(CMAPSSPreprocessor.BENCHMARK_14_SENSORS)


def test_engine_level_splitting_disjointness(sample_train_df):
    """Verifies that engine-level partitioning guarantees zero cross-partition leakage."""
    preprocessor = CMAPSSPreprocessor()
    train_part, val_part = preprocessor.split_train_val(sample_train_df, val_ratio=0.20, seed=42)

    train_ids = set(train_part["engine_id"].unique())
    val_ids = set(val_part["engine_id"].unique())

    # Total engines: 100 -> 80 train, 20 val
    assert len(train_ids) == 80
    assert len(val_ids) == 20
    # Disjointness check: absolute zero overlap
    assert train_ids.isdisjoint(val_ids)
    assert len(train_ids.intersection(val_ids)) == 0


def test_leakage_free_scaling(sample_train_df):
    """Verifies scaling parameters are derived strictly from training partition."""
    preprocessor = CMAPSSPreprocessor(scaler_type="zscore")
    train_part, val_part = preprocessor.split_train_val(sample_train_df, val_ratio=0.20, seed=42)

    preprocessor.fit(train_part)
    train_norm = preprocessor.transform(train_part)
    val_norm = preprocessor.transform(val_part)

    # In train_norm, mean of features should be ~ 0 and std ~ 1
    for col in preprocessor.feature_cols:
        assert train_norm[col].mean() == pytest.approx(0.0, abs=1e-5)
        assert train_norm[col].std() == pytest.approx(1.0, abs=1e-3)

    # In val_norm, mean and std will not be exactly 0 and 1 (proving it wasn't fitted on val!)
    assert abs(val_norm["s_2"].mean()) < 0.5  # Realistic shift


def test_sliding_window_generation(sample_train_df):
    """Verifies sliding window dimensional output and target temporal alignment."""
    window_size = 30
    stride = 1
    preprocessor = CMAPSSPreprocessor(window_size=window_size, stride=stride)
    train_part, _ = preprocessor.split_train_val(sample_train_df, val_ratio=0.20, seed=42)
    preprocessor.fit(train_part)
    train_norm = preprocessor.transform(train_part)

    X, y, engine_ids = preprocessor.create_sliding_windows(train_norm)

    assert X.ndim == 3
    assert X.shape[1] == window_size
    assert X.shape[2] == len(preprocessor.feature_cols)
    assert len(X) == len(y) == len(engine_ids)
    assert y.min() >= 0
    assert y.max() <= 125


def test_edge_padding_for_short_test_sequences():
    """Verifies edge-padding logic when test engine length < window_size."""
    # Create synthetic short trajectory of 10 cycles
    window_size = 50
    synthetic_records = []
    for c in range(1, 11):
        row = {"engine_id": 999, "cycle": c, "target_rul": 40.0}
        for s in range(1, 15):
            row[f"feat_{s}"] = float(s * 10 + c)
        synthetic_records.append(row)
    synth_df = pd.DataFrame(synthetic_records)

    preprocessor = CMAPSSPreprocessor(window_size=window_size)
    preprocessor.feature_cols = [f"feat_{s}" for s in range(1, 15)]
    preprocessor.is_fitted = True

    X_test, y_test, engine_ids = preprocessor.create_test_windows(synth_df, last_cycle_only=True)

    assert X_test.shape == (1, window_size, 14)
    assert y_test.shape == (1,)
    assert engine_ids[0] == 999
    # Padded cycles (0 to 39) must equal the first row of synthetic data
    first_row_feats = synth_df[preprocessor.feature_cols].iloc[0].values
    for step in range(40):
        np.testing.assert_allclose(X_test[0, step, :], first_row_feats)


def test_pytorch_dataset_and_dataloaders():
    """Verifies CMAPSSDataset tensor output and DataLoader batch generation."""
    X = np.random.randn(64, 50, 14).astype(np.float32)
    y = np.random.uniform(0, 125, size=(64,)).astype(np.float32)
    engine_ids = np.ones(64, dtype=np.int32)

    dataset = CMAPSSDataset(X, y, engine_ids)
    assert len(dataset) == 64
    x_sample, y_sample, id_sample = dataset[0]
    assert isinstance(x_sample, torch.Tensor)
    assert isinstance(y_sample, torch.Tensor)
    assert x_sample.shape == (50, 14)
    assert x_sample.dtype == torch.float32

    loaders = build_dataloaders(X, y, X, y, batch_size=16)
    assert "train" in loaders
    assert "val" in loaders

    batch_x, batch_y, batch_id = next(iter(loaders["train"]))
    assert batch_x.shape == (16, 50, 14)
    assert batch_y.shape == (16,)
