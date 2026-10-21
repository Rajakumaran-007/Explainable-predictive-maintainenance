"""
Training and Benchmarking Pipeline for Classical ML and Deep Learning Baselines.

Trains and systematically evaluates:
1. Ridge Regression (Statistical Tabular Window Aggregations)
2. Random Forest Regressor (Non-linear Tree Ensemble)
3. Multi-Layer Perceptron (MLP) Baseline
4. LSTM-only Recurrent Baseline
5. 1D-CNN-only Convolutional Baseline

Evaluates all baselines strictly on the final observed cycle of FD001 test engines,
computing RMSE, MAE, R², official NASA score, inference latency, and serialized model size.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from src.data_loader import CMAPSSDataLoader
from src.dataset import build_dataloaders
from src.evaluate import (
    evaluate_pytorch_model,
    evaluate_sklearn_model,
    generate_markdown_comparison_table,
    get_model_file_size_mb,
    measure_pytorch_latency,
)
from src.models import CNN1DBaseline, LSTMBaseline, MLPBaseline, count_parameters
from src.preprocessing import CMAPSSPreprocessor
from src.train import ModelTrainer
from src.utils import get_device, set_seed, setup_logger

logger = setup_logger("train_baselines")


def extract_tabular_features(X_windows: np.ndarray) -> np.ndarray:
    """Extracts summary statistics across the temporal dimension for tabular ML models.

    Computes Mean, Standard Deviation, Minimum, Maximum, and Final Cycle Value
    across each of the D sensor channels over the temporal window of length W.

    Args:
        X_windows: 3D array of shape (N_samples, window_size, num_features).

    Returns:
        2D array of shape (N_samples, num_features * 5).
    """
    mean_feat = np.mean(X_windows, axis=1)
    std_feat = np.std(X_windows, axis=1)
    min_feat = np.min(X_windows, axis=1)
    max_feat = np.max(X_windows, axis=1)
    last_feat = X_windows[:, -1, :]
    return np.hstack([mean_feat, std_feat, min_feat, max_feat, last_feat])


def plot_baseline_predictions(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    save_path: Path,
) -> None:
    """Generates comparative multi-panel plot of predicted vs ground-truth test RUL."""
    # Sort test engines by true RUL for clear visual inspection
    sort_idx = np.argsort(y_true)
    y_true_sorted = y_true[sort_idx]

    plt.figure(figsize=(14, 8), dpi=300)
    plt.plot(y_true_sorted, label="Ground Truth RUL", color="black", linewidth=2.5, linestyle="--")

    colors = {
        "Ridge Regression": "#8c564b",
        "Random Forest": "#2ca02c",
        "MLP Baseline": "#9467bd",
        "LSTM Baseline": "#ff7f0e",
        "1D-CNN Baseline": "#1f77b4",
    }

    for name, preds in predictions.items():
        preds_sorted = preds[sort_idx]
        plt.plot(preds_sorted, label=name, color=colors.get(name, "gray"), alpha=0.8, linewidth=1.5)

    plt.title("Baseline Models: RUL Prediction vs Ground Truth (FD001 Test Engines)", fontweight="bold", fontsize=14)
    plt.xlabel("Test Engine Units (Sorted by True Remaining Life)", fontsize=12)
    plt.ylabel("Remaining Useful Life (Cycles)", fontsize=12)
    plt.legend(loc="upper left", frameon=True, fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved baseline predictions comparison plot to: {save_path}")


def main():
    set_seed(42)
    device = get_device("mps")

    models_dir = PROJECT_ROOT / "models" / "baselines"
    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = PROJECT_ROOT / "models" / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    logger.info("Loading FD001 dataset...")
    loader = CMAPSSDataLoader()
    train_df = loader.load_train_df(subset="FD001", max_rul=125)
    test_df, test_rul_series = loader.load_test_df(subset="FD001", max_rul=125)

    # 2. Preprocess & Partition
    window_size = 50
    preprocessor = CMAPSSPreprocessor(
        window_size=window_size,
        stride=1,
        max_rul=125,
        include_settings=False,
        variance_threshold=0.01,
        scaler_type="zscore",
    )
    train_part, val_part = preprocessor.split_train_val(train_df, val_ratio=0.20, seed=42)
    preprocessor.fit(train_part)

    train_norm = preprocessor.transform(train_part)
    val_norm = preprocessor.transform(val_part)
    test_norm = preprocessor.transform(test_df)

    # Generate 3D Temporal Windows
    X_train, y_train, _ = preprocessor.create_sliding_windows(train_norm)
    X_val, y_val, _ = preprocessor.create_sliding_windows(val_norm)
    X_test, y_test, _ = preprocessor.create_test_windows(test_norm, last_cycle_only=True)

    logger.info(f"Window Shapes -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Generate 2D Tabular Feature Aggregations for Scikit-Learn Baselines
    X_train_tab = extract_tabular_features(X_train)
    X_val_tab = extract_tabular_features(X_val)
    X_test_tab = extract_tabular_features(X_test)
    logger.info(f"Tabular Shapes -> Train: {X_train_tab.shape}, Test: {X_test_tab.shape}")

    # Build PyTorch DataLoaders
    batch_size = 32
    loaders = build_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=batch_size, num_workers=0
    )

    results: Dict[str, Dict[str, Any]] = {}
    predictions_map: Dict[str, np.ndarray] = {}

    # --------------------------------------------------------------------------
    # Baseline 1: Ridge Regression
    # --------------------------------------------------------------------------
    logger.info("\n========== Training Baseline 1: Ridge Regression ==========")
    ridge = Ridge(alpha=1.0, random_state=42)
    t0 = time.time()
    ridge.fit(X_train_tab, y_train)
    train_time_ridge = time.time() - t0

    ridge_path = models_dir / "ridge_regression.joblib"
    joblib.dump(ridge, ridge_path)

    y_test_ridge, y_pred_ridge, metrics_ridge, lat_ridge = evaluate_sklearn_model(
        ridge, X_test_tab, y_test
    )
    predictions_map["Ridge Regression"] = y_pred_ridge
    results["Ridge Regression"] = {
        **metrics_ridge,
        "parameters": X_train_tab.shape[1] + 1,
        "latency_ms": lat_ridge,
        "size_mb": get_model_file_size_mb(ridge_path),
        "train_time_sec": train_time_ridge,
    }
    logger.info(f"Ridge Test -> RMSE: {metrics_ridge['RMSE']:.2f}, MAE: {metrics_ridge['MAE']:.2f}, NASA: {metrics_ridge['NASA_SCORE']:.1f}")

    # --------------------------------------------------------------------------
    # Baseline 2: Random Forest Regressor
    # --------------------------------------------------------------------------
    logger.info("\n========== Training Baseline 2: Random Forest Regressor ==========")
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    t0 = time.time()
    rf.fit(X_train_tab, y_train)
    train_time_rf = time.time() - t0

    rf_path = models_dir / "random_forest.joblib"
    joblib.dump(rf, rf_path)

    y_test_rf, y_pred_rf, metrics_rf, lat_rf = evaluate_sklearn_model(
        rf, X_test_tab, y_test
    )
    predictions_map["Random Forest"] = y_pred_rf
    results["Random Forest"] = {
        **metrics_rf,
        "parameters": sum(t.tree_.node_count for t in rf.estimators_),
        "latency_ms": lat_rf,
        "size_mb": get_model_file_size_mb(rf_path),
        "train_time_sec": train_time_rf,
    }
    logger.info(f"Random Forest Test -> RMSE: {metrics_rf['RMSE']:.2f}, MAE: {metrics_rf['MAE']:.2f}, NASA: {metrics_rf['NASA_SCORE']:.1f}")

    # --------------------------------------------------------------------------
    # Baseline 3: MLP Baseline
    # --------------------------------------------------------------------------
    logger.info("\n========== Training Baseline 3: Simple MLP Baseline ==========")
    mlp = MLPBaseline(window_size=window_size, num_features=len(preprocessor.feature_cols))
    mlp_trainer = ModelTrainer(
        model=mlp,
        device=device,
        learning_rate=0.001,
        model_name="baseline_mlp",
        checkpoint_dir=checkpoints_dir,
    )
    mlp_train_res = mlp_trainer.train(loaders["train"], loaders["val"], max_epochs=40, patience=8)
    mlp_chk = Path(mlp_train_res["checkpoint_path"])

    y_test_mlp, y_pred_mlp, metrics_mlp = evaluate_pytorch_model(mlp, loaders["test"], device)
    lat_mlp = measure_pytorch_latency(mlp, sample_shape=(1, window_size, len(preprocessor.feature_cols)), device=device)
    predictions_map["MLP Baseline"] = y_pred_mlp
    results["MLP Baseline"] = {
        **metrics_mlp,
        "parameters": count_parameters(mlp),
        "latency_ms": lat_mlp,
        "size_mb": get_model_file_size_mb(mlp_chk),
        "train_time_sec": mlp_train_res["training_time"],
    }
    logger.info(f"MLP Test -> RMSE: {metrics_mlp['RMSE']:.2f}, MAE: {metrics_mlp['MAE']:.2f}, NASA: {metrics_mlp['NASA_SCORE']:.1f}")

    # --------------------------------------------------------------------------
    # Baseline 4: LSTM Baseline
    # --------------------------------------------------------------------------
    logger.info("\n========== Training Baseline 4: LSTM Baseline ==========")
    lstm = LSTMBaseline(num_features=len(preprocessor.feature_cols), hidden_dim=64, num_layers=2)
    lstm_trainer = ModelTrainer(
        model=lstm,
        device=device,
        learning_rate=0.001,
        model_name="baseline_lstm",
        checkpoint_dir=checkpoints_dir,
    )
    lstm_train_res = lstm_trainer.train(loaders["train"], loaders["val"], max_epochs=40, patience=8)
    lstm_chk = Path(lstm_train_res["checkpoint_path"])

    y_test_lstm, y_pred_lstm, metrics_lstm = evaluate_pytorch_model(lstm, loaders["test"], device)
    lat_lstm = measure_pytorch_latency(lstm, sample_shape=(1, window_size, len(preprocessor.feature_cols)), device=device)
    predictions_map["LSTM Baseline"] = y_pred_lstm
    results["LSTM Baseline"] = {
        **metrics_lstm,
        "parameters": count_parameters(lstm),
        "latency_ms": lat_lstm,
        "size_mb": get_model_file_size_mb(lstm_chk),
        "train_time_sec": lstm_train_res["training_time"],
    }
    logger.info(f"LSTM Test -> RMSE: {metrics_lstm['RMSE']:.2f}, MAE: {metrics_lstm['MAE']:.2f}, NASA: {metrics_lstm['NASA_SCORE']:.1f}")

    # --------------------------------------------------------------------------
    # Baseline 5: 1D-CNN Baseline
    # --------------------------------------------------------------------------
    logger.info("\n========== Training Baseline 5: 1D-CNN Baseline ==========")
    cnn1d = CNN1DBaseline(num_features=len(preprocessor.feature_cols), conv_filters=[32, 64, 128])
    cnn1d_trainer = ModelTrainer(
        model=cnn1d,
        device=device,
        learning_rate=0.001,
        model_name="baseline_cnn1d",
        checkpoint_dir=checkpoints_dir,
    )
    cnn1d_train_res = cnn1d_trainer.train(loaders["train"], loaders["val"], max_epochs=40, patience=8)
    cnn1d_chk = Path(cnn1d_train_res["checkpoint_path"])

    y_test_cnn, y_pred_cnn, metrics_cnn = evaluate_pytorch_model(cnn1d, loaders["test"], device)
    lat_cnn = measure_pytorch_latency(cnn1d, sample_shape=(1, window_size, len(preprocessor.feature_cols)), device=device)
    predictions_map["1D-CNN Baseline"] = y_pred_cnn
    results["1D-CNN Baseline"] = {
        **metrics_cnn,
        "parameters": count_parameters(cnn1d),
        "latency_ms": lat_cnn,
        "size_mb": get_model_file_size_mb(cnn1d_chk),
        "train_time_sec": cnn1d_train_res["training_time"],
    }
    logger.info(f"1D-CNN Test -> RMSE: {metrics_cnn['RMSE']:.2f}, MAE: {metrics_cnn['MAE']:.2f}, NASA: {metrics_cnn['NASA_SCORE']:.1f}")

    # --------------------------------------------------------------------------
    # Save Metrics, Markdown Table, and Plots
    # --------------------------------------------------------------------------
    metrics_json_path = results_dir / "metrics" / "baseline_comparison.json"
    metrics_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved benchmark metrics JSON to: {metrics_json_path}")

    md_table = generate_markdown_comparison_table(results)
    table_path = results_dir / "tables" / "baseline_comparison.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with open(table_path, "w") as f:
        f.write("# Baseline Models Performance Comparison (FD001)\n\n" + md_table)
    logger.info(f"Saved Markdown table to: {table_path}")

    print("\n" + md_table)

    plot_path = results_dir / "figures" / "baseline_predictions_comparison.png"
    plot_baseline_predictions(y_test, predictions_map, plot_path)


if __name__ == "__main__":
    main()
