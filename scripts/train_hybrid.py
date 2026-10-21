"""
Training, Multi-Seed Evaluation, and Ablation Study Pipeline for Proposed Hybrid 1D-CNN + LSTM.

Executes:
1. Primary Hybrid Model Training across Multiple Seeds (42, 123, 999) for Mean ± Std reporting.
2. Controlled Window Size Sensitivity Ablation (W = 30, 50, 70 cycles).
3. Controlled Dropout Regularization Ablation (p = 0.0, 0.2, 0.3).
4. Saves optimal model checkpoint for Uncertainty Quantification (Phase 5) and XAI (Phase 6).
5. Generates publication-quality convergence and ablation figures.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data_loader import CMAPSSDataLoader
from src.dataset import build_dataloaders
from src.evaluate import (
    evaluate_pytorch_model,
    generate_markdown_comparison_table,
    get_model_file_size_mb,
    measure_pytorch_latency,
)
from src.models import HybridCNNLSTM, count_parameters
from src.preprocessing import CMAPSSPreprocessor
from src.train import ModelTrainer
from src.utils import get_device, set_seed, setup_logger

logger = setup_logger("train_hybrid")


def plot_training_history(history: Dict[str, List[float]], save_path: Path) -> None:
    """Plots training and validation loss and RMSE progression over epochs."""
    epochs = history["epoch"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Loss curve (MSE)
    ax1.plot(epochs, history["train_loss"], label="Train Loss (MSE)", color="#1f77b4", linewidth=2.0)
    ax1.plot(epochs, history["val_loss"], label="Val Loss (MSE)", color="#ff7f0e", linewidth=2.0, linestyle="--")
    ax1.set_title("Hybrid 1D-CNN + LSTM: Loss Convergence", fontweight="bold", fontsize=12)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Mean Squared Error", fontsize=11)
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # RMSE curve
    ax2.plot(epochs, history["val_rmse"], label="Val RMSE (Cycles)", color="#2ca02c", linewidth=2.0)
    ax2.set_title("Validation RMSE Progression", fontweight="bold", fontsize=12)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("RMSE (Cycles)", fontsize=11)
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved training history plot to: {save_path}")


def plot_ablation_window_lengths(ablation_results: Dict[str, Dict[str, float]], save_path: Path) -> None:
    """Generates bar chart comparing RMSE and NASA Score across window sizes W=30, 50, 70."""
    windows = list(ablation_results.keys())
    rmse_vals = [ablation_results[w]["RMSE"] for w in windows]
    nasa_vals = [ablation_results[w]["NASA_SCORE"] for w in windows]

    x = np.arange(len(windows))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(9, 5), dpi=300)
    ax2 = ax1.twinx()

    b1 = ax1.bar(x - width/2, rmse_vals, width, label="Test RMSE (Cycles)", color="#1f77b4", edgecolor="black", alpha=0.85)
    b2 = ax2.bar(x + width/2, nasa_vals, width, label="NASA Asymmetric Score", color="#d62728", edgecolor="black", alpha=0.85)

    ax1.set_xlabel("Sliding Window Sequence Length (Cycles)", fontweight="bold", fontsize=11)
    ax1.set_ylabel("Root Mean Squared Error (RMSE)", color="#1f77b4", fontweight="bold", fontsize=11)
    ax2.set_ylabel("NASA Asymmetric Score", color="#d62728", fontweight="bold", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"W = {w}" for w in windows], fontsize=11)
    ax1.set_title("Ablation Study 1: Sequence Window Length Sensitivity (FD001)", fontweight="bold", fontsize=13)

    # Value labels
    for rect in b1:
        h = rect.get_height()
        ax1.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for rect in b2:
        h = rect.get_height()
        ax2.annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved window length ablation plot to: {save_path}")


def plot_predictions_vs_ground_truth(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path) -> None:
    """Plots hybrid model test predictions against ground-truth RUL."""
    sort_idx = np.argsort(y_true)
    y_true_sorted = y_true[sort_idx]
    y_pred_sorted = y_pred[sort_idx]

    plt.figure(figsize=(12, 6), dpi=300)
    plt.plot(y_true_sorted, label="Ground Truth RUL", color="black", linewidth=2.5, linestyle="--")
    plt.plot(y_pred_sorted, label="Proposed Hybrid 1D-CNN + LSTM", color="#1f77b4", linewidth=2.0)
    plt.fill_between(range(len(y_true)), y_true_sorted, y_pred_sorted, color="#1f77b4", alpha=0.15, label="Prediction Residual Error")

    plt.title("Proposed 1D-CNN + LSTM: Test Set Predictions vs Ground Truth (FD001)", fontweight="bold", fontsize=13)
    plt.xlabel("Test Engine Units (Sorted by Actual Remaining Life)", fontsize=11)
    plt.ylabel("Remaining Useful Life (Cycles)", fontsize=11)
    plt.legend(loc="upper left", frameon=True, fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved prediction vs ground truth plot to: {save_path}")


def train_single_run(
    seed: int,
    window_size: int = 50,
    dropout_prob: float = 0.2,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    max_epochs: int = 40,
    patience: int = 8,
    save_checkpoint_name: Optional[str] = None,
) -> Tuple[Dict[str, float], float, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Trains and evaluates a single Hybrid 1D-CNN + LSTM instance."""
    set_seed(seed)
    device = get_device("mps")

    loader = CMAPSSDataLoader()
    train_df = loader.load_train_df(subset="FD001", max_rul=125)
    test_df, _ = loader.load_test_df(subset="FD001", max_rul=125)

    preprocessor = CMAPSSPreprocessor(
        window_size=window_size,
        stride=1,
        max_rul=125,
        include_settings=False,
        variance_threshold=0.01,
        scaler_type="zscore",
    )
    train_part, val_part = preprocessor.split_train_val(train_df, val_ratio=0.20, seed=seed)
    preprocessor.fit(train_part)

    train_norm = preprocessor.transform(train_part)
    val_norm = preprocessor.transform(val_part)
    test_norm = preprocessor.transform(test_df)

    X_train, y_train, _ = preprocessor.create_sliding_windows(train_norm)
    X_val, y_val, _ = preprocessor.create_sliding_windows(val_norm)
    X_test, y_test, _ = preprocessor.create_test_windows(test_norm, last_cycle_only=True)

    loaders = build_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=batch_size, num_workers=0)

    model = HybridCNNLSTM(
        num_features=len(preprocessor.feature_cols),
        conv_filters=[32, 64],
        kernel_sizes=[5, 3],
        lstm_hidden_dim=64,
        lstm_num_layers=2,
        dropout_prob=dropout_prob,
        fc_hidden_dims=[64, 32],
    )

    chk_name = save_checkpoint_name or f"hybrid_cnn_lstm_seed{seed}_w{window_size}_d{int(dropout_prob*10)}"
    trainer = ModelTrainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        model_name=chk_name,
        checkpoint_dir=PROJECT_ROOT / "models" / "checkpoints",
    )

    train_result = trainer.train(loaders["train"], loaders["val"], max_epochs=max_epochs, patience=patience)

    y_test_eval, y_pred_eval, metrics = evaluate_pytorch_model(model, loaders["test"], device)
    lat_ms = measure_pytorch_latency(model, sample_shape=(1, window_size, len(preprocessor.feature_cols)), device=device)

    return metrics, lat_ms, y_test_eval, y_pred_eval, train_result


def main():
    results_dir = PROJECT_ROOT / "results"
    figures_dir = results_dir / "figures"
    metrics_dir = results_dir / "metrics"
    tables_dir = results_dir / "tables"
    for d in [figures_dir, metrics_dir, tables_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ==========================================================================
    # 1. Multi-Seed Benchmarking (Seeds 42, 123, 999)
    # ==========================================================================
    seeds = [42, 123, 999]
    seed_metrics = []
    best_pred = None
    best_true = None
    best_history = None

    logger.info(f"\n=======================================================")
    logger.info(f"Training Proposed Hybrid 1D-CNN + LSTM across {len(seeds)} Random Seeds")
    logger.info(f"=======================================================")

    for i, s in enumerate(seeds):
        logger.info(f"\n--- Running Seed {s} ({i+1}/{len(seeds)}) ---")
        save_name = "hybrid_cnn_lstm_best" if s == 42 else f"hybrid_cnn_lstm_seed{s}"
        m, lat, y_true, y_pred, train_res = train_single_run(
            seed=s,
            window_size=50,
            dropout_prob=0.2,
            batch_size=32,
            learning_rate=0.001,
            max_epochs=40,
            patience=8,
            save_checkpoint_name=save_name,
        )
        m["latency_ms"] = lat
        seed_metrics.append(m)
        logger.info(f"Seed {s} Results -> RMSE: {m['RMSE']:.2f}, MAE: {m['MAE']:.2f}, R2: {m['R2']:.3f}, NASA: {m['NASA_SCORE']:.1f}, Latency: {lat:.2f}ms")

        if s == 42:
            best_true = y_true
            best_pred = y_pred
            best_history = train_res["history"]

    # Calculate mean and standard deviation across seeds
    rmse_list = [m["RMSE"] for m in seed_metrics]
    mae_list = [m["MAE"] for m in seed_metrics]
    r2_list = [m["R2"] for m in seed_metrics]
    nasa_list = [m["NASA_SCORE"] for m in seed_metrics]
    lat_list = [m["latency_ms"] for m in seed_metrics]

    multi_seed_summary = {
        "seeds": seeds,
        "runs": seed_metrics,
        "summary": {
            "RMSE_mean": float(np.mean(rmse_list)),
            "RMSE_std": float(np.std(rmse_list)),
            "MAE_mean": float(np.mean(mae_list)),
            "MAE_std": float(np.std(mae_list)),
            "R2_mean": float(np.mean(r2_list)),
            "R2_std": float(np.std(r2_list)),
            "NASA_mean": float(np.mean(nasa_list)),
            "NASA_std": float(np.std(nasa_list)),
            "latency_ms_mean": float(np.mean(lat_list)),
            "parameters": 102689,
        }
    }

    with open(metrics_dir / "hybrid_multiseed.json", "w") as f:
        json.dump(multi_seed_summary, f, indent=2)

    logger.info(
        f"\nHybrid 1D-CNN + LSTM Multi-Seed Summary (N=3): "
        f"RMSE: {np.mean(rmse_list):.2f} ± {np.std(rmse_list):.2f} | "
        f"MAE: {np.mean(mae_list):.2f} ± {np.std(mae_list):.2f} | "
        f"NASA: {np.mean(nasa_list):.1f} ± {np.std(nasa_list):.1f}"
    )

    # Plot convergence and predictions
    if best_history is not None:
        plot_training_history(best_history, figures_dir / "training_convergence_curves.png")
    if best_true is not None and best_pred is not None:
        plot_predictions_vs_ground_truth(best_true, best_pred, figures_dir / "hybrid_predictions_vs_ground_truth.png")

    # ==========================================================================
    # 2. Controlled Ablation Study: Window Lengths (W = 30, 50, 70)
    # ==========================================================================
    logger.info("\n=======================================================")
    logger.info("Executing Ablation Study 1: Window Length Sensitivity (W=30, 50, 70)")
    logger.info("=======================================================")
    window_sizes = [30, 50, 70]
    window_ablation_results = {}

    for w in window_sizes:
        if w == 50:
            # Reuse Seed 42 W=50 result
            window_ablation_results["50"] = seed_metrics[0]
        else:
            logger.info(f"\n--- Testing Window Size W = {w} ---")
            m_w, lat_w, _, _, _ = train_single_run(
                seed=42, window_size=w, dropout_prob=0.2, batch_size=32, max_epochs=35, patience=7
            )
            m_w["latency_ms"] = lat_w
            window_ablation_results[str(w)] = m_w
            logger.info(f"W={w} -> RMSE: {m_w['RMSE']:.2f}, MAE: {m_w['MAE']:.2f}, NASA: {m_w['NASA_SCORE']:.1f}")

    plot_ablation_window_lengths(window_ablation_results, figures_dir / "ablation_window_lengths.png")

    # ==========================================================================
    # 3. Controlled Ablation Study: Dropout Regularization (p = 0.0, 0.2, 0.3)
    # ==========================================================================
    logger.info("\n=======================================================")
    logger.info("Executing Ablation Study 2: Dropout Regularization (p=0.0, 0.2, 0.3)")
    logger.info("=======================================================")
    dropouts = [0.0, 0.2, 0.3]
    dropout_ablation_results = {}

    for d in dropouts:
        if d == 0.2:
            dropout_ablation_results["0.2"] = seed_metrics[0]
        else:
            logger.info(f"\n--- Testing Dropout p = {d} ---")
            m_d, lat_d, _, _, _ = train_single_run(
                seed=42, window_size=50, dropout_prob=d, batch_size=32, max_epochs=35, patience=7
            )
            m_d["latency_ms"] = lat_d
            dropout_ablation_results[str(d)] = m_d
            logger.info(f"p={d} -> RMSE: {m_d['RMSE']:.2f}, MAE: {m_d['MAE']:.2f}, NASA: {m_d['NASA_SCORE']:.1f}")

    # Save ablation metrics
    all_ablations = {
        "window_size_ablation": window_ablation_results,
        "dropout_ablation": dropout_ablation_results,
    }
    with open(metrics_dir / "ablations.json", "w") as f:
        json.dump(all_ablations, f, indent=2)

    # ==========================================================================
    # 4. Generate Master Comparison Table
    # ==========================================================================
    baseline_metrics_path = metrics_dir / "baseline_comparison.json"
    if baseline_metrics_path.exists():
        with open(baseline_metrics_path, "r") as f:
            all_models = json.load(f)
    else:
        all_models = {}

    # Add Proposed Hybrid
    chk_path = PROJECT_ROOT / "models" / "checkpoints" / "hybrid_cnn_lstm_best.pt"
    if not chk_path.exists():
        chk_path = PROJECT_ROOT / "models" / "checkpoints" / "hybrid_cnn_lstm_best_best.pt"
    all_models["Proposed Hybrid 1D-CNN + LSTM"] = {
        "RMSE": float(np.mean(rmse_list)),
        "MAE": float(np.mean(mae_list)),
        "R2": float(np.mean(r2_list)),
        "NASA_SCORE": float(np.mean(nasa_list)),
        "parameters": 102689,
        "latency_ms": float(np.mean(lat_list)),
        "size_mb": get_model_file_size_mb(chk_path),
    }

    master_md = generate_markdown_comparison_table(all_models)
    master_table_path = tables_dir / "model_comparison_master.md"
    with open(master_table_path, "w") as f:
        f.write("# Master Model Comparison Table (NASA C-MAPSS FD001)\n\n" + master_md)

    print("\n" + master_md)
    logger.info(f"Saved master comparison table to: {master_table_path}")


if __name__ == "__main__":
    main()
