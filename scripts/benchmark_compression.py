"""
Model Compression & Edge Deployment Profiling Pipeline on Apple Silicon M-Series.

Compares:
1. Full Hybrid 1D-CNN + LSTM: Conv1D(32, 64) + 2-layer LSTM(64) + FC(64, 32, 1)
2. Compact Edge Hybrid: Conv1D(16, 32) + 1-layer LSTM(32) + FC(32, 1) (~82% compression)
3. Micro Edge Hybrid: Conv1D(16) + 1-layer LSTM(16) + FC(16, 1) (~95% compression)

Measures on actual MacBook Air M5:
- Exact parameter counts
- Serialized model checkpoint size (KB / MB)
- Wall-clock inference latency per sample (ms) with device synchronization
- Predictive accuracy retention (RMSE, MAE, R², NASA score)
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.data_loader import CMAPSSDataLoader
from src.dataset import build_dataloaders
from src.evaluate import evaluate_pytorch_model, get_model_file_size_mb, measure_pytorch_latency
from src.models import HybridCNNLSTM, count_parameters
from src.preprocessing import CMAPSSPreprocessor
from src.train import ModelTrainer
from src.utils import get_device, set_seed, setup_logger

logger = setup_logger("benchmark_compression")

FIG_DIR = PROJECT_ROOT / "results" / "figures"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"


def plot_compression_pareto_frontier(
    results: Dict[str, Dict[str, Any]],
    save_path: Path,
) -> None:
    """Plots Pareto frontier: Inference Latency vs Test RMSE with bubble sizes showing model parameters."""
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

    names = list(results.keys())
    latencies = [results[n]["latency_ms"] for n in names]
    rmses = [results[n]["RMSE"] for n in names]
    params = [results[n]["parameters"] for n in names]
    sizes = [p / 400.0 for p in params]  # Bubble scaling

    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]

    for name, lat, rmse, s, c in zip(names, latencies, rmses, sizes, colors):
        ax.scatter(lat, rmse, s=s, color=c, alpha=0.7, edgecolors="black", linewidth=1.5, label=name)
        ax.annotate(
            f"{name}\n(RMSE: {rmse:.2f}, Latency: {lat:.2f}ms)",
            xy=(lat, rmse),
            xytext=(10, -5),
            textcoords="offset points",
            fontweight="bold",
            fontsize=9,
        )

    ax.set_title("Edge Model Compression: Pareto Frontier (MacBook Air M5 Profiling)", fontweight="bold", fontsize=13)
    ax.set_xlabel("Inference Latency per Sample (Milliseconds) [Lower is Better]", fontweight="bold", fontsize=11)
    ax.set_ylabel("Test Root Mean Squared Error (Cycles) [Lower is Better]", fontweight="bold", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved compression Pareto frontier plot to: {save_path}")


def main():
    set_seed(42)
    device = get_device("mps")

    # 1. Load Data
    loader = CMAPSSDataLoader()
    train_df = loader.load_train_df(subset="FD001", max_rul=125)
    test_df, _ = loader.load_test_df(subset="FD001", max_rul=125)

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

    norm_train = preprocessor.transform(train_part)
    norm_val = preprocessor.transform(val_part)
    norm_test = preprocessor.transform(test_df)

    X_train, y_train, _ = preprocessor.create_sliding_windows(norm_train)
    X_val, y_val, _ = preprocessor.create_sliding_windows(norm_val)
    X_test, y_test, _ = preprocessor.create_test_windows(norm_test, last_cycle_only=True)

    loaders = build_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32)

    configs = {
        "Full Hybrid (Baseline)": {
            "conv_filters": [32, 64],
            "kernel_sizes": [5, 3],
            "lstm_hidden_dim": 64,
            "lstm_num_layers": 2,
            "fc_hidden_dims": [64, 32],
            "chk_name": "hybrid_full",
        },
        "Compact Edge Hybrid": {
            "conv_filters": [16, 32],
            "kernel_sizes": [5, 3],
            "lstm_hidden_dim": 32,
            "lstm_num_layers": 1,
            "fc_hidden_dims": [32],
            "chk_name": "hybrid_compact_edge",
        },
        "Micro Edge Hybrid": {
            "conv_filters": [16],
            "kernel_sizes": [3],
            "lstm_hidden_dim": 16,
            "lstm_num_layers": 1,
            "fc_hidden_dims": [16],
            "chk_name": "hybrid_micro_edge",
        },
    }

    compression_results: Dict[str, Dict[str, Any]] = {}

    for name, cfg in configs.items():
        logger.info(f"\n=======================================================")
        logger.info(f"Training and Profiling Architecture: {name}")
        logger.info(f"=======================================================")

        model = HybridCNNLSTM(
            num_features=len(preprocessor.feature_cols),
            conv_filters=cfg["conv_filters"],
            kernel_sizes=cfg["kernel_sizes"],
            lstm_hidden_dim=cfg["lstm_hidden_dim"],
            lstm_num_layers=cfg["lstm_num_layers"],
            dropout_prob=0.2,
            fc_hidden_dims=cfg["fc_hidden_dims"],
        ).to(device)

        n_params = count_parameters(model)
        logger.info(f"{name} -> Trainable Parameters: {n_params:,}")

        trainer = ModelTrainer(
            model=model,
            device=device,
            learning_rate=0.001,
            model_name=cfg["chk_name"],
            checkpoint_dir=PROJECT_ROOT / "models" / "checkpoints",
        )

        t_res = trainer.train(loaders["train"], loaders["val"], max_epochs=25, patience=6)
        chk_file = Path(t_res["checkpoint_path"])

        _, _, metrics = evaluate_pytorch_model(model, loaders["test"], device)
        lat = measure_pytorch_latency(
            model,
            sample_shape=(1, window_size, len(preprocessor.feature_cols)),
            device=device,
            n_warmup=20,
            n_runs=100,
        )
        size_mb = get_model_file_size_mb(chk_file)

        compression_results[name] = {
            "parameters": n_params,
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "R2": metrics["R2"],
            "NASA_SCORE": metrics["NASA_SCORE"],
            "latency_ms": lat,
            "size_mb": size_mb,
            "compression_ratio_pct": float((1.0 - (n_params / 102689)) * 100.0),
        }

        logger.info(
            f"{name} Results -> RMSE: {metrics['RMSE']:.2f}, NASA: {metrics['NASA_SCORE']:.1f}, "
            f"Latency: {lat:.2f}ms, Checkpoint: {size_mb:.2f} MB ({compression_results[name]['compression_ratio_pct']:.1f}% smaller)."
        )

    # Save metrics JSON
    with open(METRICS_DIR / "compression_metrics.json", "w") as f:
        json.dump(compression_results, f, indent=2)

    # Plot Pareto Frontier
    plot_compression_pareto_frontier(compression_results, FIG_DIR / "model_compression_pareto.png")

    # Save Markdown Table
    md_table = (
        "| Architecture Variant | Parameters | Parameter Reduction (%) | RMSE (Cycles) | MAE (Cycles) | NASA Score | Latency (ms) | Speedup Factor | Serialized Size (MB) |\n"
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    )
    base_lat = compression_results["Full Hybrid (Baseline)"]["latency_ms"]
    for name, r in compression_results.items():
        speedup = base_lat / r["latency_ms"] if r["latency_ms"] > 0 else 1.0
        md_table += (
            f"| **{name}** | {r['parameters']:,} | {r['compression_ratio_pct']:.1f}% | "
            f"{r['RMSE']:.2f} | {r['MAE']:.2f} | {r['NASA_SCORE']:.1f} | "
            f"{r['latency_ms']:.2f} ms | {speedup:.2f}x | {r['size_mb']:.2f} MB |\n"
        )

    with open(TABLES_DIR / "compression_comparison.md", "w") as f:
        f.write("# Model Compression & Edge Deployment Benchmarks (MacBook Air M5)\n\n" + md_table)

    print("\n" + md_table)
    logger.info("Saved model compression benchmarks successfully!")


if __name__ == "__main__":
    main()
