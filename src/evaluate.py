"""
Evaluation, Benchmarking, and Latency Profiling Module.

Provides standard evaluation protocols, regression metrics (RMSE, MAE, R², NASA Score),
inference latency benchmarks on Apple Silicon, and parameter accounting.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models import count_parameters
from src.utils import compute_regression_metrics, setup_logger

logger = setup_logger("evaluate")


def evaluate_pytorch_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Evaluates a PyTorch model on a test or validation DataLoader.

    Args:
        model: PyTorch model module.
        loader: DataLoader generating test batches.
        device: Compute acceleration device (MPS or CPU).

    Returns:
        Tuple of (y_true, y_pred, metrics_dict).
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y, _ in loader:
            batch_x = batch_x.to(device)
            preds = model(batch_x)
            all_preds.append(preds.cpu().numpy().ravel())
            all_targets.append(batch_y.numpy().ravel())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    metrics = compute_regression_metrics(y_true, y_pred)
    return y_true, y_pred, metrics


def measure_pytorch_latency(
    model: nn.Module,
    sample_shape: Tuple[int, ...] = (1, 50, 14),
    device: Optional[torch.device] = None,
    n_warmup: int = 20,
    n_runs: int = 100,
) -> float:
    """Measures precise inference latency in milliseconds per sample.

    Includes warm-up cycles and device synchronization for Apple Silicon MPS.

    Args:
        model: Evaluated PyTorch model.
        sample_shape: Single-sample input tensor dimensions (1, W, D).
        device: Compute device.
        n_warmup: Warmup iterations before timing.
        n_runs: Measurement iterations.

    Returns:
        Inference latency in milliseconds (ms) per sample.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    dummy_input = torch.randn(*sample_shape, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy_input)
            if device.type == "mps" and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()

    # Timed runs
    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            t_start = time.perf_counter()
            _ = model(dummy_input)
            if device.type == "mps" and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
            t_end = time.perf_counter()
            latencies.append((t_end - t_start) * 1000.0)  # Convert to ms

    mean_latency = float(np.mean(latencies))
    return mean_latency


def evaluate_sklearn_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_runs: int = 100,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], float]:
    """Evaluates Scikit-Learn regression model and profiles inference latency.

    Args:
        model: Trained scikit-learn regressor (Ridge, RandomForest, etc.).
        X_test: Test feature matrix (N_samples, N_features).
        y_test: Ground truth targets.
        n_runs: Latency profiling iterations.

    Returns:
        Tuple of (y_true, y_pred, metrics_dict, latency_ms_per_sample).
    """
    y_pred = model.predict(X_test)
    metrics = compute_regression_metrics(y_test, y_pred)

    # Measure latency on single-sample inputs
    single_sample = X_test[0:1]
    # Warmup
    for _ in range(10):
        _ = model.predict(single_sample)

    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = model.predict(single_sample)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    mean_latency = float(np.mean(latencies))
    return y_test, y_pred, metrics, mean_latency


def get_model_file_size_mb(filepath: Union[str, Path]) -> float:
    """Computes serialized disk footprint of a model checkpoint in MB."""
    path = Path(filepath)
    if not path.is_file():
        return 0.0
    return float(path.stat().st_size / (1024.0 * 1024.0))


def generate_markdown_comparison_table(results: Dict[str, Dict[str, Any]]) -> str:
    """Formats structured dictionary of model benchmarks into a GitHub Markdown table.

    Args:
        results: Dictionary mapping model names to metrics and hardware benchmarks.

    Returns:
        GitHub Markdown string.
    """
    header = (
        "| Model Architecture | Trainable Params | RMSE (Cycles) | MAE (Cycles) | R² Score | NASA Asymmetric Score | Inference Latency (ms) | Serialized Size (MB) |\n"
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    )
    rows = []
    for model_name, res in results.items():
        params = f"{res.get('parameters', 0):,}" if res.get("parameters") is not None else "N/A"
        rmse = f"{res['RMSE']:.2f}"
        mae = f"{res['MAE']:.2f}"
        r2 = f"{res['R2']:.3f}"
        nasa = f"{res['NASA_SCORE']:.1f}"
        lat = f"{res.get('latency_ms', 0.0):.2f}"
        size = f"{res.get('size_mb', 0.0):.2f}"
        rows.append(f"| **{model_name}** | {params} | {rmse} | {mae} | {r2} | {nasa} | {lat} | {size} |")

    return header + "\n".join(rows) + "\n"
