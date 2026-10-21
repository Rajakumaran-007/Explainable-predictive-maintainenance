"""
Core Utility Module for Explainable Predictive Maintenance Framework.
Provides hardware acceleration management (Apple Silicon MPS / CPU fallback),
reproducible seed setting, NASA C-MAPSS scoring metric, logging, and configuration loading.
"""

import os
import random
import logging
from pathlib import Path
from typing import Dict, Any, Union, Tuple, Optional

import numpy as np
import torch
import yaml


def setup_logger(name: str = "pred_maint", level: str = "INFO") -> logging.Logger:
    """Configures a standardized console logger.

    Args:
        name: Logger identifier.
        level: Logging level string ('DEBUG', 'INFO', 'WARNING', 'ERROR').

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = setup_logger()


def set_seed(seed: int = 42) -> None:
    """Sets deterministic random seeds across Python, NumPy, and PyTorch.

    Ensures research reproducibility on Apple Silicon (macOS) and Linux/Windows.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    # Apple Silicon MPS deterministic handling
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS operations in current PyTorch use manual_seed
        pass
    
    logger.info(f"Deterministic seed set to {seed}")


def get_device(preferred: str = "mps", fallback: str = "cpu") -> torch.device:
    """Resolves target compute device with Apple MPS acceleration support.

    Validates Apple Silicon Metal Performance Shaders (MPS) availability and falls
    back gracefully to CPU if MPS is unavailable or unsupported on the platform.

    Args:
        preferred: Requested device string ("mps", "cuda", or "cpu").
        fallback: Fallback device if preferred is unavailable.

    Returns:
        torch.device instance.
    """
    if preferred == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            if torch.backends.mps.is_built():
                logger.info("Apple Metal Performance Shaders (MPS) acceleration is active.")
                return torch.device("mps")
            else:
                logger.warning("PyTorch was not built with MPS enabled. Falling back to CPU.")
        else:
            logger.warning("Apple MPS is not available on this host. Falling back to CPU.")
        return torch.device(fallback)
    elif preferred == "cuda" and torch.cuda.is_available():
        logger.info(f"NVIDIA CUDA acceleration active: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        logger.info("Using standard CPU compute device.")
        return torch.device(fallback)


def load_config(config_path: Union[str, Path] = "configs/default.yaml") -> Dict[str, Any]:
    """Loads YAML experiment configuration file safely.

    Args:
        config_path: Path to YAML configuration.

    Returns:
        Dictionary containing experiment parameters.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {path.resolve()}")
    
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def compute_nasa_score(y_true: Union[np.ndarray, torch.Tensor],
                       y_pred: Union[np.ndarray, torch.Tensor]) -> float:
    """Computes official NASA C-MAPSS asymmetric evaluation score.

    The NASA asymmetric scoring function severely penalizes late predictions (where
    estimated RUL > true RUL, creating dangerous delayed maintenance risks) compared
    to early predictions (where estimated RUL < true RUL, causing premature servicing).

    Mathematical Formulation (Saxena et al., PHM 2008):
        d_i = \\hat{y}_i - y_i  (error in cycles)
        s_i = \\exp(-d_i / 13) - 1   if d_i < 0  (Early prediction)
        s_i = \\exp(d_i / 10) - 1    if d_i >= 0 (Late prediction)
        Score = \\sum_{i=1}^{N} s_i

    Args:
        y_true: Ground truth RUL array or tensor of shape (N,).
        y_pred: Predicted RUL array or tensor of shape (N,).

    Returns:
        Total cumulative NASA asymmetric penalty score.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    d = y_pred - y_true
    # For early predictions (d < 0): exp(-d / 13) - 1
    # For late predictions (d >= 0): exp(d / 10) - 1
    scores = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(scores))


def compute_regression_metrics(y_true: Union[np.ndarray, torch.Tensor],
                               y_pred: Union[np.ndarray, torch.Tensor]) -> Dict[str, float]:
    """Computes comprehensive regression metrics for RUL prediction.

    Calculates:
        - Root Mean Squared Error (RMSE)
        - Mean Absolute Error (MAE)
        - Coefficient of Determination (R²)
        - NASA Asymmetric Score

    Args:
        y_true: Array of true values.
        y_pred: Array of predicted values.

    Returns:
        Dictionary mapping metric names to computed floating point values.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    
    nasa_score = compute_nasa_score(y_true, y_pred)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "NASA_SCORE": nasa_score
    }


def save_checkpoint(state: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """Saves model weights, optimizer state, and training metadata to disk.

    Args:
        state: Dictionary with model_state_dict, optimizer_state_dict, epoch, etc.
        filepath: Destination path for checkpoint file.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    logger.debug(f"Saved model checkpoint to {path}")


def load_checkpoint(filepath: Union[str, Path],
                    model: torch.nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Loads model weights and training metadata from a checkpoint file.

    Args:
        filepath: Source checkpoint path.
        model: Target PyTorch model module.
        optimizer: Optional optimizer to restore state into.
        device: Target compute device.

    Returns:
        Loaded state dictionary (containing epoch, best_metric, etc.).
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {path}")

    checkpoint = torch.load(path, map_location=device or torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    logger.info(f"Successfully loaded checkpoint from {path} (epoch {checkpoint.get('epoch', 'N/A')})")
    return checkpoint
