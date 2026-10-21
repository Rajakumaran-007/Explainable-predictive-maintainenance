"""
Uncertainty Quantification Module for Industrial Prognostics.

Implements:
1. Monte Carlo Dropout (MCDropoutEstimator): Approximate variational inference
   for epistemic predictive uncertainty estimation.
2. Inductive Split Conformal Prediction (SplitConformalCalibrator): Distribution-free,
   finite-sample coverage guarantees with locally-adaptive interval width scaling.
"""

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from src.utils import setup_logger

logger = setup_logger("uncertainty")


class MCDropoutEstimator:
    """Estimates epistemic predictive uncertainty using test-time Monte Carlo Dropout.

    References:
        Y. Gal and Z. Ghahramani, 'Dropout as a Bayesian Approximation: Representing Model
        Uncertainty in Deep Learning', ICML 2016.
    """

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        """Initializes estimator with target model.

        Args:
            model: PyTorch model with active dropout layers.
            device: Compute acceleration device.
        """
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.to(self.device)

    def predict_with_uncertainty(
        self,
        X: Union[np.ndarray, torch.Tensor],
        num_passes: int = 50,
        batch_size: int = 64,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Runs multiple stochastic forward passes with dropout active during inference.

        Args:
            X: Input tensor or array of shape (N_samples, window_size, num_features).
            num_passes: Number of stochastic Monte Carlo forward passes (M).
            batch_size: Mini-batch size for inference memory efficiency.

        Returns:
            Tuple of:
                - mean_prediction: Expected RUL estimate of shape (N_samples,).
                - std_prediction: Epistemic standard deviation of shape (N_samples,).
                - all_samples: Raw stochastic predictions of shape (num_passes, N_samples).
        """
        if isinstance(X, np.ndarray):
            X_tensor = torch.from_numpy(X).float()
        else:
            X_tensor = X.float()

        self.model.eval()
        n_samples = len(X_tensor)
        mc_samples = np.zeros((num_passes, n_samples), dtype=np.float32)

        logger.info(f"Executing {num_passes} Monte Carlo Dropout forward passes across {n_samples} samples...")

        # Process in mini-batches to optimize Apple unified memory usage
        for m in range(num_passes):
            batch_preds = []
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_x = X_tensor[start_idx:end_idx].to(self.device)

                with torch.no_grad():
                    # Check if model supports explicit force_dropout parameter
                    if hasattr(self.model, "forward") and "force_dropout" in self.model.forward.__code__.co_varnames:
                        out = self.model(batch_x, force_dropout=True)
                    else:
                        # Fallback: force model into train mode to activate standard dropout
                        self.model.train()
                        out = self.model(batch_x)
                        self.model.eval()

                    batch_preds.append(out.cpu().numpy().ravel())

            mc_samples[m, :] = np.concatenate(batch_preds)

        mean_pred = np.mean(mc_samples, axis=0)
        std_pred = np.std(mc_samples, axis=0)

        logger.info(
            f"MC Dropout complete: Mean Predicted RUL={mean_pred.mean():.2f} cycles, "
            f"Average Epistemic Std={std_pred.mean():.2f} cycles (min={std_pred.min():.2f}, max={std_pred.max():.2f})."
        )
        return mean_pred, std_pred, mc_samples


class SplitConformalCalibrator:
    """Inductive Split Conformal Prediction Calibrator.

    Provides distribution-free, finite-sample statistical coverage guarantees:
        P(y_test in [lower, upper]) >= 1 - alpha

    Supports both:
    1. Homoscedastic (Standard) Conformal Intervals: Fixed width 2 * q_hat.
    2. Locally Adaptive Conformal Intervals: Normalized by MC Dropout epistemic standard deviation,
       producing narrower intervals for confident predictions and wider intervals where model uncertainty is elevated.

    References:
        A. N. Angelopoulos and S. Bates, 'A Gentle Introduction to Conformal Prediction
        and Distribution-Free Uncertainty Quantification', arXiv:2107.07511, 2021.
    """

    def __init__(self, alpha: float = 0.10, adaptive: bool = True):
        """Initializes conformal calibrator.

        Args:
            alpha: Miscoverage tolerance level (e.g., 0.10 for nominal 90% coverage).
            adaptive: If True, scales intervals dynamically by epistemic standard deviation.
        """
        self.alpha = alpha
        self.adaptive = adaptive
        self.q_hat: Optional[float] = None
        self.is_calibrated: bool = False

    def fit(
        self,
        y_calib_true: np.ndarray,
        y_calib_pred: np.ndarray,
        y_calib_std: Optional[np.ndarray] = None,
    ) -> "SplitConformalCalibrator":
        """Calculates conformal quantile threshold from held-out calibration residuals.

        Args:
            y_calib_true: Ground truth targets of calibration set (N_calib,).
            y_calib_pred: Predicted values of calibration set (N_calib,).
            y_calib_std: Optional epistemic standard deviation (N_calib,) for adaptive scaling.

        Returns:
            Fitted instance of self.
        """
        y_true = np.asarray(y_calib_true).ravel()
        y_pred = np.asarray(y_calib_pred).ravel()
        n_calib = len(y_true)

        if n_calib == 0:
            raise ValueError("Calibration set cannot be empty.")

        abs_residuals = np.abs(y_true - y_pred)

        if self.adaptive and y_calib_std is not None:
            y_std = np.asarray(y_calib_std).ravel()
            # Normalize absolute error by predictive standard deviation (with small epsilon)
            scores = abs_residuals / (y_std + 1e-6)
        else:
            scores = abs_residuals

        # Conformal quantile: ceil((n + 1) * (1 - alpha)) / n
        quantile_level = np.ceil((n_calib + 1) * (1.0 - self.alpha)) / n_calib
        quantile_level = min(1.0, max(0.0, quantile_level))

        self.q_hat = float(np.quantile(scores, quantile_level, method="higher"))
        self.is_calibrated = True

        logger.info(
            f"Fitted Conformal Calibrator (N={n_calib}, alpha={self.alpha}, 1-alpha={1.0-self.alpha:.0%}): "
            f"Conformal Quantile Threshold (q_hat) = {self.q_hat:.4f} (Adaptive={self.adaptive})."
        )
        return self

    def predict_intervals(
        self,
        y_pred: np.ndarray,
        y_std: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generates calibrated prediction intervals for new test samples.

        Args:
            y_pred: Point predictions of shape (N_samples,).
            y_std: Epistemic standard deviation of shape (N_samples,) if adaptive=True.

        Returns:
            Tuple of (lower_bounds, upper_bounds).
        """
        if not self.is_calibrated or self.q_hat is None:
            raise RuntimeError("Calibrator must be fitted on calibration data before predicting intervals.")

        y_p = np.asarray(y_pred).ravel()

        if self.adaptive and y_std is not None:
            y_s = np.asarray(y_std).ravel()
            half_width = self.q_hat * (y_s + 1e-6)
        else:
            half_width = self.q_hat

        lower_bound = np.maximum(0.0, y_p - half_width)
        upper_bound = y_p + half_width

        return lower_bound, upper_bound

    @staticmethod
    def evaluate_coverage(
        y_true: np.ndarray,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluates empirical marginal coverage percentage and mean interval width.

        Args:
            y_true: Ground truth target array.
            lower_bound: Lower bounds of intervals.
            upper_bound: Upper bounds of intervals.

        Returns:
            Dictionary containing 'coverage_percentage' and 'mean_interval_width'.
        """
        y_t = np.asarray(y_true).ravel()
        l_b = np.asarray(lower_bound).ravel()
        u_b = np.asarray(upper_bound).ravel()

        covered = (y_t >= l_b) & (y_t <= u_b)
        coverage_pct = float(np.mean(covered) * 100.0)
        mean_width = float(np.mean(u_b - l_b))

        return {
            "empirical_coverage_pct": coverage_pct,
            "mean_interval_width": mean_width,
            "min_interval_width": float(np.min(u_b - l_b)),
            "max_interval_width": float(np.max(u_b - l_b)),
        }
