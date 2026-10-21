"""
Explainable AI (XAI) Module for Multivariate Temporal Deep Learning Models.

Implements:
1. IntegratedGradientsExplainer: Axiomatically grounded path-integral attribution (Captum)
   satisfying Completeness and Implementation Invariance.
2. Temporal & Sensor Aggregations: Decomposes 3D attributions (Batch, Time, Sensor)
   into fleet-wide global rankings, local engine attributions, and lifetime dynamics.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients, NoiseTunnel

from src.utils import setup_logger

logger = setup_logger("explainability")

# Sensor description dictionary linking channel IDs to physical thermodynamic functions
SENSOR_PHYSICAL_NAMES: Dict[str, str] = {
    "s_2": "LPC Outlet Temp (T24)",
    "s_3": "HPC Outlet Temp (T30)",
    "s_4": "LPT Outlet Temp (T50)",
    "s_7": "HPC Outlet Total Pressure (P30)",
    "s_8": "Physical Fan Speed (Nf)",
    "s_9": "Physical Core Speed (Nc)",
    "s_11": "HPC Outlet Static Pressure (Ps30)",
    "s_12": "Fuel Flow Ratio (Wf/Ps30)",
    "s_13": "Corrected Fan Speed (Nf_corr)",
    "s_14": "Corrected Core Speed (Nc_corr)",
    "s_15": "Bypass Ratio (BPR)",
    "s_17": "Bleed Enthalpy (htBleed)",
    "s_20": "HPT Coolant Bleed (W31)",
    "s_21": "LPT Coolant Bleed (W32)",
}


class IntegratedGradientsExplainer:
    """Axiomatic feature attribution using Integrated Gradients on PyTorch temporal models.

    References:
        M. Sundararajan, A. Taly, and Q. Yan, 'Axiomatic Attribution for Deep Networks',
        ICML 2017.
    """

    def __init__(
        self,
        model: nn.Module,
        feature_names: List[str],
        device: Optional[torch.device] = None,
    ):
        """Initializes explainer with model and feature names.

        Args:
            model: Trained PyTorch prognostic model.
            feature_names: List of sensor channel names corresponding to input channels.
            device: Compute device.
        """
        self.model = model
        self.feature_names = feature_names
        self.device = device or next(model.parameters()).device
        self.model.to(self.device)
        self.model.eval()

        self.ig = IntegratedGradients(self.model)

    def attribute(
        self,
        X: Union[np.ndarray, torch.Tensor],
        baselines: Optional[Union[np.ndarray, torch.Tensor]] = None,
        n_steps: int = 50,
        internal_batch_size: int = 32,
    ) -> np.ndarray:
        """Computes path-integral attributions for input windows.

        Args:
            X: Input tensor or array of shape (N_samples, window_size, num_features).
            baselines: Reference baseline (zeros or mean healthy reference).
            n_steps: Approximation steps for Gauss-Legendre quadrature.
            internal_batch_size: Batch size for integration steps to control memory.

        Returns:
            Attribution array of shape (N_samples, window_size, num_features).
        """
        if isinstance(X, np.ndarray):
            X_tensor = torch.from_numpy(X).float().to(self.device)
        else:
            X_tensor = X.float().to(self.device)

        if baselines is None:
            # Default reference: zeros in normalized z-score space (representing mean fleet behavior)
            baselines_tensor = torch.zeros_like(X_tensor)
        elif isinstance(baselines, np.ndarray):
            baselines_tensor = torch.from_numpy(baselines).float().to(self.device)
        else:
            baselines_tensor = baselines.float().to(self.device)

        attributions_list = []
        # Process input samples in chunks to prevent memory spikes on Apple unified memory
        chunk_size = 16
        for start_idx in range(0, len(X_tensor), chunk_size):
            end_idx = min(start_idx + chunk_size, len(X_tensor))
            batch_x = X_tensor[start_idx:end_idx]
            batch_base = baselines_tensor[start_idx:end_idx]

            attr, delta = self.ig.attribute(
                batch_x,
                baselines=batch_base,
                n_steps=n_steps,
                internal_batch_size=internal_batch_size,
                return_convergence_delta=True,
            )
            attributions_list.append(attr.detach().cpu().numpy())

        attributions = np.concatenate(attributions_list, axis=0)
        logger.info(
            f"Computed Integrated Gradients for {len(attributions)} samples (shape={attributions.shape}, n_steps={n_steps})."
        )
        return attributions

    def compute_global_importance(
        self,
        attributions: np.ndarray,
    ) -> Dict[str, float]:
        """Calculates fleet-wide global sensor importance by averaging absolute attributions.

        Args:
            attributions: Array of shape (N_samples, window_size, num_features).

        Returns:
            Dictionary mapping feature names to mean absolute importance scores, sorted descending.
        """
        # Mean absolute attribution across samples and temporal sequence: (num_features,)
        mean_abs_attr = np.mean(np.abs(attributions), axis=(0, 1))
        # Normalize to percentage sum for clear interpretability
        total_sum = np.sum(mean_abs_attr)
        normalized_scores = (mean_abs_attr / total_sum) * 100.0 if total_sum > 0 else mean_abs_attr

        importance_dict = dict(zip(self.feature_names, [float(v) for v in normalized_scores]))
        # Sort descending
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))
        return sorted_importance

    def compute_local_importance(
        self,
        sample_attributions: np.ndarray,
    ) -> Dict[str, float]:
        """Calculates signed local sensor attributions for a single engine sample.

        Positive values indicate the sensor pushed predicted RUL higher (healthy direction),
        while negative values indicate the sensor pulled predicted RUL down (degradation direction).

        Args:
            sample_attributions: Array of shape (window_size, num_features).

        Returns:
            Dictionary mapping feature names to signed cumulative attribution scores.
        """
        # Sum signed attributions across temporal window: (num_features,)
        signed_sums = np.sum(sample_attributions, axis=0)
        return dict(zip(self.feature_names, [float(v) for v in signed_sums]))

    def compute_temporal_saliency(
        self,
        attributions: np.ndarray,
    ) -> np.ndarray:
        """Computes average temporal saliency profile across the window cycles.

        Shows which operational cycles (earliest cycles vs recent cycles) dominate the model's focus.

        Args:
            attributions: Array of shape (N_samples, window_size, num_features).

        Returns:
            Array of shape (window_size,) representing mean importance per cycle.
        """
        return np.mean(np.abs(attributions), axis=(0, 2))
