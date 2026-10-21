"""
Empirical Uncertainty Estimation and Conformal Calibration Pipeline.

Evaluates:
1. Test-time Monte Carlo Dropout (M=50 passes) on proposed Hybrid 1D-CNN + LSTM.
2. Standard vs Locally-Adaptive Inductive Split Conformal Prediction intervals.
3. Empirical coverage percentage, average interval width, and calibration fidelity.
4. Epistemic uncertainty correlation with absolute prediction error.
5. Complete engine lifetime trajectory uncertainty evolution.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import torch

from src.data_loader import CMAPSSDataLoader
from src.models import HybridCNNLSTM
from src.preprocessing import CMAPSSPreprocessor
from src.uncertainty import MCDropoutEstimator, SplitConformalCalibrator
from src.utils import compute_regression_metrics, get_device, set_seed, setup_logger

logger = setup_logger("evaluate_uncertainty")

FIG_DIR = PROJECT_ROOT / "results" / "figures"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
FIG_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def plot_test_prediction_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    coverage_pct: float,
    save_path: Path,
) -> None:
    """Plots test predictions with 90% conformal prediction intervals against ground truth."""
    sort_idx = np.argsort(y_true)
    y_true_s = y_true[sort_idx]
    y_pred_s = y_pred[sort_idx]
    l_b_s = lower_bound[sort_idx]
    u_b_s = upper_bound[sort_idx]

    plt.figure(figsize=(14, 7), dpi=300)

    # 90% Conformal Prediction Shaded Region
    plt.fill_between(
        range(len(y_true)),
        l_b_s,
        u_b_s,
        color="#2b5c8f",
        alpha=0.25,
        label=f"Calibrated 90% Conformal Interval (Coverage: {coverage_pct:.1f}%)",
    )
    plt.plot(l_b_s, color="#2b5c8f", linestyle=":", linewidth=1.0, alpha=0.7)
    plt.plot(u_b_s, color="#2b5c8f", linestyle=":", linewidth=1.0, alpha=0.7)

    # Predictions and Ground Truth
    plt.plot(y_true_s, color="black", linestyle="--", linewidth=2.5, label="Ground Truth RUL")
    plt.plot(y_pred_s, color="#d62728", linewidth=1.8, label="Hybrid 1D-CNN + LSTM (MC Mean)")

    # Highlight miscovered points
    miscovered = (y_true_s < l_b_s) | (y_true_s > u_b_s)
    if np.any(miscovered):
        plt.scatter(
            np.where(miscovered)[0],
            y_true_s[miscovered],
            color="#d62728",
            s=40,
            zorder=5,
            edgecolors="black",
            label="Miscovered Engines",
        )

    plt.title(
        f"Uncertainty Quantification: Conformal Prediction Intervals on FD001 Test Set (Nominal: 90% | Empirical: {coverage_pct:.1f}%)",
        fontweight="bold",
        fontsize=13,
    )
    plt.xlabel("Test Engine Units (Sorted by True Remaining Useful Life)", fontsize=11)
    plt.ylabel("Remaining Useful Life (Cycles)", fontsize=11)
    plt.legend(loc="upper left", frameon=True, fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved test prediction intervals plot to: {save_path}")


def plot_reliability_diagram(
    nominal_levels: List[float],
    empirical_coverages: List[float],
    save_path: Path,
) -> None:
    """Plots reliability diagram comparing nominal confidence vs actual empirical coverage."""
    plt.figure(figsize=(7, 6), dpi=300)

    nom_pct = [level * 100.0 for level in nominal_levels]
    plt.plot([70, 100], [70, 100], color="black", linestyle="--", linewidth=1.5, label="Ideal Calibration Line")
    plt.plot(nom_pct, empirical_coverages, marker="o", color="#2b5c8f", linewidth=2.5, markersize=8, label="Conformal Calibrator")

    for x, y in zip(nom_pct, empirical_coverages):
        plt.annotate(f"{y:.1f}%", xy=(x, y), xytext=(-10, 8), textcoords="offset points", fontweight="bold", fontsize=9)

    plt.title("Statistical Calibration: Nominal vs Empirical Coverage (FD001)", fontweight="bold", fontsize=12)
    plt.xlabel("Nominal Target Coverage Level (%)", fontsize=11)
    plt.ylabel("Empirical Test Set Coverage (%)", fontsize=11)
    plt.xlim(75, 100)
    plt.ylim(75, 102)
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved reliability calibration plot to: {save_path}")


def plot_error_vs_uncertainty(
    abs_errors: np.ndarray,
    epistemic_stds: np.ndarray,
    save_path: Path,
) -> Tuple[float, float]:
    """Plots scatter of absolute prediction error versus epistemic standard deviation."""
    pearson_r, p_val = stats.pearsonr(epistemic_stds, abs_errors)
    spearman_rho, _ = stats.spearmanr(epistemic_stds, abs_errors)

    plt.figure(figsize=(8, 6), dpi=300)
    plt.scatter(epistemic_stds, abs_errors, color="#1f77b4", alpha=0.7, edgecolors="black", s=45)

    # Fit linear trend line
    m, b = np.polyfit(epistemic_stds, abs_errors, 1)
    x_range = np.linspace(epistemic_stds.min(), epistemic_stds.max(), 100)
    plt.plot(x_range, m * x_range + b, color="#d62728", linestyle="--", linewidth=2.0,
             label=f"Trend Line (Pearson r = {pearson_r:.2f}, Spearman ρ = {spearman_rho:.2f})")

    plt.title("Epistemic Uncertainty Quality: Absolute Error vs Predictive Std", fontweight="bold", fontsize=12)
    plt.xlabel("MC Dropout Predictive Standard Deviation (Cycles)", fontsize=11)
    plt.ylabel("Absolute Prediction Error |y - ŷ| (Cycles)", fontsize=11)
    plt.legend(loc="upper left", frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved error vs uncertainty scatter plot to: {save_path}")
    return float(pearson_r), float(spearman_rho)


def plot_trajectory_uncertainty(
    cycles: np.ndarray,
    true_ruls: np.ndarray,
    pred_ruls: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    engine_id: int,
    save_path: Path,
) -> None:
    """Plots degradation trajectory tracking for a sample test engine across all observed cycles."""
    plt.figure(figsize=(12, 6), dpi=300)

    plt.fill_between(
        cycles,
        lower_bounds,
        upper_bounds,
        color="#2b5c8f",
        alpha=0.25,
        label="90% Conformal Prediction Interval",
    )
    plt.plot(cycles, true_ruls, color="black", linestyle="--", linewidth=2.2, label="Ground Truth RUL")
    plt.plot(cycles, pred_ruls, color="#d62728", linewidth=2.0, label="Predicted RUL (MC Mean)")

    plt.title(f"Test Engine #{engine_id}: Real-Time RUL Tracking & Uncertainty Evolution", fontweight="bold", fontsize=13)
    plt.xlabel("Operational Flight Cycle", fontsize=11)
    plt.ylabel("Remaining Useful Life (Cycles)", fontsize=11)
    plt.legend(loc="upper right", frameon=True, fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved trajectory tracking uncertainty plot to: {save_path}")


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

    val_norm = preprocessor.transform(val_part)
    test_norm = preprocessor.transform(test_df)

    # Prepare Calibration Set (from validation engines)
    X_val, y_val, _ = preprocessor.create_sliding_windows(val_norm)
    # Prepare Test Set (final cycle for all 100 test engines)
    X_test, y_test, test_engine_ids = preprocessor.create_test_windows(test_norm, last_cycle_only=True)

    # 2. Load Trained Proposed Hybrid Model
    chk_path = PROJECT_ROOT / "models" / "checkpoints" / "hybrid_cnn_lstm_best.pt"
    if not chk_path.exists():
        chk_path = PROJECT_ROOT / "models" / "checkpoints" / "hybrid_cnn_lstm_best_best.pt"
    logger.info(f"Loading trained hybrid model from: {chk_path}")

    model = HybridCNNLSTM(
        num_features=len(preprocessor.feature_cols),
        conv_filters=[32, 64],
        kernel_sizes=[5, 3],
        lstm_hidden_dim=64,
        lstm_num_layers=2,
        dropout_prob=0.2,
        fc_hidden_dims=[64, 32],
    ).to(device)

    chk = torch.load(chk_path, map_location=device)
    model.load_state_dict(chk["model_state_dict"])
    model.eval()

    # 3. Monte Carlo Dropout Epistemic Uncertainty Estimation
    mc_estimator = MCDropoutEstimator(model, device=device)

    # Sample sub-sampled calibration set for computational speed
    calib_indices = np.random.choice(len(X_val), size=min(1000, len(X_val)), replace=False)
    X_calib = X_val[calib_indices]
    y_calib = y_val[calib_indices]

    logger.info("Running Monte Carlo Dropout on calibration set...")
    val_mean, val_std, _ = mc_estimator.predict_with_uncertainty(X_calib, num_passes=30, batch_size=64)

    logger.info("Running Monte Carlo Dropout on test set (M=50 passes)...")
    test_mean, test_std, all_mc_test = mc_estimator.predict_with_uncertainty(X_test, num_passes=50, batch_size=32)

    # 4. Standard MC Dropout Gaussian Confidence Intervals (μ ± 1.96 σ)
    mc_lower_95 = np.maximum(0.0, test_mean - 1.96 * test_std)
    mc_upper_95 = test_mean + 1.96 * test_std
    mc_eval_95 = SplitConformalCalibrator.evaluate_coverage(y_test, mc_lower_95, mc_upper_95)
    logger.info(f"Uncalibrated MC Dropout (μ ± 1.96σ): Coverage = {mc_eval_95['empirical_coverage_pct']:.1f}%, Mean Width = {mc_eval_95['mean_interval_width']:.1f} cycles")

    # 5. Split Conformal Prediction Calibration
    # A. 90% Nominal Coverage Target (alpha = 0.10)
    conformal_90 = SplitConformalCalibrator(alpha=0.10, adaptive=True)
    conformal_90.fit(y_calib, val_mean, val_std)
    conf_lower_90, conf_upper_90 = conformal_90.predict_intervals(test_mean, test_std)
    conf_eval_90 = conformal_90.evaluate_coverage(y_test, conf_lower_90, conf_upper_90)

    # B. Homoscedastic (Non-Adaptive) Conformal for Comparison
    homo_conformal_90 = SplitConformalCalibrator(alpha=0.10, adaptive=False)
    homo_conformal_90.fit(y_calib, val_mean)
    homo_lower_90, homo_upper_90 = homo_conformal_90.predict_intervals(test_mean)
    homo_eval_90 = homo_conformal_90.evaluate_coverage(y_test, homo_lower_90, homo_upper_90)

    logger.info(f"Locally-Adaptive 90% Conformal: Empirical Coverage = {conf_eval_90['empirical_coverage_pct']:.1f}%, Mean Width = {conf_eval_90['mean_interval_width']:.1f} cycles")
    logger.info(f"Homoscedastic 90% Conformal:    Empirical Coverage = {homo_eval_90['empirical_coverage_pct']:.1f}%, Mean Width = {homo_eval_90['mean_interval_width']:.1f} cycles")

    # 6. Multi-Level Calibration Reliability Curve (alpha in [0.20, 0.15, 0.10, 0.05])
    nom_levels = [0.80, 0.85, 0.90, 0.95]
    emp_coverages = []
    for level in nom_levels:
        c_temp = SplitConformalCalibrator(alpha=1.0 - level, adaptive=True)
        c_temp.fit(y_calib, val_mean, val_std)
        l_t, u_t = c_temp.predict_intervals(test_mean, test_std)
        emp_coverages.append(c_temp.evaluate_coverage(y_test, l_t, u_t)["empirical_coverage_pct"])

    # 7. Generate Publication Figures
    plot_test_prediction_intervals(
        y_test,
        test_mean,
        conf_lower_90,
        conf_upper_90,
        conf_eval_90["empirical_coverage_pct"],
        FIG_DIR / "uncertainty_mc_dropout_intervals.png",
    )

    plot_reliability_diagram(
        nom_levels,
        emp_coverages,
        FIG_DIR / "uncertainty_reliability_calibration.png",
    )

    abs_errors = np.abs(y_test - test_mean)
    p_r, s_rho = plot_error_vs_uncertainty(
        abs_errors,
        test_std,
        FIG_DIR / "uncertainty_error_vs_std.png",
    )

    # 8. Trajectory Uncertainty Tracking for a Representative Engine (e.g. Engine 24)
    target_engine_id = 24
    engine_mask = test_df["engine_id"] == target_engine_id
    engine_records = test_norm[engine_mask].copy()
    engine_cycles = engine_records["cycle"].values
    engine_true_ruls = engine_records["target_rul"].values

    # Generate sliding window for each cycle in engine trajectory
    engine_windows, engine_targets, _ = preprocessor.create_sliding_windows(engine_records)
    e_mean, e_std, _ = mc_estimator.predict_with_uncertainty(engine_windows, num_passes=30, batch_size=32)
    e_lower, e_upper = conformal_90.predict_intervals(e_mean, e_std)

    # Trajectory cycles align with window endpoints
    traj_cycles = engine_cycles[window_size - 1 :]
    traj_true = engine_true_ruls[window_size - 1 :]

    plot_trajectory_uncertainty(
        traj_cycles,
        traj_true,
        e_mean,
        e_lower,
        e_upper,
        target_engine_id,
        FIG_DIR / "uncertainty_engine_trajectory.png",
    )

    # 9. Save Uncertainty Metrics JSON
    uncertainty_summary = {
        "mc_dropout": {
            "num_passes": 50,
            "mean_epistemic_std": float(np.mean(test_std)),
            "min_epistemic_std": float(np.min(test_std)),
            "max_epistemic_std": float(np.max(test_std)),
            "gaussian_95_coverage_pct": mc_eval_95["empirical_coverage_pct"],
            "gaussian_95_mean_width": mc_eval_95["mean_interval_width"],
        },
        "conformal_prediction_adaptive_90": {
            "nominal_coverage_pct": 90.0,
            "empirical_coverage_pct": conf_eval_90["empirical_coverage_pct"],
            "mean_interval_width": conf_eval_90["mean_interval_width"],
            "min_interval_width": conf_eval_90["min_interval_width"],
            "max_interval_width": conf_eval_90["max_interval_width"],
            "q_hat": conformal_90.q_hat,
        },
        "conformal_prediction_homoscedastic_90": {
            "nominal_coverage_pct": 90.0,
            "empirical_coverage_pct": homo_eval_90["empirical_coverage_pct"],
            "mean_interval_width": homo_eval_90["mean_interval_width"],
            "q_hat": homo_conformal_90.q_hat,
        },
        "error_uncertainty_correlation": {
            "pearson_r": p_r,
            "spearman_rho": s_rho,
        },
        "calibration_curve": {
            "nominal_levels": nom_levels,
            "empirical_coverages": emp_coverages,
        },
    }

    metrics_out = METRICS_DIR / "uncertainty_metrics.json"
    with open(metrics_out, "w") as f:
        json.dump(uncertainty_summary, f, indent=2)
    logger.info(f"Saved uncertainty quantification metrics to: {metrics_out}")

    print("\n=== Uncertainty Quantification Summary ===")
    print(f"MC Dropout Epistemic Std (Cycles): {np.mean(test_std):.2f} ± {np.std(test_std):.2f}")
    print(f"Uncalibrated Gaussian 95% Coverage: {mc_eval_95['empirical_coverage_pct']:.1f}% (Mean Width: {mc_eval_95['mean_interval_width']:.1f} cycles)")
    print(f"Adaptive 90% Conformal Coverage:    {conf_eval_90['empirical_coverage_pct']:.1f}% (Mean Width: {conf_eval_90['mean_interval_width']:.1f} cycles)")
    print(f"Error-Uncertainty Correlation:      Pearson r = {p_r:.3f}, Spearman ρ = {s_rho:.3f}")


if __name__ == "__main__":
    main()
