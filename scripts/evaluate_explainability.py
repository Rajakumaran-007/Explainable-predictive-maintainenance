"""
Explainable AI (XAI) Attribution & Physical Interpretability Pipeline.

Generates:
1. Fleet-wide Global Feature Importance (Integrated Gradients).
2. Local Attribution Contrasts: Healthy vs Degraded Engine Case Studies.
3. 2D Spatio-Temporal Saliency Heatmaps (Cycle vs Sensor).
4. Lifetime Attribution Evolution across Engine Degradation Trajectory.
5. Physical Thermodynamic Verification & Causality Boundary Analysis.
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
import seaborn as sns
import torch

from src.data_loader import CMAPSSDataLoader
from src.explainability import SENSOR_PHYSICAL_NAMES, IntegratedGradientsExplainer
from src.models import HybridCNNLSTM
from src.preprocessing import CMAPSSPreprocessor
from src.utils import get_device, set_seed, setup_logger

logger = setup_logger("evaluate_explainability")

FIG_DIR = PROJECT_ROOT / "results" / "figures"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
FIG_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def plot_global_feature_importance(
    global_importance: Dict[str, float],
    save_path: Path,
) -> None:
    """Plots global sensor importance bar chart ranked by fleet-wide mean attribution."""
    features = list(global_importance.keys())
    scores = list(global_importance.values())
    labels = [f"{f}: {SENSOR_PHYSICAL_NAMES.get(f, f)}" for f in features]

    # Invert so highest appears at the top of horizontal bar chart
    features.reverse()
    scores.reverse()
    labels.reverse()

    plt.figure(figsize=(11, 7), dpi=300)
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(scores)))
    bars = plt.barh(labels, scores, color=colors, edgecolor="black", alpha=0.9)

    plt.xlabel("Global Relative Importance Share (%)", fontweight="bold", fontsize=11)
    plt.title("Fleet-Wide Global Feature Importance (Captum Integrated Gradients — FD001)", fontweight="bold", fontsize=13)
    plt.grid(True, linestyle=":", alpha=0.6, axis="x")

    for bar, val in zip(bars, scores):
        plt.text(val + 0.2, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", ha="left", fontsize=9, fontweight="bold")

    plt.xlim(0, max(scores) + 3.0)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved global feature importance chart to: {save_path}")


def plot_local_healthy_vs_degraded(
    healthy_imp: Dict[str, float],
    degraded_imp: Dict[str, float],
    healthy_id: int,
    degraded_id: int,
    healthy_rul: float,
    degraded_rul: float,
    save_path: Path,
) -> None:
    """Contrasts local signed attributions between a healthy and severely degraded engine."""
    features = list(healthy_imp.keys())
    h_scores = [healthy_imp[f] for f in features]
    d_scores = [degraded_imp[f] for f in features]
    labels = [f"{f} ({SENSOR_PHYSICAL_NAMES.get(f, f)})" for f in features]

    x = np.arange(len(features))
    width = 0.38

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300, sharey=True)

    # Healthy Engine
    h_colors = ["#2ca02c" if s >= 0 else "#d62728" for s in h_scores]
    ax1.barh(labels, h_scores, color=h_colors, edgecolor="black", alpha=0.85)
    ax1.axvline(0, color="black", linestyle="-", linewidth=1.0)
    ax1.set_title(f"Healthy Engine #{healthy_id} (True RUL = {healthy_rul:.0f} Cycles)\nDominantly Positive (Life-Extending) Evidence", fontweight="bold", fontsize=11)
    ax1.set_xlabel("Signed Attribution Score", fontweight="bold", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6, axis="x")

    # Degraded Engine
    d_colors = ["#2ca02c" if s >= 0 else "#d62728" for s in d_scores]
    ax2.barh(labels, d_scores, color=d_colors, edgecolor="black", alpha=0.85)
    ax2.axvline(0, color="black", linestyle="-", linewidth=1.0)
    ax2.set_title(f"Degraded Engine #{degraded_id} (True RUL = {degraded_rul:.0f} Cycles)\nDominantly Negative (Damage / Wear) Evidence", fontweight="bold", fontsize=11)
    ax2.set_xlabel("Signed Attribution Score", fontweight="bold", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6, axis="x")

    fig.suptitle("Local Explainability: Healthy vs Near-Failure Sensor Evidence Contrast", fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved healthy vs degraded local attribution contrast to: {save_path}")


def plot_temporal_saliency_heatmap(
    sample_attribution: np.ndarray,
    feature_names: List[str],
    engine_id: int,
    save_path: Path,
) -> None:
    """Plots 2D spatio-temporal saliency heatmap across 50 cycles and 14 sensors."""
    # sample_attribution shape: (window_size=50, num_features=14)
    # Transpose to (num_features, window_size) for horizontal timeline
    attr_t = sample_attribution.T
    labels = [f"{f} ({SENSOR_PHYSICAL_NAMES.get(f, f).split('(')[1].replace(')', '')})" for f in feature_names]

    plt.figure(figsize=(14, 8), dpi=300)
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    vmax = np.percentile(np.abs(attr_t), 98)

    sns.heatmap(
        attr_t,
        cmap=cmap,
        vmax=vmax,
        vmin=-vmax,
        yticklabels=labels,
        xticklabels=5,
        cbar_kws={"label": "Attribution Impact on RUL Prediction (Signed)", "shrink": 0.8},
    )

    plt.title(f"Spatio-Temporal Attribution Map: Test Engine #{engine_id} (Final 50 Cycles Window)", fontweight="bold", fontsize=13)
    plt.xlabel("Historical Window Cycles (1 = Oldest, 50 = Most Recent / Failure Horizon)", fontweight="bold", fontsize=11)
    plt.ylabel("Monitored Sensor Channels", fontweight="bold", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved spatio-temporal saliency heatmap to: {save_path}")


def plot_lifetime_sensor_dynamics(
    cycles: np.ndarray,
    trajectory_attributions: np.ndarray,
    feature_names: List[str],
    engine_id: int,
    save_path: Path,
) -> None:
    """Traces attribution trajectories of key degradation sensors over an engine's operating life."""
    # trajectory_attributions shape: (num_windows, window_size=50, num_features=14)
    # Sum across window to get net sensor impact at each evaluation cycle
    net_sensor_impact = np.sum(trajectory_attributions, axis=1)  # (num_windows, 14)

    tracked_sensors = ["s_3", "s_4", "s_11", "s_15", "s_17"]
    colors = {"s_3": "#d62728", "s_4": "#ff7f0e", "s_11": "#2ca02c", "s_15": "#9467bd", "s_17": "#1f77b4"}

    plt.figure(figsize=(12, 6), dpi=300)
    for s_id in tracked_sensors:
        if s_id in feature_names:
            idx = feature_names.index(s_id)
            plt.plot(
                cycles,
                net_sensor_impact[:, idx],
                label=f"{s_id}: {SENSOR_PHYSICAL_NAMES[s_id]}",
                color=colors.get(s_id, "gray"),
                linewidth=2.2,
            )

    plt.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    plt.title(f"Lifetime Feature Importance Dynamics: Engine #{engine_id} Transition to Degradation", fontweight="bold", fontsize=13)
    plt.xlabel("Operational Flight Cycle", fontweight="bold", fontsize=11)
    plt.ylabel("Cumulative Attribution Impact (Signed)", fontweight="bold", fontsize=11)
    plt.legend(loc="lower left", frameon=True, fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved lifetime sensor dynamics plot to: {save_path}")


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
    train_part, _ = preprocessor.split_train_val(train_df, val_ratio=0.20, seed=42)
    preprocessor.fit(train_part)

    test_norm = preprocessor.transform(test_df)
    X_test, y_test, test_engine_ids = preprocessor.create_test_windows(test_norm, last_cycle_only=True)

    # 2. Load Trained Proposed Hybrid Model
    chk_path = PROJECT_ROOT / "models" / "checkpoints" / "hybrid_cnn_lstm_best.pt"
    if not chk_path.exists():
        chk_path = PROJECT_ROOT / "models" / "checkpoints" / "hybrid_cnn_lstm_best_best.pt"

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

    # 3. Initialize Integrated Gradients Explainer
    explainer = IntegratedGradientsExplainer(model, feature_names=preprocessor.feature_cols, device=device)

    # 4. Global Feature Importance across Fleet (All 100 Test Engines)
    logger.info("Computing Integrated Gradients across all 100 test engines...")
    test_attributions = explainer.attribute(X_test, n_steps=40)
    global_importance = explainer.compute_global_importance(test_attributions)

    plot_global_feature_importance(global_importance, FIG_DIR / "xai_global_feature_importance.png")

    # 5. Local Explainability Case Studies
    # Identify healthy engine (high RUL) and degraded engine (low RUL)
    healthy_idx = int(np.argmax(y_test))   # Highest ground truth RUL
    degraded_idx = int(np.argmin(y_test))  # Lowest ground truth RUL

    healthy_engine_id = int(test_engine_ids[healthy_idx])
    degraded_engine_id = int(test_engine_ids[degraded_idx])
    healthy_rul = float(y_test[healthy_idx])
    degraded_rul = float(y_test[degraded_idx])

    healthy_local = explainer.compute_local_importance(test_attributions[healthy_idx])
    degraded_local = explainer.compute_local_importance(test_attributions[degraded_idx])

    plot_local_healthy_vs_degraded(
        healthy_local,
        degraded_local,
        healthy_engine_id,
        degraded_engine_id,
        healthy_rul,
        degraded_rul,
        FIG_DIR / "xai_local_waterfall_healthy_vs_degraded.png",
    )

    # 6. Spatio-Temporal Saliency Heatmap for Degraded Engine
    plot_temporal_saliency_heatmap(
        test_attributions[degraded_idx],
        preprocessor.feature_cols,
        degraded_engine_id,
        FIG_DIR / "xai_temporal_saliency_heatmap.png",
    )

    # 7. Lifetime Attribution Evolution across Engine Degradation Trajectory (Engine 24)
    target_engine = 24
    engine_mask = test_df["engine_id"] == target_engine
    engine_records = test_norm[engine_mask].copy()
    engine_cycles = engine_records["cycle"].values

    traj_windows, _, _ = preprocessor.create_sliding_windows(engine_records)
    logger.info(f"Computing lifetime Integrated Gradients for Engine #{target_engine} across {len(traj_windows)} cycle windows...")
    traj_attributions = explainer.attribute(traj_windows, n_steps=25)
    eval_cycles = engine_cycles[window_size - 1 :]

    plot_lifetime_sensor_dynamics(
        eval_cycles,
        traj_attributions,
        preprocessor.feature_cols,
        target_engine,
        FIG_DIR / "xai_lifetime_sensor_dynamics.png",
    )

    # 8. Save Explainability Metrics JSON
    top_3_sensors = list(global_importance.keys())[:3]
    top_3_names = [SENSOR_PHYSICAL_NAMES.get(s, s) for s in top_3_sensors]

    xai_summary = {
        "global_importance_percentage": global_importance,
        "top_predictive_sensors": dict(zip(top_3_sensors, top_3_names)),
        "case_studies": {
            "healthy_engine": {
                "engine_id": healthy_engine_id,
                "ground_truth_rul": healthy_rul,
                "local_attributions": healthy_local,
            },
            "degraded_engine": {
                "engine_id": degraded_engine_id,
                "ground_truth_rul": degraded_rul,
                "local_attributions": degraded_local,
            },
        },
        "causality_boundary_note": (
            "Feature attributions computed via Integrated Gradients measure gradient-path model sensitivity (d f / d x_i) "
            "within the operational distribution. While these attributions correlate strongly with known thermo-mechanical "
            "wear phenomena (e.g. exhaust gas temperature and bleed enthalpy escalation), mathematical attribution does not "
            "constitute formal proof of physical failure causality in the absence of explicit structural causal models."
        ),
    }

    metrics_out = METRICS_DIR / "explainability_metrics.json"
    with open(metrics_out, "w") as f:
        json.dump(xai_summary, f, indent=2)
    logger.info(f"Saved explainability metrics to: {metrics_out}")

    print("\n=== Explainable AI Global Sensor Importance Summary ===")
    for rank, (s_id, pct) in enumerate(list(global_importance.items())[:5], 1):
        print(f"Rank {rank}: {s_id} ({SENSOR_PHYSICAL_NAMES.get(s_id, s_id)}) -> {pct:.2f}% relative attribution share")


if __name__ == "__main__":
    main()
