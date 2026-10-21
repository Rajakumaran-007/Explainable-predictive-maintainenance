"""
Automated Exploratory Data Analysis (EDA) & Publication-Quality Visualization Generator.

Generates high-resolution diagnostic plots for the NASA C-MAPSS dataset:
1. Invariant vs dynamic sensor variance bar plot.
2. Spearman & Pearson correlation heatmaps with RUL.
3. Multi-engine degradation trajectory profiles over lifetime.
4. Raw linear vs piecewise-linear capped RUL distribution.
5. Operational condition clustering (FD001 vs FD002 flight envelope).
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data_loader import CMAPSSDataLoader, SENSOR_COLS, SETTING_COLS
from src.preprocessing import CMAPSSPreprocessor
from src.utils import setup_logger

logger = setup_logger("eda_generator")

# Set publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300
})

OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_sensor_variances(df: pd.DataFrame, subset: str = "FD001") -> None:
    """Generates bar chart of sensor standard deviations, highlighting invariant sensors."""
    logger.info("Generating sensor variance chart...")
    stds = df[SENSOR_COLS].std().sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#2b5c8f" if s >= 0.01 else "#d9534f" for s in stds.values]
    bars = ax.bar(stds.index, stds.values, color=colors, edgecolor="black", alpha=0.85)

    ax.axhline(0.01, color="#d9534f", linestyle="--", linewidth=1.5, label="Pruning Cutoff ($\\sigma = 0.01$)")
    ax.set_yscale("log")
    ax.set_ylabel("Standard Deviation (Log Scale)")
    ax.set_xlabel("Sensor Measurement Channels")
    ax.set_title(f"Sensor Variance Profile ({subset}) — Identification of Invariant Channels", fontweight="bold")
    ax.legend(loc="upper right")
    
    # Annotate invariant bars
    for bar, val in zip(bars, stds.values):
        if val < 0.01:
            ax.text(bar.get_x() + bar.get_width()/2., 0.005, "Drop", ha="center", va="bottom",
                    fontsize=8, color="#d9534f", fontweight="bold", rotation=90)

    plt.tight_layout()
    save_path = OUTPUT_DIR / "eda_sensor_variances.png"
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_correlation_heatmap(df: pd.DataFrame, subset: str = "FD001") -> None:
    """Generates annotated Pearson correlation heatmap between informative sensors and RUL."""
    logger.info("Generating correlation heatmap...")
    preprocessor = CMAPSSPreprocessor(include_settings=False, variance_threshold=0.01)
    features = preprocessor.select_features(df)
    
    corr_cols = features + ["target_rul"]
    corr_matrix = df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr_matrix,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient"},
        ax=ax
    )
    ax.set_title(f"Multivariate Correlation Heatmap with Target RUL ({subset})", fontweight="bold", pad=15)
    plt.tight_layout()
    save_path = OUTPUT_DIR / "eda_correlation_heatmap.png"
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_degradation_trajectories(df: pd.DataFrame, sample_engines: List[int] = [1, 2, 3, 4]) -> None:
    """Visualizes run-to-failure degradation paths for key thermodynamic indicators."""
    logger.info("Generating run-to-failure degradation trajectory plots...")
    key_sensors = [
        ("s_2", "LPC Outlet Temp (T24) [°R]"),
        ("s_3", "HPC Outlet Temp (T30) [°R]"),
        ("s_4", "LPT Outlet Temp (T50) [°R]"),
        ("s_7", "HPC Outlet Total Pressure (P30) [psia]"),
        ("s_11", "Static Pressure at HPC (Ps30) [psia]"),
        ("s_12", "Fuel Flow to Static Pressure (Wf/Ps30)"),
        ("s_15", "Bypass Ratio (BPR)"),
        ("s_17", "Bleed Enthalpy (htBleed)"),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(15, 14), sharex=False)
    axes = axes.flatten()

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, (sensor_id, sensor_name) in enumerate(key_sensors):
        ax = axes[i]
        for engine_id, color in zip(sample_engines, colors):
            sub_df = df[df["engine_id"] == engine_id]
            # Apply rolling moving average for visual clarity while keeping raw scatter faint
            ax.plot(sub_df["cycle"], sub_df[sensor_id], color=color, alpha=0.3, linewidth=1.0)
            smoothed = sub_df[sensor_id].rolling(window=10, min_periods=1).mean()
            ax.plot(sub_df["cycle"], smoothed, color=color, linewidth=2.0, label=f"Engine {engine_id}" if i == 0 else "")

        ax.set_title(f"{sensor_id}: {sensor_name}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Sensor Value")
        ax.set_xlabel("Operational Flight Cycle")
        if i == 0:
            ax.legend(loc="upper left", frameon=True)

    fig.suptitle("Run-to-Failure Degradation Trajectories Across Sample Engines (FD001)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = OUTPUT_DIR / "eda_degradation_trajectories.png"
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_rul_target_distribution(df: pd.DataFrame) -> None:
    """Visualizes raw linear vs piecewise-linear capped RUL distribution."""
    logger.info("Generating RUL target distribution comparison...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    sns.histplot(df["raw_rul"], bins=40, kde=True, ax=ax1, color="#2b5c8f", edgecolor="black")
    ax1.set_title("Raw Linear RUL Distribution", fontweight="bold")
    ax1.set_xlabel("Linear RUL (Cycles)")
    ax1.set_ylabel("Record Count")

    sns.histplot(df["target_rul"], bins=35, kde=False, ax=ax2, color="#28a745", edgecolor="black")
    ax2.axvline(125, color="#d9534f", linestyle="--", linewidth=2.0, label="Capped Ceiling (RUL_max = 125)")
    ax2.set_title("Piecewise-Linear Capped RUL (Heuristic Stabilization)", fontweight="bold")
    ax2.set_xlabel("Target RUL (Cycles)")
    ax2.set_ylabel("Record Count")
    ax2.legend()

    plt.tight_layout()
    save_path = OUTPUT_DIR / "eda_rul_distribution.png"
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_operating_conditions_comparison(loader: CMAPSSDataLoader) -> None:
    """Visualizes operational flight regime complexity in FD001 vs FD002."""
    logger.info("Generating operational condition regime comparison...")
    df_fd001 = loader.load_train_df("FD001")
    df_fd002 = loader.load_train_df("FD002")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    ax1.scatter(df_fd001["setting_1"], df_fd001["setting_2"], c="#1f77b4", alpha=0.3, s=15, edgecolors="none")
    ax1.set_title("FD001: Single Operating Condition\n(Sea Level Baseline)", fontweight="bold")
    ax1.set_xlabel("Setting 1 (Altitude Normal)")
    ax1.set_ylabel("Setting 2 (Mach Number Normal)")

    scatter = ax2.scatter(
        df_fd002["setting_1"],
        df_fd002["setting_2"],
        c=df_fd002["setting_3"],
        cmap="viridis",
        alpha=0.4,
        s=15,
        edgecolors="none"
    )
    ax2.set_title("FD002: Six Discrete Flight Envelope Regimes\n(Multi-Condition Transfer Complexity)", fontweight="bold")
    ax2.set_xlabel("Setting 1 (Altitude)")
    ax2.set_ylabel("Setting 2 (Mach Number)")
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label("Setting 3 (Throttle Resolver Angle - TRA)")

    plt.tight_layout()
    save_path = OUTPUT_DIR / "eda_flight_conditions.png"
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved: {save_path}")


def main():
    loader = CMAPSSDataLoader()
    df_fd001 = loader.load_train_df("FD001", max_rul=125)

    plot_sensor_variances(df_fd001, subset="FD001")
    plot_correlation_heatmap(df_fd001, subset="FD001")
    plot_degradation_trajectories(df_fd001, sample_engines=[1, 2, 3, 4])
    plot_rul_target_distribution(df_fd001)
    plot_operating_conditions_comparison(loader)

    logger.info("\nALL PUBLICATION-QUALITY EDA FIGURES GENERATED SUCCESSFULLY IN results/figures/!")


if __name__ == "__main__":
    main()
