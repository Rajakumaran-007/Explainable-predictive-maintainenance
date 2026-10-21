"""
Multivariate Time-Series Preprocessing, Leakage-Free Scaling, and Sliding-Window Generation.

Ensures rigorous engine-level partitioning, zero temporal data leakage, and robust
sequence edge-padding for short test trajectories on the NASA C-MAPSS benchmark.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader import CMAPSSDataLoader, IDENTIFIER_COLS, SETTING_COLS, SENSOR_COLS
from src.utils import setup_logger

logger = setup_logger("preprocessing")


class CMAPSSPreprocessor:
    """End-to-end preprocessing pipeline for C-MAPSS time-series prognostic modeling."""

    # Standard 14 prognostic sensors widely adopted in C-MAPSS benchmark literature
    # (Babu et al., 2016; Zheng et al., 2017; Li et al., 2018)
    BENCHMARK_14_SENSORS: List[str] = [
        "s_2", "s_3", "s_4", "s_7", "s_8", "s_9",
        "s_11", "s_12", "s_13", "s_14", "s_15", "s_17", "s_20", "s_21"
    ]

    def __init__(
        self,
        window_size: int = 50,
        stride: int = 1,
        max_rul: int = 125,
        drop_constant: bool = True,
        variance_threshold: float = 0.01,
        include_settings: bool = False,
        selected_features: Optional[List[str]] = None,
        scaler_type: str = "zscore",
    ):
        """Initializes preprocessor with windowing and scaling parameters.

        Args:
            window_size: Temporal sequence length (W) in operational cycles.
            stride: Step size (S) between consecutive sliding windows.
            max_rul: Capped piecewise-linear RUL ceiling in cycles.
            drop_constant: If True, automatically prunes invariant sensors below variance_threshold.
            variance_threshold: Minimum standard deviation cutoff for feature retention.
            include_settings: Whether to include operational settings (setting_1, setting_2, setting_3).
                              False for single-condition FD001/FD003, True for multi-condition FD002/FD004.
            selected_features: Optional explicit list of feature column names. Overrides auto-detection.
            scaler_type: Normalization method: 'zscore' (standardization) or 'minmax'.
        """
        self.window_size = window_size
        self.stride = stride
        self.max_rul = max_rul
        self.drop_constant = drop_constant
        self.variance_threshold = variance_threshold
        self.include_settings = include_settings
        self.selected_features = selected_features
        self.scaler_type = scaler_type.lower()

        if self.scaler_type not in ["zscore", "minmax"]:
            raise ValueError(f"Unsupported scaler_type '{scaler_type}'. Choose 'zscore' or 'minmax'.")

        self.feature_cols: List[str] = []
        self.constant_cols: List[str] = []
        self.stats_: Dict[str, Dict[str, float]] = {}
        self.is_fitted: bool = False

    def select_features(self, train_df: pd.DataFrame) -> List[str]:
        """Identifies active feature columns and removes invariant features.

        Args:
            train_df: Raw training DataFrame.

        Returns:
            List of informative feature column names.
        """
        if self.selected_features is not None:
            self.feature_cols = [c for c in self.selected_features if c in train_df.columns]
            self.constant_cols = [c for c in train_df.columns if c not in self.feature_cols and c not in IDENTIFIER_COLS + ["raw_rul", "target_rul"]]
            logger.info(f"Using explicitly specified {len(self.feature_cols)} features: {self.feature_cols}")
            return self.feature_cols

        candidate_cols = []
        if self.include_settings:
            candidate_cols.extend([c for c in SETTING_COLS if c in train_df.columns])
        candidate_cols.extend([c for c in SENSOR_COLS if c in train_df.columns])

        if self.drop_constant:
            self.constant_cols = []
            for col in candidate_cols:
                if train_df[col].std() < self.variance_threshold:
                    self.constant_cols.append(col)
            self.feature_cols = [c for c in candidate_cols if c not in self.constant_cols]
            logger.info(f"Identified {len(self.constant_cols)} invariant features (std < {self.variance_threshold}): {self.constant_cols}")
            logger.info(f"Retained {len(self.feature_cols)} informative prognostic features: {self.feature_cols}")
        else:
            self.constant_cols = []
            self.feature_cols = candidate_cols

        return self.feature_cols

    def split_train_val(
        self,
        df: pd.DataFrame,
        val_ratio: float = 0.20,
        seed: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Partitions dataset strictly at the engine unit level.

        CRITICAL LEAKAGE-FREE PROTOCOL:
        Splitting is performed by unique engine_id. Engines in the validation set
        never have historical or future cycles in the training set.

        Args:
            df: Training DataFrame containing multiple engine trajectories.
            val_ratio: Proportion of unique engines reserved for validation.
            seed: Random seed for partition reproducibility.

        Returns:
            Tuple of (train_df, val_df).
        """
        unique_engines = np.sort(df["engine_id"].unique())
        train_engines, val_engines = train_test_split(
            unique_engines,
            test_size=val_ratio,
            random_state=seed,
            shuffle=True
        )

        train_df = df[df["engine_id"].isin(train_engines)].copy()
        val_df = df[df["engine_id"].isin(val_engines)].copy()

        # Strict validation: verify complete disjointness
        assert set(train_df["engine_id"]).isdisjoint(set(val_df["engine_id"])), (
            "FATAL ERROR: Engine-level data leakage detected! Train and val share engine IDs."
        )

        logger.info(
            f"Engine-level split (val_ratio={val_ratio}): "
            f"{len(train_engines)} train engines ({len(train_df)} cycles), "
            f"{len(val_engines)} val engines ({len(val_df)} cycles)."
        )
        return train_df, val_df

    def fit(self, train_df: pd.DataFrame) -> "CMAPSSPreprocessor":
        """Computes scaling parameters strictly on the training partition.

        Args:
            train_df: Training DataFrame.

        Returns:
            Fitted instance of self.
        """
        if not self.feature_cols:
            self.select_features(train_df)

        self.stats_ = {}
        for col in self.feature_cols:
            if self.scaler_type == "zscore":
                mean_val = float(train_df[col].mean())
                std_val = float(train_df[col].std())
                if std_val < 1e-8:
                    std_val = 1.0  # Prevent zero division
                self.stats_[col] = {"mean": mean_val, "std": std_val}
            elif self.scaler_type == "minmax":
                min_val = float(train_df[col].min())
                max_val = float(train_df[col].max())
                scale = max_val - min_val
                if scale < 1e-8:
                    scale = 1.0
                self.stats_[col] = {"min": min_val, "max": max_val, "scale": scale}

        self.is_fitted = True
        logger.info(f"Fitted {self.scaler_type} normalizer on {len(self.feature_cols)} features strictly from training data.")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies fitted normalization parameters to arbitrary dataset.

        Args:
            df: DataFrame to normalize.

        Returns:
            Normalized copy of DataFrame.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted using fit() before transform() can be called.")

        df_norm = df.copy()
        for col in self.feature_cols:
            if self.scaler_type == "zscore":
                df_norm[col] = (df[col] - self.stats_[col]["mean"]) / self.stats_[col]["std"]
            elif self.scaler_type == "minmax":
                df_norm[col] = (df[col] - self.stats_[col]["min"]) / self.stats_[col]["scale"]

        return df_norm

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Fits preprocessor on train_df and returns transformed DataFrame."""
        return self.fit(train_df).transform(train_df)

    def create_sliding_windows(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generates fixed-length temporal sliding windows for deep learning models.

        Transforms 2D tabular trajectory records into 3D sequential tensors:
            X shape: (N_windows, window_size, num_features)
            y shape: (N_windows,)
            engine_ids shape: (N_windows,)

        Args:
            df: Normalized DataFrame with engine_id, cycle, feature_cols, and target_rul.

        Returns:
            Tuple of (X, y, engine_ids) as numpy arrays.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before creating sliding windows.")

        feature_matrix = df[self.feature_cols].values
        target_vector = df["target_rul"].values
        engine_vector = df["engine_id"].values

        x_list, y_list, id_list = [], [], []

        for engine_id in np.unique(engine_vector):
            mask = engine_vector == engine_id
            engine_features = feature_matrix[mask]
            engine_targets = target_vector[mask]
            traj_len = len(engine_features)

            if traj_len < self.window_size:
                # Apply edge padding if entire run-to-failure trajectory is shorter than W
                pad_len = self.window_size - traj_len
                first_row = engine_features[0:1, :]
                padding = np.repeat(first_row, pad_len, axis=0)
                padded_features = np.vstack([padding, engine_features])
                x_list.append(padded_features)
                y_list.append(engine_targets[-1])
                id_list.append(engine_id)
            else:
                for start_idx in range(0, traj_len - self.window_size + 1, self.stride):
                    end_idx = start_idx + self.window_size
                    x_list.append(engine_features[start_idx:end_idx])
                    # The regression target corresponds to the RUL at the final cycle of the window
                    y_list.append(engine_targets[end_idx - 1])
                    id_list.append(engine_id)

        X = np.asarray(x_list, dtype=np.float32)
        y = np.asarray(y_list, dtype=np.float32)
        engine_ids = np.asarray(id_list, dtype=np.int32)

        logger.debug(
            f"Generated sliding windows: X.shape={X.shape}, y.shape={y.shape}, "
            f"window_size={self.window_size}, stride={self.stride}"
        )
        return X, y, engine_ids

    def create_test_windows(
        self,
        test_df: pd.DataFrame,
        last_cycle_only: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extracts evaluation windows for test trajectories.

        For official benchmark evaluation (last_cycle_only=True):
        Extracts exactly one sequence window per test engine unit ending at the final
        observed cycle T_end. If T_end < window_size, applies edge padding (repeating the
        initial cycle) so no test engines are dropped.

        Args:
            test_df: Normalized test DataFrame.
            last_cycle_only: If True, extracts only the window at trajectory end.
                             If False, extracts all sliding windows across test sequences.

        Returns:
            Tuple of (X_test, y_test, engine_ids).
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before creating test windows.")

        if not last_cycle_only:
            return self.create_sliding_windows(test_df)

        feature_matrix = test_df[self.feature_cols].values
        target_vector = test_df["target_rul"].values
        engine_vector = test_df["engine_id"].values

        x_list, y_list, id_list = [], [], []

        for engine_id in np.unique(engine_vector):
            mask = engine_vector == engine_id
            engine_features = feature_matrix[mask]
            engine_targets = target_vector[mask]
            traj_len = len(engine_features)

            if traj_len >= self.window_size:
                # Take the final W cycles
                window = engine_features[-self.window_size :]
            else:
                # Edge padding: prepend copies of the initial observed cycle
                pad_len = self.window_size - traj_len
                first_row = engine_features[0:1, :]
                padding = np.repeat(first_row, pad_len, axis=0)
                window = np.vstack([padding, engine_features])

            x_list.append(window)
            y_list.append(engine_targets[-1])
            id_list.append(engine_id)

        X_test = np.asarray(x_list, dtype=np.float32)
        y_test = np.asarray(y_list, dtype=np.float32)
        engine_ids = np.asarray(id_list, dtype=np.int32)

        logger.info(
            f"Created test evaluation windows (last_cycle_only=True): "
            f"X_test.shape={X_test.shape}, y_test.shape={y_test.shape} for {len(engine_ids)} engines."
        )
        return X_test, y_test, engine_ids
