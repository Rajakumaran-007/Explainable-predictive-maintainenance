"""
Data Loading and Structural Validation Module for NASA C-MAPSS Turbofan Engine Dataset.

References:
    A. Saxena, K. Goebel, D. Simon, and N. Eklund, 'Damage Propagation Modeling for
    Aircraft Engine Run-to-Failure Simulation', in Proc. IEEE Int. Conf. PHM, 2008.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.utils import setup_logger

logger = setup_logger("data_loader")

# Column definitions according to official NASA C-MAPSS documentation
IDENTIFIER_COLS: List[str] = ["engine_id", "cycle"]
SETTING_COLS: List[str] = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS: List[str] = [f"s_{i}" for i in range(1, 22)]
ALL_RAW_COLS: List[str] = IDENTIFIER_COLS + SETTING_COLS + SENSOR_COLS


class CMAPSSDataLoader:
    """Robust data loader for the NASA C-MAPSS Turbofan Degradation datasets (FD001 - FD004)."""

    def __init__(self, data_dir: Union[str, Path] = "data/raw/CMAPSSData"):
        """Initializes loader with path to C-MAPSS raw data directory.

        Args:
            data_dir: Path to directory containing train_FD00*.txt, test_FD00*.txt, and RUL_FD00*.txt.
        """
        self.data_dir = Path(data_dir)
        self.valid_subsets = ["FD001", "FD002", "FD003", "FD004"]

    def _validate_subset(self, subset: str) -> None:
        if subset not in self.valid_subsets:
            raise ValueError(f"Invalid subset '{subset}'. Must be one of {self.valid_subsets}")

    def load_train_df(self, subset: str = "FD001", max_rul: Optional[int] = 125) -> pd.DataFrame:
        """Loads and structures the training run-to-failure trajectories.

        Computes the true Remaining Useful Life (RUL) for every observed cycle.
        In training data, engines run until total failure.
        For an engine with maximum observed cycle T_max:
            Linear RUL at cycle t: RUL(t) = T_max - t
            Capped (Piecewise-Linear) RUL: RUL_piecewise(t) = min(T_max - t, max_rul)

        Args:
            subset: One of 'FD001', 'FD002', 'FD003', 'FD004'.
            max_rul: Upper bound for piecewise-linear RUL label clipping. Set None for raw linear RUL.

        Returns:
            pd.DataFrame with columns: engine_id, cycle, settings, sensors, raw_rul, and target_rul.
        """
        self._validate_subset(subset)
        filepath = self.data_dir / f"train_{subset}.txt"
        if not filepath.exists():
            raise FileNotFoundError(
                f"Training file not found: {filepath.resolve()}.\n"
                f"Please run 'python scripts/prepare_data.py' to download the C-MAPSS dataset."
            )

        logger.info(f"Loading training data from {filepath.name}...")
        df = pd.read_csv(filepath, sep=r"\s+", header=None)
        
        # Remove trailing empty columns if whitespace separated
        if df.shape[1] > len(ALL_RAW_COLS):
            df = df.iloc[:, :len(ALL_RAW_COLS)]
        df.columns = ALL_RAW_COLS

        # Structural data integrity validations
        self._check_data_integrity(df, is_train=True)

        # Compute Ground Truth RUL for training trajectories
        # Group by engine_id to obtain lifetime (T_max)
        max_cycles = df.groupby("engine_id")["cycle"].transform("max")
        df["raw_rul"] = max_cycles - df["cycle"]

        if max_rul is not None and max_rul > 0:
            df["target_rul"] = df["raw_rul"].clip(upper=max_rul)
            logger.debug(f"Target RUL clipped with piecewise ceiling max_rul = {max_rul}")
        else:
            df["target_rul"] = df["raw_rul"]

        logger.info(
            f"Successfully loaded {subset} training set: {df['engine_id'].nunique()} engines, "
            f"{len(df)} total operational cycle records."
        )
        return df

    def load_test_df(self, subset: str = "FD001", max_rul: Optional[int] = 125) -> Tuple[pd.DataFrame, pd.Series]:
        """Loads test trajectories and corresponding ground-truth end-of-trajectory RUL values.

        In test data, trajectories are truncated at an arbitrary operational cycle before failure.
        The ground truth remaining life following the last observed cycle is in RUL_{subset}.txt.

        Args:
            subset: One of 'FD001', 'FD002', 'FD003', 'FD004'.
            max_rul: Upper bound for piecewise-linear target clipping.

        Returns:
            Tuple of:
                - test_df: DataFrame of test sensor trajectories with computed per-cycle RUL
                - last_cycle_rul: Series indexed by engine_id with true ground-truth RUL at trajectory end
        """
        self._validate_subset(subset)
        test_file = self.data_dir / f"test_{subset}.txt"
        rul_file = self.data_dir / f"RUL_{subset}.txt"

        if not test_file.exists() or not rul_file.exists():
            raise FileNotFoundError(
                f"Test files ({test_file.name} or {rul_file.name}) missing in {self.data_dir}."
            )

        logger.info(f"Loading test trajectories from {test_file.name} and RUL from {rul_file.name}...")
        df = pd.read_csv(test_file, sep=r"\s+", header=None)
        if df.shape[1] > len(ALL_RAW_COLS):
            df = df.iloc[:, :len(ALL_RAW_COLS)]
        df.columns = ALL_RAW_COLS

        self._check_data_integrity(df, is_train=False)

        # Load end-of-trajectory ground truth RUL
        rul_truth = pd.read_csv(rul_file, sep=r"\s+", header=None).squeeze("columns")
        # RUL file has one integer per test engine unit in sequential order (1, 2, ..., N)
        engine_ids = sorted(df["engine_id"].unique())
        if len(rul_truth) != len(engine_ids):
            raise ValueError(
                f"Mismatch in test engines: RUL file has {len(rul_truth)} values, "
                f"but test dataset contains {len(engine_ids)} engine units."
            )

        rul_map = dict(zip(engine_ids, rul_truth))
        last_cycle_rul = pd.Series(rul_map, name="ground_truth_rul")

        # Calculate exact per-cycle RUL for test data:
        # For an engine with last observed cycle T_end and ground truth remaining life R_end:
        # At cycle t <= T_end, the true RUL is: RUL(t) = R_end + (T_end - t)
        max_cycle_per_engine = df.groupby("engine_id")["cycle"].transform("max")
        end_rul_per_record = df["engine_id"].map(rul_map)
        df["raw_rul"] = end_rul_per_record + (max_cycle_per_engine - df["cycle"])

        if max_rul is not None and max_rul > 0:
            df["target_rul"] = df["raw_rul"].clip(upper=max_rul)
        else:
            df["target_rul"] = df["raw_rul"]

        logger.info(
            f"Successfully loaded {subset} test set: {len(engine_ids)} engines, "
            f"{len(df)} total cycles."
        )
        return df, last_cycle_rul

    @staticmethod
    def _check_data_integrity(df: pd.DataFrame, is_train: bool = True) -> None:
        """Validates missing values, duplicates, and cycle monotonicity."""
        # 1. Null check
        null_counts = df.isnull().sum().sum()
        if null_counts > 0:
            raise ValueError(f"Data integrity violation: found {null_counts} NaN/null values.")

        # 2. Duplicate record check
        duplicates = df.duplicated(subset=["engine_id", "cycle"]).sum()
        if duplicates > 0:
            raise ValueError(f"Data integrity violation: found {duplicates} duplicate (engine_id, cycle) pairs.")

        # 3. Monotonic cycle progression check
        for engine_id, group in df.groupby("engine_id"):
            diffs = group["cycle"].diff().dropna()
            if not (diffs == 1).all():
                raise ValueError(f"Engine {engine_id} cycles are not strictly incrementing by 1.")

    @staticmethod
    def identify_constant_sensors(df: pd.DataFrame, threshold: float = 1e-5) -> List[str]:
        """Identifies invariant sensor features having near-zero standard deviation.

        Args:
            df: DataFrame containing sensor columns.
            threshold: Standard deviation cutoff below which a sensor is deemed invariant.

        Returns:
            List of column names identified as constant.
        """
        constant_cols = []
        for col in SENSOR_COLS:
            if col in df.columns and df[col].std() < threshold:
                constant_cols.append(col)
        return constant_cols
