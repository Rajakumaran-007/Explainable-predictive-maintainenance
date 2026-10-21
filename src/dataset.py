"""
PyTorch Dataset and DataLoader Generators for Multivariate C-MAPSS Time-Series.

Optimized for Apple Silicon unified memory with zero inter-process overhead.
"""

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.utils import setup_logger

logger = setup_logger("dataset")


class CMAPSSDataset(Dataset):
    """PyTorch Dataset wrapping multivariate temporal windows and RUL regression targets."""

    def __init__(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        engine_ids: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        """Initializes dataset with temporal feature tensors and target vectors.

        Args:
            X: Array or tensor of shape (N_samples, window_size, num_features).
            y: Array or tensor of shape (N_samples,) containing RUL targets.
            engine_ids: Optional array of engine unit identifiers for tracking.
        """
        if isinstance(X, np.ndarray):
            self.X = torch.from_numpy(X).float()
        else:
            self.X = X.float()

        if isinstance(y, np.ndarray):
            self.y = torch.from_numpy(y).float()
        else:
            self.y = y.float()

        if engine_ids is not None:
            if isinstance(engine_ids, np.ndarray):
                self.engine_ids = torch.from_numpy(engine_ids).int()
            else:
                self.engine_ids = engine_ids.int()
        else:
            self.engine_ids = torch.zeros(len(self.X), dtype=torch.int32)

        if len(self.X) != len(self.y):
            raise ValueError(
                f"Length mismatch: X has {len(self.X)} windows, but y has {len(self.y)} targets."
            )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (window_features, target_rul, engine_id)."""
        return self.X[idx], self.y[idx], self.engine_ids[idx]


def build_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Dict[str, DataLoader]:
    """Constructs optimized PyTorch DataLoaders for training, validation, and testing.

    Args:
        X_train: Training window tensor (N_train, W, D).
        y_train: Training RUL targets.
        X_val: Validation window tensor (N_val, W, D).
        y_val: Validation RUL targets.
        X_test: Optional test window tensor (N_test, W, D).
        y_test: Optional test RUL targets.
        batch_size: Mini-batch size.
        num_workers: Worker subprocesses (0 recommended for macOS MPS).
        pin_memory: Whether to pin memory (False recommended for Apple unified memory).

    Returns:
        Dictionary containing 'train', 'val', and optionally 'test' DataLoaders.
    """
    train_dataset = CMAPSSDataset(X_train, y_train)
    val_dataset = CMAPSSDataset(X_val, y_val)

    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        ),
    }

    if X_test is not None and y_test is not None:
        test_dataset = CMAPSSDataset(X_test, y_test)
        loaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    logger.info(
        f"Built DataLoaders (batch_size={batch_size}): "
        f"train_batches={len(loaders['train'])}, "
        f"val_batches={len(loaders['val'])}, "
        f"test_batches={len(loaders.get('test', []))}."
    )
    return loaders
