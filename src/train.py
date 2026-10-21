"""
Training Engine for PyTorch Prognostic Deep Learning Models.

Features:
- Adam / AdamW optimization with gradient clipping
- ReduceLROnPlateau learning rate scheduling
- Early stopping based on validation loss / RMSE
- Checkpointing of optimal weights
- Reproducible training histories
"""

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.utils import compute_regression_metrics, save_checkpoint, setup_logger

logger = setup_logger("train")


class EarlyStopping:
    """Early stops the training if validation metric does not improve after a given patience."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def __call__(self, current_score: float) -> bool:
        if self.best_score is None:
            self.best_score = current_score
            return True  # Improved

        if self.mode == "min":
            improved = current_score < (self.best_score - self.min_delta)
        else:
            improved = current_score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = current_score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


class ModelTrainer:
    """Orchestrates model training, validation, early stopping, and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        gradient_clip: float = 1.0,
        checkpoint_dir: Union[str, Path] = "models/checkpoints",
        model_name: str = "model",
    ):
        self.model = model.to(device)
        self.device = device
        self.gradient_clip = gradient_clip
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

        self.criterion = nn.MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-5
        )

    def train_epoch(self, loader: DataLoader) -> float:
        """Runs one full training epoch over the dataset."""
        self.model.train()
        total_loss = 0.0
        n_samples = 0

        for batch_x, batch_y, _ in loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device).view(-1, 1)

            self.optimizer.zero_grad()
            preds = self.model(batch_x)
            loss = self.criterion(preds, batch_y)
            loss.backward()

            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

            self.optimizer.step()

            total_loss += loss.item() * len(batch_y)
            n_samples += len(batch_y)

        return total_loss / n_samples

    def validate_epoch(self, loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Evaluates model on validation dataset."""
        self.model.eval()
        total_loss = 0.0
        n_samples = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y, _ in loader:
                batch_x = batch_x.to(self.device)
                batch_y_device = batch_y.to(self.device).view(-1, 1)

                preds = self.model(batch_x)
                loss = self.criterion(preds, batch_y_device)

                total_loss += loss.item() * len(batch_y)
                n_samples += len(batch_y)

                all_preds.append(preds.cpu().numpy().ravel())
                all_targets.append(batch_y.numpy().ravel())

        mean_loss = total_loss / n_samples
        y_true = np.concatenate(all_targets)
        y_pred = np.concatenate(all_preds)
        metrics = compute_regression_metrics(y_true, y_pred)

        return mean_loss, metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int = 60,
        patience: int = 10,
    ) -> Dict[str, Any]:
        """Executes complete training loop with validation, early stopping, and checkpointing."""
        logger.info(f"Initiating training for {self.model_name} on device: {self.device}...")
        start_time = time.time()

        early_stopping = EarlyStopping(patience=patience, mode="min")
        history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_rmse": [],
            "val_mae": [],
            "val_nasa": [],
            "lr": [],
        }

        best_checkpoint_path = self.checkpoint_dir / f"{self.model_name}_best.pt"
        best_val_rmse = float("inf")

        for epoch in range(1, max_epochs + 1):
            t_epoch_start = time.time()
            train_loss = self.train_epoch(train_loader)
            val_loss, val_metrics = self.validate_epoch(val_loader)

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_loss)

            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_rmse"].append(val_metrics["RMSE"])
            history["val_mae"].append(val_metrics["MAE"])
            history["val_nasa"].append(val_metrics["NASA_SCORE"])
            history["lr"].append(current_lr)

            is_best = early_stopping(val_loss)
            if is_best:
                best_val_rmse = val_metrics["RMSE"]
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": val_loss,
                        "val_rmse": val_metrics["RMSE"],
                        "val_mae": val_metrics["MAE"],
                        "val_nasa": val_metrics["NASA_SCORE"],
                    },
                    best_checkpoint_path,
                )

            epoch_duration = time.time() - t_epoch_start
            if epoch % 5 == 0 or is_best or epoch == 1:
                star = " *" if is_best else ""
                logger.info(
                    f"Epoch [{epoch:02d}/{max_epochs:02d}] "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                    f"Val RMSE: {val_metrics['RMSE']:.2f} | Val NASA: {val_metrics['NASA_SCORE']:.1f} | "
                    f"Time: {epoch_duration:.1f}s{star}"
                )

            if early_stopping.early_stop:
                logger.info(f"Early stopping triggered at epoch {epoch}. Best Val Loss: {early_stopping.best_score:.4f}")
                break

        total_duration = time.time() - start_time
        logger.info(f"Training completed in {total_duration:.2f}s. Best checkpoint: {best_checkpoint_path}")

        # Restore optimal checkpoint
        checkpoint = torch.load(best_checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        return {
            "history": history,
            "training_time": total_duration,
            "best_epoch": checkpoint["epoch"],
            "best_val_rmse": best_val_rmse,
            "checkpoint_path": str(best_checkpoint_path),
        }
