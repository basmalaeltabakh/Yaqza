"""Unified training pipeline for all models (LSTM, Transformer, XGBoost).

Handles:
- Data loading and preprocessing
- Model instantiation
- Training with early stopping
- Learning rate scheduling
- Validation during training
- MLflow logging (optional)
- Model checkpointing
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping callback to prevent overfitting.

    Stops training when validation loss doesn't improve for *patience* epochs.
    """

    def __init__(
        self,
        patience: int = 15,
        verbose: bool = True,
        delta: float = 0.0,
        checkpoint_path: Optional[Path] = None,
    ):
        """Initialize early stopping.

        Args:
            patience: Number of epochs with no improvement to wait.
            verbose: Print messages.
            delta: Minimum change to qualify as improvement.
            checkpoint_path: Path to save best model.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.checkpoint_path = checkpoint_path

        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(
        self,
        val_loss: float,
        model: nn.Module,
    ) -> bool:
        """Check if training should stop.

        Args:
            val_loss: Validation loss at current epoch.
            model: Model to checkpoint.

        Returns:
            True if training should stop.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self._save_checkpoint(model)
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self._save_checkpoint(model)
            if self.verbose:
                logger.info(f"✓ Validation loss improved to {val_loss:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"⚠ No improvement for {self.counter}/{self.patience} epochs"
                )
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info("🛑 Early stopping triggered")
                return True
        return False

    def _save_checkpoint(self, model: nn.Module) -> None:
        """Save model checkpoint."""
        if self.checkpoint_path is not None:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), self.checkpoint_path)
            if self.verbose:
                logger.info(f"💾 Checkpoint saved to {self.checkpoint_path}")

    def load_best_model(self, model: nn.Module) -> nn.Module:
        """Load best model from checkpoint."""
        if self.checkpoint_path and self.checkpoint_path.exists():
            model.load_state_dict(torch.load(self.checkpoint_path))
            logger.info(f"📂 Loaded best model from {self.checkpoint_path}")
        return model


class Trainer:
    """Generic trainer for PyTorch models (LSTM, Transformer)."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: str = "cpu",
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        early_stopping: Optional[EarlyStopping] = None,
    ):
        """Initialize trainer.

        Args:
            model: PyTorch model.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            optimizer: Optimizer.
            criterion: Loss function.
            device: "cpu" or "cuda".
            scheduler: Learning rate scheduler.
            early_stopping: Early stopping callback.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.early_stopping = early_stopping

        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    def train_epoch(self) -> float:
        """Train for one epoch.

        Returns:
            Average training loss.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for X_batch, y_batch in self.train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device).unsqueeze(1)

            # Forward pass
            y_pred = self.model(X_batch)

            # Loss
            loss = self.criterion(y_pred, y_batch)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self) -> float:
        """Validate on validation set.

        Returns:
            Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device).unsqueeze(1)

                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def fit(self, num_epochs: int, verbose_interval: int = 5) -> Dict[str, list]:
        """Train model for multiple epochs.

        Args:
            num_epochs: Number of epochs to train.
            verbose_interval: Print stats every N epochs.

        Returns:
            Dict with ``train_losses`` and ``val_losses``.
        """
        logger.info(f"🚀 Starting training for {num_epochs} epochs")

        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)

            # LR scheduling
            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            # Early stopping
            if self.early_stopping is not None:
                if self.early_stopping(val_loss, self.model):
                    logger.info(f"Stopped at epoch {epoch + 1}")
                    self.model = self.early_stopping.load_best_model(self.model)
                    break

            # Log
            if (epoch + 1) % verbose_interval == 0:
                logger.info(
                    f"Epoch {epoch + 1:3d} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f}"
                )

        logger.info("✅ Training complete")
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }

    def predict(self, test_loader: DataLoader) -> np.ndarray:
        """Generate predictions on test set.

        Args:
            test_loader: Test data loader.

        Returns:
            Predictions of shape (n_samples,).
        """
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(self.device)
                y_pred = self.model(X_batch)
                predictions.append(y_pred.cpu().numpy())

        return np.concatenate(predictions, axis=0).flatten()


def create_trainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 15,
    checkpoint_path: Optional[Path] = None,
    device: str = "cpu",
) -> Trainer:
    """Factory function to create a Trainer with standard configuration.

    Args:
        model: PyTorch model.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        learning_rate: Initial learning rate.
        weight_decay: L2 regularization.
        patience: Early stopping patience.
        checkpoint_path: Path to save checkpoints.
        device: "cpu" or "cuda".

    Returns:
        Configured Trainer.
    """
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    criterion = nn.MSELoss()

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    early_stopping = EarlyStopping(
        patience=patience,
        checkpoint_path=checkpoint_path,
    )

    return Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        early_stopping=early_stopping,
    )
