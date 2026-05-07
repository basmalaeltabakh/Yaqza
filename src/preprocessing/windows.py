"""
Sliding-window sequence builder for time-series RUL prediction.

Why sliding windows?
--------------------
LSTMs and Transformers expect fixed-length sequences.  Instead of
feeding the entire history of an engine (variable length), we cut it
into overlapping windows of W time-steps.  Each window maps to the RUL
at its *last* cycle — this is what the model must predict.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


# ── PyTorch Dataset ────────────────────────────────────────────────────────

class CMAPSSDataset(Dataset):
    """PyTorch Dataset wrapping sliding-window sequences.

    Args:
        X: Feature array of shape ``(N, window_size, num_features)``.
        y: Target array of shape ``(N,)``.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ── Window creation ────────────────────────────────────────────────────────

def create_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    window_size: int = 30,
    stride: int = 1,
    target_col: str = "RUL",
) -> Tuple[np.ndarray, np.ndarray]:
    """Build overlapping windows from the *training* DataFrame.

    Windows are created **per engine unit** so that no window spans
    two different machines.

    Args:
        df:           Preprocessed DataFrame with columns:
                      ``unit_id``, ``cycle``, ``*feature_cols``, ``RUL``.
        feature_cols: Feature column names.
        window_size:  Number of time-steps per window.
        stride:       Number of cycles to advance between windows.
        target_col:   Regression target column.

    Returns:
        ``(X, y)`` where

        * ``X`` — shape ``(N, window_size, num_features)``
        * ``y`` — shape ``(N,)``
    """
    all_X: List[np.ndarray] = []
    all_y: List[float] = []

    for _, group in df.groupby("unit_id", sort=True):
        values  = group[feature_cols].values  # (T, F)
        targets = group[target_col].values    # (T,)
        T = len(values)

        for start in range(0, T - window_size + 1, stride):
            end = start + window_size
            all_X.append(values[start:end])
            all_y.append(targets[end - 1])

    return (
        np.array(all_X, dtype=np.float32),
        np.array(all_y, dtype=np.float32),
    )


def create_test_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    window_size: int = 30,
) -> np.ndarray:
    """Build one test sequence per engine unit (last W cycles).

    If a unit has fewer than *window_size* cycles, the sequence is
    **left-padded with zeros** so all inputs have the same shape.

    Args:
        df:           Preprocessed test DataFrame.
        feature_cols: Feature column names.
        window_size:  Sequence length.

    Returns:
        ``X`` — shape ``(num_units, window_size, num_features)``
    """
    sequences: List[np.ndarray] = []
    F = len(feature_cols)

    for _, group in df.groupby("unit_id", sort=True):
        values = group[feature_cols].values  # (T, F)
        T = len(values)

        if T >= window_size:
            sequences.append(values[-window_size:])
        else:
            # Pad on the left with zeros
            pad = np.zeros((window_size - T, F), dtype=np.float32)
            sequences.append(np.vstack([pad, values]))

    return np.array(sequences, dtype=np.float32)


# ── DataLoader builders ────────────────────────────────────────────────────

def make_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 256,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Wrap arrays in :class:`CMAPSSDataset` and return DataLoaders.

    Args:
        X_train, y_train: Training arrays.
        X_val, y_val:     Validation arrays.
        batch_size:       Mini-batch size.
        num_workers:      DataLoader worker processes.

    Returns:
        ``(train_loader, val_loader)``
    """
    train_ds = CMAPSSDataset(X_train, y_train)
    val_ds   = CMAPSSDataset(X_val,   y_val)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader


# ── Time-series aware split ────────────────────────────────────────────────

def time_series_split(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split sequences into train / validation preserving temporal order.

    Unlike random splits, this keeps the temporal structure intact:
    earlier windows go to *train*, later windows go to *validation*.
    This mimics real deployment where the model is tested on future data.

    Args:
        X:         Feature array ``(N, W, F)``.
        y:         Target array ``(N,)``.
        val_ratio: Fraction of samples assigned to validation.

    Returns:
        ``(X_train, y_train, X_val, y_val)``
    """
    n = len(y)
    split = int(n * (1 - val_ratio))
    return X[:split], y[:split], X[split:], y[split:]


# ── Multi-step targets (for seq2seq) ───────────────────────────────────────

def create_sequences_multistep(
    df: pd.DataFrame,
    feature_cols: List[str],
    window_size: int = 30,
    horizon: int = 5,
    stride: int = 1,
    target_col: str = "RUL",
) -> Tuple[np.ndarray, np.ndarray]:
    """Build sequences with multi-step targets (sequence-to-sequence).

    **Use case**: Predict RUL for the next *horizon* steps ahead.
    Useful for probabilistic RUL estimation and multi-output forecasting.

    Args:
        df:           Preprocessed DataFrame.
        feature_cols: Feature column names.
        window_size:  Input sequence length.
        horizon:      Number of future RUL values to predict.
        stride:       Stride between windows.
        target_col:   Target column name.

    Returns:
        ``(X, y)`` where ``y`` has shape ``(N, horizon)``
    """
    all_X: List[np.ndarray] = []
    all_y: List[np.ndarray] = []

    for _, group in df.groupby("unit_id", sort=True):
        values  = group[feature_cols].values
        targets = group[target_col].values
        T = len(values)

        for start in range(0, T - window_size - horizon + 1, stride):
            all_X.append(values[start : start + window_size])
            all_y.append(targets[start + window_size : start + window_size + horizon])

    return (
        np.array(all_X, dtype=np.float32),
        np.array(all_y, dtype=np.float32),
    )
