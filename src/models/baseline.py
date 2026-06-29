"""Baseline ML models (XGBoost, LightGBM) for RUL prediction.

These serve as benchmarks to compare against deep learning approaches.
They typically require 2D input (no time dimension), so sequences are
flattened or statistical summaries are used.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


class XGBoostRUL:
    """XGBoost for RUL prediction with flattened sequences.

    **Why XGBoost?**
    - Fast training.
    - Handles non-linear patterns.
    - Good baseline to compare against neural networks.
    - Interpretable via SHAP values.

    **Preprocessing**: Flattens each sequence into 1D vector.
    If seq is (batch, 30, 14), reshape to (batch, 30*14=420).
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        """Initialize XGBoost model.

        Args:
            n_estimators: Number of boosting rounds.
            max_depth: Maximum tree depth.
            learning_rate: Learning rate (shrinkage).
            subsample: Fraction of samples for each tree.
            colsample_bytree: Fraction of features for each tree.
            random_state: Random seed.
        """
        if not HAS_XGBOOST:
            raise ImportError("xgboost not installed. Install via: pip install xgboost")

        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def _flatten(self, X: np.ndarray) -> np.ndarray:
        """Flatten 3D sequences to 2D.

        Args:
            X: Shape (batch, seq_len, n_features)

        Returns:
            Shape (batch, seq_len * n_features)
        """
        batch, seq_len, n_features = X.shape
        return X.reshape(batch, seq_len * n_features)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> XGBoostRUL:
        """Train XGBoost model.

        Args:
            X_train: Shape (batch, seq_len, n_features)
            y_train: Shape (batch,)
            X_val: Optional validation set.
            y_val: Optional validation targets.

        Returns:
            Self (for chaining).
        """
        X_train_flat = self._flatten(X_train)
        X_train_flat = self.scaler.fit_transform(X_train_flat)

        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_flat = self._flatten(X_val)
            X_val_flat = self.scaler.transform(X_val_flat)
            eval_set = [(X_val_flat, y_val)]

        self.model.fit(
            X_train_flat,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict RUL.

        Args:
            X: Shape (batch, seq_len, n_features)

        Returns:
            Shape (batch,) – RUL predictions.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X_flat = self._flatten(X)
        X_flat = self.scaler.transform(X_flat)
        return self.model.predict(X_flat)

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores.

        Returns:
            Array of importance values.
        """
        return self.model.feature_importances_


class LightGBMRUL:
    """LightGBM for RUL prediction (lighter alternative to XGBoost).

    **Advantages**: Faster on large datasets, less memory usage.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        """Initialize LightGBM model.

        Args:
            n_estimators: Number of boosting rounds.
            max_depth: Maximum tree depth.
            learning_rate: Shrinkage rate.
            num_leaves: Number of leaves in decision trees.
            subsample: Fraction of samples.
            colsample_bytree: Fraction of features.
            random_state: Random seed.
        """
        if not HAS_LIGHTGBM:
            raise ImportError("lightgbm not installed. Install via: pip install lightgbm")

        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def _flatten(self, X: np.ndarray) -> np.ndarray:
        """Flatten 3D sequences."""
        batch, seq_len, n_features = X.shape
        return X.reshape(batch, seq_len * n_features)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> LightGBMRUL:
        """Train LightGBM model."""
        X_train_flat = self._flatten(X_train)
        X_train_flat = self.scaler.fit_transform(X_train_flat)

        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_flat = self._flatten(X_val)
            X_val_flat = self.scaler.transform(X_val_flat)
            eval_set = [(X_val_flat, y_val)]

        self.model.fit(
            X_train_flat,
            y_train,
            eval_set=eval_set,
            callbacks=[lgb.early_stopping(50)],
        )
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict RUL."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X_flat = self._flatten(X)
        X_flat = self.scaler.transform(X_flat)
        return self.model.predict(X_flat)

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance."""
        return self.model.feature_importances_
