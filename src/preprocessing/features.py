"""
Feature engineering pipeline for the CMAPSS turbofan dataset.

Steps
-----
1. Load raw CMAPSS text files (train / test / RUL ground-truth).
2. Compute piece-wise linear RUL labels.
3. Drop zero-variance sensors (confirmed by EDA).
4. Add rolling statistics (mean + std) as extra features.
5. Normalise with StandardScaler (fit on train only — no leakage).
6. Generate time-series aware train/val/test splits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ── Column definitions ─────────────────────────────────────────────────────
_CMAPSS_COLS: List[str] = (
    ["unit_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Sensors that are constant across all operating cycles (no predictive power)
ZERO_VAR_SENSORS: List[str] = [
    "sensor_1", "sensor_5", "sensor_6",
    "sensor_10", "sensor_16", "sensor_18", "sensor_19",
]


# ── 1. Data loading ────────────────────────────────────────────────────────

def load_cmapss(
    data_dir: Path,
    subset: str = "FD001",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load raw CMAPSS train / test files for one subset.

    The CMAPSS dataset ships as whitespace-delimited text files with no
    header.  There are four subsets (FD001–FD004) that differ in the
    number of operating conditions and fault modes.

    Args:
        data_dir: Path to the ``data/CMAPSS/`` directory.
        subset:   One of ``"FD001"`` | ``"FD002"`` | ``"FD003"`` | ``"FD004"``.

    Returns:
        ``(train_df, test_df, rul_series)`` where *rul_series* holds the
        true RUL for every test unit (indexed 0 … N-1).
    """
    sid = subset[-3:]  # "001", "002", …

    def _read(path: Path) -> pd.DataFrame:
        df = pd.read_csv(
            path, sep=r"\s+", header=None,
            names=_CMAPSS_COLS, engine="python",
        )
        # setting_3 is always constant — drop it immediately
        df.drop(columns=["setting_3"], inplace=True, errors="ignore")
        return df
    train_df = _read(data_dir / f"train_{subset}.txt")
    test_df  = _read(data_dir / f"test_{subset}.txt")
    rul_df   = pd.read_csv(
        data_dir / f"RUL_{subset}.txt",
        header=None, names=["RUL"],
    )
    rul_series = rul_df["RUL"]

    return train_df, test_df, rul_series


# ── 2. RUL computation ─────────────────────────────────────────────────────

def compute_rul(df: pd.DataFrame, max_rul: int = 125) -> pd.DataFrame:
    """Add a piece-wise linear RUL column to the *training* DataFrame.

    Why piece-wise linear?
    - Brand-new engines have very large RUL values.
    - A model trying to predict "350 cycles" learns nothing useful.
    - Capping at *max_rul* flattens the early "healthy" phase and lets
      the model focus on the degradation region that matters.

    Args:
        df:      Training DataFrame with ``unit_id`` and ``cycle`` columns.
        max_rul: Cap value (default 125 — standard for CMAPSS FD001).

    Returns:
        DataFrame with a new ``"RUL"`` column.
    """
    max_cycle = df.groupby("unit_id")["cycle"].max().rename("max_cycle")
    df = df.join(max_cycle, on="unit_id")
    df["RUL"] = (df["max_cycle"] - df["cycle"]).clip(upper=max_rul)
    df.drop(columns=["max_cycle"], inplace=True)
    return df


# ── 3. Drop zero-variance sensors ─────────────────────────────────────────

def drop_zero_variance(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Drop sensors that carry no discriminative signal.

    Args:
        df:      Input DataFrame.
        columns: Columns to drop.  Defaults to :data:`ZERO_VAR_SENSORS`.

    Returns:
        DataFrame without the specified columns.
    """
    to_drop = [c for c in (columns or ZERO_VAR_SENSORS) if c in df.columns]
    return df.drop(columns=to_drop)


# ── 4. Rolling statistics ──────────────────────────────────────────────────

def add_rolling_features(
    df: pd.DataFrame,
    sensor_cols: List[str],
    windows: List[int] = (5, 10),
) -> pd.DataFrame:
    """Append rolling mean and std for every sensor.

    Rolling statistics give the model a view of local trend and
    volatility without requiring it to learn those patterns from
    scratch.  They are computed *per engine unit* to avoid mixing
    signals across different machines.

    Args:
        df:          DataFrame sorted by ``[unit_id, cycle]``.
        sensor_cols: Base sensor column names.
        windows:     Rolling window sizes (in cycles).

    Returns:
        DataFrame with additional columns named
        ``{sensor}_roll{w}_mean`` and ``{sensor}_roll{w}_std``.
    """
    result = df.copy()
    grouped = df.groupby("unit_id", sort=False)

    for uid, group in grouped:
        idx = group.index
        for w in windows:
            for col in sensor_cols:
                r = group[col].rolling(window=w, min_periods=1)
                result.loc[idx, f"{col}_roll{w}_mean"] = r.mean().values
                result.loc[idx, f"{col}_roll{w}_std"]  = (
                    r.std(ddof=0).fillna(0.0).values
                )
    return result


# ── 5. Normalisation ───────────────────────────────────────────────────────

def normalize(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    scaler_type: str = "standard",
    scaler_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, object]:
    """Fit a scaler on *train* data and transform both splits.

    **Data-leakage rule**: the scaler is fitted **only** on training
    samples.  Applying it to the test set is a transform-only operation.

    Args:
        train_df:     Training DataFrame.
        test_df:      Test DataFrame.
        feature_cols: Columns to normalise.
        scaler_type:  ``"standard"`` (z-score) or ``"minmax"`` ([0, 1]).
        scaler_path:  Optional ``.joblib`` path to persist the scaler.

    Returns:
        ``(normalised_train_df, normalised_test_df, fitted_scaler)``
    """
    Scaler = StandardScaler if scaler_type == "standard" else MinMaxScaler
    scaler = Scaler()

    train_df = train_df.copy()
    test_df  = test_df.copy()

    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols]  = scaler.transform(test_df[feature_cols])

    if scaler_path is not None:
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)

    return train_df, test_df, scaler


# ── 6. Feature-column helper ───────────────────────────────────────────────

def get_feature_cols(df: pd.DataFrame, sensor_cols: List[str]) -> List[str]:
    """Collect all feature columns (original sensors + rolling stats).

    Args:
        df:          DataFrame after :func:`add_rolling_features`.
        sensor_cols: Base sensor column names.

    Returns:
        Sorted list of all feature column names.
    """
    base    = set(sensor_cols)
    rolling = {
        c for c in df.columns
        if any(c.startswith(s + "_roll") for s in sensor_cols)
    }
    return sorted(base | rolling)


# ── 7. End-to-end pipeline ─────────────────────────────────────────────────

def prepare_cmapss(
    data_dir: Path,
    subset: str = "FD001",
    max_rul: int = 125,
    rolling_windows: List[int] = (5, 10),
    scaler_type: str = "standard",
    scaler_path: Optional[Path] = None,
) -> Dict:
    """Run the full preprocessing pipeline for one CMAPSS subset.

    Args:
        data_dir:        Path to ``data/CMAPSS/``.
        subset:          Dataset subset (``"FD001"`` – ``"FD004"``).
        max_rul:         RUL cap value.
        rolling_windows: Window sizes for rolling statistics.
        scaler_type:     Normalisation method.
        scaler_path:     Optional path to save the fitted scaler.

    Returns:
        Dict with keys ``train_df``, ``test_df``, ``rul_truth``,
        ``feature_cols``, ``sensor_cols``, ``scaler``.
    """
    train_df, test_df, rul_truth = load_cmapss(data_dir, subset)

    # RUL labels for training
    train_df = compute_rul(train_df, max_rul=max_rul)

    # Remove no-signal sensors
    train_df = drop_zero_variance(train_df)
    test_df  = drop_zero_variance(test_df)

    sensor_cols = [
        c for c in train_df.columns
        if c.startswith("sensor_") and c not in ZERO_VAR_SENSORS
    ]

    # Temporal features
    train_df = add_rolling_features(train_df, sensor_cols, rolling_windows)
    test_df  = add_rolling_features(test_df,  sensor_cols, rolling_windows)

    feature_cols = get_feature_cols(train_df, sensor_cols)

    # Normalise
    train_df, test_df, scaler = normalize(
        train_df, test_df, feature_cols, scaler_type, scaler_path
    )

    return {
        "train_df":    train_df,
        "test_df":     test_df,
        "rul_truth":   rul_truth,
        "feature_cols": feature_cols,
        "sensor_cols": sensor_cols,
        "scaler":      scaler,
    }


def preprocess_cmapss(
    data_dir: Path,
    subset: str = "FD001",
    max_rul: int = 125,
    rolling_windows: List[int] = None,
    scaler_type: str = "standard",
    val_fraction: float = 0.2,
    output_dir: Optional[Path] = None,
) -> Dict:
    """Run complete preprocessing with train/val/test split.
    
    This is the main entry point for the training pipeline.
    
    Args:
        data_dir: Path to data/CMAPSS/ directory
        subset: Dataset subset (FD001-FD004)
        max_rul: RUL cap value
        rolling_windows: Window sizes for rolling statistics
        scaler_type: Normalization method ('standard' or 'minmax')
        val_fraction: Fraction of training data for validation
        output_dir: Optional path to save preprocessed data
        
    Returns:
        Dict with keys:
        - train_df: Training DataFrame
        - val_df: Validation DataFrame  
        - test_df: Test DataFrame with RUL column
        - feature_cols: Feature column names
        - sensor_cols: Sensor column names
        - scaler: Fitted StandardScaler/MinMaxScaler
    """
    if rolling_windows is None:
        rolling_windows = [5, 10]
    
    # Load and preprocess using existing function
    prep_result = prepare_cmapss(
        data_dir=data_dir,
        subset=subset,
        max_rul=max_rul,
        rolling_windows=rolling_windows,
        scaler_type=scaler_type,
    )
    
    train_df = prep_result["train_df"].copy()
    test_df = prep_result["test_df"].copy()
    rul_truth = prep_result["rul_truth"]
    feature_cols = prep_result["feature_cols"]
    sensor_cols = prep_result["sensor_cols"]
    scaler = prep_result["scaler"]
    
    # Add RUL values to test data (one value per unit)
    # For each unit, compute the RUL at the end of its test trajectory
    test_ruls = []
    for unit_id in sorted(test_df["unit_id"].unique()):
        idx = unit_id - 1  # 0-indexed
        if idx < len(rul_truth):
            test_ruls.append(rul_truth.iloc[idx])
        else:
            # Fallback: clip to max_rul
            test_ruls.append(max_rul)
    
    # Map RUL values to test_df
    unit_to_rul = dict(zip(sorted(test_df["unit_id"].unique()), test_ruls))
    test_df["RUL"] = test_df["unit_id"].map(unit_to_rul)
    
    # Split training data into train/val per engine (temporal causality)
    val_df_list = []
    train_df_list = []
    
    for unit_id in train_df["unit_id"].unique():
        unit_data = train_df[train_df["unit_id"] == unit_id].copy()
        n_cycles = len(unit_data)
        split_idx = int(n_cycles * (1 - val_fraction))
        
        train_df_list.append(unit_data.iloc[:split_idx])
        val_df_list.append(unit_data.iloc[split_idx:])
    
    train_df = pd.concat(train_df_list, ignore_index=True)
    val_df = pd.concat(val_df_list, ignore_index=True)
    
    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "feature_cols": feature_cols,
        "sensor_cols": sensor_cols,
        "scaler": scaler,
    }
