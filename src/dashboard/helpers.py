"""
Yaqza Dashboard — Data Loading, Caching & Utility Helpers.

Centralises all data I/O, model loading, and metrics computation
so the main dashboard file stays clean.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ── Path resolution ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR     = PROJECT_ROOT / "data" / "CMAPSS"
MODEL_DIR    = PROJECT_ROOT / "model_weights"
MLFLOW_DIR   = PROJECT_ROOT / "mlruns"

# Ensure project root is on sys.path so `src.*` imports work
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── Data loading (cached) ──────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_data(
    subset: str = "FD001",
    window_size: int = 30,
    stride: int = 1,
) -> Optional[Dict[str, Any]]:
    """Load CMAPSS data with preprocessing and windowing.

    Returns dict with keys:
        X_train, X_val, X_test, y_train, y_val, y_test,
        feature_cols, sensor_cols, train_df, val_df, test_df
    """
    try:
        from src.preprocessing.features import preprocess_cmapss
        from src.preprocessing.windows import create_sequences, create_test_sequences

        prep = preprocess_cmapss(
            DATA_DIR.parent,
            subset=subset,
            max_rul=125,
            rolling_windows=[5, 10],
            val_fraction=0.2,
        )

        train_df     = prep["train_df"]
        val_df       = prep["val_df"]
        test_df      = prep["test_df"]
        feature_cols = prep["feature_cols"]
        sensor_cols  = prep["sensor_cols"]

        X_train, y_train = create_sequences(
            train_df, feature_cols=feature_cols,
            window_size=window_size, stride=stride, target_col="RUL",
        )
        X_val, y_val = create_sequences(
            val_df, feature_cols=feature_cols,
            window_size=window_size, stride=stride, target_col="RUL",
        )
        X_test = create_test_sequences(
            test_df, feature_cols=feature_cols, window_size=window_size,
        )
        y_test = test_df.groupby("unit_id")["RUL"].first().values

        return dict(
            X_train=X_train, X_val=X_val, X_test=X_test,
            y_train=y_train, y_val=y_val, y_test=y_test,
            feature_cols=feature_cols, sensor_cols=sensor_cols,
            train_df=train_df, val_df=val_df, test_df=test_df,
        )
    except Exception as exc:
        st.error(f"❌ Data loading failed: {exc}")
        import traceback
        st.error(traceback.format_exc())
        return None


# ── Model loading ───────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_pytorch_model(model_path: Path, model_class: str, input_size: int) -> Optional[Any]:
    """Load a PyTorch model from a .pt checkpoint."""
    try:
        import torch
        if model_class == "lstm":
            from src.models.lstm_rul import ImprovedLSTM
            model = ImprovedLSTM(input_size=input_size, hidden_size=128, num_layers=3, dropout=0.3)
        elif model_class == "transformer":
            from src.models.tft_rul import TransformerEncoder
            model = TransformerEncoder(input_size=input_size, d_model=128, nhead=8, num_layers=3)
        else:
            return None

        state = torch.load(model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state)
        model.eval()
        return model
    except Exception:
        return None


def get_available_models() -> Dict[str, Path]:
    """Scan model_weights/ for available .pt files."""
    models: Dict[str, Path] = {}
    if MODEL_DIR.exists():
        for f in MODEL_DIR.glob("*.pt"):
            name = f.stem.replace("best_", "").replace("_FD001", "")
            models[name] = f
    return models


# ── MLflow metrics loading ──────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=300)
def load_mlflow_metrics() -> Dict[str, Any]:
    """Attempt to read metrics from MLflow runs."""
    metrics: Dict[str, Any] = {}
    try:
        import mlflow
        mlflow.set_tracking_uri(f"file:{MLFLOW_DIR}")
        client = mlflow.MlflowClient()
        experiments = client.search_experiments()
        for exp in experiments:
            runs = client.search_runs(exp.experiment_id, order_by=["start_time DESC"], max_results=5)
            for run in runs:
                run_metrics = run.data.metrics
                run_params  = run.data.params
                model_name  = run_params.get("model_name", run.info.run_id[:8])
                metrics[model_name] = {
                    "metrics": run_metrics,
                    "params":  run_params,
                    "run_id":  run.info.run_id,
                    "status":  run.info.status,
                }
    except Exception:
        pass
    return metrics


# ── Metrics computation helpers ─────────────────────────────────────────────

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute RMSE, MAE, R², MAPE."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)
    return {"RMSE": rmse, "MAE": mae, "R²": r2, "MAPE": mape}


def generate_demo_predictions(y_true: np.ndarray, noise_std: float = 10.0, seed: int = 42) -> np.ndarray:
    """Generate synthetic predictions for demo purposes."""
    rng = np.random.default_rng(seed)
    y_pred = y_true + rng.normal(0, noise_std, len(y_true))
    return np.clip(y_pred, 0, 125)


# ── System info helpers ─────────────────────────────────────────────────────

def get_system_info() -> Dict[str, str]:
    """Collect system information for the status page."""
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu_available else "N/A"
    except ImportError:
        gpu_available = False
        gpu_name = "N/A"

    try:
        import psutil
        ram = psutil.virtual_memory()
        ram_used  = f"{ram.used / (1024**3):.1f} GB"
        ram_total = f"{ram.total / (1024**3):.1f} GB"
        ram_pct   = f"{ram.percent}%"
        disk = psutil.disk_usage("/")
        disk_pct = f"{disk.percent}%"
        cpu_pct  = f"{psutil.cpu_percent(interval=0.5)}%"
    except ImportError:
        ram_used = ram_total = ram_pct = disk_pct = cpu_pct = "psutil not installed"

    return {
        "Python": platform.python_version(),
        "OS": f"{platform.system()} {platform.release()}",
        "CPU Usage": cpu_pct,
        "RAM Used": f"{ram_used} / {ram_total} ({ram_pct})",
        "Disk Usage": disk_pct,
        "GPU Available": str(gpu_available),
        "GPU Name": gpu_name,
    }


def check_service_status() -> Dict[str, bool]:
    """Check which services / files are available."""
    return {
        "Dataset (FD001)": (DATA_DIR / "train_FD001.txt").exists(),
        "LSTM Weights":    (MODEL_DIR / "best_lstm_FD001.pt").exists(),
        "Transformer Weights": (MODEL_DIR / "best_transformer_FD001.pt").exists(),
        "MLflow Directory": MLFLOW_DIR.exists(),
        "FastAPI App":     (PROJECT_ROOT / "src" / "serving" / "app.py").exists(),
    }


def check_api_health(base_url: str = "http://localhost:8000") -> Tuple[bool, str]:
    """Ping the FastAPI /health endpoint."""
    try:
        import httpx
        r = httpx.get(f"{base_url}/health", timeout=3)
        if r.status_code == 200:
            return True, r.json().get("status", "healthy")
        return False, f"HTTP {r.status_code}"
    except Exception as exc:
        return False, str(exc)
