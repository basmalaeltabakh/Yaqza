"""Evaluation metrics and visualization for RUL models.

Metrics:
- Standard: RMSE, MAE, R²
- Prognostics-specific: Prognostic Horizon (PH), scoring function
- Error distribution analysis
"""

from __future__ import annotations

from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ── Metrics ────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute standard regression metrics.

    Args:
        y_true: Ground truth RUL values.
        y_pred: Predicted RUL values.

    Returns:
        Dict with RMSE, MAE, R², MAPE.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R²": r2,
        "MAPE": mape,
    }


def compute_prognostics_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha1: float = 20.0,
    alpha2: float = -20.0,
) -> Dict[str, float]:
    """Compute prognostics-specific metrics from IEEE standards.

    **Scoring function** (IEEE 1856):
    - If prediction is too early (RUL too high): penalty proportional to |error|
    - If prediction is too late (RUL too low): exponentially larger penalty
    - Asymmetric: costs are higher for late predictions (safety critical)

    Args:
        y_true: Ground truth RUL.
        y_pred: Predicted RUL.
        alpha1: Early penalty coefficient (~20).
        alpha2: Late penalty coefficient (~-20).

    Returns:
        Dict with ``scoring_function``, ``alpha1_rate``, ``alpha2_rate``.
    """
    errors = y_true - y_pred

    # Compute score per sample
    scores = np.where(
        errors > 0,  # Late prediction (RUL too low, bad!)
        np.exp(errors / alpha2) - 1,
        np.exp(-errors / alpha1) - 1,
    )

    return {
        "Mean_Score": float(np.mean(scores)),
        "Early_Predictions": int(np.sum(errors > 0)),
        "Late_Predictions": int(np.sum(errors < 0)),
    }


def prognostic_horizon_ph(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold_error: float = 10.0,
) -> float:
    """Compute Prognostic Horizon (PH).

    **Definition**: How far into the future (days/cycles) the prediction
    remains within acceptable error bounds.

    For RUL prediction:
    - If predicted RUL ± threshold_error captures ground truth, PH is valid.
    - Longer PH = more warning time before failure.

    Args:
        y_true: Ground truth RUL.
        y_pred: Predicted RUL.
        threshold_error: Acceptable error margin (default 10 cycles).

    Returns:
        Prognostic Horizon as fraction (0 = no valid prediction, 1 = perfect).
    """
    within_bounds = np.abs(y_true - y_pred) <= threshold_error
    ph = np.mean(within_bounds)
    return float(ph)


# ── Visualization ──────────────────────────────────────────────────────────

def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "RUL Predictions vs Actual",
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Plot predicted vs actual RUL values.

    Args:
        y_true: Ground truth RUL.
        y_pred: Predicted RUL.
        title: Plot title.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Perfect prediction line
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "k--", label="Perfect Prediction", alpha=0.5)

    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.6, s=30)

    ax.set_xlabel("Actual RUL (cycles)")
    ax.set_ylabel("Predicted RUL (cycles)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "RUL Prediction Error Distribution",
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """Plot error statistics.

    Args:
        y_true: Ground truth.
        y_pred: Predictions.
        title: Title.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    errors = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Histogram
    axes[0].hist(errors, bins=30, edgecolor="black", alpha=0.7)
    axes[0].axvline(np.mean(errors), color="red", linestyle="--", label=f"Mean: {np.mean(errors):.2f}")
    axes[0].axvline(0, color="green", linestyle="--", label="Perfect")
    axes[0].set_xlabel("Prediction Error (Actual - Predicted)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Error Distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Box plot
    bp = axes[1].boxplot(errors, vert=True, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
    axes[1].set_ylabel("Prediction Error")
    axes[1].set_title("Error Box Plot")
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle(title)
    return fig


def plot_learning_curves(
    train_losses: list,
    val_losses: list,
    title: str = "Learning Curves",
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Plot training and validation loss over epochs.

    Args:
        train_losses: List of training losses.
        val_losses: List of validation losses.
        title: Plot title.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    epochs = np.arange(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, "o-", label="Train Loss", linewidth=2)
    ax.plot(epochs, val_losses, "s-", label="Validation Loss", linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_residuals_over_time(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residuals Over Time",
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """Plot residuals as a time series.

    Args:
        y_true: Ground truth.
        y_pred: Predictions.
        title: Title.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(range(len(residuals)), residuals, alpha=0.6, s=20)
    ax.axhline(0, color="red", linestyle="--", label="Zero Error")
    ax.axhline(np.mean(residuals), color="green", linestyle="--", label=f"Mean: {np.mean(residuals):.2f}")
    ax.fill_between(
        range(len(residuals)),
        np.std(residuals),
        -np.std(residuals),
        alpha=0.2,
        color="gray",
        label=f"±Std: ±{np.std(residuals):.2f}",
    )

    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_metrics_comparison(
    results: Dict[str, Dict[str, float]],
    metrics_keys: list = None,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """Compare metrics across multiple models.

    Args:
        results: Dict of {model_name: {metric: value}}.
        metrics_keys: Which metrics to plot (defaults to all).
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    if metrics_keys is None:
        metrics_keys = list(results[list(results.keys())[0]].keys())

    fig, axes = plt.subplots(1, len(metrics_keys), figsize=figsize)
    if len(metrics_keys) == 1:
        axes = [axes]

    model_names = list(results.keys())
    x_pos = np.arange(len(model_names))

    for idx, metric in enumerate(metrics_keys):
        values = [results[model][metric] for model in model_names]
        axes[idx].bar(x_pos, values, alpha=0.7)
        axes[idx].set_ylabel(metric)
        axes[idx].set_title(metric)
        axes[idx].set_xticks(x_pos)
        axes[idx].set_xticklabels(model_names, rotation=45, ha="right")
        axes[idx].grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    return fig


# ── Summary reporting ──────────────────────────────────────────────────────

def generate_evaluation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    output_path: str = None,
) -> str:
    """Generate a comprehensive evaluation report.

    Args:
        y_true: Ground truth RUL.
        y_pred: Predicted RUL.
        model_name: Name of the model.
        output_path: Optional file to save report.

    Returns:
        Report as a formatted string.
    """
    metrics = compute_metrics(y_true, y_pred)
    prog_metrics = compute_prognostics_metrics(y_true, y_pred)
    ph = prognostic_horizon_ph(y_true, y_pred)

    report = f"""
{'='*60}
RUL PREDICTION EVALUATION REPORT
{'='*60}

Model: {model_name}
Date: {pd.Timestamp.now()}
Samples: {len(y_true)}

STANDARD METRICS
{'-'*60}
  RMSE:                {metrics['RMSE']:.4f} cycles
  MAE:                 {metrics['MAE']:.4f} cycles
  R² Score:            {metrics['R²']:.4f}
  MAPE:                {metrics['MAPE']:.2f}%

PROGNOSTICS METRICS (IEEE 1856)
{'-'*60}
  Mean Scoring Func:   {prog_metrics['Mean_Score']:.4f}
  Early Predictions:   {prog_metrics['Early_Predictions']} / {len(y_true)}
  Late Predictions:    {prog_metrics['Late_Predictions']} / {len(y_true)}
  Prognostic Horizon:  {ph:.2%} (±10 cycles)

ERROR ANALYSIS
{'-'*60}
  Min Error:           {(y_true - y_pred).min():.2f} cycles
  Max Error:           {(y_true - y_pred).max():.2f} cycles
  Mean Error:          {(y_true - y_pred).mean():.2f} cycles
  Std Error:           {(y_true - y_pred).std():.2f} cycles

{'='*60}
"""

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)

    return report
