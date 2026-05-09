"""
Yaqza Dashboard — Reusable Plotly Chart Library.

Every chart function returns a plotly Figure object with:
- Theme-aware colours (dark / light)
- Consistent styling, hover tooltips, and interactivity
- Zoom, pan, cross-filter support
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Colour palette ──────────────────────────────────────────────────────────
PALETTE = {
    "primary":   "#6366f1",
    "secondary": "#8b5cf6",
    "accent":    "#a78bfa",
    "success":   "#10b981",
    "warning":   "#f59e0b",
    "danger":    "#ef4444",
    "info":      "#3b82f6",
    "cyan":      "#06b6d4",
    "rose":      "#f43f5e",
    "amber":     "#f59e0b",
}

SERIES_COLORS = [
    "#6366f1", "#10b981", "#f43f5e", "#f59e0b",
    "#3b82f6", "#8b5cf6", "#06b6d4", "#ec4899",
    "#14b8a6", "#a855f7", "#eab308", "#22d3ee",
]


def _base_layout(title: str = "", height: int = 450) -> dict:
    """Return shared layout properties for all charts."""
    return dict(
        title=dict(text=title, font=dict(size=16, family="Inter, sans-serif")),
        font=dict(family="Inter, sans-serif", size=12),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=50, r=30, t=55, b=45),
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font=dict(size=11),
        ),
    )


# ── Distribution charts ────────────────────────────────────────────────────

def rul_distribution(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> go.Figure:
    """Overlapping histograms for RUL across splits."""
    fig = go.Figure()
    for arr, name, color in [
        (y_train, "Train", PALETTE["primary"]),
        (y_val,   "Validation", PALETTE["success"]),
        (y_test,  "Test", PALETTE["rose"]),
    ]:
        fig.add_trace(go.Histogram(
            x=arr, name=name, opacity=0.65,
            marker_color=color, nbinsx=35,
            hovertemplate="RUL: %{x:.0f}<br>Count: %{y}<extra></extra>",
        ))
    fig.update_layout(
        **_base_layout("RUL Distribution Across Splits"),
        barmode="overlay",
        xaxis_title="RUL (cycles)",
        yaxis_title="Frequency",
    )
    return fig


def rul_violin(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> go.Figure:
    """Violin plots showing RUL spread per split."""
    df = pd.DataFrame({
        "RUL": np.concatenate([y_train, y_val, y_test]),
        "Split": (
            ["Train"] * len(y_train)
            + ["Validation"] * len(y_val)
            + ["Test"] * len(y_test)
        ),
    })
    fig = px.violin(
        df, x="Split", y="RUL", color="Split",
        box=True, points="outliers",
        color_discrete_sequence=[PALETTE["primary"], PALETTE["success"], PALETTE["rose"]],
    )
    fig.update_layout(**_base_layout("RUL Violin Plot", height=420))
    return fig


# ── Sensor / Sequence charts ───────────────────────────────────────────────

def sensor_time_series(
    sample: np.ndarray,
    feature_names: List[str],
    selected_features: List[int],
    sample_idx: int = 0,
    rul_value: float = 0.0,
) -> go.Figure:
    """Multi-line sensor readings for a single sequence."""
    fig = go.Figure()
    for i, feat_idx in enumerate(selected_features):
        if feat_idx < sample.shape[1]:
            name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"Feature {feat_idx}"
            fig.add_trace(go.Scatter(
                x=list(range(sample.shape[0])),
                y=sample[:, feat_idx],
                mode="lines",
                name=name,
                line=dict(width=2, color=SERIES_COLORS[i % len(SERIES_COLORS)]),
                hovertemplate=f"{name}<br>Step: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>",
            ))
    fig.update_layout(
        **_base_layout(f"Sample #{sample_idx} — Sensor Readings  (RUL = {rul_value:.0f})"),
        xaxis_title="Time Step",
        yaxis_title="Normalised Value",
    )
    return fig


def feature_statistics_bar(
    X: np.ndarray,
    feature_names: List[str],
    max_features: int = 15,
) -> go.Figure:
    """Grouped bar chart of mean / std / min / max per feature."""
    n = min(max_features, X.shape[2])
    names = [feature_names[i] if i < len(feature_names) else f"F{i}" for i in range(n)]
    mean_v = X.mean(axis=(0, 1))[:n]
    std_v  = X.std(axis=(0, 1))[:n]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Mean", x=names, y=mean_v, marker_color=PALETTE["primary"]))
    fig.add_trace(go.Bar(name="Std Dev", x=names, y=std_v, marker_color=PALETTE["warning"]))
    fig.update_layout(
        **_base_layout("Feature Statistics (Train Set)"),
        barmode="group",
        xaxis_title="Feature",
        yaxis_title="Value",
        xaxis_tickangle=-45,
    )
    return fig


def correlation_heatmap(df: pd.DataFrame, feature_cols: List[str]) -> go.Figure:
    """Interactive correlation matrix heatmap."""
    cols = feature_cols[:20]  # cap for readability
    corr = df[cols].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        hovertemplate="%{x} vs %{y}<br>r = %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        **_base_layout("Feature Correlation Matrix", height=550),
        xaxis_tickangle=-45,
    )
    return fig


def engine_lifecycle(
    df: pd.DataFrame,
    engine_id: int,
    sensor_col: str,
) -> go.Figure:
    """Line chart showing one engine's sensor + RUL over its lifecycle."""
    unit = df[df["unit_id"] == engine_id].sort_values("cycle")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=unit["cycle"], y=unit[sensor_col],
            name=sensor_col, mode="lines",
            line=dict(color=PALETTE["primary"], width=2),
        ),
        secondary_y=False,
    )
    if "RUL" in unit.columns:
        fig.add_trace(
            go.Scatter(
                x=unit["cycle"], y=unit["RUL"],
                name="RUL", mode="lines",
                line=dict(color=PALETTE["danger"], width=2, dash="dash"),
            ),
            secondary_y=True,
        )
    fig.update_layout(
        **_base_layout(f"Engine #{engine_id} — {sensor_col} Lifecycle", height=400),
    )
    fig.update_yaxes(title_text=sensor_col, secondary_y=False)
    fig.update_yaxes(title_text="RUL (cycles)", secondary_y=True)
    return fig


def rolling_statistics_chart(
    df: pd.DataFrame,
    engine_id: int,
    sensor_col: str,
) -> go.Figure:
    """Rolling mean + std bands for a sensor on a given engine."""
    unit = df[df["unit_id"] == engine_id].sort_values("cycle")
    roll5_mean = f"{sensor_col}_roll5_mean"
    roll5_std  = f"{sensor_col}_roll5_std"

    fig = go.Figure()
    if sensor_col in unit.columns:
        fig.add_trace(go.Scatter(
            x=unit["cycle"], y=unit[sensor_col],
            name="Raw", mode="lines", line=dict(color=PALETTE["primary"], width=1.5),
        ))
    if roll5_mean in unit.columns:
        mean_vals = unit[roll5_mean]
        std_vals  = unit.get(roll5_std, pd.Series(0, index=unit.index))
        fig.add_trace(go.Scatter(
            x=unit["cycle"], y=mean_vals,
            name="Rolling Mean (5)", mode="lines",
            line=dict(color=PALETTE["success"], width=2),
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([unit["cycle"], unit["cycle"][::-1]]),
            y=pd.concat([mean_vals + std_vals, (mean_vals - std_vals)[::-1]]),
            fill="toself", fillcolor="rgba(16,185,129,0.1)",
            line=dict(width=0), name="±1 Std",
            hoverinfo="skip",
        ))
    fig.update_layout(**_base_layout(f"Engine #{engine_id} — {sensor_col} Rolling Stats"))
    return fig


# ── Model performance charts ───────────────────────────────────────────────

def predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
) -> go.Figure:
    """Scatter of predicted vs actual RUL with perfect-prediction line."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred, mode="markers",
        marker=dict(size=6, color=PALETTE["primary"], opacity=0.6,
                    line=dict(width=0.5, color="white")),
        name="Predictions",
        hovertemplate="Actual: %{x:.1f}<br>Predicted: %{y:.1f}<extra></extra>",
    ))
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[mn, mx], y=[mn, mx], mode="lines",
        line=dict(color=PALETTE["danger"], dash="dash", width=2),
        name="Perfect",
    ))
    fig.update_layout(
        **_base_layout(f"{model_name} — Predicted vs Actual RUL"),
        xaxis_title="Actual RUL (cycles)",
        yaxis_title="Predicted RUL (cycles)",
    )
    return fig


def error_histogram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
) -> go.Figure:
    """Histogram of absolute prediction errors."""
    errors = np.abs(y_true - y_pred)
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=errors, nbinsx=40,
        marker_color=PALETTE["warning"], opacity=0.8,
        name="Absolute Error",
    ))
    fig.add_vline(x=errors.mean(), line_dash="dash", line_color=PALETTE["danger"],
                  annotation_text=f"Mean: {errors.mean():.2f}")
    fig.update_layout(
        **_base_layout(f"{model_name} — Error Distribution"),
        xaxis_title="Absolute Error (cycles)",
        yaxis_title="Frequency",
    )
    return fig


def residual_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
) -> go.Figure:
    """Residuals vs predicted values."""
    residuals = y_true - y_pred
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_pred, y=residuals, mode="markers",
        marker=dict(size=5, color=PALETTE["info"], opacity=0.5),
        hovertemplate="Predicted: %{x:.1f}<br>Residual: %{y:.1f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color=PALETTE["danger"])
    fig.update_layout(
        **_base_layout(f"{model_name} — Residual Analysis"),
        xaxis_title="Predicted RUL",
        yaxis_title="Residual (Actual − Predicted)",
    )
    return fig


def model_comparison_bar(
    metrics: Dict[str, Dict[str, float]],
    metric_key: str = "RMSE",
) -> go.Figure:
    """Horizontal bar comparing one metric across models."""
    models = list(metrics.keys())
    values = [metrics[m].get(metric_key, 0) for m in models]
    fig = go.Figure(go.Bar(
        x=values, y=models, orientation="h",
        marker=dict(
            color=values,
            colorscale=[[0, PALETTE["success"]], [1, PALETTE["danger"]]],
        ),
        text=[f"{v:.2f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        **_base_layout(f"Model Comparison — {metric_key}", height=320),
        xaxis_title=metric_key,
    )
    return fig


def learning_curves(
    train_losses: List[float],
    val_losses: List[float],
    model_name: str = "Model",
) -> go.Figure:
    """Training & validation loss curves."""
    epochs = list(range(1, len(train_losses) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, y=train_losses, name="Train Loss",
        mode="lines+markers", marker=dict(size=4),
        line=dict(color=PALETTE["primary"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=epochs, y=val_losses, name="Val Loss",
        mode="lines+markers", marker=dict(size=4),
        line=dict(color=PALETTE["warning"], width=2),
    ))
    fig.update_layout(
        **_base_layout(f"{model_name} — Learning Curves"),
        xaxis_title="Epoch",
        yaxis_title="Loss (MSE)",
    )
    return fig


def feature_importance_bar(
    importances: np.ndarray,
    feature_names: List[str],
    top_n: int = 15,
) -> go.Figure:
    """Horizontal bar chart of top feature importances."""
    idx = np.argsort(importances)[-top_n:]
    fig = go.Figure(go.Bar(
        x=importances[idx],
        y=[feature_names[i] if i < len(feature_names) else f"F{i}" for i in idx],
        orientation="h",
        marker_color=PALETTE["primary"],
    ))
    fig.update_layout(
        **_base_layout(f"Top {top_n} Feature Importances", height=420),
        xaxis_title="Importance",
    )
    return fig


# ── 3D scatter ──────────────────────────────────────────────────────────────

def rul_3d_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> go.Figure:
    """3D scatter: actual vs predicted vs error."""
    errors = np.abs(y_true - y_pred)
    fig = go.Figure(data=[go.Scatter3d(
        x=y_true, y=y_pred, z=errors,
        mode="markers",
        marker=dict(size=3, color=errors, colorscale="Viridis", opacity=0.7,
                    colorbar=dict(title="Error")),
        hovertemplate=(
            "Actual: %{x:.1f}<br>Predicted: %{y:.1f}<br>"
            "Error: %{z:.1f}<extra></extra>"
        ),
    )])
    fig.update_layout(
        **_base_layout("3D Error Landscape", height=520),
        scene=dict(
            xaxis_title="Actual RUL",
            yaxis_title="Predicted RUL",
            zaxis_title="|Error|",
        ),
    )
    return fig


# ── System monitoring ───────────────────────────────────────────────────────

def gauge_chart(value: float, title: str, max_val: float = 100) -> go.Figure:
    """Simple gauge chart for resource usage."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title=dict(text=title, font=dict(size=14, family="Inter")),
        gauge=dict(
            axis=dict(range=[0, max_val]),
            bar=dict(color=PALETTE["primary"]),
            steps=[
                dict(range=[0, max_val * 0.6], color="rgba(16,185,129,0.15)"),
                dict(range=[max_val * 0.6, max_val * 0.85], color="rgba(245,158,11,0.15)"),
                dict(range=[max_val * 0.85, max_val], color="rgba(239,68,68,0.15)"),
            ],
            threshold=dict(
                line=dict(color=PALETTE["danger"], width=2),
                thickness=0.8, value=max_val * 0.9,
            ),
        ),
    ))
    fig.update_layout(
        height=220, margin=dict(l=30, r=30, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"),
    )
    return fig
