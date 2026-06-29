"""MLflow integration for experiment tracking and model versioning.

Logs:
- Hyperparameters
- Metrics (train/val loss, RMSE, MAE, R²)
- Plots (learning curves, predictions vs actual)
- Models (saved as artifacts)
- Run metadata (model type, dataset, timestamp)
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from mlflow.models.signature import infer_signature


class MLflowTracker:
    """Wrapper for MLflow experiment tracking.

    Usage:
        with MLflowTracker(experiment_name="yaqza-rul") as tracker:
            tracker.log_param("learning_rate", 0.001)
            tracker.log_metric("rmse", 12.5)
            tracker.log_model(model, "lstm_rul")
    """

    def __init__(
        self,
        experiment_name: str = "yaqza-rul-prediction",
        tracking_uri: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        """Initialize MLflow tracker.

        Args:
            experiment_name: MLflow experiment name.
            tracking_uri: MLflow tracking server URI (defaults to local `./mlruns`).
            tags: Dict of tags to add to the run.
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or "./mlruns"
        self.tags = tags or {}

        self.run_id = None
        self.run = None

    def __enter__(self):
        """Enter context manager (start MLflow run)."""
        
        mlflow.set_tracking_uri("file:./mlruns")
        if mlflow.active_run():
            mlflow.end_run()
        mlflow.set_experiment(self.experiment_name)

        self.run = mlflow.start_run()
        self.run_id = self.run.info.run_id

        # Log tags
        for key, value in self.tags.items():
            mlflow.set_tag(key, value)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager (end MLflow run)."""
        mlflow.end_run()

    def log_param(self, key: str, value: Any) -> None:
        """Log a hyperparameter.

        Args:
            key: Parameter name.
            value: Parameter value.
        """
        mlflow.log_param(key, value)

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log multiple hyperparameters.

        Args:
            params: Dict of {param_name: value}.
        """
        for key, value in params.items():
            self.log_param(key, value)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric.

        Args:
            key: Metric name.
            value: Metric value.
            step: Optional step (epoch).
        """
        mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple metrics.

        Args:
            metrics: Dict of {metric_name: value}.
            step: Optional step.
        """
        for key, value in metrics.items():
            self.log_metric(key, value, step=step)

    def log_figure(self, fig: plt.Figure, name: str) -> None:
        """Log a matplotlib figure as an artifact.

        Args:
            fig: Matplotlib figure.
            name: Name for the artifact (e.g., "learning_curves.png").
        """
        tmp_dir = tempfile.gettempdir()
        temp_path = Path(tmp_dir) / name
        fig.savefig(temp_path, dpi=100, bbox_inches="tight")
        mlflow.log_artifact(str(temp_path), artifact_path="plots")
        temp_path.unlink()

    def log_dict(self, d: Dict[str, Any], name: str) -> None:
        """Log a dictionary as JSON artifact.

        Args:
            d: Dictionary to log.
            name: Artifact name (e.g., "metrics.json").
        """
        tmp_dir = tempfile.gettempdir()
        temp_path = Path(tmp_dir) / name
        with open(temp_path, "w") as f:
            json.dump(d, f, indent=2)
        mlflow.log_artifact(str(temp_path), artifact_path="artifacts")
        temp_path.unlink()

    def log_model(
        self,
        model: Any,
        model_name: str,
        input_example: Optional[np.ndarray] = None,
    ) -> None:
        """Log a model as an artifact.

        For PyTorch models, this saves the state_dict.
        For sklearn models, uses MLflow's sklearn flavor.

        Args:
            model: Model object.
            model_name: Name for the model.
            input_example: Optional input example for signature inference.
        """
        import torch

        if isinstance(model, torch.nn.Module):
            # PyTorch model
            model_path = f"model_{model_name}.pt"
            torch.save(model.state_dict(), model_path)
            mlflow.log_artifact(model_path, artifact_path="models")
            Path(model_path).unlink()
        else:
            # Try sklearn flavor
            try:
                mlflow.sklearn.log_model(model, model_name)
            except Exception as e:
                # Fallback: just pickle
                import pickle

                model_path = f"model_{model_name}.pkl"
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)
                mlflow.log_artifact(model_path, artifact_path="models")
                Path(model_path).unlink()

    def get_run_id(self) -> str:
        """Get the current run ID.

        Returns:
            MLflow run ID.
        """
        return self.run_id

    def set_status(self, status: str) -> None:
        """Set run status.

        Args:
            status: One of "SCHEDULED", "RUNNING", "FINISHED", "FAILED".
        """
        mlflow.set_tag("status", status)


def log_training_summary(
    tracker: MLflowTracker,
    model_name: str,
    train_losses: list,
    val_losses: list,
    test_metrics: Dict[str, float],
    test_predictions: np.ndarray = None,
    test_actuals: np.ndarray = None,
) -> None:
    """Log complete training summary to MLflow.

    Args:
        tracker: MLflowTracker instance.
        model_name: Name of the model.
        train_losses: List of training losses per epoch.
        val_losses: List of validation losses per epoch.
        test_metrics: Dict of test metrics {name: value}.
        test_predictions: Optional test predictions array.
        test_actuals: Optional test actual values array.
    """
    # Log metrics
    tracker.log_metric("best_train_loss", min(train_losses))
    tracker.log_metric("best_val_loss", min(val_losses))
    tracker.log_metrics(test_metrics)

    # Log final metrics as dict
    final_results = {
        "model": model_name,
        "epochs_trained": len(train_losses),
        "best_train_loss": float(min(train_losses)),
        "best_val_loss": float(min(val_losses)),
        "final_train_loss": float(train_losses[-1]),
        "final_val_loss": float(val_losses[-1]),
        "test_metrics": test_metrics,
    }

    if test_predictions is not None and test_actuals is not None:
        final_results["test_predictions"] = test_predictions.tolist()[:100]  # First 100
        final_results["test_actuals"] = test_actuals.tolist()[:100]

    tracker.log_dict(final_results, f"summary_{model_name}.json")

    # Create and log plots
    from src.mlops.evaluation import (
        plot_learning_curves,
        plot_predictions_vs_actual,
        plot_error_distribution,
    )

    # Learning curves
    fig_curves = plot_learning_curves(train_losses, val_losses)
    tracker.log_figure(fig_curves, f"learning_curves_{model_name}.png")
    plt.close(fig_curves)

    # Predictions vs actual
    if test_predictions is not None and test_actuals is not None:
        fig_pred = plot_predictions_vs_actual(test_actuals, test_predictions)
        tracker.log_figure(fig_pred, f"predictions_vs_actual_{model_name}.png")
        plt.close(fig_pred)

        # Error distribution
        fig_error = plot_error_distribution(test_actuals, test_predictions)
        tracker.log_figure(fig_error, f"error_distribution_{model_name}.png")
        plt.close(fig_error)
