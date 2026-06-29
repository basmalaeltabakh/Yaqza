"""
Main orchestration script for training all RUL models.

Usage:
    python scripts/train_all_models.py --subset FD001 --use-mlflow

This script:
1. Loads and preprocesses CMAPSS data
2. Creates sequences for deep learning models
3. Trains LSTM, Transformer, and XGBoost models
4. Evaluates on test set
5. Logs everything to MLflow
6. Saves results and checkpoints
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import (
    BATCH_SIZE,
    CMAPSS_DIR,
    EPOCHS,
    LEARNING_RATE,
    LSTM_CONFIG,
    MAX_RUL,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    PATIENCE,
    RANDOM_SEED,
    REPORTS_DIR,
    TRANSFORMER_CONFIG,
    WEIGHTS_DIR,
    WINDOW_SIZE,
    WEIGHT_DECAY,
)
from src.models.baseline import LightGBMRUL, XGBoostRUL
from src.models.lstm_rul import ImprovedLSTM, LSTMWithAttention
from src.models.tft_rul import TransformerEncoder
from src.mlops.evaluation import (
    compute_metrics,
    compute_prognostics_metrics,
    generate_evaluation_report,
    plot_error_distribution,
    plot_learning_curves,
    plot_predictions_vs_actual,
)
from src.mlops.mlflow_tracker import MLflowTracker, log_training_summary
from src.mlops.training import EarlyStopping, Trainer, create_trainer
from src.preprocessing.features import preprocess_cmapss
from src.preprocessing.windows import CMAPSSDataset, create_sequences, create_test_sequences, make_loaders, time_series_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_deep_learning_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    model_name: str,
    device: str = "cpu",
    epochs: int = 100,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    patience: int = 15,
    checkpoint_path: Optional[Path] = None,
) -> Dict[str, object]:
    """Train a deep learning model (LSTM or Transformer).

    Args:
        model: PyTorch model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        test_loader: Test DataLoader.
        model_name: Name of the model.
        device: "cpu" or "cuda".
        epochs: Number of epochs.
        learning_rate: Learning rate.
        weight_decay: L2 regularization.
        patience: Early stopping patience.
        checkpoint_path: Path to save best model.

    Returns:
        Dict with training history, test predictions, and metrics.
    """
    logger.info(f"🚀 Training {model_name}...")

    trainer = create_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    history = trainer.fit(num_epochs=epochs, verbose_interval=10)

    # Evaluate on test set
    test_predictions = trainer.predict(test_loader)

    return {
        "model": model,
        "train_losses": history["train_losses"],
        "val_losses": history["val_losses"],
        "test_predictions": test_predictions,
        "trainer": trainer,
    }


def train_baseline_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    model_type: str = "xgboost",
) -> Dict[str, object]:
    """Train a baseline ML model (XGBoost or LightGBM).

    Args:
        X_train: Training sequences.
        y_train: Training targets.
        X_val: Validation sequences.
        y_val: Validation targets.
        X_test: Test sequences.
        model_type: "xgboost" or "lightgbm".

    Returns:
        Dict with model and predictions.
    """
    logger.info(f"🚀 Training {model_type}...")

    if model_type == "xgboost":
        model = XGBoostRUL(n_estimators=100, max_depth=6)
    else:
        model = LightGBMRUL(n_estimators=100, max_depth=6)

    model.fit(X_train, y_train, X_val, y_val)
    test_predictions = model.predict(X_test)

    return {
        "model": model,
        "test_predictions": test_predictions,
    }


def main(args) -> None:
    """Main training pipeline."""
    set_seed(RANDOM_SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # ── 1. Load and preprocess data ────────────────────────────────────────
    logger.info(f"📂 Loading CMAPSS subset: {args.subset}")
    prep_result = preprocess_cmapss(
        data_dir=CMAPSS_DIR,
        subset=args.subset,
        max_rul=MAX_RUL,
        rolling_windows=[5, 10],
        val_fraction=0.2,
        output_dir=None,  # Set to a path to save preprocessed data
    )

    train_df = prep_result["train_df"]
    val_df = prep_result["val_df"]
    test_df = prep_result["test_df"]
    feature_cols = prep_result["feature_cols"]

    logger.info(f"✓ Loaded: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test samples")

    # ── 2. Create sequences ────────────────────────────────────────────────
    logger.info("🪟 Creating sliding window sequences...")
    X_train, y_train = create_sequences(
        train_df, feature_cols, window_size=WINDOW_SIZE, stride=1
    )
    X_val, y_val = create_sequences(
        val_df, feature_cols, window_size=WINDOW_SIZE, stride=1
    )
    X_test = create_test_sequences(test_df, feature_cols, window_size=WINDOW_SIZE)
    y_test = test_df.groupby("unit_id")["RUL"].last().values

    logger.info(
        f"✓ Sequences: X_train {X_train.shape}, X_val {X_val.shape}, X_test {X_test.shape}"
    )

    # ── 3. Create DataLoaders ──────────────────────────────────────────────
    train_loader, val_loader = make_loaders(
        X_train, y_train, X_val, y_val,
        batch_size=BATCH_SIZE,
    )

    test_dataset = CMAPSSDataset(X_test, np.zeros(len(X_test)))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ── 4. Initialize models ───────────────────────────────────────────────
    n_features = X_train.shape[2]

    models_to_train = {}

    if args.train_lstm:
        lstm_model = ImprovedLSTM(
            input_size=n_features,
            hidden_size=LSTM_CONFIG["hidden_size"],
            num_layers=LSTM_CONFIG["num_layers"],
            dropout=LSTM_CONFIG["dropout"],
            use_attention=True,
        )
        models_to_train["LSTM"] = lstm_model

    if args.train_transformer:
        transformer_model = TransformerEncoder(
            input_size=n_features,
            d_model=TRANSFORMER_CONFIG["d_model"],
            nhead=TRANSFORMER_CONFIG["nhead"],
            num_layers=TRANSFORMER_CONFIG["num_encoder_layers"],
            dim_feedforward=TRANSFORMER_CONFIG["dim_feedforward"],
            dropout=TRANSFORMER_CONFIG["dropout"],
            max_seq_len=WINDOW_SIZE,
        )
        models_to_train["Transformer"] = transformer_model

    # ── 5. Train models ───────────────────────────────────────────────────
    results = {}

    for model_name, model in models_to_train.items():
        checkpoint_path = WEIGHTS_DIR / f"best_{model_name.lower()}_{args.subset}.pt"

        result = train_deep_learning_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            model_name=model_name,
            device=device,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            patience=PATIENCE,
            checkpoint_path=checkpoint_path,
        )

        results[model_name] = result

    # ── 6. Train baseline models ───────────────────────────────────────────
    if args.train_xgboost:
        xgb_result = train_baseline_model(
            X_train, y_train, X_val, y_val, X_test, model_type="xgboost"
        )
        results["XGBoost"] = xgb_result

    if args.train_lightgbm:
        lgb_result = train_baseline_model(
            X_train, y_train, X_val, y_val, X_test, model_type="lightgbm"
        )
        results["LightGBM"] = lgb_result

    # ── 7. Evaluate all models ─────────────────────────────────────────────
    logger.info("📊 Evaluating models...")
    evaluation_results = {}

    for model_name, result in results.items():
        y_pred = result["test_predictions"]
        metrics = compute_metrics(y_test, y_pred)
        prog_metrics = compute_prognostics_metrics(y_test, y_pred)

        evaluation_results[model_name] = {
            **metrics,
            **prog_metrics,
        }

        logger.info(f"✓ {model_name} → RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}, R²: {metrics['R²']:.4f}")

    # ── 8. Log to MLflow ───────────────────────────────────────────────────
    if args.use_mlflow:
        logger.info("📈 Logging to MLflow...")

        with MLflowTracker(
            experiment_name=MLFLOW_EXPERIMENT_NAME,
            tracking_uri=MLFLOW_TRACKING_URI,
            tags={
                "dataset": args.subset,
                "window_size": str(WINDOW_SIZE),
                "max_rul": str(MAX_RUL),
            },
        ) as tracker:

            for model_name, result in results.items():
                with MLflowTracker(
                    experiment_name=MLFLOW_EXPERIMENT_NAME,
                    tracking_uri=MLFLOW_TRACKING_URI,
                    tags={
                        "model": model_name,
                        "dataset": args.subset,
                    },
                ) as model_tracker:

                    # Log hyperparameters
                    if "train_losses" in result:  # Deep learning
                        model_tracker.log_params({
                            "learning_rate": LEARNING_RATE,
                            "batch_size": BATCH_SIZE,
                            "epochs": EPOCHS,
                            "patience": PATIENCE,
                        })

                    # Log metrics
                    model_tracker.log_metrics(evaluation_results[model_name])

                    # Log plots
                    if "train_losses" in result:
                        fig = plot_learning_curves(result["train_losses"], result["val_losses"])
                        model_tracker.log_figure(fig, f"learning_curves_{model_name}.png")
                        plt.close(fig)

                    fig = plot_predictions_vs_actual(y_test, result["test_predictions"], title=f"{model_name} Predictions")
                    model_tracker.log_figure(fig, f"predictions_{model_name}.png")
                    plt.close(fig)

                    fig = plot_error_distribution(y_test, result["test_predictions"], title=f"{model_name} Errors")
                    model_tracker.log_figure(fig, f"errors_{model_name}.png")
                    plt.close(fig)

                    logger.info(f"✓ Logged {model_name} to MLflow")

    # ── 9. Save reports ────────────────────────────────────────────────────
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    for model_name, result in results.items():
        report = generate_evaluation_report(
            y_test,
            result["test_predictions"],
            model_name=model_name,
            output_path=REPORTS_DIR / f"evaluation_{model_name}_{args.subset}.txt",
        )
        logger.info(report)

    logger.info("✅ Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RUL prediction models")
    parser.add_argument(
        "--subset",
        type=str,
        default="FD001",
        help="CMAPSS subset (FD001, FD002, FD003, FD004)",
    )
    parser.add_argument(
        "--train-lstm",
        action="store_true",
        default=True,
        help="Train LSTM model",
    )
    parser.add_argument(
        "--train-transformer",
        action="store_true",
        default=True,
        help="Train Transformer model",
    )
    parser.add_argument(
        "--train-xgboost",
        action="store_true",
        default=True,
        help="Train XGBoost baseline",
    )
    parser.add_argument(
        "--train-lightgbm",
        action="store_true",
        default=False,
        help="Train LightGBM baseline",
    )
    parser.add_argument(
        "--use-mlflow",
        action="store_true",
        default=True,
        help="Log to MLflow",
    )

    args = parser.parse_args()
    main(args)
