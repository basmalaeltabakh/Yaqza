#  Yaqza Quick Reference Guide

**Fast lookups for key modules, classes, and functions**

---

##  Preprocessing Pipeline

### Loading & Preparing Data

```python
from src.preprocessing.features import preprocess_cmapss

# Single function to do everything
result = preprocess_cmapss(
    data_dir="data/CMAPSS",
    subset="FD001",
    max_rul=125,
    rolling_windows=[5, 10],
    scaler_type="standard",
    val_fraction=0.2
)

# Returns dict with:
train_df = result['train']              # (n_train_samples, 70 features)
val_df = result['val']                  # (n_val_samples, 70 features)
test_df = result['test']                # (n_test_samples, 70 features)
feature_cols = result['feature_columns'] # Column names
scaler = result['scaler']                # Fitted StandardScaler
```

### Creating Sequences

```python
from src.preprocessing.windows import create_sequences, make_loaders

# Convert DataFrames to sliding windows
X_train_seq, y_train = create_sequences(
    train_df,
    feature_cols=feature_cols,
    window_size=30,
    stride=1,
    target_col="RUL"
)
# Output: X_train_seq shape (n_windows, 30, 70), y_train shape (n_windows,)

# Create PyTorch DataLoaders
train_loader, val_loader = make_loaders(
    X_train_seq, y_train,
    X_val_seq, y_val,
    batch_size=256,
    num_workers=0
)
```

---

##  Model Architectures

### Transformer (Recommended)

```python
from src.models.tft_rul import TransformerEncoder

model = TransformerEncoder(
    input_size=70,           # Features
    d_model=128,             # Embedding dimension
    nhead=8,                 # Attention heads
    num_layers=3,            # Encoder layers
    dim_feedforward=256,
    dropout=0.1,
    max_seq_len=50,
    output_size=1            # RUL output
)

# Forward pass
output = model(X_batch)  # (batch_size, 1)
```

### LSTM

```python
from src.models.lstm_rul import ImprovedLSTM

model = ImprovedLSTM(
    input_size=70,
    hidden_size=128,
    num_layers=3,
    dropout=0.3,
    use_attention=True,
    output_size=1
)

# Forward pass
output = model(X_batch)  # (batch_size, 1)
```

### XGBoost Baseline

```python
from src.models.baseline import XGBoostRUL

model = XGBoostRUL(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1
)

# Fit (sequences automatically flattened)
model.fit(X_train_seq, y_train)

# Predict
preds = model.predict(X_test_seq)  # (n_samples,)

# Feature importance
importance = model.get_feature_importance()
```

---

## Training

### Using the Trainer Class

```python
from src.mlops.training import Trainer, EarlyStopping, create_trainer
import torch

# Create trainer (factory function with sensible defaults)
trainer = create_trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device="cuda",
    learning_rate=1e-3
)

# Train with early stopping
history = trainer.fit(num_epochs=100)

# Get predictions
test_preds = trainer.predict(test_loader)
```

### Manual Training Setup

```python
from src.mlops.training import EarlyStopping
import torch.optim as optim

early_stop = EarlyStopping(
    patience=15,
    delta=0.001,
    checkpoint_path="model_weights/best_model.pt"
)

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = torch.nn.MSELoss()

for epoch in range(100):
    # Training step
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    # Validation step
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            preds = model(X_batch)
            val_loss += criterion(preds, y_batch).item()
    
    # Early stopping check
    early_stop(val_loss, model)
    if early_stop.early_stop:
        break

# Load best weights
early_stop.load_best_model(model)
```

---

##  Evaluation

### Standard Metrics

```python
from src.mlops.evaluation import compute_metrics, compute_prognostics_metrics

# Standard regression metrics
metrics = compute_metrics(y_true, y_pred)
print(f"RMSE: {metrics['rmse']:.2f}")
print(f"MAE: {metrics['mae']:.2f}")
print(f"R²: {metrics['r2']:.4f}")

# Prognostics metrics (IEEE 1856 + PH)
prog_metrics = compute_prognostics_metrics(y_true, y_pred)
print(f"IEEE Score: {prog_metrics['ieee_score']:.1f}")
print(f"PH (±10): {prog_metrics['prognostic_horizon']:.2%}")
```

### Plotting & Reports

```python
from src.mlops.evaluation import (
    plot_predictions_vs_actual,
    plot_error_distribution,
    plot_learning_curves,
    generate_evaluation_report
)

# Create plots
plot_predictions_vs_actual(y_test, y_pred, save_path="reports/predictions.png")
plot_error_distribution(y_test, y_pred, save_path="reports/errors.png")
plot_learning_curves(train_losses, val_losses, save_path="reports/curves.png")

# Generate text report
report = generate_evaluation_report(
    model_name="Transformer",
    metrics=metrics,
    prog_metrics=prog_metrics
)
print(report)
```

---

##  MLflow Tracking

### Logging Experiments

```python
from src.mlops.mlflow_tracker import MLflowTracker, log_training_summary

# Context manager pattern
with MLflowTracker("yaqza-rul-prediction") as tracker:
    # Log hyperparameters
    tracker.log_params({
        "hidden_size": 128,
        "num_layers": 3,
        "learning_rate": 1e-3,
        "batch_size": 256
    })
    
    # Log metrics per epoch
    for epoch, val_loss in enumerate(val_losses):
        tracker.log_metric("val_loss", val_loss, step=epoch)
    
    # Log final metrics
    tracker.log_metrics({
        "rmse": metrics['rmse'],
        "r2": metrics['r2'],
        "ieee_score": prog_metrics['ieee_score']
    })
    
    # Log plots
    tracker.log_figure("predictions.png", fig)
    
    # Log model
    torch.save(model.state_dict(), "model.pt")
    tracker.log_model("model.pt", "model")

# Or use helper function
log_training_summary(
    tracker, "Transformer", 
    train_losses, val_losses,
    metrics, test_preds, y_test
)
```

### Viewing Results

```bash
# Start MLflow UI
mlflow ui --backend-store-uri ./mlruns

# Navigate to http://localhost:5000
# - View runs side-by-side
# - Compare metrics & plots
# - Download models
```

---

##  Deployment

### ONNX Export

```python
from src.serving.model_utils import export_to_onnx, ONNXPredictor
from pathlib import Path

# Export model
export_to_onnx(
    model=transformer,
    input_shape=(1, 30, 70),
    output_path=Path("model_weights/transformer.onnx"),
    dynamic_axes={
        "input": {0: "batch"},
        "output": {0: "batch"}
    }
)

# Use ONNX model for inference
predictor = ONNXPredictor(Path("model_weights/transformer.onnx"))

# Single prediction
rul = predictor.predict_single(X_sample)  # X_sample: (30, 70)

# Batch prediction
ruls = predictor.predict(X_batch)  # X_batch: (batch, 30, 70)
```

### FastAPI Service

```bash
# Start server
uvicorn src.serving.app:app --port 8000

# Health check
curl http://localhost:8000/health

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sequence": [[...]], "model": "transformer"}'

# View API docs
# Navigate to http://localhost:8000/docs
```

### Model Registry

```python
from src.serving.model_utils import ModelRegistry

# Register model
registry = ModelRegistry(Path("model_weights"))
registry.register(
    model_name="transformer",
    model=transformer,
    version="1.0.0",
    metadata={
        "config": {...},
        "metrics": {...}
    }
)

# Load model
model, metadata = registry.load(
    "transformer", "1.0.0",
    model_class=TransformerEncoder,
    device="cuda"
)
```

---

##  Testing

### Run Tests

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_preprocessing.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Quick test
python -m pytest tests/ -x  # Stop on first failure
```

### Example Test

```python
import pytest
from src.preprocessing.features import compute_rul

def test_compute_rul():
    """Test RUL calculation."""
    df = pd.DataFrame({
        'unit': [1, 1, 1],
        'RUL': [100, 50, 0]
    })
    result = compute_rul(df, max_rul=125)
    assert len(result) == len(df)
    assert result['RUL'].max() <= 125
```

---

##  Configuration

### Main Hyperparameters (`src/config.py`)

```python
from src.config import *

# Preprocessing
MAX_RUL = 125              # Piece-wise linear clipping
WINDOW_SIZE = 30           # Sequence length
ROLLING_WINDOWS = [5, 10]  # Rolling statistics

# Training
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4        # L2 regularization
PATIENCE = 15              # Early stopping patience
GRADIENT_CLIP = 1.0

# Models
LSTM_CONFIG = {
    "hidden_size": 128,
    "num_layers": 3,
    "dropout": 0.3,
}

TRANSFORMER_CONFIG = {
    "d_model": 128,
    "nhead": 8,
    "num_encoder_layers": 3,
    "dim_feedforward": 256,
    "dropout": 0.1,
}

XGBOOST_CONFIG = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
}
```

---

## Common Workflows

### Complete Training Pipeline

```python
from src.preprocessing.features import preprocess_cmapss
from src.preprocessing.windows import create_sequences, make_loaders
from src.models.tft_rul import TransformerEncoder
from src.mlops.training import create_trainer
from src.mlops.evaluation import compute_metrics
from src.mlops.mlflow_tracker import MLflowTracker

# 1. Preprocess
result = preprocess_cmapss("data/CMAPSS", "FD001")
train_df, val_df, test_df = result['train'], result['val'], result['test']

# 2. Create sequences
X_train, y_train = create_sequences(train_df, result['feature_columns'])
X_val, y_val = create_sequences(val_df, result['feature_columns'])
X_test, y_test = create_sequences(test_df, result['feature_columns'])

# 3. Create loaders
train_loader, val_loader = make_loaders(X_train, y_train, X_val, y_val)

# 4. Model & training
model = TransformerEncoder(input_size=70)
trainer = create_trainer(model, train_loader, val_loader)

# 5. Train
history = trainer.fit(num_epochs=100)

# 6. Evaluate
test_preds = trainer.predict(test_loader)
metrics = compute_metrics(y_test, test_preds)

# 7. Log to MLflow
with MLflowTracker() as tracker:
    tracker.log_metrics(metrics)
    tracker.log_model("model.pt")
```

### Multi-Model Comparison

```bash
# Train all 3 models
python scripts/train_all_models.py \
    --subset FD001 \
    --train-lstm \
    --train-transformer \
    --train-xgboost \
    --use-mlflow

# View comparison in MLflow UI
mlflow ui --backend-store-uri ./mlruns
```

---

##  Key Concepts

### Piece-Wise Linear RUL
- Early life: Constant (flat, not predictive)
- Middle life: Degradation (linear decrease)
- Late life: Rapid failure

**Solution**: Clip RUL at 125 cycles to focus on degradation region

### No Data Leakage
- Fit `StandardScaler` on **train only**
- Apply same scaler to val & test
- No statistics from test set used in training

### Time-Series Causality
- Split each engine chronologically (80/20)
- NOT random train/test split
- Preserves temporal structure

### Sliding Windows
- Create overlapping sequences (stride=1)
- Each window = 30 timesteps
- Separate windows per engine (no mixing)

---

##  Documentation Index

| Document | Purpose |
|----------|---------|
| [TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md) | Deep technical dive |
| [PRODUCTION_README.md](PRODUCTION_README.md) | Quick start guide |
| [architecture.md](docs/architecture.md) | System design |
| Code docstrings | Implementation details |
| This file | Quick reference |

---

##  Troubleshooting

### CUDA Out of Memory
```python
# Reduce batch size in src/config.py
BATCH_SIZE = 128  # from 256
```

### Data Loading Issues
```python
# Check data format
import pandas as pd
df = pd.read_csv("data/CMAPSS/train_FD001.txt", sep=r'\s+', header=None)
print(df.head())
print(df.shape)  # Should be (n_samples, 26)
```

### Model Not Converging
```python
# Check learning rate (too high = divergence, too low = slow)
LEARNING_RATE = 1e-4  # Try smaller

# Increase patience for early stopping
PATIENCE = 30
```

### ONNX Export Fails
```bash
# Install onnxruntime
pip install onnxruntime

# Or specify ONNX opset
export_to_onnx(..., opset_version=11)  # from 12
```

---

## Learning Path

1. **Start here**: [PRODUCTION_README.md](PRODUCTION_README.md)
2. **Understand preprocessing**: `src/preprocessing/features.py`
3. **Explore models**: `src/models/tft_rul.py`
4. **Study training**: `src/mlops/training.py`
5. **Deep dive**: [TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md)
6. **Run experiments**: `python scripts/train_all_models.py`

---

