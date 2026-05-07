"""
Yaqza — Global project configuration.

All tuneable constants live here so training scripts never have
magic numbers scattered through the code.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT_DIR / "data"
CMAPSS_DIR  = DATA_DIR / "CMAPSS"
WEIGHTS_DIR = ROOT_DIR / "model_weights"
REPORTS_DIR = ROOT_DIR / "reports"
MLFLOW_DIR  = ROOT_DIR / "mlruns"

# ── CMAPSS column names ────────────────────────────────────────────────────
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
SETTING_COLUMNS = ["setting_1", "setting_2", "setting_3"]
ALL_COLUMNS = ["unit_id", "cycle"] + SETTING_COLUMNS + SENSOR_COLUMNS

# Sensors with near-zero variance — confirmed by EDA → drop them
SENSORS_TO_DROP = [
    "sensor_1", "sensor_5", "sensor_6",
    "sensor_10", "sensor_16", "sensor_18", "sensor_19",
]
USEFUL_SENSORS = [s for s in SENSOR_COLUMNS if s not in SENSORS_TO_DROP]

# ── Preprocessing ──────────────────────────────────────────────────────────
MAX_RUL       = 125   # Piece-wise linear RUL cap
WINDOW_SIZE   = 30    # Sliding window length (time-steps)
STRIDE        = 1     # Window stride during training
ROLLING_WINS  = [5, 10]  # Rolling-stats window sizes

# ── Training ───────────────────────────────────────────────────────────────
BATCH_SIZE    = 256
EPOCHS        = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
PATIENCE      = 15    # Early-stopping patience (epochs)
RANDOM_SEED   = 42

# ── Model: LSTM ────────────────────────────────────────────────────────────
LSTM_CONFIG = {
    "input_size":   len(USEFUL_SENSORS),
    "hidden_size":  128,
    "num_layers":   3,
    "dropout":      0.3,
    "bidirectional": False,
}

BILSTM_CONFIG = {
    "input_size":   len(USEFUL_SENSORS),
    "hidden_size":  128,
    "num_layers":   2,
    "dropout":      0.3,
    "bidirectional": True,
}

# ── Model: Transformer ─────────────────────────────────────────────────────
TRANSFORMER_CONFIG = {
    "input_size":        len(USEFUL_SENSORS),
    "d_model":           128,
    "nhead":             8,
    "num_encoder_layers": 3,
    "dim_feedforward":   256,
    "dropout":           0.1,
    "max_seq_len":       WINDOW_SIZE,
}

# ── MLflow ─────────────────────────────────────────────────────────────────
MLFLOW_EXPERIMENT_NAME = "yaqza-rul-prediction"
# Use relative path for local MLflow (avoids Windows path issues)
MLFLOW_TRACKING_URI    = "./mlruns"

# ── Serving ────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "data_dir": str(DATA_DIR),
    "models": {
        "lstm_rul":    LSTM_CONFIG,
        "bilstm_rul":  BILSTM_CONFIG,
        "transformer": TRANSFORMER_CONFIG,
        "autoencoder": {},
        "classifier":  {},
    },
    "serving": {"port": 8000},
}
