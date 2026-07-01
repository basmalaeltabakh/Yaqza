# backend/features.py
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from datetime import datetime
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
import warnings
import sys

# ── CRITICAL: Register classes in __main__ for joblib unpickling ───────────
def _register_classes():
    """Register transformer classes in __main__ so joblib can find them"""
    import __main__
    
    if not hasattr(__main__, 'LowVarianceFeaturesRemover'):
        __main__.LowVarianceFeaturesRemover = LowVarianceFeaturesRemover
    if not hasattr(__main__, 'DataFrameMinMaxScaler'):
        __main__.DataFrameMinMaxScaler = DataFrameMinMaxScaler
    if not hasattr(__main__, 'RollTimeSeriesTransformer'):
        __main__.RollTimeSeriesTransformer = RollTimeSeriesTransformer
    if not hasattr(__main__, 'TSFreshFeaturesExtractor'):
        __main__.TSFreshFeaturesExtractor = TSFreshFeaturesExtractor
    if not hasattr(__main__, 'CustomPCA'):
        __main__.CustomPCA = CustomPCA
    if not hasattr(__main__, 'TSFreshFeaturesSelector'):
        __main__.TSFreshFeaturesSelector = TSFreshFeaturesSelector

# ── Imports for ML models (joblib unpickling) ────────────────────────────────
try:
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
except ImportError:
    NGBRegressor = None
    Normal = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    from sklearn.ensemble import RandomForestRegressor
except ImportError:
    RandomForestRegressor = None

try:
    from sklearn.linear_model import Ridge
except ImportError:
    Ridge = None

_tf = None
def _get_tf():
    global _tf
    if _tf is None:
        import tensorflow as tf
        _tf = tf
    return _tf

warnings.filterwarnings('ignore', category=UserWarning)

# ── Detect models directory ───────────────────────────────────────────────────
_POSSIBLE_MODELS_DIRS = [
    Path(__file__).parent / "models",
    Path.cwd() / "models",
    Path(__file__).parent.parent / "models",
]

MODELS_DIR = None
for d in _POSSIBLE_MODELS_DIRS:
    if d.exists():
        MODELS_DIR = d
        print(f"✅ Models directory found: {MODELS_DIR}")
        break

if MODELS_DIR is None:
    raise FileNotFoundError(
        f"Could not find 'models' directory. Tried: {_POSSIBLE_MODELS_DIRS}"
    )

SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
SETTING_COLUMNS = ["setting_1", "setting_2", "setting_3"]
FEATURE_COLS = SETTING_COLUMNS + SENSOR_COLUMNS

WINDOW_SIZE = 30
RUL_UPPER_THRESHOLD = 125


# ═════════════════════════════════════════════════════════════════════════════
# TRANSFORMER CLASSES
# ═════════════════════════════════════════════════════════════════════════════

class LowVarianceFeaturesRemover(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y=None):
        self._selector = VarianceThreshold(threshold=self.threshold)
        self._selector.fit(X)
        self.kept_columns_ = X.columns[self._selector.get_support()].tolist()
        self.dropped_columns_ = [c for c in X.columns if c not in self.kept_columns_]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.kept_columns_]

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


class DataFrameMinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range

    def fit(self, X: pd.DataFrame, y=None):
        self.columns_ = X.columns.tolist()
        self._scaler = MinMaxScaler(feature_range=self.feature_range)
        self._scaler.fit(X.values)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_arr = X[self.columns_].values
        return pd.DataFrame(
            self._scaler.transform(X_arr),
            columns=self.columns_,
            index=X.index
        )

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            self._scaler.inverse_transform(X.values),
            columns=self.columns_,
            index=X.index
        )


class RollTimeSeriesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, window_size: int = 30, min_timeshift: int = 0):
        self.window_size = window_size
        self.min_timeshift = min_timeshift

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_cols_ = [
            c for c in X.columns if c not in ('unit', 'time_cycles', 'RUL')
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        from tsfresh.utilities.dataframe_functions import roll_time_series

        cols = ['unit', 'time_cycles'] + self.feature_cols_
        X_in = X[cols].copy().reset_index(drop=True)
        rolled = roll_time_series(
            X_in,
            column_id='unit',
            column_sort='time_cycles',
            max_timeshift=self.window_size,
            min_timeshift=self.min_timeshift,
            n_jobs=0,
        )
        if 'unit' in rolled.columns:
            rolled = rolled.drop(columns=['unit'])
        return rolled


class TSFreshFeaturesExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, calc=None):
        self.calc = calc

    def fit(self, X=None, y=None):
        return self

    def _clean_features(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.loc[:, ~X.columns.duplicated()]
        X = X.dropna(axis=1, how='all')
        if X.isna().sum().sum() > 0:
            X = X.fillna(0)
        X = X.loc[:, X.std() > 0]
        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        from tsfresh import extract_features
        from tsfresh.feature_extraction import MinimalFCParameters
        import warnings as _w

        calc = self.calc or MinimalFCParameters()
        orig_filter = _w.simplefilter
        _w.simplefilter = lambda *a, **kw: None
        try:
            X_feat = extract_features(
                X,
                column_id='id',
                column_sort='time_cycles',
                default_fc_parameters=calc,
                n_jobs=0,
                show_warnings=False,
                disable_progressbar=True,
            )
        finally:
            _w.simplefilter = orig_filter

        return self._clean_features(X_feat)

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.transform(X)


class TSFreshFeaturesSelector(BaseEstimator, TransformerMixin):
    def __init__(self, selected_columns: list = None):
        self.selected_columns = selected_columns
        self.selected_columns_ = selected_columns if selected_columns is not None else []

    def __setstate__(self, state):
        if 'selected_columns_' not in state:
            state['selected_columns_'] = state.get('selected_columns', [])
        self.__dict__.update(state)

    def fit(self, X: pd.DataFrame, y=None):
        if self.selected_columns is None:
            self.selected_columns_ = X.columns.tolist()
        else:
            self.selected_columns_ = self.selected_columns
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.reindex(columns=self.selected_columns_, fill_value=0)

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


class CustomPCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None, random_state=42, explained_variance_target=None):
        self.n_components = n_components
        self.random_state = random_state
        self.explained_variance_target = explained_variance_target

    def fit(self, X: pd.DataFrame, y=None):
        self.fit_columns_ = X.columns.tolist()
        self.scaler_ = StandardScaler()
        X_sc = self.scaler_.fit_transform(X[self.fit_columns_].values)
        self.pca_ = PCA(
            n_components=self.n_components,
            random_state=self.random_state
        ).fit(X_sc)
        if self.explained_variance_target is not None:
            cum = np.cumsum(self.pca_.explained_variance_ratio_)
            self.n_selected_ = int(np.searchsorted(cum, self.explained_variance_target) + 1)
        else:
            self.n_selected_ = self.pca_.n_components_
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_aligned = X.reindex(columns=self.fit_columns_, fill_value=0)
        X_sc = self.scaler_.transform(X_aligned.values)
        X_pca = self.pca_.transform(X_sc)[:, :self.n_selected_]
        return pd.DataFrame(
            X_pca,
            columns=[f'PC{i+1}' for i in range(X_pca.shape[1])],
            index=X.index
        )


# ═════════════════════════════════════════════════════════════════════════════
# REGISTER CLASSES (after class definitions)
# ═════════════════════════════════════════════════════════════════════════════
_register_classes()


# ═════════════════════════════════════════════════════════════════════════════
# GLOBAL ARTIFACTS CACHE
# ═════════════════════════════════════════════════════════════════════════════

_ARTIFACTS_CACHE = None

def get_cached_artifacts():
    """بيرجع الـ artifacts من الـ cache، لو مش موجود بيحملهم"""
    global _ARTIFACTS_CACHE
    if _ARTIFACTS_CACHE is None:
        _ARTIFACTS_CACHE = load_artifacts()
    return _ARTIFACTS_CACHE

def clear_artifacts_cache():
    """بيClear الـ cache (مفيد في testing)"""
    global _ARTIFACTS_CACHE
    _ARTIFACTS_CACHE = None


# ═════════════════════════════════════════════════════════════════════════════
# ARTIFACT LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_artifacts():
    print(f"Loading artifacts from: {MODELS_DIR}")
    artifacts = {}

    required = {
        "lvr": "lvr.pkl",
        "scaler": "scaler.pkl",
        "selected_features": "selected_features.pkl",
        "pca_step": "pca_step.pkl",
        "ridge_model": "ridge_model.pkl",
    }

    for key, filename in required.items():
        path = MODELS_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Required artifact missing: {path}")
        artifacts[key] = joblib.load(path)
        print(f"  ✅ Loaded: {filename}")

    # tsfresh pipeline or selector
    tsfresh_pipeline_path = MODELS_DIR / "tsfresh_pipeline.pkl"
    tsfresh_selector_path = MODELS_DIR / "tsfresh_selector.pkl"

    if tsfresh_pipeline_path.exists():
        artifacts["tsfresh_pipeline"] = joblib.load(tsfresh_pipeline_path)
        print(f"  ✅ Loaded: tsfresh_pipeline.pkl")
    elif tsfresh_selector_path.exists():
        selector = joblib.load(tsfresh_selector_path)
        from sklearn.pipeline import Pipeline
        artifacts["tsfresh_pipeline"] = Pipeline([
            ('roller', RollTimeSeriesTransformer(window_size=WINDOW_SIZE)),
            ('extractor', TSFreshFeaturesExtractor()),
            ('selector', selector),
        ])
        print(f"  ✅ Loaded: tsfresh_selector.pkl (built pipeline)")
    else:
        raise FileNotFoundError(
            f"Missing tsfresh_pipeline.pkl or tsfresh_selector.pkl in {MODELS_DIR}"
        )

    # Optional models with graceful error handling
    optional_models = {
        "rf_model": "rf_model.pkl",
        "xgb_model": "xgb_model.pkl",
        "ngb_model": "ngb_model.pkl",
    }

    for key, filename in optional_models.items():
        path = MODELS_DIR / filename
        if not path.exists():
            print(f"  ⚠️  Missing (optional): {filename}")
            continue

        try:
            artifacts[key] = joblib.load(path)
            print(f"  ✅ Loaded: {filename}")
        except Exception as e:
            print(f"  ⚠️  Failed to load {filename}: {e}")
            if "ngboost" in str(e).lower():
                print("      → Install with: pip install ngboost")
            elif "xgboost" in str(e).lower():
                print("      → Install with: pip install xgboost")

    # CNN-LSTM
    cnn_path = MODELS_DIR / "cnn_lstm_model.keras"
    if cnn_path.exists():
        try:
            if tf is None:
                raise ImportError("tensorflow not installed")
            artifacts["cnn_lstm_model"] = tf.keras.models.load_model(cnn_path)
            print(f"  ✅ Loaded: cnn_lstm_model.keras")
        except Exception as e:
            print(f"  ⚠️  Failed to load CNN-LSTM: {e}")
    else:
        print(f"  ⚠️  Missing (optional): cnn_lstm_model.keras")

    print(f"\nTotal artifacts loaded: {len(artifacts)}")
    return artifacts


# ═════════════════════════════════════════════════════════════════════════════
# PREPROCESSING & PREDICTION
# ═════════════════════════════════════════════════════════════════════════════

def preprocess_readings(readings: list) -> pd.DataFrame:
    data = []
    for i, r in enumerate(readings):
        row = {
            "unit": 1,
            "time_cycles": getattr(r, 'cycle', i + 1),
        }
        for s in SETTING_COLUMNS:
            val = getattr(r, s, None)
            if val is None and hasattr(r, '__getitem__'):
                val = r.get(s, 0.0)
            row[s] = val or 0.0

        for s in SENSOR_COLUMNS:
            val = getattr(r, s, None)
            if val is None and hasattr(r, '__getitem__'):
                val = r.get(s, 0.0)
            row[s] = val or 0.0

        data.append(row)

    return pd.DataFrame(data)


def run_feature_pipeline(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """بتشغل الـ preprocessing pipeline كامل وترجع الـ PCA features"""
    
    # 1. LVR — شيلي الـ low variance columns
    df_selected = artifacts["lvr"].transform(df[FEATURE_COLS])

    # 2. Scale
    df_scaled = artifacts["scaler"].transform(df_selected)
    df_scaled = df_scaled.copy()
    df_scaled.insert(0, "time_cycles", df["time_cycles"].values)
    df_scaled.insert(0, "unit", df["unit"].values)

    # 3. Roll + Extract TSFresh features
    tsfresh_pipe = artifacts["tsfresh_pipeline"]
    roller = tsfresh_pipe.named_steps.get("roller") or RollTimeSeriesTransformer(window_size=WINDOW_SIZE)
    roller.fit(df_scaled)
    rolled = roller.transform(df_scaled)

    extractor = tsfresh_pipe.named_steps.get("extractor") or TSFreshFeaturesExtractor()
    X_features = extractor.transform(rolled)

    # 4. Selected features — بتعامل معاه سواء كان list أو transformer
    selected = artifacts["selected_features"]
    if hasattr(selected, 'transform'):
        X_selected = selected.transform(X_features)
    else:
        X_selected = X_features.reindex(columns=list(selected), fill_value=0)

    # 5. PCA
    X_pca = artifacts["pca_step"].transform(X_selected)

    return X_pca


def predict_rul_single(readings: list, model_key: str = "ridge_model") -> dict:
    artifacts = get_cached_artifacts()
    df = preprocess_readings(readings)

    if model_key == "cnn_lstm_model":
        return predict_cnn_lstm(readings, artifacts)

    X_pca = run_feature_pipeline(df, artifacts)

    if model_key not in artifacts:
        raise ValueError(f"Model {model_key} not found. Available: {list(artifacts.keys())}")

    model = artifacts[model_key]
    rul_raw = float(model.predict(X_pca.iloc[[-1]])[0])
    rul = max(0.0, round(rul_raw, 2))

    uncertainty = None
    if model_key == "ngb_model" and hasattr(model, 'pred_dist'):
        dist = model.pred_dist(X_pca.iloc[[-1]].values)
        uncertainty = {
            "std": float(dist.scale[0]),
            "ci_90_lower": float(dist.dist.ppf(0.05)[0]),
            "ci_90_upper": float(dist.dist.ppf(0.95)[0]),
        }

    return {"rul": rul, "model": model_key, "uncertainty": uncertainty}


def predict_cnn_lstm(readings: list, artifacts: dict = None) -> dict:
    if artifacts is None:
        artifacts = get_cached_artifacts()

    if "cnn_lstm_model" not in artifacts:
        raise ValueError("CNN-LSTM model not found")

    model = artifacts["cnn_lstm_model"]
    df = preprocess_readings(readings)

    df_selected = artifacts["lvr"].transform(df[FEATURE_COLS])
    df_scaled = artifacts["scaler"].transform(df_selected)

    seq = df_scaled.values

    if len(seq) < WINDOW_SIZE:
        padding = np.zeros((WINDOW_SIZE - len(seq), seq.shape[1]))
        seq = np.vstack([padding, seq])
    else:
        seq = seq[-WINDOW_SIZE:]

    X = np.expand_dims(seq, axis=0)
    rul_raw = float(model.predict(X, verbose=0)[0][0])
    rul = max(0.0, round(rul_raw, 2))

    return {"rul": rul, "model": "cnn_lstm_model", "uncertainty": None}


def predict_all_models(readings: list) -> dict:
    artifacts = get_cached_artifacts()
    results = {}

    tsfresh_models = ["ridge_model", "rf_model", "xgb_model", "ngb_model"]
    df = preprocess_readings(readings)
    X_pca = run_feature_pipeline(df, artifacts)

    for model_key in tsfresh_models:
        if model_key not in artifacts:
            results[model_key] = {"error": "Model not found"}
            continue

        try:
            model = artifacts[model_key]
            rul_raw = float(model.predict(X_pca.iloc[[-1]])[0])
            rul = max(0.0, round(rul_raw, 2))
            result = {"rul": rul, "model": model_key}

            if model_key == "ngb_model" and hasattr(model, 'pred_dist'):
                dist = model.pred_dist(X_pca.iloc[[-1]].values)
                result["uncertainty"] = {
                    "std": round(float(dist.scale[0]), 2),
                    "ci_90": [
                        round(max(0, float(dist.dist.ppf(0.05)[0])), 2),
                        round(float(dist.dist.ppf(0.95)[0]), 2),
                    ]
                }
            results[model_key] = result
        except Exception as e:
            results[model_key] = {"error": str(e)}

    if "cnn_lstm_model" in artifacts:
        try:
            cnn_result = predict_cnn_lstm(readings, artifacts)
            results["cnn_lstm_model"] = cnn_result
        except Exception as e:
            results["cnn_lstm_model"] = {"error": str(e)}
    else:
        results["cnn_lstm_model"] = {"error": "Model not found"}

    successful_ruls = [
        v["rul"] for v in results.values()
        if isinstance(v, dict) and "rul" in v
    ]

    consensus_rul = round(np.mean(successful_ruls), 2) if successful_ruls else None

    status = "NORMAL" if (consensus_rul and consensus_rul >= 50) else \
             "WARNING" if (consensus_rul and consensus_rul >= 20) else "CRITICAL"

    return {
        "predictions": results,
        "consensus_rul": consensus_rul,
        "status": status,
        "n_models_ran": len(successful_ruls),
    }


def predict_from_readings(readings: list) -> dict:
    result = predict_rul_single(readings, model_key="ridge_model")
    rul = result["rul"]

    health_score = round(min(100.0, (rul / RUL_UPPER_THRESHOLD) * 100), 1)

    latest = readings[-1] if readings else None
    if latest:
        sensor_7_vals = [getattr(r, "sensor_7", 0) or 0 for r in readings[-WINDOW_SIZE:]]
        variance_penalty = min(50, np.std(sensor_7_vals) * 10) if sensor_7_vals else 25
        confidence = float(round(max(50.0, 100.0 - variance_penalty), 1))
    else:
        confidence = float(50.0)

    failure_mode = get_failure_mode(readings)
    status = "NORMAL" if rul >= 50 else "WARNING" if rul >= 20 else "CRITICAL"

    return {
        "rul": rul,
        "failure_mode": failure_mode,
        "confidence": confidence,
        "health_score": health_score,
        "status": status,
    }


def get_failure_mode(readings: list) -> str:
    if not readings:
        return "unknown"

    latest = readings[-1]
    s9 = getattr(latest, 'sensor_9', 0) or 0
    s14 = getattr(latest, 'sensor_14', 0) or 0
    s11 = getattr(latest, 'sensor_11', 0) or 0
    s4 = getattr(latest, 'sensor_4', 0) or 0
    s15 = getattr(latest, 'sensor_15', 0) or 0

    if s9 > 9100:
        return "high_pressure_compressor_fault"
    elif s14 > 8200:
        return "turbine_inlet_temperature_high"
    elif s11 > 48:
        return "mechanical_wear"
    elif s4 < 1380:
        return "low_efficiency"
    elif s15 < 8.3:
        return "lubrication_degradation"
    else:
        return "general_degradation"