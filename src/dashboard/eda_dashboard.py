"""
EDA & ML Dashboard for RUL Prediction

Comprehensive visualization of:
- Data exploration and preprocessing
- Model training metrics
- Performance comparisons
- Predictions vs actuals
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pickle
from typing import Optional, Tuple, Dict, List
import warnings
warnings.filterwarnings("ignore")

# ─── Page Configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Yaqza RUL Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom Styling ────────────────────────────────────────────────────────
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ─── Setup Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "CMAPSS"
MODEL_DIR = PROJECT_ROOT / "model_weights"
MLFLOW_DIR = PROJECT_ROOT / "mlruns"

# ─── Streamlit Session State ───────────────────────────────────────────────
@st.cache_resource
def load_data_cache():
    """Load and cache preprocessing data"""
    try:
        from src.preprocessing.features import prepare_cmapss
        from src.preprocessing.windows import make_loaders
        
        subset = "FD001"
        data = prepare_cmapss(DATA_DIR, subset=subset, max_rul=125)
        return data
    except Exception as e:
        st.error(f"❌ Failed to load data: {e}")
        return None

@st.cache_resource
def load_model_metrics():
    """Load training metrics from MLflow"""
    try:
        metrics_file = MLFLOW_DIR / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                return json.load(f)
    except Exception as e:
        st.warning(f"⚠️ Could not load MLflow metrics: {e}")
    return {}

# ─── Header & Navigation ───────────────────────────────────────────────────
st.title("🎯 Yaqza RUL Prediction System")
st.markdown("Remaining Useful Life (RUL) Prediction for Aircraft Engines")

# Sidebar Navigation
page = st.sidebar.radio(
    "📍 Navigation",
    ["📊 EDA Overview", "🔬 Data Analysis", "🤖 Model Training", "📈 Model Comparison", "🎯 Predictions"]
)

# ─── PAGE 1: EDA OVERVIEW ──────────────────────────────────────────────────
if page == "📊 EDA Overview":
    st.header("📊 Data Overview")
    
    data = load_data_cache()
    if data is None:
        st.error("Failed to load data")
    else:
        X_train, X_val, X_test, y_train, y_val, y_test = data
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🚂 Train Samples", f"{len(X_train):,}")
        with col2:
            st.metric("✓ Val Samples", f"{len(X_val):,}")
        with col3:
            st.metric("🧪 Test Samples", f"{len(X_test):,}")
        with col4:
            st.metric("⏱️ Sequence Length", X_train.shape[1] if len(X_train.shape) > 1 else "N/A")
        
        st.markdown("---")
        
        # Data Distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("RUL Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.hist(y_train, bins=30, alpha=0.6, label="Train", color="blue", edgecolor="black")
            ax.hist(y_val, bins=30, alpha=0.6, label="Val", color="green", edgecolor="black")
            ax.hist(y_test, bins=30, alpha=0.6, label="Test", color="red", edgecolor="black")
            
            ax.set_xlabel("RUL (cycles)", fontsize=12, fontweight="bold")
            ax.set_ylabel("Frequency", fontsize=12, fontweight="bold")
            ax.set_title("RUL Distribution Across Datasets", fontsize=14, fontweight="bold")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig, use_container_width=True)
        
        with col2:
            st.subheader("RUL Statistics")
            
            stats_df = pd.DataFrame({
                "Metric": ["Mean", "Median", "Std Dev", "Min", "Max"],
                "Train": [
                    f"{y_train.mean():.2f}",
                    f"{np.median(y_train):.2f}",
                    f"{y_train.std():.2f}",
                    f"{y_train.min():.2f}",
                    f"{y_train.max():.2f}"
                ],
                "Val": [
                    f"{y_val.mean():.2f}",
                    f"{np.median(y_val):.2f}",
                    f"{y_val.std():.2f}",
                    f"{y_val.min():.2f}",
                    f"{y_val.max():.2f}"
                ],
                "Test": [
                    f"{y_test.mean():.2f}",
                    f"{np.median(y_test):.2f}",
                    f"{y_test.std():.2f}",
                    f"{y_test.min():.2f}",
                    f"{y_test.max():.2f}"
                ]
            })
            
            st.dataframe(stats_df, use_container_width=True)
            
            # Summary
            st.markdown("""
            **Key Insights:**
            - ✅ RUL capped at 125 cycles (focus on degradation region)
            - ✅ Piece-wise linear RUL calculation applied
            - ✅ Train/Val/Test split preserves temporal causality
            """)

# ─── PAGE 2: DATA ANALYSIS ─────────────────────────────────────────────────
elif page == "🔬 Data Analysis":
    st.header("🔬 Detailed Data Analysis")
    
    data = load_data_cache()
    if data is None:
        st.error("Failed to load data")
    else:
        X_train, X_val, X_test, y_train, y_val, y_test = data
        
        st.subheader("📈 Sequence Shape Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Train Sequences**")
            st.code(f"Shape: {X_train.shape}\n\nFirst sequence stats:\nMean: {X_train[0].mean():.4f}\nStd: {X_train[0].std():.4f}")
        
        with col2:
            st.write("**Val Sequences**")
            st.code(f"Shape: {X_val.shape}\n\nFirst sequence stats:\nMean: {X_val[0].mean():.4f}\nStd: {X_val[0].std():.4f}")
        
        with col3:
            st.write("**Test Sequences**")
            st.code(f"Shape: {X_test.shape}\n\nFirst sequence stats:\nMean: {X_test[0].mean():.4f}\nStd: {X_test[0].std():.4f}")
        
        st.markdown("---")
        
        # Sequence visualization
        st.subheader("📊 Sample Sequence Visualization")
        
        sample_idx = st.slider("Select sample index", 0, min(10, len(X_train)-1), 0)
        
        fig, axes = plt.subplots(5, 1, figsize=(12, 10))
        
        sample = X_train[sample_idx]
        sensor_subset = [0, 1, 2, 3, 4]  # First 5 sensors
        
        for idx, sensor in enumerate(sensor_subset):
            axes[idx].plot(sample[:, sensor], linewidth=2, color=f"C{idx}")
            axes[idx].set_ylabel(f"Sensor {sensor}", fontsize=10, fontweight="bold")
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_xlim(0, sample.shape[0]-1)
        
        axes[-1].set_xlabel("Time Step", fontsize=12, fontweight="bold")
        fig.suptitle(f"Sample #{sample_idx} - First 5 Sensors Over Time Window (RUL={y_train[sample_idx]:.1f})", 
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        
        st.pyplot(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Feature statistics
        st.subheader("📉 Feature Statistics")
        
        # Compute mean, std, min, max across all train sequences
        mean_vals = X_train.mean(axis=(0, 1))
        std_vals = X_train.std(axis=(0, 1))
        min_vals = X_train.min(axis=(0, 1))
        max_vals = X_train.max(axis=(0, 1))
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Mean
        axes[0, 0].bar(range(len(mean_vals)), mean_vals, color="steelblue", edgecolor="black")
        axes[0, 0].set_title("Mean Values Across Sensors", fontsize=12, fontweight="bold")
        axes[0, 0].set_xlabel("Sensor Index")
        axes[0, 0].set_ylabel("Mean Value")
        axes[0, 0].grid(True, alpha=0.3, axis="y")
        
        # Std
        axes[0, 1].bar(range(len(std_vals)), std_vals, color="coral", edgecolor="black")
        axes[0, 1].set_title("Std Dev Across Sensors", fontsize=12, fontweight="bold")
        axes[0, 1].set_xlabel("Sensor Index")
        axes[0, 1].set_ylabel("Std Dev")
        axes[0, 1].grid(True, alpha=0.3, axis="y")
        
        # Min
        axes[1, 0].bar(range(len(min_vals)), min_vals, color="lightgreen", edgecolor="black")
        axes[1, 0].set_title("Min Values Across Sensors", fontsize=12, fontweight="bold")
        axes[1, 0].set_xlabel("Sensor Index")
        axes[1, 0].set_ylabel("Min Value")
        axes[1, 0].grid(True, alpha=0.3, axis="y")
        
        # Max
        axes[1, 1].bar(range(len(max_vals)), max_vals, color="lightcoral", edgecolor="black")
        axes[1, 1].set_title("Max Values Across Sensors", fontsize=12, fontweight="bold")
        axes[1, 1].set_xlabel("Sensor Index")
        axes[1, 1].set_ylabel("Max Value")
        axes[1, 1].grid(True, alpha=0.3, axis="y")
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

# ─── PAGE 3: MODEL TRAINING ────────────────────────────────────────────────
elif page == "🤖 Model Training":
    st.header("🤖 Model Training Metrics")
    
    st.subheader("Training Progress")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🧠 LSTM", "In Progress ⏳", "-")
    with col2:
        st.metric("⚡ Transformer", "Pending", "-")
    with col3:
        st.metric("🌳 XGBoost", "Pending", "-")
    
    st.markdown("---")
    
    st.info("""
    **Training Status:**
    - ✅ Data loaded and preprocessed
    - ✅ Models initialized
    - ⏳ LSTM training in progress...
    - ⏳ Metrics being logged to MLflow
    
    **Expected Training Time (CPU):**
    - LSTM: ~2-3 hours
    - Transformer: ~1.5-2 hours  
    - XGBoost: ~30-45 minutes
    """)
    
    st.subheader("Model Architectures")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🧠 LSTM RUL
        
        - **Type:** Recurrent Neural Network
        - **Layers:** 3-layer LSTM
        - **Hidden Units:** 64 per layer
        - **Dropout:** 0.3
        - **Attention:** Multi-head attention
        - **Input Norm:** Layer normalization
        - **Loss:** MSE
        - **Optimizer:** Adam (lr=0.001)
        """)
    
    with col2:
        st.markdown("""
        ### ⚡ Transformer Encoder
        
        - **Type:** Attention-based
        - **Layers:** 3 encoder layers
        - **Heads:** 8-head attention
        - **Embedding Dim:** 128
        - **FFN Hidden:** 512
        - **Dropout:** 0.1
        - **Pos. Encoding:** Sinusoidal
        - **Loss:** MSE
        """)
    
    with col3:
        st.markdown("""
        ### 🌳 XGBoost RUL
        
        - **Type:** Gradient Boosting
        - **Trees:** 100 estimators
        - **Max Depth:** 6
        - **Learning Rate:** 0.1
        - **Subsample:** 0.8
        - **Input Shape:** Flattened (420)
        - **Objective:** Regression
        - **Metric:** MSE
        """)

# ─── PAGE 4: MODEL COMPARISON ──────────────────────────────────────────────
elif page == "📈 Model Comparison":
    st.header("📈 Model Performance Comparison")
    
    st.info("⏳ Models are currently training. Metrics will appear here upon completion.")
    
    st.subheader("Expected Metrics Framework")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Standard Metrics
        
        - **RMSE:** Root Mean Squared Error
        - **MAE:** Mean Absolute Error
        - **R²:** Coefficient of Determination
        - **MAPE:** Mean Absolute Percentage Error
        """)
    
    with col2:
        st.markdown("""
        ### Prognostics Metrics (IEEE 1856)
        
        - **Alpha-Lambda Score:** Asymmetric accuracy
        - **Prognostic Horizon:** % of life with <30% error
        - **Cumulative Accuracy:** Integral metric
        - **Safety Index:** Early prediction bonus
        """)
    
    st.markdown("---")
    
    st.subheader("Comparison Template")
    
    comparison_df = pd.DataFrame({
        "Model": ["LSTM", "Transformer", "XGBoost"],
        "RMSE": ["Pending", "Pending", "Pending"],
        "MAE": ["Pending", "Pending", "Pending"],
        "R²": ["Pending", "Pending", "Pending"],
        "Training Time": ["Pending", "Pending", "Pending"]
    })
    
    st.dataframe(comparison_df, use_container_width=True)

# ─── PAGE 5: PREDICTIONS ───────────────────────────────────────────────────
elif page == "🎯 Predictions":
    st.header("🎯 Model Predictions")
    
    st.info("⏳ Models are currently training. Predictions will be available after training completes.")
    
    st.subheader("Prediction Visualization Templates")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Predicted vs Actual
        
        - Scatter plot of predictions
        - Perfect prediction line (y=x)
        - Error distribution analysis
        - Residual plots
        """)
    
    with col2:
        st.markdown("""
        ### Error Analysis
        
        - Absolute error histograms
        - Error by RUL range
        - Early/Late prediction distribution
        - Confidence intervals
        """)
    
    st.markdown("---")
    
    st.subheader("Sample Prediction Framework")
    
    # Create dummy prediction data for visualization
    n_samples = 100
    y_actual = np.random.uniform(0, 125, n_samples)
    y_pred_lstm = y_actual + np.random.normal(0, 15, n_samples)
    y_pred_lstm = np.clip(y_pred_lstm, 0, 125)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Predictions vs Actual
    axes[0].scatter(y_actual, y_pred_lstm, alpha=0.6, s=50, color="steelblue", edgecolor="black")
    axes[0].plot([0, 125], [0, 125], "r--", linewidth=2, label="Perfect Prediction")
    axes[0].set_xlabel("Actual RUL", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Predicted RUL", fontsize=12, fontweight="bold")
    axes[0].set_title("Predicted vs Actual RUL (LSTM)", fontsize=13, fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect("equal")
    
    # Error distribution
    errors = np.abs(y_actual - y_pred_lstm)
    axes[1].hist(errors, bins=30, color="coral", alpha=0.7, edgecolor="black")
    axes[1].axvline(errors.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean Error: {errors.mean():.2f}")
    axes[1].set_xlabel("Absolute Error (cycles)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Frequency", fontsize=12, fontweight="bold")
    axes[1].set_title("Prediction Error Distribution", fontsize=13, fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

# ─── Footer ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
**Yaqza RUL Prediction System** | 
🏭 Aircraft Engine Remaining Useful Life Prediction | 
📊 Advanced ML Dashboard
""")
