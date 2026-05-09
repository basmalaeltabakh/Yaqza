"""
Yaqza Production-Grade RUL Prediction Dashboard

A premium SaaS-style AI analytics platform for predictive maintenance.
Includes executive overview, data exploration, model performance,
real-time monitoring, and advanced inference capabilities.

Architecture:
- Modular page system with clear separation of concerns
- Cached data loading and model inference
- Premium glassmorphism UI with dark theme
- Interactive Plotly charts throughout
- Real-time system monitoring
- MLflow experiment tracking integration
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import streamlit as st

# ── Path Setup ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Imports ─────────────────────────────────────────────────────────────────
from src.dashboard.styles import get_custom_css, render_hero
from src.dashboard import pages

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Yaqza — AI Predictive Maintenance Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# STYLING
# ──────────────────────────────────────────────────────────────────────────────

st.markdown(get_custom_css(), unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        render_hero(
            "Yaqza",
            "Aircraft Engine RUL Prediction",
        ),
        unsafe_allow_html=True,
    )

    st.divider()

    page_selection = st.radio(
        "📍 Navigation",
        [
            "🏠 Executive Overview",
            "📊 Dataset Explorer",
            "🔬 Sensor Analytics",
            "🤖 Model Performance",
            "🎯 Prediction Explorer",
            "📡 Real-Time Monitoring",
            "⚕️ System Health",
            "🔄 MLflow Experiments",
            "🔌 API Monitoring",
            "🌳 Feature Analysis",
            "⚡ Failure Analysis",
        ],
        key="page_nav",
    )

    st.divider()

    with st.expander("⚙️ Settings"):
        st.selectbox(
            "Dataset",
            options=["FD001", "FD002", "FD003", "FD004"],
            key="selected_dataset",
            help="Select which CMAPSS failure mode to analyze",
        )

        st.number_input(
            "Window Size",
            min_value=10,
            max_value=100,
            value=30,
            step=5,
            key="window_size",
            help="Sequence length for model input",
        )

        st.selectbox(
            "Model",
            options=["LSTM", "Transformer"],
            key="selected_model",
            help="Choose the model architecture",
        )

    with st.expander("ℹ️ About"):
        st.markdown(
            """
        **Yaqza** is a production-grade predictive maintenance platform using
        deep learning for aircraft engine Remaining Useful Life (RUL) prediction.

        **Key Features:**
        - Interactive data exploration
        - Real-time anomaly detection
        - Advanced model performance analysis
        - Live system monitoring
        - MLflow experiment tracking

        **Dataset:** CMAPSS (Commercial Modular Aero-Propulsion System Simulation)
        """
        )

    st.divider()
    st.caption("🛡️ Yaqza v1.0 — Enterprise AI Platform")

# ──────────────────────────────────────────────────────────────────────────────
# PAGE ROUTING
# ──────────────────────────────────────────────────────────────────────────────

if "🏠 Executive Overview" in page_selection:
    pages.executive_overview()

elif "📊 Dataset Explorer" in page_selection:
    pages.dataset_explorer()

elif "🔬 Sensor Analytics" in page_selection:
    pages.sensor_analytics()

elif "🤖 Model Performance" in page_selection:
    pages.model_performance()

elif "🎯 Prediction Explorer" in page_selection:
    pages.prediction_explorer()

elif "📡 Real-Time Monitoring" in page_selection:
    pages.realtime_monitoring()

elif "⚕️ System Health" in page_selection:
    pages.system_health()

elif "🔄 MLflow Experiments" in page_selection:
    pages.mlflow_experiments()

elif "🔌 API Monitoring" in page_selection:
    pages.api_monitoring()

elif "🌳 Feature Analysis" in page_selection:
    pages.feature_analysis()

elif "⚡ Failure Analysis" in page_selection:
    pages.failure_analysis()
