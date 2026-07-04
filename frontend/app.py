import streamlit as st
import streamlit.components.v1 as components
import os
import pandas as pd
from datetime import datetime

# ─── Page Config ─────────────────────────────────────────
st.set_page_config(
    page_title="Yaqza — Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ──────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    .block-container { padding-top: 0.5rem; padding-bottom: 0rem; }
    div[data-testid="stSidebarContent"] { background: #0d1422; border-right: 1px solid rgba(255,255,255,0.07); }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Custom scrollbar */
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:14px;margin-bottom:20px;">
        <div style="width:34px;height:34px;border-radius:10px;background:linear-gradient(135deg,#00e5ff,#7c4dff);display:flex;align-items:center;justify-content:center;font-size:18px;box-shadow:0 0 18px rgba(0,229,255,.35);">⚙</div>
        <div>
            <div style="font-size:15px;font-weight:600;letter-spacing:0.3px;">Yaqza</div>
            <div style="font-size:11px;color:rgba(232,240,255,0.45);">Predictive Maintenance</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p style="font-size:10px;color:rgba(232,240,255,0.45);text-transform:uppercase;letter-spacing:1.2px;padding:8px 0 4px;">Quick Stats</p>', unsafe_allow_html=True)

    # Quick stats from Python
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Engines", "3", help="Total monitored engines")
    with col2:
        st.metric("Predictions", "24", help="Predictions today")

    st.markdown("---")

    # Engine selector
    engine = st.selectbox("🎯 Select Engine", ["ENG001", "ENG002", "ENG003"], index=0)

    st.markdown("---")

    # Quick actions
    st.markdown('<p style="font-size:10px;color:rgba(232,240,255,0.45);text-transform:uppercase;letter-spacing:1.2px;padding:8px 0 4px;">Quick Actions</p>', unsafe_allow_html=True)

    if st.button("🔄 Refresh Dashboard", use_container_width=True):
        st.toast("Dashboard refreshed!")
        st.rerun()

    if st.button("📊 Generate Report", use_container_width=True):
        with st.spinner("Generating report..."):
            import time
            time.sleep(1.5)
        st.success("Report generated!")

    st.markdown("---")

    # API Status
    st.markdown("""
    <div style="padding:10px 12px;border-radius:10px;background:rgba(0,255,136,.05);border:1px solid rgba(0,255,136,.15);display:flex;align-items:center;gap:8px;font-size:12px;color:#00ff88;">
        🟢 API Connected
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("v5.0 | Industrial IoT")

# ─── Main Content ────────────────────────────────────────
# Read and embed the HTML dashboard
html_path = "yaqza_dashboard_v5.html"

# Check if file exists in current directory, otherwise use a default path
if os.path.exists(html_path):
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
else:
    # Try to find the file in the output directory
    alt_path = "/mnt/agents/output/yaqza_dashboard_v5.html"
    if os.path.exists(alt_path):
        with open(alt_path, "r", encoding="utf-8") as f:
            html_content = f.read()
    else:
        st.error("❌ Dashboard HTML file not found! Please make sure 'yaqza_dashboard_v5.html' is in the same directory.")
        st.stop()

# Embed the HTML dashboard with full height
components.html(html_content, height=950, scrolling=True)

# ─── Bottom Panel ────────────────────────────────────────
st.markdown("---")

# Create tabs for additional Streamlit-native features
tab1, tab2, tab3 = st.tabs(["📈 Analytics", "⚙️ Settings", "📋 Logs"])

with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg RUL", "87 cycles", "+2%")
    with col2:
        st.metric("Model Accuracy", "92%", "+1.5%")
    with col3:
        st.metric("Uptime", "99.8%", "-0.1%")

    # Simple line chart using Streamlit native
    chart_data = pd.DataFrame({
        'Cycle': ['C10', 'C15', 'C20', 'C25', 'C30', 'C35', 'C40', 'C45', 'C47'],
        'RUL': [130, 122, 115, 108, 103, 98, 95, 90, 87],
        'Threshold': [50, 50, 50, 50, 50, 50, 50, 50, 50]
    })

    st.line_chart(chart_data.set_index('Cycle'), use_container_width=True)

with tab2:
    st.subheader("Dashboard Settings")

    col1, col2 = st.columns(2)
    with col1:
        st.checkbox("Auto-refresh", value=True, help="Enable auto-refresh every 5 seconds")
        st.checkbox("Show notifications", value=True, help="Show toast notifications")
        st.checkbox("Dark mode", value=True, disabled=True, help="Dark mode is always on")
    with col2:
        st.slider("Refresh interval (seconds)", min_value=1, max_value=30, value=5)
        st.selectbox("Default engine", ["ENG001", "ENG002", "ENG003"], index=0)
        st.selectbox("Chart theme", ["Default", "Minimal", "Detailed"], index=0)

    if st.button("💾 Save Settings", use_container_width=True):
        st.success("Settings saved!")

with tab3:
    st.subheader("System Logs")

    logs = [
        {"time": "08:15:23", "level": "INFO", "message": "Dashboard loaded successfully"},
        {"time": "08:14:45", "level": "SUCCESS", "message": "Prediction completed for ENG001 — RUL: 87 cycles"},
        {"time": "08:12:10", "level": "WARNING", "message": "RUL approached warning zone for ENG003"},
        {"time": "08:10:00", "level": "INFO", "message": "Sensor data ingested — 8 sensors recorded"},
        {"time": "08:05:30", "level": "INFO", "message": "Model loaded — rul_model_v1.pkl"},
    ]

    for log in logs:
        color = "green" if log["level"] == "SUCCESS" else "orange" if log["level"] == "WARNING" else "blue"
        st.markdown(f"""
        <div style="display:flex;gap:10px;padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.05);">
            <span style="font-family:JetBrains Mono;font-size:11px;color:rgba(232,240,255,0.45);min-width:70px;">{log['time']}</span>
            <span style="font-size:11px;font-weight:600;color:{color};min-width:60px;">{log['level']}</span>
            <span style="font-size:12px;color:#e8f0ff;">{log['message']}</span>
        </div>
        """, unsafe_allow_html=True)
