import streamlit as st
import requests
import json
import time
from pathlib import Path

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════
try:
    API_BASE = st.secrets.get("api_base", "http://127.0.0.1:8000")
except Exception:
    API_BASE = "http://127.0.0.1:8000"

# ═════════════════════════════════════════════════════════════════════════════
# PAGE SETUP
# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Yaqza — Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ═════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS - Hide Streamlit chrome
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stSidebar"] {display: none !important;}
    [data-testid="collapsedControl"] {display: none !important;}
    .block-container { padding: 0 !important; max-width: 100% !important; }
    iframe { border: none; }
</style>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# API PROXY FUNCTIONS (Python side - no CORS issues!)
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=10)
def api_get(path, timeout=15):
    """Make GET request to FastAPI backend"""
    try:
        response = requests.get(f"{API_BASE}{path}", timeout=timeout)
        return response.json() if response.ok else {"error": response.text, "status": response.status_code}
    except Exception as e:
        return {"error": str(e)}

def api_post(path, data, timeout=15):
    """Make POST request to FastAPI backend"""
    try:
        response = requests.post(f"{API_BASE}{path}", json=data, timeout=timeout)
        return response.json() if response.ok else {"error": response.text, "status": response.status_code}
    except Exception as e:
        return {"error": str(e)}

# ═════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═════════════════════════════════════════════════════════════════════════════
if "engine" not in st.session_state:
    st.session_state.engine = "ENG001"
if "model" not in st.session_state:
    st.session_state.model = "ridge"
if "language" not in st.session_state:
    st.session_state.language = "en"  # 'en' or 'ar'
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "history" not in st.session_state:
    st.session_state.history = []
if "engines" not in st.session_state:
    st.session_state.engines = ["ENG001", "ENG002", "ENG003"]
if "comparison" not in st.session_state:
    st.session_state.comparison = None
if "recommendation" not in st.session_state:
    st.session_state.recommendation = None

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR - Streamlit Native (replaces HTML sidebar)
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:14px;margin-bottom:20px;">
        <div style="width:34px;height:34px;border-radius:10px;background:linear-gradient(135deg,#00e5ff,#7c4dff);display:flex;align-items:center;justify-content:center;font-size:18px;color:white;">⚙</div>
        <div>
            <div style="font-size:15px;font-weight:600;color:white;">Yaqza</div>
            <div style="font-size:11px;color:rgba(232,240,255,0.45);">Predictive Maintenance</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Fetch engines from API
    engines_data = api_get("/engines")
    if "engines" in engines_data and engines_data["engines"]:
        st.session_state.engines = [e["engine_id"] for e in engines_data["engines"]]

    st.markdown('<p style="font-size:10px;color:rgba(232,240,255,0.45);text-transform:uppercase;letter-spacing:1.2px;">Engine Selection</p>', unsafe_allow_html=True)

    selected_engine = st.selectbox(
        "Select Engine",
        st.session_state.engines,
        index=st.session_state.engines.index(st.session_state.engine) if st.session_state.engine in st.session_state.engines else 0,
        label_visibility="collapsed"
    )
    st.session_state.engine = selected_engine

    st.markdown("---")

    # Fetch models from API - DYNAMIC MODEL SELECTION
    models_data = api_get("/models")
    available_models = []
    if "models" in models_data:
        for m in models_data["models"]:
            if m.get("available"):
                # Map backend key to frontend key
                key = m["key"]
                if key == "ridge_model":
                    frontend_key = "ridge"
                elif key == "rf_model":
                    frontend_key = "rf"
                elif key == "xgb_model":
                    frontend_key = "xgboost"
                elif key == "ngb_model":
                    frontend_key = "ngboost"
                elif key == "cnn_lstm_model":
                    frontend_key = "cnn_lstm"
                else:
                    frontend_key = key.replace("_model", "")
                available_models.append((frontend_key, m["name"]))

    if not available_models:
        available_models = [("ridge", "Ridge Regression")]

    # Find current model index
    current_model_index = 0
    for i, (key, name) in enumerate(available_models):
        if key == st.session_state.model:
            current_model_index = i
            break

    model_display = st.selectbox(
        "Select Model",
        [m[1] for m in available_models],
        index=current_model_index,
        label_visibility="collapsed"
    )
    model_key = [m[0] for m in available_models if m[1] == model_display][0]
    st.session_state.model = model_key

    st.markdown("---")

    # Language Selector
    st.markdown('<p style="font-size:10px;color:rgba(232,240,255,0.45);text-transform:uppercase;letter-spacing:1.2px;">Language / اللغة</p>', unsafe_allow_html=True)

    lang_options = {"en": "🇬🇧 English", "ar": "🇸🇦 العربية"}
    selected_lang = st.selectbox(
        "Select Language",
        list(lang_options.keys()),
        format_func=lambda x: lang_options[x],
        index=0 if st.session_state.language == "en" else 1,
        label_visibility="collapsed"
    )
    st.session_state.language = selected_lang

    st.markdown("---")

    # Quick Actions
    st.markdown('<p style="font-size:10px;color:rgba(232,240,255,0.45);text-transform:uppercase;letter-spacing:1.2px;">Quick Actions</p>', unsafe_allow_html=True)

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    if st.button("📊 Run Prediction", use_container_width=True):
        with st.spinner("Running prediction..."):
            pred = api_get(f"/predict/{st.session_state.engine}/model/{st.session_state.model}")
            st.session_state.prediction = pred
            # Also fetch history
            hist = api_get(f"/history/{st.session_state.engine}")
            st.session_state.history = hist if isinstance(hist, list) else []
        st.success("Prediction updated!")

    if st.button("📐 Compare Models", use_container_width=True):
        with st.spinner("Running comparison..."):
            comp = api_get(f"/predict/{st.session_state.engine}/compare")
            st.session_state.comparison = comp
        st.success("Comparison done!")

    if st.button("🛠️ Get Recommendation", use_container_width=True):
        with st.spinner("Getting recommendation..."):
            rec = api_get(f"/recommend/{st.session_state.engine}")
            st.session_state.recommendation = rec
        st.success("Recommendation ready!")

    st.markdown("---")

    # API Status
    health = api_get("/health", timeout=5)
    if health.get("status") == "ok":
        st.markdown("""
        <div style="padding:10px 12px;border-radius:10px;background:rgba(0,255,136,.05);border:1px solid rgba(0,255,136,.15);display:flex;align-items:center;gap:8px;font-size:12px;color:#00ff88;">
            🟢 API Connected
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="padding:10px 12px;border-radius:10px;background:rgba(255,69,96,.05);border:1px solid rgba(255,69,96,.15);display:flex;align-items:center;gap:8px;font-size:12px;color:#ff4560;">
            🔴 API Disconnected
        </div>
        """, unsafe_allow_html=True)

    st.caption(f"🔗 {API_BASE}")

# ═════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT AREA - HTML Dashboard (Pure Display, No API Calls)
# ═════════════════════════════════════════════════════════════════════════════

# Prepare data to inject into HTML
prediction_json = json.dumps(st.session_state.prediction) if st.session_state.prediction else "null"
history_json = json.dumps(st.session_state.history) if st.session_state.history else "[]"
comparison_json = json.dumps(st.session_state.comparison) if st.session_state.comparison else "null"
recommendation_json = json.dumps(st.session_state.recommendation) if st.session_state.recommendation else "null"
engine_json = json.dumps(st.session_state.engine)
model_json = json.dumps(st.session_state.model)
language_json = json.dumps(st.session_state.language)

# Read the HTML template
base_dir = Path(__file__).resolve().parent
html_path = base_dir / "yaqza_dashboard_v6_fixed.html"

if not html_path.exists():
    # Try alternative locations
    alt_paths = [
        base_dir.parent / "frontend" / "yaqza_dashboard_v6_fixed.html",
        base_dir.parent / "yaqza_dashboard_v6_fixed.html",
        base_dir / "yaqza_dashboard_v5_fixed.html",
        base_dir / "yaqza_dashboard_v5.html",
    ]
    for p in alt_paths:
        if p.exists():
            html_path = p
            break

if html_path.exists():
    with open(html_path, "r", encoding="utf-8") as f:
        html_template = f.read()

    # Inject data into HTML before rendering
    data_injection = f"""
    <script>
        // Injected by Streamlit - REAL DATA from API
        window.YAQZA_INITIAL_DATA = {{
            engine: {engine_json},
            model: {model_json},
            language: {language_json},
            prediction: {prediction_json},
            history: {history_json},
            comparison: {comparison_json},
            recommendation: {recommendation_json},
            apiBase: "{API_BASE}",
            isStreamlitEmbedded: true
        }};
    </script>
    """

    # Insert after <head> or before </head>
    if "</head>" in html_template:
        html_content = html_template.replace("</head>", data_injection + "</head>")
    else:
        html_content = data_injection + html_template

    # Render HTML component
    import streamlit.components.v1 as components
    components.html(html_content, height=950, scrolling=True)

else:
    st.error("❌ Dashboard HTML file not found! Please ensure 'yaqza_dashboard_v6_fixed.html' exists.")
    st.info(f"💡 Searched in: {base_dir}")

    # Fallback: Show data in native Streamlit
    st.subheader("📊 Current Status (Fallback View)")
    if st.session_state.prediction:
        st.json(st.session_state.prediction)
    else:
        st.info("No prediction data. Use the sidebar buttons to fetch data.")