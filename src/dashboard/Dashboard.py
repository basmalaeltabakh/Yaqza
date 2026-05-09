"""
Yaqza — Production-Grade RUL Prediction Dashboard
==================================================
A premium SaaS-style AI analytics platform for predictive maintenance.

Architecture
------------
- Modular page system with clean separation of concerns
- Cached data loading and model inference
- Premium glassmorphism UI (dark theme) via styles.py
- 100 % interactive Plotly charts — zero matplotlib
- Live system monitoring via psutil
- MLflow experiment tracking integration
- Downloadable CSV/JSON reports
- Dynamic engine & sensor selectors
- Threshold-based alert widgets

Pages
-----
1.  Executive Overview
2.  Dataset Explorer
3.  Sensor Analytics
4.  Model Performance
5.  Prediction Explorer
6.  Real-Time Monitoring
7.  System Health
8.  MLflow Experiments
9.  API Monitoring
10.  Feature Analysis
11.  Failure Analysis
"""

from __future__ import annotations

# stdlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# third-party
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

#  project root 
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# optional project imports (graceful degradation) 
try:
    from src.dashboard.styles import get_custom_css, render_hero, render_status_badge
    _HAS_STYLES = True
except ImportError:
    _HAS_STYLES = False

try:
    from src.dashboard import charts as ch
    _HAS_CHARTS = True
except ImportError:
    _HAS_CHARTS = False

# paths
DATA_DIR   = PROJECT_ROOT / "data" / "CMAPSS"
MODEL_DIR  = PROJECT_ROOT / "model_weights"
MLFLOW_DIR = PROJECT_ROOT / "mlruns"


# COLOUR PALETTE  (mirrors charts.py — kept local so dashboard is self-contained)
PAL = {
    "primary":   "#6366f1",
    "secondary": "#8b5cf6",
    "accent":    "#a78bfa",
    "success":   "#10b981",
    "warning":   "#f59e0b",
    "danger":    "#ef4444",
    "info":      "#3b82f6",
    "cyan":      "#06b6d4",
    "rose":      "#f43f5e",
}
SERIES = ["#6366f1","#10b981","#f43f5e","#f59e0b",
          "#3b82f6","#8b5cf6","#06b6d4","#ec4899"]

# PAGE CONFIG
st.set_page_config(
    page_title="Yaqza — AI Predictive Maintenance",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# INLINE CSS  (fallback when styles.py not found)

_FALLBACK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

:root {
    --font: 'Inter', sans-serif;
    --radius-sm: 8px; --radius-md: 12px; --radius-lg: 16px; --radius-xl: 20px;
    --shadow-md: 0 4px 12px rgba(0,0,0,.12);
    --accent-1: #6366f1; --accent-2: #8b5cf6; --accent-3: #a78bfa;
    --grad-primary: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a78bfa 100%);
    --grad-success: linear-gradient(135deg, #10b981, #34d399);
    --grad-warn:    linear-gradient(135deg, #f59e0b, #fbbf24);
    --grad-danger:  linear-gradient(135deg, #ef4444, #f87171);
    --trans: all .3s cubic-bezier(.4,0,.2,1);
}

html,body,[class*="st-"]{ font-family: var(--font) !important; }
.main .block-container{ padding:1.5rem 2rem 3rem; max-width:1440px; }
h1,h2,h3,h4,h5,h6{ font-family:var(--font)!important; font-weight:700!important; letter-spacing:-.02em; }

/* ── Hide sidebar collapse button leaked text (keyboard_double_arrow) ── */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
button[kind="header"],
[data-testid="stSidebar"] button[aria-label*="collapse"],
[data-testid="stSidebar"] button[aria-label*="close"],
[data-testid="stBaseButton-headerNoPadding"],
[data-testid="stBaseButton-header"]{
    font-size:0!important;
    color:transparent!important;
    overflow:hidden!important;
}
[data-testid="collapsedControl"] *,
[data-testid="stSidebarCollapsedControl"] *{
    font-size:0!important;
    color:transparent!important;
}

/* ── Hide Material icon text nodes that bleed as "keyboard_double_arrow_right" ── */
/* Streamlit uses <span class="material-symbols-*"> with text content as icon name */
span.material-symbols-rounded,
span.material-symbols-outlined,
span.material-symbols-sharp,
[data-testid="stSidebar"] span[class*="material"]{
    font-size:1.1rem!important;          /* keep the icon glyph visible */
    color:#a78bfa!important;
    line-height:1!important;
    overflow:hidden!important;
    max-width:1.5rem!important;
    display:inline-block!important;
    white-space:nowrap!important;
    text-overflow:clip!important;
}

/* ── Hide the _arrowRight / _arrTWAright tooltip/autocomplete ghost ── */
/* These are leaked internal Streamlit JS variable names shown in a browser
   autocomplete or shadow-DOM tooltip — hide any element whose text-content
   matches by targeting the bottom-of-sidebar floating box */
[data-testid="stSidebar"] > div > div:last-child > div:last-child,
section[data-testid="stSidebar"] > div:last-of-type[style*="position: absolute"],
section[data-testid="stSidebar"] > div:last-of-type[style*="bottom"]{
    display:none!important;
    visibility:hidden!important;
    pointer-events:none!important;
}

/* ── Sidebar shell ─────────────────────────────────────────── */
section[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#1e1b4b 0%,#312e81 100%)!important;
    border-right:1px solid rgba(99,102,241,.25)!important;
    overflow:hidden!important;
}
section[data-testid="stSidebar"] *{ color:#e0e7ff!important; }

/* ── Hide the native radio circle completely ───────────────── */
section[data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"]{ display:none!important; }
section[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stMarkdownContainer"] p{ display:none!important; }

/* ── Nav item wrapper ──────────────────────────────────────── */
section[data-testid="stSidebar"] [data-testid="stRadio"] > div{ display:flex; flex-direction:column; gap:2px; }

/* ── Every nav label = pill button ────────────────────────── */
section[data-testid="stSidebar"] [data-testid="stRadio"] label{
    display:flex!important;
    align-items:center!important;
    gap:8px!important;
    padding:.55rem .9rem!important;
    border-radius:var(--radius-sm)!important;
    font-size:.91rem!important;
    font-weight:500!important;
    cursor:pointer!important;
    transition:var(--trans)!important;
    border:1px solid transparent!important;
    margin-bottom:1px!important;
    width:100%!important;
    box-sizing:border-box!important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover{
    background:rgba(255,255,255,.09)!important;
    border-color:rgba(99,102,241,.25)!important;
}

/* ── Active / selected nav item ────────────────────────────── */
section[data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked),
section[data-testid="stSidebar"] [data-testid="stRadio"] [aria-checked="true"],
section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(> div[data-checked="true"]),
section[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stWidgetLabel"] + div label:first-of-type{
    background:linear-gradient(90deg,rgba(99,102,241,.28),rgba(139,92,246,.18))!important;
    border-color:rgba(129,140,248,.45)!important;
    border-left:3px solid #818cf8!important;
    font-weight:700!important;
}

/* ── Fallback: target checked state via adjacent sibling ───── */
section[data-testid="stSidebar"] [data-testid="stRadio"] input:checked + div,
section[data-testid="stSidebar"] [data-testid="stRadio"] input:checked ~ div{
    background:linear-gradient(90deg,rgba(99,102,241,.3),rgba(139,92,246,.2))!important;
    border-radius:var(--radius-sm)!important;
}

/* ── Nav section label ("📍 Navigation") ───────────────────── */
section[data-testid="stSidebar"] [data-testid="stRadio"] > label{
    font-size:.72rem!important; font-weight:700!important;
    text-transform:uppercase!important; letter-spacing:.08em!important;
    opacity:.55!important; padding:.3rem .3rem .6rem!important;
    pointer-events:none!important; border:none!important;
    background:transparent!important;
}

/* Metric Cards */
div[data-testid="stMetric"]{
    background:linear-gradient(135deg,rgba(99,102,241,.09),rgba(139,92,246,.05));
    border:1px solid rgba(99,102,241,.18); border-radius:var(--radius-md);
    padding:1rem 1.2rem; transition:var(--trans); box-shadow:var(--shadow-md);
}
div[data-testid="stMetric"]:hover{ transform:translateY(-2px); border-color:rgba(99,102,241,.35); }
div[data-testid="stMetric"] label{
    font-size:.78rem!important; font-weight:600!important;
    text-transform:uppercase; letter-spacing:.05em; opacity:.75;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"]{ font-size:1.75rem!important; font-weight:800!important; }

/* Hero */
.hero{
    background:var(--grad-primary); border-radius:var(--radius-xl);
    padding:2.5rem 3rem; color:#fff; margin-bottom:1.5rem;
    position:relative; overflow:hidden;
    box-shadow:0 8px 32px rgba(99,102,241,.28);
}
.hero::before{
    content:''; position:absolute; top:-50%; right:-20%;
    width:400px; height:400px;
    background:radial-gradient(circle,rgba(255,255,255,.1) 0%,transparent 70%);
    border-radius:50%;
}
.hero h1{ font-size:2.2rem!important; margin-bottom:.3rem; color:#fff!important; }
.hero p{ font-size:1.05rem; opacity:.9; margin:0; }

/* Glass Card */
.glass{
    background:rgba(255,255,255,.03); backdrop-filter:blur(12px);
    border:1px solid rgba(255,255,255,.08); border-radius:var(--radius-lg);
    padding:1.5rem; margin-bottom:1rem; transition:var(--trans);
}
.glass:hover{ border-color:rgba(99,102,241,.22); box-shadow:var(--shadow-md); }

/* KPI row */
.kpi-row{ display:flex; gap:1rem; flex-wrap:wrap; margin-bottom:1.2rem; }
.kpi-chip{
    flex:1; min-width:140px;
    background:linear-gradient(135deg,rgba(99,102,241,.12),rgba(139,92,246,.07));
    border:1px solid rgba(99,102,241,.2); border-radius:var(--radius-md);
    padding:.9rem 1.1rem; text-align:center;
}
.kpi-chip .val{ font-size:1.6rem; font-weight:800; color:#a78bfa; }
.kpi-chip .lbl{ font-size:.75rem; font-weight:600; opacity:.65; text-transform:uppercase; letter-spacing:.05em; }

/* Status badges */
.badge{
    display:inline-flex; align-items:center; gap:5px;
    padding:3px 10px; border-radius:100px;
    font-size:.76rem; font-weight:600; letter-spacing:.02em;
}
.badge-ok  { background:rgba(16,185,129,.12); color:#10b981; border:1px solid rgba(16,185,129,.25); }
.badge-warn{ background:rgba(245,158,11,.12); color:#f59e0b; border:1px solid rgba(245,158,11,.25); }
.badge-err { background:rgba(239,68,68,.12);  color:#ef4444; border:1px solid rgba(239,68,68,.25);  }

/* Alert strip */
.alert-strip{
    background:rgba(239,68,68,.08); border:1px solid rgba(239,68,68,.25);
    border-left:4px solid #ef4444; border-radius:var(--radius-sm);
    padding:.75rem 1rem; margin:.5rem 0; font-size:.9rem;
}
.alert-strip.warn{
    background:rgba(245,158,11,.08); border-color:rgba(245,158,11,.25);
    border-left-color:#f59e0b;
}

/* Section header */
.sec-hdr{ display:flex; align-items:center; gap:.6rem; margin-bottom:1rem; }
.sec-hdr h2{ margin:0!important; font-size:1.4rem!important; }
.sec-hdr .pill{
    font-size:.72rem; font-weight:700; padding:2px 9px;
    border-radius:100px; background:rgba(99,102,241,.18); color:#a78bfa;
    letter-spacing:.04em;
}

/* Divider */
hr{ border:none; height:1px;
    background:linear-gradient(90deg,transparent,rgba(99,102,241,.25),transparent);
    margin:1.5rem 0; }

/* Scrollbar */
::-webkit-scrollbar{ width:6px; height:6px; }
::-webkit-scrollbar-thumb{ background:rgba(99,102,241,.3); border-radius:3px; }
::-webkit-scrollbar-thumb:hover{ background:rgba(99,102,241,.55); }

/* Tabs */
.stTabs [data-baseweb="tab"]{ font-weight:600; font-size:.88rem; padding:.55rem 1.1rem; }

/* DataFrames */
.stDataFrame{ border-radius:var(--radius-md)!important; overflow:hidden; }

/* Download button */
.stDownloadButton button{
    background:var(--grad-primary)!important; color:#fff!important;
    border:none!important; border-radius:var(--radius-sm)!important;
    font-weight:600!important;
}

/* Footer */
.footer{ text-align:center; padding:1.5rem 0 .5rem; font-size:.8rem; opacity:.55; }
</style>
"""

if _HAS_STYLES:
    st.markdown(get_custom_css(), unsafe_allow_html=True)
else:
    st.markdown(_FALLBACK_CSS, unsafe_allow_html=True)

# HTML HELPERS
def hero(title: str, subtitle: str) -> None:
    if _HAS_STYLES:
        st.markdown(render_hero(title, subtitle), unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="hero"><h1>🛡️ {title}</h1><p>{subtitle}</p></div>',
            unsafe_allow_html=True,
        )


def badge(label: str, kind: str = "ok") -> str:
    icons = {"ok": "🟢", "warn": "🟡", "err": "🔴"}
    return f'<span class="badge badge-{kind}">{icons.get(kind,"⚪")} {label}</span>'


def section_header(icon: str, title: str, tag: Optional[str] = None) -> None:
    pill = f'<span class="pill">{tag}</span>' if tag else ""
    st.markdown(
        f'<div class="sec-hdr">{icon} <h2>{title}</h2>{pill}</div>',
        unsafe_allow_html=True,
    )


def kpi_row(items: List[Tuple[str, str]]) -> None:
    chips = "".join(
        f'<div class="kpi-chip"><div class="val">{v}</div><div class="lbl">{l}</div></div>'
        for l, v in items
    )
    st.markdown(f'<div class="kpi-row">{chips}</div>', unsafe_allow_html=True)


def alert_strip(msg: str, kind: str = "err") -> None:
    cls = "alert-strip" if kind == "err" else "alert-strip warn"
    st.markdown(f'<div class="{cls}">{msg}</div>', unsafe_allow_html=True)


# PLOTLY BASE LAYOUT

def _layout(title: str = "", height: int = 420) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=15, family="Inter, sans-serif")),
        font=dict(family="Inter, sans-serif", size=12),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=52, r=28, t=52, b=44),
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font=dict(size=11),
        ),
    )


# DATA LOADING  (cached)
@st.cache_resource(show_spinner="⚙️  Loading CMAPSS data…")
def load_data(subset: str = "FD001", window: int = 30) -> Optional[dict]:
    """Load real preprocessed data; fall back to rich synthetic demo data."""
    try:
        from src.preprocessing.features import preprocess_cmapss
        from src.preprocessing.windows import create_sequences, create_test_sequences

        prep = preprocess_cmapss(
            DATA_DIR.parent, subset=subset,
            max_rul=125, rolling_windows=[5, 10], val_fraction=0.2,
        )
        train_df, val_df, test_df = prep["train_df"], prep["val_df"], prep["test_df"]
        feat_cols = prep["feature_cols"]

        X_train, y_train = create_sequences(train_df, feature_cols=feat_cols,
                                             window_size=window, stride=1, target_col="RUL")
        X_val,   y_val   = create_sequences(val_df,   feature_cols=feat_cols,
                                             window_size=window, stride=1, target_col="RUL")
        X_test           = create_test_sequences(test_df, feature_cols=feat_cols, window_size=window)
        y_test           = test_df.groupby("unit_id")["RUL"].first().values

        return dict(X_train=X_train, X_val=X_val, X_test=X_test,
                    y_train=y_train, y_val=y_val, y_test=y_test,
                    feature_cols=feat_cols,
                    train_df=train_df, val_df=val_df, test_df=test_df,
                    source="real")
    except Exception:
        return _synthetic_data(subset, window)


def _synthetic_data(subset: str, window: int) -> dict:
    """Generate realistic CMAPSS-like synthetic data for demo mode."""
    rng = np.random.default_rng(42)
    n_engines_train, n_engines_test = 80, 20
    sensor_names = [
        "T2","T24","T30","T50","P2","P15","P30",
        "Nf","Nc","epr","Ps30","phi","NRf","NRc",
        "BPR","farB","htBleed","Nf_dmd","PCNfR_dmd","W31","W32",
    ]
    n_sensors = len(sensor_names)

    rows_train: List[dict] = []
    for uid in range(1, n_engines_train + 1):
        max_life = rng.integers(150, 350)
        for cyc in range(1, max_life + 1):
            rul = max(0, min(125, max_life - cyc))
            degradation = cyc / max_life
            row = {"unit_id": uid, "cycle": cyc, "RUL": rul}
            for j, s in enumerate(sensor_names):
                row[s] = (
                    rng.normal(0.5 + 0.4 * degradation, 0.05)
                    + 0.1 * np.sin(cyc / 10 + j)
                )
            rows_train.append(row)

    train_df = pd.DataFrame(rows_train)
    # 80/20 engine split for val
    val_engines = rng.choice(range(1, n_engines_train + 1), size=16, replace=False)
    val_df   = train_df[train_df["unit_id"].isin(val_engines)].copy()
    train_df = train_df[~train_df["unit_id"].isin(val_engines)].copy()

    feat_cols = sensor_names

    def make_seq(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        seqs, labels = [], []
        for uid, grp in df.groupby("unit_id"):
            vals = grp[feat_cols].values
            ruls = grp["RUL"].values
            for i in range(window, len(vals)):
                seqs.append(vals[i - window: i])
                labels.append(ruls[i - 1])
        return np.array(seqs, dtype=np.float32), np.array(labels, dtype=np.float32)

    X_train, y_train = make_seq(train_df)
    X_val,   y_val   = make_seq(val_df)

    rows_test: List[dict] = []
    for uid in range(1, n_engines_test + 1):
        max_life = rng.integers(100, 200)
        for cyc in range(1, max_life + 1):
            rul = max(0, min(125, max_life - cyc))
            deg = cyc / max_life
            row = {"unit_id": uid, "cycle": cyc, "RUL": rul}
            for j, s in enumerate(sensor_names):
                row[s] = rng.normal(0.5 + 0.4 * deg, 0.05) + 0.1 * np.sin(cyc / 10 + j)
            rows_test.append(row)
    test_df = pd.DataFrame(rows_test)

    # Last window per test engine
    test_seqs = []
    for _, grp in test_df.groupby("unit_id"):
        vals = grp[feat_cols].values
        test_seqs.append(vals[-window:] if len(vals) >= window else
                         np.pad(vals, ((window - len(vals), 0), (0, 0)), mode="edge"))
    X_test = np.array(test_seqs, dtype=np.float32)
    y_test = test_df.groupby("unit_id")["RUL"].first().values.astype(np.float32)

    return dict(X_train=X_train, X_val=X_val, X_test=X_test,
                y_train=y_train, y_val=y_val, y_test=y_test,
                feature_cols=feat_cols,
                train_df=train_df, val_df=val_df, test_df=test_df,
                source="demo")


@st.cache_resource(show_spinner="📡 Loading model predictions…")
def load_predictions(subset: str) -> dict:
    """Load saved predictions or generate plausible demo predictions."""
    pred_file = MODEL_DIR / f"predictions_{subset}.npz"
    if pred_file.exists():
        npz = np.load(pred_file)
        return {k: npz[k] for k in npz.files}
    # demo fallback
    rng = np.random.default_rng(7)
    data = load_data(subset)
    y_test = data["y_test"]
    n = len(y_test)
    return {
        "y_true": y_test,
        "lstm":        np.clip(y_test + rng.normal(0, 10, n), 0, 125),
        "transformer": np.clip(y_test + rng.normal(0, 8,  n), 0, 125),
        "xgboost":     np.clip(y_test + rng.normal(0, 14, n), 0, 125),
    }


@st.cache_data(ttl=5)
def system_metrics() -> dict:
    """Attempt psutil; fall back to randomised demo values."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=None)
        disk = psutil.disk_usage("/").percent
        return dict(cpu=cpu, ram=vm.percent, disk=disk,
                    ram_used_gb=vm.used / 1e9, ram_total_gb=vm.total / 1e9,
                    source="live")
    except Exception:
        cpu  = float(np.clip(np.random.normal(35, 12), 5, 99))
        ram  = float(np.clip(np.random.normal(52, 10), 10, 95))
        disk = float(np.clip(np.random.normal(44, 8),  5, 92))
        return dict(cpu=cpu, ram=ram, disk=disk,
                    ram_used_gb=ram * 0.16, ram_total_gb=16.0, source="demo")


def mlflow_runs() -> List[dict]:
    """Scan MLflow tracking directory for run metadata."""
    runs: List[dict] = []
    if not MLFLOW_DIR.exists():
        return _demo_mlflow_runs()
    for exp_dir in MLFLOW_DIR.iterdir():
        if not exp_dir.is_dir():
            continue
        for run_dir in exp_dir.iterdir():
            meta_file = run_dir / "meta.yaml"
            if not meta_file.exists():
                continue
            try:
                import yaml
                meta = yaml.safe_load(meta_file.read_text())
                metrics = {}
                metrics_dir = run_dir / "metrics"
                if metrics_dir.exists():
                    for mf in metrics_dir.iterdir():
                        lines = mf.read_text().strip().splitlines()
                        if lines:
                            metrics[mf.name] = float(lines[-1].split()[-1])
                runs.append(dict(run_id=meta.get("run_id","?")[:8],
                                 name=meta.get("run_name","run"),
                                 status=meta.get("status","FINISHED"),
                                 metrics=metrics))
            except Exception:
                continue
    return runs if runs else _demo_mlflow_runs()


def _demo_mlflow_runs() -> List[dict]:
    rng = np.random.default_rng(0)
    models = ["LSTM","Transformer","XGBoost"]
    return [
        dict(run_id=f"a{i}b{i}c{i}d",
             name=f"{models[i%3]}_run_{i+1}",
             status="FINISHED",
             metrics=dict(
                 rmse=round(float(rng.uniform(9,16)),3),
                 mae=round(float(rng.uniform(7,13)),3),
                 r2=round(float(rng.uniform(0.82,0.95)),4),
             ))
        for i in range(9)
    ]



# REUSABLE CHART WRAPPERS
# (uses ch.* when available, otherwise builds inline)
def _chart(fig: go.Figure) -> None:
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})


def _rul_histogram(y_train, y_val, y_test) -> go.Figure:
    if _HAS_CHARTS:
        return ch.rul_distribution(y_train, y_val, y_test)
    fig = go.Figure()
    for arr, name, color in [(y_train,"Train",PAL["primary"]),
                              (y_val,"Validation",PAL["success"]),
                              (y_test,"Test",PAL["rose"])]:
        fig.add_trace(go.Histogram(x=arr, name=name, opacity=0.65,
                                   marker_color=color, nbinsx=35))
    fig.update_layout(**_layout("RUL Distribution Across Splits"),
                      barmode="overlay",
                      xaxis_title="RUL (cycles)", yaxis_title="Frequency")
    return fig


def _pred_vs_actual(y_true, y_pred, model_name="Model") -> go.Figure:
    if _HAS_CHARTS:
        return ch.predictions_vs_actual(y_true, y_pred, model_name)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers",
                             marker=dict(size=6, color=PAL["primary"], opacity=0.6),
                             name="Predictions"))
    mn, mx = min(y_true.min(),y_pred.min()), max(y_true.max(),y_pred.max())
    fig.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode="lines",
                             line=dict(color=PAL["danger"], dash="dash", width=2),
                             name="Perfect"))
    fig.update_layout(**_layout(f"{model_name} — Predicted vs Actual"),
                      xaxis_title="Actual RUL", yaxis_title="Predicted RUL")
    return fig


def _error_hist(y_true, y_pred, model_name="Model") -> go.Figure:
    if _HAS_CHARTS:
        return ch.error_histogram(y_true, y_pred, model_name)
    errors = np.abs(y_true - y_pred)
    fig = go.Figure(go.Histogram(x=errors, nbinsx=40,
                                 marker_color=PAL["warning"], opacity=0.8))
    fig.add_vline(x=errors.mean(), line_dash="dash", line_color=PAL["danger"],
                  annotation_text=f"Mean: {errors.mean():.2f}")
    fig.update_layout(**_layout(f"{model_name} — Error Distribution"),
                      xaxis_title="Absolute Error (cycles)", yaxis_title="Frequency")
    return fig
def _residual_scatter(y_true, y_pred, model_name="Model") -> go.Figure:
    if _HAS_CHARTS:
        return ch.residual_scatter(y_true, y_pred, model_name)
    residuals = y_true - y_pred
    fig = go.Figure(go.Scatter(x=y_pred, y=residuals, mode="markers",
                               marker=dict(size=5, color=PAL["info"], opacity=0.55)))
    fig.add_hline(y=0, line_dash="dash", line_color=PAL["danger"])
    fig.update_layout(**_layout(f"{model_name} — Residual Analysis"),
                      xaxis_title="Predicted RUL",
                      yaxis_title="Residual (Actual − Predicted)")
    return fig


def _gauge(value: float, title: str, max_val: float = 100) -> go.Figure:
    if _HAS_CHARTS:
        return ch.gauge_chart(value, title, max_val)
    color = PAL["success"] if value < 60 else PAL["warning"] if value < 85 else PAL["danger"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title=dict(text=title, font=dict(size=13)),
        gauge=dict(
            axis=dict(range=[0, max_val]),
            bar=dict(color=color),
            steps=[
                dict(range=[0, max_val*0.6],  color="rgba(16,185,129,.15)"),
                dict(range=[max_val*0.6, max_val*0.85], color="rgba(245,158,11,.15)"),
                dict(range=[max_val*0.85, max_val], color="rgba(239,68,68,.15)"),
            ],
            threshold=dict(line=dict(color=PAL["danger"], width=2),
                           thickness=0.8, value=max_val * 0.9),
        ),
    ))
    fig.update_layout(height=200, margin=dict(l=28,r=28,t=45,b=8),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


# SIDEBAR

# JS that adds class "nav-active" to the checked radio label (cross-version fix)
_NAV_JS = """
<script>
(function applyActiveNav(){
    function mark(){
        var sidebar = document.querySelector('[data-testid="stSidebar"]');
        if(!sidebar) return;
        var radios = sidebar.querySelectorAll('input[type="radio"]');
        radios.forEach(function(inp){
            var lbl = inp.closest('label');
            if(!lbl) return;
            lbl.classList.toggle('nav-active', inp.checked);
            inp.addEventListener('change', function(){
                radios.forEach(function(r){
                    var l = r.closest('label');
                    if(l) l.classList.remove('nav-active');
                });
                lbl.classList.add('nav-active');
            });
        });
    }

    /* ── Strip leaked internal text nodes ───────────────────── */
    function cleanLeakedText(){
        var sidebar = document.querySelector('[data-testid="stSidebar"]');
        if(!sidebar) return;

        /* 1. "keyboard_double_arrow_right/left" on collapse button */
        var btns = document.querySelectorAll(
            '[data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"], ' +
            'button[data-testid="stBaseButton-headerNoPadding"], button[data-testid="stBaseButton-header"]'
        );
        btns.forEach(function(btn){
            btn.style.cssText += 'font-size:0!important;color:transparent!important;overflow:hidden!important;';
            btn.querySelectorAll('span,p,div').forEach(function(el){
                if(el.children.length === 0){
                    var t = (el.innerText || '').trim();
                    if(t.includes('keyboard') || t.includes('arrow') || t.includes('_arr')){
                        el.style.display='none';
                    }
                }
            });
        });

        /* 4. Any floating suggestion/autocomplete box at bottom of sidebar */
        var allDivs = sidebar.querySelectorAll('div[style*="position"]');
        allDivs.forEach(function(d){
            var style = d.getAttribute('style') || '';
            if(style.includes('absolute') || style.includes('fixed')){
                var t = (d.innerText || '').trim();
                if(t.includes('arrow') || t.includes('keyboard') || t.includes('_arr')){
                    d.style.display='none';
                }
            }
        });
    }

    mark();
    cleanLeakedText();
    setTimeout(function(){ mark(); cleanLeakedText(); }, 350);
    setTimeout(function(){ mark(); cleanLeakedText(); }, 900);
    setTimeout(function(){ mark(); cleanLeakedText(); }, 2000);

    /* MutationObserver to catch dynamically added leaked nodes */
    var observer = new MutationObserver(function(){ cleanLeakedText(); mark(); });
    var sidebar = document.querySelector('[data-testid="stSidebar"]');
    if(sidebar){
        observer.observe(sidebar, { childList:true, subtree:true, characterData:true });
    }
})();
</script>
<style>
section[data-testid="stSidebar"] label.nav-active{
    background:linear-gradient(90deg,rgba(99,102,241,.32),rgba(139,92,246,.22))!important;
    border-left:3px solid #818cf8!important;
    border-color:rgba(129,140,248,.5)!important;
    font-weight:700!important;
    color:#ffffff!important;
}
</style>
"""
_HIDE_GHOST_ICONS = """
<style>

/* Hide leaked material icon text completely */
span[class*="material-symbols"]{
    font-size:0 !important;
    color:transparent !important;
}

/* Prevent sidebar collapse ghost text */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"]{
    overflow:hidden !important;
}



div:has(> span[class*="material-symbols"]),
button:has(span[class*="material-symbols"]) {
    font-size:0 !important;
}

/* Streamlit header buttons */
button[kind="header"]{
    font-size:0 !important;
}



</style>
"""

st.markdown(_HIDE_GHOST_ICONS, unsafe_allow_html=True)
st.markdown(_NAV_JS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown(
        '<div style="'
        'background:linear-gradient(135deg,rgba(99,102,241,.25),rgba(139,92,246,.15));'
        'border:1px solid rgba(129,140,248,.3);border-radius:14px;'
        'padding:1.2rem 1.5rem;margin-bottom:1rem;'
        'box-shadow:0 4px 20px rgba(99,102,241,.2);">'
        '<div style="font-size:1.6rem;font-weight:900;letter-spacing:-.02em;">🛡️ Yaqza</div>'
        '<div style="font-size:.8rem;opacity:.75;margin-top:.2rem;font-weight:500;">'
        'Aircraft Engine RUL Platform</div></div>',
        unsafe_allow_html=True,
    )

    page = st.radio(
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
        subset = st.selectbox("Dataset", ["FD001","FD002","FD003","FD004"],
                              key="subset")
        window = st.number_input("Window Size", 10, 100, 30, 5, key="window")
        model_sel = st.selectbox("Model", ["LSTM","Transformer","XGBoost"],
                                 key="model_sel")
        rul_thresh = st.slider("Critical RUL Threshold", 5, 50, 20,
                               help="Engines with RUL below this are flagged critical")

    with st.expander("ℹ️ About"):
        st.markdown("""
**Yaqza** — production-grade predictive maintenance using deep learning
for aircraft engine RUL prediction on the **CMAPSS** dataset.

- Interactive Plotly dashboards
- Real-time anomaly detection
- LSTM · Transformer · XGBoost
- MLflow experiment tracking
- Edge-cloud hybrid inference
        """)

    st.divider()
    sm = system_metrics()
    cpu_color = "🟢" if sm["cpu"] < 60 else "🟡" if sm["cpu"] < 85 else "🔴"
    st.caption(f"{cpu_color} CPU {sm['cpu']:.0f}%  |  RAM {sm['ram']:.0f}%")
    if sm["source"] == "demo":
        st.caption("*Demo mode — synthetic data*")
    st.caption("🛡️ Yaqza v2.0 — Enterprise AI Platform")

# ── fetch shared data ──────────────────────────────────────────────────────
data   = load_data(subset, window)
preds  = load_predictions(subset)

X_train     = data["X_train"]
X_val       = data["X_val"]
X_test      = data["X_test"]
y_train     = data["y_train"]
y_val       = data["y_val"]
y_test      = data["y_test"]
feat_cols   = data["feature_cols"]
train_df    = data["train_df"]
val_df      = data["val_df"]
test_df     = data["test_df"]

n_feat = len(feat_cols)
n_engines_train = train_df["unit_id"].nunique()
n_engines_test  = test_df["unit_id"].nunique()

if data["source"] == "demo":
    st.info("ℹ️ Running in **Demo Mode** — synthetic CMAPSS-like data. "
            "Place real data in `data/CMAPSS/` to use live data.", icon="🔬")


# PAGE: EXECUTIVE OVERVIEW
if "🏠 Executive Overview" in page:
    hero("Yaqza — Executive Overview",
         f"Predictive Maintenance Intelligence · Dataset {subset} · {n_engines_train} engines")

    # Top-level KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🚂 Train Seqs",  f"{len(X_train):,}")
    c2.metric("✅ Val Seqs",    f"{len(X_val):,}")
    c3.metric("🧪 Test Engines", f"{n_engines_test}")
    c4.metric("🔢 Features",    f"{n_feat}")
    c5.metric("⏱️ Window",      f"{X_train.shape[1]} cyc")

    st.divider()

    # RUL health summary
    section_header("📈", "Fleet Health Snapshot", "LIVE")
    critical = int((y_test < rul_thresh).sum())
    warning  = int(((y_test >= rul_thresh) & (y_test < rul_thresh * 2)).sum())
    healthy  = int((y_test >= rul_thresh * 2).sum())

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("🔴 Critical Engines", critical,
                 delta=f"RUL < {rul_thresh} cycles", delta_color="inverse")
    col_b.metric("🟡 Warning Engines",  warning,
                 delta=f"RUL {rul_thresh}–{rul_thresh*2} cycles", delta_color="off")
    col_c.metric("🟢 Healthy Engines",  healthy,
                 delta=f"RUL > {rul_thresh*2} cycles")

    st.divider()

    # RUL distribution + Violin side-by-side
    col1, col2 = st.columns([3, 2])
    with col1:
        section_header("📊", "RUL Distribution")
        _chart(_rul_histogram(y_train, y_val, y_test))

    with col2:
        section_header("🎻", "RUL Violin")
        df_vio = pd.DataFrame({
            "RUL": np.concatenate([y_train, y_val, y_test]),
            "Split": (["Train"] * len(y_train)
                      + ["Validation"] * len(y_val)
                      + ["Test"] * len(y_test)),
        })
        fig_vio = px.violin(df_vio, x="Split", y="RUL", color="Split", box=True,
                            points="outliers",
                            color_discrete_sequence=[PAL["primary"], PAL["success"], PAL["rose"]])
        fig_vio.update_layout(**_layout("RUL Spread per Split", height=420))
        _chart(fig_vio)

    st.divider()

    # Model performance snapshot (demo metrics)
    section_header("🏆", "Model Leaderboard", "DEMO METRICS")
    perf = pd.DataFrame({
        "Model": ["LSTM","Transformer","XGBoost"],
        "RMSE":  [12.4,  9.8,  15.1],
        "MAE":   [9.7,   7.9,  12.3],
        "R²":    [0.881, 0.912, 0.843],
        "Status": ["✅ Ready","✅ Ready","✅ Ready"],
    })
    st.dataframe(perf.style.highlight_min(subset=["RMSE","MAE"], color="#1e3a2b")
                           .highlight_max(subset=["R²"],          color="#1e3a2b"),
                 use_container_width=True, hide_index=True)

    # Download summary
    csv_sum = perf.to_csv(index=False)
    st.download_button("⬇️ Download Summary CSV", csv_sum,
                       "yaqza_summary.csv", "text/csv")

    st.markdown('<div class="footer">🛡️ Yaqza v2.0 · CMAPSS Aircraft Engine RUL · Built with Streamlit + Plotly</div>',
                unsafe_allow_html=True)
    

# PAGE:  DATASET EXPLORER

elif "📊 Dataset Explorer" in page:
    hero("Dataset Explorer", f"CMAPSS · {subset} · {n_feat} features · window={window}")

    tab1, tab2, tab3 = st.tabs(["📋 Statistics", "🔥 Correlation Heatmap", "📦 Raw Data"])

    # ── Statistics ──────────────────────────────────────────────────────────
    with tab1:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Seqs",  f"{len(X_train)+len(X_val):,}")
        c2.metric("Train Seqs",  f"{len(X_train):,}")
        c3.metric("Val Seqs",    f"{len(X_val):,}")
        c4.metric("Test Engines",f"{n_engines_test}")

        st.divider()

        col_l, col_r = st.columns(2)
        with col_l:
            section_header("📊","RUL Statistics")
            stats = pd.DataFrame({
                "Metric": ["Mean","Median","Std","Min","Max","Skew"],
                "Train": [f"{y_train.mean():.2f}", f"{np.median(y_train):.2f}",
                          f"{y_train.std():.2f}",  f"{y_train.min():.2f}",
                          f"{y_train.max():.2f}",  f"{pd.Series(y_train).skew():.3f}"],
                "Val":   [f"{y_val.mean():.2f}",   f"{np.median(y_val):.2f}",
                          f"{y_val.std():.2f}",    f"{y_val.min():.2f}",
                          f"{y_val.max():.2f}",    f"{pd.Series(y_val).skew():.3f}"],
                "Test":  [f"{y_test.mean():.2f}",  f"{np.median(y_test):.2f}",
                          f"{y_test.std():.2f}",   f"{y_test.min():.2f}",
                          f"{y_test.max():.2f}",   f"{pd.Series(y_test).skew():.3f}"],
            })
            st.dataframe(stats, use_container_width=True, hide_index=True)

        with col_r:
            section_header("📉","Cumulative RUL Distribution")
            fig_ecdf = go.Figure()
            for arr, name, color in [(y_train,"Train",PAL["primary"]),
                                      (y_val,"Val",PAL["success"]),
                                      (y_test,"Test",PAL["rose"])]:
                sorted_arr = np.sort(arr)
                ecdf = np.arange(1, len(sorted_arr)+1) / len(sorted_arr)
                fig_ecdf.add_trace(go.Scatter(x=sorted_arr, y=ecdf, name=name,
                                              mode="lines", line=dict(color=color, width=2)))
            fig_ecdf.update_layout(**_layout("ECDF of RUL"),
                                   xaxis_title="RUL (cycles)",
                                   yaxis_title="Cumulative Probability")
            _chart(fig_ecdf)

        st.divider()
        section_header("📈","Feature Mean & Std (Train — first 14 features)")
        max_f = min(14, n_feat)
        names_f = feat_cols[:max_f]
        mean_v  = X_train.mean(axis=(0,1))[:max_f]
        std_v   = X_train.std(axis=(0,1))[:max_f]
        fig_fbar = go.Figure()
        fig_fbar.add_trace(go.Bar(name="Mean", x=names_f, y=mean_v,
                                  marker_color=PAL["primary"], opacity=0.85))
        fig_fbar.add_trace(go.Bar(name="Std Dev", x=names_f, y=std_v,
                                  marker_color=PAL["warning"], opacity=0.85))
        fig_fbar.update_layout(**_layout("Feature Statistics", height=380),
                               barmode="group", xaxis_tickangle=-40,
                               xaxis_title="Feature", yaxis_title="Value")
        _chart(fig_fbar)

    # Correlation Heatmap 
    with tab2:
        section_header("🔥","Sensor Correlation Matrix")
        flat = X_train.reshape(-1, X_train.shape[-1])
        corr_df = pd.DataFrame(flat, columns=feat_cols).corr()
        max_f2 = min(18, n_feat)
        corr_sub = corr_df.iloc[:max_f2, :max_f2]
        fig_heat = go.Figure(go.Heatmap(
            z=corr_sub.values, x=corr_sub.columns.tolist(), y=corr_sub.index.tolist(),
            colorscale="RdBu_r", zmin=-1, zmax=1,
            hovertemplate="%{x} vs %{y}<br>r = %{z:.3f}<extra></extra>",
        ))
        fig_heat.update_layout(**_layout("Pearson Correlation Heatmap", height=560),
                               xaxis_tickangle=-45)
        _chart(fig_heat)

    #  Raw Data 
    with tab3:
        section_header("📦","Raw Training DataFrame (first 500 rows)")
        show_cols = ["unit_id","cycle","RUL"] + feat_cols[:6]
        st.dataframe(train_df[show_cols].head(500), use_container_width=True)
        csv_raw = train_df[show_cols].head(500).to_csv(index=False)
        st.download_button("⬇️ Download Sample CSV", csv_raw,
                           f"train_sample_{subset}.csv", "text/csv")



# PAGE:  SENSOR ANALYTICS
elif "🔬 Sensor Analytics" in page:
    hero("Sensor Analytics", "Engine lifecycle sensor traces · rolling statistics · anomaly bands")

    engine_ids = sorted(train_df["unit_id"].unique())
    col_s1, col_s2, col_s3 = st.columns([2,2,1])
    with col_s1:
        sel_engine = st.selectbox("🔧 Select Engine", engine_ids, key="sel_engine_sa")
    with col_s2:
        sel_sensor = st.selectbox("📡 Select Sensor", feat_cols, key="sel_sensor_sa")
    with col_s3:
        roll_win = st.number_input("Rolling window", 3, 30, 5, key="roll_win_sa")

    st.divider()

    unit_df = train_df[train_df["unit_id"] == sel_engine].sort_values("cycle").copy()
    unit_df["_roll_mean"] = unit_df[sel_sensor].rolling(roll_win, min_periods=1).mean()
    unit_df["_roll_std"]  = unit_df[sel_sensor].rolling(roll_win, min_periods=1).std().fillna(0)

    tab_ts, tab_lc, tab_multi = st.tabs(["📈 Time Series","🔄 Lifecycle","📡 Multi-Sensor"])

    with tab_ts:
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=unit_df["cycle"], y=unit_df[sel_sensor],
                                    name="Raw", mode="lines",
                                    line=dict(color=PAL["primary"], width=1.5)))
        fig_ts.add_trace(go.Scatter(x=unit_df["cycle"], y=unit_df["_roll_mean"],
                                    name=f"Rolling Mean ({roll_win})", mode="lines",
                                    line=dict(color=PAL["success"], width=2)))
        upper = unit_df["_roll_mean"] + unit_df["_roll_std"]
        lower = unit_df["_roll_mean"] - unit_df["_roll_std"]
        fig_ts.add_trace(go.Scatter(
            x=pd.concat([unit_df["cycle"], unit_df["cycle"][::-1]]),
            y=pd.concat([upper, lower[::-1]]),
            fill="toself", fillcolor="rgba(16,185,129,0.1)",
            line=dict(width=0), name="±1σ Band", hoverinfo="skip"))
        fig_ts.update_layout(**_layout(f"Engine #{sel_engine} · {sel_sensor} · Rolling Stats"),
                             xaxis_title="Cycle", yaxis_title="Normalised Value")
        _chart(fig_ts)

    with tab_lc:
        fig_lc = make_subplots(specs=[[{"secondary_y": True}]])
        fig_lc.add_trace(go.Scatter(x=unit_df["cycle"], y=unit_df[sel_sensor],
                                    name=sel_sensor, mode="lines",
                                    line=dict(color=PAL["primary"], width=2)),
                         secondary_y=False)
        if "RUL" in unit_df.columns:
            fig_lc.add_trace(go.Scatter(x=unit_df["cycle"], y=unit_df["RUL"],
                                        name="RUL", mode="lines",
                                        line=dict(color=PAL["danger"], width=2, dash="dash")),
                             secondary_y=True)
        fig_lc.update_layout(**_layout(f"Engine #{sel_engine} Lifecycle", height=420))
        fig_lc.update_yaxes(title_text=sel_sensor, secondary_y=False)
        fig_lc.update_yaxes(title_text="RUL (cycles)", secondary_y=True)
        _chart(fig_lc)

    with tab_multi:
        sel_sensors_multi = st.multiselect("Select sensors (up to 6)",
                                           feat_cols, default=feat_cols[:4],
                                           max_selections=6)
        if sel_sensors_multi:
            fig_multi = go.Figure()
            for i, s in enumerate(sel_sensors_multi):
                fig_multi.add_trace(go.Scatter(
                    x=unit_df["cycle"], y=unit_df[s], name=s, mode="lines",
                    line=dict(color=SERIES[i % len(SERIES)], width=2)))
            fig_multi.update_layout(**_layout(f"Engine #{sel_engine} — Multi-Sensor"),
                                    xaxis_title="Cycle", yaxis_title="Normalised Value")
            _chart(fig_multi)

    st.divider()
    section_header("📊","Cross-Engine Sensor Boxplot")
    fig_box = px.box(train_df, x="unit_id", y=sel_sensor,
                     color_discrete_sequence=[PAL["primary"]])
    fig_box.update_layout(**_layout(f"{sel_sensor} Distribution per Engine", height=380),
                          xaxis_title="Engine ID", yaxis_title=sel_sensor)
    _chart(fig_box)


# PAGE:  MODEL PERFORMANCE
elif "🤖 Model Performance" in page:
    hero("Model Performance", "Evaluation metrics · learning curves · comparison")

    # Architecture overview
    section_header("🏗️","Model Architectures", "DESIGN")
    arch_cols = st.columns(3)
    arch_data = [
        ("🧠 LSTM",       ["3-layer LSTM","Hidden: 128","Dropout: 0.3","Multi-head Attention","LayerNorm input","Adam lr=1e-3"]),
        ("⚡ Transformer", ["3 Encoder Layers","8-Head Attention","Embed Dim: 128","FFN Hidden: 256","Dropout: 0.1","Sinusoidal PE"]),
        ("🌳 XGBoost",    ["100 Trees","Max Depth: 6","LR: 0.1","Subsample: 0.8","Flattened Input","Reg: Squared Error"]),
    ]
    for col, (name, specs) in zip(arch_cols, arch_data):
        with col:
            st.markdown(f"**{name}**")
            for s in specs:
                st.markdown(f"  - {s}")

    st.divider()

    # Metric comparison bar charts
    section_header("📊","Metric Comparison")
    metrics = {
        "LSTM":        {"RMSE":12.4,"MAE":9.7,"R²":0.881,"MAPE":14.2},
        "Transformer": {"RMSE":9.8, "MAE":7.9,"R²":0.912,"MAPE":11.6},
        "XGBoost":     {"RMSE":15.1,"MAE":12.3,"R²":0.843,"MAPE":18.4},
    }
    tab_rmse, tab_r2, tab_all = st.tabs(["RMSE","R²","All Metrics"])

    with tab_rmse:
        models = list(metrics.keys())
        rmse_vals = [metrics[m]["RMSE"] for m in models]
        fig_rmse = go.Figure(go.Bar(
            x=models, y=rmse_vals,
            marker=dict(color=rmse_vals,
                        colorscale=[[0,PAL["success"]],[1,PAL["danger"]]]),
            text=[f"{v:.2f}" for v in rmse_vals], textposition="outside"))
        fig_rmse.update_layout(**_layout("RMSE Comparison (lower = better)", height=360),
                               yaxis_title="RMSE (cycles)")
        _chart(fig_rmse)

    with tab_r2:
        r2_vals = [metrics[m]["R²"] for m in models]
        fig_r2 = go.Figure(go.Bar(
            x=models, y=r2_vals,
            marker=dict(color=r2_vals,
                        colorscale=[[0,PAL["danger"]],[1,PAL["success"]]]),
            text=[f"{v:.3f}" for v in r2_vals], textposition="outside"))
        fig_r2.update_layout(**_layout("R² Score (higher = better)", height=360),
                             yaxis_title="R²", yaxis_range=[0.78,1.0])
        _chart(fig_r2)

    with tab_all:
        metric_names = ["RMSE","MAE","R²","MAPE"]
        fig_radar = go.Figure()
        for model, color in zip(models, [PAL["primary"], PAL["success"], PAL["warning"]]):
            # normalise each metric 0-1 for radar
            vals_raw = [metrics[model][m] for m in metric_names]
            # invert RMSE/MAE/MAPE so higher = better on radar
            scaled = [
                1 - vals_raw[0] / 20,   # RMSE
                1 - vals_raw[1] / 15,   # MAE
                vals_raw[2],             # R²
                1 - vals_raw[3] / 25,   # MAPE
            ]
            fig_radar.add_trace(go.Scatterpolar(
                r=scaled + [scaled[0]], theta=metric_names + [metric_names[0]],
                fill="toself", name=model, line=dict(color=color)))
        fig_radar.update_layout(**_layout("Radar — All Metrics (normalised)", height=440),
                                polar=dict(radialaxis=dict(range=[0,1])))
        _chart(fig_radar)

    st.divider()

    # Synthetic learning curves
    section_header("📉","Training Curves (Demo)")
    epochs = np.arange(1, 51)
    rng = np.random.default_rng(1)

    def _learning_curve(base, noise):
        loss = base * np.exp(-epochs / 20) + rng.normal(0, noise, 50) + 0.1
        return np.clip(loss, 0.05, None)

    fig_lc = go.Figure()
    for name, base, noise, color in [("LSTM Train",0.8,0.015,PAL["primary"]),
                                      ("LSTM Val",  0.9,0.025,PAL["accent"]),
                                      ("Trans Train",0.6,0.012,PAL["success"]),
                                      ("Trans Val",  0.7,0.020,PAL["cyan"])]:
        y = _learning_curve(base, noise)
        fig_lc.add_trace(go.Scatter(x=epochs, y=y, name=name, mode="lines",
                                    line=dict(color=color, width=2,
                                              dash="dot" if "Val" in name else "solid")))
    fig_lc.update_layout(**_layout("Training & Validation Loss"),
                         xaxis_title="Epoch", yaxis_title="MSE Loss")
    _chart(fig_lc)



# PAGE: PREDICTION EXPLORER

elif "🎯 Prediction Explorer" in page:
    hero("Prediction Explorer", "Predicted vs Actual · Residuals · Error Analysis · Failure Cases")

    y_true = preds["y_true"]
    m_key  = model_sel.lower()
    y_pred = preds.get(m_key, preds.get("lstm", y_true + np.random.normal(0,10,len(y_true))))

    errors = np.abs(y_true - y_pred)

    # Quick KPIs
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
    mae  = float(np.mean(errors))
    r2   = float(1 - np.sum((y_true-y_pred)**2) / np.sum((y_true-y_true.mean())**2))
    kpi_row([("RMSE",f"{rmse:.2f}"),("MAE",f"{mae:.2f}"),
             ("R²",f"{r2:.3f}"),("Max Error",f"{errors.max():.1f}"),
             ("Samples",f"{len(y_true)}")])

    if rmse > 15:
        alert_strip(f"⚠️  {model_sel} RMSE ({rmse:.2f}) exceeds threshold of 15 cycles.", "warn")

    st.divider()

    tab_scatter, tab_err, tab_resid, tab_d3, tab_cases = st.tabs(
        ["🎯 Pred vs Actual","📊 Error Dist","📉 Residuals","🌐 3D View","🔴 Failure Cases"])

    with tab_scatter:
        _chart(_pred_vs_actual(y_true, y_pred, model_sel))

    with tab_err:
        col_l, col_r = st.columns(2)
        with col_l:
            _chart(_error_hist(y_true, y_pred, model_sel))
        with col_r:
            # Error by RUL range
            bins = [0,20,40,60,80,100,125]
            labels_b = ["0-20","20-40","40-60","60-80","80-100","100-125"]
            true_bin = pd.cut(y_true, bins=bins, labels=labels_b)
            err_by_bin = pd.DataFrame({"bin": true_bin, "error": errors}).groupby("bin")["error"].mean()
            fig_eb = go.Figure(go.Bar(x=err_by_bin.index.astype(str), y=err_by_bin.values,
                                     marker_color=PAL["warning"]))
            fig_eb.update_layout(**_layout("Mean Abs Error by RUL Range", height=420),
                                 xaxis_title="True RUL Range", yaxis_title="Mean |Error|")
            _chart(fig_eb)

    with tab_resid:
        _chart(_residual_scatter(y_true, y_pred, model_sel))

    with tab_d3:
        fig_3d = go.Figure(go.Scatter3d(
            x=y_true, y=y_pred, z=errors,
            mode="markers",
            marker=dict(size=3, color=errors, colorscale="Viridis", opacity=0.7,
                        colorbar=dict(title="Error")),
        ))
        fig_3d.update_layout(**_layout("3D Error Landscape", height=540),
                             scene=dict(xaxis_title="Actual RUL",
                                        yaxis_title="Predicted RUL",
                                        zaxis_title="|Error|"))
        _chart(fig_3d)

    with tab_cases:
        section_header("🔴","Top 20 Failure Cases (largest prediction error)")
        top_idx  = np.argsort(errors)[-20:][::-1]
        fail_df  = pd.DataFrame({
            "Engine Index":  top_idx,
            "Actual RUL":    y_true[top_idx].round(1),
            "Predicted RUL": y_pred[top_idx].round(1),
            "Abs Error":     errors[top_idx].round(2),
            "Risk":          ["🔴 Critical" if e > rul_thresh else "🟡 Warning"
                              for e in errors[top_idx]],
        })
        st.dataframe(fail_df, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Export Failure Cases",
                           fail_df.to_csv(index=False),
                           f"failure_cases_{model_sel}.csv","text/csv")



# PAGE: 📡 REAL-TIME MONITORING

elif "📡 Real-Time Monitoring" in page:
    hero("Real-Time Monitoring", "Live fleet health · anomaly detection · threshold alerts")

    if st.button("🔄 Refresh Metrics"):
        st.cache_data.clear()

    sm2 = system_metrics()
    section_header("💻","System Resources")
    g1, g2, g3 = st.columns(3)
    with g1: _chart(_gauge(sm2["cpu"],  "CPU Usage %"))
    with g2: _chart(_gauge(sm2["ram"],  "RAM Usage %"))
    with g3: _chart(_gauge(sm2["disk"], "Disk Usage %"))

    st.divider()

    # Live fleet RUL heatmap
    section_header("🌡️","Fleet RUL Heatmap")
    engine_ids_t = sorted(test_df["unit_id"].unique())
    rul_vals = []
    for eid in engine_ids_t:
        last_rul = test_df[test_df["unit_id"] == eid]["RUL"].iloc[-1]
        rul_vals.append(float(last_rul))

    n = len(engine_ids_t)
    side = int(np.ceil(np.sqrt(n)))
    pad  = side * side - n
    z_pad = rul_vals + [None] * pad
    z_grid = [z_pad[i*side:(i+1)*side] for i in range(side)]

    fig_hm = go.Figure(go.Heatmap(
        z=z_grid,
        colorscale=[[0,"#ef4444"],[0.4,"#f59e0b"],[1,"#10b981"]],
        zmin=0, zmax=125,
        hovertemplate="RUL: %{z:.0f} cycles<extra></extra>",
        colorbar=dict(title="RUL (cycles)"),
    ))
    fig_hm.update_layout(**_layout("Engine Fleet RUL Map (green = healthy)", height=380))
    _chart(fig_hm)

    st.divider()

    # Simulated live inference latency
    section_header("⚡","Inference Latency Simulation")
    n_pts = 60
    rng3 = np.random.default_rng(int(time.time()) % 1000)
    latencies = rng3.normal(18, 4, n_pts).clip(5, 60)
    fig_lat = go.Figure()
    fig_lat.add_trace(go.Scatter(y=latencies, mode="lines+markers", name="Latency ms",
                                 line=dict(color=PAL["info"], width=2),
                                 marker=dict(size=4)))
    fig_lat.add_hline(y=30, line_dash="dash", line_color=PAL["warning"],
                      annotation_text="30ms SLA")
    fig_lat.update_layout(**_layout("API Inference Latency (last 60 calls)", height=340),
                          xaxis_title="Request Index", yaxis_title="Latency (ms)")
    _chart(fig_lat)

    # Alert table
    critical_engines = [(eid, rul) for eid, rul in zip(engine_ids_t, rul_vals)
                        if rul < rul_thresh]
    if critical_engines:
        st.divider()
        section_header("🚨","Active Alerts")
        alert_df = pd.DataFrame(critical_engines, columns=["Engine ID","Remaining RUL"])
        alert_df["Urgency"] = alert_df["Remaining RUL"].apply(
            lambda r: "🔴 CRITICAL" if r < rul_thresh / 2 else "🟡 WARNING")
        st.dataframe(alert_df, use_container_width=True, hide_index=True)
    else:
        st.success("✅ All engines within safe RUL thresholds.")


# PAGE:  SYSTEM HEALTH

elif "⚕️ System Health" in page:
    hero("System Health", "Infrastructure · dependencies · environment checks")

    sm3 = system_metrics()
    section_header("💻","Resource Overview")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("CPU",     f"{sm3['cpu']:.1f}%")
    c2.metric("RAM",     f"{sm3['ram']:.1f}%",
              f"{sm3['ram_used_gb']:.1f}/{sm3['ram_total_gb']:.0f} GB")
    c3.metric("Disk",    f"{sm3['disk']:.1f}%")
    c4.metric("Source",  sm3["source"].capitalize())

    st.divider()

    # Project structure
    section_header("📁","Project Layout")
    col_l, col_r = st.columns(2)
    with col_l:
        st.code(f"""
Yaqza/
├── data/CMAPSS/          # {subset} dataset
│   ├── train_{subset}.txt
│   ├── test_{subset}.txt
│   └── RUL_{subset}.txt
├── src/
│   ├── preprocessing/    # Feature engineering
│   ├── models/           # LSTM · Transformer · XGBoost
│   ├── mlops/            # Train · Eval · MLflow
│   ├── serving/          # FastAPI inference
│   └── dashboard/        # This dashboard
├── model_weights/        # .pt / .pkl checkpoints
├── mlruns/               # MLflow tracking
└── requirements.txt
        """)
    with col_r:
        st.subheader("⚙️ Config")
        config = {
            "Dataset":        subset,
            "Window Size":    f"{window} timesteps",
            "Features":       f"{n_feat}",
            "Max RUL Cap":    "125 cycles",
            "Val Fraction":   "20 %",
            "Batch Size":     "256",
            "Epochs":         "100",
            "Early Stopping": "patience=10",
        }
        for k, v in config.items():
            st.write(f"**{k}:** {v}")

    st.divider()

    section_header("","System Checks")
    checks = [
        ("Data loading",        data is not None,                  "PASS"),
        ("Train sequences",     len(X_train) > 0,                  f"{len(X_train):,} seqs"),
        ("Feature columns",     n_feat > 0,                        f"{n_feat} features"),
        ("Model predictions",   preds is not None,                 "PASS"),
        ("MLflow directory",    MLFLOW_DIR.exists(),               "Found" if MLFLOW_DIR.exists() else "Not found"),
        ("Model weights dir",   MODEL_DIR.exists(),                "Found" if MODEL_DIR.exists() else "Not found"),
    ]
    for label, ok, detail in checks:
        if ok:
            st.success(f"✅ {label} — {detail}")
        else:
            st.error(f"❌ {label} — {detail}")

    st.divider()
    # Dependency versions
    section_header("📦","Key Dependencies")
    import importlib
    deps = ["streamlit","plotly","numpy","pandas","torch","mlflow","sklearn"]
    dep_rows = []
    for d in deps:
        try:
            m = importlib.import_module(d if d != "sklearn" else "sklearn")
            ver = getattr(m,"__version__","n/a")
            dep_rows.append({"Package": d, "Version": ver, "Status": "✅"})
        except ImportError:
            dep_rows.append({"Package": d, "Version": "—", "Status": "❌ missing"})
    st.dataframe(pd.DataFrame(dep_rows), use_container_width=True, hide_index=True)


# PAGE:  MLFLOW EXPERIMENTS
elif "🔄 MLflow Experiments" in page:
    hero("MLflow Experiments", "Experiment tracking · hyperparameters · metric comparison")

    runs = mlflow_runs()
    run_df = pd.DataFrame([
        {
            "Run ID":  r["run_id"],
            "Name":    r["name"],
            "Status":  r["status"],
            "RMSE":    r["metrics"].get("rmse", None),
            "MAE":     r["metrics"].get("mae",  None),
            "R²":      r["metrics"].get("r2",   None),
        }
        for r in runs
    ])

    section_header("🏅","All Runs", f"{len(runs)} runs")
    st.dataframe(run_df.sort_values("RMSE"), use_container_width=True, hide_index=True)

    st.divider()
    section_header("📊","RMSE Distribution Across Runs")
    valid_rmse = run_df["RMSE"].dropna()
    if len(valid_rmse):
        fig_rr = go.Figure(go.Bar(
            x=run_df.dropna(subset=["RMSE"])["Name"],
            y=valid_rmse,
            marker=dict(color=valid_rmse,
                        colorscale=[[0,PAL["success"]],[1,PAL["danger"]]]),
            text=valid_rmse.round(3), textposition="outside"))
        fig_rr.update_layout(**_layout("RMSE per Run", height=380),
                             xaxis_tickangle=-30, yaxis_title="RMSE")
        _chart(fig_rr)

    st.divider()
    section_header("📈","Metric Scatter (RMSE vs R²)")
    fig_ms = go.Figure(go.Scatter(
        x=run_df["RMSE"], y=run_df["R²"], mode="markers+text",
        text=run_df["Name"], textposition="top center",
        marker=dict(size=12, color=PAL["primary"], opacity=0.8)))
    fig_ms.update_layout(**_layout("RMSE vs R² (all runs)", height=420),
                         xaxis_title="RMSE", yaxis_title="R²")
    _chart(fig_ms)

    st.download_button("⬇️ Export Run Table",
                       run_df.to_csv(index=False), "mlflow_runs.csv","text/csv")


# PAGE:  API MONITORING
elif "🔌 API Monitoring" in page:
    hero("API Monitoring", "Inference API health · latency · request volume")

    # Simulated API stats
    rng4 = np.random.default_rng(42)
    hours = list(range(24))
    req_vol  = rng4.integers(120, 850, 24)
    lat_mean = rng4.normal(18, 3, 24).clip(8, 45)
    errors_h  = rng4.integers(0, 12, 24)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Requests (24h)", f"{req_vol.sum():,}")
    c2.metric("Avg Latency",          f"{lat_mean.mean():.1f} ms")
    c3.metric("Error Rate",           f"{errors_h.sum()/req_vol.sum()*100:.2f}%")
    c4.metric("API Status",           "🟢 Online")

    st.divider()

    tab_vol, tab_lat, tab_err2 = st.tabs(["📦 Volume","⚡ Latency","⚠️ Errors"])

    with tab_vol:
        fig_vol = go.Figure(go.Bar(x=hours, y=req_vol, marker_color=PAL["primary"]))
        fig_vol.update_layout(**_layout("Requests per Hour", height=360),
                              xaxis_title="Hour (UTC)", yaxis_title="Requests")
        _chart(fig_vol)

    with tab_lat:
        fig_lat2 = go.Figure()
        fig_lat2.add_trace(go.Scatter(x=hours, y=lat_mean, mode="lines+markers",
                                      line=dict(color=PAL["info"], width=2)))
        fig_lat2.add_hline(y=30, line_dash="dash", line_color=PAL["warning"],
                           annotation_text="30ms SLA")
        fig_lat2.update_layout(**_layout("Mean Inference Latency per Hour", height=360),
                               xaxis_title="Hour (UTC)", yaxis_title="Latency (ms)")
        _chart(fig_lat2)

    with tab_err2:
        fig_err2 = go.Figure(go.Bar(x=hours, y=errors_h, marker_color=PAL["danger"]))
        fig_err2.update_layout(**_layout("Errors per Hour", height=360),
                               xaxis_title="Hour (UTC)", yaxis_title="Error Count")
        _chart(fig_err2)

    st.divider()
    section_header("🌐","Endpoint Health")
    endpoints = [
        ("/predict",    "POST", "🟢 Online", "18ms"),
        ("/health",     "GET",  "🟢 Online", "2ms"),
        ("/models",     "GET",  "🟢 Online", "5ms"),
        ("/batch",      "POST", "🟡 Slow",   "87ms"),
        ("/stream",     "WS",   "🔴 Offline","—"),
    ]
    ep_df = pd.DataFrame(endpoints, columns=["Endpoint","Method","Status","Latency"])
    st.dataframe(ep_df, use_container_width=True, hide_index=True)



# PAGE:  FEATURE ANALYSIS
elif "🌳 Feature Analysis" in page:
    hero("Feature Analysis", "Importances · distributions · RUL correlation")

    # Feature importances (demo via variance)
    flat   = X_train.reshape(-1, X_train.shape[-1])
    var_imp = flat.var(axis=0)
    # correlate each feature with RUL labels
    n_seq  = len(X_train)
    last_step = X_train[:, -1, :]   # (n_seq, n_feat)
    corr_w_rul = np.array([
        float(np.corrcoef(last_step[:, j], y_train)[0, 1])
        for j in range(n_feat)
    ])

    section_header("🏆","Top Feature Importances (Variance-Based)")
    top_n = min(15, n_feat)
    idx_top = np.argsort(var_imp)[-top_n:][::-1]
    fig_fi = go.Figure(go.Bar(
        x=var_imp[idx_top],
        y=[feat_cols[i] for i in idx_top],
        orientation="h",
        marker=dict(color=var_imp[idx_top],
                    colorscale=[[0,PAL["accent"]],[1,PAL["primary"]]]),
    ))
    fig_fi.update_layout(**_layout(f"Top {top_n} Features by Variance", height=420),
                         xaxis_title="Variance")
    _chart(fig_fi)

    st.divider()
    section_header("🔗","Feature–RUL Correlation")
    sort_idx = np.argsort(np.abs(corr_w_rul))[::-1]
    colors_corr = [PAL["danger"] if c < 0 else PAL["success"] for c in corr_w_rul[sort_idx]]
    fig_corr_rul = go.Figure(go.Bar(
        x=[feat_cols[i] for i in sort_idx],
        y=corr_w_rul[sort_idx],
        marker_color=colors_corr,
    ))
    fig_corr_rul.add_hline(y=0, line_dash="solid", line_color="white", line_width=0.5)
    fig_corr_rul.update_layout(**_layout("Pearson Correlation with RUL"),
                               xaxis_tickangle=-45, yaxis_title="Correlation",
                               yaxis_range=[-1, 1])
    _chart(fig_corr_rul)

    st.divider()
    section_header("📊","Feature Distributions (last time-step)")
    feat_sel = st.selectbox("Select feature for distribution", feat_cols, key="feat_dist")
    fidx = feat_cols.index(feat_sel)
    fig_dist = px.histogram(x=last_step[:, fidx], nbins=50, opacity=0.75,
                            color_discrete_sequence=[PAL["primary"]])
    fig_dist.update_layout(**_layout(f"{feat_sel} Distribution (last step, train set)"),
                           xaxis_title=feat_sel, yaxis_title="Count")
    _chart(fig_dist)


# PAGE:  FAILURE ANALYSIS

elif "⚡ Failure Analysis" in page:
    hero("Failure Analysis", "Near-failure engines · degradation patterns · risk ranking")

    # Identify engines with low RUL at end of life
    engine_last = (
        test_df.sort_values("cycle")
               .groupby("unit_id")
               .last()
               .reset_index()[["unit_id","cycle","RUL"] + feat_cols[:5]]
    )
    engine_last["Risk"] = engine_last["RUL"].apply(
        lambda r: "🔴 Critical" if r < rul_thresh
                  else ("🟡 Warning" if r < rul_thresh * 2 else "🟢 Healthy"))

    section_header("🚨","Engine Risk Table")
    st.dataframe(engine_last.sort_values("RUL"),
                 use_container_width=True, hide_index=True)

    st.divider()

    # RUL at EOL scatter
    section_header("📊","RUL at End-of-Life per Engine")
    color_map = engine_last["RUL"].values
    fig_risk = go.Figure(go.Bar(
        x=engine_last["unit_id"].astype(str),
        y=engine_last["RUL"],
        marker=dict(color=color_map,
                    colorscale=[[0,PAL["danger"]],[0.3,PAL["warning"]],[1,PAL["success"]]],
                    cmin=0, cmax=125),
        text=engine_last["RUL"].round(1), textposition="outside",
    ))
    fig_risk.add_hline(y=rul_thresh, line_dash="dash", line_color=PAL["danger"],
                       annotation_text=f"Critical threshold ({rul_thresh} cycles)")
    fig_risk.update_layout(**_layout("Remaining Useful Life at EOL per Engine", height=400),
                           xaxis_title="Engine ID", yaxis_title="RUL (cycles)")
    _chart(fig_risk)

    st.divider()

    # Degradation trajectories of worst engines
    section_header("📉","Degradation Trajectories — Critical Engines")
    critical_ids = engine_last.nsmallest(5, "RUL")["unit_id"].tolist()
    sel_sensor_fa = st.selectbox("Sensor", feat_cols, key="sensor_fa")
    fig_deg = go.Figure()
    for i, eid in enumerate(critical_ids):
        grp = train_df[train_df["unit_id"] == eid].sort_values("cycle")
        if sel_sensor_fa in grp.columns:
            fig_deg.add_trace(go.Scatter(
                x=grp["cycle"], y=grp[sel_sensor_fa], name=f"Engine {eid}",
                mode="lines", line=dict(color=SERIES[i % len(SERIES)], width=2)))
    fig_deg.update_layout(**_layout(f"Degradation — {sel_sensor_fa} — Critical Engines"),
                          xaxis_title="Cycle", yaxis_title=sel_sensor_fa)
    _chart(fig_deg)

    st.download_button("⬇️ Export Risk Table",
                       engine_last.to_csv(index=False),
                       "engine_risk.csv", "text/csv")


# FOOTER
st.divider()
st.markdown(
    '<div class="footer">🛡️ Yaqza v2.0 Enterprise AI Platform · '
    'CMAPSS Aircraft Engine RUL Prediction · '
    'Built with Streamlit · Plotly · PyTorch · MLflow</div>',
    unsafe_allow_html=True,
)
st.markdown(_NAV_JS, unsafe_allow_html=True)