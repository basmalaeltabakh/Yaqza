"""
Yaqza Dashboard — Premium CSS Theming Module.

Provides theme-aware styling for the entire dashboard including:
- Glassmorphism metric cards with gradient borders
- Animated KPI counters
- Professional typography (Inter font family)
- Dark/Light mode adaptive components
- Smooth transitions and micro-animations
- Responsive layout utilities
"""

from __future__ import annotations


def get_custom_css() -> str:
    """Return the full custom CSS for the dashboard."""
    return """
    <style>
    /* ── Google Font ─────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ── Root Variables ──────────────────────────────────────────── */
    :root {
        --font-main: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --radius-xl: 20px;
        --shadow-sm: 0 1px 3px rgba(0,0,0,0.08);
        --shadow-md: 0 4px 12px rgba(0,0,0,0.1);
        --shadow-lg: 0 8px 30px rgba(0,0,0,0.12);
        --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        --accent-1: #6366f1;
        --accent-2: #8b5cf6;
        --accent-3: #a78bfa;
        --gradient-primary: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a78bfa 100%);
        --gradient-success: linear-gradient(135deg, #10b981 0%, #34d399 100%);
        --gradient-warning: linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%);
        --gradient-danger: linear-gradient(135deg, #ef4444 0%, #f87171 100%);
        --gradient-info: linear-gradient(135deg, #3b82f6 0%, #60a5fa 100%);
    }

    /* ── Global ──────────────────────────────────────────────────── */
    html, body, [class*="st-"] {
        font-family: var(--font-main) !important;
    }
    .main .block-container {
        padding: 1.5rem 2rem 3rem 2rem;
        max-width: 1400px;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: var(--font-main) !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }

    /* ── Sidebar ─────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1b4b 0%, #312e81 100%) !important;
        border-right: 1px solid rgba(99,102,241,0.2);
    }
    section[data-testid="stSidebar"] * {
        color: #e0e7ff !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 0.95rem !important;
        padding: 0.55rem 0.8rem !important;
        border-radius: var(--radius-sm) !important;
        transition: var(--transition) !important;
        margin-bottom: 2px !important;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(255,255,255,0.08) !important;
    }
    section[data-testid="stSidebar"] .stRadio label[data-checked="true"],
    section[data-testid="stSidebar"] .stRadio [aria-checked="true"] + label {
        background: rgba(99,102,241,0.25) !important;
        border-left: 3px solid #818cf8 !important;
    }

    /* ── Metric Cards ────────────────────────────────────────────── */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(99,102,241,0.08) 0%, rgba(139,92,246,0.05) 100%);
        border: 1px solid rgba(99,102,241,0.15);
        border-radius: var(--radius-md);
        padding: 1rem 1.2rem;
        transition: var(--transition);
        box-shadow: var(--shadow-sm);
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
        border-color: rgba(99,102,241,0.3);
    }
    div[data-testid="stMetric"] label {
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.75;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 800 !important;
    }

    /* ── Hero Section ────────────────────────────────────────────── */
    .hero-banner {
        background: var(--gradient-primary);
        border-radius: var(--radius-xl);
        padding: 2.5rem 3rem;
        color: white;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(99,102,241,0.25);
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-banner h1 {
        font-size: 2.2rem !important;
        margin-bottom: 0.3rem;
        color: white !important;
    }
    .hero-banner p {
        font-size: 1.05rem;
        opacity: 0.9;
        margin: 0;
    }

    /* ── Status Badges ───────────────────────────────────────────── */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 100px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .status-online {
        background: rgba(16,185,129,0.12);
        color: #10b981;
        border: 1px solid rgba(16,185,129,0.25);
    }
    .status-offline {
        background: rgba(239,68,68,0.12);
        color: #ef4444;
        border: 1px solid rgba(239,68,68,0.25);
    }
    .status-warning {
        background: rgba(245,158,11,0.12);
        color: #f59e0b;
        border: 1px solid rgba(245,158,11,0.25);
    }

    /* ── Glass Card ──────────────────────────────────────────────── */
    .glass-card {
        background: rgba(255,255,255,0.03);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: var(--transition);
    }
    .glass-card:hover {
        border-color: rgba(99,102,241,0.2);
        box-shadow: var(--shadow-md);
    }

    /* ── Architecture Card ───────────────────────────────────────── */
    .arch-card {
        background: linear-gradient(145deg, rgba(99,102,241,0.06) 0%, rgba(139,92,246,0.03) 100%);
        border: 1px solid rgba(99,102,241,0.12);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        transition: var(--transition);
    }
    .arch-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 24px rgba(99,102,241,0.1);
    }
    .arch-card h3 {
        font-size: 1.1rem !important;
        margin-bottom: 0.8rem;
    }

    /* ── Tabs ────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: var(--radius-sm) var(--radius-sm) 0 0;
        font-weight: 600;
        font-size: 0.88rem;
        padding: 0.6rem 1.2rem;
        transition: var(--transition);
    }

    /* ── Expanders ───────────────────────────────────────────────── */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        border-radius: var(--radius-sm) !important;
    }

    /* ── DataFrames ──────────────────────────────────────────────── */
    .stDataFrame {
        border-radius: var(--radius-md) !important;
        overflow: hidden;
    }

    /* ── Download Buttons ────────────────────────────────────────── */
    .stDownloadButton button {
        background: var(--gradient-primary) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--radius-sm) !important;
        font-weight: 600 !important;
        transition: var(--transition) !important;
    }
    .stDownloadButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(99,102,241,0.3) !important;
    }

    /* ── Info / Warning / Error Boxes ────────────────────────────── */
    .stAlert {
        border-radius: var(--radius-md) !important;
        border-left-width: 4px !important;
    }

    /* ── Divider ─────────────────────────────────────────────────── */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, rgba(99,102,241,0.2) 50%, transparent 100%);
        margin: 1.5rem 0;
    }

    /* ── Footer ──────────────────────────────────────────────────── */
    .footer-bar {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
        font-size: 0.82rem;
        opacity: 0.6;
        letter-spacing: 0.01em;
    }

    /* ── Animations ──────────────────────────────────────────────── */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(15px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .animate-in {
        animation: fadeInUp 0.5s ease-out forwards;
    }

    /* ── Scrollbar ───────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: rgba(99,102,241,0.3);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,0.5); }
    </style>
    """


def render_hero(title: str, subtitle: str) -> str:
    """Return HTML for the hero banner."""
    return f"""
    <div class="hero-banner animate-in">
        <h1>🛡️ {title}</h1>
        <p>{subtitle}</p>
    </div>
    """


def render_status_badge(label: str, status: str = "online") -> str:
    """Return HTML for a coloured status badge."""
    icon = {"online": "🟢", "offline": "🔴", "warning": "🟡"}.get(status, "⚪")
    return f'<span class="status-badge status-{status}">{icon} {label}</span>'
