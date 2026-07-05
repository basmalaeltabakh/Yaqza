# ═════════════════════════════════════════════════════════════════════════════
# Yaqza Predictive Maintenance API - Production Dockerfile
# ═════════════════════════════════════════════════════════════════════════════

FROM python:3.11-slim-bookworm

# ── System Dependencies ─────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Working Directory ────────────────────────────────────────────────────────
WORKDIR /app

# ── Python Environment ───────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONFAULTHANDLER=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# ── Install Dependencies ──────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application Code ────────────────────────────────────────────────────────
COPY backend/ ./backend/
COPY data/ ./data/
COPY models/ ./models/

# ── Set Working Directory to Backend ────────────────────────────────────────
WORKDIR /app/backend

# ── Health Check ────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# ── Production Server ─────────────────────────────────────────────────────────
# Gunicorn + Uvicorn workers for production
# Workers = (2 × CPU cores) + 1
CMD ["gunicorn", 
     "main:app", 
     "--workers", "4", 
     "--worker-class", "uvicorn.workers.UvicornWorker",
     "--bind", "0.0.0.0:8000",
     "--timeout", "120",
     "--keep-alive", "5",
     "--max-requests", "1000",
     "--max-requests-jitter", "100",
     "--access-logfile", "-",
     "--error-logfile", "-",
     "--capture-output",
     "--enable-stdio-inheritance"]