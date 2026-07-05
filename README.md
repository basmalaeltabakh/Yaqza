<div align="center">

# ⚙️ Yaqza — Predictive Maintenance System for Turbofan Engines
A production-ready machine learning system for predicting Remaining Useful Life (RUL) of turbofan engines using the NASA CMAPSS dataset. Built with FastAPI, SQLAlchemy, and multiple ML models including ensemble trees, probabilistic boosting, and deep learning.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)](https://streamlit.io)
[![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-orange?logo=google)](https://ai.google.dev)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)



</div>

---

## 📖 About

**Yaqza** (يقظة) is an end-to-end predictive maintenance platform that forecasts the **Remaining Useful Life (RUL)** of industrial turbofan engines using the [NASA CMAPSS dataset](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data). The system ingests live sensor readings, runs five ML models in parallel, and generates bilingual maintenance reports via Google Gemini AI.

### Key Capabilities

- 🔮 **Multi-model RUL prediction** — Ridge, Random Forest, XGBoost, NGBoost, CNN-LSTM
- 📊 **Sensor trend analysis** — 14 critical sensors monitored for anomalies
- 🌐 **Bilingual AI reports** — Arabic + English maintenance recommendations via Gemini 2.5 Flash
- 📡 **Live dashboard** — Real-time Streamlit UI with interactive charts
- 🔄 **REST API** — 10 endpoints covering ingestion, prediction, comparison, and analysis

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Yaqza System                          │
│                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────┐  │
│  │   Streamlit  │───▶│   FastAPI    │───▶│  SQLite  │  │
│  │  Dashboard   │    │   Backend    │    │    DB    │  │
│  └──────────────┘    └──────┬───────┘    └──────────┘  │
│                             │                           │
│                    ┌────────▼────────┐                  │
│                    │  ML Pipeline    │                  │
│                    │  ┌───────────┐  │                  │
│                    │  │ LVR → Scale│  │                  │
│                    │  │ TSFresh   │  │                  │
│                    │  │ PCA       │  │                  │
│                    │  └─────┬─────┘  │                  │
│                    │        │        │                  │
│                    │  ┌─────▼─────┐  │                  │
│                    │  │  5 Models │  │                  │
│                    │  │ Ridge     │  │                  │
│                    │  │ RF/XGB    │  │                  │
│                    │  │ NGBoost   │  │                  │
│                    │  │ CNN-LSTM  │  │                  │
│                    │  └─────┬─────┘  │                  │
│                    └────────┼────────┘                  │
│                             │                           │
│                    ┌────────▼────────┐                  │
│                    │  Gemini 2.5     │                  │
│                    │  Flash API      │                  │
│                    │  (Bilingual)    │                  │
│                    └─────────────────┘                  │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Yaqza/
├── backend/
│   ├── main.py               # FastAPI app — 10 endpoints
│   ├── features.py           # ML pipeline & prediction logic
│   ├── gemini_advisor.py     # Gemini bilingual analysis (pure HTTP)
│   ├── models.py             # SQLAlchemy DB models
│   ├── schemas.py            # Pydantic request/response schemas
│   ├── crud.py               # Database CRUD operations
│   ├── database.py           # SQLite connection & init
│   ├── seed_data.py          # Seed synthetic readings for ENG001/002/003
│   ├── schema.sql            # DB schema reference
│   ├── requirements.txt      # Python dependencies
│   └── models/               # ML model artifacts (not in repo)
│       ├── ridge_model.pkl
│       ├── rf_model.pkl
│       ├── xgb_model.pkl
│       ├── ngb_model.pkl
│       ├── cnn_lstm_model.keras
│       ├── lvr.pkl
│       ├── scaler.pkl
│       ├── selected_features.pkl
│       ├── pca_step.pkl
│       └── tsfresh_pipeline.pkl
├── frontend/
│   ├── dashboard.py          # Streamlit app
│   └── yaqza_dashboard_v5.html  # Embedded HTML dashboard
├── ml/
│   ├── 1-EDA.ipynb
│   ├── 2-target-metrics-baseline.ipynb
│   ├── 3-features_engineering.ipynb
│   └── 4-predict_rul_with_ML.ipynb
├── data/
│   ├── train_FD001.txt       # NASA CMAPSS training data
│   ├── test_FD001.txt
│   └── RUL_FD001.txt
├── .gitignore
└── README.md
```

---

##  Quick Start

### Prerequisites

- Python 3.11+
- Git

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/Yaqza.git
cd Yaqza
```

### 2. Setup backend

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Add model artifacts

Copy your trained `.pkl` and `.keras` files into `backend/models/`:

```
backend/models/
├── ridge_model.pkl       ← required
├── rf_model.pkl          ← required
├── xgb_model.pkl         ← required
├── ngb_model.pkl         ← required
├── cnn_lstm_model.keras  ← required
├── lvr.pkl               ← required
├── scaler.pkl            ← required
├── selected_features.pkl ← required
├── pca_step.pkl          ← required
└── tsfresh_pipeline.pkl  ← required
```

### 4. Configure environment

Create `backend/.env`:

```env
DATABASE_URL=sqlite:///./yaqza.db
MODEL_PATH=models/
GEMINI_API_KEY=your_gemini_api_key_here
```

Get your Gemini API key from [aistudio.google.com](https://aistudio.google.com).

### 5. Seed the database

```bash
cd backend
python seed_data.py
```

Expected output:
```
✅ Seeded 50 readings for ENG001
✅ Seeded 50 readings for ENG002
✅ Seeded 50 readings for ENG003
✅ All engines seeded successfully
```

### 6. Run the API

```bash
uvicorn main:app --reload
```

API available at: `http://127.0.0.1:8000`  
Interactive docs: `http://127.0.0.1:8000/docs`

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | API health check |
| `GET` | `/engines` | List all monitored engines |
| `GET` | `/models` | List available ML models |
| `POST` | `/ingest` | Ingest a sensor reading |
| `GET` | `/predict/{engine_id}` | Predict RUL (Ridge default) |
| `GET` | `/predict/{engine_id}/model/{model}` | Predict with a specific model |
| `GET` | `/predict/{engine_id}/compare` | Compare all 5 models |
| `GET` | `/recommend/{engine_id}` | Gemini maintenance recommendation |
| `GET` | `/analyze/{engine_id}` | Full bilingual analysis (AR + EN) |
| `GET` | `/history/{engine_id}` | Prediction history |

### Example: Ingest a reading

```bash
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "engine_id": "ENG001",
    "cycle": 47,
    "op_setting_1": 0.0023,
    "op_setting_2": 0.0003,
    "op_setting_3": 100.0,
    "sensor_2": 641.82,
    "sensor_3": 1589.7,
    "sensor_7": 554.3,
    "sensor_11": 47.2
  }'
```

### Example: Compare all models

```bash
curl "http://localhost:8000/predict/ENG001/compare"
```

Response:
```json
{
  "engine_id": "ENG001",
  "consensus_rul": 112.6,
  "overall_status": "NORMAL",
  "n_models_evaluated": 5,
  "predictions": [
    { "model_name": "RIDGE",    "rul": 177.46, "status": "success" },
    { "model_name": "RF",       "rul": 123.82, "status": "success" },
    { "model_name": "XGB",      "rul": 123.33, "status": "success" },
    { "model_name": "NGB",      "rul": 124.10, "status": "success",
      "uncertainty": { "std": 5.46, "ci_90": [115.11, 133.09] } },
    { "model_name": "CNN-LSTM", "rul": 14.27,  "status": "success" }
  ],
  "recommendation": {
    "recommended_model": "NGBoost",
    "reason": "Provides uncertainty quantification for risk-aware maintenance",
    "rul": 124.1
  }
}
```

---

## 🧠 ML Pipeline

```
Raw Sensor Readings (21 sensors × N cycles)
        │
        ▼
LowVarianceFeaturesRemover  ← removes constant sensors
        │
        ▼
DataFrameMinMaxScaler       ← normalizes to [0, 1]
        │
        ▼
RollTimeSeriesTransformer   ← sliding window (30 cycles)
        │
        ▼
TSFreshFeaturesExtractor    ← statistical feature extraction
        │
        ▼
Feature Selection           ← top relevant features
        │
        ▼
CustomPCA                   ← dimensionality reduction (12 PCs)
        │
        ├──▶ Ridge Regression
        ├──▶ Random Forest
        ├──▶ XGBoost
        └──▶ NGBoost (+ uncertainty intervals)

Raw Scaled Sequences (30 × n_features)
        └──▶ CNN-LSTM (bypasses TSFresh/PCA)
```

### Model Performance (CMAPSS FD001 test set)

| Model | Type | Features |
|-------|------|----------|
| Ridge Regression | Linear baseline | TSFresh + PCA |
| Random Forest | Ensemble | TSFresh + PCA |
| XGBoost | Gradient boosting | TSFresh + PCA |
| NGBoost | Probabilistic | TSFresh + PCA + uncertainty |
| CNN-LSTM | Deep learning | Raw sequences |

---

## 🌐 Gemini AI Analysis

The `/analyze/{engine_id}` endpoint generates a **full bilingual diagnostic report**:

```json
{
  "report_en": "Engine ENG001 is currently in stable condition...",
  "report_ar": "يتمتع المحرك ENG001 بحالة مستقرة...",
  "actions_en": ["Inspect LPT outlet temperature sensor...", "..."],
  "actions_ar": ["فحص حساس درجة حرارة مخرج التوربين...", "..."],
  "risk_level": "LOW",
  "urgency": "SCHEDULED",
  "critical_sensors": [...],
  "model_insight_en": "NGBoost is most reliable due to tight confidence intervals..."
}
```

---

## 👥 Team

| Role | Member | Responsibility |
|------|--------|---------------|
| Basmala Saeed | Backend Engineer | FastAPI, endpoints, Gemini integration |
| Aya Yasser| Database Engineer | SQLite, CRUD, seed data |
| Omar Abdullah | ML Engineer | Model training, feature pipeline, .pkl artifacts |
| Esraa Mohammed Morsy | Frontend Engineer | Streamlit dashboard, charts, UI |
| AHmed Badawy | DevOps / Integration | Docker, CI/CD, deployment |

---

## 📦 Dependencies

```
fastapi          uvicorn        sqlalchemy
pydantic         python-dotenv  joblib
numpy            pandas         scikit-learn
tsfresh          xgboost        ngboost
tensorflow       httpx
```

Install all:
```bash
pip install -r backend/requirements.txt
```

---

## ⚠️ Notes

- Model `.pkl` and `.keras` files are **not included** in the repository due to size. Contact the ML engineer (C) for the artifact bundle.
- The Gemini advisor uses **pure HTTP** (no Google SDK) to avoid protobuf conflicts with TensorFlow.
- SQLite DB is created automatically on first run via `init_db()`.

---



<div align="center">
Built with ❤️ by the Yaqza Team — DEBI Graduation Project 2026
</div>
