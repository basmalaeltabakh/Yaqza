# Yaqza | يقظة 🔍
### Predictive Maintenance System for Industrial IoT
**Microsoft ML Engineering Track · DEPI**

---

## Project Summary
Yaqza predicts **Remaining Useful Life (RUL)** of industrial machinery from real-time multi-sensor streams — giving factory operators a 48–72 hour window to schedule maintenance before equipment failure.

**Core Stack:** Azure IoT Hub · Python · PyTorch · FastAPI · MLflow · Power BI / Grafana

---

## Team

| ID | Role | Branch |
|---|---|---|
| M1 | IoT & Data Engineer | `feature/m1-iot` |
| M2 | ML Engineer — Modeling | `feature/m2-models` |
| M3 | Cloud & Edge Engineer | `feature/m3-deploy` |
| M4 | MLOps Engineer | `feature/m4-mlops` |
| M5 | Dashboard & Monitoring | `feature/m5-dashboard` |

---

## Repository Structure

```
yaqza/
├── data/                        # Datasets (gitignored — download manually)
│   ├── CMAPSS/                  # NASA Turbofan — RUL prediction (M1 downloads)
│   ├── PRONOSTIA/               # FEMTO Bearing — anomaly detection (M1 downloads)
│   ├── XJTU-SY/                 # XJTU Bearing — accelerated life test (M1 downloads)
│   └── README.md                # Download instructions for all datasets
│
├── notebooks/                   # Jupyter notebooks — numbered by phase
│   ├── 01_EDA_CMAPSS.ipynb      # M2: EDA on CMAPSS FD001
│   ├── 02_EDA_Bearing.ipynb     # M2: EDA on PRONOSTIA & XJTU-SY
│   ├── 03_Preprocessing.ipynb   # M2: Normalization, sliding windows, split
│   ├── 04_LSTM_RUL.ipynb        # M2: LSTM RUL model training
│   ├── 05_TFT_RUL.ipynb         # M4: Temporal Fusion Transformer training
│   ├── 06_Autoencoder.ipynb     # M2: LSTM Autoencoder anomaly detection
│   └── 07_Classifier.ipynb      # M2: XGBoost fault type classifier
│
├── src/                         # Production Python source code
│   ├── ingestion/               # M1: IoT simulator + Azure IoT Hub client
│   ├── preprocessing/           # M2: Feature engineering + sliding windows
│   ├── models/                  # M2/M4: Model definitions (LSTM, TFT, XGBoost)
│   ├── serving/                 # M3: FastAPI inference service
│   ├── mlops/                   # M4: Drift detection + retraining pipeline
│   └── dashboard/               # M5: Azure Data Lake query helpers
│
├── edge/
│   └── anomaly_module/          # M1: ONNX edge inference module for IoT Edge
│
├── tests/                       # Unit + integration tests
│   ├── test_preprocessing.py    # M2 owns
│   ├── test_models.py           # M2 owns
│   ├── test_api.py              # M3 owns
│   └── test_drift.py            # M4 owns
│
├── docs/                        # Architecture diagrams + final report (M5)
├── Dockerfile                   # M3: Docker image for model API
├── docker-compose.yml           # M3: Local multi-container setup
├── requirements.txt             # All Python dependencies
└── .gitignore                   # Excludes data/, weights, __pycache__
```

---

## Setup (Do This First — Everyone)

### 1. Clone the repo
```bash
git clone https://github.com/<your-org>/yaqza.git
cd yaqza
```

### 2. Create your virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac / Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download datasets
- See `data/CMAPSS/README.md` for CMAPSS download + format details ← **Start here (Phase 1)**
- See `data/PRONOSTIA/README.md` for PRONOSTIA
- See `data/XJTU-SY/README.md` for XJTU-SY
- M1 is responsible for confirming all datasets are accessible

### 5. Launch Jupyter
```bash
jupyter lab
```
Open `notebooks/01_EDA_CMAPSS.ipynb` to begin Phase 1.

---

## Branching Rules

```
main          ← Production only. Never push directly.
  └── develop ← Integration branch. All features merge here via PR.
        ├── feature/m1-iot
        ├── feature/m2-models
        ├── feature/m3-deploy
        ├── feature/m4-mlops
        └── feature/m5-dashboard
```

**PR Rule:** Every feature branch requires **at least 1 reviewer approval** before merging into `develop`.  
**M1 merges** `develop → main` at the end of each phase.

### Creating your branch (first time)
```bash
git checkout develop
git pull origin develop
git checkout -b feature/m2-models   # replace with your branch name
```

### Daily workflow
```bash
git add .
git commit -m "feat(m2): add sliding window preprocessing"
git push origin feature/m2-models
# Then open a Pull Request on GitHub targeting develop
```

---

## Running Tests
```bash
pytest tests/ -v --cov=src
```

---

## Phase 1 Focus — What to do right now

| Who | Task |
|---|---|
| **M1** | Download CMAPSS (all 4 subsets) into `data/CMAPSS/`. See that folder's README. |
| **M1** | Set up Azure free account + IoT Hub |
| **M2** | Open `notebooks/01_EDA_CMAPSS.ipynb` and start EDA on FD001 |
| **M3** | Set up Azure free account + Data Lake Gen2 |
| **M4** | Install MLflow, set up tracking server locally |
| **M5** | Start drafting `docs/architecture.md` |

---


