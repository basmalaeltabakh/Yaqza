# backend/main.py
import sys
import logging
import numpy as np
import gemini_advisor

# ── Setup logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ── Lazy imports for heavy / conflict-prone packages ────────────────────────
_tf = None

def _get_tf():
    global _tf
    if _tf is None:
        try:
            import tensorflow as tf
            _tf = tf
            logger.info("TensorFlow loaded (lazy import)")
        except ImportError as exc:
            logger.warning("TensorFlow not installed or failed to import: %s", exc)
            _tf = False
    return _tf if _tf is not False else None


# ── Imports for joblib unpickling ───────────────────────────────────────────
try:
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
    sys.modules['__main__'].NGBRegressor = NGBRegressor
    sys.modules['__main__'].Normal = Normal
except ImportError:
    logger.warning("ngboost not installed")
    pass

try:
    import xgboost as xgb
except ImportError:
    logger.warning("xgboost not installed")
    pass

try:
    from sklearn.ensemble import RandomForestRegressor
except ImportError:
    pass

try:
    from sklearn.linear_model import Ridge
except ImportError:
    pass


import features as feat

sys.modules['__main__'].LowVarianceFeaturesRemover = feat.LowVarianceFeaturesRemover
sys.modules['__main__'].DataFrameMinMaxScaler = feat.DataFrameMinMaxScaler
sys.modules['__main__'].RollTimeSeriesTransformer = feat.RollTimeSeriesTransformer
sys.modules['__main__'].TSFreshFeaturesExtractor = feat.TSFreshFeaturesExtractor
sys.modules['__main__'].CustomPCA = feat.CustomPCA
sys.modules['__main__'].TSFreshFeaturesSelector = feat.TSFreshFeaturesSelector

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import os
import crud
from database import get_db, init_db
from schemas import (
    SensorReadingCreate,
    PredictionResponse,
    PredictionHistoryItem,
    SuccessResponse,
    HealthResponse,
    ModelComparisonResponse,
    AvailableModelInfo,
    EnginesListResponse,
    ModelsListResponse,
    PredictRequest,
    CompareRequest,
    EngineInfo,
    MaintenanceRecommendation,
    AnalysisResponse
)

init_db()

app = FastAPI(
    title="Yaqza API",
    description="Predictive Maintenance API — CMAPSS Dataset",
    version="0.2.0"
)

PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip()

# ═════════════════════════════════════════════════════════════════════════════
# CORS CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════
_default_origins = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "null",
]

env_origins = os.getenv("ALLOWED_ORIGINS", "")
if env_origins and env_origins != "*":
    allowed_origins = [o.strip() for o in env_origins.split(",") if o.strip()]
    if "null" not in allowed_origins:
        allowed_origins.append("null")
else:
    allowed_origins = _default_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Total-Count"],
    max_age=600,
)

@app.middleware("http")
async def add_cors_headers(request, call_next):
    response = await call_next(request)
    origin = request.headers.get("origin", "")
    if origin == "null" or not origin:
        response.headers["Access-Control-Allow-Origin"] = "null"
        response.headers["Access-Control-Allow-Credentials"] = "true"
    return response

N_READINGS = 40
MIN_READINGS = 30

MODEL_MAP = {
    "ridge": "ridge_model",
    "rf": "rf_model",
    "xgboost": "xgb_model",
    "ngboost": "ngb_model",
    "cnn_lstm": "cnn_lstm_model",
}


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _build_prediction_response(
    engine_id: str,
    rul: float,
    readings: list,
    model_name: str = "ridge"
) -> PredictionResponse:
    health_score = round(min(100.0, (rul / 125.0) * 100), 1)
    failure_mode = feat.get_failure_mode(readings)
    status = "NORMAL" if rul >= 50 else "WARNING" if rul >= 20 else "CRITICAL"

    sensor_7_vals = [getattr(r, "sensor_7", 0) or 0 for r in readings[-30:]]
    variance_penalty = min(50, float(np.std(sensor_7_vals)) * 10) if sensor_7_vals else 25
    confidence = round(max(50.0, 100.0 - variance_penalty), 1)

    return PredictionResponse(
        engine_id=engine_id.upper(),
        rul=rul,
        confidence=confidence,
        failure_mode=failure_mode,
        health_score=health_score,
        status=status,
        model_used=model_name,
    )


def _save_prediction(db: Session, engine_id: str, response: PredictionResponse, model_name: str = "ridge"):
    crud.create_prediction(
        db=db,
        engine_id=engine_id,
        rul=response.rul,
        failure_mode=response.failure_mode,
        confidence=response.confidence,
        health_score=response.health_score,
        status=response.status,
        model_used=model_name,
    )


# ═════════════════════════════════════════════════════════════════════════════
# LIST / DROPDOWN ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/engines", response_model=EnginesListResponse)
def list_engines(min_readings: int = 0, db: Session = Depends(get_db)):
    try:
        logger.info(f"Listing engines with min_readings={min_readings}")

        if min_readings > 0:
            engines_data = crud.get_engines_with_readings_count(db, min_readings)
        else:
            from sqlalchemy import distinct, func
            from models import SensorReading, Prediction

            result = db.query(
                SensorReading.engine_id,
                func.count(SensorReading.id).label('readings_count')
            ).group_by(SensorReading.engine_id).all()

            engines_data = [{"engine_id": r[0], "readings_count": r[1]} for r in result]

            pred_engines = [r[0] for r in db.query(distinct(Prediction.engine_id)).all()]
            existing = {e["engine_id"] for e in engines_data}
            for eid in pred_engines:
                if eid not in existing:
                    engines_data.append({"engine_id": eid, "readings_count": 0})

        logger.info(f"Found {len(engines_data)} engines")

        engines = []
        for ed in engines_data:
            eid = ed["engine_id"]
            try:
                latest_pred = crud.get_latest_prediction(db, eid)
                engines.append(EngineInfo(
                    engine_id=eid,
                    readings_count=ed.get("readings_count", 0),
                    last_prediction=latest_pred.timestamp if latest_pred else None,
                    status=latest_pred.status if latest_pred else "NO_PREDICTION"
                ))
            except Exception as e:
                logger.error(f"Error getting prediction for {eid}: {e}")
                engines.append(EngineInfo(
                    engine_id=eid,
                    readings_count=ed.get("readings_count", 0),
                    last_prediction=None,
                    status="ERROR"
                ))

        return EnginesListResponse(engines=engines, total=len(engines))

    except Exception as e:
        logger.exception("Error in list_engines")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/models", response_model=ModelsListResponse)
def list_models(db: Session = Depends(get_db)):
    try:
        logger.info("Loading artifacts for /models")
        artifacts = feat.get_cached_artifacts()
        available_keys = list(artifacts.keys())
        logger.info(f"Loaded artifacts: {available_keys}")
    except Exception as e:
        logger.exception("Failed to load artifacts")
        available_keys = []

    model_info = {
        "ridge_model": {"name": "Ridge Regression", "type": "linear", "features": "TSFresh + PCA", "description": "Linear baseline with L2 regularization"},
        "rf_model": {"name": "Random Forest", "type": "ensemble", "features": "TSFresh + PCA", "description": "Bagging ensemble of decision trees"},
        "xgb_model": {"name": "XGBoost", "type": "gradient_boosting", "features": "TSFresh + PCA", "description": "Gradient boosted trees with early stopping"},
        "ngb_model": {"name": "NGBoost", "type": "probabilistic", "features": "TSFresh + PCA", "description": "Natural Gradient Boosting with uncertainty quantification"},
        "cnn_lstm_model": {"name": "CNN-LSTM", "type": "deep_learning", "features": "Raw Sequences", "description": "1D-CNN + LSTM hybrid for end-to-end learning"},
    }

    models = []
    for key, info in model_info.items():
        models.append(AvailableModelInfo(
            key=key,
            name=info["name"],
            type=info["type"],
            features=info["features"],
            description=info["description"],
            available=key in available_keys
        ))

    return ModelsListResponse(
        models=models,
        total_available=len([m for m in models if m.available]),
        total_expected=5
    )


# ═════════════════════════════════════════════════════════════════════════════
# PREDICTION ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="ok")

@app.get("/status")
def public_status():
    return {
        "status": "ok",
        "service": "yaqza-api",
        "public_base_url": PUBLIC_BASE_URL or None,
        "models_available": sorted([k for k in MODEL_MAP.keys()]),
    }


@app.post("/ingest", response_model=SuccessResponse, status_code=201)
def ingest_sensor(data: SensorReadingCreate, db: Session = Depends(get_db)):
    reading = crud.create_sensor_reading(db, data)
    return SuccessResponse(id=reading.id, status="created")


@app.post("/predict", response_model=PredictionResponse)
def predict_with_body(request: PredictRequest, db: Session = Depends(get_db)):
    engine_id = request.engine_id
    model_name = request.model_name

    if model_name not in MODEL_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {model_name}. Available: {list(MODEL_MAP.keys())}"
        )

    count = crud.get_readings_count(db, engine_id)
    if count < MIN_READINGS:
        raise HTTPException(
            status_code=404,
            detail=f"not enough readings for engine {engine_id} (have {count}, need {MIN_READINGS})"
        )

    readings = crud.get_last_n_readings(db, engine_id, n=N_READINGS)

    try:
        if model_name == "ridge":
            result = feat.predict_from_readings(readings)
            response = _build_prediction_response(engine_id, result["rul"], readings, model_name)
        else:
            result = feat.predict_rul_single(readings, model_key=MODEL_MAP[model_name])
            response = _build_prediction_response(engine_id, result["rul"], readings, model_name)

            if model_name == "ngboost" and result.get("uncertainty"):
                response.confidence = round(max(50.0, 100.0 - result["uncertainty"]["std"] * 2), 1)
    except Exception as e:
        logger.exception(f"Prediction failed for {engine_id}")
        raise HTTPException(status_code=500, detail=f"model failed: {str(e)}")

    _save_prediction(db, engine_id, response, model_name)

    return response


@app.get("/predict/{engine_id}", response_model=PredictionResponse)
def predict_rul(engine_id: str, db: Session = Depends(get_db)):
    count = crud.get_readings_count(db, engine_id)

    if count < MIN_READINGS:
        raise HTTPException(
            status_code=404,
            detail=f"not enough readings for engine {engine_id} (have {count}, need {MIN_READINGS})"
        )

    readings = crud.get_last_n_readings(db, engine_id, n=N_READINGS)

    try:
        result = feat.predict_from_readings(readings)
    except Exception as e:
        logger.exception(f"Prediction failed for {engine_id}")
        raise HTTPException(status_code=500, detail=f"model failed: {str(e)}")

    response = _build_prediction_response(engine_id, result["rul"], readings, model_name="ridge")
    _save_prediction(db, engine_id, response, model_name="ridge")

    return response


@app.get("/predict/{engine_id}/model/{model_name}", response_model=PredictionResponse)
def predict_with_model(engine_id: str, model_name: str, db: Session = Depends(get_db)):
    if model_name not in MODEL_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {model_name}. Available: {list(MODEL_MAP.keys())}"
        )

    count = crud.get_readings_count(db, engine_id)
    if count < MIN_READINGS:
        raise HTTPException(
            status_code=404,
            detail=f"not enough readings for engine {engine_id}"
        )

    readings = crud.get_last_n_readings(db, engine_id, n=N_READINGS)

    try:
        result = feat.predict_rul_single(readings, model_key=MODEL_MAP[model_name])
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"model failed: {str(e)}")

    confidence = 75.0
    if model_name == "ngboost" and result.get("uncertainty"):
        confidence = round(max(50.0, 100.0 - result["uncertainty"]["std"] * 2), 1)

    health_score = round(min(100.0, (result["rul"] / 125.0) * 100), 1)
    failure_mode = feat.get_failure_mode(readings)
    status = "NORMAL" if result["rul"] >= 50 else "WARNING" if result["rul"] >= 20 else "CRITICAL"

    response = PredictionResponse(
        engine_id=engine_id.upper(),
        rul=result["rul"],
        confidence=confidence,
        failure_mode=failure_mode,
        health_score=health_score,
        status=status,
        model_used=model_name,
    )

    _save_prediction(db, engine_id, response, model_name=model_name)
    return response


@app.post("/compare", response_model=ModelComparisonResponse)
def compare_with_body(request: CompareRequest, db: Session = Depends(get_db)):
    engine_id = request.engine_id

    count = crud.get_readings_count(db, engine_id)
    if count < MIN_READINGS:
        raise HTTPException(
            status_code=404,
            detail=f"not enough readings for engine {engine_id} (have {count}, need {MIN_READINGS})"
        )

    readings = crud.get_last_n_readings(db, engine_id, n=N_READINGS)

    try:
        comparison = feat.predict_all_models(readings)
    except Exception as e:
        logger.exception(f"Comparison failed for {engine_id}")
        raise HTTPException(status_code=500, detail=f"model comparison failed: {str(e)}")

    predictions = []
    for model_key, pred in comparison["predictions"].items():
        entry = {
            "model_name": model_key.replace("_model", "").replace("_", "-").upper(),
            "model_key": model_key,
        }
        if "error" in pred:
            entry["status"] = "error"
            entry["error"] = pred["error"]
        else:
            entry["status"] = "success"
            entry["rul"] = pred["rul"]
            if pred.get("uncertainty"):
                entry["uncertainty"] = pred["uncertainty"]
        predictions.append(entry)

    successful = [p for p in predictions if p.get("status") == "success"]
    recommendation = None
    if successful:
        ngboost_pred = next((p for p in successful if p["model_key"] == "ngb_model"), None)
        if ngboost_pred:
            recommendation = {
                "recommended_model": "NGBoost",
                "reason": "Provides uncertainty quantification for risk-aware maintenance",
                "rul": ngboost_pred["rul"],
            }
        else:
            safest = min(successful, key=lambda x: x["rul"])
            recommendation = {
                "recommended_model": safest["model_name"],
                "reason": "Most conservative estimate (lowest predicted RUL)",
                "rul": safest["rul"],
            }

    return ModelComparisonResponse(
        engine_id=engine_id.upper(),
        consensus_rul=comparison.get("consensus_rul"),
        overall_status=comparison.get("status"),
        n_models_evaluated=comparison.get("n_models_ran", 0),
        predictions=predictions,
        recommendation=recommendation,
    )


@app.get("/recommend/{engine_id}", response_model=MaintenanceRecommendation)
def get_recommendation(engine_id: str, db: Session = Depends(get_db)):
    """
    بيعمل model comparison + sensor trend analysis + Gemini bilingual report
    ويرجع توصية maintenance مفصّلة مع critical sensors
    """
    count = crud.get_readings_count(db, engine_id)
    if count < MIN_READINGS:
        raise HTTPException(
            status_code=404,
            detail=f"not enough readings for engine {engine_id}"
        )

    readings = crud.get_last_n_readings(db, engine_id, n=N_READINGS)

    # 1. شغّلي الـ comparison
    try:
        comparison = feat.predict_all_models(readings)
    except Exception as e:
        logger.exception(f"Comparison failed for {engine_id}")
        raise HTTPException(status_code=500, detail=f"model comparison failed: {str(e)}")

    # 2. حوّلي النتيجة لـ format مناسب للـ gemini_advisor
    predictions_list = []
    for model_key, pred in comparison["predictions"].items():
        entry = {
            "model_name": model_key.replace("_model", "").replace("_", "-").upper(),
            "model_key": model_key,
        }
        if "error" in pred:
            entry["status"] = "error"
            entry["error"] = pred["error"]
        else:
            entry["status"] = "success"
            entry["rul"] = pred["rul"]
            if pred.get("uncertainty"):
                entry["uncertainty"] = pred["uncertainty"]
        predictions_list.append(entry)

    successful = [p for p in predictions_list if p.get("status") == "success"]
    ngboost_pred = next((p for p in successful if p["model_key"] == "ngb_model"), None)
    recommendation_info = {
        "recommended_model": "NGBoost" if ngboost_pred else (successful[0]["model_name"] if successful else "N/A"),
        "rul": ngboost_pred["rul"] if ngboost_pred else (successful[0]["rul"] if successful else 0)
    }

    comparison_for_gemini = {
        "predictions": predictions_list,
        "consensus_rul": comparison.get("consensus_rul"),
        "overall_status": comparison.get("status"),
        "recommendation": recommendation_info,
    }

    # 3. ابعتي لـ Gemini مع sensor trends (الجديد)
    try:
        result = gemini_advisor.get_full_engine_analysis(
            engine_id=engine_id.upper(),
            readings=readings,
            comparison_result=comparison_for_gemini,
        )
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception(f"Gemini recommendation failed for {engine_id}")
        raise HTTPException(status_code=500, detail=f"Gemini failed: {str(e)}")

    # 4. Map analysis fields to recommendation schema for backward compatibility
    # get_full_engine_analysis returns report_en/ar but MaintenanceRecommendation expects recommendation_en/ar
    mapped_result = {
        "engine_id": result["engine_id"],
        "recommended_model": result["recommended_model"],
        "consensus_rul": result["consensus_rul"],
        "risk_level": result["risk_level"],
        "urgency": result["urgency"],
        # Map report -> recommendation for schema compatibility
        "recommendation": result.get("report", result.get("recommendation", "")),
        "recommendation_en": result.get("report_en", ""),
        "recommendation_ar": result.get("report_ar", ""),
        "actions": result.get("actions", []),
        "actions_en": result.get("actions_en", []),
        "actions_ar": result.get("actions_ar", []),
        "model_insight": result.get("model_insight", ""),
        "model_insight_en": result.get("model_insight_en", ""),
        "model_insight_ar": result.get("model_insight_ar", ""),
        "generated_by": result.get("generated_by", ""),
        # Include critical sensors from analysis
        "critical_sensors": result.get("critical_sensors", []),
    }

    return MaintenanceRecommendation(**mapped_result)


@app.get("/predict/{engine_id}/compare", response_model=ModelComparisonResponse)
def compare_models(engine_id: str, db: Session = Depends(get_db)):
    count = crud.get_readings_count(db, engine_id)
    if count < MIN_READINGS:
        raise HTTPException(
            status_code=404,
            detail=f"not enough readings for engine {engine_id} (have {count}, need {MIN_READINGS})"
        )

    readings = crud.get_last_n_readings(db, engine_id, n=N_READINGS)

    try:
        comparison = feat.predict_all_models(readings)
    except Exception as e:
        logger.exception(f"Comparison failed for {engine_id}")
        raise HTTPException(status_code=500, detail=f"model comparison failed: {str(e)}")

    predictions = []
    for model_key, pred in comparison["predictions"].items():
        entry = {
            "model_name": model_key.replace("_model", "").replace("_", "-").upper(),
            "model_key": model_key,
        }
        if "error" in pred:
            entry["status"] = "error"
            entry["error"] = pred["error"]
        else:
            entry["status"] = "success"
            entry["rul"] = pred["rul"]
            if pred.get("uncertainty"):
                entry["uncertainty"] = pred["uncertainty"]
        predictions.append(entry)

    successful = [p for p in predictions if p.get("status") == "success"]
    recommendation = None
    if successful:
        ngboost_pred = next((p for p in successful if p["model_key"] == "ngb_model"), None)
        if ngboost_pred:
            recommendation = {
                "recommended_model": "NGBoost",
                "reason": "Provides uncertainty quantification for risk-aware maintenance",
                "rul": ngboost_pred["rul"],
            }
        else:
            safest = min(successful, key=lambda x: x["rul"])
            recommendation = {
                "recommended_model": safest["model_name"],
                "reason": "Most conservative estimate (lowest predicted RUL)",
                "rul": safest["rul"],
            }

    return ModelComparisonResponse(
        engine_id=engine_id.upper(),
        consensus_rul=comparison.get("consensus_rul"),
        overall_status=comparison.get("status"),
        n_models_evaluated=comparison.get("n_models_ran", 0),
        predictions=predictions,
        recommendation=recommendation,
    )


@app.get("/history/{engine_id}", response_model=list[PredictionHistoryItem])
def get_history(engine_id: str, db: Session = Depends(get_db)):
    try:
        predictions = crud.get_prediction_history(db, engine_id)
        n_predictions = len(predictions)

        if n_predictions == 0:
            return []

        readings = crud.get_last_n_readings(db, engine_id, n=n_predictions)

        history = []
        for i, p in enumerate(reversed(predictions)):
            cycle = readings[i].cycle if i < len(readings) else i
            history.append(PredictionHistoryItem(cycle=cycle, rul=p.rul))
        return history
    except Exception as e:
        logger.exception(f"Error in get_history for {engine_id}")
        raise HTTPException(status_code=500, detail=f"History error: {str(e)}")


@app.get("/analyze/{engine_id}")
def full_engine_analysis(engine_id: str, db: Session = Depends(get_db)):
    """
    التحليل الكامل: sensor trends + model comparison + Gemini bilingual report
    """
    count = crud.get_readings_count(db, engine_id)
    if count < MIN_READINGS:
        raise HTTPException(
            status_code=404,
            detail=f"not enough readings for engine {engine_id}"
        )

    readings = crud.get_last_n_readings(db, engine_id, n=N_READINGS)

    try:
        comparison = feat.predict_all_models(readings)
    except Exception as e:
        logger.exception(f"Comparison failed for {engine_id}")
        raise HTTPException(status_code=500, detail=f"model comparison failed: {str(e)}")

    predictions_list = []
    for model_key, pred in comparison["predictions"].items():
        entry = {
            "model_name": model_key.replace("_model", "").replace("_", "-").upper(),
            "model_key":  model_key,
        }
        if "error" in pred:
            entry["status"] = "error"
            entry["error"]  = pred["error"]
        else:
            entry["status"] = "success"
            entry["rul"]    = pred["rul"]
            if pred.get("uncertainty"):
                entry["uncertainty"] = pred["uncertainty"]
        predictions_list.append(entry)

    successful   = [p for p in predictions_list if p.get("status") == "success"]
    ngb_pred     = next((p for p in successful if p["model_key"] == "ngb_model"), None)
    rec_info     = {
        "recommended_model": "NGBoost" if ngb_pred else (successful[0]["model_name"] if successful else "N/A"),
        "rul": ngb_pred["rul"] if ngb_pred else (successful[0]["rul"] if successful else 0)
    }

    comparison_formatted = {
        "predictions":    predictions_list,
        "consensus_rul":  comparison.get("consensus_rul"),
        "overall_status": comparison.get("status"),
        "recommendation": rec_info,
    }

    try:
        result = gemini_advisor.get_full_engine_analysis(
            engine_id=engine_id.upper(),
            readings=readings,
            comparison_result=comparison_formatted,
        )
    except Exception as e:
        logger.exception(f"Analysis failed for {engine_id}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    return result