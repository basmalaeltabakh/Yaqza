# backend/schemas.py
"""
Pydantic schemas for Yaqza API.
"""
from pydantic import BaseModel
from datetime import datetime
from typing import Optional


# ── Sensor Reading ─────────────────────────────────────────
class SensorReadingCreate(BaseModel):
    engine_id: str
    cycle: int
    setting_1: float = 0.0
    setting_2: float = 0.0
    setting_3: float = 0.0
    sensor_1: float = 0.0
    sensor_2: float = 0.0
    sensor_3: float = 0.0
    sensor_4: float = 0.0
    sensor_5: float = 0.0
    sensor_6: float = 0.0
    sensor_7: float = 0.0
    sensor_8: float = 0.0
    sensor_9: float = 0.0
    sensor_10: float = 0.0
    sensor_11: float = 0.0
    sensor_12: float = 0.0
    sensor_13: float = 0.0
    sensor_14: float = 0.0
    sensor_15: float = 0.0
    sensor_16: float = 0.0
    sensor_17: float = 0.0
    sensor_18: float = 0.0
    sensor_19: float = 0.0
    sensor_20: float = 0.0
    sensor_21: float = 0.0


# ── Prediction Response ────────────────────────────────────
class PredictionResponse(BaseModel):
    engine_id: str
    rul: float
    confidence: float
    failure_mode: str
    health_score: float
    status: str
    model_used: str


# ── Prediction History ───────────────────────────────────
class PredictionHistoryItem(BaseModel):
    cycle: int
    rul: float


# ── Success Response ─────────────────────────────────────
class SuccessResponse(BaseModel):
    id: int
    status: str


# ── Health Response ──────────────────────────────────────
class HealthResponse(BaseModel):
    status: str


# ── Model Comparison ─────────────────────────────────────
class PredictionEntry(BaseModel):
    model_name: str
    model_key: str
    status: str
    rul: Optional[float] = None
    error: Optional[str] = None
    uncertainty: Optional[dict] = None


class ModelRecommendation(BaseModel):
    recommended_model: str
    reason: str
    rul: Optional[float] = None


class ModelComparisonResponse(BaseModel):
    engine_id: str
    consensus_rul: Optional[float] = None
    overall_status: Optional[str] = None
    n_models_evaluated: int = 0
    predictions: list = []
    recommendation: Optional[ModelRecommendation] = None


# ── Available Models ─────────────────────────────────────
class AvailableModelInfo(BaseModel):
    key: str
    name: str
    type: str
    features: str
    description: str
    available: bool


class ModelsListResponse(BaseModel):
    models: list = []
    total_available: int = 0
    total_expected: int = 5


# ── Engines ──────────────────────────────────────────────
class EngineInfo(BaseModel):
    engine_id: str
    readings_count: int = 0
    last_prediction: Optional[datetime] = None
    status: str = "NO_PREDICTION"


class EnginesListResponse(BaseModel):
    engines: list = []
    total: int = 0


# ── Request Bodies ─────────────────────────────────────
class PredictRequest(BaseModel):
    engine_id: str
    model_name: str = "ridge"


class CompareRequest(BaseModel):
    engine_id: str


# ── Maintenance Recommendation (BILINGUAL) ───────────────
class MaintenanceRecommendation(BaseModel):
    engine_id: str
    recommended_model: str
    consensus_rul: float
    risk_level: str
    urgency: str
    # Single fields for backward compatibility (auto-filled from bilingual)
    recommendation: str = ""
    actions: list = []
    model_insight: str = ""
    # Bilingual fields
    recommendation_en: str = ""
    recommendation_ar: str = ""
    actions_en: list = []
    actions_ar: list = []
    model_insight_en: str = ""
    model_insight_ar: str = ""
    generated_by: str = ""


# ── Full Engine Analysis Response (from /analyze endpoint) ───────────────────
class SensorTrend(BaseModel):
    sensor_name: str
    sensor_key: str
    current_value: float
    mean_value: float
    trend: str
    trend_rate: float
    anomaly: bool

class AnalysisResponse(BaseModel):
    engine_id: str
    analysis_timestamp: str
    consensus_rul: float
    risk_level: str
    urgency: str
    recommended_model: str
    critical_sensors: list = []
    stable_sensors: list = []
    total_sensors_analyzed: int = 0
    # Single fields for backward compatibility
    report: str = ""
    actions: list = []
    model_insight: str = ""
    # Bilingual fields
    report_en: str = ""
    report_ar: str = ""
    actions_en: list = []
    actions_ar: list = []
    model_insight_en: str = ""
    model_insight_ar: str = ""
    generated_by: str = ""