# backend/schemas.py
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime


class SensorReadingCreate(BaseModel):
    engine_id: str
    cycle: int
    op_setting_1: Optional[float] = 0.0
    op_setting_2: Optional[float] = 0.0
    op_setting_3: Optional[float] = 0.0
    sensor_1: Optional[float] = 0.0
    sensor_2: Optional[float] = 0.0
    sensor_3: Optional[float] = 0.0
    sensor_4: Optional[float] = 0.0
    sensor_5: Optional[float] = 0.0
    sensor_6: Optional[float] = 0.0
    sensor_7: Optional[float] = 0.0
    sensor_8: Optional[float] = 0.0
    sensor_9: Optional[float] = 0.0
    sensor_10: Optional[float] = 0.0
    sensor_11: Optional[float] = 0.0
    sensor_12: Optional[float] = 0.0
    sensor_13: Optional[float] = 0.0
    sensor_14: Optional[float] = 0.0
    sensor_15: Optional[float] = 0.0
    sensor_16: Optional[float] = 0.0
    sensor_17: Optional[float] = 0.0
    sensor_18: Optional[float] = 0.0
    sensor_19: Optional[float] = 0.0
    sensor_20: Optional[float] = 0.0
    sensor_21: Optional[float] = 0.0


class SensorReadingResponse(BaseModel):
    id: int
    engine_id: str
    cycle: int
    op_setting_1: Optional[float]
    op_setting_2: Optional[float]
    op_setting_3: Optional[float]
    sensor_1: Optional[float]
    sensor_2: Optional[float]
    sensor_3: Optional[float]
    sensor_4: Optional[float]
    sensor_5: Optional[float]
    sensor_6: Optional[float]
    sensor_7: Optional[float]
    sensor_8: Optional[float]
    sensor_9: Optional[float]
    sensor_10: Optional[float]
    sensor_11: Optional[float]
    sensor_12: Optional[float]
    sensor_13: Optional[float]
    sensor_14: Optional[float]
    sensor_15: Optional[float]
    sensor_16: Optional[float]
    sensor_17: Optional[float]
    sensor_18: Optional[float]
    sensor_19: Optional[float]
    sensor_20: Optional[float]
    sensor_21: Optional[float]
    timestamp: Optional[datetime] = None

    class Config:
        from_attributes = True


class PredictionResponse(BaseModel):
    engine_id: str
    rul: float
    confidence: float
    failure_mode: str
    health_score: float
    status: str
    model_used: Optional[str] = "ridge"


class PredictionHistoryItem(BaseModel):
    cycle: int
    rul: float


class PredictionRecordResponse(BaseModel):
    id: int
    engine_id: str
    rul: float
    failure_mode: str
    confidence: float
    health_score: float
    status: str
    model_used: Optional[str] = "ridge"
    timestamp: Optional[datetime] = None

    class Config:
        from_attributes = True


class ModelPredictionDetail(BaseModel):
    model_name: str
    model_key: str
    status: str
    rul: Optional[float] = None
    uncertainty: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ModelRecommendation(BaseModel):
    recommended_model: str
    reason: str
    rul: float


class ModelComparisonResponse(BaseModel):
    engine_id: str
    consensus_rul: Optional[float]
    overall_status: str
    n_models_evaluated: int
    predictions: List[ModelPredictionDetail]
    recommendation: Optional[ModelRecommendation]


class AvailableModelInfo(BaseModel):
    key: str
    name: str
    type: str
    features: str
    description: str
    available: bool


class AvailableModelsResponse(BaseModel):
    models: List[AvailableModelInfo]
    total_available: int
    total_expected: int


class EngineInfo(BaseModel):
    engine_id: str
    readings_count: Optional[int] = None
    last_prediction: Optional[datetime] = None
    status: Optional[str] = None


class EnginesListResponse(BaseModel):
    engines: List[EngineInfo]
    total: int


class ModelsListResponse(BaseModel):
    models: List[AvailableModelInfo]
    total_available: int
    total_expected: int


class PredictRequest(BaseModel):
    engine_id: str
    model_name: str = "ridge"


class CompareRequest(BaseModel):
    engine_id: str


class SuccessResponse(BaseModel):
    id: int
    status: str


class HealthResponse(BaseModel):
    status: str


class ErrorResponse(BaseModel):
    detail: str
    
class MaintenanceRecommendation(BaseModel):
    engine_id: str
    recommended_model: str
    consensus_rul: float
    risk_level: str
    recommendation: str
    actions: List[str]
    urgency: str
    model_insight: Optional[str] = None
    generated_by: str = "gemini-3-flash-preview"