from pydantic import BaseModel, Field, validator
from typing import Optional
from datetime import datetime


# ── Input schema: Sensor data received from the equipment ─────────────
class SensorReadingCreate(BaseModel):
    equipment_id: str = Field(..., example="ENG001")
    sensor_name: str = Field(..., example="sensor_2")
    value: float = Field(..., example=0.567)
    cycle: int = Field(..., gt=0, example=145)

    @validator("equipment_id")
    def equipment_id_not_empty(cls, v):
        if not v.strip():
            raise ValueError("equipment_id cannot be empty")
        return v.upper()


# ── Response schema: Stored sensor reading ────────────────────────────
class SensorReadingResponse(BaseModel):
    id: int
    equipment_id: str
    sensor_name: str
    value: float
    cycle: int
    timestamp: datetime

    class Config:
        from_attributes = True


# ── Request schema: Prediction request for a specific equipment ───────
class PredictionRequest(BaseModel):
    equipment_id: str = Field(..., example="ENG001")


# ── Response schema: Predicted Remaining Useful Life (RUL) ────────────
class PredictionResponse(BaseModel):
    equipment_id: str
    rul_prediction: float = Field(..., description="Remaining Useful Life in cycles")
    failure_mode: str = Field(..., example="bearing_wear")
    confidence: float = Field(..., ge=0.0, le=1.0, example=0.85)
    timestamp: datetime

    class Config:
        from_attributes = True


# ── Response schema: Prediction history records ───────────────────────
class PredictionHistoryItem(BaseModel):
    timestamp: datetime
    rul_prediction: float
    failure_mode: str
    confidence: float

    class Config:
        from_attributes = True


# ── Generic API response schemas ──────────────────────────────────────
class SuccessResponse(BaseModel):
    status: str = "stored"
    id: int


class HealthResponse(BaseModel):
    status: str
    database: str
    model: str