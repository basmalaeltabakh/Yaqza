# backend/main.py
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
import os

from database import get_db, init_db
from models import SensorReading, Prediction
from schemas import (
    SensorReadingCreate,
    SensorReadingResponse,
    PredictionResponse,
    PredictionHistoryItem,
    SuccessResponse,
    HealthResponse,
)

init_db()

app = FastAPI(
    title="Yaqza API",
    description="Predictive Maintenance API — CMAPSS Dataset",
    version="0.1.0"
)


# ─────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health_check(db: Session = Depends(get_db)):
    """
    Check whether the API, database, and ML model are available.
    """
    try:
        db.execute(__import__("sqlalchemy").text("SELECT 1"))
        db_status = "ok"
    except Exception:
        db_status = "error"

    model_path = os.getenv("MODEL_PATH", "ml/models/rul_model_v1.pkl")
    model_status = "ok" if os.path.exists(model_path) else "not_loaded_yet"

    return HealthResponse(
        status="ok" if db_status == "ok" else "degraded",
        database=db_status,
        model=model_status
    )


# ─────────────────────────────────────────
# POST /ingest
# ─────────────────────────────────────────
@app.post("/ingest", response_model=SuccessResponse, status_code=201)
def ingest_sensor(data: SensorReadingCreate, db: Session = Depends(get_db)):
    """
    Receive a single sensor reading and store it in the database.
    """
    # TODO (Day 3): Replace with crud.create_sensor_reading()
    sensor_fields = {f"sensor{i}": getattr(data, f"sensor{i}", None) for i in range(1, 22)}

    reading = SensorReading(
        equipment_id=data.equipment_id,
        cycle=data.cycle,
        setting1=data.setting1,
        setting2=data.setting2,
        setting3=data.setting3,
        **sensor_fields
    )

    db.add(reading)
    db.commit()
    db.refresh(reading)

    return SuccessResponse(status="stored", id=reading.id)


# ─────────────────────────────────────────
# POST /predict/{equipment_id}
# ─────────────────────────────────────────
@app.post("/predict/{equipment_id}", response_model=PredictionResponse)
def predict_rul(equipment_id: str, db: Session = Depends(get_db)):
    """
    Retrieve the latest 30 sensor readings and predict the Remaining Useful Life (RUL).
    """
    count = db.query(SensorReading).filter(
        SensorReading.equipment_id == equipment_id.upper()
    ).count()

    if count == 0:
        raise HTTPException(status_code=404, detail=f"Equipment {equipment_id} not found")

    if count < 30:
        raise HTTPException(
            status_code=422,
            detail=f"Need at least 30 readings, only {count} found for {equipment_id}"
        )

    # TODO (Day 4): Integrate extract_features() + model.predict()
    from datetime import datetime
    return PredictionResponse(
        equipment_id=equipment_id.upper(),
        rul_prediction=-1.0,
        failure_mode="pending",
        confidence=0.0,
        timestamp=datetime.utcnow()
    )


# ─────────────────────────────────────────
# GET /history/{equipment_id}
# ─────────────────────────────────────────
@app.get("/history/{equipment_id}", response_model=list[PredictionHistoryItem])
def get_history(equipment_id: str, limit: int = 100, db: Session = Depends(get_db)):
    """
    Retrieve the most recent prediction history for the specified equipment.
    """
    # TODO (Day 5): Replace with crud.get_prediction_history()
    return []