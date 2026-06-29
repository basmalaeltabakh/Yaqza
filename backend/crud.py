# backend/crud.py
from sqlalchemy.orm import Session
from sqlalchemy import desc
from models import SensorReading, Prediction
from schemas import SensorReadingCreate
import datetime


# ─────────────────────────────────────────
# Sensor Readings
# ─────────────────────────────────────────

def create_sensor_reading(db: Session, data: SensorReadingCreate) -> SensorReading:
    """حفظ قراءة sensor جديدة في الـ DB"""
    sensor_fields = {
        f"sensor{i}": getattr(data, f"sensor{i}", None)
        for i in range(1, 22)
    }

    reading = SensorReading(
        equipment_id=data.equipment_id.upper(),
        cycle=data.cycle,
        setting1=data.setting1,
        setting2=data.setting2,
        setting3=data.setting3,
        **sensor_fields
    )

    db.add(reading)
    db.commit()
    db.refresh(reading)
    return reading


def get_last_n_readings(
    db: Session,
    equipment_id: str,
    n: int = 30
) -> list[SensorReading]:
    """آخر N قراءة للـ equipment مرتبة من الأقدم للأحدث"""
    rows = (
        db.query(SensorReading)
        .filter(SensorReading.equipment_id == equipment_id.upper())
        .order_by(desc(SensorReading.cycle))
        .limit(n)
        .all()
    )
    return list(reversed(rows))  # الأقدم أول عشان C يحسب الـ trend صح


def get_readings_count(db: Session, equipment_id: str) -> int:
    """عدد القراءات الكلي للـ equipment"""
    return (
        db.query(SensorReading)
        .filter(SensorReading.equipment_id == equipment_id.upper())
        .count()
    )


# ─────────────────────────────────────────
# Predictions
# ─────────────────────────────────────────

def create_prediction(
    db: Session,
    equipment_id: str,
    rul: float,
    failure_mode: str,
    confidence: float
) -> Prediction:
    """حفظ prediction جديدة"""
    pred = Prediction(
        equipment_id=equipment_id.upper(),
        rul=rul,
        failure_mode=failure_mode,
        confidence=confidence,
        timestamp=datetime.datetime.utcnow()
    )

    db.add(pred)
    db.commit()
    db.refresh(pred)
    return pred


def get_prediction_history(
    db: Session,
    equipment_id: str,
    limit: int = 100
) -> list[Prediction]:
    """آخر N predictions للـ equipment"""
    return (
        db.query(Prediction)
        .filter(Prediction.equipment_id == equipment_id.upper())
        .order_by(desc(Prediction.timestamp))
        .limit(limit)
        .all()
    )