# backend/crud.py
from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from models import SensorReading, Prediction
from schemas import SensorReadingCreate


def create_sensor_reading(db: Session, data: SensorReadingCreate) -> SensorReading:
    reading = SensorReading(
        engine_id=data.engine_id,
        cycle=data.cycle,
        op_setting_1=data.op_setting_1,
        op_setting_2=data.op_setting_2,
        op_setting_3=data.op_setting_3,
        sensor_1=data.sensor_1,
        sensor_2=data.sensor_2,
        sensor_3=data.sensor_3,
        sensor_4=data.sensor_4,
        sensor_5=data.sensor_5,
        sensor_6=data.sensor_6,
        sensor_7=data.sensor_7,
        sensor_8=data.sensor_8,
        sensor_9=data.sensor_9,
        sensor_10=data.sensor_10,
        sensor_11=data.sensor_11,
        sensor_12=data.sensor_12,
        sensor_13=data.sensor_13,
        sensor_14=data.sensor_14,
        sensor_15=data.sensor_15,
        sensor_16=data.sensor_16,
        sensor_17=data.sensor_17,
        sensor_18=data.sensor_18,
        sensor_19=data.sensor_19,
        sensor_20=data.sensor_20,
        sensor_21=data.sensor_21,
    )
    db.add(reading)
    db.commit()
    db.refresh(reading)
    return reading


def get_readings_count(db: Session, engine_id: str) -> int:
    return db.query(SensorReading).filter(
        SensorReading.engine_id == engine_id
    ).count()


def get_last_n_readings(db: Session, engine_id: str, n: int = 40):
    readings = db.query(SensorReading).filter(
        SensorReading.engine_id == engine_id
    ).order_by(
        desc(SensorReading.cycle)
    ).limit(n).all()
    return list(reversed(readings))


def get_all_engines(db: Session):
    from sqlalchemy import distinct
    result = db.query(distinct(SensorReading.engine_id)).all()
    return [r[0] for r in result]


def get_engines_with_readings_count(db: Session, min_readings: int = 30):
    result = db.query(
        SensorReading.engine_id,
        func.count(SensorReading.id).label('readings_count')
    ).group_by(SensorReading.engine_id).having(
        func.count(SensorReading.id) >= min_readings
    ).all()
    return [{"engine_id": r[0], "readings_count": r[1]} for r in result]


def create_prediction(db: Session, engine_id: str, rul: float, failure_mode: str,
                      confidence: float, health_score: float, status: str,
                      model_used: str = "ridge") -> Prediction:
    prediction = Prediction(
        engine_id=engine_id,
        rul=rul,
        failure_mode=failure_mode,
        confidence=confidence,
        health_score=health_score,
        status=status,
        model_used=model_used,
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)
    return prediction


def get_prediction_history(db: Session, engine_id: str, limit: int = 100):
    return db.query(Prediction).filter(
        Prediction.engine_id == engine_id
    ).order_by(
        desc(Prediction.timestamp)
    ).limit(limit).all()


def get_latest_prediction(db: Session, engine_id: str):
    return db.query(Prediction).filter(
        Prediction.engine_id == engine_id
    ).order_by(
        desc(Prediction.timestamp)
    ).first()


def get_predictions_stats(db: Session, engine_id: str):
    stats = db.query(
        func.avg(Prediction.rul).label('avg_rul'),
        func.min(Prediction.rul).label('min_rul'),
        func.max(Prediction.rul).label('max_rul'),
        func.count(Prediction.id).label('total_predictions')
    ).filter(
        Prediction.engine_id == engine_id
    ).first()
    return {
        "avg_rul": stats.avg_rul,
        "min_rul": stats.min_rul,
        "max_rul": stats.max_rul,
        "total_predictions": stats.total_predictions
    }