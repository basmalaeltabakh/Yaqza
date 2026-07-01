# backend/models.py
from sqlalchemy import Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base
import datetime

Base = declarative_base()


class SensorReading(Base):
    __tablename__ = "sensor_readings"

    id = Column(Integer, primary_key=True, index=True)
    engine_id = Column(String, index=True)
    cycle = Column(Integer)

    op_setting_1 = Column(Float, nullable=True)
    op_setting_2 = Column(Float, nullable=True)
    op_setting_3 = Column(Float, nullable=True)

    sensor_1 = Column(Float, nullable=True)
    sensor_2 = Column(Float, nullable=True)
    sensor_3 = Column(Float, nullable=True)
    sensor_4 = Column(Float, nullable=True)
    sensor_5 = Column(Float, nullable=True)
    sensor_6 = Column(Float, nullable=True)
    sensor_7 = Column(Float, nullable=True)
    sensor_8 = Column(Float, nullable=True)
    sensor_9 = Column(Float, nullable=True)
    sensor_10 = Column(Float, nullable=True)
    sensor_11 = Column(Float, nullable=True)
    sensor_12 = Column(Float, nullable=True)
    sensor_13 = Column(Float, nullable=True)
    sensor_14 = Column(Float, nullable=True)
    sensor_15 = Column(Float, nullable=True)
    sensor_16 = Column(Float, nullable=True)
    sensor_17 = Column(Float, nullable=True)
    sensor_18 = Column(Float, nullable=True)
    sensor_19 = Column(Float, nullable=True)
    sensor_20 = Column(Float, nullable=True)
    sensor_21 = Column(Float, nullable=True)

    timestamp = Column(DateTime, default=datetime.datetime.utcnow)


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    engine_id = Column(String, index=True)
    rul = Column(Float)
    health_score = Column(Float)
    status = Column(String)
    failure_mode = Column(String, nullable=True)
    confidence = Column(Float)
    model_used = Column(String, default="ridge")
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)