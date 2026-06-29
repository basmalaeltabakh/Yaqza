from sqlalchemy import Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base
import datetime

Base = declarative_base()


class SensorReading(Base):
    __tablename__ = "sensor_readings"

    id = Column(Integer, primary_key=True, index=True)
    equipment_id = Column(String, index=True)
    cycle = Column(Integer)

    # Operational Settings
    setting1 = Column(Float)
    setting2 = Column(Float)
    setting3 = Column(Float)

    # 21 Sensors 
    sensor1  = Column(Float)   # Total temperature fan inlet
    sensor2  = Column(Float)   # Total temperature LPC outlet
    sensor3  = Column(Float)   # Total temperature HPC outlet
    sensor4  = Column(Float)   # Total temperature LPT outlet
    sensor5  = Column(Float)   # Pressure fan inlet
    sensor6  = Column(Float)   # Total pressure fan inlet
    sensor7  = Column(Float)   # Total pressure HPC outlet
    sensor8  = Column(Float)   # Physical fan speed
    sensor9  = Column(Float)   # Physical core speed
    sensor10 = Column(Float)   # Engine pressure ratio
    sensor11 = Column(Float)   # Static pressure HPC outlet
    sensor12 = Column(Float)   # Fuel flow ratio
    sensor13 = Column(Float)   # Corrected fan speed
    sensor14 = Column(Float)   # Corrected core speed
    sensor15 = Column(Float)   # Bypass ratio
    sensor16 = Column(Float)   # Burner fuel-air ratio
    sensor17 = Column(Float)   # Bleed enthalpy
    sensor18 = Column(Float)   # Demanded fan speed
    sensor19 = Column(Float)   # Demanded corrected fan speed
    sensor20 = Column(Float)   # HPT coolant bleed
    sensor21 = Column(Float)   # LPT coolant bleed

    timestamp = Column(DateTime, default=datetime.datetime.utcnow)


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    equipment_id = Column(String, index=True)
    rul = Column(Float)
    failure_mode = Column(String, nullable=True)
    confidence = Column(Float)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)