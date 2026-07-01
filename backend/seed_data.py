# backend/seed_data.py
import sys
import os
sys.path.append(os.path.dirname(__file__))

from database import SessionLocal, engine
from models import Base, SensorReading
import random

Base.metadata.create_all(bind=engine)

ENGINE_IDS = ["ENG001", "ENG002", "ENG003"]


def seed_engine(db, engine_id: str, base_degradation_speed: float = 1.0):
    """بتعمل seed لمحرك واحد بـ 50 قراءة"""
    db.query(SensorReading).filter(
        SensorReading.engine_id == engine_id
    ).delete()
    db.commit()

    readings = []
    for cycle in range(1, 51):
        degradation = (cycle / 50.0) * base_degradation_speed

        reading = SensorReading(
            engine_id=engine_id,
            cycle=cycle,
            op_setting_1=round(random.uniform(-0.0087, 0.0087), 4),
            op_setting_2=round(random.uniform(-0.0003, 0.0003), 4),
            op_setting_3=round(random.choice([0.0, 20.0, 25.0, 35.0, 42.0, 100.0]), 1),
            sensor_1=round(518.67 + random.uniform(-0.5, 0.5), 2),
            sensor_2=round(642.68 + degradation * 5 + random.uniform(-1, 1), 2),
            sensor_3=round(1583.4 + degradation * 10 + random.uniform(-2, 2), 2),
            sensor_4=round(1400.0 + degradation * 8 + random.uniform(-2, 2), 2),
            sensor_5=round(14.62 + random.uniform(-0.1, 0.1), 2),
            sensor_6=round(21.61 + random.uniform(-0.1, 0.1), 2),
            sensor_7=round(554.36 + degradation * 3 + random.uniform(-1, 1), 2),
            sensor_8=round(2388.0 + random.uniform(-5, 5), 2),
            sensor_9=round(9050.0 + random.uniform(-10, 10), 2),
            sensor_10=round(1.3 + random.uniform(-0.01, 0.01), 3),
            sensor_11=round(47.47 + degradation * 2 + random.uniform(-0.5, 0.5), 2),
            sensor_12=round(521.66 + degradation * 4 + random.uniform(-1, 1), 2),
            sensor_13=round(2388.0 + random.uniform(-5, 5), 2),
            sensor_14=round(8138.0 + random.uniform(-10, 10), 2),
            sensor_15=round(8.4195 + random.uniform(-0.05, 0.05), 4),
            sensor_16=round(0.03 + random.uniform(-0.001, 0.001), 4),
            sensor_17=round(392.0 + degradation * 2 + random.uniform(-1, 1), 2),
            sensor_18=round(2388.0 + random.uniform(-5, 5), 2),
            sensor_19=round(100.0 + random.uniform(-0.5, 0.5), 2),
            sensor_20=round(38.86 + degradation + random.uniform(-0.5, 0.5), 2),
            sensor_21=round(23.42 + degradation + random.uniform(-0.3, 0.3), 2),
        )
        readings.append(reading)

    db.add_all(readings)
    db.commit()
    print(f"✅ Seeded 50 readings for {engine_id}")


def seed():
    db = SessionLocal()

    # كل محرك بسرعة تدهور مختلفة عشان النتايج تبقى متنوعة فالداشبورد
    seed_engine(db, "ENG001", base_degradation_speed=1.0)   # طبيعي
    seed_engine(db, "ENG002", base_degradation_speed=2.2)   # متدهور أسرع (RUL أقل)
    seed_engine(db, "ENG003", base_degradation_speed=0.5)   # سليم نسبياً

    db.close()
    print("\n✅ All engines seeded successfully")


if __name__ == "__main__":
    seed()