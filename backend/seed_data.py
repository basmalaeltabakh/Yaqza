# backend/seed_data.py
import sys
import os
sys.path.append(os.path.dirname(__file__))

from database import SessionLocal, engine
from models import Base, SensorReading
import random

Base.metadata.create_all(bind=engine)


def seed():
    db = SessionLocal()

    # امسح الداتا القديمة لو موجودة
    db.query(SensorReading).filter(
        SensorReading.equipment_id == "ENG001"
    ).delete()
    db.commit()

    readings = []
    for cycle in range(1, 51):
        # قراءات واقعية بناءً على CMAPSS ranges
        degradation = cycle / 50.0  # كلما زاد الـ cycle، زاد الـ degradation

        reading = SensorReading(
            equipment_id="ENG001",
            cycle=cycle,
            setting1=round(random.uniform(-0.0087, 0.0087), 4),
            setting2=round(random.uniform(-0.0003, 0.0003), 4),
            setting3=round(random.choice([0.0, 20.0, 25.0, 35.0, 42.0, 100.0]), 1),
            sensor1=round(518.67 + random.uniform(-0.5, 0.5), 2),
            sensor2=round(642.68 + degradation * 5 + random.uniform(-1, 1), 2),
            sensor3=round(1583.4 + degradation * 10 + random.uniform(-2, 2), 2),
            sensor4=round(1400.0 + degradation * 8 + random.uniform(-2, 2), 2),
            sensor5=round(14.62 + random.uniform(-0.1, 0.1), 2),
            sensor6=round(21.61 + random.uniform(-0.1, 0.1), 2),
            sensor7=round(554.36 + degradation * 3 + random.uniform(-1, 1), 2),
            sensor8=round(2388.0 + random.uniform(-5, 5), 2),
            sensor9=round(9050.0 + random.uniform(-10, 10), 2),
            sensor10=round(1.3 + random.uniform(-0.01, 0.01), 3),
            sensor11=round(47.47 + degradation * 2 + random.uniform(-0.5, 0.5), 2),
            sensor12=round(521.66 + degradation * 4 + random.uniform(-1, 1), 2),
            sensor13=round(2388.0 + random.uniform(-5, 5), 2),
            sensor14=round(8138.0 + random.uniform(-10, 10), 2),
            sensor15=round(8.4195 + random.uniform(-0.05, 0.05), 4),
            sensor16=round(0.03 + random.uniform(-0.001, 0.001), 4),
            sensor17=round(392.0 + degradation * 2 + random.uniform(-1, 1), 2),
            sensor18=round(2388.0 + random.uniform(-5, 5), 2),
            sensor19=round(100.0 + random.uniform(-0.5, 0.5), 2),
            sensor20=round(38.86 + degradation + random.uniform(-0.5, 0.5), 2),
            sensor21=round(23.42 + degradation + random.uniform(-0.3, 0.3), 2),
        )
        readings.append(reading)

    db.add_all(readings)
    db.commit()
    db.close()

    print(f"✅ Seeded 50 readings for ENG001 successfully")


if __name__ == "__main__":
    seed()