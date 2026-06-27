from sqlalchemy import Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base  # التحديث الجديد للـ SQLAlchemy 2.0
import datetime

# إنشاء الـ Base بالطريقة الجديدة لتجنب الـ Warning
Base = declarative_base()


# 1. جدول قراءات الحساسات متوافق مع داتاست NASA CMAPSS
class SensorReading(Base):
    __tablename__ = "sensor_readings"

    id = Column(Integer, primary_key=True, index=True)
    equipment_id = Column(String, index=True)  # رقم المحرك (مثلاً Engine_001)
    cycle = Column(Integer)  # رقم الدورة الزمنية من الداتاست (Cycle)

    # Operational Settings الثلاثة الخاصة بناسا
    setting1 = Column(Float)
    setting2 = Column(Float)
    setting3 = Column(Float)

    # الـ 21 حساس كـ String بصيغة JSON
    sensor_data = Column(String)

    timestamp = Column(DateTime, default=datetime.datetime.utcnow)


# 2. جدول التوقعات (RUL Predictions)
class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    equipment_id = Column(String, index=True)  # رقم المحرك
    rul = Column(Float)  # العمر الافتراضي المتبقي المتوقع (Remaining Useful Life)
    failure_mode = Column(String, nullable=True)  # نوع العطل
    confidence = Column(Float)  # نسبة التأكد من التوقع
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)