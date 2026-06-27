import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# 1. تحديد مسار قاعدة البيانات
# هنقرا المسار من الـ Environment Variables عشان Docker (زي ما DevOps طالب)
# ولو مش موجود، هيعمل ملف محلي اسمه maintenance.db
DATABASE_URL = os.getenv("https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data/resource/5224bcd1-ad61-490b-93b9-2817288accb8", "sqlite:///./maintenance.db")

# 2. إنشاء المحرك (Engine) لـ SQLite
# connect_args={"check_same_thread": False} دي ضرورية جداً مع SQLite في FastAPI
engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)

# 3. إعداد مصنع الجلسات (Session Local)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 4. فانكشن مساعدة (Dependency) الـ Backend هيستخدمها لفتح وقفل الاتصال تلقائياً
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



# استيراد Base من ملف الموديلز اللي انتِ لسه كاتباها
from models import Base

# السطر ده هو السحر اللي بياخد السكيما بتاعتك ويكريتها جداول حقيقية في SQLite
def init_db():
    Base.metadata.create_all(bind=engine)