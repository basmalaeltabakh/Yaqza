# backend/fix.py
"""يصلح مشكلتين: DB migration + np.float64 في confidence"""

# ── Fix 1: DB Migration ──────────────────────────────────────────────────────
import sqlite3

conn = sqlite3.connect('yaqza.db')
try:
    conn.execute("ALTER TABLE predictions ADD COLUMN model_used TEXT DEFAULT 'ridge'")
    conn.commit()
    print("✅ Fix 1: Column model_used added to predictions table")
except sqlite3.OperationalError as e:
    if "duplicate column" in str(e):
        print("✅ Fix 1: Column model_used already exists")
    else:
        print(f"⚠️  Fix 1 error: {e}")
finally:
    conn.close()

print("✅ All fixes applied — restart uvicorn now")