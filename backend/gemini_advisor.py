# backend/gemini_advisor.py
"""
Gemini Advisor — Pure HTTP, zero Google SDK.
Full bilingual analysis: sensor trends + model comparison.
"""
import os
import json
import time
import httpx
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from models import SensorReading

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME     = "gemini-3-flash-preview"
BASE_URL       = "https://generativelanguage.googleapis.com/v1beta/models"

# الـ sensors المهمة في CMAPSS مع أسمائها الحقيقية
SENSOR_NAMES = {
    "sensor_2":  "LPC Outlet Temperature (deg R)",
    "sensor_3":  "HPC Outlet Temperature (deg R)",
    "sensor_4":  "LPT Outlet Temperature (deg R)",
    "sensor_7":  "HPC Outlet Pressure (psia)",
    "sensor_8":  "Physical Fan Speed (rpm)",
    "sensor_9":  "Physical Core Speed (rpm)",
    "sensor_11": "HPC Outlet Static Pressure (psia)",
    "sensor_12": "Fuel Flow Ratio (pps/psi)",
    "sensor_13": "Corrected Fan Speed (rpm)",
    "sensor_14": "Corrected Core Speed (rpm)",
    "sensor_15": "Bypass Ratio",
    "sensor_17": "Bleed Enthalpy",
    "sensor_20": "HPT Coolant Bleed (lbm/s)",
    "sensor_21": "LPT Coolant Bleed (lbm/s)",
}
MAX_RETRIES = 3
BASE_DELAY  = 2.0


# ═══════════════════════════════════════════════════════
# HTTP CALL
# ═══════════════════════════════════════════════════════

def _call_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set in .env")

    url = f"{BASE_URL}/{MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 2048},
    }

    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            with httpx.Client(timeout=45.0) as client:
                r = client.post(url, json=payload)
                r.raise_for_status()
                return r.json()["candidates"][0]["content"]["parts"][0]["text"]
        except httpx.HTTPStatusError as exc:
            last_exc = exc
            if exc.response.status_code in (429, 503):
                wait = BASE_DELAY * (2 ** attempt)
                print(f"[Gemini] {exc.response.status_code} retry {attempt+1}/{MAX_RETRIES} in {wait}s")
                time.sleep(wait)
            else:
                raise
        except Exception:
            raise
    raise last_exc


def _parse_json(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text  = "\n".join(lines[1:-1]).strip()
    return json.loads(text)


# ═══════════════════════════════════════════════════════
# SENSOR TREND ANALYSIS
# ═══════════════════════════════════════════════════════

def analyze_sensor_trends(readings: list) -> dict:
    """بتحلل الـ sensors وبتطلع trends وanomalies."""
    if not readings:
        return {"critical": [], "stable": [], "total": 0}

    critical_sensors = []
    stable_sensors   = []

    for sensor_key, sensor_display in SENSOR_NAMES.items():
        values = [getattr(r, sensor_key, None) for r in readings]
        values = [v for v in values if v is not None and v != 0.0]

        if len(values) < 5:
            continue

        arr        = np.array(values, dtype=float)
        mean_val   = float(np.mean(arr))
        std_val    = float(np.std(arr))
        latest_val = float(arr[-1])

        # حساب الـ trend بـ linear regression
        x         = np.arange(len(arr))
        trend_rate = float(np.polyfit(x, arr, 1)[0])

        # تحديد اتجاه الـ trend
        if abs(trend_rate) < 0.001 * abs(mean_val):
            trend_dir = "stable"
        elif trend_rate > 0:
            trend_dir = "increasing"
        else:
            trend_dir = "decreasing"

        # كشف الـ anomaly لو القيمة الأخيرة خارج نطاق 2 std
        anomaly = abs(latest_val - mean_val) > 2 * std_val if std_val > 0 else False

        sensor_info = {
            "sensor_name":    sensor_display,
            "sensor_key":     sensor_key,
            "current_value":  round(latest_val, 3),
            "mean_value":     round(mean_val, 3),
            "trend":          trend_dir,
            "trend_rate":     round(trend_rate, 4),
            "anomaly":        anomaly,
        }

        if anomaly or trend_dir != "stable":
            critical_sensors.append(sensor_info)
        else:
            stable_sensors.append(sensor_info)

    # رتّبي الـ critical بالأكثر تأثيراً أول
    critical_sensors.sort(key=lambda x: abs(x["trend_rate"]), reverse=True)

    return {
        "critical": critical_sensors[:6],  # أهم 6 sensors
        "stable":   stable_sensors,
        "total":    len(SENSOR_NAMES),
    }


# ═══════════════════════════════════════════════════════
# MAIN ANALYSIS FUNCTION
# ═══════════════════════════════════════════════════════

def get_full_engine_analysis(
    engine_id: str,
    readings: list,
    comparison_result: dict,
) -> dict:
    """
    التحليل الكامل: sensor trends + model comparison + Gemini bilingual report.
    """

    # 1. حلّلي الـ sensor trends
    trends = analyze_sensor_trends(readings)

    # 2. جهّزي ملخص الموديلز
    model_lines = []
    for pred in comparison_result.get("predictions", []):
        name = pred.get("model_name", "?")
        if pred.get("status") == "success":
            line = f"  • {name}: {pred['rul']} cycles"
            unc  = pred.get("uncertainty")
            if unc:
                line += f" ± {unc.get('std','?')} (90% CI: {unc.get('ci_90',['?','?'])[0]}–{unc.get('ci_90',['?','?'])[1]})"
            model_lines.append(line)
        else:
            model_lines.append(f"  • {name}: FAILED")

    consensus_rul = float(comparison_result.get("consensus_rul", 0))
    rec_info      = comparison_result.get("recommendation", {})
    best_model    = rec_info.get("recommended_model", "N/A")
    best_rul      = rec_info.get("rul", "N/A")
    status        = comparison_result.get("overall_status", "NORMAL")

    # 3. جهّزي ملخص الـ sensors
    sensor_lines = []
    for s in trends["critical"]:
        direction_ar = "مرتفع ↑" if s["trend"] == "increasing" else "منخفض ↓" if s["trend"] == "decreasing" else "مستقر"
        anomaly_flag = " ⚠️ ANOMALY" if s["anomaly"] else ""
        sensor_lines.append(
            f"  • {s['sensor_name']}: {s['current_value']} (trend: {s['trend']}, rate: {s['trend_rate']}/cycle){anomaly_flag}"
        )

    sensors_block = "\n".join(sensor_lines) if sensor_lines else "  All sensors within normal range."

    # 4. risk level
    if consensus_rul >= 100:
        risk, urgency = "LOW", "SCHEDULED"
    elif consensus_rul >= 50:
        risk, urgency = "MEDIUM", "WITHIN_WEEK"
    elif consensus_rul >= 20:
        risk, urgency = "HIGH", "WITHIN_24H"
    else:
        risk, urgency = "CRITICAL", "IMMEDIATE"

    # 5. Gemini prompt
    prompt = f"""You are a senior jet-engine predictive maintenance engineer and technical report writer.

You have received a full diagnostic report for engine {engine_id} from a multi-model AI system.

══════════════════════════════════════════════
ENGINE STATUS SUMMARY
══════════════════════════════════════════════
Engine ID      : {engine_id}
Risk Level     : {risk}
Urgency        : {urgency}
Consensus RUL  : {consensus_rul} cycles remaining
Best Model     : {best_model} → {best_rul} cycles

MODEL PREDICTIONS (5 ML Models)
{chr(10).join(model_lines)}

SENSOR TREND ANALYSIS ({len(trends['critical'])} sensors showing changes)
{sensors_block}

Stable sensors: {len(trends['stable'])} sensors within normal parameters
══════════════════════════════════════════════

TASK: Generate a complete bilingual diagnostic report.

Return ONLY a valid JSON object with NO markdown fences:

{{
  "report_en": "A professional 4-5 sentence report in English for the factory owner. Cover: (1) overall engine health summary, (2) what the sensor trends mean physically, (3) model agreement/disagreement analysis, (4) business impact if ignored, (5) recommended next step.",

  "report_ar": "تقرير مهني من 4-5 جمل بالعربية لصاحب المصنع. يغطي: (1) ملخص الحالة العامة للمحرك، (2) ماذا تعني قراءات الحساسات من الناحية الفنية، (3) تحليل اتفاق أو اختلاف النماذج، (4) الأثر التجاري إذا تم تجاهله، (5) الخطوة التالية الموصى بها.",

  "actions_en": [
    "Immediate action with specific component or sensor name",
    "Inspection step with timeline",
    "Maintenance decision based on RUL",
    "Documentation or escalation requirement",
    "Follow-up monitoring recommendation"
  ],

  "actions_ar": [
    "إجراء فوري مع اسم المكون أو الحساس المحدد",
    "خطوة فحص مع الجدول الزمني",
    "قرار صيانة بناءً على العمر الافتراضي المتبقي",
    "متطلب التوثيق أو التصعيد",
    "توصية المراقبة المتابعة"
  ],

  "model_insight_en": "One sentence about model consensus and which to trust most.",
  "model_insight_ar": "جملة واحدة عن اتفاق النماذج وأيها يمكن الوثوق به أكثر."
}}"""

    # 6. كلّمي Gemini
    try:
        raw    = _call_gemini(prompt)
        parsed = _parse_json(raw)
        source = MODEL_NAME
    except Exception as exc:
        print(f"[Gemini] Failed: {exc} — using fallback")
        parsed = _build_fallback(engine_id, consensus_rul, risk, urgency, best_model, trends)
        source = "fallback-rule-engine"

    return {
        "engine_id"             : engine_id.upper(),
        "analysis_timestamp"    : datetime.utcnow().isoformat(),
        "consensus_rul"         : consensus_rul,
        "risk_level"            : risk,
        "urgency"               : urgency,
        "model_predictions"     : comparison_result.get("predictions", []),
        "recommended_model"     : best_model,
        "critical_sensors"      : trends["critical"],
        "stable_sensors"        : trends["stable"],
        "total_sensors_analyzed": trends["total"],
        "report_en"             : parsed.get("report_en", ""),
        "report_ar"             : parsed.get("report_ar", ""),
        "actions_en"            : parsed.get("actions_en", []),
        "actions_ar"            : parsed.get("actions_ar", []),
        "model_insight_en"      : parsed.get("model_insight_en", ""),
        "model_insight_ar"      : parsed.get("model_insight_ar", ""),
        "generated_by"          : source,
    }


# ═══════════════════════════════════════════════════════
# FALLBACK
# ═══════════════════════════════════════════════════════

def _build_fallback(engine_id, consensus_rul, risk, urgency, best_model, trends) -> dict:
    critical_names_en = [s["sensor_name"] for s in trends["critical"][:3]]
    critical_names_ar = critical_names_en

    return {
        "report_en": (
            f"Engine {engine_id} has been assessed with a consensus RUL of {consensus_rul} cycles, "
            f"indicating a {risk} risk level. "
            f"Sensor analysis reveals changes in {', '.join(critical_names_en) if critical_names_en else 'multiple systems'}. "
            f"Immediate action classification: {urgency}. "
            f"Recommend consulting {best_model} model predictions for maintenance scheduling."
        ),
        "report_ar": (
            f"تم تقييم المحرك {engine_id} بعمر افتراضي متبقي قدره {consensus_rul} دورة، "
            f"مما يشير إلى مستوى خطر {risk}. "
            f"كشف تحليل الحساسات عن تغييرات في {', '.join(critical_names_ar) if critical_names_ar else 'أنظمة متعددة'}. "
            f"تصنيف الإجراء الفوري: {urgency}. "
            f"يوصى بمراجعة توقعات نموذج {best_model} لجدولة الصيانة."
        ),
        "actions_en": [
            f"Inspect components related to {critical_names_en[0] if critical_names_en else 'core systems'}.",
            f"Schedule maintenance within timeframe: {urgency.replace('_', ' ')}.",
            "Document all sensor readings and log current engine state.",
            "Consult senior maintenance engineer before next operation.",
            "Re-run prediction after any maintenance action.",
        ],
        "actions_ar": [
            f"فحص المكونات المرتبطة بـ {critical_names_ar[0] if critical_names_ar else 'الأنظمة الأساسية'}.",
            f"جدولة الصيانة خلال: {urgency.replace('_', ' ')}.",
            "توثيق جميع قراءات الحساسات وتسجيل الحالة الحالية للمحرك.",
            "استشارة المهندس الأول قبل أي تشغيل.",
            "إعادة تشغيل التوقع بعد أي إجراء صيانة.",
        ],
        "model_insight_en": f"Fallback analysis — {best_model} is the recommended reference model.",
        "model_insight_ar": f"تحليل احتياطي — {best_model} هو النموذج المرجعي الموصى به.",
    }


# ── للتوافق مع الكود القديم ──────────────────────────
def get_maintenance_recommendation(engine_id: str, comparison_result: dict) -> dict:
    """Wrapper للتوافق مع الـ /recommend endpoint القديم."""
    consensus_rul = float(comparison_result.get("consensus_rul", 0))
    rec_info      = comparison_result.get("recommendation", {})
    best_model    = rec_info.get("recommended_model", "N/A")
    best_rul      = rec_info.get("rul", "N/A")

    if consensus_rul >= 100:
        risk, urgency = "LOW", "SCHEDULED"
    elif consensus_rul >= 50:
        risk, urgency = "MEDIUM", "WITHIN_WEEK"
    elif consensus_rul >= 20:
        risk, urgency = "HIGH", "WITHIN_24H"
    else:
        risk, urgency = "CRITICAL", "IMMEDIATE"

    lines = []
    for pred in comparison_result.get("predictions", []):
        name = pred.get("model_name", "?")
        if pred.get("status") == "success":
            lines.append(f"  • {name}: RUL = {pred['rul']} cycles")
        else:
            lines.append(f"  • {name}: FAILED")

    prompt = f"""You are a senior jet-engine predictive maintenance engineer.
Engine {engine_id} — Consensus RUL: {consensus_rul} cycles — Status: {risk}

MODEL PREDICTIONS:
{chr(10).join(lines)}

Return ONLY valid JSON (no markdown):
{{
  "risk_level": "{risk}",
  "urgency": "{urgency}",
  "recommendation": "2-3 sentences for the factory owner.",
  "actions": ["Action 1", "Action 2", "Action 3", "Action 4"],
  "model_insight": "One sentence about model reliability."
}}"""

    try:
        raw    = _call_gemini(prompt)
        parsed = _parse_json(raw)
        source = MODEL_NAME
    except Exception as exc:
        print(f"[Gemini] Failed: {exc}")
        parsed = {
            "risk_level":     risk,
            "urgency":        urgency,
            "recommendation": f"Engine {engine_id} requires {urgency.lower().replace('_',' ')} attention. Consensus RUL is {consensus_rul} cycles.",
            "actions":        ["Inspect critical components.", "Schedule maintenance.", "Document findings.", "Monitor sensors."],
            "model_insight":  f"{best_model} is the recommended reference model.",
        }
        source = "fallback-rule-engine"

    return {
        "engine_id"        : engine_id.upper(),
        "recommended_model": best_model,
        "consensus_rul"    : consensus_rul,
        "risk_level"       : parsed.get("risk_level", risk),
        "urgency"          : parsed.get("urgency", urgency),
        "recommendation"   : parsed.get("recommendation", ""),
        "actions"          : parsed.get("actions", []),
        "model_insight"    : parsed.get("model_insight", ""),
        "generated_by"     : source,
    }