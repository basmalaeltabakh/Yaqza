# backend/gemini_advisor.py
"""
Gemini Advisor — Pure HTTP, zero Google SDK.
Avoids all protobuf conflicts with TensorFlow.
Includes retry logic + caching + graceful fallback.
"""
import os
import json
import time
import hashlib
import httpx
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME     = "gemini-3-flash-preview"
BASE_URL       = "https://generativelanguage.googleapis.com/v1beta/models"

# ── Retry config ────────────────────────────────────────────────────────────
MAX_RETRIES = 3
BASE_DELAY  = 2.0   # seconds


def _call_gemini(prompt: str) -> str:
    """Direct HTTP call to Gemini — no SDK, no protobuf.
    Retry with exponential backoff on 429 / 503.
    """
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set in .env")

    url = f"{BASE_URL}/{MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"

    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ],
        "generationConfig": {
            "temperature":     0.2,
            "maxOutputTokens": 1024,
        }
    }

    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            with httpx.Client(timeout=30.0) as client:
                r = client.post(url, json=payload)
                r.raise_for_status()
                return r.json()["candidates"][0]["content"]["parts"][0]["text"]
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            last_exc = exc
            if status in (429, 503):
                wait = BASE_DELAY * (2 ** attempt)
                print(f"[Gemini] {status} on attempt {attempt + 1}/{MAX_RETRIES}, waiting {wait:.1f}s...")
                time.sleep(wait)
                continue
            raise  # other 4xx/5xx — don't retry
        except Exception:
            raise  # network errors — don't retry

    raise last_exc  # all retries exhausted


def _parse_json_response(raw: str) -> dict:
    """بتشيل الـ markdown وبتعمل JSON parse."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text  = "\n".join(lines[1:-1]).strip()
    return json.loads(text)


def _build_fallback_recommendation(
    engine_id: str,
    consensus_rul: float,
    best_model: str,
    best_rul,
) -> dict:
    """توصية fallback منطقية لو Gemini فشل خالص."""
    if consensus_rul >= 100:
        risk = "LOW"
        urgency = "SCHEDULED"
        rec = f"Engine {engine_id} is in good health with {consensus_rul} cycles remaining. Continue routine monitoring and scheduled maintenance."
        actions = [
            "Perform routine visual inspection during next scheduled check.",
            "Continue standard sensor monitoring protocols.",
            "Schedule next maintenance per standard interval.",
            "Log current readings for trend analysis."
        ]
    elif consensus_rul >= 50:
        risk = "MEDIUM"
        urgency = "WITHIN_WEEK"
        rec = f"Engine {engine_id} shows moderate degradation. RUL of {consensus_rul} cycles requires attention within the week to prevent escalation."
        actions = [
            "Inspect high-wear components (turbine blades, bearings).",
            "Verify sensor calibration for sensors 7, 12, and 15.",
            "Schedule detailed inspection within 5-7 days.",
            "Document any abnormal vibration or temperature readings."
        ]
    elif consensus_rul >= 20:
        risk = "HIGH"
        urgency = "WITHIN_24H"
        rec = f"Engine {engine_id} is approaching critical wear. With only {consensus_rul} cycles remaining, immediate action is required within 24 hours."
        actions = [
            "Ground engine and perform full diagnostic scan.",
            "Replace or inspect all critical rotating components.",
            "Escalate to senior maintenance engineer immediately.",
            "Prepare replacement parts and maintenance crew for urgent repair."
        ]
    else:
        risk = "CRITICAL"
        urgency = "IMMEDIATE"
        rec = f"Engine {engine_id} is in CRITICAL condition. RUL of {consensus_rul} cycles demands immediate grounding and emergency maintenance."
        actions = [
            "GROUND ENGINE IMMEDIATELY — do not operate.",
            "Initiate emergency maintenance protocol.",
            "Notify operations manager and safety officer.",
            "Prepare full engine teardown and component replacement plan."
        ]

    return {
        "risk_level": risk,
        "urgency": urgency,
        "recommendation": rec,
        "actions": actions,
        "model_insight": f"Models indicate {risk} risk based on consensus RUL of {consensus_rul} cycles. {best_model} provided the most reliable estimate ({best_rul} cycles).",
    }


def _hash_comparison(comparison_result: dict) -> str:
    """Hash للـ caching — بنحسب hash من الـ predictions فقط."""
    # ناخد predictions + consensus_rul فقط (اللي بيأثّر على الـ prompt)
    key_data = {
        "predictions": comparison_result.get("predictions", []),
        "consensus_rul": comparison_result.get("consensus_rul", 0),
    }
    return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()


@lru_cache(maxsize=128)
def _cached_call_gemini(prompt_hash: str, prompt: str) -> str:
    """Cached wrapper — نفس الـ prompt = نفس الـ response."""
    return _call_gemini(prompt)


def get_maintenance_recommendation(
    engine_id: str,
    comparison_result: dict,
) -> dict:
    """
    بتاخد نتيجة الـ model comparison وبترجع توصية maintenance من Gemini.
    لو Gemini فشل (429/503) → fallback recommendation منطقية.
    """
    # جهّزي ملخص التوقعات
    lines = []
    for pred in comparison_result.get("predictions", []):
        name = pred.get("model_name", "?")
        if pred.get("status") == "success":
            line = f"  • {name}: RUL = {pred['rul']} cycles"
            unc  = pred.get("uncertainty")
            if unc:
                line += f" | std={unc.get('std','?')}"
                ci   = unc.get("ci_90")
                if ci:
                    line += f" | 90% CI [{ci[0]} – {ci[1]}]"
            lines.append(line)
        else:
            lines.append(f"  • {name}: FAILED — {pred.get('error','unknown')}")

    predictions_block = "\n".join(lines)
    consensus_rul     = comparison_result.get("consensus_rul", 0)
    status            = comparison_result.get("overall_status", "UNKNOWN")
    rec               = comparison_result.get("recommendation", {})
    best_model        = rec.get("recommended_model", "N/A")
    best_rul          = rec.get("rul", "N/A")

    prompt = f"""You are a senior jet-engine predictive maintenance engineer.
Five ML models have predicted the Remaining Useful Life (RUL) of engine {engine_id}
using the NASA CMAPSS turbofan degradation dataset.

════════════════════════════════════
ENGINE      : {engine_id}
STATUS      : {status}
CONSENSUS   : {consensus_rul} cycles  (average of all successful models)
BEST MODEL  : {best_model}  →  {best_rul} cycles
════════════════════════════════════

MODEL BREAKDOWN
{predictions_block}

CONTEXT
- RUL = cycles until the engine is expected to fail
- NGBoost provides a confidence interval (uncertainty); CNN-LSTM learns from raw sensor sequences
- If models disagree, trust NGBoost (probabilistic) and CNN-LSTM (raw sequences) over linear Ridge
- Maintenance crews need clear, actionable guidance

════════════════════════════════════
TASK
Return ONLY a JSON object — no markdown fences, no prose outside the JSON:

{{
  "risk_level"    : "LOW | MEDIUM | HIGH | CRITICAL",
  "urgency"       : "SCHEDULED | WITHIN_WEEK | WITHIN_24H | IMMEDIATE",
  "recommendation": "Two or three sentences: current engine health, what will happen if ignored, recommended action.",
  "actions"       : [
    "Action 1 — specific inspection or part check",
    "Action 2 — monitoring or sensor calibration step",
    "Action 3 — maintenance scheduling decision",
    "Action 4 — documentation or escalation step"
  ],
  "model_insight" : "One sentence: do the models agree? Which is most reliable here and why?"
}}

RISK RULES (apply strictly based on consensus RUL):
  >= 100 cycles → LOW   + SCHEDULED
  50–99  cycles → MEDIUM + WITHIN_WEEK
  20–49  cycles → HIGH  + WITHIN_24H
  < 20   cycles → CRITICAL + IMMEDIATE"""

    # جرّبي Gemini — مع retry + cache
    try:
        prompt_hash = _hash_comparison(comparison_result)
        raw    = _cached_call_gemini(prompt_hash, prompt)
        parsed = _parse_json_response(raw)
        source = MODEL_NAME
    except Exception as exc:
        print(f"[Gemini] All retries failed: {exc}")
        print("[Gemini] Using fallback recommendation.")
        parsed = _build_fallback_recommendation(
            engine_id, consensus_rul, best_model, best_rul
        )
        source = "fallback-rule-engine"

    return {
        "engine_id"        : engine_id.upper(),
        "recommended_model": best_model,
        "consensus_rul"    : float(consensus_rul),
        "risk_level"       : parsed.get("risk_level",     "UNKNOWN"),
        "urgency"          : parsed.get("urgency",        "UNKNOWN"),
        "recommendation"   : parsed.get("recommendation", ""),
        "actions"          : parsed.get("actions",        []),
        "model_insight"    : parsed.get("model_insight",  ""),
        "generated_by"     : source,
    }