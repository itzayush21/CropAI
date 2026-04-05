"""
gemini_advisors.py
------------------
Gemini 2.0-powered advisory modules for CropAI:
- Fertilizer planning
- Pest control
- Financial insights
- AI doubt resolver
"""

import os
import json
import google.generativeai as genai

# ✅ Gemini Configuration
genai.configure(api_key=os.getenv("Google_Api_Key2"))
MODEL_NAME = "gemini-2.5-flash"


# ----------------------------
# 🧩 Utility: Safe Gemini Call
# ----------------------------
def _call_gemini(prompt: str, json_output=False):
    """Safely call Gemini and handle malformed responses."""
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        text = response.text.strip()

        if not text:
            return [] if json_output else "⚠️ Empty AI response."

        if json_output:
            try:
                return json.loads(text)
            except Exception:
                # Clean malformed JSON (extra text before/after)
                start, end = text.find("{"), text.rfind("}") + 1
                if start != -1 and end != -1:
                    try:
                        cleaned = text[start:end]
                        return json.loads(cleaned)
                    except Exception as e:
                        print(f"[Gemini JSON Parse Error] {e}")
                # Try JSON array
                start_a, end_a = text.find("["), text.rfind("]") + 1
                if start_a != -1 and end_a != -1:
                    try:
                        cleaned_a = text[start_a:end_a]
                        return json.loads(cleaned_a)
                    except Exception as e:
                        print(f"[Gemini JSON Array Parse Error] {e}")
                return []
        return text

    except Exception as e:
        print(f"[Gemini Error] {e}")
        return [] if json_output else f"AI Error: {e}"


# ----------------------------
# 🧠 Context Helper
# ----------------------------
def _build_context(crop: str, context: dict):
    """Ensures context dict always has clean fields."""
    default = {
        "crop": crop,
        "region": "Unknown Region",
        "soil_type": "Unknown Soil",
        "ph_level": "N/A",
        "soil_summary": "No soil information available.",
        "weather_summary": "No weather data available.",
        "avg_temp": "N/A",
        "humidity": "N/A",
        "rainfall": "N/A",
        "ndvi": "N/A",
        "ndwi": "N/A",
        "land_summary": "No land data.",
        "land_area": "N/A",
        "expected_yield": "N/A",
        "estimated_budget": "N/A",
        "current_stage": "N/A"
    }
    default.update(context or {})
    return default


# ----------------------------
# 🧪 Fertilizer Advisor
# ----------------------------
def get_fertilizer_suggestions(crop: str, context: dict):
    ctx = _build_context(crop, context)
    prompt = f"""
You are an agricultural expert AI. Analyze the following context and recommend fertilizers.

Crop: {ctx['crop']}
Region: {ctx['region']}
Soil Type: {ctx['soil_type']}
pH Level: {ctx['ph_level']}
Soil Summary: {ctx['soil_summary']}
Weather Summary: {ctx['weather_summary']}
Avg Temperature: {ctx['avg_temp']}°C
Humidity: {ctx['humidity']}%
Rainfall: {ctx['rainfall']} mm
NDVI: {ctx['ndvi']}
Land Summary: {ctx['land_summary']}
Land Area: {ctx['land_area']} acres

Return 3–5 fertilizer recommendations as JSON:
[
  {{
    "fertilizer": "Name of fertilizer",
    "purpose": "Why it is needed",
    "dosage": "Amount (kg/acre or ml/litre)",
    "frequency": "Application interval",
    "stage": "Crop stage for use"
  }}
]
"""
    result = _call_gemini(prompt, json_output=True)
    print("[Gemini Fertilizer Suggestions]", result)
    return result or [{
        "fertilizer": "Urea",
        "purpose": "Provides nitrogen for plant growth.",
        "dosage": "50 kg/acre",
        "frequency": "Every 25–30 days",
        "stage": "Vegetative"
    }]
    
    
    


# ----------------------------
# 🐛 Pest Control Advisor
# ----------------------------
def get_pest_guidelines(crop: str, context: dict):
    ctx = _build_context(crop, context)
    prompt = f"""
You are a pest management AI for sustainable farming.
Given the following data, list current pest/disease risks and eco-friendly controls.

Crop: {ctx['crop']}
Region: {ctx['region']}
Soil: {ctx['soil_type']}
Weather: {ctx['weather_summary']}
Temperature: {ctx['avg_temp']}°C
Humidity: {ctx['humidity']}%
Rainfall: {ctx['rainfall']} mm
Stage: {ctx['current_stage']}
Land Summary: {ctx['land_summary']}

Return 2–4 objects in JSON:
[
  {{
    "pest": "Name",
    "symptoms": "Visible signs",
    "organic_control": "Eco-friendly measure",
    "chemical_control": "Optional chemical (only if necessary)"
  }}
]
"""
    result = _call_gemini(prompt, json_output=True)
    return result or [{
        "pest": "Aphids",
        "symptoms": "Sticky residue and curled leaves",
        "organic_control": "Neem oil spray every 10 days",
        "chemical_control": "Dimethoate 30EC @ 1 ml/litre if severe"
    }]


# ----------------------------
# 💰 Financial Advisor
# ----------------------------
def get_financial_advice(crop: str, context: dict):
    ctx = _build_context(crop, context)
    prompt = f"""
You are an agri-finance advisor AI.
Provide actionable strategies to improve ROI and reduce farming costs.

Crop: {ctx['crop']}
Region: {ctx['region']}
Soil: {ctx['soil_type']}
Weather: {ctx['weather_summary']}
Land Area: {ctx['land_area']} acres
Expected Yield: {ctx['expected_yield']}
Estimated Budget: ₹{ctx['estimated_budget']}
NDVI: {ctx['ndvi']}
NDWI: {ctx['ndwi']}

Return 3–5 actionable financial tips as JSON:
[
  {{
    "tip": "Title of advice",
    "description": "2–3 line explanation",
    "potential_saving": "₹ or % (if applicable)"
  }}
]
"""
    result = _call_gemini(prompt, json_output=True)
    return result or [{
        "tip": "Use Government Subsidy for Drip Irrigation",
        "description": "Apply for PMKSY scheme to reduce irrigation cost by 30–40%.",
        "potential_saving": "₹2000–3000/acre"
    }]


# ----------------------------
# 🤖 AI Doubt Solver
# ----------------------------
def get_ai_doubt_response(query: str, context: dict):
    ctx = _build_context(context.get("crop", "Unknown"), context)
    prompt = f"""
You are CropAI, a concise and reliable agricultural advisor.
Answer the farmer’s question based on this context:

Crop: {ctx['crop']}
Region: {ctx['region']}
Soil Type: {ctx['soil_type']}
Weather: {ctx['weather_summary']}
Stage: {ctx['current_stage']}
Land Summary: {ctx['land_summary']}
Soil Summary: {ctx['soil_summary']}

Farmer's Question: "{query}"

Answer clearly in 2–4 lines. Be factual, local-context aware, and avoid fluff.
"""
    response = _call_gemini(prompt, json_output=False)
    return response or "I'm sorry, I couldn't retrieve information for that right now. Try again soon."
