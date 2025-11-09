"""
gemini_crop_pipeline.py
------------------------
Gemini 2.0-based step generator for CropAI.
Handles generation of initial and next crop steps dynamically with soil, land, and weather context.
"""

import os
import json
from datetime import datetime, timezone
import google.generativeai as genai  # pip install google-generativeai

# ✅ Configure Gemini client
API_KEY =os.getenv("Google_Api_Key2")
if not API_KEY:
    raise EnvironmentError("❌ Missing Gemini API Key: Set GEMINI_API_KEY or Google_Api_Key2")
genai.configure(api_key=API_KEY)

MODEL_NAME = "gemini-2.0-flash"  # can also use "gemini-2.0-flash" for faster responses


# 🧩 --- Utility: Safe Gemini JSON Call ---
def _call_gemini(prompt: str):
    """
    Calls Gemini safely and extracts JSON output even if wrapped in markdown or text.
    """
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)

        # ✅ Extract text safely (Gemini sometimes returns candidates instead of text)
        if hasattr(response, "text"):
            text = response.text.strip()
        elif hasattr(response, "candidates") and response.candidates:
            parts = response.candidates[0].content.parts
            text = " ".join([p.text for p in parts if hasattr(p, "text")]).strip()
        else:
            raise ValueError("Empty response from Gemini")

        # 🧹 Clean markdown wrappers
        text = text.replace("```json", "").replace("```", "").strip()

        # 🧠 Try to parse JSON directly
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Extract inner JSON (handles Gemini’s extra words)
            start, end = text.find("{"), text.rfind("}") + 1
            if start != -1 and end != -1:
                return json.loads(text[start:end])
            else:
                raise ValueError("Malformed JSON response")

    except Exception as e:
        print(f"[Gemini Error] Failed to parse: {e}")
        # fallback default
        return {
            "title": "General Crop Care",
            "description": "Continue standard crop monitoring, irrigation, and nutrient management."
        }


# 🧠 --- Prompt Builder ---
def build_prompt(chosen_crop, soil_type, region, weather_summary,
                 soil_summary=None, land_summary=None, current_stage=None,
                 ndvi=None, ndwi=None, land_area=None):
    """
    Builds a context-rich prompt for Gemini based on crop & environment.
    """
    context = f"""
Crop: {chosen_crop}
Region: {region}
Soil Type: {soil_type}
Soil Summary: {soil_summary or "Not available"}
Land Summary: {land_summary or "Not available"}
Land Area: {land_area or "Unknown"} acres
NDVI: {ndvi or "N/A"}
NDWI: {ndwi or "N/A"}
Weather Summary: {weather_summary or "N/A"}
"""

    if current_stage:
        # 🌿 Generate next step prompt
        return f"""
You are CropAI, an agricultural expert helping farmers progress their crop cultivation intelligently.

{context}

Current Stage: {current_stage}

Predict the **next best step** (2–3 lines) to perform after this stage, 
based on practical farm experience, soil, and weather conditions.

Return JSON ONLY in this format:
{{
  "title": "Step Title (e.g., Pest Monitoring & Early Irrigation)",
  "description": "2–3 line explanation of what to do next and why it's important."
}}
"""
    else:
        # 🌱 Generate initial step prompt
        return f"""
You are CropAI, an intelligent assistant helping farmers start their crop cycle effectively.

{context}

Predict the **first cultivation step** to begin with (before sowing).
Keep it practical and concise (2–3 lines).

Return JSON ONLY in this format:
{{
  "title": "Step Title (e.g., Land Preparation & Soil Conditioning)",
  "description": "2–3 line actionable guidance to start the farming process."
}}
"""


# 🌾 --- Generate Initial Step ---
def generate_initial_step(chosen_crop, soil_type, region, weather_summary,
                          soil_summary=None, land_summary=None,
                          ndvi=None, ndwi=None, land_area=None):
    """
    Generates the initial AI-driven step when a crop-room is created.
    """
    prompt = build_prompt(chosen_crop, soil_type, region, weather_summary,
                          soil_summary, land_summary, None, ndvi, ndwi, land_area)
    print("[Gemini] Generating Initial Step...")
    result = _call_gemini(prompt)
    return {
        "title": result.get("title", "Land Preparation & Soil Conditioning"),
        "description": result.get("description", "Start by tilling, leveling, and enriching soil before sowing.")
    }


# 🌿 --- Generate Next Step ---
def generate_next_step(chosen_crop, soil_type, region, weather_summary, current_stage,
                       soil_summary=None, land_summary=None,
                       ndvi=None, ndwi=None, land_area=None):
    """
    Generates the next stage step dynamically using Gemini AI.
    """
    prompt = build_prompt(chosen_crop, soil_type, region, weather_summary,
                          soil_summary, land_summary, current_stage, ndvi, ndwi, land_area)
    print(f"[Gemini] Generating Next Step for stage: {current_stage}")
    result = _call_gemini(prompt)
    return {
        "title": result.get("title", "Next Step"),
        "description": result.get("description", "Continue with standard crop care and monitor progress.")
    }


# 🗃️ --- Formatter for Database Storage ---
def format_step_for_db(step_data):
    """
    Adds timestamp and structures Gemini output for DB storage.
    """
    return {
        "title": step_data["title"],
        "description": step_data["description"],
        "generated_at": datetime.now(timezone.utc).isoformat()
    }
