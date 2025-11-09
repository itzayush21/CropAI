# modules/crop_suggestion_ai.py
import os
import json
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("Google_Api_Key2"))

# Gemini model (fast + accurate)
model = genai.GenerativeModel("gemini-2.0-flash")

# --------------------------
# DYNAMIC CROP SUGGESTION (AI-Driven)
# --------------------------
def generate_crop_suggestion(data, soil_type, region, weather_summary):
    """
    Generate AI-based crop suggestion based on soil, region, and weather context.
    Falls back to static suggestion if LLM fails.
    """
    crop = data.get("chosen_crop", "Crop")
    soil_type = soil_type or "general soil"
    region = region or "your region"
    weather_summary = weather_summary or "current climate data"

    # Create a structured prompt for Gemini
    prompt = f"""
    You are CropAI’s Agricultural Advisor.
    Given the following conditions:
    - Crop: {crop}
    - Region: {region}
    - Soil Type: {soil_type}
    - Weather Summary: {weather_summary}

    Analyze whether this crop is suitable for the given conditions.
    Then, generate a JSON object with these keys:
    {{
      "summary": "1–2 line overview of suitability and reasoning",
      "key_recommendations": ["Tip 1", "Tip 2", "Tip 3"],
      "insight": "Brief weather or soil insight",
      "confidence_score": float between 0.75 and 1.0
    }}

    Keep language simple, conversational, and factual.
    Ensure it is valid JSON with no additional text.
    """

    try:
        response = model.generate_content(prompt)
        text = response.text.strip()

        # Extract JSON safely
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        ai_output = json.loads(text[json_start:json_end])

        # Ensure all keys exist
        if "summary" not in ai_output or "key_recommendations" not in ai_output:
            raise ValueError("Incomplete AI response")

        # ✅ Add timestamp + metadata
        ai_output["generated_at"] = datetime.utcnow().isoformat()
        ai_output["source"] = "Gemini-2.0"

        return ai_output

    except Exception as e:
        print("⚠️ Gemini fallback triggered:", e)

        # 🔁 Fallback static logic
        weather_hint = (
            "The region shows moderate rainfall and suitable humidity for cereal crops."
            if "humid" in weather_summary.lower()
            else "The area experiences dry spells; ensure irrigation management."
        )

        suggestion = {
            "summary": f"{crop} is suitable for {region} based on soil type ({soil_type}) and {weather_summary.lower()}.",
            "key_recommendations": [
                f"Ensure proper irrigation during the {crop} vegetative stage.",
                "Add organic compost or FYM before sowing to enrich soil nutrients.",
                "Monitor pest activity every 10–12 days.",
                "Maintain soil moisture, especially during critical growth periods.",
                "Use weather forecasts to plan fertilizer application and irrigation."
            ],
            "insight": weather_hint,
            "confidence_score": 0.92,
            "generated_at": datetime.utcnow().isoformat(),
            "source": "Fallback Logic"
        }
        return suggestion
