# module/decision_engine.py

import os
import json
import random
import google.generativeai as genai


# ==========================================
# 🔑 CONFIG
# ==========================================
GEMINI_API_KEY = os.getenv("Google_Api_Key2")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not set")

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel(
    "gemini-2.5-flash",
    generation_config={"response_mime_type": "application/json"}
)


# ==========================================
# 🧠 ENGINE CLASS
# ==========================================
class DecisionEngine:

    # -------------------------------
    # CONTEXT BUILDER
    # -------------------------------
    def build_context(self, user_data):
        return {
            "land_area_acres": float(user_data.get("land_area", "1").split()[0]),
            "soil_type": str(user_data.get("soil_type", "loam")).lower(),
            "soil_ph": float(user_data.get("soil_ph", 6.5)),
            "rainfall_level": str(user_data.get("rainfall", "medium")).lower(),
            "temperature_level": str(user_data.get("temperature", "moderate")).lower(),
            "location": str(user_data.get("location", "")).lower(),
            "language": str(user_data.get("language", "english")).lower()
        }

    # -------------------------------
    # ENVIRONMENT MAPPING
    # -------------------------------
    def map_environment(self, context):
        rainfall_map = {"low": 400, "medium": 800, "high": 1200}
        temp_map = {"cool": 20, "moderate": 25, "hot": 32}

        context["rainfall_value"] = rainfall_map.get(context["rainfall_level"], 800)
        context["temperature_value"] = temp_map.get(context["temperature_level"], 25)

        return context

    # -------------------------------
    # ML PLACEHOLDERS
    # -------------------------------
    def profit_model(self, crop, context):
        base = {
            "rice": 50000,
            "wheat": 40000,
            "maize": 45000,
            "cotton": 60000
        }

        profit = base.get(crop.lower(), 30000)

        if context["rainfall_value"] > 1000:
            profit *= 1.1
        elif context["rainfall_value"] < 500:
            profit *= 0.85

        return round(profit * random.uniform(0.9, 1.1), 2)

    def eco_model(self, crop, context):
        water_factor = {
            "rice": 0.9,
            "wheat": 0.6,
            "maize": 0.4,
            "cotton": 0.8
        }

        water = water_factor.get(crop.lower(), 0.7)
        climate_fit = 1 if 20 <= context["temperature_value"] <= 30 else 0.6

        eco = (0.3 * (1 - water)) + (0.3 * (1 - water)) + (0.4 * climate_fit)
        eco += random.uniform(-0.05, 0.05)

        return round(max(0, min(eco, 1)), 3)

    def suitability_model(self, crop, context):
        score = 0.5

        if crop.lower() == "rice" and context["soil_type"] == "clay":
            score += 0.3
        if crop.lower() == "wheat" and context["soil_type"] == "loam":
            score += 0.3

        if crop.lower() == "rice" and context["rainfall_value"] > 1000:
            score += 0.2
        if crop.lower() == "wheat" and context["rainfall_value"] < 800:
            score += 0.2

        score += random.uniform(-0.05, 0.05)
        return round(max(0, min(score, 1)), 3)

    # -------------------------------
    # SCORING
    # -------------------------------
    def compute_score(self, profit, eco, suitability):
        profit_norm = profit / 100000
        return round((0.4 * profit_norm) + (0.3 * eco) + (0.3 * suitability), 3)

    # -------------------------------
    # COMPARISON
    # -------------------------------
    def compare(self, crops, context):
        context = self.map_environment(context)

        results = []

        for crop in crops:
            profit = self.profit_model(crop, context)
            eco = self.eco_model(crop, context)
            suitability = self.suitability_model(crop, context)

            results.append({
                "crop": crop,
                "profit_estimate": profit,
                "eco_score": eco,
                "suitability_score": suitability,
                "heuristic_score": self.compute_score(profit, eco, suitability)
            })

        return sorted(results, key=lambda x: x["heuristic_score"], reverse=True)

    # -------------------------------
    # AI REASONING
    # -------------------------------
    def generate_explanation(self, context, results):

        prompt = f"""
You are CropAI, an expert agricultural decision intelligence system.

Your task is NOT to repeat model scores.
Your task IS to re-evaluate, reason, and justify.

----------------------------------
USER CONTEXT (REAL WORLD)
----------------------------------
{json.dumps(context, indent=2)}

----------------------------------
ML MODEL SIGNALS (ADVISORY ONLY)
----------------------------------
{json.dumps(results, indent=2)}

----------------------------------
INSTRUCTIONS
----------------------------------
• Scores are noisy indicators, not ground truth
• Penalize fragile crops under low rainfall & heat
• Reward resilience and farmer stability
• Ignore tiny score gaps unless agronomically meaningful
• You may overrule the ranking if justified

----------------------------------
OUTPUT FORMAT (STRICT JSON)
----------------------------------
{{
  "final_recommendation": {{
    "best_crop": "",
    "confidence_level": "High | Medium | Low",
    "reasoning": [
      "Concrete agronomic reasoning",
      "Explicit trade-offs"
    ]
  }},
  "crop_comparison": [
    {{
      "crop": "",
      "verdict": "Recommended | Acceptable | Risky | Not Suitable",
      "why": "",
      "main_risks": [],
      "when_it_makes_sense": ""
    }}
  ],
  "decision_summary": {{
    "key_deciding_factors": [],
    "score_override_explanation": ""
  }}
}}
"""

        try:
            response = model.generate_content(prompt)
            return json.loads(response.text)
        except:
            return {"error": "AI reasoning failed"}

    # -------------------------------
    # TRANSLATION
    # -------------------------------
    def translate(self, data, language):
        if language == "english":
            return data

        prompt = f"""
Translate VALUES to {language}. Keep keys same.

{json.dumps(data, indent=2, ensure_ascii=False)}
"""

        try:
            response = model.generate_content(prompt)
            return json.loads(response.text)
        except:
            return data

    # -------------------------------
    # MAIN PIPELINE
    # -------------------------------
    def run(self, crops, user_data, translate=True):

        context = self.build_context(user_data)
        results = self.compare(crops, context)

        explanation = self.generate_explanation(context, results)

        output = {
            "ranked_signals": results,
            "ai_decision": explanation
        }

        if translate:
            output = self.translate(output, context["language"])

        return output


# ==========================================
# 🔥 ENTRY FUNCTION
# ==========================================
engine = DecisionEngine()

def run_decision_engine(crops, user_data):
    return engine.run(crops, user_data)