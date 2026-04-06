# ================================================================
# 🚜 CropAI — Equipment Recommendation System (Gemini Only)
# ================================================================

import os
import json
import time
import re
import google.generativeai as genai
import dotenv


# ================================================================
# 📘 JSON Schema — Equipment Recommendation
# ================================================================
SYSTEM_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "district": {"type": "string"},
        "state": {"type": "string"},
        "recommended_equipment": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "equipment": {"type": "string"},
                    "purpose": {"type": "string"},
                    "suitable_land_size": {"type": "string"},
                    "power_requirement": {"type": "string"},
                    "fuel_or_energy": {"type": "string"},
                    "estimated_cost": {"type": "string"},
                    "operation_steps": {"type": "string"},
                    "maintenance_tips": {"type": "string"},
                    "recommended_stage": {"type": "string"},
                    "replacement_or_upgrade_cycle": {"type": "string"}
                },
                "required": ["equipment", "purpose"]
            }
        },
        "reasoning": {"type": "string"}
    },
    "required": ["recommended_equipment", "reasoning"]
}

# ================================================================
# 🧩 Safe JSON Parser
# ================================================================
def safe_json_loads(text: str):
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        cleaned = match.group(0)
        cleaned = cleaned.replace("“", "\"").replace("”", "\"")
        cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)
        return json.loads(cleaned)
    except Exception:
        return None

# ================================================================
# 🧠 Prompt Builder (NO RAG)
# ================================================================
def build_prompt(context: dict, user_query: str | None = None) -> str:
    return f"""
You are CropAI's expert farm mechanization advisor.

Your goal is to recommend the TOP 5 farm equipment choices
best suited for the farmer's conditions.

CONTEXT:
{json.dumps(context, indent=2, ensure_ascii=False)}

REQUIREMENTS:
- Output STRICT JSON ONLY (no text outside JSON).
- Follow this JSON schema exactly:
{json.dumps(SYSTEM_JSON_SCHEMA, indent=2, ensure_ascii=False)}
- "recommended_equipment" must contain EXACTLY 5 items.
- Be realistic for Indian conditions.
- Provide approximate cost in INR (₹).
- Prefer durability, fuel efficiency, and maintenance ease.
- Assume small–medium farmers unless specified otherwise.

{("USER_QUERY:\n" + user_query) if user_query else ""}

ONLY RETURN JSON. Do NOT include explanations outside JSON.
""".strip()

# ================================================================
# 🚜 MAIN GENERATION FUNCTION
# ================================================================
def generate_equipment_recommendations(
    context: dict,
    user_query: str | None = None,
    max_retries: int = 3
) -> dict:
    """
    Gemini-only equipment recommendation pipeline.
    Safe retries + JSON validation.
    """

    model = genai.GenerativeModel(
        "gemini-2.5-flash",
        generation_config={
            "response_mime_type": "application/json"
        }
    )
    prompt = build_prompt(context, user_query)

    last_raw_output = None

    for attempt in range(1, max_retries + 1):
        print(f"🧠 Gemini attempt {attempt}...")
        try:
            resp = model.generate_content(prompt)
            last_raw_output = getattr(resp, "text", "")
            parsed = safe_json_loads(last_raw_output)

            if parsed:
                print("✅ Valid JSON received.")
                return parsed

            print("⚠️ Invalid JSON, retrying...")
            time.sleep(1)

        except Exception as e:
            print(f"⚠️ Gemini error: {e}")
            time.sleep(1)

    return {
        "error": "Failed to generate valid equipment recommendation JSON",
        "raw_output": last_raw_output
    }