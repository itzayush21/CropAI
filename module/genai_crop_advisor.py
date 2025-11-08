import os
import json
import google.generativeai as genai
import time

# Expect env var Google_Api_Key2
genai.configure(api_key=os.getenv("Google_Api_Key2"))
MODEL_NAME = "gemini-2.0-flash"

import json, re

def safe_json_loads(gemini_output: str):
    """
    Safely extracts and parses JSON from Gemini output.
    Handles trailing commas, smart quotes, and extra text.
    """
    try:
        # 1️⃣ Extract only the JSON part
        match = re.search(r'\{.*\}', gemini_output, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in text")
        json_text = match.group(0)

        # 2️⃣ Replace smart quotes & normalize
        json_text = json_text.replace('“', '"').replace('”', '"').replace('’', "'")

        # 3️⃣ Remove trailing commas before } or ]
        json_text = re.sub(r',(\s*[}\]])', r'\1', json_text)

        # 4️⃣ Parse JSON safely
        return json.loads(json_text)

    except json.JSONDecodeError as e:
        print("JSON decoding error:", e)
        print("Raw text:", gemini_output[:500])  # show snippet
        return None


SYSTEM_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "district": {"type": "string"},
        "state": {"type": "string"},
        "suitable_crops": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "crop": {"type": "string"},
                    "rationale": {"type": "string"},
                    "season": {"type": "string"},
                    "timeframe": {"type": "string"},
                    "steps": {"type": "string"},
                    "fertilizers": {"type": "string"},
                    "risks": {"type": "string"},
                    "market_tips": {"type": "string"},
                    "expected_yield": {"type": "string"},
                    "water_need": {"type": "string"}
                },
                "required": ["crop", "rationale"]
            }
        },
        "reasoning": {"type": "string"}
    },
    "required": ["suitable_crops", "reasoning"]
}

def _build_prompt(context_json: dict, user_query: str | None) -> str:
    # context_json may include: user, soil, weather, overrides
    return f"""
You are CropAI's expert agronomy assistant.
Given this CONTEXT (JSON), recommend the TOP-5 crops as structured JSON ONLY (no prose outside JSON).
Be practical for Indian conditions, and align with soil, pH, NDVI/NDWI hints, weather (temp/rain/humidity),
user preferences (budget, irrigation, risk, market), and land area.

CONTEXT:
{json.dumps(context_json, ensure_ascii=False, indent=2)}

REQUIREMENTS:
- Output STRICT JSON matching this schema:
{json.dumps(SYSTEM_JSON_SCHEMA, ensure_ascii=False, indent=2)}
- "suitable_crops": length exactly 5.
- "steps": concise process to grow (#1–#4).
- "fertilizers": precise N-P-K guidance or named fertilizers + stages.
- "timeframe": sowing-to-harvest duration or key stages with weeks/months.
- "market_tips": brief marketing/storage/sell-window advice.
- "expected_yield": typical yield range per acre (units).
- "water_need": qualitative (Low/Medium/High).
- "reasoning": a short conclusion that explains WHY these 5 are best.

{("USER_FOLLOWUP:\n" + user_query) if user_query else ""}
ONLY RETURN JSON. No additional commentary.
""".strip()


import time
import google.generativeai as genai

def generate_crop_suggestions(context_json: dict, user_query: str | None = None) -> dict:
    """
    Calls Gemini to get crop suggestions as structured JSON.
    Retries on invalid JSON, returns dict or raises Exception on failure.
    """
    model = genai.GenerativeModel(MODEL_NAME)
    max_retries = 3
    retry_delay = 2  # seconds
    prompt = _build_prompt(context_json, user_query)

    resp = None
    parsed = None

    for attempt in range(max_retries + 1):
        print(f"🧠 Attempt {attempt + 1}: Requesting Gemini response...")

        try:
            resp = model.generate_content(prompt)
        except Exception as e:
            print(f"⚠️ Gemini API error on attempt {attempt + 1}: {e}")
            if attempt < max_retries:
                time.sleep(retry_delay)
                continue
            raise RuntimeError(f"Gemini API call failed after {max_retries + 1} attempts: {e}")

        # Check if response text exists
        if not hasattr(resp, "text") or not resp.text.strip():
            print("⚠️ Empty response text from Gemini.")
            if attempt < max_retries:
                time.sleep(retry_delay)
                continue
            raise RuntimeError("Gemini returned an empty response.")

        parsed = safe_json_loads(resp.text)
        if parsed:
            print(f"✅ Successfully parsed JSON on attempt {attempt + 1}")
            return parsed

        print(f"⚠️ Invalid JSON on attempt {attempt + 1}. Retrying in {retry_delay} seconds...")
        time.sleep(retry_delay)

    # If all retries exhausted
    print("❌ All retries failed. Returning raw output for debugging.")
    return {
        "error": "Invalid JSON after retries",
        "raw_output": getattr(resp, "text", None)
    }
