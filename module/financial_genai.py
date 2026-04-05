# financial_genai.py
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("Google_Api_Key2"))

# Use Gemini conversational model
model = genai.GenerativeModel("gemini-2.5-flash")

# In-memory conversation memory per session (optional: persist to DB/Redis later)
FIN_CONVERSATION_MEMORY = {}

def _safe_extract_json(text):
    """
    Try to locate a JSON object inside text and parse it safely.
    Returns dict on success else raises ValueError.
    """
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model response.")
    json_text = text[start:end+1]
    return json.loads(json_text)

def generate_financial_advice(context: dict, session_id: str):
    """
    Conversational Financial Advisor.
    - context: dict (user profile + form inputs + user_query if any)
    - session_id: unique id for conversation memory
    Returns structured Python dict:
    {
      "reply": "...",
      "cost_breakdown": {...},
      "expected_profit": 0.0,
      "roi": "xx%",
      "tips": [...],
      "conclusion": "..."
    }
    """
    if session_id not in FIN_CONVERSATION_MEMORY:
        FIN_CONVERSATION_MEMORY[session_id] = []

    history = FIN_CONVERSATION_MEMORY[session_id]

    prompt = f"""
You are CropAI's Financial Advisor for farmers. You're an expert in crop economics, input costing,
yield-to-revenue estimates, ROI, risk notes, and practical investment tips. The user will provide
context (crop, land area, budget preference, region/district/state, soil hints, and optional prior info).
Respond in **valid JSON only** using the schema exactly as shown:

{{
  "reply": "Short conversational reply acknowledging user's request.",
  "cost_breakdown": {{
    "seeds": 0.0,
    "fertilizers": 0.0,
    "pesticides": 0.0,
    "labor": 0.0,
    "irrigation": 0.0,
    "other": 0.0,
    "total_cost": 0.0
  }},
  "expected_profit": 0.0,
  "roi": "xx.x%", 
  "tips": ["string", "string"],
  "conclusion": "Short summary recommendation"
}}

Rules:
- Use region/district, soil summary, NDVI/NDWI if provided in context to adjust numbers.
- If user asks follow-up (e.g., 'explain labour costs'), continue the conversation, referencing previous context.
- Always return JSON only (no extra commentary outside the JSON).
"""

    # Build messages to pass to model: include prompt, previous turns, and the current context as the latest user content
    messages = [prompt] + [turn["content"] for turn in history] + [json.dumps(context)]

    try:
        response = model.generate_content(messages)
        text = response.text.strip()

        parsed = _safe_extract_json(text)

        # store both the raw user context and the assistant response in memory
        history.append({"role": "user", "content": json.dumps(context)})
        history.append({"role": "assistant", "content": parsed})
        FIN_CONVERSATION_MEMORY[session_id] = history

        return parsed

    except Exception as e:
        # fallback structured response to keep UI working
        fallback = {
            "reply": f"Sorry, I couldn't compute accurate figures right now (error: {str(e)}). Here's a conservative fallback estimate.",
            "cost_breakdown": {
                "seeds": 1000.0,
                "fertilizers": 1500.0,
                "pesticides": 500.0,
                "labor": 2000.0,
                "irrigation": 400.0,
                "other": 200.0,
                "total_cost": 5600.0
            },
            "expected_profit": 8400.0,
            "roi": "50.0%",
            "tips": [
                "Use drip irrigation to reduce irrigation cost by ~20-30%.",
                "Buy inputs in bulk during off-season to save costs."
            ],
            "conclusion": "Fallback: conservative plan until model recovers."
        }
        # keep fallback in memory as assistant reply
        history.append({"role": "user", "content": json.dumps(context)})
        history.append({"role": "assistant", "content": fallback})
        FIN_CONVERSATION_MEMORY[session_id] = history
        return fallback
