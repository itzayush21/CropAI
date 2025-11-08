import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("Google_Api_Key2"))

# Gemini conversational model
model = genai.GenerativeModel("gemini-2.0-flash")

# Session-based memory
CONVERSATION_MEMORY = {}

def generate_fertilizer_recommendations(context, session_id):
    """
    Conversational fertilizer advisor with memory.
    Returns structured JSON (reply, recommendations, schedule, insights).
    """

    if session_id not in CONVERSATION_MEMORY:
        CONVERSATION_MEMORY[session_id] = []

    history = CONVERSATION_MEMORY[session_id]

    prompt = f"""
    You are CropAI’s **Fertilizer Advisor**, an interactive agricultural assistant.
    Analyze the provided context carefully.

    Context:
    {json.dumps(context, indent=2)}

    Respond ONLY in valid JSON:
    {{
        "reply": "Conversational reply explaining your suggestion or clarification.",
        "recommendations": [
            {{
                "name": "Fertilizer name",
                "type": "Organic/Chemical/Mixed",
                "dosage": "Recommended amount per acre",
                "alternative": "Possible alternative (if any)",
                "reason": "Why suitable for this crop and soil"
            }}
        ],
        "schedule": "Phase-wise or week-wise fertilizer application schedule",
        "insights": "Additional context or care tips"
    }}

    Notes:
    - Maintain natural conversation flow.
    - If user asks follow-up (e.g., 'explain urea'), reuse past data.
    - Always return JSON.
    """

    # Build conversation history
    messages = [prompt]
    for turn in history:
        messages.append(turn["content"])
    messages.append(json.dumps(context))

    try:
        response = model.generate_content(messages)
        text = response.text.strip()

        # Safe JSON extraction
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        parsed = json.loads(text[json_start:json_end])

        # Update memory
        history.append({"role": "user", "content": json.dumps(context)})
        history.append({"role": "assistant", "content": parsed})
        CONVERSATION_MEMORY[session_id] = history

        return parsed

    except Exception as e:
        # Fallback static reply
        return {
            "reply": f"Encountered an issue, showing fallback: {str(e)}",
            "recommendations": [
                {
                    "name": "Urea (46% N)",
                    "type": "Chemical",
                    "dosage": "50kg/acre split twice",
                    "alternative": "Compost or DAP",
                    "reason": "Provides quick nitrogen support during growth."
                }
            ],
            "schedule": "Apply half before sowing and half after 30 days.",
            "insights": "Ensure soil moisture before applying urea."
        }
