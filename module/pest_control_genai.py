import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("Google_Api_Key2"))

model = genai.GenerativeModel("gemini-2.0-flash")

# Memory for conversational continuity (can later move to Redis or DB)
PEST_CONVERSATION_MEMORY = {}


def generate_pest_control_advice(context, session_id):
    """
    Conversational pest control advisor that uses Gemini 2.0 Flash.
    Returns structured JSON for chat, including reply, pest list, and recommendations.
    """

    if session_id not in PEST_CONVERSATION_MEMORY:
        PEST_CONVERSATION_MEMORY[session_id] = []

    history = PEST_CONVERSATION_MEMORY[session_id]

    prompt = f"""
    You are CropAI’s **Pest Control Advisor**, an agricultural assistant specializing in pest diagnosis and management.
    
    Context:
    {json.dumps(context, indent=2)}

    Respond **only in valid JSON** like this:
    {{
      "reply": "Conversational reply to the user.",
      "pests": [
        {{
          "name": "Pest name",
          "symptoms": "Common visible symptoms",
          "organic_control": "Eco-friendly control measures",
          "chemical_control": "Approved chemical treatments (with dosage)",
          "prevention_tips": "How to prevent recurrence"
        }}
      ],
      "recommendations": "Summary and final recommendations."
    }}

    Rules:
    - If user asks a follow-up (like “how to identify aphids?”), continue the conversation logically.
    - Base suggestions on crop type, soil, and climate context.
    - Always return a valid JSON object.
    """

    try:
        messages = [prompt] + [turn["content"] for turn in history] + [json.dumps(context)]
        response = model.generate_content(messages)
        text = response.text.strip()

        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        parsed = json.loads(text[json_start:json_end])

        # Update in-memory conversation
        history.append({"role": "user", "content": json.dumps(context)})
        history.append({"role": "assistant", "content": parsed})
        PEST_CONVERSATION_MEMORY[session_id] = history

        return parsed

    except Exception as e:
        return {
            "reply": f"Encountered a minor issue, using fallback. ({str(e)})",
            "pests": [
                {
                    "name": "Aphids",
                    "symptoms": "Curled leaves, sticky residue, stunted growth.",
                    "organic_control": "Neem oil spray (5ml/litre) every 5 days.",
                    "chemical_control": "Imidacloprid 17.8% SL, 0.3ml/litre.",
                    "prevention_tips": "Encourage ladybugs; avoid over-fertilizing."
                }
            ],
            "recommendations": "Inspect leaves weekly and maintain clean field edges."
        }
