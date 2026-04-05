# module/ai_explainer_engine.py

import os
import json
import re
import uuid
import google.generativeai as genai
from gtts import gTTS


# ==========================================
# 🔑 CONFIG
# ==========================================
GEMINI_API_KEY = os.getenv("Google_Api_Key2")

if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not set in environment")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")


# ==========================================
# 🌍 LANGUAGE CONFIG
# ==========================================
LANGUAGE_CONFIG = {
    "english": {"code": "en", "labels": {
        "materials": "Materials required.",
        "mistakes": "Common mistakes.",
        "timeline": "Timeline."
    }},
    "hindi": {"code": "hi", "labels": {
        "materials": "सामग्री।",
        "mistakes": "सामान्य गलतियाँ।",
        "timeline": "समयरेखा।"
    }},
    "tamil": {"code": "ta", "labels": {
        "materials": "தேவையான பொருட்கள்.",
        "mistakes": "பொதுவான தவறுகள்.",
        "timeline": "நேர அட்டவணை."
    }}
}


def resolve_language(language: str):
    return LANGUAGE_CONFIG.get(language.lower(), LANGUAGE_CONFIG["english"])


# ==========================================
# 🧠 CORE ENGINE CLASS
# ==========================================
class AIExplainerEngine:

    def __init__(self):
        pass

    # -------------------------------
    # CONTEXT BUILDER
    # -------------------------------
    def build_context(self, user_input, user_data=None):
        context = {"query": user_input}
        if user_data:
            context.update({k: v for k, v in user_data.items() if v})
        return context

    # -------------------------------
    # RULE ENGINE
    # -------------------------------
    def apply_rules(self, context):
        rules = []

        try:
            ph = float(context.get("soil_ph", 0))
            if ph < 5.5:
                rules.append("Soil is acidic")
            elif ph > 7.5:
                rules.append("Soil is alkaline")
        except:
            pass

        return rules

    # -------------------------------
    # PROMPT BUILDER
    # -------------------------------
    def build_prompt(self, context, rules):
        return f"""
You are an expert agriculture teacher conducting a structured lesson.

Context:
{json.dumps(context, indent=2, ensure_ascii=False)}

Rules:
{rules}

STRICT RULES:
- Teach step-by-step
- No unnecessary explanation
- Output ONLY valid JSON

FORMAT:
{{
  "Lesson Title": "",
  "Materials Needed": [],
  "Step 1": "",
  "Step 2": "",
  "Step 3": "",
  "Step 4": "",
  "Step 5": "",
  "Common Mistakes": [],
  "Pro Tip": "",
  "Summary": ""
}}
"""

    # -------------------------------
    # GEMINI CALL
    # -------------------------------
    def generate_lesson(self, prompt):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            raise RuntimeError(f"AI generation failed: {str(e)}")

    # -------------------------------
    # JSON PARSER
    # -------------------------------
    def parse_json(self, text):
        match = re.search(r"\{[\s\S]*\}", text)

        if not match:
            raise ValueError("❌ No valid JSON from AI")

        try:
            return json.loads(match.group())
        except:
            raise ValueError("❌ JSON parsing failed")

    # -------------------------------
    # TRANSLATION (OPTIONAL)
    # -------------------------------
    def translate(self, data, language):
        if language.lower() == "english":
            return data

        prompt = f"""
Translate this JSON into {language}.
Keep keys unchanged.

Return ONLY JSON.

{json.dumps(data, indent=2, ensure_ascii=False)}
"""
        raw = model.generate_content(prompt).text
        return self.parse_json(raw)

    # -------------------------------
    # AUDIO GENERATION
    # -------------------------------
    def generate_audio(self, lesson, language):
        folder = "static/audio"
        os.makedirs(folder, exist_ok=True)

        lang_cfg = resolve_language(language)
        lang_code = lang_cfg["code"]

        audio_map = {}

        for key, value in lesson.items():
            if isinstance(value, str) and value.strip():

                file_id = uuid.uuid4().hex[:8]
                filename = f"{key.replace(' ', '_').lower()}_{file_id}.mp3"
                filepath = os.path.join(folder, filename)

                try:
                    gTTS(text=value, lang=lang_code).save(filepath)
                    audio_map[key] = f"/{filepath}"
                except Exception:
                    continue

        return audio_map

    # -------------------------------
    # MAIN PIPELINE
    # -------------------------------
    def run(self, user_input, language="english", user_data=None):

        context = self.build_context(user_input, user_data)
        rules = self.apply_rules(context)

        prompt = self.build_prompt(context, rules)

        raw = self.generate_lesson(prompt)
        lesson = self.parse_json(raw)

        lesson = self.translate(lesson, language)

        audio = self.generate_audio(lesson, language)

        return {
            "lesson": lesson,
            "audio_chunks": audio
        }


# ==========================================
# 🔥 SINGLE ENTRY FUNCTION (FOR FLASK)
# ==========================================
engine = AIExplainerEngine()

def run_ai_explainer(user_input, language="english", user_data=None):
    return engine.run(user_input, language, user_data)