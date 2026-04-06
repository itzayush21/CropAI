# ============================================================
# 🧠 CropAI — Inventory AI Summary Module (Gemini)
# ============================================================

import json
import re
import time
import google.generativeai as genai


# ============================================================
# 🧩 SAFE JSON PARSER
# ============================================================
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


# ============================================================
# 🧠 PROMPT BUILDER
# ============================================================
def build_inventory_prompt(inventory):

    return f"""
You are CropAI's intelligent farm inventory advisor.

Analyze the farmer's inventory and generate:

1. A concise summary of current inventory status
2. Key risks (low stock, expiry, overuse)
3. Actionable recommendations

INVENTORY DATA:
{json.dumps(inventory, indent=2)}

RULES:
- Output STRICT JSON ONLY
- Be practical for Indian farmers
- Keep summary short but insightful

FORMAT:
{{
  "summary": "short overview",
  "risks": ["risk1","risk2"],
  "recommendations": ["action1","action2"]
}}

DO NOT output anything outside JSON.
"""


# ============================================================
# 🚀 GEMINI GENERATION
# ============================================================
def generate_inventory_summary(inventory, retries=3):

    model = genai.GenerativeModel(
        "gemini-2.5-flash",
        generation_config={"response_mime_type": "application/json"}
    )

    prompt = build_inventory_prompt(inventory)

    last_output = None

    for _ in range(retries):
        try:
            res = model.generate_content(prompt)
            last_output = res.text

            parsed = safe_json_loads(last_output)

            if parsed:
                return parsed

        except Exception as e:
            print("[GENAI ERROR]", e)

        time.sleep(1)

    return {
        "summary": "Unable to generate AI insights",
        "risks": [],
        "recommendations": []
    }