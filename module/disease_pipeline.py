# ==========================================================
# 🌾 CropAI - AI Pipeline: YOLOv11 + Gemini 2.0 Flash
# ==========================================================
from ultralytics import YOLO
import google.generativeai as genai
from dotenv import load_dotenv
import os, torch

# ==========================================================
# 🔹 Environment Setup
# ==========================================================
load_dotenv()
genai.configure(api_key=os.getenv("Google_Api_Key2"))

# Initialize Gemini model
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# Load YOLOv11 classification model
model = YOLO("model/best.pt")  # 🔹 Replace with path to your trained weights


# ==========================================================
# 🧠 Step 1: YOLOv11 Classification
# ==========================================================
def run_yolo_classification(image_path: str):
    """
    Run YOLOv11 classification inference and return top-5 predictions.
    """
    try:
        results = model.predict(image_path, imgsz=224, conf=0.25, verbose=False)
        res = results[0]

        if res.probs is None:
            return {"error": "No classification probabilities found in model output."}

        probs = res.probs.data.cpu().numpy()
        top_indices = probs.argsort()[-5:][::-1]

        predictions = [
            {"class": res.names[i], "confidence": round(float(probs[i]), 3)}
            for i in top_indices
        ]

        return {"top1": predictions[0], "top5": predictions}

    except Exception as e:
        return {"error": str(e)}


# ==========================================================
# 🌿 Step 2: Gemini 2.0 Flash - Disease Guidance
# ==========================================================
def get_gemini_guidance(top_predictions, crop_type: str, preference: str):
    """
    Generate realistic disease diagnosis and guidance using Gemini 2.0 Flash.
    Focuses on practical crop symptom analysis and treatment correlation.
    """
    try:
        diseases_text = "\n".join([
            f"- {p['class']} ({p['confidence']*100:.1f}%)"
            for p in top_predictions
        ])
        main_disease = top_predictions[0]["class"]

        prompt = f"""
        You are an experienced agricultural scientist and plant pathologist assisting farmers with 
        accurate diagnosis and disease management of their crops.

        🌾 Crop Type: {crop_type}
        🧬 Detected Patterns (Top-5 Model Predictions):
        {diseases_text}

        The AI system suspects the crop might be showing symptoms similar to **{main_disease}**,
        but you must not depend only on the disease name. 
        Instead, reason scientifically about what conditions or pathogens could lead to these visual patterns 
        on the given crop.

        The farmer prefers **{preference}** treatment methods.

        Please provide:
        1. A realistic diagnosis of what might be happening to this crop (based on visual cues and common conditions).
        2. Probable causes (like fungus, bacteria, nutrient deficiency, or environmental stress).
        3. Observable symptoms to help confirm the situation in the field.
        4. Recommended immediate actions or treatments — tailored to {preference.lower()} farming style.
        5. Preventive steps and sustainable practices for future health.
        6. A short, reassuring message to guide the farmer toward sustainable recovery.

        ⚠️ Important:
        - Do not rigidly focus on the disease name given by the model.
        - Instead, relate your reasoning to the crop type, climate, and common agricultural conditions.
        - Your tone should be empathetic, clear, and easy for a rural farmer to understand.
        """

        response = gemini_model.generate_content(prompt)
        return response.text.strip() if response and response.text else "No response generated from Gemini."

    except Exception as e:
        return f"Gemini error: {str(e)}"

