import os
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
import io

# Load environment variables from .env
load_dotenv()

# Fetch key
api_key = os.getenv("Google_Api_Key2")

if not api_key:
    raise ValueError("❌ GEMINI_API_KEY not found. Check your .env file and path.")

genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-2.5-flash")


def analyze_farmland_image(image_path):
    """Use Gemini AI to describe the land surface and conditions."""
    with open(image_path, 'rb') as f:
        img_bytes = f.read()
    image = Image.open(io.BytesIO(img_bytes))

    prompt = """
    You are an expert in analyzing agricultural land using images.

    Carefully observe the given image of farmland and provide summarized agricultural insights:
    1. Surface condition (e.g., dry, wet, cracked, grassy)
    2. Vegetation presence
    3. Moisture / waterlogging
    4. Soil fertility impression
    5. Terrain and topography
    Provide the result in 4-5 sentences only.
    """

    response = model.generate_content([prompt, image])
    return response.text


print(analyze_farmland_image('images\Gaya_20251104170515.png'))