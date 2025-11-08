import os
import cv2
import io
import numpy as np
import threading
import requests
from datetime import datetime
from PIL import Image
from opencage.geocoder import OpenCageGeocode
import google.generativeai as genai
import dotenv

dotenv.load_dotenv()

# ---------- Gemini AI Setup ----------
genai.configure(api_key=os.getenv('Google_Api_Key2'))
model = genai.GenerativeModel('gemini-2.0-flash')

# ---------- OpenCage Setup ----------
GEOCODER_KEY = "7edf47c44cce409b892facc8ba369bbf"
geocoder = OpenCageGeocode(GEOCODER_KEY)


# ---------- 1️⃣ Geocode Location ----------
def get_lat_lon_from_location(address, district, state):
    """Get latitude and longitude using OpenCage API."""
    query = f"{address}, {district}, {state}, India"
    results = geocoder.geocode(query)
    if results and len(results):
        loc = results[0]['geometry']
        return loc['lat'], loc['lng']
    else:
        raise ValueError(f"Could not geocode location: {query}")


# ---------- 2️⃣ Satellite Image Downloader ----------
def project_with_scale(lat, lon, scale):
    siny = np.sin(lat * np.pi / 180)
    siny = min(max(siny, -0.9999), 0.9999)
    x = scale * (0.5 + lon / 360)
    y = scale * (0.5 - np.log((1 + siny) / (1 - siny)) / (4 * np.pi))
    return x, y

def download_tile(url, headers, channels):
    response = requests.get(url, headers=headers)
    arr = np.asarray(bytearray(response.content), dtype=np.uint8)
    return cv2.imdecode(arr, 1 if channels == 3 else -1)

def download_image(lat1, lon1, lat2, lon2, zoom, url, headers, tile_size=256, channels=3):
    scale = 1 << zoom
    tl_proj_x, tl_proj_y = project_with_scale(lat1, lon1, scale)
    br_proj_x, br_proj_y = project_with_scale(lat2, lon2, scale)

    tl_pixel_x = int(tl_proj_x * tile_size)
    tl_pixel_y = int(tl_proj_y * tile_size)
    br_pixel_x = int(br_proj_x * tile_size)
    br_pixel_y = int(br_proj_y * tile_size)

    tl_tile_x = int(tl_proj_x)
    tl_tile_y = int(tl_proj_y)
    br_tile_x = int(br_proj_x)
    br_tile_y = int(br_proj_y)

    img_w = abs(tl_pixel_x - br_pixel_x)
    img_h = br_pixel_y - tl_pixel_y
    img = np.zeros((img_h, img_w, channels), np.uint8)

    def build_row(tile_y):
        for tile_x in range(tl_tile_x, br_tile_x + 1):
            tile = download_tile(url.format(x=tile_x, y=tile_y, z=zoom), headers, channels)
            if tile is not None:
                tl_rel_x = tile_x * tile_size - tl_pixel_x
                tl_rel_y = tile_y * tile_size - tl_pixel_y
                br_rel_x = tl_rel_x + tile_size
                br_rel_y = tl_rel_y + tile_size

                img_x_l = max(0, tl_rel_x)
                img_x_r = min(img_w + 1, br_rel_x)
                img_y_l = max(0, tl_rel_y)
                img_y_r = min(img_h + 1, br_rel_y)

                cr_x_l = max(0, -tl_rel_x)
                cr_x_r = tile_size + min(0, img_w - br_rel_x)
                cr_y_l = max(0, -tl_rel_y)
                cr_y_r = tile_size + min(0, img_h - br_rel_y)

                img[img_y_l:img_y_r, img_x_l:img_x_r] = tile[cr_y_l:cr_y_r, cr_x_l:cr_x_r]

    threads = []
    for tile_y in range(tl_tile_y, br_tile_y + 1):
        t = threading.Thread(target=build_row, args=[tile_y])
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    return img


# ---------- 3️⃣ Compute NDVI / NDWI ----------
def compute_indices(image):
    """Compute NDVI & NDWI (RGB approximation)."""
    b, g, r = cv2.split(image.astype(float))
    ndvi = (r - b) / (r + b + 1e-6)
    ndwi = (g - r) / (g + r + 1e-6)
    return float(np.mean(ndvi)), float(np.mean(ndwi))


# ---------- 4️⃣ Gemini Analysis ----------
def analyze_farmland_image(image_path):
    """Use Gemini AI to describe land condition."""
    with open(image_path, 'rb') as f:
        img_bytes = f.read()
    image = Image.open(io.BytesIO(img_bytes))

    prompt = """
    You are an expert in agricultural land analysis.
    Observe the given farmland image and describe:
    - Surface condition (dry/wet/vegetated)
    - Vegetation density
    - Moisture or waterlogging signs
    - Terrain condition (flat/slope)
    - Overall land usability
    Respond in 3–5 concise sentences.
    """

    response = model.generate_content([prompt, image])
    return response.text


# ---------- 5️⃣ Unified Preprocessing Pipeline ----------
def enrich_user_data(address=None, district=None, state=None, latitude=None, longitude=None):
    """
    1. If latitude & longitude given → use directly.
    2. Else → geocode using address, district, state.
    3. Download satellite image.
    4. Compute NDVI/NDWI.
    5. Analyze land using Gemini.
    """

    # Determine coordinates
    if latitude and longitude:
        lat, lon = float(latitude), float(longitude)
        print(f"📍 Using provided coordinates: {lat}, {lon}")
    else:
        lat, lon = get_lat_lon_from_location(address, district, state)
        print(f"📍 Coordinates from geocoding: {lat}, {lon}")

    prefs = {
        'url': 'https://mt.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        'tile_size': 256,
        'channels': 3,
        'headers': {'user-agent': 'Mozilla/5.0'},
    }

    delta = 0.01
    lat1, lon1 = lat + delta, lon - delta
    lat2, lon2 = lat - delta, lon + delta

    img = download_image(lat1, lon1, lat2, lon2, zoom=17, url=prefs['url'], headers=prefs['headers'])
    os.makedirs("images", exist_ok=True)
    image_path = f"images/farmland_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    cv2.imwrite(image_path, img)

    ndvi, ndwi = compute_indices(img)
    land_summary = analyze_farmland_image(image_path)

    return {
        "latitude": lat,
        "longitude": lon,
        "ndvi": round(ndvi, 4),
        "ndwi": round(ndwi, 4),
        "land_summary": land_summary
    }
