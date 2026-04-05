# module/nearby_services_engine.py

import requests
import math

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
HEADERS = {"User-Agent": "CropAI/1.0"}
RADII = [5000, 10000, 20000]

SERVICE_TAGS = {
    "fertilizer": [
        "amenity=agricultural_supplier",
        "shop=agriculture",
        "shop=farm"
    ],
    "market": [
        "amenity=marketplace",
        "shop=supermarket"
    ],
    "warehouse": [
        "building=warehouse",
        "man_made=storage"
    ]
}


# ==========================================
# 📏 DISTANCE
# ==========================================
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2 +
        math.cos(math.radians(lat1)) *
        math.cos(math.radians(lat2)) *
        math.sin(dlon / 2) ** 2
    )
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))


# ==========================================
# 🔍 OSM FETCH
# ==========================================
def build_query(lat, lon, radius, tags):
    parts = []
    for tag in tags:
        key, value = tag.split("=")
        parts.append(f'node["{key}"="{value}"](around:{radius},{lat},{lon});')
        parts.append(f'way["{key}"="{value}"](around:{radius},{lat},{lon});')

    return f"""
    [out:json];
    (
        {"".join(parts)}
    );
    out center;
    """


def fetch_osm_services(lat, lon, service_type):

    for radius in RADII:
        try:
            query = build_query(lat, lon, radius, SERVICE_TAGS[service_type])

            res = requests.post(
                OVERPASS_URL,
                data=query,
                headers=HEADERS,
                timeout=20
            )

            data = res.json()

        except Exception:
            continue

        results = []

        for el in data.get("elements", []):
            try:
                if "lat" in el:
                    lat2, lon2 = el["lat"], el["lon"]
                else:
                    lat2, lon2 = el["center"]["lat"], el["center"]["lon"]

                dist = calculate_distance(lat, lon, lat2, lon2)

                results.append({
                    "name": el.get("tags", {}).get("name", "Local Store"),
                    "lat": lat2,
                    "lon": lon2,
                    "distance_km": round(dist, 2),
                    "source": "OSM"
                })
            except:
                continue

        if results:
            return sorted(results, key=lambda x: x["distance_km"])[:10]

    return []


# ==========================================
# 🚀 MAIN FUNCTION
# ==========================================
def find_nearby_services(lat, lon):

    output = {}

    for svc in ["fertilizer", "market", "warehouse"]:
        output[svc] = fetch_osm_services(lat, lon, svc)

    return output