# ================================================================
# 🌾 CropAI — Gemini + Optional RAG Integration (Stable + Safe)
# ================================================================

import os, json, time, re, traceback, shutil
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import dotenv
dotenv.load_dotenv()

# ================================================================
# 🔑 Gemini Setup
# ================================================================
genai.configure(api_key=os.getenv("Google_Api_Key2"))
MODEL_NAME = "gemini-2.0-flash"

# ================================================================
# 🧠 Local Embedding Model (Optional RAG)
# ================================================================
try:
    embedding_model = SentenceTransformer(r"./minilm_model", device="cpu")
    print("✅ Loaded local embedding model.")
except Exception as e:
    print("⚠️ Failed to load embedding model:", e)
    embedding_model = None

# ================================================================
# 🌾 Safe ChromaDB Loader (Handles Missing / Corrupted DB)
# ================================================================
def get_safe_chroma_client(db_path: str = "./rag_model/cropai_chromadb"):
    import subprocess, sys
    print("\n🔧 Initializing ChromaDB client…")
    os.makedirs(db_path, exist_ok=True)

    try:
        client = chromadb.PersistentClient(path=db_path)
        _ = client.list_collections()
        print(f"✅ Using persistent ChromaDB at {os.path.abspath(db_path)}")
        return client
    except Exception as e:
        print("⚠️ Persistent mode failed:", e)
        print("🚨 Falling back to in-memory ChromaDB (no persistence).")
        return chromadb.Client()

try:
    client = get_safe_chroma_client("./rag_model/cropai_chromadb")
    collection = client.get_or_create_collection("cropai_embeddings")
    print(f"🗂️ Using ChromaDB collection: {collection.name}")
except Exception as e:
    print("⚠️ RAG unavailable:", e)
    client = None
    collection = None

# ================================================================
# 🔍 RAG Retriever (Optional)
# ================================================================
def retrieve_rag_context(query_text: str, n_results: int = 5) -> str:
    """Retrieve most relevant stored agricultural info, fallback-safe."""
    if not collection or not embedding_model:
        return "No retrieved context available (RAG disabled)."
    try:
        embedding = embedding_model.encode(query_text).tolist()
        results = collection.query(query_embeddings=[embedding], n_results=n_results)
        if not results or not results.get("documents") or not results["documents"][0]:
            return "No retrieved context available."
        docs = results["documents"][0]
        print(f"✅ RAG retrieved {len(docs)} context items.")
        return "\n\n".join(docs)
    except Exception as e:
        print("⚠️ RAG retrieval error:", e)
        return "No retrieved context available."

# ================================================================
# 🧩 JSON Schema (From Your Original Pipeline)
# ================================================================
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

# ================================================================
# 🧩 Safe JSON Parser (Unchanged)
# ================================================================
def safe_json_loads(gemini_output: str):
    try:
        match = re.search(r'\{.*\}', gemini_output, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in text")
        json_text = match.group(0)
        json_text = json_text.replace('“', '"').replace('”', '"').replace('’', "'")
        json_text = re.sub(r',(\s*[}\]])', r'\1', json_text)
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        print("JSON decoding error:", e)
        print("Raw text:", gemini_output[:500])
        return None

# ================================================================
# 🧠 Prompt Builder (Now with Optional RAG Context)
# ================================================================
def _build_prompt(context_json: dict, user_query: str | None, rag_context: str | None = None) -> str:
    rag_part = (
        f"\nRETRIEVED KNOWLEDGE (RAG CONTEXT):\n{rag_context}\n\n"
        if rag_context and "No retrieved context" not in rag_context
        else "\n(RETRIEVED KNOWLEDGE unavailable — using Gemini reasoning only)\n"
    )
    return f"""
You are CropAI's expert agronomy assistant.
{rag_part}
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

# ================================================================
# 🌾 Gemini + RAG Integrated Main Function (Yours + RAG Added)
# ================================================================
def generate_crop_suggestions(context_json: dict, user_query: str | None = None) -> dict:
    """Gemini pipeline + optional RAG retrieval (never fails silently)."""
    model = genai.GenerativeModel(MODEL_NAME)
    max_retries = 3
    retry_delay = 2

    # ---- Build query for RAG ----
    query_text = (
        f"Suggest crops for {context_json.get('district','')} in {context_json.get('state','')} "
        f"with {context_json.get('soil_type','')} soil, pH {context_json.get('soil_ph','')}, "
        f"rainfall {context_json.get('rainfall','')} mm, temperature {context_json.get('temperature','')}°C, "
        f"irrigation {context_json.get('irrigation','')}, season {context_json.get('season','')}, "
        f"and preference {context_json.get('preference','')}."
    )

    print("\n🔍 Fetching RAG context...")
    rag_context = retrieve_rag_context(query_text)
    print("📘 RAG context (first 300 chars):", rag_context[:300])

    prompt = _build_prompt(context_json, user_query, rag_context)
    resp = None
    parsed = None

    for attempt in range(max_retries + 1):
        print(f"\n🧠 Attempt {attempt + 1}: Requesting Gemini response...")
        try:
            resp = model.generate_content(prompt)
        except Exception as e:
            print(f"⚠️ Gemini API error on attempt {attempt + 1}: {e}")
            if attempt < max_retries:
                time.sleep(retry_delay)
                continue
            raise RuntimeError(f"Gemini API call failed after {max_retries + 1} attempts: {e}")

        if not hasattr(resp, "text") or not resp.text.strip():
            print("⚠️ Empty response from Gemini.")
            if attempt < max_retries:
                time.sleep(retry_delay)
                continue
            raise RuntimeError("Gemini returned empty text after retries.")

        parsed = safe_json_loads(resp.text)
        if parsed:
            print(f"✅ Successfully parsed JSON on attempt {attempt + 1}")
            return parsed

        print(f"⚠️ Invalid JSON on attempt {attempt + 1}. Retrying in {retry_delay}s...")
        time.sleep(retry_delay)

    print("❌ All retries failed. Returning raw output for debugging.")
    return {
        "error": "Invalid JSON after retries",
        "raw_output": getattr(resp, "text", None)
    }


