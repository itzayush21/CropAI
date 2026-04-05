# ================================================================
# 🌾 CropAI – Fertilizer Advisor (Gemini + Optional RAG Integration)
# ================================================================

import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
import traceback

# Optional RAG imports
import chromadb
from sentence_transformers import SentenceTransformer

# ----------------------------
# 🔑 Setup Gemini
# ----------------------------
load_dotenv()
genai.configure(api_key=os.getenv("Google_Api_Key2"))
model = genai.GenerativeModel("gemini-2.5-flash")

# ----------------------------
# 🧠 RAG Initialization (Safe)
# ----------------------------
try:
    embedding_model = SentenceTransformer(r"./minilm_model", device="cpu")
    print("✅ RAG embedding model loaded successfully.")
except Exception as e:
    print("⚠️ Failed to load embedding model:", e)
    embedding_model = None

def get_safe_chroma_client(db_path="./rag_model/fertilizer_chromadb"):
    try:
        os.makedirs(db_path, exist_ok=True)
        client = chromadb.PersistentClient(path=db_path)
        _ = client.list_collections()
        print(f"✅ ChromaDB connected at {os.path.abspath(db_path)}")
        return client
    except Exception as e:
        print("⚠️ Persistent ChromaDB failed, switching to in-memory mode:", e)
        return chromadb.Client()

try:
    client = get_safe_chroma_client()
    collection = client.get_or_create_collection("fertilizer_docs")
    print(f"📘 Using RAG collection: {collection.name}")
except Exception as e:
    print("⚠️ RAG unavailable:", e)
    client = None
    collection = None

# ----------------------------
# 🔍 RAG Retrieval
# ----------------------------
def retrieve_rag_context(query_text: str, n_results: int = 5) -> str:
    """Retrieve relevant fertilizer info from ChromaDB (fallback safe)."""
    if not collection or not embedding_model:
        return "No RAG context available."
    try:
        emb = embedding_model.encode(query_text).tolist()
        results = collection.query(query_embeddings=[emb], n_results=n_results)
        if not results or not results.get("documents") or not results["documents"][0]:
            return "No RAG context available."
        docs = results["documents"][0]
        print(f"✅ Retrieved {len(docs)} fertilizer context docs from RAG.")
        return "\n\n".join(docs)
    except Exception as e:
        print("⚠️ RAG retrieval error:", e)
        traceback.print_exc(limit=1)
        return "No RAG context available."

# ----------------------------
# 💬 Conversation Memory
# ----------------------------
CONVERSATION_MEMORY = {}

# ----------------------------
# 🌾 Fertilizer Advisor with RAG
# ----------------------------
def generate_fertilizer_recommendations(context, session_id):
    """
    Conversational fertilizer advisor with memory + RAG.
    Returns structured JSON (reply, recommendations, schedule, insights).
    """

    if session_id not in CONVERSATION_MEMORY:
        CONVERSATION_MEMORY[session_id] = []

    history = CONVERSATION_MEMORY[session_id]

    # Build RAG query text from context
    query_text = (
    f"The farmer is growing {context.get('crop', 'a crop')} in {context.get('soil_type', 'unspecified')} soil "
    f"with a pH of around {context.get('soil_ph', '7.0')} and {context.get('irrigation', 'moderate')} irrigation. "
    f"The field has received about {context.get('rainfall', 'normal')} millimeters of rainfall. "
    f"The crop shows the following problem: {context.get('deficiency_symptoms', 'general nutrient stress')}. "
    f"Suggest appropriate fertilizers and nutrient management practices, including chemical options like urea, DAP, MOP, "
    f"and organic or biofertilizers, following Indian agricultural recommendations and soil management guidelines."
)
    print(query_text)


    # Retrieve RAG context
    print("\n🔍 Fetching RAG context for fertilizer recommendation...")
    rag_context = retrieve_rag_context(query_text)
    print("📗 RAG Context (first 300 chars):", rag_context[:300])

    # Inject RAG context into the existing Gemini prompt
    prompt = f"""
    You are CropAI’s Fertilizer Advisor, an interactive agricultural assistant.
    Analyze the provided context carefully and return top 5 fertilizers recommendation.

    Retrieved Knowledge (RAG Context, Do not give much importance if irrelevant):
    {rag_context}

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
    - Provide practical, India-specific top 5 fertilizer advice.
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
        print("⚠️ Gemini pipeline error:", e)
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


