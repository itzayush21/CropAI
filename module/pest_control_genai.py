import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
import traceback
# Optional RAG imports
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()
genai.configure(api_key=os.getenv("Google_Api_Key2"))

model = genai.GenerativeModel("gemini-2.0-flash")

# Memory for conversational continuity (can later move to Redis or DB)
PEST_CONVERSATION_MEMORY = {}

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
    - provide practical, region-specific top 5 pest management advice.
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
