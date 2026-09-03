"""
BIS AI Assistant - Prototype (Step 6: True Agent with Tool Use)
-------------------------------------------------------------------
Upgrades the RAG pipeline into an AGENT: instead of always running
retrieval first and handing the LLM a fixed context block, Gemini is
given a set of TOOLS (functions) and decides for itself which one(s)
to call based on the user's message. This is real function calling /
tool use, not a scripted retrieve-then-generate flow.

Tools available to the agent:
  - search_product(query)              : find a product's IS standard/scheme
  - get_certification_steps(scheme)    : get certification workflow steps
  - list_product_categories()          : list all product categories covered
  - list_products_in_category(category): list products within a category

Setup:
    pip install flask google-genai scikit-learn

    Windows PowerShell:
        $env:GEMINI_API_KEY="your-key-here"

Usage:
    python app_agent.py
    Then open http://127.0.0.1:5000
"""

import json
import os
import sys
from flask import Flask, request, jsonify, send_from_directory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Missing dependency. Run: pip install google-genai")
    sys.exit(1)

DATA_FILE = os.path.join(os.path.dirname(__file__), "bis_mock_dataset.json")
MODEL_NAME = "gemini-3.6-flash"

app = Flask(__name__, static_folder="static")

# ---------- Load data once at startup ----------

with open(DATA_FILE, "r", encoding="utf-8") as f:
    _data = json.load(f)

_products = _data["products"]
_workflow_steps = _data["certification_workflow_steps"]

_corpus = [
    f"{p['product_name']} {p['category']} {p['scope_summary']}"
    for p in _products
]
_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
_tfidf_matrix = _vectorizer.fit_transform(_corpus)


# ---------- TOOLS ----------
# Each function below is exposed to the LLM as a callable tool.
# The docstring and type hints are what Gemini reads to understand
# what the tool does and what arguments it needs - keep them clear.

def search_product(query: str) -> str:
    """Search the BIS product database for a product matching the given
    query (a product name or description). Returns the applicable Indian
    Standard (IS number), certification scheme, whether it is mandatory,
    the legal basis, scope, and regulating body for the best-matching
    product(s). Returns an empty result if nothing matches well.

    Args:
        query: A product name or short description, e.g. "pressure cooker"
            or "LED bulb" or "kitchen cooking vessel".
    """
    query_vec = _vectorizer.transform([query])
    sims = cosine_similarity(query_vec, _tfidf_matrix).flatten()
    ranked_idx = sims.argsort()[::-1][:3]
    matches = [(_products[i], sims[i]) for i in ranked_idx if sims[i] >= 0.05]

    if not matches:
        return json.dumps({"found": False, "message": "No matching product found in the database."})

    results = []
    for product, score in matches:
        results.append({
            "product_name": product["product_name"],
            "category": product["category"],
            "is_standard": product["is_standard"],
            "scheme": product["scheme"],
            "mandatory": product["mandatory"],
            "legal_basis": product["legal_basis"],
            "scope_summary": product["scope_summary"],
            "regulating_ministry": product["regulating_ministry"],
            "match_confidence": round(float(score), 3),
        })
    return json.dumps({"found": True, "results": results})


def get_certification_steps(scheme: str) -> str:
    """Get the ordered certification/licensing steps for a given BIS
    certification scheme.

    Args:
        scheme: The scheme name. Must be one of "ISI", "CRS", or
            "Hallmarking" (case-sensitive, use exactly these values).
    """
    steps = _workflow_steps.get(scheme)
    if not steps:
        return json.dumps({
            "found": False,
            "message": f"No workflow found for scheme '{scheme}'. "
                       f"Valid schemes are: {list(_workflow_steps.keys())}"
        })
    return json.dumps({"found": True, "scheme": scheme, "steps": steps})


def list_product_categories() -> str:
    """List all distinct product categories currently covered in the
    BIS product database (e.g. 'Electronics & IT', 'Construction
    Materials', 'Jewellery'). Useful when the user wants to browse
    what's covered rather than search for a specific product.
    """
    categories = sorted(set(p["category"] for p in _products))
    return json.dumps({"categories": categories})


def list_products_in_category(category: str) -> str:
    """List all products in the database belonging to a given category.

    Args:
        category: A category name, e.g. "Electronics & IT" or
            "Construction Materials". Should match (or closely match)
            one of the categories returned by list_product_categories.
    """
    category_lower = category.lower()
    matches = [
        p["product_name"] for p in _products
        if category_lower in p["category"].lower()
    ]
    if not matches:
        all_categories = sorted(set(p["category"] for p in _products))
        return json.dumps({
            "found": False,
            "message": f"No products found in category '{category}'.",
            "available_categories": all_categories,
        })
    return json.dumps({"found": True, "category": category, "products": matches})


TOOLS = [search_product, get_certification_steps, list_product_categories, list_products_in_category]


# ---------- Agent system instruction ----------

LANGUAGE_INSTRUCTION = (
    "Match the language and style of the user's message:\n"
    "- Plain English -> respond in clear, simple English.\n"
    "- Hindi (Devanagari) -> respond in Hindi (Devanagari).\n"
    "- Hinglish (Hindi-English mix in Roman script) -> respond in Hinglish, Roman script.\n"
    "- DEFAULT TO ENGLISH for short, ambiguous messages (e.g. a bare product "
    "name, or a greeting like 'hi') unless they clearly contain Hindi/Hinglish "
    "words or grammar.\n"
    "Always keep technical terms (IS numbers, scheme names, product names) in English."
)

SYSTEM_PROMPT = f"""You are the BIS AI Assistant, an official-style agent for the \
Bureau of Indian Standards (BIS). You help industries and consumers find \
applicable Indian Standards, certification schemes, and licensing steps.

You have tools available to look up real data. USE THEM rather than answering \
from memory - you must never state an IS standard number, scheme, or \
certification step that didn't come from a tool result.

TOOL USE GUIDANCE:
- If the user asks about a specific product, call search_product first.
- If the user then asks how to get certified/licensed for that product, call \
get_certification_steps with the scheme name returned by search_product.
- If the user asks what products/categories you cover, call \
list_product_categories or list_products_in_category as appropriate.
- You may call multiple tools in sequence if needed to fully answer a question.
- If a tool returns "found": false, do not guess - tell the user honestly \
that it wasn't found, and suggest they rephrase or ask what's covered.
- For greetings or small talk with no product mentioned, respond warmly \
without calling any tool, and briefly explain what you can help with.

LANGUAGE: {LANGUAGE_INSTRUCTION}

Keep answers concise and practical, written for a small business owner or \
consumer, not a legal expert. Present certification steps in order.
"""


# ---------- Gemini agent setup ----------

_api_key = os.environ.get("GEMINI_API_KEY")
_client = genai.Client(api_key=_api_key) if _api_key else None

# One chat session per server process (fine for a single-user local demo).
# For multi-user production use, you'd keep a dict of chats keyed by session id.
_chat = None


def get_chat():
    global _chat
    if _chat is None:
        _chat = _client.chats.create(
            model=MODEL_NAME,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=TOOLS,
            ),
        )
    return _chat


# ---------- Routes ----------

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/chat", methods=["POST"])
def chat_endpoint():
    if _client is None:
        return jsonify({"error": "GEMINI_API_KEY is not set on the server."}), 500

    data = request.get_json(force=True)
    query = (data.get("message") or "").strip()
    if not query:
        return jsonify({"error": "Empty message."}), 400

    try:
        chat = get_chat()
        response = chat.send_message(query)
        answer = response.text
    except Exception as e:
        error_text = str(e)
        if "503" in error_text or "UNAVAILABLE" in error_text:
            return jsonify({"error": "Gemini's servers are busy right now. Please try again."}), 503
        return jsonify({"error": f"Agent error: {e}"}), 500

    return jsonify({"answer": answer})


@app.route("/api/reset", methods=["POST"])
def reset_chat():
    """Reset the conversation (clears the agent's memory of prior turns)."""
    global _chat
    _chat = None
    return jsonify({"status": "reset"})


if __name__ == "__main__":
    if not _api_key:
        print("WARNING: GEMINI_API_KEY not set. The chat endpoint will fail.")
    print("Starting agent server at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
