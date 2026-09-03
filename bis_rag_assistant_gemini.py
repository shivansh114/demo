"""
BIS AI Assistant - Prototype (Step 3: RAG with LLM - Google Gemini version)
-------------------------------------------------------------------------------
This connects the TF-IDF retrieval with Google's Gemini API (free tier,
no credit card required) to generate natural-language answers that are
GROUNDED in the retrieved BIS data - not the model's general knowledge.

RAG (Retrieval-Augmented Generation) pattern:
  1. User asks a question
  2. Retrieve the most relevant product(s) from bis_mock_dataset.json
  3. Pass ONLY that retrieved data + the user's question to the LLM
  4. LLM generates a natural, cited answer using ONLY that context

Setup:
    pip install scikit-learn google-genai

    Get a free API key (no credit card needed):
      1. Go to https://aistudio.google.com
      2. Sign in with your Google account
      3. Click "Get API Key" -> "Create API Key"

    Windows PowerShell:
        $env:GEMINI_API_KEY="your-key-here"
    Windows cmd:
        set GEMINI_API_KEY=your-key-here
    Mac/Linux:
        export GEMINI_API_KEY="your-key-here"

Usage:
    python bis_rag_assistant_gemini.py
"""

import json
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from google import genai
except ImportError:
    print("Missing dependency. Run: pip install google-genai")
    sys.exit(1)

DATA_FILE = os.path.join(os.path.dirname(__file__), "bis_mock_dataset.json")
MODEL_NAME = "gemini-3.6-flash"  # free tier model


# ---------- Data loading & retrieval ----------

def load_data():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def build_corpus(products):
    return [
        f"{p['product_name']} {p['category']} {p['scope_summary']}"
        for p in products
    ]


def build_index(products):
    corpus = build_corpus(products)
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, tfidf_matrix


def retrieve(query, products, vectorizer, tfidf_matrix, top_n=3, cutoff=0.05):
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    ranked_idx = sims.argsort()[::-1][:top_n]
    return [(products[i], sims[i]) for i in ranked_idx if sims[i] >= cutoff]


# ---------- RAG: build context + call LLM ----------

def build_context_block(matches, workflow_steps):
    blocks = []
    for product, score in matches:
        scheme = product["scheme"]
        steps = workflow_steps.get(scheme, [])
        steps_text = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps))

        block = f"""
Product: {product['product_name']}
Category: {product['category']}
Applicable IS Standard: {product['is_standard']}
Certification Scheme: {scheme} ({'Mandatory' if product['mandatory'] else 'Voluntary'})
Legal Basis: {product['legal_basis']}
Scope: {product['scope_summary']}
Regulating Body: {product['regulating_ministry']}
Certification Steps:
{steps_text if steps_text else "  (No mandatory workflow - voluntary certification only)"}
""".strip()
        blocks.append(block)

    return "\n\n---\n\n".join(blocks)


SYSTEM_PROMPT = """You are the BIS AI Assistant, an official-style helper for the \
Bureau of Indian Standards (BIS). You help industries and consumers understand \
applicable Indian Standards, certification schemes, and licensing steps.

STRICT RULES:
1. If RETRIEVED DATA is provided below, answer ONLY using that context. Do not \
use outside knowledge about specific BIS standards, even if you think you know more.
2. If RETRIEVED DATA says "NONE" (no matching product was found), do NOT invent \
a standard or product match. Instead: if the user's message is a greeting or \
small talk (e.g. "hi", "hello", "thanks"), respond warmly and briefly explain \
what you can help with (e.g. "Hi! Tell me a product name like 'pressure cooker' \
or 'LED bulb' and I'll find the applicable Indian Standard and certification \
steps for it."). If it looks like a genuine product question but nothing matched, \
say you couldn't find that product in the current dataset and ask them to \
rephrase or try a different product name.
3. Always mention the exact IS standard number and scheme type when relevant \
and available in RETRIEVED DATA.
4. Keep answers concise, clear, and practical - written for a small business \
owner or consumer, not a legal expert.
5. When explaining certification steps, present them in order.
"""


def ask_llm(client, user_query, context_block):
    context_text = context_block if context_block else "NONE"
    full_prompt = f"""{SYSTEM_PROMPT}

RETRIEVED DATA:
{context_text}

USER MESSAGE:
{user_query}

Respond following the system rules above."""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=full_prompt,
    )
    return response.text


# ---------- Main loop ----------

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable not set.")
        print("Get a free key at https://aistudio.google.com (no card needed),")
        print("then set it, e.g.:")
        print('  PowerShell : $env:GEMINI_API_KEY="your-key-here"')
        print("  cmd        : set GEMINI_API_KEY=your-key-here")
        print('  Mac/Linux  : export GEMINI_API_KEY="your-key-here"')
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    data = load_data()
    products = data["products"]
    workflow_steps = data["certification_workflow_steps"]
    vectorizer, tfidf_matrix = build_index(products)

    print("=" * 60)
    print(" BIS AI Assistant - Prototype (Step 3: RAG + Gemini)")
    print("=" * 60)
    print("Ask a natural question, e.g.:")
    print("  'What standard applies to my LED bulbs?'")
    print("  'How do I get ISI certification for a pressure cooker?'")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("You > ").strip()
        if query.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if not query:
            continue

        matches = retrieve(query, products, vectorizer, tfidf_matrix)
        context_block = build_context_block(matches, workflow_steps) if matches else None

        try:
            answer = ask_llm(client, query, context_block)
        except Exception as e:
            print(f"\nAssistant > (Error calling Gemini API: {e})\n")
            continue

        print(f"\nAssistant > {answer}\n")


if __name__ == "__main__":
    main()
