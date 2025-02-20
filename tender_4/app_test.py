import os
import csv
import json
import numpy as np
import faiss
from flask import Flask, render_template, request, redirect, url_for
from sentence_transformers import SentenceTransformer
import openai  # GPT-4 API

# --------------------------------------------------------------------
# Load OpenAI API Key (Set externally for security)
# openai.api_key = "YOUR_OPENAI_API_KEY"

# --------------------------------------------------------------------
# Check and Download AllMiniLM-L6-v2 Model
# --------------------------------------------------------------------
MODEL_PATH = "model_repo/all-MiniLM-L6-v2"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def download_model_if_needed():
    if not os.path.exists(MODEL_PATH):
        print("🔹 Model not found. Downloading AllMiniLM-L6-v2...")
        model = SentenceTransformer(MODEL_NAME)
        model.save(MODEL_PATH)
        print("✅ Model downloaded and saved at:", MODEL_PATH)
    else:
        print("✅ Model already present at:", MODEL_PATH)

download_model_if_needed()

# Load SentenceTransformer for FAISS embeddings
embedding_model = SentenceTransformer(MODEL_PATH)
EMBEDDING_DIM = embedding_model.get_sentence_embedding_dimension()

app = Flask(__name__)

# --------------------------------------------------------------------
# GLOBAL VARIABLES
# --------------------------------------------------------------------
EVENT_INDEX = None
EVENTS_DATA = []

# --------------------------------------------------------------------
# FAISS Indexing for Events
# --------------------------------------------------------------------
def build_faiss_index_for_events(csv_path):
    global EVENT_INDEX, EVENTS_DATA
    events = []
    try:
        with open(csv_path, newline='', encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                events.append(row)
    except Exception as e:
        print("❌ Error loading event CSV:", e)
        return

    if not events:
        print("❌ No events found in CSV.")
        return

    texts = [
        " ".join([r.get("Title",""), r.get("Description",""), r.get("Commodity","")])
        for r in events
    ]
    embeddings = embedding_model.encode(texts, convert_to_numpy=True)

    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(embeddings)

    EVENT_INDEX = index
    EVENTS_DATA = events

def faiss_search_events(title, commodity, top_k=50):
    """FAISS search by event title and commodity."""
    if EVENT_INDEX is None:
        print("❌ FAISS not built.")
        return []

    query_text = f"{title} {commodity}".strip()
    query_emb = embedding_model.encode([query_text], convert_to_numpy=True)
    distances, indices = EVENT_INDEX.search(query_emb, top_k)
    
    return [EVENTS_DATA[idx] for idx in indices[0] if idx < len(EVENTS_DATA)]

# --------------------------------------------------------------------
# GPT-4 for Events (Full Event Handling)
# --------------------------------------------------------------------
def call_gpt4_for_events(events_list, user_form_data):
    """
    Calls GPT-4 for AI insights on events. Ensures full events are sent without splitting.
    """
    events_json = json.dumps(events_list, indent=2)
    prompt = f"""
    You are an AI procurement assistant. Based on these events (JSON format) and buyer input, provide:
    - score
    - reason
    - explanation
    - match_score
    - region
    - risks
    - AI insights: trends, optimizations.

    Events: {events_json}
    Buyer Input: {user_form_data}

    Respond with a JSON array where each object contains:
    {{"score": 95, "reason": "...", "explanation": "...", "match_score": 0.92, "region": "...",
    "risks": "...", "ai_insights": {{"trends": [...], "optimizations": [...]}} }}
    """

    response = openai.ChatCompletion.create(
        model='gpt-4',
        messages=[{'role': 'system', 'content': 'You are a procurement AI assistant.'},
                  {'role': 'user', 'content': prompt}],
        max_tokens=2000,
        temperature=0
    )

    return json.loads(response['choices'][0]['message']['content'])

# --------------------------------------------------------------------
# GPT-4 for Supplier Quotations
# --------------------------------------------------------------------
def call_gpt4_for_suppliers(supplier_text, buyer_req):
    """
    Calls GPT-4 for supplier insights.
    """
    prompt = f"""
    You are an AI procurement assistant. Based on these supplier quotations and buyer requirements:
    
    Buyer Requirements:
    {buyer_req}

    Supplier Quotations:
    {supplier_text}

    Provide JSON structured output ranking suppliers based on:
    - price_per_unit
    - delivery_date
    - additional_terms
    - score
    - AI insights: trends, risks, optimizations.
    """

    response = openai.ChatCompletion.create(
        model='gpt-4',
        messages=[{'role': 'system', 'content': 'You are a procurement AI assistant.'},
                  {'role': 'user', 'content': prompt}],
        max_tokens=2000,
        temperature=0
    )

    return json.loads(response['choices'][0]['message']['content'])

# --------------------------------------------------------------------
# Flask Setup
# --------------------------------------------------------------------
@app.before_first_request
def init_app():
    csv_path = "data/small_tender_system_event_data.csv"
    build_faiss_index_for_events(csv_path)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate_project", methods=["POST"])
def generate_project():
    form_data = request.form.to_dict()
    event_title = form_data.get("project_name", "").strip()
    commodity = form_data.get("commodity", "").strip()

    matched_events = faiss_search_events(event_title, commodity, top_k=50)

    # Call GPT-4 for event insights
    if matched_events:
        event_insights = call_gpt4_for_events(matched_events, form_data)
        for i, ev in enumerate(matched_events):
            ev["ai_data"] = event_insights[i]

    return render_template("compareEvents.html", events=matched_events, form_data=form_data)

@app.route("/compare_quotes/<event_id>")
def compare_quotes(event_id):
    txt_path = "data/supplier_quotations.txt"
    with open(txt_path, 'r', encoding='utf-8') as f:
        supplier_text = f.read()

    buyer_req = f"Event ID: {event_id}"
    supplier_insights_list = call_gpt4_for_suppliers(supplier_text, buyer_req)

    return render_template("compare_quotes.html", event_id=event_id, supplier_insights_list=supplier_insights_list)

@app.route("/award", methods=["POST"])
def award():
    selected_suppliers = request.form.getlist("selected_suppliers")
    return render_template("award_result.html", awarded_suppliers=selected_suppliers)

if __name__ == "__main__":
    app.run(debug=True)
