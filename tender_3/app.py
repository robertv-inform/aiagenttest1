import os
import csv
import json
import numpy as np
import faiss
from flask import Flask, render_template, request, redirect, url_for
from sentence_transformers import SentenceTransformer

# Uncomment these lines to enable real GPT-4 usage:
# import openai
# openai.api_key = "YOUR_OPENAI_API_KEY"

# --------------------------------------------------------------------
# Check and Download AllMiniLM-L6-v2 Model
# --------------------------------------------------------------------
MODEL_PATH = "model_repo/all-MiniLM-L6-v2"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def download_model_if_needed():
    if not os.path.exists(MODEL_PATH):
        print("🔹 Model not found. Downloading AllMiniLM-L6-v2...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL_NAME)
        model.save(MODEL_PATH)
        print("✅ Model downloaded and saved at:", MODEL_PATH)
    else:
        print("✅ Model already present at:", MODEL_PATH)

download_model_if_needed()

# Load the AllMiniLM-L6-v2 model for FAISS embeddings
embedding_model = SentenceTransformer(MODEL_PATH)
EMBEDDING_DIM = embedding_model.get_sentence_embedding_dimension()

app = Flask(__name__)

# --------------------------------------------------------------------
# GLOBALS for events
# --------------------------------------------------------------------
EVENT_INDEX = None
EVENT_EMBEDDINGS = None
EVENTS_DATA = []

# --------------------------------------------------------------------
# Build FAISS index from CSV
# --------------------------------------------------------------------
def build_faiss_index_for_events(csv_path):
    global EVENT_INDEX, EVENT_EMBEDDINGS, EVENTS_DATA
    events = []
    try:
        with open(csv_path, newline='', encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                events.append(row)
    except Exception as e:
        print("Error loading events CSV:", e)
        return

    if not events:
        print("No events found in CSV.")
        return

    # Convert each row to a single string for embeddings
    texts = [" ".join([row.get("Title",""), row.get("Description",""), row.get("Commodity","")]) for row in events]
    embeddings = embedding_model.encode(texts, convert_to_numpy=True)

    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(embeddings)

    EVENT_INDEX = index
    EVENT_EMBEDDINGS = embeddings
    EVENTS_DATA = events

def faiss_search_events(commodity, top_k=50):
    """Filter events by commodity using FAISS index."""
    if EVENT_INDEX is None:
        print("FAISS index not built.")
        return []
    commodity_emb = embedding_model.encode([commodity], convert_to_numpy=True)
    distances, indices = EVENT_INDEX.search(commodity_emb, top_k)
    return [EVENTS_DATA[idx] for idx in indices[0] if idx < len(EVENTS_DATA)]

# --------------------------------------------------------------------
# Dummy GPT Functions (OpenAI code commented)
# --------------------------------------------------------------------
def dummy_gpt4_event_insights(event):
    """Returns dummy AI insights for a single event."""
    return {
        "score": np.random.randint(70, 99),
        "reason": "Dummy synergy reason",
        "explanation": "Dummy chunk-based explanation",
        "match_score": round(np.random.rand(), 2),
        "region": event.get("Region", "Unknown"),
        "risks": "Minimal risk from dummy data",
        "ai_insights": {
            "trends": ["Dummy trend in region."],
            "optimizations": ["Dummy optimization approach."]
        }
    }

def dummy_gpt4_supplier_insights():
    """Returns dummy AI insights for suppliers, no raw data."""
    return [
        {
            "rank": 1,
            "supplier_name": "Supplier A",
            "price_per_unit": "$1,200",
            "delivery_date": "2025-03-01",
            "additional_terms": "Fast shipping, extended warranty",
            "score": 95,
            "explanation": "Most competitive pricing and good terms.",
            "ai_suggested_percentage": 35
        },
        {
            "rank": 2,
            "supplier_name": "Supplier B",
            "price_per_unit": "$1,250",
            "delivery_date": "2025-03-05",
            "additional_terms": "5% discount on large orders",
            "score": 90,
            "explanation": "Slightly higher cost but flexible terms.",
            "ai_suggested_percentage": 30
        }
    ]

# Example of chunk-based approach if you need to handle large data
def chunk_data_for_gpt4(data_list, chunk_size=5):
    for i in range(0, len(data_list), chunk_size):
        yield data_list[i:i+chunk_size]

"""
def call_gpt4_for_events_batched(events_list, form_data):
    # Real GPT-4 code goes here
    # We'll chunk events_list, produce final results, etc.
    pass
"""

"""
def call_gpt4_for_suppliers(event_data, supplier_text):
    # Real GPT-4 code to parse supplier data
    pass
"""

# --------------------------------------------------------------------
# Supplier Quotation Loader (Optional)
# --------------------------------------------------------------------
def load_supplier_quotations(txt_path):
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print("Error loading supplier quotations:", e)
        return ""

# --------------------------------------------------------------------
# Flask Setup
# --------------------------------------------------------------------
@app.before_first_request
def init_app():
    csv_path = os.path.join("data", "small_tender_system_event_data.csv")
    build_faiss_index_for_events(csv_path)

@app.route("/")
def index():
    """Page 1: Extended input fields for the user."""
    return render_template("index.html")

@app.route("/generate_project", methods=["POST"])
def generate_project():
    """
    1) Gather form data
    2) Search events
    3) Dummy GPT-4 call for event AI insights
    4) Show compareEvents.html
    """
    form_data = request.form.to_dict()
    commodity = form_data.get("commodity", "").strip()

    matched_events = faiss_search_events(commodity, top_k=100)
    # Attach dummy GPT-4 event data
    for evt in matched_events:
        evt["ai_data"] = dummy_gpt4_event_insights(evt)

    # Sort by AI score descending
    matched_events.sort(key=lambda x: x["ai_data"]["score"], reverse=True)

    # Dummy global insights
    global_insights = {
        "trends": ["Global demand for sustainable solutions."],
        "risks": ["Potential raw material shortages."],
        "optimizations": ["Leverage multi-year contracts for price stability."]
    }

    return render_template(
        "compareEvents.html",
        events=matched_events,
        global_insights=global_insights,
        form_data=form_data
    )

@app.route("/event_details/<event_id>")
def event_details(event_id):
    """Page 3: Detailed event page after Add button in compareEvents."""
    event = next((e for e in EVENTS_DATA if e.get("EventID") == event_id), None)
    section_a = "Page 3a: Historical performance..."
    section_b = "Page 3b: Cost analysis / ROI..."
    section_c = "Page 3c: Additional disclaimers..."
    return render_template(
        "event_details.html",
        event=event,
        section_a=section_a,
        section_b=section_b,
        section_c=section_c
    )

@app.route("/quotation_ai/<event_id>")
def quotation_ai(event_id):
    """Page 4: Quotation AI => leads to Compare Quotes."""
    return render_template("quotation_ai.html", event_id=event_id)

@app.route("/compare_quotes/<event_id>")
def compare_quotes(event_id):
    """
    Page 5: Supplier quotes with dummy GPT-4. 
    No raw data is shown, as requested.
    """
    # We skip showing raw data
    # Return dummy GPT-4 supplier insights
    supplier_insights_list = dummy_gpt4_supplier_insights()
    # Sort by dummy score
    supplier_insights_list.sort(key=lambda x: x["score"], reverse=True)

    return render_template(
        "compare_quotes.html",
        event_id=event_id,
        supplier_insights_list=supplier_insights_list,
        supplier_text=""  # Blank or remove as user doesn't want raw data
    )

@app.route("/award", methods=["POST"])
def award():
    """Final awarding page."""
    selected_suppliers = request.form.getlist("selected_suppliers")
    return render_template("award_result.html", awarded_suppliers=selected_suppliers)

if __name__ == "__main__":
    # If you want real GPT-4 calls, uncomment and set your API key
    # openai.api_key = "YOUR_OPENAI_API_KEY"
    app.run(debug=True)
