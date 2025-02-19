import os
import csv
import json
import numpy as np
import faiss
from flask import Flask, render_template, request, redirect, url_for
from sentence_transformers import SentenceTransformer

# Import openai for GPT-4 usage
import openai

# Set your OpenAI API key environment variable or do openai.api_key = "YOUR_API_KEY"
# e.g. openai.api_key = os.getenv("OPENAI_API_KEY")

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
        print("✅ Model downloaded at:", MODEL_PATH)
    else:
        print("✅ Model already present at:", MODEL_PATH)

download_model_if_needed()
embedding_model = SentenceTransformer(MODEL_PATH)
EMBEDDING_DIM = embedding_model.get_sentence_embedding_dimension()

app = Flask(__name__)

# --------------------------------------------------------------------
# GLOBALS for Events
# --------------------------------------------------------------------
EVENT_INDEX = None
EVENT_EMBEDDINGS = None
EVENTS_DATA = []

# --------------------------------------------------------------------
# LOAD EVENT CSV => FAISS
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

    texts = [" ".join([row.get("Title",""), row.get("Description",""), row.get("Commodity","")]) for row in events]
    embeddings = embedding_model.encode(texts, convert_to_numpy=True)

    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(embeddings)

    EVENT_INDEX = index
    EVENT_EMBEDDINGS = embeddings
    EVENTS_DATA = events

def faiss_search_events(commodity, top_k=50):
    """Search the events by commodity using FAISS."""
    if EVENT_INDEX is None:
        print("FAISS index not initialized.")
        return []
    commodity_emb = embedding_model.encode([commodity], convert_to_numpy=True)
    distances, indices = EVENT_INDEX.search(commodity_emb, top_k)
    return [EVENTS_DATA[idx] for idx in indices[0] if idx < len(EVENTS_DATA)]

# --------------------------------------------------------------------
# LOAD SUPPLIER QUOTATIONS
# --------------------------------------------------------------------
def load_supplier_quotations(txt_path):
    """Reads the entire supplier_quotations.txt and returns it as a string for GPT-4."""
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print("Error loading supplier quotations:", e)
        return ""

# --------------------------------------------------------------------
# Batching Logic for GPT-4 (Events)
# --------------------------------------------------------------------
def chunk_data_for_gpt4(data_list, chunk_size=5):
    for i in range(0, len(data_list), chunk_size):
        yield data_list[i:i+chunk_size]

def call_gpt4_for_events_batched(events_list, form_data):
    """
    Sends events_list in small chunks to GPT-4, merges results.
    """
    final_results = []
    for chunk in chunk_data_for_gpt4(events_list, chunk_size=5):
        chunk_json = json.dumps(chunk, indent=2)

        prompt = f"""
        You are an AI procurement assistant. Based on these events (in JSON) and buyer form data, provide:
        - score
        - reason
        - explanation
        - match_score
        - region
        - risks
        - ai_insights (trends, optimizations)
        Return JSON for each event in 'chunk'.
        Events (JSON): {chunk_json}
        Buyer Input: {form_data}
        """

        response = openai.ChatCompletion.create(
            model='gpt-4',
            messages=[
              {'role':'system', 'content':'You are a procurement AI assistant.'},
              {'role':'user', 'content': prompt}
            ],
            max_tokens=2000,
            temperature=0
        )
        chunk_result = json.loads(response['choices'][0]['message']['content'])
        final_results.extend(chunk_result)
    return final_results

# --------------------------------------------------------------------
# GPT-4 for Supplier Quotations
# --------------------------------------------------------------------
def call_gpt4_for_suppliers(events_form_data, supplier_text):
    """
    Sends supplier_text to GPT-4 for analyzing quotations.
    """
    prompt = f"""
    You are an AI procurement assistant. Based on the following supplier quotations and buyer requirements, provide:
    1. Ranked supplier quotes (price, delivery time, quality, additional terms).
    2. Explanations for each ranking, highlighting trade-offs.
    3. AI insights including trends, risks, optimizations.

    Buyer Requirements: {events_form_data}
    Supplier Quotations (raw text): {supplier_text}

    Respond in JSON format, e.g.:
    [
      {{
        "rank": 1,
        "supplier_name": "...",
        "price_per_unit": "...",
        "delivery_date": "...",
        "additional_terms": "...",
        "score": 95,
        "explanation": "...",
        "ai_suggested_percentage": 30
      }},
      ...
    ]
    """

    response = openai.ChatCompletion.create(
        model='gpt-4',
        messages=[
            {'role':'system','content':'You are a helpful procurement AI assistant.'},
            {'role':'user','content':prompt}
        ],
        max_tokens=2500,
        temperature=0
    )
    return json.loads(response['choices'][0]['message']['content'])

# --------------------------------------------------------------------
# Flask Setup
# --------------------------------------------------------------------
app = Flask(__name__)

@app.before_first_request
def init_app():
    csv_path = os.path.join("data", "small_tender_system_event_data.csv")
    build_faiss_index_for_events(csv_path)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate_project", methods=["POST"])
def generate_project():
    """
    1) Gather form data
    2) Vector search events
    3) call_gpt4_for_events_batched for large events
    4) Return compareEvents with AI insights
    """
    form_data = request.form.to_dict()
    commodity = form_data.get("commodity","").strip()

    matched_events = faiss_search_events(commodity, top_k=100)

    # BATCH GPT-4 for events
    # Convert matched_events to a minimal JSON for GPT-4 to read
    # For example, pass Title, Description, Commodity, EventID
    # This is an example. Adjust as needed for your real data
    events_for_gpt = []
    for evt in matched_events:
        events_for_gpt.append({
            "EventID": evt.get("EventID"),
            "Title": evt.get("Title"),
            "Description": evt.get("Description"),
            "Commodity": evt.get("Commodity"),
            "Region": evt.get("Region", "Unknown")
        })

    if len(events_for_gpt) > 0:
        gpt4_results = call_gpt4_for_events_batched(events_for_gpt, form_data)
        # Attach to matched_events
        for i, e in enumerate(matched_events):
            e["ai_data"] = gpt4_results[i]
    else:
        # If no events found or no GPT-4
        pass

    # Sort by AI score if present
    matched_events.sort(key=lambda x: x["ai_data"]["score"] if x.get("ai_data") else 0, reverse=True)

    # Dummy global insights
    global_insights = {
        "trends": ["Global demand for sustainable solutions."],
        "risks": ["Potential raw material shortages."],
        "optimizations": ["Leverage multi-year contracts for price stability."]
    }

    return render_template("compareEvents.html",
                           events=matched_events,
                           global_insights=global_insights,
                           form_data=form_data)

@app.route("/event_details/<event_id>")
def event_details(event_id):
    # Show event detail
    event = next((e for e in EVENTS_DATA if e.get("EventID") == event_id), None)
    section_a = "Historical data..."
    section_b = "Cost analysis / ROI..."
    section_c = "Additional disclaimers..."
    return render_template("event_details.html",
                           event=event,
                           section_a=section_a,
                           section_b=section_b,
                           section_c=section_c)

@app.route("/quotation_ai/<event_id>")
def quotation_ai(event_id):
    # Page 4
    return render_template("quotation_ai.html", event_id=event_id)

@app.route("/compare_quotes/<event_id>")
def compare_quotes(event_id):
    """
    1) Load supplier_quotations.txt
    2) call_gpt4_for_suppliers => JSON
    3) Display in compare_quotes
    """
    txt_path = os.path.join("data", "supplier_quotations.txt")
    supplier_text = ""
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            supplier_text = f.read()
    except Exception as e:
        print("Error loading supplier quotations:", e)

    # We can pass the event ID or form_data from the previous steps
    # For demonstration, we'll just pass the event ID as "buyer requirements"
    event_data = f"EventID: {event_id}"

    # GPT-4 call
    supplier_insights_list = call_gpt4_for_suppliers(event_data, supplier_text)
    # We expect a list of supplier quotes objects
    # e.g. [ { 'rank':1, 'supplier_name':'...', ...}, {...} ]

    # Sort by score
    supplier_insights_list.sort(key=lambda x: x["score"], reverse=True)

    return render_template("compare_quotes.html",
                           event_id=event_id,
                           supplier_insights_list=supplier_insights_list,
                           supplier_text=supplier_text)

@app.route("/award", methods=["POST"])
def award():
    """Collect awarded suppliers and show them."""
    selected_suppliers = request.form.getlist("selected_suppliers")
    return render_template("award_result.html", awarded_suppliers=selected_suppliers)

if __name__ == "__main__":
    # Provide your OpenAI API Key
    # openai.api_key = "YOUR_OPENAI_API_KEY"
    app.run(debug=True)
