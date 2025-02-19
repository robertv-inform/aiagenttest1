import os
import csv
import json
import numpy as np
import faiss
from flask import Flask, render_template, request, redirect, url_for
from sentence_transformers import SentenceTransformer

import openai  # GPT-4 usage

# --------------------------------------------------------------------
# Provide your OpenAI API Key here or via env var
# e.g. openai.api_key = "YOUR_OPENAI_API_KEY"
# --------------------------------------------------------------------

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

# Load SentenceTransformer for FAISS embeddings
embedding_model = SentenceTransformer(MODEL_PATH)
EMBEDDING_DIM = embedding_model.get_sentence_embedding_dimension()

app = Flask(__name__)

# --------------------------------------------------------------------
# GLOBALS for events
# --------------------------------------------------------------------
EVENT_INDEX = None
EVENTS_DATA = []

# --------------------------------------------------------------------
# Build FAISS index from CSV
# --------------------------------------------------------------------
def build_faiss_index_for_events(csv_path):
    """
    Loads entire event CSV, builds embeddings, indexes them in FAISS.
    """
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

    # Convert each row to a single string for embeddings
    texts = [
        " ".join([r.get("Title",""), r.get("Description",""), r.get("Commodity","")])
        for r in events
    ]
    embeddings = embedding_model.encode(texts, convert_to_numpy=True)

    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(embeddings)

    EVENT_INDEX = index
    EVENTS_DATA = events

def faiss_search_events(commodity, top_k=50):
    """FAISS search by commodity."""
    if EVENT_INDEX is None:
        print("❌ FAISS not built.")
        return []
    query_emb = embedding_model.encode([commodity], convert_to_numpy=True)
    distances, indices = EVENT_INDEX.search(query_emb, top_k)
    matched = [EVENTS_DATA[idx] for idx in indices[0] if idx < len(EVENTS_DATA)]
    return matched

# --------------------------------------------------------------------
# GPT-4 Batching for full events
# --------------------------------------------------------------------
def chunk_data_for_gpt4(data_list, chunk_size=5):
    """
    We ensure each event row is included in its entirety in a chunk
    so we never pass partial events to GPT-4.
    """
    for i in range(0, len(data_list), chunk_size):
        yield data_list[i:i+chunk_size]

def call_gpt4_for_events_batched(events_list, user_form_data):
    """
    Splits events_list into small chunks, calls GPT-4 for each chunk,
    merges results into one big list.
    Each chunk uses complete event data, no partial rows.
    """
    final_event_ai = []
    for chunk in chunk_data_for_gpt4(events_list, chunk_size=5):
        # Convert chunk to JSON
        chunk_json = json.dumps(chunk, indent=2)

        prompt = f"""
        You are an AI procurement assistant. For these events (in JSON) plus buyer input, return a JSON array
        of the same length. For each event, fields needed are:
        - score
        - reason
        - explanation
        - match_score
        - region
        - risks
        - ai_insights {{ trends, optimizations }}

        Events chunk (JSON): {chunk_json}
        Buyer Form Data: {user_form_data}

        Respond with a JSON array, e.g.
        [
          {{
            "score": 95,
            "reason": "...",
            "explanation": "...",
            "match_score": 0.92,
            "region": "...",
            "risks": "...",
            "ai_insights": {{
               "trends": [...],
               "optimizations": [...]
            }}
          }},
          ...
        ]
        """

        response = openai.ChatCompletion.create(
            model='gpt-4',
            messages=[
                {'role':'system', 'content':'You are a helpful procurement AI assistant.'},
                {'role':'user', 'content': prompt}
            ],
            max_tokens=2000,
            temperature=0
        )
        chunk_results = json.loads(response['choices'][0]['message']['content'])
        final_event_ai.extend(chunk_results)

    return final_event_ai

def call_gpt4_for_global_insights(events_list, user_form_data):
    """
    Calls GPT-4 once to produce aggregated insights (trends, risks, optimizations)
    for the entire set of events. No partial row usage, just a separate aggregator.
    """
    events_json = json.dumps(events_list, indent=2)
    prompt = f"""
    You are an AI procurement assistant. We have these events (in JSON) plus user form data.
    Provide global-level AI insights: 
    - 'trends': [ ... ],
    - 'risks': [ ... ],
    - 'optimizations': [ ... ]

    Events (JSON): {events_json}
    Buyer Form Data: {user_form_data}

    Respond in JSON, e.g.
    {{
      "trends": [...],
      "risks": [...],
      "optimizations": [...]
    }}
    """

    response = openai.ChatCompletion.create(
        model='gpt-4',
        messages=[
            {'role':'system', 'content':'You are a procurement AI aggregator.'},
            {'role':'user', 'content':prompt}
        ],
        max_tokens=1200,
        temperature=0
    )
    return json.loads(response['choices'][0]['message']['content'])

# --------------------------------------------------------------------
# GPT-4 for Supplier Quotations
# --------------------------------------------------------------------
def call_gpt4_for_suppliers(buyer_req, supplier_text):
    """
    Single GPT-4 call for supplier quotes. No partial row usage, we pass the entire text.
    """
    prompt = f"""
    You are an AI procurement assistant. Based on these supplier quotations (raw text) and buyer requirements:
    {buyer_req}

    {supplier_text}

    Provide a JSON array of quotes, each with:
    - rank
    - supplier_name
    - price_per_unit
    - delivery_date
    - additional_terms
    - score
    - explanation
    - ai_suggested_percentage
    """

    response = openai.ChatCompletion.create(
        model='gpt-4',
        messages=[
            {'role':'system','content':'You are a procurement AI assistant for quotations.'},
            {'role':'user','content':prompt}
        ],
        max_tokens=2000,
        temperature=0
    )
    return json.loads(response['choices'][0]['message']['content'])

# --------------------------------------------------------------------
# Supplier Quotation Loader
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
app = Flask(__name__)

@app.before_first_request
def init_app():
    csv_path = os.path.join("data", "small_tender_system_event_data.csv")
    build_faiss_index_for_events(csv_path)
    # openai.api_key = "YOUR_OPENAI_API_KEY"  # Or set it externally

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate_project", methods=["POST"])
def generate_project():
    """
    1) Gather form data
    2) FAISS filter events
    3) chunk-based GPT-4 for each event => attach
    4) aggregator GPT-4 => global insights
    5) show compareEvents
    """
    ## If to use both event data and commodity, then uncomment below
    # event_title = form_data.get("event_title", "").strip()
    # commodity   = form_data.get("commodity", "").strip()

    # # Combine them into a single string query for FAISS
    # # e.g. "Smart Traffic System Electronics"
    # combined_query = f"{event_title} {commodity}".strip()

    # # Then FAISS filter by combined_query
    # matched_events = faiss_search_events(combined_query, top_k=100)
    
    
    form_data = request.form.to_dict()
    commodity = form_data.get("commodity","").strip()

    matched_events = faiss_search_events(commodity, top_k=100)

    # 1) Prepare minimal JSON for chunk-based GPT-4
    events_for_gpt = []
    for e in matched_events:
        events_for_gpt.append({
            "EventID": e.get("EventID"),
            "Title": e.get("Title"),
            "Description": e.get("Description"),
            "Commodity": e.get("Commodity"),
            "Region": e.get("Region", "Unknown"),
        })

    # 2) Call GPT-4 for each event in chunks
    if events_for_gpt:
        event_insights = call_gpt4_for_events_batched(events_for_gpt, form_data)
        for i, ev in enumerate(matched_events):
            ev["ai_data"] = event_insights[i]

    # 3) Sort events by AI score
    matched_events.sort(
        key=lambda x: x["ai_data"]["score"] if x.get("ai_data") else 0,
        reverse=True
    )

    # 4) aggregator GPT-4 => produce global insights
    global_insights = {"trends":[],"risks":[],"optimizations":[]}
    if events_for_gpt:
        # We pass the entire list to aggregator
        global_insights = call_gpt4_for_global_insights(events_for_gpt, form_data)

    return render_template("compareEvents.html",
                           events=matched_events,
                           global_insights=global_insights,
                           form_data=form_data)

@app.route("/event_details/<event_id>")
def event_details(event_id):
    """
    Page 3: Detailed event info after 'Add' button
    """
    event = next((x for x in EVENTS_DATA if x.get("EventID") == event_id), None)
    section_a = "Historical performance data..."
    section_b = "Cost analysis / ROI..."
    section_c = "Additional disclaimers..."
    return render_template("event_details.html",
                           event=event,
                           section_a=section_a,
                           section_b=section_b,
                           section_c=section_c)

@app.route("/quotation_ai/<event_id>")
def quotation_ai(event_id):
    """
    Page 4: Quotation AI => leads to Compare Quotes
    """
    return render_template("quotation_ai.html", event_id=event_id)

@app.route("/compare_quotes/<event_id>")
def compare_quotes(event_id):
    """
    Page 5: Supplier quotes with GPT-4
    """
    txt_path = os.path.join("data", "supplier_quotations.txt")
    supplier_txt = load_supplier_quotations(txt_path)

    # We'll pass the event ID as part of the buyer requirements
    buyer_req = f"EventID: {event_id}"

    # 1) GPT-4 call for suppliers
    quotes_data = call_gpt4_for_suppliers(buyer_req, supplier_txt)
    # 2) Sort by 'score'
    quotes_data.sort(key=lambda x: x["score"], reverse=True)

    return render_template("compare_quotes.html",
                           event_id=event_id,
                           supplier_insights_list=quotes_data,
                           supplier_text="")  # not showing raw data

@app.route("/award", methods=["POST"])
def award():
    """Final awarding page."""
    selected_suppliers = request.form.getlist("selected_suppliers")
    return render_template("award_result.html", awarded_suppliers=selected_suppliers)

if __name__ == "__main__":
    # e.g. openai.api_key = "YOUR_OPENAI_API_KEY"
    app.run(debug=True)
