import os
import csv
import faiss
import numpy as np
from flask import Flask, render_template, request, jsonify
# import openai  # Uncomment for real GPT-4 integration.
from transformers import pipeline
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# -----------------------------------------------------------
# Configuration & Model Initialization
# -----------------------------------------------------------
# For real GPT-4 calls, set your OpenAI API key in your environment.
# openai.api_key = os.environ.get("OPENAI_API_KEY")

# Initialize a summarization pipeline (using Facebook's BART model)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Define the model repository directory and model name.
model_repo_dir = "model_repo"
model_name = "all-MiniLM-L6-v2"

# Ensure the model repository directory exists
os.makedirs(model_repo_dir, exist_ok=True)

# Check if the model is already present in the model_repo directory.
# The SentenceTransformer downloads its model into a subfolder named after the model.
model_local_path = os.path.join(model_repo_dir, model_name)
if os.path.exists(model_local_path):
    print(f"Found {model_name} in {model_repo_dir}. Loading from cache.")
else:
    print(f"{model_name} not found in {model_repo_dir}. Downloading and caching now...")

# Initialize the SentenceTransformer model for computing embeddings using the cache folder.
print("Loading SentenceTransformer model...")
embedding_model = SentenceTransformer(model_name, cache_folder=model_repo_dir)
print("Model loaded!")

# -----------------------------------------------------------
# Load Historical Event Data (for similar event recommendation)
# -----------------------------------------------------------
historical_events = []
# Load Historical Event Data from CSV (inside data folder)
csv_filename = os.path.join("data", "historical_events.csv")
try:
    with open(csv_filename, newline='', encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            historical_events.append(row)
except Exception as e:
    print(f"Error loading {csv_filename}: {e}")

def extract_event_features(event):
    """
    Combine key fields from an event record into a text string and summarize it.
    Fields include event_id, commodity, item, line_item, line_number, supplier details,
    bid_amount, bid_status, submission_timestamp, and evaluation_score.
    """
    text = (
        f"Tender ID: {event.get('event_id', '')}. "
        f"Commodity: {event.get('commodity', '')}. "
        f"Item: {event.get('item', '')}. "
        f"Line Item: {event.get('line_item', '')}, Line Number: {event.get('line_number', '')}. "
        f"Supplier: {event.get('supplier_name', '')} (ID: {event.get('supplier_id', '')}). "
        f"Bid Amount: {event.get('bid_amount', '')}. "
        f"Status: {event.get('bid_status', '')}. "
        f"Submitted: {event.get('submission_timestamp', '')}. "
        f"Score: {event.get('evaluation_score', '')}."
    )
    try:
        summary = summarizer(text, max_length=60, min_length=20, do_sample=False)[0]['summary_text']
    except Exception as e:
        summary = text  # Fallback to raw text if summarization fails
        print(f"Summarization error: {e}")
    return summary

event_features = []
for idx, event in enumerate(historical_events):
    features = extract_event_features(event)
    event_features.append({
        "id": idx,
        "features": features,
        "raw_event": event
    })

# Compute embeddings for each event summary using the SentenceTransformer model.
feature_texts = [item["features"] for item in event_features]
event_embeddings = embedding_model.encode(feature_texts, convert_to_numpy=True).astype("float32")

# Build a FAISS index using L2 distance.
embedding_dim = event_embeddings.shape[1]
faiss_index_events = faiss.IndexFlatL2(embedding_dim)
faiss_index_events.add(event_embeddings)
print(f"FAISS index for events built with {faiss_index_events.ntotal} records.")

# -----------------------------------------------------------
# Load Supplier Quotations (for quote analysis; unchanged)
# -----------------------------------------------------------
supplier_quotations = ""
# Load Supplier Quotations from the data folder
quotations_file = os.path.join("data", "supplier_quotations.txt")
try:
    with open(quotations_file, "r", encoding="utf-8") as f:
        supplier_quotations = f.read()
except Exception as e:
    print(f"Error loading {quotations_file}: {e}")

# ----------------------------
# Routes for UI Pages
# ----------------------------
@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/similar_event_recommendation")
def similar_event_recommendation_page():
    return render_template("similar_event_recommendation.html")

@app.route("/analyze_quotes")
def analyze_quotes_page():
    return render_template("analyze_quotes.html", supplier_quotations=supplier_quotations)

# ----------------------------
# API Endpoint: Similar Event Recommendation
# ----------------------------
@app.route("/api/similar_event", methods=["POST"])
def similar_event():
    data = request.json
    buyer_input = data.get("buyer_input", "")
    if not buyer_input:
        return jsonify({"error": "Missing buyer input"}), 400

    # Summarize the buyer's current event input.
    try:
        buyer_summary = summarizer(buyer_input, max_length=60, min_length=20, do_sample=False)[0]['summary_text']
    except Exception as e:
        buyer_summary = buyer_input
        print(f"Summarization error for buyer input: {e}")
    
    # Compute the embedding for the buyer's event summary.
    buyer_embedding = embedding_model.encode([buyer_summary], convert_to_numpy=True).astype("float32")
    top_k = 5
    distances, indices = faiss_index_events.search(buyer_embedding, top_k)
    filtered_events = [event_features[i] for i in indices[0] if i < len(event_features)]
    
    # Construct a GPT-4 prompt for similar event recommendation.
    # (The following GPT-4 API call is commented out and replaced with mocked output for testing.)
    """
    prompt = f'''
    You are an AI procurement event recommender. Based on the following historical event data and buyer input, provide:
    1. Ranked event recommendations that are most similar to the buyer input.
    2. For each recommended event, include a match percentage and a brief explanation for the ranking.
    3. Identify any key risks or opportunities associated with the recommended events.
    4. Provide overall AI insights including trends and potential optimizations.

    Historical Event Data:
    {filtered_events}

    Buyer Input:
    {buyer_input}

    Respond in JSON format:
    {{
      "ranked_events": [
          {{
              "event_id": "TND-2025-001",
              "match_percentage": 95,
              "ranking_reason": "Strong similarity in commodity and bid patterns.",
              "risks": ["Potential supply delays", "High bid variance"],
              "ai_insights": {{
                  "trends": ["Increasing bid amounts over time"],
                  "optimizations": ["Consolidate orders for cost efficiency"]
              }}
          }},
          ...
      ],
      "ai_insights": {{
          "trends": [...],
          "risks": [...],
          "optimizations": [...]
      }}
    }}
    '''
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI procurement event recommender."},
                {"role": "user", "content": prompt}
            ]
        )
        output = eval(response['choices'][0]['message']['content'])
    except Exception as e:
        print(f"OpenAI GPT-4 Error: {e}")
        return jsonify({"error": "Failed to get AI recommendations."}), 500
    """
    # Mocked GPT-4 output for similar event recommendation:
    output = {
        "ranked_events": [
            {
                "event_id": "TND-2025-001",
                "match_percentage": 95,
                "ranking_reason": "High similarity in commodity (Server) and bid details.",
                "risks": ["Slightly high bid amount may indicate budget constraints."],
                "ai_insights": {
                    "trends": ["Bid amounts have been increasing steadily."],
                    "optimizations": ["Consider negotiating volume discounts."]
                }
            },
            {
                "event_id": "TND-2025-002",
                "match_percentage": 92,
                "ranking_reason": "Similar commodity (Networking) and competitive bid parameters.",
                "risks": ["Lower evaluation scores may indicate quality concerns."],
                "ai_insights": {
                    "trends": ["Networking equipment shows seasonal demand fluctuations."],
                    "optimizations": ["Review past supplier performance before awarding."]
                }
            }
        ],
        "ai_insights": {
            "trends": ["Overall, historical events indicate a trend towards increasing bid amounts in IT infrastructure."],
            "risks": ["Market volatility may affect supplier performance."],
            "optimizations": ["Leverage historical trends to negotiate better terms."]
        }
    }
    return jsonify(output)

# ----------------------------
# API Endpoint: Analyze Quotes (Quotation Analysis)
# ----------------------------
@app.route("/api/analyze_quotes", methods=["POST"])
def analyze_quotes():
    buyer_requirements = request.json.get("buyer_requirements", "")
    if not buyer_requirements:
        return jsonify({"error": "Missing buyer requirements"}), 400

    # Construct a GPT-4 prompt for analyzing supplier quotations.
    # (The GPT-4 API code is commented out and replaced with mocked output.)
    """
    prompt = f'''
    You are an AI procurement assistant. Based on the following supplier quotations and buyer requirements, provide:
    1. Ranked supplier quotes based on price, delivery time, quality, and additional terms.
    2. Explanations for the rankings, highlighting trade-offs.
    3. AI insights including trends, risks, optimizations, and sustainability.

    Buyer Requirements:
    {buyer_requirements}

    Supplier Quotations:
    {supplier_quotations}

    Respond in JSON format:
    {{
      "ranked_quotes": [ ... ],
      "ai_insights": {{ ... }}
    }}
    '''
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a procurement AI assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        output = eval(response['choices'][0]['message']['content'])
    except Exception as e:
        print(f"OpenAI GPT-4 Error: {e}")
        return jsonify({"error": "Failed to get AI insights."}), 500
    """
    # Mocked output for analyze quotes:
    quotes = [
        {
            "supplier_name": "TechWorld",
            "price_per_unit": 1200,
            "total_cost": 120000,
            "delivery_date": "2025-01-25",
            "additional_terms": "Free shipping for orders above $10,000.",
            "score": 96,
            "explanation": "Ranked high due to competitive pricing, fast delivery, and excellent terms."
        },
        {
            "supplier_name": "DataMax",
            "price_per_unit": 1150,
            "total_cost": 138000,
            "delivery_date": "2025-01-24",
            "additional_terms": "5% discount for orders above 150 units.",
            "score": 92,
            "explanation": "Strong ranking due to discounted pricing and early delivery."
        }
    ]
    analysis = {
        "ranked_quotes": quotes,
        "ai_insights": {
            "trends": ["Bid stability is noted in the current quarter.", "Early deliveries are a competitive advantage."],
            "risks": ["Potential delays due to global chip shortages."],
            "optimizations": ["Negotiate bulk discounts.", "Consider splitting orders to mitigate risk."],
            "sustainability": ["Eco-friendly practices could be further incentivized."]
        }
    }
    return jsonify(analysis)

if __name__ == "__main__":
    app.run(debug=False)
