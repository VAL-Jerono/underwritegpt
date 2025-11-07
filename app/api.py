# api.py
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd

app = FastAPI()

# Load your actual models
df = pd.read_csv('data/processed/train_data_with_summaries.csv')
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index('models/faiss_index.bin')

@app.post("/analyze")
def analyze_application(query: str):
    # Your existing RAG logic
    features = extract_features(query)
    risk = calculate_risk(features)
    similar_cases = search_faiss(query)
    decision = make_decision(risk, similar_cases)
    
    return {
        "decision": decision,
        "risk": risk,
        "cases": similar_cases[:10],
        "features": features
    }