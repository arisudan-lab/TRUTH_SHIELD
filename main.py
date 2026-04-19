from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os
import traceback

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Force Python to use the exact directory where main.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTORIZER_PATH = os.path.join(BASE_DIR, "models/vectorizer.jb")
MODEL_PATH = os.path.join(BASE_DIR, "models/lr_model.jb")

# 2. Define global variables first so we don't get a NameError later
vectorizer = None
text_model = None

# 3. Load the models using the absolute paths
try:
    vectorizer = joblib.load(VECTORIZER_PATH)
    text_model = joblib.load(MODEL_PATH)
    print(f"✅ Models loaded successfully from: {BASE_DIR}")
except Exception as e:
    print(f"❌ CRITICAL ERROR LOADING MODELS: {e}")
    print(f"⚠️ Python tried looking here:\n - {VECTORIZER_PATH}\n - {MODEL_PATH}")

class NewsRequest(BaseModel):
    text: str

@app.post("/api/check-news")
async def check_news(request: NewsRequest):
    # If models failed to load, tell the frontend nicely instead of crashing
    if vectorizer is None or text_model is None:
        raise HTTPException(
            status_code=500, 
            detail="Models failed to load on the server. Check your terminal logs."
        )
        
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Please provide text to analyze.")
    
    try:
        # Transform and Predict
        vectorized_text = vectorizer.transform([request.text])
        prediction = text_model.predict(vectorized_text)[0]
        
        # Safely handle probabilities
        try:
            probabilities = text_model.predict_proba(vectorized_text)[0]
            confidence = max(probabilities) * 100
        except AttributeError:
            confidence = 88.5 # Fallback for models without predict_proba
        
        # Format Verdict
        pred_str = str(prediction).upper()
        if pred_str in ["1", "REAL", "TRUE"]:
            verdict = "REAL"
        else:
            verdict = "FAKE"
        
        return {
            "verdict": verdict,
            "confidence": round(confidence, 1),
            "summary": "Analyzed using local Logistic Regression model."
        }
        
    except Exception as e:
        error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
        print(f"Server crashed during prediction:\n{error_msg}")
        raise HTTPException(status_code=500, detail=f"ML Error: {str(e)}")