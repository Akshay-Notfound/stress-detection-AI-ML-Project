import os
import sys
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import numpy as np
from pydantic import BaseModel
import uvicorn

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import your existing modules
try:
    from src.preprocessing import preprocess_data
    from src.features import extract_features
    from src.models import load_model
except ImportError:
    # Fallback if imports fail
    def preprocess_data(df):
        return df
    
    def extract_features(df):
        return df
    
    def load_model(model_name='xgboost'):
        return None

app = FastAPI(title="Stress Detection API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StressPredictionResponse(BaseModel):
    stress_level: str
    confidence: float
    features_used: list

@app.get("/")
async def root():
    return {"message": "Stress Detection API", "status": "online"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=StressPredictionResponse)
async def predict_stress(file: UploadFile = File(...)):
    try:
        # Read the uploaded CSV file
        contents = await file.read()
        df = pd.read_csv(pd.compat.StringIO(contents.decode('utf-8')))
        
        # Preprocess the data
        processed_data = preprocess_data(df)
        
        # Extract features
        features = extract_features(processed_data)
        
        # Load model and make prediction (placeholder)
        model = load_model()
        
        # Placeholder prediction logic
        # In a real implementation, you would use your trained model
        stress_level = "Moderate"  # Placeholder
        confidence = 0.85  # Placeholder
        
        return StressPredictionResponse(
            stress_level=stress_level,
            confidence=confidence,
            features_used=list(features.columns) if hasattr(features, 'columns') else []
        )
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)