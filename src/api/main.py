import os
import pandas as pd
import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from src.api.pydantic_models import CustomerPredictionInput, PredictionResponse

# Initialize the FastAPI Application
app = FastAPI(
    title="Credit Risk Proxy Modeling API Engine",
    description="Production-ready REST API gateway delivering real-time customer risk assessment predictions.",
    version="1.0.0"
)

# Connect FastAPI to your local MLflow tracking database
MODEL_NAME = "RandomForest_Champion"
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Global variable to hold our loaded champion model in system memory
model = None

@app.on_event("startup")
def load_champion_model():
    """
    Executes automatically when the API server boots up.
    Fetches the latest production-grade model version from the MLflow registry.
    """
    global model
    try:
        print(f"📡 Connecting to MLflow Registry to download model: '{MODEL_NAME}'...")
        # Pulls version 1 of our champion model dynamically from storage
        model_uri = f"models:/{MODEL_NAME}/1"
        model = mlflow.sklearn.load_model(model_uri)
        print("🏆 Champion Model loaded into API server memory successfully!")
    except Exception as e:
        print(f"❌ Critical Error loading model from registry: {str(e)}")
        # Fallback safeguard in case registry lookup drops or paths are misaligned
        model = None

@app.get("/")
def read_root():
    """
    Health check endpoint to ensure the API server is breathing.
    """
    return {
        "status": "online",
        "api_name": "Credit Risk Proxy Assessment Gateway",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_credit_risk(payload: CustomerPredictionInput):
    """
    Receives validated customer transaction data, maps it to a DataFrame,
    and returns high-risk probability matrix properties.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Machine Learning Model engine is currently offline or uninitialized.")
    
    try:
        # 1. Convert incoming Pydantic data into a single-row Pandas DataFrame matching model expectations
        input_data = pd.DataFrame([{
            "CustomerId": payload.CustomerId,
            "Amount": payload.Amount,
            "TransactionStartTime": payload.TransactionStartTime
        }])
        
        # 2. Run raw feature inference calculations through the Scikit-Learn Pipeline
        probabilities = model.predict_proba(input_data)[0]
        high_risk_prob = float(probabilities[1])
        
        # 3. Generate binary class target prediction (0 or 1)
        binary_prediction = int(model.predict(input_data)[0])
        
        # 4. Neatly package and return the validated JSON response back to the client app
        return PredictionResponse(
            customer_id=payload.CustomerId,
            risk_probability=high_risk_prob,
            is_high_risk=binary_prediction
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference execution failed: {str(e)}")