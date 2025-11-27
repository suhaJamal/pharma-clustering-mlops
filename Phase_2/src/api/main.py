"""
FastAPI Application
Main API with prediction endpoints
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.predictor import ModelPredictor
from src.api.schemas import (
    PredictionRequest, 
    PredictionResponse, 
    HealthResponse,
    BatchPredictionRequest,
    BatchPredictionResponse
)

# Initialize FastAPI app
app = FastAPI(
    title="Pharmaceutical Market Segmentation API",
    description="Predicts market clusters for countries based on pharmaceutical spending patterns",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor (loads model once)
try:
    predictor = ModelPredictor(model_dir="models/v1.0")
except Exception as e:
    print(f"⚠️ Failed to load model: {e}")
    predictor = None


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Pharmaceutical Market Segmentation API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health and model status"""
    model_loaded = predictor is not None
    model_version = predictor.metadata['model_version'] if model_loaded else None
    
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        model_version=model_version
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Predict market cluster for a country
    
    Takes pharmaceutical spending features and returns:
    - Cluster assignment (0, 1, or 2)
    - Cluster name
    - Strategic recommendation
    - Confidence score
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Extract features from request
        features = {
            'PC_HEALTHXP_growth': request.PC_HEALTHXP_growth,
            'PC_GDP_growth': request.PC_GDP_growth,
            'USD_CAP_growth': request.USD_CAP_growth,
            'PC_HEALTHXP_avg': request.PC_HEALTHXP_avg,
            'PC_GDP_avg': request.PC_GDP_avg,
            'USD_CAP_avg': request.USD_CAP_avg,
            'PC_HEALTHXP_volatility': request.PC_HEALTHXP_volatility,
            'PC_GDP_volatility': request.PC_GDP_volatility,
            'USD_CAP_volatility': request.USD_CAP_volatility
        }
        
        # Make prediction
        result = predictor.predict(features, request.country)

        return PredictionResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict market clusters for multiple countries
    
    More efficient than calling /predict multiple times.
    Processes all countries in a single request.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    import time
    start_time = time.time()
    
    try:
        predictions = []
        
        # Process each country
        for country_data in request.countries:
            features = {
                'PC_HEALTHXP_growth': country_data.PC_HEALTHXP_growth,
                'PC_GDP_growth': country_data.PC_GDP_growth,
                'USD_CAP_growth': country_data.USD_CAP_growth,
                'PC_HEALTHXP_avg': country_data.PC_HEALTHXP_avg,
                'PC_GDP_avg': country_data.PC_GDP_avg,
                'USD_CAP_avg': country_data.USD_CAP_avg,
                'PC_HEALTHXP_volatility': country_data.PC_HEALTHXP_volatility,
                'PC_GDP_volatility': country_data.PC_GDP_volatility,
                'USD_CAP_volatility': country_data.USD_CAP_volatility
            }
            
            result = predictor.predict(features, country_data.country)
            predictions.append(PredictionResponse(**result))
        
        processing_time = time.time() - start_time
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_countries=len(predictions),
            processing_time_seconds=round(processing_time, 3)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
    
@app.get("/model/info", tags=["Model"])
async def model_info():
    """Get model metadata and performance metrics"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return predictor.get_model_info()