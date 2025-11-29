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

from src.monitoring.logger import PredictionLogger
import time

from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from src.monitoring.drift_detector import DriftDetector

# Initialize FastAPI app
app = FastAPI(
    title="Pharmaceutical Market Segmentation API",
    description="Predicts market clusters for countries based on pharmaceutical spending patterns",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="src/api/static"), name="static")

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
    predictor = ModelPredictor(model_dir="models/v1.2.0")
except Exception as e:
    print(f"⚠️ Failed to load model: {e}")
    predictor = None

prediction_logger = PredictionLogger()
# Initialize drift detector
try:
    drift_detector = DriftDetector(model_dir="models/v1.2.0")
    print("✅ Drift detector initialized")
except Exception as e:
    print(f"⚠️ Drift detector not available: {e}")
    drift_detector = None

@app.get("/", response_class=HTMLResponse, tags=["Root"])
async def root():
    """Serve the web UI"""
    with open("src/api/static/index.html", "r", encoding="utf-8") as f:
        return f.read()

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

    start_time = time.time()

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
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Log prediction
        prediction_logger.log_prediction(
            country=result['country'],
            cluster=result['cluster'],
            confidence=result['confidence'],
            model_version=result['model_version'],
            response_time=response_time
        )
        result['processing_time_seconds'] = round(response_time, 4) 
        
        # Track for drift detection
        if drift_detector:
            drift_detector.add_prediction(features)

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

        # Log batch predictions
        prediction_logger.log_batch_prediction(
            predictions=[p.model_dump() for p in predictions],
            total_response_time=processing_time
        )

        return BatchPredictionResponse(
            predictions=predictions,
            total_countries=len(predictions),
            processing_time_seconds=round(processing_time, 3)
        )
        
    except Exception as e:
        import traceback
        print("BATCH ERROR:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
    
@app.get("/model/info", tags=["Model"])
async def model_info():
    """Get model metadata and performance metrics"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return predictor.get_model_info()

@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """
    Get prediction metrics and statistics
    
    Returns:
    - Total predictions made
    - Predictions per cluster
    - Average response time
    - API uptime
    """
    return prediction_logger.get_metrics()

@app.get("/drift/status", tags=["Monitoring"])
async def drift_status():
    """
    Check for data drift
    
    Compares recent production data to training data distribution.
    Returns drift status for each feature.
    """
    if drift_detector is None:
        raise HTTPException(
            status_code=503, 
            detail="Drift detection not available. Retrain model to generate training statistics."
        )
    
    return drift_detector.detect_drift()

@app.post("/generate-test-data", tags=["Testing"])
async def generate_test_data(num_predictions: int = 30):
    """
    Generate random test predictions for drift detection testing
    
    Creates predictions for fictional countries with randomized features
    """
    import random
    
    if drift_detector is None:
        raise HTTPException(status_code=503, detail="Drift detector not available")
    
    # Fictional country names
    countries = [
        "Atlantis", "Wakanda", "Genovia", "Latveria", "Sokovia", 
        "Themyscira", "Krypton", "Asgard", "Pandora", "Narnia",
        "Westeros", "Essos", "Middle-earth", "Gondor", "Rohan",
        "Tatooine", "Coruscant", "Naboo", "Alderaan", "Hoth",
        "Panem", "District-12", "Gilead", "Oceania", "Airstrip-One",
        "Wonderland", "Oz", "Neverland", "Hundred-Acre", "Camelot",
        "Fantasia", "Narnia-2", "Xanadu", "Shangri-La", "El-Dorado"
    ]
    
    predictions_made = 0
    
    for i in range(min(num_predictions, len(countries))):
        # Generate random features (slightly varied from training ranges)
        features = {
            'PC_HEALTHXP_growth': random.uniform(-8, 8),
            'PC_GDP_growth': random.uniform(-3, 8),
            'USD_CAP_growth': random.uniform(-5, 15),
            'PC_HEALTHXP_avg': random.uniform(6, 18),
            'PC_GDP_avg': random.uniform(0.5, 4),
            'USD_CAP_avg': random.uniform(150, 1200),
            'PC_HEALTHXP_volatility': random.uniform(0.1, 1.5),
            'PC_GDP_volatility': random.uniform(0.02, 0.8),
            'USD_CAP_volatility': random.uniform(10, 150)
        }
        
        # Make prediction
        result = predictor.predict(features, countries[i])
        
        # Track for drift detection
        if drift_detector:
            drift_detector.add_prediction(features)
        
        predictions_made += 1
    
    return {
        "message": f"Generated {predictions_made} test predictions",
        "predictions_made": predictions_made,
        "drift_status_ready": predictions_made >= 30,
        "check_drift_at": "/drift/status"
    }