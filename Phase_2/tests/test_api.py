"""
API endpoint tests
"""
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint returns welcome message"""
    response = client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    assert data['message'] == "Pharmaceutical Market Segmentation API"
    assert data['version'] == "1.0.0"


def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == "healthy"
    assert data['model_loaded'] == True
    assert data['model_version'] == "1.0.0"


def test_predict_endpoint():
    """Test single prediction endpoint"""
    payload = {
        "country": "Germany",
        "PC_HEALTHXP_growth": -0.51,
        "PC_GDP_growth": 1.31,
        "USD_CAP_growth": 4.24,
        "PC_HEALTHXP_avg": 14.14,
        "PC_GDP_avg": 1.60,
        "USD_CAP_avg": 790.71,
        "PC_HEALTHXP_volatility": 0.27,
        "PC_GDP_volatility": 0.06,
        "USD_CAP_volatility": 102.02
    }
    
    response = client.post("/predict", json=payload)
    print(response.json()) 
    assert response.status_code == 200
    data = response.json()
    assert data['country'] == "Germany"
    assert data['cluster'] in [0, 1, 2]
    assert data['cluster_name'] in ["Declining Markets", "Stable Moderate Markets", "High-Value Pharma Markets"]


def test_batch_predict_endpoint():
    """Test batch prediction endpoint"""
    payload = {
        "countries": [
            {
                "country": "Germany",
                "PC_HEALTHXP_growth": -0.51,
                "PC_GDP_growth": 1.31,
                "USD_CAP_growth": 4.24,
                "PC_HEALTHXP_avg": 14.14,
                "PC_GDP_avg": 1.60,
                "USD_CAP_avg": 790.71,
                "PC_HEALTHXP_volatility": 0.27,
                "PC_GDP_volatility": 0.06,
                "USD_CAP_volatility": 102.02
            },
            {
                "country": "France",
                "PC_HEALTHXP_growth": -2.5,
                "PC_GDP_growth": 0.8,
                "USD_CAP_growth": 2.1,
                "PC_HEALTHXP_avg": 15.2,
                "PC_GDP_avg": 1.9,
                "USD_CAP_avg": 650.5,
                "PC_HEALTHXP_volatility": 0.9,
                "PC_GDP_volatility": 0.05,
                "USD_CAP_volatility": 45.3
            }
        ]
    }
    
    response = client.post("/predict/batch", json=payload)
    print(response.json())
    assert response.status_code == 200
    data = response.json()
    assert data['total_countries'] == 2
    assert len(data['predictions']) == 2
    assert 'processing_time_seconds' in data


def test_model_info_endpoint():
    """Test model info endpoint"""
    response = client.get("/model/info")
    
    assert response.status_code == 200
    data = response.json()
    assert data['version'] == "1.0.0"
    assert data['n_clusters'] == 3