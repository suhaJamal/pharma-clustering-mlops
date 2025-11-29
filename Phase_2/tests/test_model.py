"""
Unit tests for model predictor
"""
import pytest
import numpy as np
from src.models.predictor import ModelPredictor


def test_model_loads():
    """Test that model loads successfully"""
    predictor = ModelPredictor(model_dir="models/v1.2.0")
    
    assert predictor.model is not None
    assert predictor.scaler is not None
    assert predictor.metadata is not None


def test_model_prediction():
    """Test that model makes valid predictions"""
    predictor = ModelPredictor(model_dir="models/v1.2.0")
    
    # Test data (Germany)
    features = {
        'PC_HEALTHXP_growth': -0.51,
        'PC_GDP_growth': 1.31,
        'USD_CAP_growth': 4.24,
        'PC_HEALTHXP_avg': 14.14,
        'PC_GDP_avg': 1.60,
        'USD_CAP_avg': 790.71,
        'PC_HEALTHXP_volatility': 0.27,
        'PC_GDP_volatility': 0.06,
        'USD_CAP_volatility': 102.02
    }
    
    result = predictor.predict(features, "Germany")
    
    # Assertions
    assert result['country'] == "Germany"
    assert result['cluster'] in [0, 1, 2]
    assert 0 <= result['confidence'] <= 1
    assert result['model_version'] == "1.2.0"
    assert 'cluster_name' in result
    assert 'recommendation' in result


def test_model_metadata():
    """Test that model metadata is correct"""
    predictor = ModelPredictor(model_dir="models/v1.2.0")
    
    info = predictor.get_model_info()
    
    assert info['version'] == "1.2.0"
    assert info['n_clusters'] == 3
    assert 'silhouette_score' in info