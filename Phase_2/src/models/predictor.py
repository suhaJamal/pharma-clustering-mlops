"""
Model Predictor
Loads trained model and makes predictions
"""
import joblib
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


class ModelPredictor:
    """
    Loads and serves the clustering model for predictions
    Singleton pattern - model loaded once and reused
    """
    
    def __init__(self, model_dir: str = "models/v1.0"):
        """
        Initialize predictor and load model artifacts
        
        Args:
            model_dir: Path to model version directory
        """
        self.model_dir = Path(model_dir)
        self.model = None
        self.scaler = None
        self.metadata = None
        self.feature_names = None
        
        # Cluster interpretations
        self.cluster_names = {
            0: "Declining Markets",
            1: "Stable Moderate Markets",
            2: "High-Value Pharma Markets"
        }
        
        self.cluster_recommendations = {
            0: "❌ NOT recommended for new market entry",
            1: "✅ IDEAL for market expansion",
            2: "⭐ PRIORITY for innovative products"
        }
        
        self._load_model()
    
    def _load_model(self):
        """Load model, scaler, and metadata from disk"""
        try:
            # Load model and scaler
            self.model = joblib.load(self.model_dir / 'model.pkl')
            self.scaler = joblib.load(self.model_dir / 'scaler.pkl')
            
            # Load metadata
            with open(self.model_dir / 'metadata.json', 'r') as f:
                self.metadata = json.load(f)
            
            self.feature_names = self.metadata['training_data']['feature_names']
            
            print(f"✅ Model v{self.metadata['model_version']} loaded successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def predict(self, features: Dict[str, float], country: str) -> Dict[str, Any]:
        """
        Make prediction for a single country
        
        Args:
            features: Dictionary of feature values
            country: Country name
            
        Returns:
            Dictionary with prediction results
        """
        # Convert features dict to array in correct order
        feature_array = np.array([
            [features[name] for name in self.feature_names]
        ])
        
        # Scale features
        features_scaled = self.scaler.transform(feature_array)
        
        # Predict cluster
        cluster = int(self.model.predict(features_scaled)[0])
        
        # Calculate confidence (distance to centroid)
        distances = self.model.transform(features_scaled)[0]
        confidence = float(1.0 / (1.0 + distances[cluster]))
        
        # Build response
        result = {
            'country': country,
            'cluster': cluster,
            'cluster_name': self.cluster_names[cluster],
            'recommendation': self.cluster_recommendations[cluster],
            'confidence': round(confidence, 3),
            'model_version': self.metadata['model_version'],
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata"""
        return {
            'version': self.metadata['model_version'],
            'created_at': self.metadata['creation_date'],
            'n_clusters': self.metadata['model_parameters']['n_clusters'],
            'silhouette_score': self.metadata['performance_metrics']['silhouette_score']
        }