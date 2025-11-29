"""
Drift Detection System
Detects when production data differs from training data
"""
import json
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Dict, List, Any
from collections import deque


class DriftDetector:
    """
    Detects data drift using statistical tests
    Compares production features to training distribution
    """
    
    def __init__(self, model_dir: str = "models/v1.0.0", max_samples: int = 100):
        """
        Initialize drift detector
        
        Args:
            model_dir: Path to model directory with training_stats.json
            max_samples: Number of recent predictions to keep for comparison
        """
        self.model_dir = Path(model_dir)
        self.max_samples = max_samples
        
        # Load training statistics
        self.training_stats = self._load_training_stats()
        
        # Store recent production data (one deque per feature)
        self.production_data = {
            feature: deque(maxlen=max_samples) 
            for feature in self.training_stats.keys()
        }
        
        # Drift threshold (p-value)
        self.drift_threshold = 0.05  # p < 0.05 indicates significant drift
    
    def _load_training_stats(self) -> Dict:
        """Load training statistics from file"""
        stats_file = self.model_dir / 'training_stats.json'
        
        if not stats_file.exists():
            raise FileNotFoundError(
                f"Training statistics not found at {stats_file}. "
                "Please retrain the model to generate statistics."
            )
        
        with open(stats_file, 'r') as f:
            return json.load(f)
    
    def add_prediction(self, features: Dict[str, float]):
        """
        Add a prediction's features to production data
        
        Args:
            features: Dictionary of feature values
        """
        for feature_name, value in features.items():
            if feature_name in self.production_data:
                self.production_data[feature_name].append(value)
    
    def detect_drift(self) -> Dict[str, Any]:
        """
        Detect drift across all features
        
        Returns:
            Dictionary with drift analysis results
        """
        # Need at least 30 samples for reliable statistical test
        min_samples = min(len(data) for data in self.production_data.values())
        
        if min_samples < 30:
            return {
                'drift_detected': False,
                'ready': False,
                'message': f'Need {30 - min_samples} more predictions for drift detection (minimum 30 required)',
                'current_samples': min_samples
            }
        
        drift_results = {}
        features_drifted = []
        
        for feature_name in self.training_stats.keys():
            result = self._test_feature_drift(feature_name)
            drift_results[feature_name] = result
            
            if result['status'] == 'drifted':
                features_drifted.append(feature_name)
        
        # Overall drift detected if any feature drifted
        drift_detected = len(features_drifted) > 0
        
        # Generate recommendation
        if len(features_drifted) >= 3:
            recommendation = "⚠️ CRITICAL: Multiple features drifted. Retrain model immediately."
        elif len(features_drifted) > 0:
            recommendation = "⚠️ WARNING: Some features drifted. Monitor closely and consider retraining."
        else:
            recommendation = "✅ OK: All features stable. No action needed."
        
        return {
            'drift_detected': drift_detected,
            'ready': True,
            'features_drifted': features_drifted,
            'drift_scores': drift_results,
            'recent_predictions_analyzed': min_samples,
            'overall_recommendation': recommendation,
            'drift_threshold': self.drift_threshold
        }
    
    def _test_feature_drift(self, feature_name: str) -> Dict[str, Any]:
        """
        Test single feature for drift using Kolmogorov-Smirnov test
        
        Args:
            feature_name: Name of feature to test
            
        Returns:
            Dictionary with test results
        """
        # Get training and production data
        training_values = np.array(self.training_stats[feature_name]['values'])
        production_values = np.array(list(self.production_data[feature_name]))
        
        # Perform KS test
        ks_statistic, p_value = stats.ks_2samp(training_values, production_values)
        
        # Determine drift status
        if p_value < self.drift_threshold:
            status = 'drifted'
            recommendation = 'Monitor closely or retrain model'
        elif p_value < 0.1:
            status = 'warning'
            recommendation = 'Watch for continued drift'
        else:
            status = 'stable'
            recommendation = 'OK'
        
        # Calculate distribution shift
        training_mean = self.training_stats[feature_name]['mean']
        production_mean = float(np.mean(production_values))
        mean_shift_pct = ((production_mean - training_mean) / training_mean * 100) if training_mean != 0 else 0
        
        return {
            'p_value': round(float(p_value), 4),
            'ks_statistic': round(float(ks_statistic), 4),
            'status': status,
            'recommendation': recommendation,
            'training_mean': round(training_mean, 4),
            'production_mean': round(production_mean, 4),
            'mean_shift_percent': round(mean_shift_pct, 2)
        }
    
    def get_feature_comparison(self, feature_name: str) -> Dict[str, Any]:
        """
        Get detailed comparison for a single feature
        
        Args:
            feature_name: Name of feature
            
        Returns:
            Comparison statistics
        """
        if feature_name not in self.training_stats:
            raise ValueError(f"Unknown feature: {feature_name}")
        
        production_values = list(self.production_data[feature_name])
        
        if len(production_values) == 0:
            return {
                'feature': feature_name,
                'error': 'No production data available yet'
            }
        
        return {
            'feature': feature_name,
            'training': self.training_stats[feature_name],
            'production': {
                'mean': float(np.mean(production_values)),
                'std': float(np.std(production_values)),
                'min': float(np.min(production_values)),
                'max': float(np.max(production_values)),
                'count': len(production_values)
            }
        }
    
    def reset(self):
        """Reset all production data"""
        for feature in self.production_data.keys():
            self.production_data[feature].clear()