"""
Prediction Logger
Logs all predictions for monitoring and analysis
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import threading


class PredictionLogger:
    """
    Thread-safe prediction logger
    Logs predictions to JSON file for monitoring
    """
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.log_file = self.log_dir / "predictions.jsonl"
        self.metrics_file = self.log_dir / "metrics.json"
        
        self.lock = threading.Lock()
        
        # Initialize metrics
        self._init_metrics()
    
    def _init_metrics(self):
        """Initialize or load metrics"""
        if not self.metrics_file.exists():
            self.metrics = {
                'total_predictions': 0,
                'predictions_by_cluster': {
                    '0': 0,
                    '1': 0,
                    '2': 0
                },
                'total_response_time': 0.0,
                'start_time': datetime.now().isoformat()
            }
            self._save_metrics()
        else:
            with open(self.metrics_file, 'r') as f:
                self.metrics = json.load(f)
    
    def _save_metrics(self):
        """Save metrics to file"""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def log_prediction(self, 
                      country: str, 
                      cluster: int, 
                      confidence: float,
                      model_version: str,
                      response_time: float):
        """
        Log a single prediction
        
        Args:
            country: Country name
            cluster: Predicted cluster
            confidence: Prediction confidence
            model_version: Model version used
            response_time: Response time in seconds
        """
        with self.lock:
            # Create log entry
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'country': country,
                'cluster': cluster,
                'confidence': confidence,
                'model_version': model_version,
                'response_time_seconds': response_time
            }
            
            # Append to log file (JSONL format)
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
            
            # Update metrics
            self.metrics['total_predictions'] += 1
            self.metrics['predictions_by_cluster'][str(cluster)] += 1
            self.metrics['total_response_time'] += response_time
            
            self._save_metrics()
    
    def log_batch_prediction(self, 
                            predictions: list,
                            total_response_time: float):
        """
        Log batch predictions
        
        Args:
            predictions: List of prediction results
            total_response_time: Total time for batch
        """
        with self.lock:
            for pred in predictions:
                log_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'country': pred['country'],
                    'cluster': pred['cluster'],
                    'confidence': pred['confidence'],
                    'model_version': pred['model_version'],
                    'response_time_seconds': total_response_time / len(predictions),
                    'batch': True
                }
                
                with open(self.log_file, 'a') as f:
                    f.write(json.dumps(log_entry) + '\n')
                
                # Update metrics
                self.metrics['total_predictions'] += 1
                self.metrics['predictions_by_cluster'][str(pred['cluster'])] += 1
            
            self.metrics['total_response_time'] += total_response_time
            self._save_metrics()
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics
        
        Returns:
            Dictionary with metrics
        """
        with self.lock:
            avg_response_time = (
                self.metrics['total_response_time'] / self.metrics['total_predictions']
                if self.metrics['total_predictions'] > 0
                else 0
            )
            
            # Calculate uptime
            start_time = datetime.fromisoformat(self.metrics['start_time'])
            uptime_seconds = (datetime.now() - start_time).total_seconds()
            
            return {
                'total_predictions': self.metrics['total_predictions'],
                'predictions_by_cluster': self.metrics['predictions_by_cluster'],
                'average_response_time_seconds': round(avg_response_time, 4),
                'uptime_seconds': round(uptime_seconds, 2),
                'uptime_hours': round(uptime_seconds / 3600, 2),
                'start_time': self.metrics['start_time']
            }
    
    def reset_metrics(self):
        """Reset all metrics (admin function)"""
        with self.lock:
            self.metrics = {
                'total_predictions': 0,
                'predictions_by_cluster': {
                    '0': 0,
                    '1': 0,
                    '2': 0
                },
                'total_response_time': 0.0,
                'start_time': datetime.now().isoformat()
            }
            self._save_metrics()