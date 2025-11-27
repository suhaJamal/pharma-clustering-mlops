"""
Automated Model Training Script
Trains new model version and saves with metadata
"""
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))


def get_next_version(models_dir):
    """
    Get next version number based on existing models
    
    Returns:
        str: Next version (e.g., "1.1.0")
    """
    models_path = Path(models_dir)
    
    if not models_path.exists():
        return "1.0.0"
    
    # Find all version directories
    versions = []
    for v_dir in models_path.glob("v*"):
        if v_dir.is_dir():
            version = v_dir.name.replace("v", "")
            # Handle both "1.0" and "1.0.0" formats
            parts = version.split(".")
            if len(parts) == 2:
                version = f"{parts[0]}.{parts[1]}.0"
            versions.append(version)
    
    if not versions:
        return "1.0.0"
    
    # Get latest version and increment
    latest = sorted(versions)[-1]
    major, minor, patch = latest.split(".")
    
    # Increment minor version
    new_version = f"{major}.{int(minor) + 1}.0"
    
    return new_version


def load_training_data(data_path):
    """Load and prepare training data"""
    df = pd.read_csv(data_path)
    
    # Extract features (exclude COUNTRY column)
    feature_cols = [col for col in df.columns if col not in ['COUNTRY', 'Cluster_k3', 'Cluster_Name']]
    X = df[feature_cols].values
    feature_names = feature_cols
    
    return X, feature_names, df


def train_model(X, n_clusters=3, random_state=42):
    """
    Train K-means clustering model
    
    Returns:
        model, scaler, metrics
    """
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
        max_iter=300
    )
    kmeans.fit(X_scaled)
    
    # Calculate metrics
    silhouette = silhouette_score(X_scaled, kmeans.labels_)
    inertia = kmeans.inertia_
    
    metrics = {
        'silhouette_score': float(silhouette),
        'inertia': float(inertia)
    }
    
    return kmeans, scaler, metrics


def save_model(model, scaler, version, feature_names, X, metrics, models_dir):
    """Save model with versioning and metadata"""
    # Create version directory
    version_dir = Path(models_dir) / f"v{version}"
    version_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model artifacts
    joblib.dump(model, version_dir / 'model.pkl')
    joblib.dump(scaler, version_dir / 'scaler.pkl')
    
    # Create metadata
    metadata = {
        'model_version': version,
        'creation_date': datetime.now().isoformat(),
        'model_type': 'KMeans Clustering',
        'model_parameters': {
            'n_clusters': model.n_clusters,
            'random_state': model.random_state,
            'n_init': model.n_init,
            'max_iter': model.max_iter
        },
        'training_data': {
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'feature_names': feature_names
        },
        'performance_metrics': metrics,
        'cluster_size': {
            f'cluster_{i}': int((model.labels_ == i).sum())
            for i in range(model.n_clusters)
        }
    }
    
    # Save metadata
    with open(version_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Model v{version} saved to {version_dir}")
    
    return metadata


def compare_models(new_metrics, old_version_dir):
    """Compare new model with previous version"""
    
    if not old_version_dir.exists():
        print("ℹ️  No previous version to compare")
        return True
    
    # Load old metadata
    with open(old_version_dir / 'metadata.json', 'r') as f:
        old_metadata = json.load(f)
    
    old_metrics = old_metadata['performance_metrics']
    
    print("\n📊 MODEL COMPARISON:")
    print(f"{'Metric':<20} {'Old Version':<15} {'New Version':<15} {'Change':<10}")
    print("-" * 65)
    
    silhouette_old = old_metrics['silhouette_score']
    silhouette_new = new_metrics['silhouette_score']
    silhouette_change = silhouette_new - silhouette_old
    
    print(f"{'Silhouette Score':<20} {silhouette_old:<15.4f} {silhouette_new:<15.4f} {silhouette_change:+.4f}")
    
    inertia_old = old_metrics['inertia']
    inertia_new = new_metrics['inertia']
    inertia_change = inertia_new - inertia_old
    
    print(f"{'Inertia':<20} {inertia_old:<15.2f} {inertia_new:<15.2f} {inertia_change:+.2f}")
    
    # Decision: better if silhouette increases
    is_better = silhouette_new > silhouette_old
    
    if is_better:
        print("\n✅ New model is BETTER (higher silhouette score)")
    else:
        print("\n⚠️  New model is WORSE (lower silhouette score)")
    
    return is_better


def main():
    """Main training pipeline"""
    
    print("🚀 Starting model training pipeline...\n")
    
    # Paths
    data_path = Path("data/features/engineered_features.csv")
    models_dir = Path("models")
    print(models_dir)
    # Get next version
    new_version = get_next_version(models_dir)
    print(f"📦 Training model version: {new_version}")
    
    # Load data
    print("📁 Loading training data...")
    X, feature_names, df = load_training_data(data_path)
    print(f"   Loaded {X.shape[0]} samples with {X.shape[1]} features")
    
    # Train model
    print("🔧 Training K-means model...")
    model, scaler, metrics = train_model(X, n_clusters=3, random_state=22)
    print(f"   Silhouette Score: {metrics['silhouette_score']:.4f}")
    print(f"   Inertia: {metrics['inertia']:.2f}")
    
    # Save model
    print("💾 Saving model...")
    metadata = save_model(model, scaler, new_version, feature_names, X, metrics, models_dir)
    
    # Compare with previous version
    old_version_dir = models_dir / f"v{get_previous_version(new_version)}"
    is_better = compare_models(metrics, old_version_dir)
    
    print(f"\n🎉 Training complete! Model v{new_version} ready.")
    
    if is_better:
        print("💡 Recommendation: Deploy this version to production")
    else:
        print("💡 Recommendation: Keep previous version in production")


def get_previous_version(current_version):
    """Get previous version number"""
    major, minor, patch = current_version.split(".")
    
    if int(minor) > 0:
        prev_version = f"{major}.{int(minor) - 1}.0"
    else:
        prev_version = f"{int(major) - 1}.0.0"
    
    return prev_version


if __name__ == "__main__":
    main()