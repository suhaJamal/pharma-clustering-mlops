# 🏥 Pharma Clustering MLOps

**Production ML System for Pharmaceutical Market Segmentation**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/tests-8%20passing-brightgreen.svg)](https://github.com/suhaJamal/pharma-clustering-mlops/actions)

---

## 🎯 Project Overview

A production-ready ML system that segments OECD countries by pharmaceutical spending patterns using K-means clustering. Features complete MLOps pipeline with automated training, testing, deployment, and monitoring.

**Business Value:** Identifies three distinct pharmaceutical market segments (Declining Markets, Stable Markets, High-Value Markets) to inform strategic market entry decisions.

---

## 📊 Two-Phase Development

### **Phase 1: Data Science Analysis** (Team Project)
- Exploratory analysis on 10 years of OECD pharmaceutical data (2011-2020)
- K-means clustering identifying 3 market segments
- Strategic business recommendations

**Team:** Ahil Khuwaja, Fabiana Camargo Franco Barril, Mohammad Faisal, Saranya Manoharan, Suha Islaih

### **Phase 2: Production ML System** (Individual MLOps Implementation)
- REST API with FastAPI
- Docker containerization
- Automated testing and CI/CD
- Model versioning and comparison
- Production monitoring

**Developer:** Suha Islaih

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker (optional)
- Git

### Installation

```bash
# Clone repository
git clone https://github.com/suhaJamal/pharma-clustering-mlops.git
cd pharma-clustering-mlops/Phase_2

# Create virtual environment
python -m venv myenv
source myenv/bin/activate  # Windows: myenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run API Locally

```bash
# Start FastAPI server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

**Access:**
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health
- Metrics: http://localhost:8000/metrics

### Run with Docker

```bash
# Build and run
docker-compose up

# Or build manually
docker build -t pharma-api:v1.0 .
docker run -p 8000:8000 pharma-api:v1.0
```

---

## 📡 API Endpoints

### **POST /predict** - Single Country Prediction

**Request:**
```json
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
}
```

**Response:**
```json
{
  "country": "Germany",
  "cluster": 2,
  "cluster_name": "High-Value Pharma Markets",
  "recommendation": "⭐ PRIORITY for innovative products",
  "confidence": 0.281,
  "model_version": "1.0.0",
  "timestamp": "2025-11-27T18:48:29.847937",
  "processing_time_seconds": 0.015
}
```

### **POST /predict/batch** - Batch Predictions

**Request:**
```json
{
  "countries": [
    {
      "country": "Germany",
      "PC_HEALTHXP_growth": -0.51,
      ...
    },
    {
      "country": "France",
      "PC_HEALTHXP_growth": -2.5,
      ...
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [...],
  "total_countries": 2,
  "processing_time_seconds": 0.023
}
```

### **GET /health** - Health Check

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1.0.0"
}
```

### **GET /metrics** - Monitoring Metrics

```json
{
  "total_predictions": 150,
  "predictions_by_cluster": {
    "0": 10,
    "1": 95,
    "2": 45
  },
  "average_response_time_seconds": 0.0074,
  "uptime_hours": 24.5,
  "start_time": "2025-11-27T12:00:00"
}
```

### **GET /model/info** - Model Metadata

```json
{
  "version": "1.0.0",
  "created_at": "2025-11-26T16:55:31",
  "n_clusters": 3,
  "silhouette_score": 0.2894
}
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CLIENT REQUEST                        │
│            (Web Browser / API Client)                    │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                 FASTAPI APPLICATION                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │  Real-time  │  │    Batch    │  │ Monitoring  │    │
│  │  /predict   │  │/predict/batch│  │  /metrics   │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│              MODEL SERVING LAYER                         │
│  - Versioned Models (v1.0.0, v1.1.0)                    │
│  - StandardScaler for feature preprocessing             │
│  - Prediction logging and monitoring                    │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│           MONITORING & LOGGING                           │
│  - Prediction logs (JSONL format)                       │
│  - Performance metrics tracking                         │
│  - Response time monitoring                             │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
pharma-clustering-mlops/
│
├── .github/
│   └── workflows/
│       └── ci.yml                    # GitHub Actions CI/CD
│
├── Phase_1/                          # Original data science project
│   ├── data/
│   ├── notebooks/
│   └── README.md
│
├── Phase_2/                          # Production ML system
│   ├── src/
│   │   ├── api/
│   │   │   ├── main.py              # FastAPI application
│   │   │   └── schemas.py           # Pydantic models
│   │   ├── models/
│   │   │   └── predictor.py         # Model serving logic
│   │   └── monitoring/
│   │       └── logger.py            # Prediction logging
│   │
│   ├── tests/
│   │   ├── test_api.py              # API endpoint tests
│   │   └── test_model.py            # Model unit tests
│   │
│   ├── scripts/
│   │   └── train.py                 # Automated training
│   │
│   ├── models/                      # Versioned model artifacts
│   │   ├── v1.0.0/
│   │   └── v1.1.0/
│   │
│   ├── logs/                        # Prediction logs
│   │   ├── predictions.jsonl
│   │   └── metrics.json
│   │
│   ├── deployment/
│   │   ├── Dockerfile
│   │   ├── docker-compose.yml
│   │   └── requirements.txt
│   │
│   └── data/
│       ├── features/                # Engineered features
│       └── training/                # Training data
│
└── README.md                        # This file
```

---

## 🎓 MLOps Features Implemented

| Feature | Implementation | Status |
|---------|---------------|--------|
| **Real-time Inference** | `/predict` endpoint with <50ms latency | ✅ Complete |
| **Batch Inference** | `/predict/batch` for multiple countries | ✅ Complete |
| **Model Versioning** | Semantic versioning (v1.0.0, v1.1.0) | ✅ Complete |
| **Automated Testing** | 8 pytest tests (unit + integration) | ✅ Complete |
| **CI/CD Pipeline** | GitHub Actions on every push | ✅ Complete |
| **Docker Containerization** | Dockerfile + docker-compose | ✅ Complete |
| **Automated Training** | `scripts/train.py` with version increment | ✅ Complete |
| **Model Comparison** | Compare performance before deployment | ✅ Complete |
| **Production Monitoring** | Prediction logging + metrics endpoint | ✅ Complete |
| **API Documentation** | Auto-generated Swagger/OpenAPI docs | ✅ Complete |
| **Drift Detection** | Statistical tests for data drift | 🚧 Planned |
| **Cloud Deployment** | GCP Cloud Run / Vertex AI | 📋 Future |

---

## 🔧 Development Workflow

### Run Tests

```bash
cd Phase_2
pytest tests/ -v
```

**Expected output:** 8 tests passing

### Train New Model Version

```bash
python scripts/train.py
```

**What it does:**
- Loads training data
- Trains K-means model
- Auto-increments version (v1.0.0 → v1.1.0)
- Compares to previous version
- Recommends deployment if better

### Check Logs

```bash
# View prediction logs
cat logs/predictions.jsonl

# View metrics
cat logs/metrics.json
```

---

## 📊 Model Performance

**Current Production Model:** v1.0.0

| Metric | Value |
|--------|-------|
| Silhouette Score | 0.2894 |
| Inertia | 188.86 |
| Number of Clusters | 3 |
| Training Samples | 36 countries |
| Features | 9 engineered features |

**Cluster Distribution:**
- Cluster 0 (Declining Markets): 5 countries
- Cluster 1 (Stable Markets): 23 countries
- Cluster 2 (High-Value Markets): 8 countries

---

## 🧪 Testing

### Test Coverage

- ✅ Model loading and initialization
- ✅ Single prediction endpoint
- ✅ Batch prediction endpoint
- ✅ Health check endpoint
- ✅ Model info endpoint
- ✅ Model metadata validation
- ✅ Prediction response format
- ✅ Error handling

### Run Tests Locally

```bash
pytest tests/ -v --cov=src
```

### CI/CD Testing

Every push triggers automated testing in GitHub Actions:
- Python 3.11 environment setup
- Dependency installation
- Test execution
- Model artifact validation

---

## 🐳 Docker Deployment

### Build Image

```bash
docker build -t pharma-api:v1.0 .
```

### Run Container

```bash
docker run -p 8000:8000 pharma-api:v1.0
```

### Use Docker Compose

```bash
# Start
docker-compose up

# Stop
docker-compose down
```

---

## 📈 Monitoring & Observability

### Prediction Logging

Every prediction is logged with:
- Timestamp
- Country name
- Predicted cluster
- Confidence score
- Model version used
- Response time

**Log format:** JSONL (one JSON object per line)

### Metrics Tracking

Real-time metrics available at `/metrics`:
- Total predictions count
- Cluster distribution
- Average response time
- System uptime

### Performance

- **Latency:** ~7ms average response time
- **Throughput:** ~200 requests/second
- **Model size:** 2.3 MB (compressed)
- **Docker image:** ~150 MB

---

## 🗺️ Roadmap

### ✅ Completed
- FastAPI REST API
- Batch prediction support
- Model versioning system
- Automated testing (pytest)
- CI/CD with GitHub Actions
- Docker containerization
- Training automation
- Model comparison logic
- Production monitoring

### 🚧 In Progress
- Drift detection implementation
- Extended monitoring dashboard

### 📋 Future Enhancements
- GCP Cloud Run deployment
- Feature store integration
- A/B testing framework
- Online learning implementation
- Cost optimization analysis
- Fairness and bias metrics

---

## 📚 Technologies Used

**ML/Data Science:**
- scikit-learn 1.7.2 (K-means, StandardScaler)
- pandas 1.5.3
- numpy 1.26.4

**API/Backend:**
- FastAPI 0.104.1
- Pydantic 2.5.0
- Uvicorn 0.24.0

**Testing:**
- pytest 7.4.3
- httpx 0.25.2

**DevOps:**
- Docker
- GitHub Actions
- docker-compose

**Monitoring:**
- Custom logging system
- JSONL format logs

---

## 📞 Contact

**Suha Islaih**
- LinkedIn: [linkedin.com/in/suha-islaih](https://linkedin.com/in/suha-islaih)
- Email: suha@smartdiversity.ca
- GitHub: [@suhaJamal](https://github.com/suhaJamal)

---

## 🙏 Acknowledgments

- **Team DS-4** for Phase 1 data science work
- **University of Toronto Data Science Program** for project framework
- **OECD** for pharmaceutical spending dataset

---

## 📄 License

MIT License - See LICENSE file for details

---

**Last Updated:** November 27, 2025  
**Project Status:** ✅ Production-ready with active monitoring  
**Version:** Phase 2 - v1.0.0
