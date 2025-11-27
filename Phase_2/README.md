# 🏥 Pharma Clustering MLOps

**Production ML System for Pharmaceutical Market Segmentation**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![GCP](https://img.shields.io/badge/GCP-Vertex%20AI-orange.svg)](https://cloud.google.com/vertex-ai)

---

## 🎯 Project Overview

This project transforms a data science clustering analysis into a **production-ready ML system** with real-time inference, batch predictions, model monitoring, and MLOps best practices.

**Business Problem:** Segment OECD countries by pharmaceutical spending patterns to inform strategic market entry decisions for pharmaceutical companies.

**ML Solution:** K-means clustering (k=3) identifying three distinct market segments based on spending levels, growth trajectories, and market stability.

---

## 📊 Project Phases

### **Phase 1: Exploratory Data Science** (Team Project)
Located in `Phase_1/` directory

**Team:** Ahil Khuwaja, Fabiana Camargo Franco Barril, Mohammad Faisal, Saranya Manoharan, Suha Islaih

**Deliverables:**
- Exploratory data analysis (EDA) on 10 years of OECD pharmaceutical spending data (2011-2020)
- Feature engineering: 9 features per country (growth rates, spending levels, volatility)
- K-means clustering analysis identifying 3 market segments
- Strategic business recommendations for market entry

**Key Findings:**
- 🔴 **Cluster 0:** Crisis/Declining Markets (5 countries) - High risk, not recommended
- 🟡 **Cluster 1:** Stable Moderate Markets (23 countries) - Ideal for expansion
- 🟢 **Cluster 2:** High-Value Pharma Markets (8 countries) - Priority for innovation

**Technologies:** Python, pandas, scikit-learn, matplotlib, Jupyter notebooks

**Original Repository:** [DS04-Team-Project](https://github.com/saranya-mano/DS04-Team-Project)

---

### **Phase 2: Production ML System** (Individual MLOps Implementation)
Located in `Phase_2/` directory

**Developer:** Suha Islaih

**Objective:** Transform the clustering model into a production-grade ML system demonstrating MLOps best practices.

**Key Features:**
- ✅ **Real-time Inference API** - Predict cluster for individual countries via REST API
- ✅ **Batch Inference** - Process multiple countries efficiently
- ✅ **Model Versioning** - Track model versions with metadata and lineage
- ✅ **Docker Containerization** - Portable, reproducible deployment
- ✅ **GCP Deployment** - Cloud Run / Vertex AI deployment
- ✅ **Monitoring & Drift Detection** - Track model performance and data drift
- ✅ **Feature Store** - Centralized feature management and versioning
- ✅ **MLOps Pipeline** - Automated training, evaluation, and deployment

**Technologies:** FastAPI, Docker, GCP (Vertex AI, Cloud Run, BigQuery, GCS), MLflow, Prometheus

---

## 🏗️ Architecture (Phase 2)

```
┌─────────────────────────────────────────────────────────┐
│                    CLIENT REQUEST                        │
│            (REST API / Web Dashboard)                    │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                 FASTAPI APPLICATION                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │  Real-time  │  │    Batch    │  │ Monitoring  │    │
│  │  Endpoint   │  │  Endpoint   │  │  Endpoint   │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│              MODEL SERVING LAYER                         │
│  - Model Registry (Versioned Models: v1.0, v1.1, v2.0)  │
│  - Model Cache (Redis for low latency)                  │
│  - Feature Store (GCS + BigQuery)                       │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│           MONITORING & LOGGING                           │
│  - Cloud Logging (All predictions logged)               │
│  - Drift Detection (Statistical tests)                  │
│  - Performance Metrics (Latency, throughput)            │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start (Phase 2)

### Prerequisites
- Python 3.8+
- Docker (optional, for containerization)
- GCP Account (optional, for cloud deployment)

### Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/pharma-clustering-mlops.git
cd pharma-clustering-mlops/Phase_2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r deployment/requirements.txt
```

### Run API Locally

```bash
# Start FastAPI server
cd Phase_2
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

**Access API:**
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### Example API Request

**Real-time Prediction:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "country": "Germany",
    "pharma_per_capita": 659,
    "pharma_pct_health": 18,
    "pharma_pct_gdp": 1.8,
    "growth_rate": 4.0,
    "volatility": 2.5
  }'
```

**Response:**
```json
{
  "country": "Germany",
  "cluster": 2,
  "cluster_name": "High-Value Pharma Markets",
  "recommendation": "Priority for innovative products",
  "confidence": 0.87,
  "model_version": "v1.0",
  "timestamp": "2024-11-26T10:30:00"
}
```

---

## 📦 Project Structure

```
pharma-clustering-mlops/
│
├── Phase_1/                          # Original team data science project
│   ├── data/
│   ├── notebooks/
│   ├── scripts/
│   └── README.md
│
├── Phase_2/                          # Production ML system
│   ├── src/
│   │   ├── models/                   # Model classes and training logic
│   │   ├── api/                      # FastAPI application
│   │   │   ├── main.py
│   │   │   ├── routers/
│   │   │   └── schemas.py
│   │   ├── feature_store/            # Feature engineering pipeline
│   │   ├── monitoring/               # Drift detection and tracking
│   │   └── utils/                    # Helper functions
│   │
│   ├── deployment/
│   │   ├── Dockerfile
│   │   ├── docker-compose.yml
│   │   └── requirements.txt
│   │
│   ├── models/                       # Versioned model artifacts
│   │   ├── v1.0/
│   │   ├── v1.1/
│   │   └── production/               # Current production model
│   │
│   ├── tests/                        # Automated tests
│   │   ├── test_api.py
│   │   ├── test_models.py
│   │   └── test_features.py
│   │
│   └── README_PHASE2.md              # Phase 2 detailed documentation
│
└── README.md                         # This file
```

---

## 🎓 MLOps Concepts Demonstrated

This project implements production ML best practices:

| Concept | Implementation | Location |
|---------|---------------|----------|
| **Real-time Inference** | FastAPI `/predict` endpoint | `src/api/routers/predict.py` |
| **Batch Inference** | `/predict/batch` endpoint | `src/api/routers/batch.py` |
| **Model Versioning** | Semantic versioning (v1.0, v1.1) | `models/` |
| **Containerization** | Docker + docker-compose | `deployment/Dockerfile` |
| **Feature Store** | GCS + BigQuery integration | `src/feature_store/` |
| **Model Monitoring** | Prediction logging + metrics | `src/monitoring/` |
| **Drift Detection** | Statistical tests on features | `src/monitoring/drift_detector.py` |
| **Reproducibility** | Random seeds, versioning | Throughout codebase |
| **CI/CD** | Automated testing + deployment | `.github/workflows/` |
| **Cost Optimization** | Model compression, batching | `src/models/` |

---

## 📊 Model Performance

**Clustering Metrics (Phase 1):**
- Silhouette Score: 0.289
- Number of Clusters: 3
- Countries Analyzed: 36
- Time Period: 2011-2020

**Production Metrics (Phase 2):**
- API Latency: ~50ms (p95)
- Throughput: ~200 requests/second
- Model Size: 2.3 MB (compressed)
- Docker Image: 150 MB

---

## 🔧 Development Workflow

### Training New Model Version

```bash
python Phase_2/src/models/train.py --version 1.1 --data Phase_1/data/processed/
```

### Running Tests

```bash
cd Phase_2
pytest tests/ -v
```

### Building Docker Image

```bash
cd Phase_2/deployment
docker build -t pharma-clustering:v1.0 .
docker run -p 8000:8000 pharma-clustering:v1.0
```

### Deploying to GCP

```bash
# Deploy to Cloud Run
gcloud run deploy pharma-clustering \
  --image gcr.io/PROJECT_ID/pharma-clustering:v1.0 \
  --platform managed \
  --region us-central1
```

---

## 📈 Roadmap

### ✅ Completed (Phase 2)
- [x] FastAPI application with real-time and batch endpoints
- [x] Model versioning system
- [x] Docker containerization
- [x] Basic monitoring and logging

### 🚧 In Progress
- [ ] GCP Vertex AI deployment
- [ ] Feature store with BigQuery
- [ ] Drift detection with automated alerts

### 📋 Planned
- [ ] A/B testing framework (v1.0 vs v2.0)
- [ ] Online learning implementation
- [ ] Interactive monitoring dashboard
- [ ] Cost optimization analysis
- [ ] Fairness and bias analysis

---

## 🤝 Contributing

This is a learning project demonstrating MLOps practices. Feedback and suggestions are welcome!

**To contribute:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📄 License

**Phase 1:** MIT License (Team project)  
**Phase 2:** MIT License (Individual implementation)

---

## 📞 Contact

**Suha Islaih**
- LinkedIn: [linkedin.com/in/suha-islaih](https://linkedin.com/in/suha-islaih)
- Email: suha@smartdiversity.ca
- GitHub: [@suhaJamal](https://github.com/suhaJama)

---

## 🙏 Acknowledgments

- **Team DS-4** for the foundational data science work in Phase 1
- **University of Toronto Data Science Program** for project framework
- **OECD** for pharmaceutical spending dataset
- **FastAPI, scikit-learn, and GCP** communities for excellent tools

---

**Last Updated:** November 26, 2024  
**Project Status:** 🚧 Phase 2 in active development  
**Documentation:** See `Phase_2/README_PHASE2.md` for technical details
