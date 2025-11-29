"""
API Request/Response Schemas
Defines data structures for FastAPI endpoints
"""
from pydantic import BaseModel, Field
from typing import Optional


class PredictionRequest(BaseModel):
    """Request model for single country prediction"""
    
    country: str = Field(..., example="Germany")
    PC_HEALTHXP_growth: float = Field(..., example=-0.51)
    PC_GDP_growth: float = Field(..., example=1.31)
    USD_CAP_growth: float = Field(..., example=4.24)
    PC_HEALTHXP_avg: float = Field(..., example=14.14)
    PC_GDP_avg: float = Field(..., example=1.60)
    USD_CAP_avg: float = Field(..., example=790.71)
    PC_HEALTHXP_volatility: float = Field(..., example=0.27)
    PC_GDP_volatility: float = Field(..., example=0.06)
    USD_CAP_volatility: float = Field(..., example=102.02)
    
    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    model_config = {"protected_namespaces": ()}
    country: str
    cluster: int
    cluster_name: str
    recommendation: str
    confidence: float
    model_version: str
    timestamp: str
    processing_time_seconds: float | None = None
    

class HealthResponse(BaseModel):
    """Response model for health check"""
    model_config = {"protected_namespaces": ()}
    status: str
    model_loaded: bool
    model_version: Optional[str] = None

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    
    countries: list[PredictionRequest]
    
    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    
    model_config = {"protected_namespaces": ()}
    
    predictions: list[PredictionResponse]
    total_countries: int
    processing_time_seconds: float