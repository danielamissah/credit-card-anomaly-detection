from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class AnomalyType(str, Enum):
    normal     = "normal"
    high_value = "high_value"
    velocity   = "velocity"
    geographic = "geographic"
    odd_hour   = "odd_hour"
    unknown    = "unknown"


class TransactionRequest(BaseModel):
    transaction_id:  str   = Field(..., description="Unique transaction identifier")
    user_id:         int   = Field(..., ge=0)
    amount:          float = Field(..., gt=0, le=1_000_000)
    mcc:             int
    country:         str   = Field(..., min_length=2, max_length=3)
    home_country:    str   = Field(..., min_length=2, max_length=3)
    hour:            int   = Field(..., ge=0, le=23)
    day_of_week:     int   = Field(..., ge=0, le=6)
    is_card_present: int   = Field(..., ge=0, le=1)
    credit_limit:    float = Field(..., gt=0)
    avg_user_spend:  float = Field(..., gt=0)
    timestamp:       Optional[datetime] = None

    @field_validator("country", "home_country")
    @classmethod
    def uppercase_country(cls, v):
        return v.upper()

    model_config = {
        "json_schema_extra": {
            "example": {
                "transaction_id": "TXN_0001234", "user_id": 42,
                "amount": 1250.00, "mcc": 5944,
                "country": "CN", "home_country": "DE",
                "hour": 3, "day_of_week": 1,
                "is_card_present": 0, "credit_limit": 5000.0, "avg_user_spend": 45.0
            }
        }
    }


class BatchTransactionRequest(BaseModel):
    transactions: list[TransactionRequest] = Field(..., min_length=1, max_length=100)


class ModelScores(BaseModel):
    isolation_forest: float
    lof:              float
    ensemble:         float


class RuleTrigger(BaseModel):
    rule_name:   str
    description: str
    severity:    str
    value:       float
    threshold:   float


class AnomalyResponse(BaseModel):
    transaction_id:  str
    is_anomaly:      bool
    anomaly_score:   float
    anomaly_type:    AnomalyType
    confidence:      str
    model_scores:    ModelScores
    rules_triggered: list[RuleTrigger]
    explanation:     str
    processing_ms:   float


class BatchAnomalyResponse(BaseModel):
    results:       list[AnomalyResponse]
    total:         int
    anomaly_count: int
    anomaly_rate:  float
    processing_ms: float


class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
    version:      str
    uptime_s:     float


class ModelInfoResponse(BaseModel):
    models:       list[str]
    feature_cols: list[str]
    thresholds:   dict
    evaluation:   dict
