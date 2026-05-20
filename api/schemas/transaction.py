"""
api/schemas/transaction.py
───────────────────────────
Pydantic models for request validation and response serialisation.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class AnomalyType(str, Enum):
    normal      = "normal"
    high_value  = "high_value"
    velocity    = "velocity"
    geographic  = "geographic"
    odd_hour    = "odd_hour"
    unknown     = "unknown"


# ── Request ───────────────────────────────────────────────────────────────────

class TransactionRequest(BaseModel):
    """
    A single credit card transaction to be scored for anomalies.
    All fields mirror real transaction data available at authorisation time.
    """
    transaction_id:   str           = Field(..., description="Unique transaction identifier")
    user_id:          int           = Field(..., ge=0, description="Internal user/card ID")
    amount:           float         = Field(..., gt=0, le=1_000_000, description="Transaction amount in EUR")
    mcc:              int           = Field(..., description="Merchant Category Code (ISO 18245)")
    country:          str           = Field(..., min_length=2, max_length=3, description="Transaction country (ISO 3166-1 alpha-2)")
    home_country:     str           = Field(..., min_length=2, max_length=3, description="Cardholder home country")
    hour:             int           = Field(..., ge=0, le=23, description="Hour of transaction (0-23)")
    day_of_week:      int           = Field(..., ge=0, le=6, description="Day of week (0=Monday)")
    is_card_present:  int           = Field(..., ge=0, le=1, description="1 if physical card used, 0 for CNP")
    credit_limit:     float         = Field(..., gt=0, description="Cardholder credit limit in EUR")
    avg_user_spend:   float         = Field(..., gt=0, description="User's historical average transaction amount")
    timestamp:        Optional[datetime] = Field(default=None, description="Transaction timestamp (ISO 8601)")

    @field_validator("country", "home_country")
    @classmethod
    def uppercase_country(cls, v):
        return v.upper()

    model_config = {
        "json_schema_extra": {
            "example": {
                "transaction_id":  "TXN_0001234",
                "user_id":         42,
                "amount":          1250.00,
                "mcc":             5944,
                "country":         "CN",
                "home_country":    "DE",
                "hour":            3,
                "day_of_week":     1,
                "is_card_present": 0,
                "credit_limit":    5000.0,
                "avg_user_spend":  45.0,
                "timestamp":       "2024-06-15T03:22:00",
            }
        }
    }


class BatchTransactionRequest(BaseModel):
    """Batch of up to 100 transactions scored in one request."""
    transactions: list[TransactionRequest] = Field(..., min_length=1, max_length=100)


# ── Response ──────────────────────────────────────────────────────────────────

class ModelScores(BaseModel):
    isolation_forest: float = Field(..., description="Isolation Forest anomaly score [0,1]")
    lof:              float = Field(..., description="Local Outlier Factor anomaly score [0,1]")
    ensemble:         float = Field(..., description="Weighted ensemble score [0,1]")


class RuleTrigger(BaseModel):
    rule_name:   str   = Field(..., description="Name of the triggered rule")
    description: str   = Field(..., description="Human-readable explanation")
    severity:    str   = Field(..., description="low | medium | high")
    value:       float = Field(..., description="The value that triggered the rule")
    threshold:   float = Field(..., description="The threshold that was exceeded")


class AnomalyResponse(BaseModel):
    transaction_id:   str         = Field(..., description="Echo of input transaction ID")
    is_anomaly:       bool        = Field(..., description="True if transaction is flagged as anomalous")
    anomaly_score:    float       = Field(..., description="Final ensemble anomaly score [0,1]")
    anomaly_type:     AnomalyType = Field(..., description="Most likely anomaly type if flagged")
    confidence:       str         = Field(..., description="low | medium | high")
    model_scores:     ModelScores = Field(..., description="Individual model scores")
    rules_triggered:  list[RuleTrigger] = Field(default_factory=list, description="Fraud rules triggered")
    explanation:      str         = Field(..., description="Human-readable explanation of the decision")
    processing_ms:    float       = Field(..., description="Server-side processing time in milliseconds")

    model_config = {
        "json_schema_extra": {
            "example": {
                "transaction_id":  "TXN_0001234",
                "is_anomaly":      True,
                "anomaly_score":   0.87,
                "anomaly_type":    "geographic",
                "confidence":      "high",
                "model_scores": {
                    "isolation_forest": 0.82,
                    "lof":              0.94,
                    "ensemble":         0.87,
                },
                "rules_triggered": [
                    {
                        "rule_name":   "foreign_transaction",
                        "description": "Transaction in CN while home country is DE",
                        "severity":    "high",
                        "value":       1.0,
                        "threshold":   0.0,
                    }
                ],
                "explanation": "High anomaly score driven by geographic anomaly (CN vs home DE) and unusual hour (03:00). 2 fraud rules triggered.",
                "processing_ms": 4.2,
            }
        }
    }


class BatchAnomalyResponse(BaseModel):
    results:        list[AnomalyResponse]
    total:          int
    anomaly_count:  int
    anomaly_rate:   float
    processing_ms:  float


class HealthResponse(BaseModel):
    status:      str
    model_loaded: bool
    version:     str
    uptime_s:    float


class ModelInfoResponse(BaseModel):
    models:      list[str]
    feature_cols: list[str]
    thresholds:  dict
    evaluation:  dict