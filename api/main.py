import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from api.models.detector import detector
from api.metrics.prometheus_metrics import (
    MODEL_LOADED_GAUGE, ANOMALY_RATE_GAUGE, BATCH_SIZE_HISTOGRAM, record_prediction,
)
from api.schemas.transaction import (
    TransactionRequest, BatchTransactionRequest,
    AnomalyResponse, BatchAnomalyResponse, HealthResponse, ModelInfoResponse,
)

APP_VERSION   = "1.0.0"
_start_time   = time.time()
_recent_flags = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading anomaly detection models ...")
    try:
        detector.load()
        MODEL_LOADED_GAUGE.set(1)
        print("Models loaded successfully.")
    except RuntimeError as e:
        print(f"Model loading failed: {e}")
        MODEL_LOADED_GAUGE.set(0)
    yield


app = FastAPI(
    title       = "Credit Card Anomaly Detection API",
    description = "Real-time anomaly detection using Isolation Forest + LOF ensemble with fraud rule engine.",
    version     = APP_VERSION,
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app, endpoint="/metrics")


def _update_anomaly_rate(is_anomaly: bool, window: int = 1000):
    global _recent_flags
    _recent_flags.append(int(is_anomaly))
    if len(_recent_flags) > window:
        _recent_flags = _recent_flags[-window:]
    ANOMALY_RATE_GAUGE.set(sum(_recent_flags) / len(_recent_flags))


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    return HealthResponse(
        status       = "healthy" if detector.is_loaded else "degraded",
        model_loaded = detector.is_loaded,
        version      = APP_VERSION,
        uptime_s     = round(time.time() - _start_time, 1),
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    if not detector.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return ModelInfoResponse(
        models       = ["isolation_forest", "local_outlier_factor"],
        feature_cols = detector.feature_cols,
        thresholds   = detector.thresholds,
        evaluation   = detector.evaluation,
    )


@app.post("/predict", response_model=AnomalyResponse, tags=["Prediction"])
async def predict(txn: TransactionRequest):
    if not detector.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded. Run ml/train.py first.")
    t0       = time.perf_counter()
    response = detector.predict(txn)
    total_ms = (time.perf_counter() - t0) * 1000
    record_prediction(
        is_anomaly=response.is_anomaly, anomaly_type=response.anomaly_type.value,
        ensemble_score=response.model_scores.ensemble,
        if_score=response.model_scores.isolation_forest,
        lof_score=response.model_scores.lof,
        rules_triggered=response.rules_triggered,
        inference_ms=response.processing_ms,
        rule_ms=0, total_ms=total_ms, amount=txn.amount,
    )
    _update_anomaly_rate(response.is_anomaly)
    return response


@app.post("/predict/batch", response_model=BatchAnomalyResponse, tags=["Prediction"])
async def predict_batch(request: BatchTransactionRequest):
    if not detector.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded. Run ml/train.py first.")
    BATCH_SIZE_HISTOGRAM.observe(len(request.transactions))
    t0      = time.perf_counter()
    results = []
    for txn in request.transactions:
        response = detector.predict(txn)
        results.append(response)
        record_prediction(
            is_anomaly=response.is_anomaly, anomaly_type=response.anomaly_type.value,
            ensemble_score=response.model_scores.ensemble,
            if_score=response.model_scores.isolation_forest,
            lof_score=response.model_scores.lof,
            rules_triggered=response.rules_triggered,
            inference_ms=response.processing_ms,
            rule_ms=0, total_ms=response.processing_ms, amount=txn.amount,
        )
        _update_anomaly_rate(response.is_anomaly)
    total_ms      = (time.perf_counter() - t0) * 1000
    anomaly_count = sum(1 for r in results if r.is_anomaly)
    return BatchAnomalyResponse(
        results=results, total=len(results),
        anomaly_count=anomaly_count,
        anomaly_rate=round(anomaly_count / len(results), 4),
        processing_ms=round(total_ms, 2),
    )
