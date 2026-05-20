import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from api.schemas.transaction import AnomalyResponse, AnomalyType, ModelScores

MOCK = AnomalyResponse(
    transaction_id="T1", is_anomaly=True, anomaly_score=0.85,
    anomaly_type=AnomalyType.geographic, confidence="high",
    model_scores=ModelScores(isolation_forest=0.80, lof=0.92, ensemble=0.85),
    rules_triggered=[], explanation="Test.", processing_ms=3.5,
)

TXN = {
    "transaction_id":"T1","user_id":42,"amount":1250.0,"mcc":5944,
    "country":"CN","home_country":"DE","hour":3,"day_of_week":1,
    "is_card_present":0,"credit_limit":5000.0,"avg_user_spend":45.0,
}

@pytest.fixture
def client():
    from api.main import app
    with patch("api.main.detector") as m:
        m.is_loaded=True; m.predict.return_value=MOCK
        m.feature_cols=["log_amount"]; m.thresholds={"ensemble":0.5}; m.evaluation={}
        with TestClient(app) as c: yield c

class TestHealth:
    def test_200(self, client):          assert client.get("/health").status_code == 200
    def test_loaded(self, client):       assert client.get("/health").json()["model_loaded"] is True

class TestPredict:
    def test_200(self, client):          assert client.post("/predict", json=TXN).status_code == 200
    def test_fields(self, client):
        d = client.post("/predict", json=TXN).json()
        assert all(k in d for k in ["is_anomaly","anomaly_score","model_scores","explanation"])
    def test_bad_amount(self, client):   assert client.post("/predict", json={**TXN,"amount":-1}).status_code == 422
    def test_bad_hour(self, client):     assert client.post("/predict", json={**TXN,"hour":25}).status_code == 422

class TestBatch:
    def test_batch_ok(self, client):
        r = client.post("/predict/batch", json={"transactions":[TXN,{**TXN,"transaction_id":"T2"}]})
        assert r.status_code==200 and r.json()["total"]==2
    def test_empty_rejected(self, client):   assert client.post("/predict/batch",json={"transactions":[]}).status_code==422
    def test_too_large_rejected(self, client): assert client.post("/predict/batch",json={"transactions":[TXN]*101}).status_code==422

class TestMetrics:
    def test_200(self, client):          assert client.get("/metrics").status_code == 200
