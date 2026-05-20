from prometheus_client import Counter, Histogram, Gauge

PREDICTIONS_TOTAL = Counter(
    "anomaly_predictions_total",
    "Total transactions scored",
    ["result"],
)
PREDICTIONS_BY_TYPE = Counter(
    "anomaly_predictions_by_type_total",
    "Predictions by anomaly type",
    ["anomaly_type"],
)
ANOMALY_SCORE_HISTOGRAM = Histogram(
    "anomaly_score_distribution",
    "Ensemble anomaly score distribution",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
RULE_TRIGGERS_TOTAL = Counter(
    "fraud_rule_triggers_total",
    "Fraud rule trigger counts",
    ["rule_name", "severity"],
)
RULES_PER_TRANSACTION = Histogram(
    "fraud_rules_per_transaction",
    "Rules triggered per transaction",
    buckets=[0, 1, 2, 3, 4, 5, 6, 7],
)
INFERENCE_LATENCY = Histogram(
    "model_inference_latency_seconds",
    "Model inference time",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
TOTAL_REQUEST_LATENCY = Histogram(
    "prediction_request_latency_seconds",
    "End-to-end request latency",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
)
ANOMALY_RATE_GAUGE = Gauge(
    "current_anomaly_rate",
    "Rolling anomaly rate over last 1000 transactions",
)
BATCH_SIZE_HISTOGRAM = Histogram(
    "batch_request_size",
    "Transactions per batch request",
    buckets=[1, 5, 10, 25, 50, 75, 100],
)
MODEL_LOADED_GAUGE = Gauge(
    "model_loaded",
    "1 if models are loaded, 0 otherwise",
)
TRANSACTION_AMOUNT_HISTOGRAM = Histogram(
    "transaction_amount_eur",
    "Transaction amounts in EUR",
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000, 5000, 10000],
)


def record_prediction(
    is_anomaly, anomaly_type, ensemble_score, if_score,
    lof_score, rules_triggered, inference_ms, rule_ms, total_ms, amount,
):
    result = "anomaly" if is_anomaly else "normal"
    PREDICTIONS_TOTAL.labels(result=result).inc()
    PREDICTIONS_BY_TYPE.labels(anomaly_type=anomaly_type).inc()
    ANOMALY_SCORE_HISTOGRAM.observe(ensemble_score)
    RULES_PER_TRANSACTION.observe(len(rules_triggered))
    for rule in rules_triggered:
        RULE_TRIGGERS_TOTAL.labels(rule_name=rule.rule_name, severity=rule.severity).inc()
    INFERENCE_LATENCY.observe(inference_ms / 1000)
    TOTAL_REQUEST_LATENCY.observe(total_ms / 1000)
    TRANSACTION_AMOUNT_HISTOGRAM.observe(amount)
