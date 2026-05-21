
# Credit Card Anomaly Detection (CCAD)

### Project Brief & Business Use Case

## Summary

This project is an end-to-end, production-ready Machine Learning system designed to detect fraudulent credit card transactions in real-time. This implements a robust, containerised architecture capable of handling **1,000,000 rows** of transaction data, mirroring how tier-1 fintech companies can deploy ML models to production.

The system relies on a **3-model ensemble** (Isolation Forest, Local Outlier Factor, and a PyTorch TabNet Autoencoder) trained entirely on normal transactions (unsupervised learning) to flag deviations. It is wrapped in a highly-optimised FastAPI backend, instrumented with Prometheus, and visualised via a live Grafana MLOps dashboard. End-users and Data Scientists interact with the system via a stunning, multi-tab **Streamlit** portal.

## The Problem

Credit card fraud costs the global economy billions annually. Traditional rule-based systems are brittle and generate too many false positives, leading to blocked cards and frustrated customers. Furthermore, supervised ML models struggle because fraud patterns constantly evolve (concept drift), and labelled fraud data is extremely scarce and heavily imbalanced.

## The Solution

We treat fraud detection as an **unsupervised anomaly detection** problem.
By training our models exclusively on *normal* user spending behaviour, the system learns the complex, multidimensional signature of a legitimate transaction. When a fraudster attempts a transaction using a stolen card, it will naturally deviate from this learned signature.

### The 3-Model Ensemble

Our ensemble approach covers all blind spots, heavily optimized to train on 1M rows without bottlenecking system memory:

1. **Isolation Forest:** Catches global anomalies (e.g., massive purchase amounts relative to the entire dataset).
2. **Local Outlier Factor (LOF):** Catches local density anomalies (e.g., a transaction that blends in globally, but is highly unusual for that specific user's geographic region). *Note: Optimized with intelligent downsampling to bypass its native $O(n^2)$ memory limits.*
3. **TabNet Autoencoder (PyTorch):** Catches complex, non-linear feature interactions (e.g., a normal amount, but at an unusual hour, at a high-risk merchant, in a foreign country).

### Explainable AI (Feature Attribution)

Black-box models are unacceptable in the financial sector. During inference, the API calculates the standard deviation (`σ`) of every feature against the global mean using the `StandardScaler`. Any feature deviating by more than `2.5σ` is instantly extracted and translated into a plain-English explanation for the end-user (e.g., *"The AI flagged this because the amount is unusually high compared to typical spending, and the transaction occurred at an unusual hour."*). This provides true Feature Attribution natively without needing heavy libraries like SHAP that would bottleneck the 15ms inference latency.

## Primary Business Use Cases

### 1. Live Transaction Authorisation (The "Payment Gateway")

When a customer swipes their card, the payment gateway sends a payload to this API. The system evaluates the transaction against the user's historical baseline, scores it through the 3-model ensemble, checks it against deterministic fraud rules, and returns an `Allow` or `Block` decision—all in under 15 milliseconds. This is visualized in real-time on the **Streamlit Live Detector** app.

### 2. High-Throughput Risk Audits

Banks evaluate millions of transactions overnight to catch "sleeper" fraud (e.g., card testing where fraudsters make tiny $1.00 transactions to see if a stolen card is active). The system is load-tested to handle **100,000 concurrent simulated requests**, utilizing the `/predict/batch` endpoint to allow risk teams to push thousands of transactions through the system asynchronously.

### 3. MLOps & System Observability

Model drift is a major issue in production. Every single prediction made by the API is scraped by **Prometheus** every 10 seconds. This data feeds into a live **Grafana** dashboard that tracks the P99 latency of the model, the current rolling anomaly detection rate, and real-time score distributions. If the model starts degrading, engineers know instantly.

## How to Interact With It

You can run this entire ecosystem on your local machine or server with a single command: `docker compose up -d --build`.

1. **Streamlit Portal (Port 8502):** The unified app where you can simulate live transactions, view ROC evaluation plots, and explore the 1,000,000 row dataset.
2. **Grafana Dashboard (Port 3000):** Watch the dials spin as you send traffic through the API, monitoring the system's P99 latency and anomaly rates.
3. **FastAPI Swagger (Port 8000/docs):** Test the raw endpoints programmatically.
