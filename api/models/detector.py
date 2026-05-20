"""
api/models/detector.py
───────────────────────
Loads all three trained model artifacts and scores incoming transactions.

Models loaded at startup:
  - Isolation Forest  (sklearn)
  - Local Outlier Factor (sklearn)
  - TabNet Autoencoder  (PyTorch)

Ensemble scoring:
  final_score = 0.35*IF + 0.35*LOF + 0.30*TabNet
  + rule boost (up to +0.25 from fraud rule engine)
"""

import json
import math
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from api.schemas.transaction import (
    TransactionRequest, AnomalyResponse, AnomalyType, ModelScores, RuleTrigger,
)
from api.rules.fraud_rules import evaluate_rules, rule_severity_score

ARTIFACT_DIR        = Path(__file__).parent.parent.parent / "ml" / "artifacts"
HIGH_RISK_COUNTRIES = {"NG", "CN", "RU", "UA", "BR", "VN", "PK", "ID", "RO", "BG"}
RULE_BOOST_FACTOR   = 0.25

DEVICE = (
    torch.device("mps")  if torch.backends.mps.is_available() else
    torch.device("cuda") if torch.cuda.is_available()          else
    torch.device("cpu")
)


# ── TabNet architecture (must match train.py exactly) ─────────────────────────

class TabNetEncoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.input_dim   = config["input_dim"]
        self.hidden_dims = config["hidden_dims"]
        self.n_steps     = config["n_steps"]
        self.dropout     = nn.Dropout(config["dropout"])
        self.initial_bn  = nn.BatchNorm1d(self.input_dim)
        step_out_dim     = self.hidden_dims[0]
        self.steps       = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_dim, step_out_dim * 2),
                nn.BatchNorm1d(step_out_dim * 2),
            )
            for _ in range(self.n_steps)
        ])
        self.step_attn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_dim, self.input_dim),
                nn.BatchNorm1d(self.input_dim),
                nn.Softmax(dim=-1),
            )
            for _ in range(self.n_steps)
        ])
        layers = []
        in_dim = step_out_dim
        for out_dim in self.hidden_dims[1:]:
            layers += [
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(config["dropout"]),
            ]
            in_dim = out_dim
        self.deep = nn.Sequential(*layers)

    def forward(self, x):
        x   = self.initial_bn(x)
        agg = torch.zeros(x.size(0), self.hidden_dims[0], device=x.device)
        prior = torch.ones(x.size(0), self.input_dim, device=x.device)
        for step, attn in zip(self.steps, self.step_attn):
            mask  = attn(x * prior)
            prior = prior * (1 - mask + 1e-8)
            h     = step(x * mask)
            h     = torch.relu(h[:, :self.hidden_dims[0]]) + \
                    torch.sigmoid(h[:, self.hidden_dims[0]:]) * h[:, :self.hidden_dims[0]]
            agg   = agg + h / self.n_steps
        return self.deep(self.dropout(agg))


class TabNetDecoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        dims   = list(reversed(config["hidden_dims"]))
        layers = []
        for i in range(len(dims) - 1):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(config["dropout"]),
            ]
        layers += [nn.Linear(dims[-1], config["input_dim"])]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class TabNetAutoencoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.encoder = TabNetEncoder(config)
        self.decoder = TabNetDecoder(config)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    def reconstruction_error(self, x: torch.Tensor) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            recon, _ = self(x)
            mse      = ((x - recon) ** 2).mean(dim=1)
        return mse.cpu().numpy()


# ── Detector ──────────────────────────────────────────────────────────────────

class AnomalyDetector:

    def __init__(self):
        self.if_model      = None
        self.lof_model     = None
        self.tabnet        = None
        self.scaler        = None
        self.feature_cols  = None
        self.thresholds    = None
        self.evaluation    = None
        self.tabnet_config = None
        self._loaded       = False

        # Store training-time reconstruction error stats for normalisation
        self._tab_p1  = 0.0
        self._tab_p99 = 1.0

    def load(self):
        if self._loaded:
            return
        try:
            with open(ARTIFACT_DIR / "isolation_forest.pkl", "rb") as f:
                self.if_model = pickle.load(f)
            with open(ARTIFACT_DIR / "lof.pkl", "rb") as f:
                self.lof_model = pickle.load(f)
            with open(ARTIFACT_DIR / "scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)
            with open(ARTIFACT_DIR / "feature_cols.json") as f:
                self.feature_cols = json.load(f)
            with open(ARTIFACT_DIR / "thresholds.json") as f:
                self.thresholds = json.load(f)
            with open(ARTIFACT_DIR / "evaluation_report.json") as f:
                self.evaluation = json.load(f)
            with open(ARTIFACT_DIR / "tabnet_config.json") as f:
                self.tabnet_config = json.load(f)

            # Load TabNet
            self.tabnet = TabNetAutoencoder(self.tabnet_config).to(DEVICE)
            self.tabnet.encoder.load_state_dict(
                torch.load(ARTIFACT_DIR / "tabnet_encoder.pt", map_location=DEVICE, weights_only=True)
            )
            self.tabnet.decoder.load_state_dict(
                torch.load(ARTIFACT_DIR / "tabnet_decoder.pt", map_location=DEVICE, weights_only=True)
            )
            self.tabnet.eval()

            self._loaded = True
            print(f"  Models loaded on {DEVICE}.")

        except FileNotFoundError as e:
            raise RuntimeError(
                f"Artifacts not found at {ARTIFACT_DIR}. "
                f"Run 'python ml/train.py' first. Missing: {e.filename}"
            )

    @property
    def is_loaded(self):
        return self._loaded

    # ── Feature engineering ───────────────────────────────────────────────────

    def _build_features(self, txn: TransactionRequest) -> np.ndarray:
        mcc_risk = {
            5411:0.1, 5812:0.1, 5541:0.2, 5999:0.4, 4722:0.3,
            7922:0.2, 6011:0.6, 5912:0.1, 5734:0.5, 5944:0.7,
        }
        hour_bin = 0 if txn.hour <= 5 else 1 if txn.hour <= 11 else 2 if txn.hour <= 17 else 3
        return np.array([
            math.log1p(txn.amount),
            txn.amount / (txn.avg_user_spend + 1e-6),
            txn.amount / (txn.credit_limit + 1e-6),
            int(txn.country != txn.home_country),
            txn.hour, hour_bin,
            int(txn.day_of_week >= 5),
            txn.is_card_present,
            mcc_risk.get(txn.mcc, 0.3),
            txn.day_of_week,
        ], dtype=float).reshape(1, -1)

    # ── Per-model scoring ─────────────────────────────────────────────────────

    def _score_if(self, X_scaled: np.ndarray) -> float:
        raw = float(self.if_model.score_samples(X_scaled)[0])
        return float(np.clip(0.5 - raw, 0, 1))

    def _score_lof(self, X_scaled: np.ndarray) -> float:
        raw = float(self.lof_model.score_samples(X_scaled)[0])
        return float(np.clip((-raw) / 2.0, 0, 1))

    def _score_tabnet(self, X_scaled: np.ndarray) -> float:
        x_t    = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)
        error  = float(self.tabnet.reconstruction_error(x_t)[0])
        # Normalise using stored percentile bounds (fallback: clip to [0,1])
        p1, p99 = self._tab_p1, self._tab_p99
        if p99 == p1:
            return 0.0
        return float(np.clip((error - p1) / (p99 - p1), 0, 1))

    # ── Classification ────────────────────────────────────────────────────────

    def _classify_type(self, txn, rules, score) -> AnomalyType:
        rule_names = {r.rule_name for r in rules}
        if "atm_foreign_country" in rule_names or "foreign_transaction" in rule_names:
            return AnomalyType.geographic
        if "high_value_ratio" in rule_names or "near_credit_limit" in rule_names:
            return AnomalyType.high_value
        if "odd_hour_large_transaction" in rule_names:
            return AnomalyType.odd_hour
        if txn.country != txn.home_country:
            return AnomalyType.geographic
        if score >= self.thresholds.get("ensemble", 0.5):
            return AnomalyType.unknown
        return AnomalyType.normal

    def _explain(self, txn, is_anom, score, rules, anom_type) -> str:
        if not is_anom:
            return f"Transaction appears normal. Ensemble score: {score:.2f}."
        parts = [f"Anomaly detected (score: {score:.2f})."]
        high = [r for r in rules if r.severity == "high"]
        med  = [r for r in rules if r.severity == "medium"]
        if high:
            parts.append(f"High-severity rules: {', '.join(r.rule_name for r in high)}.")
        if med:
            parts.append(f"Medium-severity rules: {', '.join(r.rule_name for r in med)}.")
        if anom_type == AnomalyType.geographic:
            parts.append(f"Geographic anomaly: {txn.country} vs home {txn.home_country}.")
        elif anom_type == AnomalyType.high_value:
            parts.append(f"Amount is {txn.amount/(txn.avg_user_spend+1e-6):.0f}x user avg.")
        elif anom_type == AnomalyType.odd_hour:
            parts.append(f"Unusual time: {txn.hour:02d}:xx.")
        return " ".join(parts)

    # ── Main predict ──────────────────────────────────────────────────────────

    def predict(self, txn: TransactionRequest) -> AnomalyResponse:
        if not self._loaded:
            raise RuntimeError("Models not loaded.")

        t0       = time.perf_counter()
        X        = self._build_features(txn)
        X_scaled = self.scaler.transform(X)

        if_s  = self._score_if(X_scaled)
        lof_s = self._score_lof(X_scaled)
        tab_s = self._score_tabnet(X_scaled)
        ens   = 0.35 * if_s + 0.35 * lof_s + 0.30 * tab_s

        rules       = evaluate_rules(txn)
        boost       = RULE_BOOST_FACTOR * rule_severity_score(rules)
        final_score = min(1.0, ens + boost)

        threshold  = self.thresholds.get("ensemble", 0.5)
        is_anomaly = final_score >= threshold
        anom_type  = self._classify_type(txn, rules, final_score)
        confidence = "high" if final_score >= 0.75 else "medium" if final_score >= 0.50 else "low"
        total_ms   = (time.perf_counter() - t0) * 1000

        return AnomalyResponse(
            transaction_id  = txn.transaction_id,
            is_anomaly      = is_anomaly,
            anomaly_score   = round(final_score, 4),
            anomaly_type    = anom_type if is_anomaly else AnomalyType.normal,
            confidence      = confidence,
            model_scores    = ModelScores(
                isolation_forest = round(if_s,  4),
                lof              = round(lof_s, 4),
                tabnet           = round(tab_s, 4),
                ensemble         = round(ens,   4),
            ),
            rules_triggered = rules,
            explanation     = self._explain(txn, is_anomaly, final_score, rules, anom_type),
            processing_ms   = round(total_ms, 2),
        )


detector = AnomalyDetector()