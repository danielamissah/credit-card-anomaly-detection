"""
ml/train.py
────────────
Trains a 3-model ensemble for credit card anomaly detection
using 5-fold cross-validation with early stopping.

Models
──────
1. Isolation Forest  — tree-based unsupervised outlier detection
2. Local Outlier Factor — density-based local anomaly detection
3. TabNet Autoencoder — deep learning model trained to reconstruct
   normal transactions; high reconstruction error = anomaly

Why TabNet for tabular anomaly detection?
  TabNet (Arik & Pfister, 2021) uses sequential attention to select
  relevant features at each decision step — it does not treat all
  features equally. For fraud detection where only a subset of
  features drive anomalous behaviour (e.g. amount + country but not
  day_of_week), this is a genuine advantage over MLPs.

  We use it as an AUTOENCODER: train encoder+decoder on normal data,
  then use reconstruction error as the anomaly score at inference.
  High error = the transaction does not look like anything the model
  has seen during training = anomaly.

Convergence strategy
────────────────────
TabNet trains until early stopping fires — validation reconstruction
loss must improve by at least MIN_DELTA within PATIENCE epochs,
otherwise training stops. No fixed epoch budget.

Ensemble weights
────────────────
  Isolation Forest  : 0.35
  LOF               : 0.35
  TabNet            : 0.30

5-Fold Cross-Validation
───────────────────────
Each fold:
  - Train split: normal transactions only (unsupervised)
  - Val split  : full mix of normal + anomalous
  - All 3 models fitted on train normals
  - Scored and evaluated on val set
  - Per-fold F1, precision, recall, AUC-ROC reported

Final models: retrained on ALL normal data using mean CV thresholds.

Artifacts saved to ml/artifacts/
  isolation_forest.pkl
  lof.pkl
  tabnet_encoder.pt       ← TabNet encoder state dict
  tabnet_decoder.pt       ← TabNet decoder state dict
  tabnet_config.json      ← architecture hyperparameters
  scaler.pkl
  feature_cols.json
  thresholds.json
  evaluation_report.json
  cv_summary.json

Run:
  python ml/train.py
"""

import json
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Paths 

ROOT_DIR     = Path(__file__).parent.parent
DATA_DIR     = ROOT_DIR / "data"
ARTIFACT_DIR = Path(__file__).parent / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

# ── Feature config 

FEATURE_COLS = [
    "log_amount",
    "amount_to_avg_ratio",
    "amount_to_limit_ratio",
    "is_foreign",
    "hour",
    "hour_bin",
    "is_weekend",
    "is_card_present",
    "mcc_risk_score",
    "day_of_week",
]
N_FEATURES = len(FEATURE_COLS)

# ── CV config 

N_FOLDS     = 5
RANDOM_SEED = 42

# ── Ensemble weights 

W_IF     = 0.35
W_LOF    = 0.35
W_TABNET = 0.30

# ── sklearn model params 

IF_PARAMS = {
    "n_estimators":  200,
    "contamination": 0.05,
    "max_samples":   "auto",
    "random_state":  RANDOM_SEED,
    "n_jobs":        -1,
}

LOF_PARAMS = {
    "n_neighbors": 20,
    "contamination": 0.05,
    "novelty":     True,
    "n_jobs":      -1,
}

# ── TabNet autoencoder hyperparameters 

TABNET_CONFIG = {
    "input_dim":    N_FEATURES,
    "hidden_dims":  [64, 32, 16],   # encoder layer sizes
    "n_steps":      3,              # TabNet attention steps
    "dropout":      0.1,
    "batch_size":   512,
    "lr":           1e-3,
    "patience":     20,             # early stopping patience (epochs)
    "min_delta":    1e-5,           # minimum improvement to reset patience
    "val_frac":     0.15,           # fraction of normal data held out for ES
    "weight_decay": 1e-5,
}

# ── Device 

DEVICE = (
    torch.device("mps")  if torch.backends.mps.is_available() else
    torch.device("cuda") if torch.cuda.is_available()          else
    torch.device("cpu")
)


# ══════════════════════════════════════════════════════════════════════════════
# TabNet Autoencoder Architecture
# ══════════════════════════════════════════════════════════════════════════════

class TabNetAttentionStep(nn.Module):
    """
    Single attention step in TabNet.
    Learns a soft mask over features at each step — forces the model
    to justify which features it uses rather than attending to all of them.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Softmax(dim=-1),          # attention weights sum to 1
        )
        self.transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GLU(dim=-1) if hidden_dim % 2 == 0 else nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, prior_mask: torch.Tensor):
        # Attention mask weighted by prior (penalises reuse of same features)
        mask       = self.attention(x * prior_mask)
        masked_x   = x * mask
        out_dim    = self.transform[0].out_features
        # Safe transform — handle GLU halving output dim
        h          = self.transform[0](masked_x)
        h          = self.transform[1](h)
        if h.shape[-1] != out_dim // 2:
            h = h
        return h, mask


class TabNetEncoder(nn.Module):
    """
    TabNet encoder with sequential attention steps.
    Each step attends to a different subset of features.
    Final representation is the sum of all step outputs.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.input_dim   = config["input_dim"]
        self.hidden_dims = config["hidden_dims"]
        self.n_steps     = config["n_steps"]
        self.dropout     = nn.Dropout(config["dropout"])

        # Initial batch norm on raw features
        self.initial_bn  = nn.BatchNorm1d(self.input_dim)

        # Build attention steps — each maps input → hidden_dims[0]
        step_out_dim     = self.hidden_dims[0]
        self.steps       = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_dim, step_out_dim * 2),
                nn.BatchNorm1d(step_out_dim * 2),
            )
            for _ in range(self.n_steps)
        ])
        self.step_attn   = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_dim, self.input_dim),
                nn.BatchNorm1d(self.input_dim),
                nn.Softmax(dim=-1),
            )
            for _ in range(self.n_steps)
        ])

        # Deep layers after attention aggregation
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
        self.latent_dim = in_dim

    def forward(self, x: torch.Tensor):
        x   = self.initial_bn(x)
        agg = torch.zeros(x.size(0), self.hidden_dims[0], device=x.device)

        # Prior scales — discourages repeated attention to same features
        prior = torch.ones(x.size(0), self.input_dim, device=x.device)

        for step, attn in zip(self.steps, self.step_attn):
            mask   = attn(x * prior)
            prior  = prior * (1 - mask + 1e-8)         # update prior
            h      = step(x * mask)                    # shape: [B, step_out*2]
            h      = torch.relu(h[:, :self.hidden_dims[0]]) + \
                     torch.sigmoid(h[:, self.hidden_dims[0]:]) * h[:, :self.hidden_dims[0]]
            agg    = agg + h / self.n_steps

        agg = self.dropout(agg)
        return self.deep(agg)


class TabNetDecoder(nn.Module):
    """
    Decoder mirrors the encoder in reverse.
    Reconstructs the original feature vector from the latent representation.
    """

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

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class TabNetAutoencoder(nn.Module):
    """Full autoencoder = encoder + decoder."""

    def __init__(self, config: dict):
        super().__init__()
        self.encoder = TabNetEncoder(config)
        self.decoder = TabNetDecoder(config)

    def forward(self, x: torch.Tensor):
        z    = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

    def reconstruction_error(self, x: torch.Tensor) -> np.ndarray:
        """Per-sample mean squared reconstruction error."""
        self.eval()
        with torch.no_grad():
            recon, _ = self(x)
            mse      = ((x - recon) ** 2).mean(dim=1)
        return mse.cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
# TabNet Training with Early Stopping
# ══════════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    """
    Stops training when validation loss stops improving.
    Saves the best model state so we always return the best checkpoint.
    """

    def __init__(self, patience: int, min_delta: float):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_state = None
        self.stopped_epoch = 0

    def step(self, val_loss: float, model: nn.Module, epoch: int) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped_epoch = epoch
                return True   # stop training
        return False

    def restore_best(self, model: nn.Module):
        if self.best_state:
            model.load_state_dict(self.best_state)


def train_tabnet(
    X_train_norm: np.ndarray,
    config: dict,
    fold_idx: int = 0,
    verbose: bool = True,
) -> TabNetAutoencoder:
    """
    Train TabNet autoencoder on normal transactions until early stopping.

    Args:
        X_train_norm : scaled normal training data, shape [N, n_features]
        config       : hyperparameter dict (TABNET_CONFIG)
        fold_idx     : current CV fold (for logging)
        verbose      : print epoch logs

    Returns:
        Trained TabNetAutoencoder with best weights restored.
    """
    # Split off a validation set for early stopping
    n_val        = max(1, int(len(X_train_norm) * config["val_frac"]))
    idx          = np.random.RandomState(RANDOM_SEED + fold_idx).permutation(len(X_train_norm))
    val_idx      = idx[:n_val]
    train_idx    = idx[n_val:]

    X_tr = torch.tensor(X_train_norm[train_idx], dtype=torch.float32).to(DEVICE)
    X_va = torch.tensor(X_train_norm[val_idx],   dtype=torch.float32).to(DEVICE)

    model     = TabNetAutoencoder(config).to(DEVICE)
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr           = config["lr"],
        weight_decay = config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )
    criterion  = nn.MSELoss()
    es         = EarlyStopping(config["patience"], config["min_delta"])
    batch_size = config["batch_size"]
    dataset    = torch.utils.data.TensorDataset(X_tr)
    loader     = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if verbose:
        print(f"    TabNet training | train:{len(X_tr):,} val:{len(X_va):,} | device:{DEVICE}")
        print(f"    Early stopping: patience={config['patience']} min_delta={config['min_delta']}")

    epoch      = 0
    t_start    = time.time()

    while True:
        epoch += 1
        model.train()
        train_loss = 0.0
        for (batch,) in loader:
            optimiser.zero_grad()
            recon, _   = model(batch)
            loss       = criterion(recon, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            train_loss += loss.item() * len(batch)
        train_loss /= len(X_tr)

        # Validation loss
        model.eval()
        with torch.no_grad():
            recon_val, _ = model(X_va)
            val_loss     = criterion(recon_val, X_va).item()

        scheduler.step(val_loss)

        if verbose and (epoch % 10 == 0 or epoch <= 5):
            elapsed = time.time() - t_start
            lr_now  = optimiser.param_groups[0]["lr"]
            print(f"    Epoch {epoch:4d} | train_loss:{train_loss:.6f} | val_loss:{val_loss:.6f} | lr:{lr_now:.2e} | {elapsed:.1f}s")

        if es.step(val_loss, model, epoch):
            if verbose:
                print(f"    Early stopping at epoch {epoch} | best val_loss:{es.best_loss:.6f}")
            break

    es.restore_best(model)
    if verbose:
        print(f"    Converged at epoch {epoch} | best val_loss: {es.best_loss:.6f}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Scoring helpers
# ══════════════════════════════════════════════════════════════════════════════

def normalise_if(raw: np.ndarray) -> np.ndarray:
    return np.clip(0.5 - raw, 0, 1)

def normalise_lof(raw: np.ndarray) -> np.ndarray:
    return np.clip((-raw) / 2.0, 0, 1)

def normalise_tabnet(errors: np.ndarray) -> np.ndarray:
    """Normalise reconstruction errors to [0,1] using percentile clipping."""
    p1, p99 = np.percentile(errors, 1), np.percentile(errors, 99)
    clipped  = np.clip(errors, p1, p99)
    if p99 == p1:
        return np.zeros_like(clipped)
    return (clipped - p1) / (p99 - p1)

def ensemble(if_s, lof_s, tab_s) -> np.ndarray:
    return W_IF * if_s + W_LOF * lof_s + W_TABNET * tab_s

def tune_threshold(scores: np.ndarray, y_true: np.ndarray) -> float:
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.2, 0.98, 0.02):
        f1 = f1_score(y_true, (scores >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(round(best_t, 2))

def score_fold(scores: np.ndarray, y_true: np.ndarray, threshold: float) -> dict:
    preds = (scores >= threshold).astype(int)
    try:    auc = float(roc_auc_score(y_true, scores))
    except: auc = 0.0
    return {
        "f1":        round(f1_score(y_true, preds, zero_division=0), 4),
        "precision": round(precision_score(y_true, preds, zero_division=0), 4),
        "recall":    round(recall_score(y_true, preds, zero_division=0), 4),
        "auc_roc":   round(auc, 4),
        "threshold": threshold,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5-Fold Cross-Validation
# ══════════════════════════════════════════════════════════════════════════════

def run_cross_validation(full_df: pd.DataFrame) -> dict:
    print(f"\n{'='*60}")
    print(f"  5-Fold Cross-Validation  (IF + LOF + TabNet Ensemble)")
    print(f"{'='*60}")

    X_all = full_df[FEATURE_COLS].fillna(0).values
    y_all = full_df["is_anomaly"].values
    skf   = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    fold_results = {
        "isolation_forest": [],
        "lof":              [],
        "tabnet":           [],
        "ensemble":         [],
    }

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all), start=1):
        print(f"\n  ── Fold {fold_idx}/{N_FOLDS} {'─'*45}")

        X_train_fold = X_all[train_idx]
        y_train_fold = y_all[train_idx]
        X_val        = X_all[val_idx]
        y_val        = y_all[val_idx]

        # Only normals for unsupervised training
        normal_mask  = y_train_fold == 0
        X_norm       = X_train_fold[normal_mask]

        print(f"  Train normals: {normal_mask.sum():,} | Val: {len(y_val):,} | Val anomalies: {y_val.sum():,}")

        # Scaler fitted on normals only
        scaler     = StandardScaler()
        X_norm_sc  = scaler.fit_transform(X_norm)
        X_val_sc   = scaler.transform(X_val)

        # ── Isolation Forest ───────────────────────────────────────────────────
        print(f"\n  [IF] Training ...")
        if_model = IsolationForest(**IF_PARAMS)
        if_model.fit(X_norm_sc)
        if_scores  = normalise_if(if_model.score_samples(X_val_sc))

        # ── LOF ────────────────────────────────────────────────────────────────
        print(f"  [LOF] Training ...")
        lof_model = LocalOutlierFactor(**LOF_PARAMS)
        if len(X_norm_sc) > 100_000:
            idx = np.random.choice(len(X_norm_sc), 100_000, replace=False)
            lof_model.fit(X_norm_sc[idx])
        else:
            lof_model.fit(X_norm_sc)
        lof_scores = normalise_lof(lof_model.score_samples(X_val_sc))

        # ── TabNet ─────────────────────────────────────────────────────────────
        print(f"\n  [TabNet] Training until convergence ...")
        tabnet_model  = train_tabnet(X_norm_sc, TABNET_CONFIG, fold_idx=fold_idx)
        X_val_tensor  = torch.tensor(X_val_sc, dtype=torch.float32).to(DEVICE)
        tab_errors    = tabnet_model.reconstruction_error(X_val_tensor)
        tab_scores    = normalise_tabnet(tab_errors)

        # ── Ensemble ───────────────────────────────────────────────────────────
        ens_scores = ensemble(if_scores, lof_scores, tab_scores)

        # ── Thresholds ─────────────────────────────────────────────────────────
        t_if  = tune_threshold(if_scores,  y_val)
        t_lof = tune_threshold(lof_scores, y_val)
        t_tab = tune_threshold(tab_scores, y_val)
        t_ens = tune_threshold(ens_scores, y_val)

        # ── Results ────────────────────────────────────────────────────────────
        r_if  = score_fold(if_scores,  y_val, t_if)
        r_lof = score_fold(lof_scores, y_val, t_lof)
        r_tab = score_fold(tab_scores, y_val, t_tab)
        r_ens = score_fold(ens_scores, y_val, t_ens)

        fold_results["isolation_forest"].append(r_if)
        fold_results["lof"].append(r_lof)
        fold_results["tabnet"].append(r_tab)
        fold_results["ensemble"].append(r_ens)

        print(f"\n  Fold {fold_idx} results:")
        print(f"    IF      → F1:{r_if['f1']:.3f}  P:{r_if['precision']:.3f}  R:{r_if['recall']:.3f}  AUC:{r_if['auc_roc']:.3f}  t={t_if}")
        print(f"    LOF     → F1:{r_lof['f1']:.3f}  P:{r_lof['precision']:.3f}  R:{r_lof['recall']:.3f}  AUC:{r_lof['auc_roc']:.3f}  t={t_lof}")
        print(f"    TabNet  → F1:{r_tab['f1']:.3f}  P:{r_tab['precision']:.3f}  R:{r_tab['recall']:.3f}  AUC:{r_tab['auc_roc']:.3f}  t={t_tab}")
        print(f"    Ensemble→ F1:{r_ens['f1']:.3f}  P:{r_ens['precision']:.3f}  R:{r_ens['recall']:.3f}  AUC:{r_ens['auc_roc']:.3f}  t={t_ens}")

    return fold_results


def aggregate_cv_results(fold_results: dict) -> dict:
    metrics = ["f1", "precision", "recall", "auc_roc", "threshold"]
    summary = {}

    print(f"\n{'='*60}")
    print(f"  CV Summary — mean ± std over {N_FOLDS} folds")
    print(f"{'='*60}")

    for model_name, folds in fold_results.items():
        summary[model_name] = {}
        print(f"\n  [{model_name}]")
        for metric in metrics:
            values = [f[metric] for f in folds]
            mean   = round(float(np.mean(values)), 4)
            std    = round(float(np.std(values)),  4)
            summary[model_name][metric]            = mean
            summary[model_name][f"{metric}_std"]   = std
            summary[model_name][f"{metric}_folds"] = [round(v, 4) for v in values]
            if metric != "threshold":
                print(f"    {metric:<12}: {mean:.4f} ± {std:.4f}   folds: {[round(v,3) for v in values]}")
        print(f"    {'threshold':<12}: {summary[model_name]['threshold']:.2f} ± {summary[model_name]['threshold_std']:.2f}")

    return summary


# ══════════════════════════════════════════════════════════════════════════════
# Final model training
# ══════════════════════════════════════════════════════════════════════════════

def train_final_models(full_df: pd.DataFrame) -> tuple:
    print(f"\n{'='*60}")
    print(f"  Training Final Models on All Normal Data")
    print(f"{'='*60}")

    normal_df  = full_df[full_df["is_anomaly"] == 0]
    X_norm     = normal_df[FEATURE_COLS].fillna(0).values
    print(f"  Normal samples: {len(X_norm):,}")

    scaler     = StandardScaler()
    X_scaled   = scaler.fit_transform(X_norm)

    print("\n  [IF] Fitting final Isolation Forest ...")
    if_model   = IsolationForest(**IF_PARAMS)
    if_model.fit(X_scaled)

    print("  [LOF] Fitting final Local Outlier Factor ...")
    lof_model  = LocalOutlierFactor(**LOF_PARAMS)
    if len(X_scaled) > 100_000:
        idx = np.random.choice(len(X_scaled), 100_000, replace=False)
        lof_model.fit(X_scaled[idx])
    else:
        lof_model.fit(X_scaled)

    print("\n  [TabNet] Training final autoencoder until convergence ...")
    tabnet_model = train_tabnet(X_scaled, TABNET_CONFIG, fold_idx=99, verbose=True)

    return if_model, lof_model, tabnet_model, scaler


# ══════════════════════════════════════════════════════════════════════════════
# Save artifacts
# ══════════════════════════════════════════════════════════════════════════════

def save_artifacts(if_model, lof_model, tabnet_model, scaler, thresholds, eval_report, cv_summary):
    with open(ARTIFACT_DIR / "isolation_forest.pkl", "wb") as f:
        pickle.dump(if_model, f)
    with open(ARTIFACT_DIR / "lof.pkl", "wb") as f:
        pickle.dump(lof_model, f)
    with open(ARTIFACT_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Save TabNet state dicts separately (more portable than pickling the whole model)
    torch.save(tabnet_model.encoder.state_dict(), ARTIFACT_DIR / "tabnet_encoder.pt")
    torch.save(tabnet_model.decoder.state_dict(), ARTIFACT_DIR / "tabnet_decoder.pt")

    with open(ARTIFACT_DIR / "tabnet_config.json", "w") as f:
        json.dump(TABNET_CONFIG, f, indent=2)
    with open(ARTIFACT_DIR / "feature_cols.json", "w") as f:
        json.dump(FEATURE_COLS, f, indent=2)
    with open(ARTIFACT_DIR / "thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    with open(ARTIFACT_DIR / "evaluation_report.json", "w") as f:
        json.dump(eval_report, f, indent=2)
    with open(ARTIFACT_DIR / "cv_summary.json", "w") as f:
        json.dump(cv_summary, f, indent=2)

    print(f"\n  Artifacts saved to {ARTIFACT_DIR}/")
    for f in sorted(ARTIFACT_DIR.iterdir()):
        print(f"    {f.name:<38} {f.stat().st_size:>10,} bytes")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 60)
    print("  Credit Card Anomaly Detection — Training")
    print("  Models : Isolation Forest + LOF + TabNet Autoencoder")
    print(f"  CV     : {N_FOLDS}-fold stratified")
    print(f"  Device : {DEVICE}")
    print("=" * 60)

    # Load full dataset
    full_df = pd.read_csv(DATA_DIR / "transactions.csv")
    print(f"\n  Dataset: {len(full_df):,} rows | "
          f"Normal: {(full_df['is_anomaly']==0).sum():,} | "
          f"Anomalous: {(full_df['is_anomaly']==1).sum():,} | "
          f"Rate: {full_df['is_anomaly'].mean():.1%}")

    t0 = time.time()

    # Step 1: Cross-validation
    fold_results = run_cross_validation(full_df)

    # Step 2: Aggregate
    cv_summary = aggregate_cv_results(fold_results)

    # Step 3: Extract mean thresholds from CV
    thresholds = {
        model: round(cv_summary[model]["threshold"], 2)
        for model in cv_summary
    }
    print(f"\n  Mean CV thresholds: {thresholds}")

    # Step 4: Train final models on all normal data
    if_model, lof_model, tabnet_model, scaler = train_final_models(full_df)

    # Step 5: Holdout evaluation
    print(f"\n{'='*60}")
    print(f"  Holdout Evaluation")
    print(f"{'='*60}")

    test_df = pd.read_csv(DATA_DIR / "test.csv")
    X_test  = scaler.transform(test_df[FEATURE_COLS].fillna(0).values)
    y_test  = test_df["is_anomaly"].values

    if_scores  = normalise_if(if_model.score_samples(X_test))
    lof_scores = normalise_lof(lof_model.score_samples(X_test))

    X_test_t   = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    tab_errors = tabnet_model.reconstruction_error(X_test_t)
    tab_scores = normalise_tabnet(tab_errors)
    ens_scores = ensemble(if_scores, lof_scores, tab_scores)

    eval_report = {"cv_summary": cv_summary, "holdout": {}, "n_folds": N_FOLDS,
                   "ensemble_weights": {"isolation_forest": W_IF, "lof": W_LOF, "tabnet": W_TABNET}}

    all_scores = {
        "isolation_forest": if_scores,
        "lof":              lof_scores,
        "tabnet":           tab_scores,
        "ensemble":         ens_scores,
    }

    print(f"\n  Test: {len(y_test):,} rows | Anomaly rate: {y_test.mean():.1%}\n")
    for name, scores in all_scores.items():
        t     = thresholds.get(name, 0.5)
        preds = (scores >= t).astype(int)
        try:    auc = round(float(roc_auc_score(y_test, scores)), 4)
        except: auc = 0.0
        result = {
            "f1":        round(f1_score(y_test, preds, zero_division=0), 4),
            "precision": round(precision_score(y_test, preds, zero_division=0), 4),
            "recall":    round(recall_score(y_test, preds, zero_division=0), 4),
            "auc_roc":   auc,
            "threshold": t,
        }
        eval_report["holdout"][name] = result
        print(f"  [{name}]")
        print(f"    F1:{result['f1']:.4f}  P:{result['precision']:.4f}  R:{result['recall']:.4f}  AUC:{result['auc_roc']:.4f}")

    # Step 6: Save
    save_artifacts(if_model, lof_model, tabnet_model, scaler, thresholds, eval_report, cv_summary)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Training complete in {elapsed/60:.1f} minutes.")
    print(f"  Run: docker compose up -d")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()