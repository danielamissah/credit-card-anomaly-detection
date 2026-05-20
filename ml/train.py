"""
ml/train.py
────────────
Trains an ensemble anomaly detection model with 5-fold cross-validation.

Why cross-validation for anomaly detection?
  Standard k-fold CV doesn't directly apply to unsupervised models —
  we can't "fit on k-1 folds and evaluate on the kth fold" in the usual
  supervised sense. Instead we use a stratified approach:

  For each fold:
    - Training set: normal transactions only (unsupervised fitting)
    - Validation set: held-out mix of normal + anomalous transactions
    - Models are fit on training normals, scored on validation set
    - F1, precision, recall, AUC are computed per fold

  After 5 folds:
    - Mean and std of each metric are reported
    - This gives a honest estimate of generalisation performance
    - Final models are retrained on ALL normal data using the
      mean threshold from CV as the decision boundary

  This mirrors how unsupervised anomaly detection is evaluated in
  production ML systems at companies like Stripe, N26, and Adyen.

Outputs saved to ml/artifacts/:
  - isolation_forest.pkl      : final model trained on all normal data
  - lof.pkl                   : final model trained on all normal data
  - scaler.pkl                : fitted on all normal data
  - feature_cols.json         : ordered feature column list
  - thresholds.json           : mean CV thresholds per model
  - evaluation_report.json    : full CV results (per-fold + mean/std)
  - cv_summary.json           : clean summary for the API /model/info endpoint

Run:
  python ml/train.py
"""

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)

warnings.filterwarnings("ignore")

# ── Paths 

ROOT_DIR     = Path(__file__).parent.parent
DATA_DIR     = ROOT_DIR / "data"
ARTIFACT_DIR = Path(__file__).parent / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

# ── Config 

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

N_FOLDS            = 5
RANDOM_SEED        = 42
ENSEMBLE_WEIGHT_IF = 0.6
ENSEMBLE_WEIGHT_LOF = 0.4

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


# ── Scoring helpers 

def normalise_if(raw: np.ndarray) -> np.ndarray:
    """IF: more negative = more anomalous. Invert to [0,1]."""
    return np.clip(0.5 - raw, 0, 1)

def normalise_lof(raw: np.ndarray) -> np.ndarray:
    """LOF: more negative = more anomalous. Divide by 2 to roughly map to [0,1]."""
    return np.clip((-raw) / 2.0, 0, 1)

def ensemble_score(if_s: np.ndarray, lof_s: np.ndarray) -> np.ndarray:
    return ENSEMBLE_WEIGHT_IF * if_s + ENSEMBLE_WEIGHT_LOF * lof_s

def tune_threshold(scores: np.ndarray, y_true: np.ndarray) -> float:
    """Find threshold maximising F1 over a grid search."""
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


# ── Cross-validation 

def run_cross_validation(full_df: pd.DataFrame) -> dict:
    """
    5-fold stratified cross-validation for unsupervised anomaly detection.

    Strategy:
      - Stratify splits by is_anomaly label to ensure each fold has
        a representative mix of normal and anomalous samples
      - Train models on the NORMAL subset of the training fold only
      - Evaluate on the full validation fold (normal + anomalous)
      - Report per-fold and aggregate metrics
    """
    print(f"\n{'='*55}")
    print(f"  5-Fold Cross-Validation")
    print(f"{'='*55}")

    X_all = full_df[FEATURE_COLS].fillna(0).values
    y_all = full_df["is_anomaly"].values

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    fold_results = {
        "isolation_forest": [],
        "lof":              [],
        "ensemble":         [],
    }

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all), start=1):
        print(f"\n  Fold {fold_idx}/{N_FOLDS}")
        print(f"  {'─'*40}")

        X_train_fold = X_all[train_idx]
        y_train_fold = y_all[train_idx]
        X_val        = X_all[val_idx]
        y_val        = y_all[val_idx]

        # Keep only NORMAL samples for training (unsupervised setup)
        normal_mask  = y_train_fold == 0
        X_train_norm = X_train_fold[normal_mask]

        print(f"  Train normals: {normal_mask.sum():,} | Val total: {len(y_val):,} | Val anomalies: {y_val.sum():,}")

        # Fit scaler on training normals only
        scaler_fold  = StandardScaler()
        X_train_sc   = scaler_fold.fit_transform(X_train_norm)
        X_val_sc     = scaler_fold.transform(X_val)

        # Train models
        if_model  = IsolationForest(**IF_PARAMS)
        if_model.fit(X_train_sc)

        lof_model = LocalOutlierFactor(**LOF_PARAMS)
        lof_model.fit(X_train_sc)

        # Score validation set
        if_scores  = normalise_if(if_model.score_samples(X_val_sc))
        lof_scores = normalise_lof(lof_model.score_samples(X_val_sc))
        ens_scores = ensemble_score(if_scores, lof_scores)

        # Tune thresholds on this fold's validation set
        t_if  = tune_threshold(if_scores,  y_val)
        t_lof = tune_threshold(lof_scores, y_val)
        t_ens = tune_threshold(ens_scores, y_val)

        # Score metrics
        r_if  = score_fold(if_scores,  y_val, t_if)
        r_lof = score_fold(lof_scores, y_val, t_lof)
        r_ens = score_fold(ens_scores, y_val, t_ens)

        fold_results["isolation_forest"].append(r_if)
        fold_results["lof"].append(r_lof)
        fold_results["ensemble"].append(r_ens)

        print(f"  IF   → F1:{r_if['f1']:.3f}  P:{r_if['precision']:.3f}  R:{r_if['recall']:.3f}  AUC:{r_if['auc_roc']:.3f}  t={t_if}")
        print(f"  LOF  → F1:{r_lof['f1']:.3f}  P:{r_lof['precision']:.3f}  R:{r_lof['recall']:.3f}  AUC:{r_lof['auc_roc']:.3f}  t={t_lof}")
        print(f"  ENS  → F1:{r_ens['f1']:.3f}  P:{r_ens['precision']:.3f}  R:{r_ens['recall']:.3f}  AUC:{r_ens['auc_roc']:.3f}  t={t_ens}")

    return fold_results


def aggregate_cv_results(fold_results: dict) -> dict:
    """Compute mean ± std across all folds for each metric and model."""
    metrics  = ["f1", "precision", "recall", "auc_roc", "threshold"]
    summary  = {}

    print(f"\n{'='*55}")
    print(f"  Cross-Validation Summary (mean ± std over {N_FOLDS} folds)")
    print(f"{'='*55}")

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
                print(f"    {metric:<12}: {mean:.4f} ± {std:.4f}  (folds: {[round(v,3) for v in values]})")
        print(f"    {'threshold':<12}: {summary[model_name]['threshold']:.2f} ± {summary[model_name]['threshold_std']:.2f}")

    return summary


# ── Final model training 

def train_final_models(full_df: pd.DataFrame) -> tuple:
    """
    Retrain final models on ALL normal data using mean CV thresholds.
    This is the model that gets deployed to the API.
    """
    print(f"\n{'='*55}")
    print(f"  Training Final Models on All Normal Data")
    print(f"{'='*55}")

    normal_df   = full_df[full_df["is_anomaly"] == 0]
    X_all_norm  = normal_df[FEATURE_COLS].fillna(0).values

    print(f"  Normal training samples: {len(X_all_norm):,}")

    scaler      = StandardScaler()
    X_scaled    = scaler.fit_transform(X_all_norm)

    print("  Fitting Isolation Forest ...")
    if_model    = IsolationForest(**IF_PARAMS)
    if_model.fit(X_scaled)

    print("  Fitting Local Outlier Factor ...")
    lof_model   = LocalOutlierFactor(**LOF_PARAMS)
    lof_model.fit(X_scaled)

    print("  Final models trained.")
    return if_model, lof_model, scaler


# ── Save artifacts 

def save_artifacts(if_model, lof_model, scaler, thresholds, eval_report, cv_summary):
    with open(ARTIFACT_DIR / "isolation_forest.pkl", "wb") as f:
        pickle.dump(if_model, f)
    with open(ARTIFACT_DIR / "lof.pkl", "wb") as f:
        pickle.dump(lof_model, f)
    with open(ARTIFACT_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
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
        print(f"    {f.name:<35} {f.stat().st_size:>10,} bytes")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Credit Card Anomaly Detection — Training")
    print(f"  Strategy: {N_FOLDS}-Fold Cross-Validation")
    print("=" * 55)

    # Load full dataset (normal + anomalies combined)
    full_df = pd.read_csv(DATA_DIR / "transactions.csv")
    print(f"\n  Full dataset: {len(full_df):,} rows")
    print(f"  Normal:       {(full_df['is_anomaly']==0).sum():,}")
    print(f"  Anomalous:    {(full_df['is_anomaly']==1).sum():,}")
    print(f"  Anomaly rate: {full_df['is_anomaly'].mean():.1%}")

    # ── Step 1: Cross-validation 
    fold_results = run_cross_validation(full_df)

    # ── Step 2: Aggregate results 
    cv_summary = aggregate_cv_results(fold_results)

    # Extract mean thresholds from CV to use for final model
    thresholds = {
        model: round(cv_summary[model]["threshold"], 2)
        for model in cv_summary
    }
    print(f"\n  Mean thresholds from CV: {thresholds}")

    # ── Step 3: Train final models on all normal data 
    if_model, lof_model, scaler = train_final_models(full_df)

    # ── Step 4: Final holdout evaluation 
    print(f"\n{'='*55}")
    print(f"  Final Holdout Evaluation")
    print(f"{'='*55}")

    test_df  = pd.read_csv(DATA_DIR / "test.csv")
    X_test   = scaler.transform(test_df[FEATURE_COLS].fillna(0).values)
    y_test   = test_df["is_anomaly"].values

    if_scores  = normalise_if(if_model.score_samples(X_test))
    lof_scores = normalise_lof(lof_model.score_samples(X_test))
    ens_scores = ensemble_score(if_scores, lof_scores)

    eval_report = {"cv_summary": cv_summary, "holdout": {}, "n_folds": N_FOLDS}
    all_scores  = {
        "isolation_forest": if_scores,
        "lof":              lof_scores,
        "ensemble":         ens_scores,
    }

    print(f"\n  Test set: {len(y_test):,} rows | Anomaly rate: {y_test.mean():.1%}")
    for model_name, scores in all_scores.items():
        t     = thresholds[model_name]
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
        eval_report["holdout"][model_name] = result
        print(f"\n  [{model_name}]")
        print(f"    F1:        {result['f1']:.4f}")
        print(f"    Precision: {result['precision']:.4f}")
        print(f"    Recall:    {result['recall']:.4f}")
        print(f"    AUC-ROC:   {result['auc_roc']:.4f}")
        print(f"    Threshold: {result['threshold']}")

    # ── Step 5: Save 
    save_artifacts(if_model, lof_model, scaler, thresholds, eval_report, cv_summary)

    print(f"\n{'='*55}")
    print(f"  Training complete.")
    print(f"  Run: docker compose up -d")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()