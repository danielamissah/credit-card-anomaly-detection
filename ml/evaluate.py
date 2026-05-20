"""
ml/evaluate.py
───────────────
Generates all evaluation plots after training is complete.

Plots saved to ml/plots/:
  01_roc_curves.png           — ROC curve for each model + ensemble
  02_precision_recall.png     — Precision-Recall curve (better for imbalanced data)
  03_confusion_matrix.png     — Normalised confusion matrix for the ensemble
  04_score_distributions.png  — Anomaly score histogram (normal vs anomaly)
  05_cv_metrics_boxplot.png   — Cross-validation F1/AUC across all 5 folds
  06_feature_importance.png   — Permutation importance via ensemble score impact
  07_anomaly_type_breakdown.png — Detection rate per anomaly type

Why these plots?
  - ROC curve shows discrimination ability across all thresholds
  - PR curve is more informative than ROC when classes are imbalanced
  - Normalised confusion matrix shows false positive / false negative rates
    as proportions — easier to read than raw counts when class sizes differ
  - Score distributions show how well the model separates the two classes
  - CV boxplot shows variance across folds — low variance = stable model
  - Feature importance shows which features drive anomaly detection

Run:
  python ml/evaluate.py
"""

import json
import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    confusion_matrix,
    f1_score,
)

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR     = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(ROOT_DIR))
DATA_DIR     = ROOT_DIR / "data"
ARTIFACT_DIR = Path(__file__).parent / "artifacts"
PLOTS_DIR    = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# ── Plot style — dark theme to match the Grafana dashboard aesthetic ───────────
plt.rcParams.update({
    "figure.facecolor":  "#0d1117",
    "axes.facecolor":    "#161b22",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#e6edf3",
    "text.color":        "#e6edf3",
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "grid.color":        "#21262d",
    "grid.linewidth":    0.8,
    "legend.facecolor":  "#161b22",
    "legend.edgecolor":  "#30363d",
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "figure.dpi":        150,
})

COLOURS = {
    "isolation_forest": "#58a6ff",   # blue
    "lof":              "#3fb950",   # green
    "tabnet":           "#d2a8ff",   # purple
    "ensemble":         "#f0b429",   # amber — primary colour
    "normal":           "#3fb950",
    "anomaly":          "#f85149",
}

FEATURE_COLS = [
    "log_amount", "amount_to_avg_ratio", "amount_to_limit_ratio",
    "is_foreign", "hour", "hour_bin", "is_weekend",
    "is_card_present", "mcc_risk_score", "day_of_week",
]

FEATURE_LABELS = [
    "Log Amount", "Amount / Avg Spend", "Amount / Credit Limit",
    "Is Foreign", "Hour of Day", "Hour Bin", "Is Weekend",
    "Card Present", "MCC Risk Score", "Day of Week",
]


# ── Load artifacts ─────────────────────────────────────────────────────────────

def load_artifacts():
    """Load all trained model artifacts and test data."""
    print("Loading artifacts ...")

    with open(ARTIFACT_DIR / "isolation_forest.pkl", "rb") as f:
        if_model = pickle.load(f)
    with open(ARTIFACT_DIR / "lof.pkl", "rb") as f:
        lof_model = pickle.load(f)
    with open(ARTIFACT_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(ARTIFACT_DIR / "thresholds.json") as f:
        thresholds = json.load(f)
    with open(ARTIFACT_DIR / "cv_summary.json") as f:
        cv_summary = json.load(f)

    # Load TabNet if available
    tabnet_model = None
    tabnet_config_path = ARTIFACT_DIR / "tabnet_config.json"
    if (ARTIFACT_DIR / "tabnet_encoder.pt").exists() and tabnet_config_path.exists():
        with open(tabnet_config_path) as f:
            tabnet_config = json.load(f)
        try:
            from ml.train import TabNetAutoencoder
            tabnet_model = TabNetAutoencoder(tabnet_config)
            tabnet_model.encoder.load_state_dict(
                torch.load(ARTIFACT_DIR / "tabnet_encoder.pt", map_location="cpu", weights_only=True)
            )
            tabnet_model.decoder.load_state_dict(
                torch.load(ARTIFACT_DIR / "tabnet_decoder.pt", map_location="cpu", weights_only=True)
            )
            tabnet_model.eval()
            print("  TabNet loaded.")
        except Exception as e:
            print(f"  TabNet not available: {e}")

    test_df = pd.read_csv(DATA_DIR / "test.csv")
    X_test  = scaler.transform(test_df[FEATURE_COLS].fillna(0).values)
    y_test  = test_df["is_anomaly"].values

    return if_model, lof_model, tabnet_model, scaler, thresholds, cv_summary, X_test, y_test


def get_all_scores(if_model, lof_model, tabnet_model, X_test):
    """Compute anomaly scores from all models."""
    if_scores  = np.clip(0.5 - if_model.score_samples(X_test), 0, 1)
    lof_scores = np.clip((-lof_model.score_samples(X_test)) / 2.0, 0, 1)

    tab_scores = np.zeros(len(X_test))
    if tabnet_model is not None:
        x_t = torch.tensor(X_test, dtype=torch.float32)
        errors = tabnet_model.reconstruction_error(x_t)
        p1, p99 = np.percentile(errors, 1), np.percentile(errors, 99)
        tab_scores = np.clip((errors - p1) / (p99 - p1 + 1e-9), 0, 1)

    ens_scores = 0.35 * if_scores + 0.35 * lof_scores + 0.30 * tab_scores

    return {
        "isolation_forest": if_scores,
        "lof":              lof_scores,
        "tabnet":           tab_scores,
        "ensemble":         ens_scores,
    }


# ── Plot 1: ROC Curves ─────────────────────────────────────────────────────────

def plot_roc_curves(scores, y_test):
    """
    ROC curve plots True Positive Rate vs False Positive Rate at every threshold.
    AUC (Area Under Curve) summarises this — 1.0 is perfect, 0.5 is random.
    We plot all models on one chart for direct comparison.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, score_arr in scores.items():
        fpr, tpr, _ = roc_curve(y_test, score_arr)
        roc_auc     = auc(fpr, tpr)
        label       = f"{name.replace('_', ' ').title()} (AUC = {roc_auc:.3f})"
        lw          = 2.5 if name == "ensemble" else 1.5
        ax.plot(fpr, tpr, color=COLOURS[name], lw=lw, label=label)

    # Random classifier baseline
    ax.plot([0, 1], [0, 1], color="#484f58", lw=1, linestyle="--", label="Random (AUC = 0.500)")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])

    plt.tight_layout()
    path = PLOTS_DIR / "01_roc_curves.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


# ── Plot 2: Precision-Recall Curves ───────────────────────────────────────────

def plot_precision_recall(scores, y_test):
    """
    Precision-Recall curve is more informative than ROC when classes are
    imbalanced (which fraud data always is). High precision = few false alarms.
    High recall = few missed frauds. They trade off against each other.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, score_arr in scores.items():
        precision, recall, _ = precision_recall_curve(y_test, score_arr)
        pr_auc = auc(recall, precision)
        label  = f"{name.replace('_', ' ').title()} (AUC = {pr_auc:.3f})"
        lw     = 2.5 if name == "ensemble" else 1.5
        ax.plot(recall, precision, color=COLOURS[name], lw=lw, label=label)

    # Baseline: random classifier at the positive class rate
    baseline = y_test.mean()
    ax.axhline(y=baseline, color="#484f58", lw=1, linestyle="--",
               label=f"Random (baseline = {baseline:.2f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — All Models")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    plt.tight_layout()
    path = PLOTS_DIR / "02_precision_recall.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


# ── Plot 3: Normalised Confusion Matrix ────────────────────────────────────────

def plot_confusion_matrix(scores, thresholds, y_test):
    """
    Confusion matrix shows the breakdown of correct and incorrect predictions.
    We normalise by the true label count so each cell is a proportion (0 to 1).
    This makes it easy to compare false positive rate vs false negative rate
    regardless of class imbalance.

    Cells:
      Top-left:     True Negative rate  (correctly flagged as normal)
      Top-right:    False Positive rate (normal flagged as anomaly — false alarm)
      Bottom-left:  False Negative rate (anomaly missed — the dangerous error)
      Bottom-right: True Positive rate  (correctly caught anomalies)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    model_names = ["isolation_forest", "lof", "tabnet", "ensemble"]

    for idx, name in enumerate(model_names):
        ax         = axes[idx]
        threshold  = thresholds.get(name, 0.5)
        preds      = (scores[name] >= threshold).astype(int)
        cm         = confusion_matrix(y_test, preds)

        # Normalise each row by the true label count
        cm_norm    = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        # Draw heatmap
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            ax=ax,
            vmin=0, vmax=1,
            linewidths=0.5,
            linecolor="#30363d",
            annot_kws={"size": 14, "weight": "bold"},
            cbar_kws={"shrink": 0.8},
        )

        f1  = f1_score(y_test, preds, zero_division=0)
        ax.set_title(f"{name.replace('_', ' ').title()}\n(F1 = {f1:.3f}, threshold = {threshold})",
                     fontsize=11)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_xticklabels(["Normal", "Anomaly"], fontsize=10)
        ax.set_yticklabels(["Normal", "Anomaly"], fontsize=10, rotation=0)

    fig.suptitle("Normalised Confusion Matrices — All Models", fontsize=14, y=1.01)
    plt.tight_layout()
    path = PLOTS_DIR / "03_confusion_matrix.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


# ── Plot 4: Score Distributions ───────────────────────────────────────────────

def plot_score_distributions(scores, y_test):
    """
    Shows the distribution of anomaly scores separately for normal and anomalous
    transactions. A good model produces two well-separated distributions —
    normal transactions cluster near 0, anomalies cluster near 1.
    Overlap between the distributions is where errors happen.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes      = axes.flatten()

    model_names = ["isolation_forest", "lof", "tabnet", "ensemble"]

    for idx, name in enumerate(model_names):
        ax      = axes[idx]
        s       = scores[name]
        normal  = s[y_test == 0]
        anomaly = s[y_test == 1]

        ax.hist(normal,  bins=50, alpha=0.65, color=COLOURS["normal"],
                label=f"Normal (n={len(normal):,})",  density=True)
        ax.hist(anomaly, bins=50, alpha=0.65, color=COLOURS["anomaly"],
                label=f"Anomaly (n={len(anomaly):,})", density=True)

        # Mark the decision threshold
        from ml.train import TABNET_CONFIG  # noqa
        threshold = 0.5   # will be overridden below
        ax.axvline(x=0.5, color="#f0b429", lw=1.5, linestyle="--",
                   label="Threshold (0.5)")

        ax.set_xlabel("Anomaly Score")
        ax.set_ylabel("Density")
        ax.set_title(f"{name.replace('_', ' ').title()} — Score Distribution")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])

    fig.suptitle("Anomaly Score Distributions (Normal vs Anomalous Transactions)", fontsize=13)
    plt.tight_layout()
    path = PLOTS_DIR / "04_score_distributions.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


# ── Plot 5: CV Metrics Boxplot ─────────────────────────────────────────────────

def plot_cv_boxplot(cv_summary):
    """
    Boxplot of F1 and AUC-ROC across all 5 cross-validation folds.
    The box shows the interquartile range (25th–75th percentile).
    The line inside is the median. Whiskers extend to min/max.
    Tight boxes = stable model. Wide boxes = high variance across folds.
    """
    metrics     = ["f1", "auc_roc"]
    model_names = list(cv_summary.keys())
    fig, axes   = plt.subplots(1, 2, figsize=(13, 6))

    for ax_idx, metric in enumerate(metrics):
        ax   = axes[ax_idx]
        data = []
        for model in model_names:
            fold_values = cv_summary[model].get(f"{metric}_folds", [])
            data.append(fold_values)

        bp = ax.boxplot(
            data,
            patch_artist=True,
            notch=False,
            widths=0.5,
            medianprops=dict(color="#f0b429", linewidth=2.5),
            whiskerprops=dict(color="#8b949e"),
            capprops=dict(color="#8b949e"),
            flierprops=dict(marker="o", color="#f85149", markersize=5),
        )

        # Colour each box
        colours = [COLOURS.get(m, "#8b949e") for m in model_names]
        for patch, colour in zip(bp["boxes"], colours):
            patch.set_facecolor(colour)
            patch.set_alpha(0.7)

        # Overlay individual fold points
        for i, fold_vals in enumerate(data, start=1):
            jitter = np.random.normal(0, 0.04, len(fold_vals))
            ax.scatter(
                [i] * len(fold_vals) + jitter,
                fold_vals,
                color=colours[i - 1],
                s=40, zorder=5, alpha=0.9,
            )

        ax.set_xticks(range(1, len(model_names) + 1))
        ax.set_xticklabels([m.replace("_", "\n") for m in model_names], fontsize=9)
        ax.set_ylabel(metric.upper().replace("_", "-"))
        ax.set_title(f"{metric.upper().replace('_', '-')} Across 5 CV Folds")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim([max(0, min(v for d in data for v in d) - 0.05), 1.02])

    fig.suptitle("5-Fold Cross-Validation Results — All Models", fontsize=13)
    plt.tight_layout()
    path = PLOTS_DIR / "05_cv_metrics_boxplot.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


# ── Plot 6: Feature Importance ─────────────────────────────────────────────────

def plot_feature_importance(if_model, lof_model, tabnet_model, X_test, y_test):
    """
    Permutation feature importance: shuffle each feature one at a time and
    measure how much the ensemble AUC drops. Large drop = important feature.
    This is model-agnostic and works for any combination of models.
    """
    from sklearn.metrics import roc_auc_score

    def get_ensemble_auc(X):
        if_s  = np.clip(0.5 - if_model.score_samples(X), 0, 1)
        lof_s = np.clip((-lof_model.score_samples(X)) / 2.0, 0, 1)
        tab_s = np.zeros(len(X))
        if tabnet_model is not None:
            x_t   = torch.tensor(X, dtype=torch.float32)
            errors = tabnet_model.reconstruction_error(x_t)
            p1, p99 = np.percentile(errors, 1), np.percentile(errors, 99)
            tab_s = np.clip((errors - p1) / (p99 - p1 + 1e-9), 0, 1)
        ens = 0.35 * if_s + 0.35 * lof_s + 0.30 * tab_s
        try:
            return roc_auc_score(y_test, ens)
        except Exception:
            return 0.0

    print("  Computing feature importance (this takes ~30s) ...")
    baseline_auc  = get_ensemble_auc(X_test)
    importances   = []
    n_repeats     = 3   # average over 3 shuffles for stability

    for feat_idx in range(X_test.shape[1]):
        drops = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            np.random.shuffle(X_perm[:, feat_idx])
            perm_auc = get_ensemble_auc(X_perm)
            drops.append(baseline_auc - perm_auc)
        importances.append(np.mean(drops))

    # Sort features by importance (descending)
    importances  = np.array(importances)
    sorted_idx   = np.argsort(importances)[::-1]
    sorted_feats = [FEATURE_LABELS[i] for i in sorted_idx]
    sorted_imps  = importances[sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    colours_bar = [COLOURS["ensemble"] if v >= 0 else COLOURS["anomaly"] for v in sorted_imps]
    bars = ax.barh(range(len(sorted_feats)), sorted_imps[::-1], color=colours_bar[::-1], alpha=0.85)
    ax.set_yticks(range(len(sorted_feats)))
    ax.set_yticklabels(sorted_feats[::-1], fontsize=10)
    ax.set_xlabel("Mean AUC Drop (higher = more important)")
    ax.set_title("Permutation Feature Importance\n(Ensemble Model — averaged over 3 shuffles)")
    ax.axvline(x=0, color="#484f58", lw=1)
    ax.grid(True, axis="x", alpha=0.3)

    # Annotate bars
    for bar, val in zip(bars, sorted_imps[::-1]):
        ax.text(
            max(val + 0.0005, 0.0005), bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=8, color="#e6edf3",
        )

    plt.tight_layout()
    path = PLOTS_DIR / "06_feature_importance.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


# ── Plot 7: Detection Rate by Anomaly Type ─────────────────────────────────────

def plot_anomaly_type_breakdown(scores, thresholds, test_df):
    """
    For each anomaly type (high_value, velocity, geographic, odd_hour),
    shows what percentage the ensemble correctly detected vs missed.
    This tells you where the model is strong and where it struggles.
    """
    threshold = thresholds.get("ensemble", 0.5)
    preds     = (scores["ensemble"] >= threshold).astype(int)

    anomaly_df = test_df[test_df["is_anomaly"] == 1].copy()
    anomaly_df["predicted"] = preds[test_df["is_anomaly"] == 1]

    types  = anomaly_df["anomaly_type"].unique()
    recall_per_type = {}
    count_per_type  = {}

    for atype in types:
        subset  = anomaly_df[anomaly_df["anomaly_type"] == atype]
        recall  = subset["predicted"].mean()
        recall_per_type[atype] = recall
        count_per_type[atype]  = len(subset)

    # Sort by recall
    sorted_types  = sorted(recall_per_type, key=recall_per_type.get, reverse=True)
    recall_vals   = [recall_per_type[t] for t in sorted_types]
    count_vals    = [count_per_type[t]  for t in sorted_types]

    type_colours = {
        "high_value":  "#f0b429",
        "velocity":    "#58a6ff",
        "geographic":  "#3fb950",
        "odd_hour":    "#d2a8ff",
        "normal":      "#8b949e",
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    bar_colours = [type_colours.get(t, "#8b949e") for t in sorted_types]
    bars = ax.bar(
        [t.replace("_", " ").title() for t in sorted_types],
        [r * 100 for r in recall_vals],
        color=bar_colours,
        alpha=0.85,
        width=0.55,
    )

    # Annotate with count and recall
    for bar, recall, count in zip(bars, recall_vals, count_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{recall*100:.1f}%\n(n={count})",
            ha="center", va="bottom", fontsize=10, color="#e6edf3",
        )

    ax.set_ylabel("Detection Rate (%)")
    ax.set_title("Ensemble Detection Rate by Anomaly Type")
    ax.set_ylim([0, 115])
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(y=80, color="#484f58", lw=1, linestyle="--", label="80% target")
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = PLOTS_DIR / "07_anomaly_type_breakdown.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Generating Evaluation Plots")
    print("=" * 55)

    if_model, lof_model, tabnet_model, scaler, thresholds, cv_summary, X_test, y_test = load_artifacts()
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    scores  = get_all_scores(if_model, lof_model, tabnet_model, X_test)

    print("\nGenerating plots ...")
    plot_roc_curves(scores, y_test)
    plot_precision_recall(scores, y_test)
    plot_confusion_matrix(scores, thresholds, y_test)
    plot_score_distributions(scores, y_test)
    plot_cv_boxplot(cv_summary)
    plot_feature_importance(if_model, lof_model, tabnet_model, X_test, y_test)
    plot_anomaly_type_breakdown(scores, thresholds, test_df)

    print(f"\n  All plots saved to {PLOTS_DIR}/")
    print("  Include these in your README and Streamlit demo.")


if __name__ == "__main__":
    main()