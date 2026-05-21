"""
streamlit_demo.py

Pages:
  1. Live Detector   — submit a transaction and see the anomaly score live
  2. Model Overview  — architecture, evaluation metrics, CV results
  3. Evaluation Plots — all generated plots from ml/evaluate.py
  4. Dataset Explorer — explore the synthetic training data

Run locally:
  streamlit run streamlit_demo.py

"""

import json
import math
import pickle
import pycountry
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── Page config — must be first Streamlit call 
st.set_page_config(
    page_title = "Credit Card Anomaly Detection",
    page_icon  = "",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Paths 
ROOT_DIR     = Path(__file__).parent
ARTIFACT_DIR = ROOT_DIR / "ml" / "artifacts"
PLOTS_DIR    = ROOT_DIR / "ml" / "plots"
DATA_DIR     = ROOT_DIR / "data"

# Theme Configuration
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

:root {
    --accent:  #ff9800;
    --green:   #3fb950;
    --red:     #f85149;
}

@media (prefers-color-scheme: dark) {
    :root {
        --bg:      #001f1c;
        --card:    #00332e;
        --border:  #004d40;
        --text:    #e0f2f1;
        --muted:   #80cbc4;
    }
}

@media (prefers-color-scheme: light) {
    :root {
        --bg:      #e0f2f1;
        --card:    #ffffff;
        --border:  #b2dfdb;
        --text:    #00332e;
        --muted:   #00695c;
    }
}

.stApp { background: var(--bg); font-family: 'Open Sans', sans-serif; color: var(--text); }
.metric-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 8px; padding: 1.2rem 1.5rem; text-align: center;
}
.metric-label { font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }
.metric-value { font-family: 'DM Serif Display', serif; font-size: 2rem; color: var(--text); }
.anomaly-badge { background: #f8514920; border: 1px solid #f85149; color: #f85149; padding: 0.3rem 0.8rem; border-radius: 4px; font-weight: 600; }
.normal-badge  { background: #3fb95020; border: 1px solid #3fb950; color: #3fb950; padding: 0.3rem 0.8rem; border-radius: 4px; font-weight: 600; }
#MainMenu { visibility: hidden; } footer { visibility: hidden; } header { visibility: hidden; }
[data-testid="stSidebar"] { background: var(--card) !important; border-right: 1px solid var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# Initialize models

@st.cache_resource
def load_models():
    """
    Load all trained model artifacts into memory.
    Uses Streamlit's cache so models are only loaded once,
    not on every page interaction.
    """
    try:
        with open(ARTIFACT_DIR / "isolation_forest.pkl", "rb") as f:
            if_model = pickle.load(f)
        with open(ARTIFACT_DIR / "lof.pkl", "rb") as f:
            lof_model = pickle.load(f)
        with open(ARTIFACT_DIR / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open(ARTIFACT_DIR / "thresholds.json") as f:
            thresholds = json.load(f)
        with open(ARTIFACT_DIR / "evaluation_report.json") as f:
            evaluation = json.load(f)
        with open(ARTIFACT_DIR / "cv_summary.json") as f:
            cv_summary = json.load(f)

        # Try loading TabNet
        tabnet_model = None
        if (ARTIFACT_DIR / "tabnet_encoder.pt").exists():
            try:
                import torch
                sys.path.insert(0, str(ROOT_DIR))
                from ml.train import TabNetAutoencoder
                with open(ARTIFACT_DIR / "tabnet_config.json") as f:
                    config = json.load(f)
                tabnet_model = TabNetAutoencoder(config)
                tabnet_model.encoder.load_state_dict(
                    torch.load(ARTIFACT_DIR / "tabnet_encoder.pt", map_location="cpu", weights_only=True)
                )
                tabnet_model.decoder.load_state_dict(
                    torch.load(ARTIFACT_DIR / "tabnet_decoder.pt", map_location="cpu", weights_only=True)
                )
                tabnet_model.eval()
            except Exception as e:
                pass

        return if_model, lof_model, tabnet_model, scaler, thresholds, evaluation, cv_summary, True

    except FileNotFoundError:
        return None, None, None, None, {}, {}, {}, False


def score_transaction(txn_features: dict, if_model, lof_model, tabnet_model, scaler, thresholds):
    """
    Score a single transaction through all three models and return
    individual scores, the ensemble score, and the final decision.
    """
    import torch

    # Build feature vector in the same order as training
    mcc_risk = {5411:0.1,5812:0.1,5541:0.2,5999:0.4,4722:0.3,7922:0.2,6011:0.6,5912:0.1,5734:0.5,5944:0.7}
    hour     = txn_features["hour"]
    hour_bin = 0 if hour <= 5 else 1 if hour <= 11 else 2 if hour <= 17 else 3

    X = np.array([[
        math.log1p(txn_features["amount"]),
        txn_features["amount"] / (txn_features["avg_user_spend"] + 1e-6),
        txn_features["amount"] / (txn_features["credit_limit"] + 1e-6),
        int(txn_features["country"] != txn_features["home_country"]),
        hour, hour_bin,
        int(txn_features["day_of_week"] >= 5),
        txn_features["is_card_present"],
        mcc_risk.get(txn_features["mcc"], 0.3),
        txn_features["day_of_week"],
    ]])

    X_scaled = scaler.transform(X)

    # Score each model
    if_score  = float(np.clip(0.5 - if_model.score_samples(X_scaled)[0], 0, 1))
    lof_score = float(np.clip((-lof_model.score_samples(X_scaled)[0]) / 2.0, 0, 1))

    tab_score = 0.0
    if tabnet_model is not None:
        x_t      = torch.tensor(X_scaled, dtype=torch.float32)
        error    = float(tabnet_model.reconstruction_error(x_t)[0])
        tab_score = float(np.clip(error, 0, 1))

    ensemble = 0.35 * if_score + 0.35 * lof_score + 0.30 * tab_score
    threshold = thresholds.get("ensemble", 0.5)

    # Model Explainability
    feature_explanations = {
        0: "the transaction amount is significantly larger than typical transactions",
        1: "the amount is unusually high compared to typical spending",
        2: "the transaction utilizes an unusually large portion of the credit limit",
        3: "the transaction occurred in a country different from the home country",
        4: "the transaction occurred at an unusual hour",
        5: "the transaction occurred at an unusual time of day",
        6: "the transaction occurred on a weekend, which is unusual",
        7: "the physical presence of the card is unusual for this type of transaction",
        8: "the merchant category has a high risk profile",
        9: "the transaction occurred on an unusual day of the week"
    }
    
    is_anomaly = ensemble >= threshold
    deviations = []
    if is_anomaly:
        for i, val in enumerate(X_scaled[0]):
            if abs(val) > 2.5:
                deviations.append(feature_explanations[i])
                
        if len(deviations) > 1:
            feature_attributions = ", ".join(deviations[:-1]) + ", and " + deviations[-1]
        elif deviations:
            feature_attributions = deviations[0]
        else:
            feature_attributions = ""
            
        if feature_attributions:
            explanation = f"The AI flagged this because {feature_attributions}."
        else:
            explanation = "The AI flagged this because of a complex combination of features that collectively deviate from your typical spending patterns."
    else:
        explanation = "The AI evaluated this transaction and found all patterns to be within normal behavior."

    return {
        "isolation_forest": round(if_score, 4),
        "lof":              round(lof_score, 4),
        "tabnet":           round(tab_score, 4),
        "ensemble":         round(ensemble, 4),
        "is_anomaly":       is_anomaly,
        "threshold":        threshold,
        "explanation":      explanation
    }


# ── Sidebar navigation 

with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 1.5rem; border-bottom:1px solid #30363d; margin-bottom:1rem;'>
        <div style='font-family:DM Serif Display,serif;font-size:1.2rem;'>💳 CCAD</div>
        <div style='font-size:0.65rem;color:#484f58;font-family:JetBrains Mono;'>CREDIT CARD ANOMALY DETECTION</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", ["Live Detector", "Model Overview", "Evaluation Plots", "Dataset Explorer"],
                    label_visibility="collapsed")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.7rem;color:#484f58;font-family:JetBrains Mono;'>
    MODELS<br>
    <span style='color:#8b949e;'>
    Isolation Forest<br>
    Local Outlier Factor<br>
    TabNet Autoencoder<br>
    </span><br>
    CV<br>
    <span style='color:#8b949e;'>5-fold stratified</span><br><br>
    SOURCE<br>
    <span style='color:#58a6ff;'><a href='https://github.com/danielamissah/credit-card-anomaly-detection' style='color:#58a6ff;'>GitHub ↗</a></span>
    </div>
    """, unsafe_allow_html=True)

# ── Load models 
if_model, lof_model, tabnet_model, scaler, thresholds, evaluation, cv_summary, models_loaded = load_models()

# ══════════════════════════════════════════════════════════════════════════════
# Page 1: Live Detector
# ══════════════════════════════════════════════════════════════════════════════

if page == "Live Detector":
    st.markdown("<h1 style='font-family:DM Serif Display,serif;font-size:2rem;'>Live Transaction Detector</h1>", unsafe_allow_html=True)
    st.markdown("Submit a credit card transaction and see the anomaly score from all three models in real time.")

    if not models_loaded:
        st.error("Models not loaded. Run `python ml/train.py` first, then restart the app.")
        st.stop()

    # ── Input form 
    st.markdown("### Transaction Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        amount        = st.number_input("Amount (EUR)", min_value=0.01, max_value=50000.0, value=1250.0, step=10.0)
        avg_spend     = st.number_input("User Avg Spend (EUR)", min_value=1.0, max_value=10000.0, value=45.0)
        credit_limit  = st.number_input("Credit Limit (EUR)", min_value=100.0, max_value=100000.0, value=5000.0, step=100.0)

    with col2:
        ALL_COUNTRIES = {country.alpha_2: country.name for country in pycountry.countries}
        # Sort countries alphabetically by name, but put some common ones at the top for convenience
        popular = ["DE", "US", "GB", "FR", "CN", "BR", "NG"]
        sorted_codes = popular + [c for c in sorted(ALL_COUNTRIES.keys(), key=lambda x: ALL_COUNTRIES[x]) if c not in popular]
        
        country       = st.selectbox("Transaction Country", sorted_codes, index=0, format_func=lambda x: ALL_COUNTRIES.get(x, x))
        home_country  = st.selectbox("Home Country", sorted_codes, index=0, format_func=lambda x: ALL_COUNTRIES.get(x, x))
        mcc           = st.selectbox("Merchant Category (MCC)",
                                      [5411, 5812, 5541, 5999, 4722, 7922, 6011, 5912, 5734, 5944],
                                      format_func=lambda x: {
                                          5411:"Grocery",5812:"Restaurant",5541:"Gas Station",
                                          5999:"Online Retail",4722:"Travel",7922:"Entertainment",
                                          6011:"ATM Withdrawal",5912:"Pharmacy",5734:"Electronics",5944:"Luxury/Jewellery"
                                      }.get(x, str(x)), index=9)

    with col3:
        hour          = st.slider("Hour of Day", 0, 23, 3)
        day_of_week   = st.selectbox("Day of Week", [0,1,2,3,4,5,6],
                                      format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
        card_present  = st.radio("Card Present?", ["No (Online/CNP)", "Yes (Physical)"], index=0)
        is_card_present = 0 if "No" in card_present else 1

    if st.button("Analyse Transaction", type="primary", use_container_width=True):
        txn = {
            "amount": amount, "avg_user_spend": avg_spend, "credit_limit": credit_limit,
            "country": country, "home_country": home_country, "mcc": mcc,
            "hour": hour, "day_of_week": day_of_week, "is_card_present": is_card_present,
        }
        result = score_transaction(txn, if_model, lof_model, tabnet_model, scaler, thresholds)

        st.markdown("---")
        st.markdown("### Result")

        badge = "<span class='anomaly-badge'>ANOMALY DETECTED</span>" if result["is_anomaly"] \
                else "<span class='normal-badge'>NORMAL</span>"
        st.markdown(badge, unsafe_allow_html=True)

        # Score cards
        c1, c2, c3, c4 = st.columns(4)
        for col, name, score, colour in [
            (c1, "Isolation Forest", result["isolation_forest"], "#58a6ff"),
            (c2, "LOF",              result["lof"],              "#3fb950"),
            (c3, "TabNet",          result["tabnet"],            "#d2a8ff"),
            (c4, "Ensemble",        result["ensemble"],          "#ff9800"),
        ]:
            col.markdown(f"""
            <div class='metric-card' style='border-color:{colour}40;'>
                <div class='metric-label'>{name}</div>
                <div class='metric-value' style='color:{colour};'>{score:.3f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        if result["explanation"]:
            if result["is_anomaly"]:
                st.warning(f"**AI Explanation:** {result['explanation']}")
            else:
                st.success(f"**AI Explanation:** {result['explanation']}")

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode  = "gauge+number",
            value = result["ensemble"],
            domain = {"x": [0, 1], "y": [0, 1]},
            title  = {"text": "Ensemble Anomaly Score", "font": {"color": "#e6edf3", "size": 14}},
            number = {"font": {"color": "#f0b429", "size": 36}},
            gauge  = {
                "axis":  {"range": [0, 1], "tickcolor": "#8b949e"},
                "bar":   {"color": "#f85149" if result["is_anomaly"] else "#3fb950"},
                "bgcolor": "#161b22",
                "bordercolor": "#30363d",
                "steps": [
                    {"range": [0, thresholds.get("ensemble", 0.5)], "color": "#1c2230"},
                    {"range": [thresholds.get("ensemble", 0.5), 1], "color": "#2d1b1b"},
                ],
                "threshold": {
                    "line": {"color": "#f0b429", "width": 3},
                    "value": thresholds.get("ensemble", 0.5),
                },
            },
        ))
        fig.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font={"color": "#8b949e"},
            height=280, margin=dict(l=30, r=30, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Page 2: Model Overview
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Model Overview":
    st.markdown("<h1 style='font-family:DM Serif Display,serif;font-size:2rem;'>Model Overview</h1>", unsafe_allow_html=True)

    # Architecture summary
    st.markdown("### Ensemble Architecture")
    arch_data = {
        "Model":       ["Isolation Forest", "Local Outlier Factor", "TabNet Autoencoder", "**Ensemble**"],
        "Type":        ["Tree-based", "Density-based", "Deep Learning", "Weighted Average"],
        "Weight":      ["0.35", "0.35", "0.30", "—"],
        "Training":    ["Unsupervised", "Unsupervised", "Autoencoder on normals", "—"],
        "Strength":    ["Global outliers", "Local density anomalies", "Feature interaction anomalies", "All of the above"],
    }
    st.dataframe(pd.DataFrame(arch_data), use_container_width=True, hide_index=True)

    if models_loaded and cv_summary:
        st.markdown("### Cross-Validation Results (5-fold)")

        rows = []
        for model, metrics in cv_summary.items():
            rows.append({
                "Model":     model.replace("_", " ").title(),
                "F1 (mean)": f"{metrics.get('f1', 0):.4f}",
                "F1 (±std)": f"±{metrics.get('f1_std', 0):.4f}",
                "Precision": f"{metrics.get('precision', 0):.4f}",
                "Recall":    f"{metrics.get('recall', 0):.4f}",
                "AUC-ROC":   f"{metrics.get('auc_roc', 0):.4f}",
                "Threshold": f"{metrics.get('threshold', 0):.2f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Per-fold F1 bar chart
        st.markdown("### F1 per Fold")
        fold_data = []
        for model, metrics in cv_summary.items():
            for fold_i, val in enumerate(metrics.get("f1_folds", []), start=1):
                fold_data.append({"Model": model.replace("_"," ").title(), "Fold": f"Fold {fold_i}", "F1": val})

        if fold_data:
            df_folds = pd.DataFrame(fold_data)
            fig = px.bar(
                df_folds, x="Fold", y="F1", color="Model", barmode="group",
                color_discrete_map={
                    "Isolation Forest": "#58a6ff", "Lof": "#3fb950",
                    "Tabnet": "#d2a8ff", "Ensemble": "#f0b429",
                },
            )
            fig.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                font={"color": "#8b949e"}, height=350,
                xaxis={"gridcolor":"#21262d"}, yaxis={"gridcolor":"#21262d"},
                legend={"bgcolor":"#161b22"},
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Anomaly Types Detected")
    types_df = pd.DataFrame([
        {"Type": "High Value",  "Description": "Single transaction 10x+ above user's spending baseline", "Signal": "amount_to_avg_ratio, log_amount"},
        {"Type": "Velocity",    "Description": "8+ small transactions within 30 minutes (card testing)", "Signal": "hour, is_card_present, mcc_risk_score"},
        {"Type": "Geographic",  "Description": "Transaction from high-risk or unexpected country",         "Signal": "is_foreign, mcc_risk_score"},
        {"Type": "Odd Hour",    "Description": "Large transaction at 01:00–04:59 for a daytime user",     "Signal": "hour, hour_bin, log_amount"},
    ])
    st.dataframe(types_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# Page 3: Evaluation Plots
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Evaluation Plots":
    st.markdown("<h1 style='font-family:DM Serif Display,serif;font-size:2rem;'>Evaluation Plots</h1>", unsafe_allow_html=True)

    plot_files = {
        "01_roc_curves.png":           ("ROC Curves", "AUC-ROC for each model. Higher = better. Ensemble (amber) should be top."),
        "02_precision_recall.png":     ("Precision-Recall", "More informative than ROC for imbalanced fraud data."),
        "03_confusion_matrix.png":     ("Normalised Confusion Matrix", "Row-normalised. Bottom-right = catch rate. Top-right = false alarm rate."),
        "04_score_distributions.png":  ("Score Distributions", "Well-separated peaks = model clearly distinguishes normal from anomalous."),
        "05_cv_metrics_boxplot.png":   ("CV Metrics Boxplot", "Tight boxes = stable model. Wide boxes = high variance across folds."),
        "06_feature_importance.png":   ("Feature Importance", "Permutation importance — how much AUC drops when each feature is shuffled."),
        "07_anomaly_type_breakdown.png":("Detection Rate by Type", "Which fraud patterns does the ensemble catch vs miss?"),
    }

    if not PLOTS_DIR.exists() or not any(PLOTS_DIR.glob("*.png")):
        st.warning("No plots found. Run `python ml/evaluate.py` after training to generate them.")
    else:
        for filename, (title, description) in plot_files.items():
            path = PLOTS_DIR / filename
            if path.exists():
                st.markdown(f"### {title}")
                st.caption(description)
                st.image(str(path), use_column_width=True)
                st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# Page 4: Dataset Explorer
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Dataset Explorer":
    st.markdown("<h1 style='font-family:DM Serif Display,serif;font-size:2rem;'>Dataset Explorer</h1>", unsafe_allow_html=True)

    csv_path = DATA_DIR / "transactions.csv"
    if not csv_path.exists():
        st.warning("Dataset not found. Run `python data/generate.py` first.")
        st.stop()

    df = pd.read_csv(csv_path)

    # Summary stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", f"{len(df):,}")
    c2.metric("Normal",    f"{(df['is_anomaly']==0).sum():,}")
    c3.metric("Anomalous", f"{(df['is_anomaly']==1).sum():,}")
    c4.metric("Anomaly Rate", f"{df['is_anomaly'].mean():.1%}")

    st.markdown("### Anomaly Type Distribution")
    type_counts = df[df["is_anomaly"]==1]["anomaly_type"].value_counts()
    fig = px.bar(
        x=type_counts.index, y=type_counts.values,
        labels={"x": "Anomaly Type", "y": "Count"},
        color=type_counts.index,
        color_discrete_sequence=["#f0b429","#58a6ff","#3fb950","#d2a8ff"],
    )
    fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                      font={"color":"#8b949e"}, height=300, showlegend=False,
                      xaxis={"gridcolor":"#21262d"}, yaxis={"gridcolor":"#21262d"})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Amount Distribution (log scale)")
    fig2 = go.Figure()
    for label, colour in [("Normal", "#3fb950"), ("Anomaly", "#f85149")]:
        mask = df["is_anomaly"] == (0 if label == "Normal" else 1)
        fig2.add_trace(go.Histogram(
            x=np.log1p(df[mask]["amount"]), name=label,
            marker_color=colour, opacity=0.7, nbinsx=60,
        ))
    fig2.update_layout(
        barmode="overlay", paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        font={"color":"#8b949e"}, height=300, xaxis_title="log(1 + Amount)",
        xaxis={"gridcolor":"#21262d"}, yaxis={"gridcolor":"#21262d"},
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Sample Transactions")
    n_show = st.slider("Rows to show", 5, 50, 10)
    show_type = st.radio("Show", ["All", "Normal only", "Anomalies only"], horizontal=True)
    display_df = df
    if show_type == "Normal only":
        display_df = df[df["is_anomaly"] == 0]
    elif show_type == "Anomalies only":
        display_df = df[df["is_anomaly"] == 1]
    st.dataframe(
        display_df[["transaction_id","amount","mcc","country","home_country",
                    "hour","is_card_present","is_anomaly","anomaly_type"]].head(n_show),
        use_container_width=True, hide_index=True,
    )