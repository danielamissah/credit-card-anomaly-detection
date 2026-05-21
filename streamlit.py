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
import warnings
import streamlit as st

warnings.filterwarnings("ignore", module="sklearn")

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
PLOTS_DIR    = ROOT_DIR / "assets"
DATA_DIR     = ROOT_DIR / "data"

# Theme Configuration
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

.stApp { font-family: 'Open Sans', sans-serif; }
.metric-card {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 8px; padding: 1.2rem 1.5rem; text-align: center;
    box-shadow: 0 4px 6px rgba(0,0,0,0.02);
}
.metric-label { font-size: 0.72rem; color: #004d40; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600; }
.metric-value { font-family: 'DM Serif Display', serif; font-size: 2rem; color: #f57c00; }
.anomaly-badge { background: #f8514915; border: 1px solid #f85149; color: #f85149; padding: 0.3rem 0.8rem; border-radius: 4px; font-weight: 600; }
.normal-badge  { background: #3fb95015; border: 1px solid #3fb950; color: #3fb950; padding: 0.3rem 0.8rem; border-radius: 4px; font-weight: 600; }
#MainMenu { visibility: hidden; } footer { visibility: hidden; } header { visibility: hidden; }
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


@st.cache_data(ttl=3600)
def get_exchange_rates():
    import requests
    try:
        response = requests.get("https://open.er-api.com/v6/latest/EUR", timeout=5)
        return response.json().get("rates", {})
    except Exception:
        return {}

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

        CURRENCY_MAP = {"AW": {"code": "AWG", "name": "Aruban Florin"}, "AF": {"code": "AFN", "name": "Afghan Afghani"}, "AO": {"code": "AOA", "name": "Angolan Kwanza"}, "AI": {"code": "XCD", "name": "East Caribbean Dollar"}, "AX": {"code": "EUR", "name": "Euro"}, "AL": {"code": "ALL", "name": "Albanian Lek"}, "AD": {"code": "EUR", "name": "Euro"}, "AE": {"code": "AED", "name": "United Arab Emirates Dirham"}, "AR": {"code": "ARS", "name": "Argentine Peso"}, "AM": {"code": "AMD", "name": "Armenian Dram"}, "AS": {"code": "USD", "name": "US Dollar"}, "TF": {"code": "EUR", "name": "Euro"}, "AG": {"code": "XCD", "name": "East Caribbean Dollar"}, "AU": {"code": "AUD", "name": "Australian Dollar"}, "AT": {"code": "EUR", "name": "Euro"}, "AZ": {"code": "AZN", "name": "Azerbaijani Manat"}, "BI": {"code": "BIF", "name": "Burundian Franc"}, "BE": {"code": "EUR", "name": "Euro"}, "BJ": {"code": "XOF", "name": "West African CFA Franc"}, "BQ": {"code": "USD", "name": "US Dollar"}, "BF": {"code": "XOF", "name": "West African CFA Franc"}, "BD": {"code": "BDT", "name": "Bangladeshi Taka"}, "BG": {"code": "BGN", "name": "Bulgarian Lev"}, "BH": {"code": "BHD", "name": "Bahraini Dinar"}, "BS": {"code": "BSD", "name": "Bahamian Dollar"}, "BA": {"code": "BAM", "name": "Bosnia-Herzegovina Convertible Mark"}, "BL": {"code": "EUR", "name": "Euro"}, "BY": {"code": "BYN", "name": "Belarusian Ruble"}, "BZ": {"code": "BZD", "name": "Belize Dollar"}, "BM": {"code": "BMD", "name": "Bermudan Dollar"}, "BO": {"code": "BOB", "name": "Bolivian Boliviano"}, "BR": {"code": "BRL", "name": "Brazilian Real"}, "BB": {"code": "BBD", "name": "Barbadian Dollar"}, "BN": {"code": "BND", "name": "Brunei Dollar"}, "BT": {"code": "INR", "name": "Indian Rupee"}, "BV": {"code": "NOK", "name": "Norwegian Krone"}, "BW": {"code": "BWP", "name": "Botswanan Pula"}, "CF": {"code": "XAF", "name": "Central African CFA Franc"}, "CA": {"code": "CAD", "name": "Canadian Dollar"}, "CC": {"code": "AUD", "name": "Australian Dollar"}, "CH": {"code": "CHF", "name": "Swiss Franc"}, "CL": {"code": "CLP", "name": "Chilean Peso"}, "CN": {"code": "CNY", "name": "Chinese Yuan"}, "CI": {"code": "XOF", "name": "West African CFA Franc"}, "CM": {"code": "XAF", "name": "Central African CFA Franc"}, "CD": {"code": "CDF", "name": "Congolese Franc"}, "CG": {"code": "XAF", "name": "Central African CFA Franc"}, "CK": {"code": "NZD", "name": "New Zealand Dollar"}, "CO": {"code": "COP", "name": "Colombian Peso"}, "KM": {"code": "KMF", "name": "Comorian Franc"}, "CV": {"code": "CVE", "name": "Cape Verdean Escudo"}, "CR": {"code": "CRC", "name": "Costa Rican Colón"}, "CU": {"code": "CUP", "name": "Cuban Peso"}, "CW": {"code": "XCG", "name": "Caribbean guilder"}, "CX": {"code": "AUD", "name": "Australian Dollar"}, "KY": {"code": "KYD", "name": "Cayman Islands Dollar"}, "CY": {"code": "EUR", "name": "Euro"}, "CZ": {"code": "CZK", "name": "Czech Koruna"}, "DE": {"code": "EUR", "name": "Euro"}, "DJ": {"code": "DJF", "name": "Djiboutian Franc"}, "DM": {"code": "XCD", "name": "East Caribbean Dollar"}, "DK": {"code": "DKK", "name": "Danish Krone"}, "DO": {"code": "DOP", "name": "Dominican Peso"}, "DZ": {"code": "DZD", "name": "Algerian Dinar"}, "EC": {"code": "USD", "name": "US Dollar"}, "EG": {"code": "EGP", "name": "Egyptian Pound"}, "ER": {"code": "ERN", "name": "Eritrean Nakfa"}, "EH": {"code": "MAD", "name": "Moroccan Dirham"}, "ES": {"code": "EUR", "name": "Euro"}, "EE": {"code": "EUR", "name": "Euro"}, "ET": {"code": "ETB", "name": "Ethiopian Birr"}, "FI": {"code": "EUR", "name": "Euro"}, "FJ": {"code": "FJD", "name": "Fijian Dollar"}, "FK": {"code": "FKP", "name": "Falkland Islands Pound"}, "FR": {"code": "EUR", "name": "Euro"}, "FO": {"code": "DKK", "name": "Danish Krone"}, "FM": {"code": "USD", "name": "US Dollar"}, "GA": {"code": "XAF", "name": "Central African CFA Franc"}, "GB": {"code": "GBP", "name": "British Pound"}, "GE": {"code": "GEL", "name": "Georgian Lari"}, "GG": {"code": "GBP", "name": "British Pound"}, "GH": {"code": "GHS", "name": "Ghanaian Cedi"}, "GI": {"code": "GIP", "name": "Gibraltar Pound"}, "GN": {"code": "GNF", "name": "Guinean Franc"}, "GP": {"code": "EUR", "name": "Euro"}, "GM": {"code": "GMD", "name": "Gambian Dalasi"}, "GW": {"code": "XOF", "name": "West African CFA Franc"}, "GQ": {"code": "XAF", "name": "Central African CFA Franc"}, "GR": {"code": "EUR", "name": "Euro"}, "GD": {"code": "XCD", "name": "East Caribbean Dollar"}, "GL": {"code": "DKK", "name": "Danish Krone"}, "GT": {"code": "GTQ", "name": "Guatemalan Quetzal"}, "GF": {"code": "EUR", "name": "Euro"}, "GU": {"code": "USD", "name": "US Dollar"}, "GY": {"code": "GYD", "name": "Guyanaese Dollar"}, "HK": {"code": "HKD", "name": "Hong Kong Dollar"}, "HM": {"code": "AUD", "name": "Australian Dollar"}, "HN": {"code": "HNL", "name": "Honduran Lempira"}, "HR": {"code": "EUR", "name": "Euro"}, "HT": {"code": "HTG", "name": "Haitian Gourde"}, "HU": {"code": "HUF", "name": "Hungarian Forint"}, "ID": {"code": "IDR", "name": "Indonesian Rupiah"}, "IM": {"code": "GBP", "name": "British Pound"}, "IN": {"code": "INR", "name": "Indian Rupee"}, "IO": {"code": "USD", "name": "US Dollar"}, "IE": {"code": "EUR", "name": "Euro"}, "IR": {"code": "IRR", "name": "Iranian Rial"}, "IQ": {"code": "IQD", "name": "Iraqi Dinar"}, "IS": {"code": "ISK", "name": "Icelandic Króna"}, "IL": {"code": "ILS", "name": "Israeli New Shekel"}, "IT": {"code": "EUR", "name": "Euro"}, "JM": {"code": "JMD", "name": "Jamaican Dollar"}, "JE": {"code": "GBP", "name": "British Pound"}, "JO": {"code": "JOD", "name": "Jordanian Dinar"}, "JP": {"code": "JPY", "name": "Japanese Yen"}, "KZ": {"code": "KZT", "name": "Kazakhstani Tenge"}, "KE": {"code": "KES", "name": "Kenyan Shilling"}, "KG": {"code": "KGS", "name": "Kyrgystani Som"}, "KH": {"code": "KHR", "name": "Cambodian Riel"}, "KI": {"code": "AUD", "name": "Australian Dollar"}, "KN": {"code": "XCD", "name": "East Caribbean Dollar"}, "KR": {"code": "KRW", "name": "South Korean Won"}, "KW": {"code": "KWD", "name": "Kuwaiti Dinar"}, "LA": {"code": "LAK", "name": "Laotian Kip"}, "LB": {"code": "LBP", "name": "Lebanese Pound"}, "LR": {"code": "LRD", "name": "Liberian Dollar"}, "LY": {"code": "LYD", "name": "Libyan Dinar"}, "LC": {"code": "XCD", "name": "East Caribbean Dollar"}, "LI": {"code": "CHF", "name": "Swiss Franc"}, "LK": {"code": "LKR", "name": "Sri Lankan Rupee"}, "LS": {"code": "ZAR", "name": "South African Rand"}, "LT": {"code": "EUR", "name": "Euro"}, "LU": {"code": "EUR", "name": "Euro"}, "LV": {"code": "EUR", "name": "Euro"}, "MO": {"code": "MOP", "name": "Macanese Pataca"}, "MF": {"code": "EUR", "name": "Euro"}, "MA": {"code": "MAD", "name": "Moroccan Dirham"}, "MC": {"code": "EUR", "name": "Euro"}, "MD": {"code": "MDL", "name": "Moldovan Leu"}, "MG": {"code": "MGA", "name": "Malagasy Ariary"}, "MV": {"code": "MVR", "name": "Maldivian Rufiyaa"}, "MX": {"code": "MXN", "name": "Mexican Peso"}, "MH": {"code": "USD", "name": "US Dollar"}, "MK": {"code": "MKD", "name": "Macedonian Denar"}, "ML": {"code": "XOF", "name": "West African CFA Franc"}, "MT": {"code": "EUR", "name": "Euro"}, "MM": {"code": "MMK", "name": "Myanmar Kyat"}, "ME": {"code": "EUR", "name": "Euro"}, "MN": {"code": "MNT", "name": "Mongolian Tugrik"}, "MP": {"code": "USD", "name": "US Dollar"}, "MZ": {"code": "MZN", "name": "Mozambican Metical"}, "MR": {"code": "MRU", "name": "Mauritanian Ouguiya"}, "MS": {"code": "XCD", "name": "East Caribbean Dollar"}, "MQ": {"code": "EUR", "name": "Euro"}, "MU": {"code": "MUR", "name": "Mauritian Rupee"}, "MW": {"code": "MWK", "name": "Malawian Kwacha"}, "MY": {"code": "MYR", "name": "Malaysian Ringgit"}, "YT": {"code": "EUR", "name": "Euro"}, "NA": {"code": "ZAR", "name": "South African Rand"}, "NC": {"code": "XPF", "name": "CFP Franc"}, "NE": {"code": "XOF", "name": "West African CFA Franc"}, "NF": {"code": "AUD", "name": "Australian Dollar"}, "NG": {"code": "NGN", "name": "Nigerian Naira"}, "NI": {"code": "NIO", "name": "Nicaraguan Córdoba"}, "NU": {"code": "NZD", "name": "New Zealand Dollar"}, "NL": {"code": "EUR", "name": "Euro"}, "NO": {"code": "NOK", "name": "Norwegian Krone"}, "NP": {"code": "NPR", "name": "Nepalese Rupee"}, "NR": {"code": "AUD", "name": "Australian Dollar"}, "NZ": {"code": "NZD", "name": "New Zealand Dollar"}, "OM": {"code": "OMR", "name": "Omani Rial"}, "PK": {"code": "PKR", "name": "Pakistani Rupee"}, "PA": {"code": "PAB", "name": "Panamanian Balboa"}, "PN": {"code": "NZD", "name": "New Zealand Dollar"}, "PE": {"code": "PEN", "name": "Peruvian Sol"}, "PH": {"code": "PHP", "name": "Philippine Peso"}, "PW": {"code": "USD", "name": "US Dollar"}, "PG": {"code": "PGK", "name": "Papua New Guinean Kina"}, "PL": {"code": "PLN", "name": "Polish Zloty"}, "PR": {"code": "USD", "name": "US Dollar"}, "KP": {"code": "KPW", "name": "North Korean Won"}, "PT": {"code": "EUR", "name": "Euro"}, "PY": {"code": "PYG", "name": "Paraguayan Guarani"}, "PS": {"code": "ILS", "name": "Israeli New Shekel"}, "PF": {"code": "XPF", "name": "CFP Franc"}, "QA": {"code": "QAR", "name": "Qatari Riyal"}, "RE": {"code": "EUR", "name": "Euro"}, "RO": {"code": "RON", "name": "Romanian Leu"}, "RU": {"code": "RUB", "name": "Russian Ruble"}, "RW": {"code": "RWF", "name": "Rwandan Franc"}, "SA": {"code": "SAR", "name": "Saudi Riyal"}, "SD": {"code": "SDG", "name": "Sudanese Pound"}, "SN": {"code": "XOF", "name": "West African CFA Franc"}, "SG": {"code": "SGD", "name": "Singapore Dollar"}, "GS": {"code": "GBP", "name": "British Pound"}, "SH": {"code": "SHP", "name": "St. Helena Pound"}, "SJ": {"code": "NOK", "name": "Norwegian Krone"}, "SB": {"code": "SBD", "name": "Solomon Islands Dollar"}, "SL": {"code": "SLE", "name": "Sierra Leonean Leone"}, "SV": {"code": "USD", "name": "US Dollar"}, "SM": {"code": "EUR", "name": "Euro"}, "SO": {"code": "SOS", "name": "Somali Shilling"}, "PM": {"code": "EUR", "name": "Euro"}, "RS": {"code": "RSD", "name": "Serbian Dinar"}, "SS": {"code": "SSP", "name": "South Sudanese Pound"}, "ST": {"code": "STN", "name": "São Tomé & Príncipe Dobra"}, "SR": {"code": "SRD", "name": "Surinamese Dollar"}, "SK": {"code": "EUR", "name": "Euro"}, "SI": {"code": "EUR", "name": "Euro"}, "SE": {"code": "SEK", "name": "Swedish Krona"}, "SZ": {"code": "SZL", "name": "Swazi Lilangeni"}, "SX": {"code": "XCG", "name": "Caribbean guilder"}, "SC": {"code": "SCR", "name": "Seychellois Rupee"}, "SY": {"code": "SYP", "name": "Syrian Pound"}, "TC": {"code": "USD", "name": "US Dollar"}, "TD": {"code": "XAF", "name": "Central African CFA Franc"}, "TG": {"code": "XOF", "name": "West African CFA Franc"}, "TH": {"code": "THB", "name": "Thai Baht"}, "TJ": {"code": "TJS", "name": "Tajikistani Somoni"}, "TK": {"code": "NZD", "name": "New Zealand Dollar"}, "TM": {"code": "TMT", "name": "Turkmenistani Manat"}, "TL": {"code": "USD", "name": "US Dollar"}, "TO": {"code": "TOP", "name": "Tongan Paʻanga"}, "TT": {"code": "TTD", "name": "Trinidad & Tobago Dollar"}, "TN": {"code": "TND", "name": "Tunisian Dinar"}, "TR": {"code": "TRY", "name": "Turkish Lira"}, "TV": {"code": "AUD", "name": "Australian Dollar"}, "TW": {"code": "TWD", "name": "New Taiwan Dollar"}, "TZ": {"code": "TZS", "name": "Tanzanian Shilling"}, "UG": {"code": "UGX", "name": "Ugandan Shilling"}, "UA": {"code": "UAH", "name": "Ukrainian Hryvnia"}, "UM": {"code": "USD", "name": "US Dollar"}, "UY": {"code": "UYU", "name": "Uruguayan Peso"}, "US": {"code": "USD", "name": "US Dollar"}, "UZ": {"code": "UZS", "name": "Uzbekistani Som"}, "VA": {"code": "EUR", "name": "Euro"}, "VC": {"code": "XCD", "name": "East Caribbean Dollar"}, "VE": {"code": "VES", "name": "Venezuelan Bolívar"}, "VG": {"code": "USD", "name": "US Dollar"}, "VI": {"code": "USD", "name": "US Dollar"}, "VN": {"code": "VND", "name": "Vietnamese Dong"}, "VU": {"code": "VUV", "name": "Vanuatu Vatu"}, "WF": {"code": "XPF", "name": "CFP Franc"}, "WS": {"code": "WST", "name": "Samoan Tala"}, "YE": {"code": "YER", "name": "Yemeni Rial"}, "ZA": {"code": "ZAR", "name": "South African Rand"}, "ZM": {"code": "ZMW", "name": "Zambian Kwacha"}, "ZW": {"code": "USD", "name": "US Dollar"}}
        
        currency_info = CURRENCY_MAP.get(home_country, {"code": "EUR", "name": "Local Currency"})
        currency_name = currency_info["name"]
        currency_code = currency_info["code"]
        
        rates = get_exchange_rates()
        exchange_rate = rates.get(currency_code, 1.0)
        
        amount        = st.number_input(f"Amount ({currency_name})", min_value=0.01, value=float(round(1250.0 * exchange_rate, 2)))
        avg_spend     = st.number_input(f"User Avg Spend ({currency_name})", min_value=1.0, value=float(round(45.0 * exchange_rate, 2)))
        credit_limit  = st.number_input(f"Credit Limit ({currency_name})", min_value=100.0, value=float(round(5000.0 * exchange_rate, 2)))

    with col3:
        hour          = st.slider("Hour of Day", 0, 23, 3)
        day_of_week   = st.selectbox("Day of Week", [0,1,2,3,4,5,6],
                                      format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
        card_present  = st.radio("Card Present?", ["No (Online/CNP)", "Yes (Physical)"], index=0)
        is_card_present = 0 if "No" in card_present else 1

    if st.button("Analyse Transaction", type="primary", use_container_width=True):
        txn = {
            "amount": amount / exchange_rate, "avg_user_spend": avg_spend / exchange_rate, "credit_limit": credit_limit / exchange_rate,
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

        st.markdown("<br><br>", unsafe_allow_html=True)

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
                st.image(str(path), use_container_width=True)
                st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# Page 4: Dataset Explorer
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Dataset Explorer":
    st.markdown("<h1 style='font-family:DM Serif Display,serif;font-size:2rem;'>Dataset Explorer</h1>", unsafe_allow_html=True)

    csv_path = DATA_DIR / "transactions.csv"
    sample_path = DATA_DIR / "sample_transactions.csv"
    
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    elif sample_path.exists():
        df = pd.read_csv(sample_path)
        st.info("Using a 10,000 row sample dataset for demonstration purposes because the full 1M row dataset is too large to host on GitHub.")
    else:
        st.warning("Dataset not found. Run `python data/generate.py` first.")
        st.stop()
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