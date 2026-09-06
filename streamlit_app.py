# ============================================================
# STREAMLIT MEDICAL COST PREDICTION APPLICATION
# Latest Gradient Boosting Model:
# Blended LightGBM + XGBoost with Stored Interaction Rules
# ============================================================

from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
import os
from urllib.parse import quote

from streamlit_geolocation import streamlit_geolocation


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Medical Cost Prediction & Healthcare Finder",
    page_icon="🏥",
    layout="centered",
)


# ============================================================
# 1A. USER-INTERFACE STYLING
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --app-primary: var(--st-primary-color, #2b6cb0);
        --app-bg: var(--st-background-color, #f7fbff);
        --app-surface: var(--st-secondary-background-color, #ffffff);
        --app-text: var(--st-text-color, #173b5e);
        --app-border: var(--st-border-color, rgba(52,120,183,.18));
        --app-soft: color-mix(in srgb, var(--app-primary) 8%, var(--app-surface));
        --app-soft-2: color-mix(in srgb, var(--app-primary) 14%, var(--app-surface));
        --app-muted: color-mix(in srgb, var(--app-text) 68%, transparent);
        --app-shadow: 0 10px 24px color-mix(in srgb, var(--app-text) 8%, transparent);
        --app-shadow-soft: 0 4px 12px color-mix(in srgb, var(--app-text) 6%, transparent);
    }

    .stApp {
        background: linear-gradient(
            180deg,
            color-mix(in srgb, var(--app-primary) 4%, var(--app-bg)) 0%,
            var(--app-bg) 45%,
            color-mix(in srgb, var(--app-primary) 2%, var(--app-bg)) 100%
        );
        color: var(--app-text);
    }

    .block-container {
        max-width: 1120px;
        padding-top: 1.8rem;
        padding-bottom: 5.5rem;
    }

    html, body, [class*="css"], p, label, span, small,
    .stMarkdown, .stCaption, h1, h2, h3, h4 {
        color: var(--app-text);
    }

    footer { visibility: hidden; }

    .app-hero {
        padding: 1.6rem 1.7rem;
        border: 1px solid var(--app-border);
        border-radius: 20px;
        margin-bottom: 1.2rem;
        background: linear-gradient(135deg, var(--app-surface), var(--app-soft));
        box-shadow: var(--app-shadow);
    }

    .app-hero h1 {
        margin: 0;
        color: var(--app-text);
        font-size: 2.1rem;
        line-height: 1.2;
        font-weight: 780;
    }

    .app-hero p {
        margin: .6rem 0 0 0;
        color: var(--app-muted);
        font-size: 1rem;
        line-height: 1.55;
        max-width: 880px;
    }

    /* Tabs Styling */
    button[data-baseweb="tab"] {
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        padding: 0.8rem 1.4rem !important;
        border-radius: 12px 12px 0 0 !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--app-primary) !important;
        border-bottom: 3px solid var(--app-primary) !important;
    }

    .info-card {
        padding: 1.1rem;
        min-height: 115px;
        border: 1px solid var(--app-border);
        border-radius: 16px;
        background: var(--app-surface);
        box-shadow: var(--app-shadow-soft);
        transition: transform 160ms ease, box-shadow 160ms ease;
    }

    .info-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--app-shadow);
    }

    .info-card-title {
        color: var(--app-primary);
        font-size: .76rem;
        font-weight: 760;
        text-transform: uppercase;
        letter-spacing: .065em;
        margin-bottom: .35rem;
    }

    .info-card-value {
        color: var(--app-text);
        font-size: 1.1rem;
        font-weight: 780;
        margin-bottom: .25rem;
    }

    .info-card-text {
        color: var(--app-muted);
        font-size: .88rem;
        line-height: 1.45;
    }

    /* Facility Custom Card */
    .facility-card {
        border: 1px solid var(--app-border);
        border-radius: 18px;
        background: var(--app-surface);
        box-shadow: var(--app-shadow-soft);
        padding: 1.25rem 1.35rem;
        margin-bottom: 1.2rem;
        transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
    }

    .facility-card:hover {
        transform: translateY(-2px);
        border-color: color-mix(in srgb, var(--app-primary) 40%, var(--app-border));
        box-shadow: var(--app-shadow);
    }

    .facility-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 0.6rem;
    }

    .facility-name {
        font-size: 1.2rem;
        font-weight: 780;
        color: var(--app-text);
        margin: 0;
    }

    .badge-hospital {
        display: inline-block;
        background: #ebf8ff;
        color: #2b6cb0;
        font-size: 0.76rem;
        font-weight: 700;
        padding: 0.22rem 0.65rem;
        border-radius: 999px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        border: 1px solid rgba(43, 108, 176, 0.25);
    }

    .badge-clinic {
        display: inline-block;
        background: #f0fff4;
        color: #276749;
        font-size: 0.76rem;
        font-weight: 700;
        padding: 0.22rem 0.65rem;
        border-radius: 999px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        border: 1px solid rgba(39, 103, 73, 0.25);
    }

    .distance-tag {
        background: var(--app-soft-2);
        color: var(--app-primary);
        font-weight: 800;
        padding: 0.35rem 0.75rem;
        border-radius: 12px;
        font-size: 0.95rem;
        text-align: right;
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
    }

    .facility-address {
        color: var(--app-muted);
        font-size: 0.92rem;
        line-height: 1.45;
        margin-top: 0.35rem;
    }

    /* Floating Chat Launcher */
    .st-key-floating_chat_launcher {
        position: fixed;
        right: 24px;
        bottom: 24px;
        z-index: 999999;
        width: auto;
    }

    .st-key-floating_chat_launcher button[data-testid="stPopoverButton"] {
        min-width: 90px;
        min-height: 58px;
        border-radius: 999px !important;
        font-weight: 800 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 2. MODEL PATHS AND SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "complete_gradient_boosting_pipeline.pkl"
MODEL_VERSION = "Gradient Boosting 2.0 — Interaction LightGBM + XGBoost Blend"
TARGET_NAME = "log_qc701"


# ============================================================
# 3. SECRETS & ENV CONFIG
# ============================================================

def get_secret(key: str) -> str | None:
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key)


GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
EXCHANGE_RATE_API_KEY = get_secret("EXCHANGE_RATE_API_KEY")


# ============================================================
# 4. ENCODING & LABELS
# ============================================================

FEATURE_LABELS = {
    "age": "Age",
    "gender": "Gender",
    "bmi": "BMI",
    "qp401": "Any chronic illness",
    "qq201": "Smoking status",
    "log_past_qc701": "Previous-year inpatient cost",
    "qc401": "Hospitalized during the past 6 months",
    "log_qc7b": "Outpatient cost",
    "qp201": "Self-rated health",
    "qgb1": "Employment status",
    "health_fair": "Self-rated health: Fair",
    "health_good": "Self-rated health: Good",
    "health_poor": "Self-rated health: Poor",
    "health_very_good": "Self-rated health: Very good",
    "qp102": "Body weight",
    "qp605_s_1": "Medical insurance category",
    "log_qi202": "Retired allowance",
    "log_qc7b bmi": "Outpatient cost × BMI",
    "log_qc7b age": "Outpatient cost × Age",
    "log_qc7b log_past_qc701": "Outpatient cost × Previous-year inpatient cost",
    "bmi age": "BMI × Age",
    "bmi log_past_qc701": "BMI × Previous-year inpatient cost",
    "age log_past_qc701": "Age × Previous-year inpatient cost",
    "qc401 age": "Hospitalization × Age",
    "qc401 bmi": "Hospitalization × BMI",
}

YES_NO_MAPPING = {"No": 0, "Yes": 1}
GENDER_MAPPING = {"Female": 0, "Male": 1}
HEALTH_MAPPING = {"Excellent": 1, "Very good": 2, "Good": 3, "Fair": 4, "Poor": 5}
EMPLOYMENT_MAPPING = {"Not employed": 0, "Employed": 1}

CURRENCY_OPTIONS = {
    "Chinese Yuan (CNY)": {"code": "CNY", "symbol": "¥"},
    "Malaysian Ringgit (MYR)": {"code": "MYR", "symbol": "RM"},
    "US Dollar (USD)": {"code": "USD", "symbol": "$"},
    "Singapore Dollar (SGD)": {"code": "SGD", "symbol": "S$"},
    "Euro (EUR)": {"code": "EUR", "symbol": "€"},
    "British Pound (GBP)": {"code": "GBP", "symbol": "£"},
}


def get_readable_feature_name(feature_name: str) -> str:
    feature_name = str(feature_name).strip()
    if feature_name in FEATURE_LABELS:
        return FEATURE_LABELS[feature_name]
    if " " in feature_name:
        parts = feature_name.split()
        return " × ".join([FEATURE_LABELS.get(p, p.replace("_", " ").title()) for p in parts])
    return feature_name.replace("_", " ").title()


# ============================================================
# 5. MODEL PIPELINE LOADING & INFERENCE
# ============================================================

@st.cache_resource
def load_model_artifact(model_path: str) -> dict[str, Any]:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at: {path}")

    raw = joblib.load(path)
    lgb_model = raw.get("lightgbm_model") or raw.get("lgb_model")
    xgb_model = raw.get("xgboost_model") or raw.get("xgb_model")
    lgb_weight = raw.get("lightgbm_weight") or raw.get("blend_weight")
    xgb_weight = raw.get("xgboost_weight") or (1.0 - float(lgb_weight) if lgb_weight is not None else None)

    original_features = [str(f).strip() for f in (raw.get("original_feature_names") or raw.get("feature_names"))]
    final_features = [str(f).strip() for f in (raw.get("final_feature_names") or raw.get("feature_names"))]
    interaction_sources = [str(f).strip() for f in raw.get("interaction_source_features", [])]

    return {
        "raw_artifact": raw,
        "lgb_model": lgb_model,
        "xgb_model": xgb_model,
        "lgb_weight": float(lgb_weight),
        "xgb_weight": float(xgb_weight),
        "original_feature_names": original_features,
        "interaction_source_features": interaction_sources,
        "final_feature_names": final_features,
        "model_name": raw.get("model_type", MODEL_VERSION),
    }


def safe_log1p(value: float) -> float:
    return float(np.log1p(max(0.0, float(value))))


def create_feature_candidates(*, age: int, gender_code: int, height_cm: float, weight_kg: float,
                              chronic_code: int, smoking_code: int, previous_inpatient_cost: float,
                              hospitalized_code: int, outpatient_cost: float, health_code: int,
                              employed_code: int) -> dict[str, float]:
    bmi = float(weight_kg / ((height_cm / 100.0) ** 2))
    return {
        "age": float(age),
        "gender": float(gender_code),
        "bmi": bmi,
        "qp401": float(chronic_code),
        "qq201": float(smoking_code),
        "log_past_qc701": safe_log1p(previous_inpatient_cost),
        "qc401": float(hospitalized_code),
        "log_qc7b": safe_log1p(outpatient_cost),
        "qp201": float(health_code),
        "qgb1": float(employed_code),
        "qp102": float(weight_kg * 2.0),
        "health_very_good": float(int(health_code == 2)),
        "health_good": float(int(health_code == 3)),
        "health_fair": float(int(health_code == 4)),
        "health_poor": float(int(health_code == 5)),
    }


def create_model_dataframe(artifact: dict[str, Any], candidates: dict[str, float]) -> pd.DataFrame:
    df = pd.DataFrame([{f: candidates[f] for f in artifact["original_feature_names"]}])
    sources = artifact["interaction_source_features"]
    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            df[f"{sources[i]} {sources[j]}"] = df[sources[i]] * df[sources[j]]
    return df.loc[:, artifact["final_feature_names"]].apply(pd.to_numeric, errors="coerce")


def predict_medical_cost(*, artifact: dict[str, Any], candidates: dict[str, float]) -> dict[str, Any]:
    engineered_df = create_model_dataframe(artifact, candidates)
    lgb_pred = float(np.asarray(artifact["lgb_model"].predict(engineered_df)).reshape(-1)[0])
    xgb_pred = float(np.asarray(artifact["xgb_model"].predict(engineered_df)).reshape(-1)[0])

    blended_log = float(artifact["lgb_weight"] * lgb_pred + artifact["xgb_weight"] * xgb_pred)
    predicted_cost = float(max(0.0, np.expm1(blended_log)))

    return {
        "predicted_log_cost": blended_log,
        "predicted_cost_cny": predicted_cost,
        "engineered_df": engineered_df,
    }


def calculate_top_contributors(artifact: dict[str, Any], engineered_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    import shap
    shap_vectors = []
    if artifact["lgb_weight"] > 1e-12:
        explainer = shap.TreeExplainer(artifact["lgb_model"])
        vals = np.asarray(getattr(explainer(engineered_df), "values"))
        shap_vectors.append(artifact["lgb_weight"] * (vals[0] if vals.ndim == 2 else vals[0, :, 0]))

    if artifact["xgb_weight"] > 1e-12:
        explainer = shap.TreeExplainer(artifact["xgb_model"])
        vals = np.asarray(getattr(explainer(engineered_df), "values"))
        shap_vectors.append(artifact["xgb_weight"] * (vals[0] if vals.ndim == 2 else vals[0, :, 0]))

    blended = np.sum(shap_vectors, axis=0)
    df = pd.DataFrame({
        "Feature": [get_readable_feature_name(f) for f in artifact["final_feature_names"]],
        "Raw": artifact["final_feature_names"],
        "SHAP contribution": blended,
        "Absolute contribution": np.abs(blended),
        "Effect": np.where(blended >= 0, "Increased prediction", "Decreased prediction")
    })
    return df.sort_values("Absolute contribution", ascending=False).head(top_n).reset_index(drop=True)


# ============================================================
# 6. CURRENCY HELPERS
# ============================================================

@st.cache_data(ttl=3600)
def get_exchange_rates(api_key: str) -> dict[str, float]:
    if not api_key:
        return {"CNY": 1.0, "MYR": 0.65, "USD": 0.14, "SGD": 0.19, "EUR": 0.13, "GBP": 0.11}
    try:
        url = f"https://v6.exchangerate-api.com/v6/{api_key}/latest/CNY"
        res = requests.get(url, timeout=8).json()
        return res.get("conversion_rates", {})
    except Exception:
        return {"CNY": 1.0, "MYR": 0.65, "USD": 0.14, "SGD": 0.19, "EUR": 0.13, "GBP": 0.11}


# ============================================================
# 7. MAPS & OVERPASS (HEALTHCARE FINDER)
# ============================================================

OVERPASS_API_URLS = [
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0)**2
    return float(2.0 * r * np.arcsin(np.sqrt(a)))


@st.cache_data(ttl=600, show_spinner=False)
def search_nearby_facilities(lat: float, lon: float, radius_m: int = 5000) -> list[dict[str, Any]]:
    query = f"""
    [out:json][timeout:15];
    (
      nwr["amenity"="hospital"](around:{radius_m},{lat},{lon});
      nwr["amenity"="clinic"](around:{radius_m},{lat},{lon});
    );
    out center tags;
    """
    payload = None
    for api_url in OVERPASS_API_URLS:
        try:
            res = requests.post(api_url, data=query, headers={"User-Agent": "HealthFinderApp/1.0"}, timeout=15)
            if res.status_code == 200:
                payload = res.json()
                break
        except Exception:
            continue

    if not payload:
        return []

    facilities = []
    for el in payload.get("elements", []):
        tags = el.get("tags", {})
        el_lat = el.get("lat") or (el.get("center") or {}).get("lat")
        el_lon = el.get("lon") or (el.get("center") or {}).get("lon")
        if el_lat is None or el_lon is None:
            continue

        amenity = str(tags.get("amenity", "")).lower()
        if amenity not in {"hospital", "clinic"}:
            continue

        name = tags.get("name") or tags.get("official_name") or "Unnamed Facility"
        addr_parts = [tags.get("addr:housenumber"), tags.get("addr:street"), tags.get("addr:city")]
        address = ", ".join([p for p in addr_parts if p]) or "Address details not registered"

        facilities.append({
            "name": name,
            "type": "Hospital" if amenity == "hospital" else "Clinic",
            "lat": float(el_lat),
            "lon": float(el_lon),
            "distance_km": haversine_distance_km(lat, lon, float(el_lat), float(el_lon)),
            "address": address,
            "phone": tags.get("phone") or tags.get("contact:phone") or "",
            "website": tags.get("website") or tags.get("contact:website") or ""
        })

    facilities.sort(key=lambda x: x["distance_km"])
    return facilities[:12]


# ============================================================
# 8. LOAD APPLICATION DATA
# ============================================================

try:
    artifact = load_model_artifact(str(MODEL_PATH))
except Exception as e:
    st.error(f"Error loading model pipeline: {e}")
    st.stop()

if "latest_prediction_context" not in st.session_state:
    st.session_state.latest_prediction_context = None

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "assistant", "content": "Hello! Submit a prediction and I can explain the result and factors for you."}
    ]


# ============================================================
# 9. HEADER
# ============================================================

st.markdown(
    """
    <div class="app-hero">
        <h1>🏥 Medical Cost Analytics & Healthcare Hub</h1>
        <p>
            Estimate inpatient medical expenses powered by an ensemble gradient boosting pipeline,
            or locate verified hospitals and medical clinics around you.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 10. TAB NAVIGATION
# ============================================================

tab_prediction, tab_locator, tab_model_info = st.tabs([
    "📊 Medical Cost Predictor",
    "📍 Nearby Healthcare Facilities",
    "ℹ️ Model & System Specs"
])


# ============================================================
# TAB 1: PREDICTION ENGINE
# ============================================================

with tab_prediction:
    st.markdown("### Patient Parameters & Expenditure")
    st.caption("Select your preferred currency to automatically standardize spending inputs into the pipeline.")

    selected_curr_label = st.selectbox(
        "Display and Calculation Currency",
        options=list(CURRENCY_OPTIONS.keys()),
        index=0,
    )
    curr_info = CURRENCY_OPTIONS[selected_curr_label]
    rates = get_exchange_rates(EXCHANGE_RATE_API_KEY)
    exchange_rate = float(rates.get(curr_info["code"], 1.0))

    with st.form("prediction_form"):
        st.markdown("##### Personal Details")
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", 1, 119, 40, 1)
            gender = st.selectbox("Gender", options=list(GENDER_MAPPING.keys()))
        with c2:
            height = st.number_input("Height (cm)", 50.0, 250.0, 168.0, 0.5)
            weight = st.number_input("Weight (kg)", 10.0, 250.0, 65.0, 0.5)

        bmi = float(weight / ((height / 100.0)**2))
        bmi_color = "#2b6cb0" if 18.5 <= bmi <= 24.9 else "#c53030"
        st.markdown(f"<small>Calculated Body Mass Index (BMI): <b style='color:{bmi_color}'>{bmi:.2f}</b></small>", unsafe_allow_html=True)

        st.divider()
        st.markdown("##### Health & Lifestyle Factors")
        h1, h2 = st.columns(2)
        with h1:
            chronic = st.selectbox("Diagnosed with Chronic Illness", options=list(YES_NO_MAPPING.keys()))
            smoking = st.selectbox("Current Smoker", options=list(YES_NO_MAPPING.keys()))
        with h2:
            hospitalized = st.selectbox("Hospitalized in Past 6 Months", options=list(YES_NO_MAPPING.keys()))
            health = st.selectbox("Self-Assessed Health Condition", options=list(HEALTH_MAPPING.keys()), index=2)

        employed = st.selectbox("Employment Status", options=list(EMPLOYMENT_MAPPING.keys()), index=1)

        st.divider()
        st.markdown("##### Associated Medical Expenses")
        e1, e2 = st.columns(2)
        with e1:
            outpatient_in = st.number_input(f"Outpatient Costs ({curr_info['code']})", 0.0, 500000.0, 0.0, 50.0)
        with e2:
            prev_inpatient_in = st.number_input(f"Previous Inpatient Costs ({curr_info['code']})", 0.0, 500000.0, 0.0, 100.0)

        submitted = st.form_submit_button("✨ Compute Estimated Medical Cost", type="primary", use_container_width=True)

    if submitted:
        outpatient_cny = outpatient_in / exchange_rate if curr_info["code"] != "CNY" else outpatient_in
        prev_inpatient_cny = prev_inpatient_in / exchange_rate if curr_info["code"] != "CNY" else prev_inpatient_in

        candidates = create_feature_candidates(
            age=age, gender_code=GENDER_MAPPING[gender], height_cm=height, weight_kg=weight,
            chronic_code=YES_NO_MAPPING[chronic], smoking_code=YES_NO_MAPPING[smoking],
            previous_inpatient_cost=prev_inpatient_cny, hospitalized_code=YES_NO_MAPPING[hospitalized],
            outpatient_cost=outpatient_cny, health_code=HEALTH_MAPPING[health],
            employed_code=EMPLOYMENT_MAPPING[employed]
        )

        res = predict_medical_cost(artifact=artifact, candidates=candidates)
        cost_cny = res["predicted_cost_cny"]
        cost_selected = cost_cny * exchange_rate

        st.success("Prediction calculated successfully.")

        res_c1, res_c2 = st.columns(2)
        with res_c1:
            st.metric(f"Predicted Cost ({curr_info['code']})", f"{curr_info['symbol']} {cost_selected:,.2f}")
        with res_c2:
            st.metric("Base Model Output (CNY)", f"¥ {cost_cny:,.2f}")

        # Explainability
        try:
            top_df = calculate_top_contributors(artifact, res["engineered_df"], top_n=5)
            st.markdown("#### Primary Model Determinants (SHAP Values)")
            st.bar_chart(top_df.set_index("Feature")[["Absolute contribution"]], use_container_width=True)

            st.session_state.latest_prediction_context = {
                "predicted_cost_cny": cost_cny,
                "predicted_log_cost": res["predicted_log_cost"],
                "selected_currency_code": curr_info["code"],
                "selected_currency_symbol": curr_info["symbol"],
                "predicted_cost_selected_currency": cost_selected,
                "age": age, "bmi": bmi, "gender": gender, "chronic_illness": chronic,
                "smoking_status": smoking, "hospitalized": hospitalized,
                "health_status": health, "employment_status": employed,
                "top_factors": top_df.to_dict(orient="records"),
                "raw_inputs": candidates,
            }
        except Exception as e:
            st.warning(f"Feature contributions unavailable: {e}")


# ============================================================
# TAB 2: NEARBY HEALTHCARE FACILITIES (RESTRUCTURED & BEAUTIFIED)
# ============================================================

with tab_locator:
    st.markdown("### 🏥 Real-Time Healthcare Provider Locator")
    st.write(
        "Find nearby accredited hospitals and outpatient clinics around your current coordinates. "
        "Click the location trigger below to start."
    )

    loc_col1, loc_col2 = st.columns([1, 2])
    with loc_col1:
        user_loc = streamlit_geolocation()

    if user_loc and user_loc.get("latitude") and user_loc.get("longitude"):
        u_lat, u_lon = float(user_loc["latitude"]), float(user_loc["longitude"])

        with loc_col2:
            st.success(f"📍 GPS Identified: `{u_lat:.4f}, {u_lon:.4f}`")

        # Map filters & radius
        filter_col1, filter_col2 = st.columns([1, 1])
        with filter_col1:
            radius_choice = st.select_slider(
                "Search Radius (Kilometers)",
                options=[1, 3, 5, 10, 15],
                value=5
            )
        with filter_col2:
            type_filter = st.radio("Show Facilities", ["All", "Hospitals Only", "Clinics Only"], horizontal=True)

        with st.spinner("Scanning OpenStreetMap healthcare nodes..."):
            raw_facilities = search_nearby_facilities(u_lat, u_lon, radius_m=radius_choice * 1000)

        # Apply facility type filter
        if type_filter == "Hospitals Only":
            facilities = [f for f in raw_facilities if f["type"] == "Hospital"]
        elif type_filter == "Clinics Only":
            facilities = [f for f in raw_facilities if f["type"] == "Clinic"]
        else:
            facilities = raw_facilities

        if not facilities:
            st.info(f"No {type_filter.lower()} found within {radius_choice} km. Try expanding the search radius slider.")
        else:
            st.markdown(f"##### Showing {len(facilities)} Medical Facilities Nearby")

            # Geographic Overview Map
            map_data = pd.DataFrame([
                {"lat": f["lat"], "lon": f["lon"]} for f in facilities
            ] + [{"lat": u_lat, "lon": u_lon}])
            st.map(map_data, zoom=12, use_container_width=True)

            # Restructured Cards
            for i, fac in enumerate(facilities, start=1):
                badge_class = "badge-hospital" if fac["type"] == "Hospital" else "badge-clinic"
                st.markdown(
                    f"""
                    <div class="facility-card">
                        <div class="facility-header">
                            <div>
                                <h4 class="facility-name">{i}. {fac['name']}</h4>
                                <span class="{badge_class}">{fac['type']}</span>
                                <div class="facility-address">📍 {fac['address']}</div>
                            </div>
                            <div class="distance-tag">
                                🚗 {fac['distance_km']:.2f} km
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Action toolbar
                b1, b2, b3 = st.columns(3)
                with b1:
                    dir_url = f"https://www.google.com/maps/dir/?api=1&destination={fac['lat']},{fac['lon']}"
                    st.link_button("🗺️ Get Directions", dir_url, use_container_width=True)
                with b2:
                    search_url = f"https://www.google.com/maps/search/?api=1&query={quote(fac['name'] + ' ' + str(fac['lat']) + ',' + str(fac['lon']))}"
                    st.link_button("🔍 Google Maps", search_url, use_container_width=True)
                with b3:
                    if fac["phone"]:
                        clean_phone = re.sub(r"[^0-9+]", "", fac["phone"])
                        st.link_button("📞 Call Clinic", f"tel:{clean_phone}", use_container_width=True)
                    elif fac["website"]:
                        site = fac["website"] if fac["website"].startswith("http") else f"https://{fac['website']}"
                        st.link_button("🌐 Website", site, use_container_width=True)
                    else:
                        st.button("No Phone Listed", disabled=True, key=f"dis_{i}", use_container_width=True)
    else:
        st.info("Click the location button above to identify medical centers near your current location.")


# ============================================================
# TAB 3: SYSTEM SPECIFICATIONS
# ============================================================

with tab_model_info:
    st.markdown("### Model Architecture & Diagnostics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Pipeline Type", "Ensemble Blend")
    c2.metric("LightGBM Weight", f"{artifact['lgb_weight']:.2%}")
    c3.metric("XGBoost Weight", f"{artifact['xgb_weight']:.2%}")

    st.markdown("##### Interaction Variables & Feature Set")
    st.write(f"**Total Features Processed:** {len(artifact['final_feature_names'])}")
    with st.expander("Inspect Model Predictors"):
        st.write(artifact["final_feature_names"])


# ============================================================
# 11. FLOATING ASSISTANT
# ============================================================

with st.container(key="floating_chat_launcher"):
    with st.popover("💬 AI Assistant"):
        st.markdown("##### Medical Cost Explainer")
        chat_box = st.container(height=260)
        for msg in st.session_state.chat_messages:
            role = "You" if msg["role"] == "user" else "Assistant"
            chat_box.write(f"**{role}:** {msg['content']}")

        prompt = st.chat_input("Ask about costs or SHAP values...")
        if prompt:
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            ctx = st.session_state.latest_prediction_context
            reply = f"Your latest predicted cost was {ctx['selected_currency_symbol']}{ctx['predicted_cost_selected_currency']:,.2f}." if ctx else "Submit a prediction first."
            st.session_state.chat_messages.append({"role": "assistant", "content": reply})
            st.rerun()
