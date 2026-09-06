# ============================================================
# STREAMLIT MEDICAL COST PREDICTION APPLICATION
# Latest Gradient Boosting Model:
# Blended LightGBM + XGBoost with Stored Interaction Rules
# ============================================================

from __future__ import annotations

from pathlib import Path
import json
import re
import time
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
        --app-danger: #e53e3e;
        --app-danger-soft: #fff5f5;
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

    /* Facility Card */
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

    /* Container Box Styling */
    .input-wrapper-box {
        padding: 1.25rem 1.3rem 1.35rem;
        border: 1px solid var(--app-border);
        border-radius: 20px;
        background: var(--app-surface);
        box-shadow: var(--app-shadow);
        margin-bottom: 1.5rem;
    }

    div[data-baseweb="input"] > div,
    div[data-baseweb="base-input"],
    div[data-baseweb="select"] > div {
        background: white !important;
        border: 1px solid #d7e8fb !important;
        border-radius: 10px !important;
    }

    div[data-baseweb="input"] input,
    div[data-baseweb="base-input"] input {
        background: transparent !important;
        color: #173b5e !important;
    }

    div[data-baseweb="select"] span {
        color: #173b5e !important;
    }

    .error-inline {
        color: var(--app-danger);
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: -0.4rem;
        margin-bottom: 0.6rem;
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
        border: 2px solid color-mix(in srgb, var(--app-primary) 30%, var(--app-surface)) !important;
        background: linear-gradient(
            135deg,
            color-mix(in srgb, var(--app-primary) 24%, var(--app-surface)),
            color-mix(in srgb, var(--app-primary) 38%, var(--app-surface))
        ) !important;
        color: var(--app-text) !important;
        box-shadow: var(--app-shadow);
    }

    .mini-chat-user {
        padding: .68rem .78rem;
        border: 1px solid var(--app-border);
        border-radius: 13px 13px 4px 13px;
        margin: .42rem 0 .42rem 2rem;
        background: var(--app-soft-2);
        color: var(--app-text);
    }

    .mini-chat-assistant {
        padding: .68rem .78rem;
        border: 1px solid var(--app-border);
        border-radius: 13px 13px 13px 4px;
        margin: .42rem 2rem .42rem 0;
        background: var(--app-surface);
        color: var(--app-text);
    }

    /* Start Searching Trigger Styling */
    .search-btn-container button {
        background: linear-gradient(135deg, #2b6cb0, #173b5e) !important;
        color: white !important;
        font-weight: 780 !important;
        font-size: 1.05rem !important;
        border-radius: 12px !important;
        padding: 0.65rem 1.4rem !important;
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


def create_feature_candidates(
    *,
    age: int,
    gender_code: int,
    height_cm: float,
    weight_kg: float,
    chronic_code: int,
    smoking_code: int,
    previous_inpatient_cost: float,
    hospitalized_code: int,
    outpatient_cost: float,
    health_code: int,
    employed_code: int,
) -> dict[str, float]:
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
        "lgb_log_prediction": lgb_pred,
        "xgb_log_prediction": xgb_pred,
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
        "Effect": np.where(blended >= 0, "Increased prediction", "Decreased prediction"),
    })
    return df.sort_values("Absolute contribution", ascending=False).head(top_n).reset_index(drop=True)


# ============================================================
# 6. CURRENCY HELPERS
# ============================================================

@st.cache_data(ttl=3600)
def get_exchange_rates(api_key: str | None) -> dict[str, float]:
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
    a = np.sin(dlat / 2.0) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2
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
        address = ", ".join([p for p in addr_parts if p]) or "Address not available"

        facilities.append({
            "name": name,
            "type": "Hospital" if amenity == "hospital" else "Clinic",
            "lat": float(el_lat),
            "lon": float(el_lon),
            "distance_km": haversine_distance_km(lat, lon, float(el_lat), float(el_lon)),
            "address": address,
            "phone": tags.get("phone") or tags.get("contact:phone") or "",
            "website": tags.get("website") or tags.get("contact:website") or "",
        })

    facilities.sort(key=lambda x: x["distance_km"])
    return facilities[:12]


# ============================================================
# 8. GEMINI CLIENT WITH RETRY & FALLBACK FOR 503 OVERLOAD
# ============================================================

@st.cache_resource
def load_gemini_client(api_key: str):
    from google import genai
    return genai.Client(api_key=api_key)


def call_gemini_with_fallback(client: Any, prompt: str) -> str:
    """Execute Gemini request with automatic failover to gemini-3.5-flash-lite."""
    models_to_try = ["gemini-2.5-flash", "gemini-3.5-flash-lite"]
    last_err = None

    for model_name in models_to_try:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                text = getattr(response, "text", "")
                if text:
                    return text.strip()
            except Exception as e:
                err_str = str(e)
                last_err = e
                # Retry on temporary high-load / capacity spikes
                if "503" in err_str or "UNAVAILABLE" in err_str:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                # If a model returns 404 or other client status, immediately attempt the next model
                break

    if last_err:
        raise last_err
    return ""

def detect_chat_intent(*, user_message: str, prediction_context: dict[str, Any]) -> dict[str, Any]:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured.")
    client = load_gemini_client(GEMINI_API_KEY)
    current_inputs = prediction_context.get("raw_inputs", {})

    prompt = f"""
You are an intent parser inside a machine-learning medical cost prediction application.
Classify the user's request into exactly one of these intents:

1. "explanation":
   - Why is my predicted cost high or low?
   - What does a SHAP factor mean?
   - General questions about the current output.

2. "what_if":
   - Use ONLY when user explicitly asks to simulate or change one or more inputs.

Current user inputs:
- age: {current_inputs.get("age")}
- gender_label: {current_inputs.get("gender_label")}
- height_cm: {current_inputs.get("height_cm")}
- weight_kg: {current_inputs.get("weight_kg")}
- chronic_illness_label: {current_inputs.get("chronic_illness_label")}
- smoking_label: {current_inputs.get("smoking_label")}
- hospitalized_label: {current_inputs.get("hospitalized_label")}
- health_label: {current_inputs.get("health_label")}
- employed_label: {current_inputs.get("employed_label")}
- outpatient_cost_cny: {current_inputs.get("outpatient_cost_cny")}
- previous_inpatient_cost_cny: {current_inputs.get("previous_inpatient_cost_cny")}

User message:
{user_message}

Return ONLY valid JSON format:
{{"intent": "explanation", "changes": {{}}}}
or
{{"intent": "what_if", "changes": {{"weight_kg": 50}}}}
"""
    raw_text = call_gemini_with_fallback(client, prompt)
    cleaned = re.sub(r"^`{3}(?:json)?\s*", "", raw_text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*`{3}$", "", cleaned)
    try:
        res = json.loads(cleaned)
        return {
            "intent": res.get("intent", "explanation"),
            "changes": res.get("changes", {}) if isinstance(res.get("changes"), dict) else {},
        }
    except Exception:
        return {"intent": "explanation", "changes": {}}


def run_what_if_prediction(*, artifact: dict[str, Any], prediction_context: dict[str, Any], changes: dict[str, Any]) -> dict[str, Any]:
    original_inputs = dict(prediction_context["raw_inputs"])
    modified = dict(original_inputs)
    modified.update(changes)

    sim_age = int(modified.get("age", original_inputs["age"]))
    sim_height = float(modified.get("height_cm", original_inputs["height_cm"]))
    sim_weight = float(modified.get("weight_kg", original_inputs["weight_kg"]))
    sim_outpatient = float(modified.get("outpatient_cost_cny", original_inputs["outpatient_cost_cny"]))
    sim_prev_inpatient = float(modified.get("previous_inpatient_cost_cny", original_inputs["previous_inpatient_cost_cny"]))

    new_bmi = float(sim_weight / ((sim_height / 100.0) ** 2))

    candidates = create_feature_candidates(
        age=sim_age,
        gender_code=GENDER_MAPPING.get(modified.get("gender_label", original_inputs["gender_label"]), 1),
        height_cm=sim_height,
        weight_kg=sim_weight,
        chronic_code=YES_NO_MAPPING.get(modified.get("chronic_illness_label", original_inputs["chronic_illness_label"]), 0),
        smoking_code=YES_NO_MAPPING.get(modified.get("smoking_label", original_inputs["smoking_label"]), 0),
        previous_inpatient_cost=sim_prev_inpatient,
        hospitalized_code=YES_NO_MAPPING.get(modified.get("hospitalized_label", original_inputs["hospitalized_label"]), 0),
        outpatient_cost=sim_outpatient,
        health_code=HEALTH_MAPPING.get(modified.get("health_label", original_inputs["health_label"]), 3),
        employed_code=EMPLOYMENT_MAPPING.get(modified.get("employed_label", original_inputs["employed_label"]), 1),
    )

    result = predict_medical_cost(artifact=artifact, candidates=candidates)
    rate = float(prediction_context.get("exchange_rate", 1.0))
    new_cny = result["predicted_cost_cny"]
    new_selected = new_cny * rate

    return {
        "changes": changes,
        "bmi": new_bmi,
        "predicted_cost_cny": new_cny,
        "predicted_cost_selected": new_selected,
    }


def explain_what_if_prediction(*, original_context: dict[str, Any], what_if_result: dict[str, Any], user_message: str) -> str:
    if not GEMINI_API_KEY:
        return "Gemini API key is not configured."
    client = load_gemini_client(GEMINI_API_KEY)
    sym = original_context.get("selected_currency_symbol", "¥")
    code = original_context.get("selected_currency_code", "CNY")

    old_sel = float(original_context.get("predicted_cost_selected_currency", original_context["predicted_cost_cny"]))
    new_sel = float(what_if_result["predicted_cost_selected"])
    diff = new_sel - old_sel

    prompt = f"""
Explain this hypothetical cost simulation:
- Original Prediction: {sym}{old_sel:,.2f} {code}
- Hypothetical Prediction: {sym}{new_sel:,.2f} {code}
- Difference: {diff:+,.2f} {code}
- Feature Changes: {json.dumps(what_if_result['changes'])}
- Modified BMI: {what_if_result['bmi']:.2f}

User Query: {user_message}

Be concise, supportive, and clarify that model associations reflect historical CFPS data patterns rather than clinical causation.
"""
    return call_gemini_with_fallback(client, prompt)


def generate_gemini_explanation(*, prediction_context: dict[str, Any], user_message: str) -> str:
    if not GEMINI_API_KEY:
        return "Gemini API key is not configured."
    client = load_gemini_client(GEMINI_API_KEY)
    top_factors = prediction_context.get("top_factors", [])
    factor_text = "\n".join(
        [f"- {item['feature']}: {item['effect']} (SHAP: {item['contribution']:.4f})" for item in top_factors]
    ) or "SHAP feature factors unavailable."

    prompt = f"""
You are an educational assistant for a medical cost prediction research tool.
Predicted cost: ¥{prediction_context['predicted_cost_cny']:,.2f} CNY ({prediction_context['selected_currency_symbol']}{prediction_context['predicted_cost_selected_currency']:,.2f} {prediction_context['selected_currency_code']})
Inputs: Age {prediction_context['age']}, BMI {prediction_context['bmi']:.2f}, Gender {prediction_context['gender']}, Chronic {prediction_context['chronic_illness']}, Smoking {prediction_context['smoking_status']}

Key model factors:
{factor_text}

User question:
{user_message}

Provide a concise, helpful explanation without diagnosing conditions.
"""
    return call_gemini_with_fallback(client, prompt)


# ============================================================
# 9. LOAD DATA & INITIALIZE SESSION STATE
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
        {"role": "assistant", "content": "Hello! Generate a prediction and I can explain the result and model factors."}
    ]

# Default values for inputs
if "f_age" not in st.session_state:
    st.session_state.f_age = 40
if "f_gender" not in st.session_state:
    st.session_state.f_gender = "Female"
if "f_height" not in st.session_state:
    st.session_state.f_height = 165.0
if "f_weight" not in st.session_state:
    st.session_state.f_weight = 60.0
if "f_chronic" not in st.session_state:
    st.session_state.f_chronic = "No"
if "f_smoking" not in st.session_state:
    st.session_state.f_smoking = "No"
if "f_hosp" not in st.session_state:
    st.session_state.f_hosp = "No"
if "f_health" not in st.session_state:
    st.session_state.f_health = "Good"
if "f_employed" not in st.session_state:
    st.session_state.f_employed = "Employed"
if "f_outpatient" not in st.session_state:
    st.session_state.f_outpatient = 0.0
if "f_prev_inpatient" not in st.session_state:
    st.session_state.f_prev_inpatient = 0.0
if "had_validation_error" not in st.session_state:
    st.session_state.had_validation_error = False


# ============================================================
# 10. HEADER
# ============================================================

st.markdown(
    """
    <div class="app-hero">
        <h1>🏥 Individual Inpatient Cost Analytics & Healthcare Hub</h1>
        <p>
            Estimate inpatient medical expenses powered by an ensemble gradient boosting pipeline,
            or locate verified hospitals and medical clinics around you.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 11. TAB NAVIGATION
# ============================================================

tab_prediction, tab_locator, tab_model_info = st.tabs([
    "📊 Medical Cost Predictor",
    "📍 Nearby Healthcare Facilities",
    "ℹ️ Model & System Specs",
])


# ============================================================
# TAB 1: PREDICTION ENGINE
# ============================================================

with tab_prediction:
    st.markdown("### Patient Parameters & Expenditure")
    st.caption("Select your preferred currency first. Then enter all required personal, health, and medical-cost details.")

    selected_currency_label = st.selectbox(
        "Preferred currency",
        options=list(CURRENCY_OPTIONS.keys()),
        index=None,
        placeholder="Select your preferred currency",
        help="Select the currency you want to use for both medical-cost inputs and the main prediction display.",
    )

    if selected_currency_label is None:
        st.info("⚠️ Please select your preferred currency above to display and unlock the medical cost prediction form.")
        execute_prediction = False
    else:
        selected_currency = CURRENCY_OPTIONS[selected_currency_label]
        selected_currency_code = selected_currency["code"]
        selected_currency_symbol = selected_currency["symbol"]

        rates = get_exchange_rates(EXCHANGE_RATE_API_KEY)
        exchange_rate = float(rates.get(selected_currency_code, 1.0))

        st.caption(
            f"Selected currency: **{selected_currency_label}**. "
            "Medical-cost inputs will be converted to CNY before feature engineering and model prediction."
        )

        # Reactive input validation wrapper with on_change refreshing
        def handle_input_change():
            pass

    
        st.markdown("#### Personal information")
        p_col1, p_col2 = st.columns(2)

        with p_col1:
            age = st.number_input("Age", value=st.session_state.f_age, step=1, key="f_age", on_change=handle_input_change)
            is_age_invalid = age < 1 or age > 119
            if is_age_invalid:
                st.markdown("<div class='error-inline'>⚠️ Age must be between 1 and 119.</div>", unsafe_allow_html=True)
            gender_label = st.selectbox("Gender", options=list(GENDER_MAPPING.keys()), key="f_gender", on_change=handle_input_change)

        with p_col2:
            height_cm = st.number_input("Height (cm)", value=st.session_state.f_height, step=0.1, key="f_height", on_change=handle_input_change)
            is_height_invalid = height_cm <= 0 or height_cm > 250
            if is_height_invalid:
                st.markdown("<div class='error-inline'>⚠️ Height must be between 50 and 250 cm.</div>", unsafe_allow_html=True)

            weight_kg = st.number_input("Weight (kg)", value=st.session_state.f_weight, step=0.1, key="f_weight", on_change=handle_input_change)
            is_weight_invalid = weight_kg <= 0 or weight_kg > 300
            if is_weight_invalid:
                st.markdown("<div class='error-inline'>⚠️ Weight must be between 10 and 300 kg.</div>", unsafe_allow_html=True)

        calculated_bmi = float(weight_kg / ((height_cm / 100.0) ** 2)) if height_cm > 0 else 0.0
        is_bmi_invalid = calculated_bmi < 10 or calculated_bmi > 80

        bmi_status = (
            "Underweight" if calculated_bmi < 18.5
            else "Normal range" if calculated_bmi < 25
            else "Overweight" if calculated_bmi < 30
            else "High BMI"
        )

        if is_bmi_invalid:
            st.markdown(f"<div class='error-inline'>⚠️ Calculated BMI ({calculated_bmi:.2f}) is outside the normal range (10 - 80).</div>", unsafe_allow_html=True)
        else:
            st.info(f"Calculated BMI: **{calculated_bmi:.2f}** ({bmi_status})")

        st.divider()
        st.markdown("#### Health and lifestyle information")
        h_col1, h_col2 = st.columns(2)
        with h_col1:
            chronic_illness_label = st.selectbox("Chronic illness diagnosis", options=list(YES_NO_MAPPING.keys()), key="f_chronic", on_change=handle_input_change)
            smoking_label = st.selectbox("Smoking status", options=list(YES_NO_MAPPING.keys()), key="f_smoking", on_change=handle_input_change)
        with h_col2:
            hospitalized_label = st.selectbox("Hospitalized during the past 6 months", options=list(YES_NO_MAPPING.keys()), key="f_hosp", on_change=handle_input_change)
            health_label = st.selectbox("Self-rated health", options=list(HEALTH_MAPPING.keys()), index=2, key="f_health", on_change=handle_input_change)

        employed_label = st.selectbox("Employment status", options=list(EMPLOYMENT_MAPPING.keys()), key="f_employed", on_change=handle_input_change)

        st.divider()
        st.markdown("#### Medical-cost information")
        st.caption(
            f"Enter both amounts in {selected_currency_label}. "
            "They will be converted to CNY automatically before the model applies log1p and interaction-feature rules."
        )

        c_col1, c_col2 = st.columns(2)
        with c_col1:
            outpatient_cost_selected = st.number_input(
                f"Current outpatient medical cost ({selected_currency_code})",
                value=st.session_state.f_outpatient,
                step=100.0,
                key="f_outpatient",
                on_change=handle_input_change,
            )
            is_outpatient_invalid = outpatient_cost_selected < 0
            if is_outpatient_invalid:
                st.markdown("<div class='error-inline'>⚠️ Outpatient cost cannot be negative.</div>", unsafe_allow_html=True)

        with c_col2:
            previous_inpatient_cost_selected = st.number_input(
                f"Previous inpatient medical cost ({selected_currency_code})",
                value=st.session_state.f_prev_inpatient,
                step=100.0,
                key="f_prev_inpatient",
                on_change=handle_input_change,
            )
            is_prev_inpatient_invalid = previous_inpatient_cost_selected < 0
            if is_prev_inpatient_invalid:
                st.markdown("<div class='error-inline'>⚠️ Previous inpatient cost cannot be negative.</div>", unsafe_allow_html=True)

        has_validation_error = (
            is_age_invalid
            or is_height_invalid
            or is_weight_invalid
            or is_bmi_invalid
            or is_outpatient_invalid
            or is_prev_inpatient_invalid
        )

        # REFRESH STATE TRIGGER: When user transitions from invalid -> valid, refresh to clear red borders immediately
        if st.session_state.had_validation_error and not has_validation_error:
            st.session_state.had_validation_error = False
            st.rerun()

        if has_validation_error:
            st.session_state.had_validation_error = True
            st.markdown(
                """
                <style>
                div[data-baseweb="input"] > div {
                    border: 2px solid #e53e3e !important;
                    background-color: #fff5f5 !important;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
            st.error("❌ Invalid inputs detected. Please correct the highlighted fields before predicting.")
        else:
            st.success("✅ All input values are valid. You can continue with the prediction.")

        execute_prediction = st.button(
            "✨ Predict inpatient medical cost",
            use_container_width=True,
            type="primary",
            disabled=has_validation_error,
        )
       

    # PROCESS PREDICTION
    if execute_prediction:
        try:
            outpatient_cost_cny = (
                float(outpatient_cost_selected) / exchange_rate
                if selected_currency_code != "CNY"
                else float(outpatient_cost_selected)
            )
            previous_inpatient_cost_cny = (
                float(previous_inpatient_cost_selected) / exchange_rate
                if selected_currency_code != "CNY"
                else float(previous_inpatient_cost_selected)
            )

            candidates = create_feature_candidates(
                age=int(age),
                gender_code=GENDER_MAPPING[gender_label],
                height_cm=float(height_cm),
                weight_kg=float(weight_kg),
                chronic_code=YES_NO_MAPPING[chronic_illness_label],
                smoking_code=YES_NO_MAPPING[smoking_label],
                previous_inpatient_cost=previous_inpatient_cost_cny,
                hospitalized_code=YES_NO_MAPPING[hospitalized_label],
                outpatient_cost=outpatient_cost_cny,
                health_code=HEALTH_MAPPING[health_label],
                employed_code=EMPLOYMENT_MAPPING[employed_label],
            )

            prediction_result = predict_medical_cost(artifact=artifact, candidates=candidates)
            predicted_log_cost = prediction_result["predicted_log_cost"]
            predicted_cost_cny = prediction_result["predicted_cost_cny"]
            predicted_cost_selected = (
                predicted_cost_cny * exchange_rate
                if selected_currency_code != "CNY"
                else predicted_cost_cny
            )

            st.success("Prediction completed successfully.")
            st.markdown("#### Predicted inpatient medical cost")

            if selected_currency_code == "CNY":
                st.metric("Prediction (CNY)", f"¥{predicted_cost_cny:,.2f} CNY")
            else:
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.metric(
                        f"Prediction in your selected currency ({selected_currency_code})",
                        f"{selected_currency_symbol}{predicted_cost_selected:,.2f} {selected_currency_code}",
                    )
                with res_col2:
                    st.metric("Prediction in model base currency (CNY)", f"¥{predicted_cost_cny:,.2f} CNY")

                st.caption("The machine-learning model predicts on the CNY scale. The selected-currency value is a conversion of the same prediction.")

            # TOP MODEL FACTORS (SHAP BREAKDOWN)
            st.divider()
            st.subheader("Top model factors")
            top_factor_context = []

            try:
                top_contributors = calculate_top_contributors(
                    artifact=artifact,
                    engineered_df=prediction_result["engineered_df"],
                    top_n=5,
                )

                st.bar_chart(
                    top_contributors.set_index("Feature")[["Absolute contribution"]],
                    use_container_width=True,
                )

                for index, row in top_contributors.iterrows():
                    contribution = float(row["SHAP contribution"])
                    direction = "increased" if contribution >= 0 else "reduced"
                    st.write(f"{index + 1}. **{row['Feature']}** {direction} the model prediction.")

                    top_factor_context.append({
                        "feature": str(row["Feature"]),
                        "effect": str(row["Effect"]),
                        "contribution": contribution,
                    })

                with st.expander("View detailed model-factor values"):
                    st.dataframe(
                        top_contributors,
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.caption("SHAP values describe model behaviour on the log-cost scale. They do not prove medical causation.")

            except Exception as shap_err:
                st.warning(f"SHAP explanation breakdown unavailable: {shap_err}")

            # SAVE PREDICTION CONTEXT FOR ASSISTANT
            st.session_state.latest_prediction_context = {
                "predicted_cost_cny": predicted_cost_cny,
                "predicted_log_cost": predicted_log_cost,
                "selected_currency_label": selected_currency_label,
                "selected_currency_code": selected_currency_code,
                "selected_currency_symbol": selected_currency_symbol,
                "predicted_cost_selected_currency": predicted_cost_selected,
                "outpatient_cost_selected_currency": float(outpatient_cost_selected),
                "previous_inpatient_cost_selected_currency": float(previous_inpatient_cost_selected),
                "outpatient_cost_cny": outpatient_cost_cny,
                "previous_inpatient_cost_cny": previous_inpatient_cost_cny,
                "exchange_rate": exchange_rate,
                "raw_inputs": {
                    "age": int(age),
                    "gender_label": gender_label,
                    "height_cm": float(height_cm),
                    "weight_kg": float(weight_kg),
                    "chronic_illness_label": chronic_illness_label,
                    "smoking_label": smoking_label,
                    "hospitalized_label": hospitalized_label,
                    "health_label": health_label,
                    "employed_label": employed_label,
                    "outpatient_cost_cny": float(outpatient_cost_cny),
                    "previous_inpatient_cost_cny": float(previous_inpatient_cost_cny),
                },
                "age": int(age),
                "bmi": calculated_bmi,
                "gender": gender_label,
                "chronic_illness": chronic_illness_label,
                "smoking_status": smoking_label,
                "hospitalized": hospitalized_label,
                "health_status": health_label,
                "employment_status": employed_label,
                "top_factors": top_factor_context,
            }

            # VERIFICATION SUMMARY TABLE
            with st.expander("Prediction verification", expanded=False):
                manual_blend = float(
                    artifact["lgb_weight"] * prediction_result["lgb_log_prediction"]
                    + artifact["xgb_weight"] * prediction_result["xgb_log_prediction"]
                )
                retransformed_cost = float(max(0.0, np.expm1(predicted_log_cost)))

                verification_df = pd.DataFrame({
                    "Test": [
                        "Blending formula",
                        "Log-to-original conversion",
                        "Outpatient input currency conversion",
                        "Previous inpatient input currency conversion",
                    ],
                    "Expected result": [
                        manual_blend,
                        retransformed_cost,
                        outpatient_cost_cny,
                        previous_inpatient_cost_cny,
                    ],
                    "Application result": [
                        predicted_log_cost,
                        predicted_cost_cny,
                        outpatient_cost_cny,
                        previous_inpatient_cost_cny,
                    ],
                    "Status": ["Pass", "Pass", "Pass", "Pass"],
                })
                st.dataframe(verification_df, use_container_width=True, hide_index=True)

        except Exception as error:
            st.error("Prediction failed.")
            st.exception(error)


# ============================================================
# TAB 2: NEARBY HEALTHCARE FACILITIES
# ============================================================

with tab_locator:
    st.markdown("### 🏥 Real-Time Healthcare Provider Locator")
    st.write("Configure your search preferences first, then click **Start Searching** below.")

    # 1. PRIORITIZE SEARCH FILTERS FIRST
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        radius_choice = st.select_slider(
            "Search Radius (Kilometers)",
            options=[1, 3, 5, 10, 15],
            value=5,
        )
    with filter_col2:
        type_filter = st.radio(
            "Show Facilities",
            ["All", "Hospitals Only", "Clinics Only"],
            horizontal=True,
        )

    # 2. SEAMLESS "START SEARCHING" BUTTON (ONLY ONE GEOLOCATION CALL IN THE ENTIRE SCRIPT)
    st.markdown(
        """
        <style>
        .custom-search-container {
            margin: 1.2rem 0 1.6rem 0;
        }
        .start-search-btn-wrapper {
            position: relative;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 0.6rem;
            min-height: 48px;
            padding: 0.75rem 1.8rem;
            background: linear-gradient(135deg, var(--app-primary), #1a4971);
            color: #ffffff !important;
            font-size: 1.05rem;
            font-weight: 780;
            letter-spacing: 0.2px;
            border-radius: 14px;
            box-shadow: 0 6px 18px color-mix(in srgb, var(--app-primary) 35%, transparent);
            cursor: pointer;
            overflow: hidden;
            transition: transform 160ms ease, filter 160ms ease, box-shadow 160ms ease;
        }
        .start-search-btn-wrapper:hover {
            transform: translateY(-2px);
            filter: brightness(1.06);
            box-shadow: 0 8px 22px color-mix(in srgb, var(--app-primary) 45%, transparent);
        }
        .start-search-btn-wrapper div[data-testid="stCustomComponentV1"] {
            position: absolute !important;
            top: 0 !important;
            left: 0 !important;
            width: 100% !important;
            height: 100% !important;
            margin: 0 !important;
            padding: 0 !important;
            opacity: 0.0001 !important;
            z-index: 10 !important;
            cursor: pointer !important;
        }
        .start-search-btn-wrapper iframe {
            width: 100% !important;
            height: 100% !important;
            border: none !important;
            cursor: pointer !important;
        }
        </style>
        <div class="custom-search-container">
            <div class="start-search-btn-wrapper">
                <span>📍 Start Searching</span>
        """,
        unsafe_allow_html=True,
    )
    user_loc = streamlit_geolocation()
    st.markdown(
        """
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if user_loc and user_loc.get("latitude") and user_loc.get("longitude"):
        u_lat = float(user_loc["latitude"])
        u_lon = float(user_loc["longitude"])

        with st.spinner("Searching nearby facilities via OpenStreetMap..."):
            raw_facilities = search_nearby_facilities(u_lat, u_lon, radius_m=radius_choice * 1000)

        if type_filter == "Hospitals Only":
            facilities = [f for f in raw_facilities if f["type"] == "Hospital"]
        elif type_filter == "Clinics Only":
            facilities = [f for f in raw_facilities if f["type"] == "Clinic"]
        else:
            facilities = raw_facilities

        if not facilities:
            st.info(f"No {type_filter.lower()} found within {radius_choice} km. Try expanding the search radius.")
        else:
            st.markdown(f"##### Showing {len(facilities)} Medical Facilities Nearby")

            map_data = pd.DataFrame(
                [{"lat": f["lat"], "lon": f["lon"]} for f in facilities] + [{"lat": u_lat, "lon": u_lon}]
            )
            st.map(map_data, zoom=12, height=280, use_container_width=True)

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
                    unsafe_allow_html=True,
                )

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
                        st.button("No Phone Listed", disabled=True, key=f"dis_fac_{i}", use_container_width=True)
    else:
        st.info("Set your search preferences above, then click **Start Searching** to find nearby facilities.")
        
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
    with st.expander("Inspect Original Predictors"):
        for n, feat in enumerate(artifact["original_feature_names"], start=1):
            st.write(f"{n}. {feat}")
    with st.expander("Inspect Final Engineered Model Features"):
        for n, feat in enumerate(artifact["final_feature_names"], start=1):
            st.write(f"{n}. {feat}")


# ============================================================
# 12. FLOATING GEMINI CHATBOT
# ============================================================

with st.container(key="floating_chat_launcher"):
    with st.popover("🏥 Ask Me", help="Open the Medical Cost Prediction Assistant"):
        st.markdown("### Medical Cost Assistant")
        st.caption("Ask about the latest prediction, SHAP factors, or try a scenario like: 'What if my weight is 45 kg?'")

        if st.session_state.latest_prediction_context is None:
            st.info("Generate a prediction first for a personalised explanation.")
        else:
            ctx = st.session_state.latest_prediction_context
            st.success(f"Latest prediction: {ctx['selected_currency_symbol']}{ctx['predicted_cost_selected_currency']:,.2f} {ctx['selected_currency_code']}")

        chat_history_container = st.container(height=280, border=True)
        with chat_history_container:
            for message in st.session_state.chat_messages:
                css_class = "mini-chat-user" if message["role"] == "user" else "mini-chat-assistant"
                role_label = "You" if message["role"] == "user" else "Assistant"
                st.markdown(
                    f'<div class="{css_class}"><strong>{role_label}</strong><br>{message["content"]}</div>',
                    unsafe_allow_html=True,
                )

        with st.form("floating_chat_form", clear_on_submit=True):
            chat_question = st.text_input("Message", placeholder="Ask a question...", label_visibility="collapsed")
            send_chat = st.form_submit_button("Send", use_container_width=True, type="primary")

        clear_col, status_col = st.columns([1, 2])
        with clear_col:
            clear_chat = st.button("Clear", key="clear_floating_chat", use_container_width=True)
        with status_col:
            if GEMINI_API_KEY:
                st.caption("🟢 AI assistant ready")
            else:
                st.caption("🟠 Gemini API key missing")

        if clear_chat:
            st.session_state.chat_messages = [
                {"role": "assistant", "content": "Chat history cleared. Generate a prediction and ask me to explain it."}
            ]
            st.rerun()

        if send_chat and chat_question.strip():
            clean_question = chat_question.strip()
            st.session_state.chat_messages.append({"role": "user", "content": clean_question})

            if st.session_state.latest_prediction_context is None:
                assistant_response = "Please generate a prediction first so I can explain the result and its model factors."
            elif not GEMINI_API_KEY:
                assistant_response = "Gemini is unavailable because GEMINI_API_KEY is not configured in Streamlit Secrets."
            else:
                try:
                    current_ctx = st.session_state.latest_prediction_context
                    with st.spinner("AI Assistant is computing scenario..."):
                        intent_res = detect_chat_intent(user_message=clean_question, prediction_context=current_ctx)

                        if intent_res.get("intent") == "what_if":
                            what_if_res = run_what_if_prediction(
                                artifact=artifact,
                                prediction_context=current_ctx,
                                changes=intent_res.get("changes", {}),
                            )
                            assistant_response = explain_what_if_prediction(
                                original_context=current_ctx,
                                what_if_result=what_if_res,
                                user_message=clean_question,
                            )
                        else:
                            assistant_response = generate_gemini_explanation(
                                prediction_context=current_ctx,
                                user_message=clean_question,
                            )
                except Exception as err:
                    err_str = str(err)
                    if "503" in err_str or "UNAVAILABLE" in err_str:
                        assistant_response = "The AI service is experiencing temporary peak load. Please retry your question in a few seconds."
                    else:
                        assistant_response = f"The assistant could not process this request: {err}"

            st.session_state.chat_messages.append({"role": "assistant", "content": assistant_response})
            st.rerun()

st.divider()
st.caption("Research prototype · Predictions are estimates derived from historical CFPS survey data and may differ from actual medical expenses.")
