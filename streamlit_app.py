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
    page_title="Medical Cost Prediction",
    page_icon="🏥",
    layout="centered",
)


# ============================================================
# 1A. USER-INTERFACE STYLING
# Adaptive professional blue theme for light and dark modes
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --app-primary: var(--st-primary-color, #4f8fc9);
        --app-bg: var(--st-background-color, #f7fbff);
        --app-surface: var(--st-secondary-background-color, #ffffff);
        --app-text: var(--st-text-color, #173b5e);
        --app-border: var(--st-border-color, rgba(52,120,183,.18));
        --app-soft: color-mix(in srgb, var(--app-primary) 10%, var(--app-surface));
        --app-soft-2: color-mix(in srgb, var(--app-primary) 18%, var(--app-surface));
        --app-muted: color-mix(in srgb, var(--app-text) 68%, transparent);
        --app-shadow: 0 10px 28px color-mix(in srgb, var(--app-text) 10%, transparent);
        --app-shadow-soft: 0 5px 16px color-mix(in srgb, var(--app-text) 7%, transparent);
    }

    .stApp {
        background: linear-gradient(
            180deg,
            color-mix(in srgb, var(--app-primary) 5%, var(--app-bg)) 0%,
            var(--app-bg) 42%,
            color-mix(in srgb, var(--app-primary) 3%, var(--app-bg)) 100%
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
        padding: 1.55rem 1.65rem;
        border: 1px solid var(--app-border);
        border-radius: 20px;
        margin-bottom: 1.1rem;
        background: linear-gradient(135deg, var(--app-surface), var(--app-soft));
        box-shadow: var(--app-shadow);
    }

    .app-hero h1 {
        margin: 0;
        color: var(--app-text);
        font-size: 2.05rem;
        line-height: 1.2;
        font-weight: 760;
    }

    .app-hero p {
        margin: .62rem 0 0 0;
        color: var(--app-muted);
        font-size: 1rem;
        line-height: 1.55;
        max-width: 840px;
    }

    .info-card {
        padding: 1.08rem 1.15rem;
        min-height: 116px;
        border: 1px solid var(--app-border);
        border-radius: 16px;
        background: var(--app-surface);
        box-shadow: var(--app-shadow-soft);
        transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
    }

    .info-card:hover {
        transform: translateY(-2px);
        border-color: color-mix(in srgb, var(--app-primary) 38%, var(--app-border));
        box-shadow: var(--app-shadow);
    }

    .info-card-title {
        color: var(--app-primary);
        font-size: .76rem;
        font-weight: 760;
        text-transform: uppercase;
        letter-spacing: .065em;
        margin-bottom: .38rem;
    }

    .info-card-value {
        color: var(--app-text);
        font-size: 1.06rem;
        font-weight: 760;
        margin-bottom: .24rem;
    }

    .info-card-text {
        color: var(--app-muted);
        font-size: .89rem;
        line-height: 1.45;
    }

    div[data-testid="stForm"] {
        padding: 1.25rem 1.3rem 1.35rem;
        border: 1px solid var(--app-border);
        border-radius: 20px;
        background: var(--app-surface);
        box-shadow: var(--app-shadow);
    }

    /* Input fields */
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

    input, textarea {
        color: var(--app-text) !important;
        caret-color: var(--app-primary) !important;
    }

    input::placeholder, textarea::placeholder {
        color: var(--app-muted) !important;
    }

    hr { border-color: var(--app-border) !important; }

    div[data-testid="stAlert"] {
        border: 1px solid var(--app-border);
        border-radius: 13px;
        box-shadow: var(--app-shadow-soft);
    }

    button { border-radius: 10px !important; }

    div[data-testid="stFormSubmitButton"] button {
        min-height: 3.1rem;
        border: 1px solid var(--app-primary) !important;
        border-radius: 12px !important;
        background: linear-gradient(
            135deg,
            color-mix(in srgb, var(--app-primary) 90%, #ffffff),
            var(--app-primary)
        ) !important;
        color: #ffffff !important;
        font-size: 1.05rem !important;
        font-weight: 800 !important;
        letter-spacing: .15px;
        box-shadow: 0 8px 20px color-mix(in srgb, var(--app-primary) 28%, transparent);
    }

    div[data-testid="stFormSubmitButton"] button:hover {
        transform: translateY(-1px);
        filter: brightness(1.04);
    }

    div[data-testid="stFormSubmitButton"] button,
    div[data-testid="stFormSubmitButton"] button span,
    div[data-testid="stFormSubmitButton"] button p,
    div[data-testid="stFormSubmitButton"] button div {
        color: #ffffff !important;
        font-weight: 800 !important;
    }

    div[data-testid="stDownloadButton"] button,
    .stButton button {
        border: 1px solid var(--app-border) !important;
        background: var(--app-surface) !important;
        color: var(--app-primary) !important;
        font-weight: 650 !important;
    }

    div[data-testid="stDownloadButton"] button:hover,
    .stButton button:hover {
        border-color: var(--app-primary) !important;
        background: var(--app-soft) !important;
    }

    div[data-testid="stMetric"] {
        padding: .95rem 1.05rem;
        border: 1px solid var(--app-border);
        border-radius: 16px;
        background: linear-gradient(145deg, var(--app-surface), var(--app-soft));
        box-shadow: var(--app-shadow-soft);
    }

    div[data-testid="stMetricLabel"] { color: var(--app-muted); }
    div[data-testid="stMetricValue"] { color: var(--app-text); }

    div[data-testid="stExpander"] {
        overflow: hidden;
        border: 1px solid var(--app-border);
        border-radius: 14px;
        background: var(--app-surface);
        box-shadow: var(--app-shadow-soft);
    }

    div[data-testid="stExpander"] details summary {
        color: var(--app-text);
        font-weight: 650;
    }

    div[data-testid="stDataFrame"] {
        overflow: hidden;
        border: 1px solid var(--app-border);
        border-radius: 12px;
        box-shadow: var(--app-shadow-soft);
    }

    button[data-baseweb="tab"] {
        color: var(--app-muted);
        font-weight: 650;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--app-primary);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--app-soft) 0%, var(--app-surface) 58%, var(--app-bg) 100%);
        border-right: 1px solid var(--app-border);
    }

    section[data-testid="stSidebar"] > div { padding-top: 1.35rem; }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        color: var(--app-text);
    }

    .st-key-floating_chat_launcher {
        position: fixed;
        right: 24px;
        bottom: 24px;
        z-index: 999999;
        width: auto;
        animation: floatingChatButton 3.2s ease-in-out infinite;
    }

    @keyframes floatingChatButton {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-6px); }
    }

    .st-key-floating_chat_launcher button[data-testid="stPopoverButton"] {
        min-width: 94px;
        min-height: 62px;
        padding: .72rem 1rem !important;
        border: 2px solid color-mix(in srgb, var(--app-primary) 30%, var(--app-surface)) !important;
        border-radius: 999px !important;
        background: linear-gradient(
            135deg,
            color-mix(in srgb, var(--app-primary) 24%, var(--app-surface)),
            color-mix(in srgb, var(--app-primary) 38%, var(--app-surface))
        ) !important;
        color: var(--app-text) !important;
        box-shadow: var(--app-shadow);
    }

    .st-key-floating_chat_launcher button[data-testid="stPopoverButton"]:hover {
        transform: scale(1.04);
        filter: brightness(1.03);
    }

    .st-key-floating_chat_launcher button[data-testid="stPopoverButton"] span,
    .st-key-floating_chat_launcher button[data-testid="stPopoverButton"] p,
    .st-key-floating_chat_launcher button[data-testid="stPopoverButton"] div {
        color: var(--app-text) !important;
        font-size: 1.25rem !important;
        font-weight: 800 !important;
        line-height: 1.1 !important;
    }

    div[data-baseweb="popover"] {
        max-width: min(420px, calc(100vw - 32px));
    }

    div[data-baseweb="popover"] > div {
        border: 1px solid var(--app-border);
        border-radius: 16px;
        background: var(--app-surface);
        color: var(--app-text);
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

    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: var(--app-soft); }
    ::-webkit-scrollbar-thumb {
        border: 2px solid var(--app-soft);
        border-radius: 999px;
        background: color-mix(in srgb, var(--app-primary) 55%, transparent);
    }

    @media (max-width: 700px) {
        .block-container {
            padding-left: .85rem;
            padding-right: .85rem;
            padding-top: 1.05rem;
        }

        .app-hero {
            padding: 1.2rem 1.1rem;
            border-radius: 17px;
        }

        .app-hero h1 { font-size: 1.58rem; }
        .info-card { min-height: auto; }
        div[data-testid="stForm"] { padding: 1rem .9rem 1.1rem; }
        .st-key-floating_chat_launcher { right: 16px; bottom: 16px; }

        .st-key-floating_chat_launcher button[data-testid="stPopoverButton"] {
            min-width: 84px;
            min-height: 56px;
            padding: .62rem .85rem !important;
        }

        .st-key-floating_chat_launcher button[data-testid="stPopoverButton"] span,
        .st-key-floating_chat_launcher button[data-testid="stPopoverButton"] p,
        .st-key-floating_chat_launcher button[data-testid="stPopoverButton"] div {
            font-size: 1.08rem !important;
        }
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: .01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: .01ms !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 2. PATHS AND MODEL SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "complete_gradient_boosting_pipeline.pkl"
MODEL_VERSION = "Gradient Boosting 2.0 — Interaction LightGBM + XGBoost Blend"
TARGET_NAME = "log_qc701"
ORIGINAL_TARGET_NAME = "qc701"


# ============================================================
# 3. OPTIONAL STREAMLIT SECRETS & ENV VARS
# ============================================================

def get_secret(key: str) -> str | None:
    """Safely read a Streamlit secret or environment variable."""
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key)


GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
EXCHANGE_RATE_API_KEY = get_secret("EXCHANGE_RATE_API_KEY")

if GEMINI_API_KEY:
    st.caption("🟢 AI assistant ready")
else:
    st.caption("🔴 AI assistant unavailable")


# ============================================================
# 4. HUMAN-READABLE FEATURE LABELS
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

    # Interaction features
    "log_qc7b bmi": "Outpatient cost × BMI",
    "log_qc7b age": "Outpatient cost × Age",
    "log_qc7b log_past_qc701": "Outpatient cost × Previous-year inpatient cost",
    "bmi age": "BMI × Age",
    "bmi log_past_qc701": "BMI × Previous-year inpatient cost",
    "age log_past_qc701": "Age × Previous-year inpatient cost",
    "qc401 age": "Hospitalization × Age",
    "qc401 bmi": "Hospitalization × BMI",
}


# ============================================================
# 5. USER-INPUT MAPPINGS
# ============================================================

YES_NO_MAPPING = {
    "No": 0,
    "Yes": 1,
}

GENDER_MAPPING = {
    "Female": 0,
    "Male": 1,
}

HEALTH_MAPPING = {
    "Excellent": 1,
    "Very good": 2,
    "Good": 3,
    "Fair": 4,
    "Poor": 5,
}

EMPLOYMENT_MAPPING = {
    "Not employed": 0,
    "Employed": 1,
}


def get_readable_feature_name(feature_name: str) -> str:
    """Convert saved model feature names into user-friendly labels."""
    feature_name = str(feature_name).strip()

    if feature_name in FEATURE_LABELS:
        return FEATURE_LABELS[feature_name]

    if " " in feature_name:
        parts = feature_name.split()
        readable_parts = [
            FEATURE_LABELS.get(part, part.replace("_", " ").title())
            for part in parts
        ]
        return " × ".join(readable_parts)

    return feature_name.replace("_", " ").title()


# ============================================================
# 6. CURRENCY OPTIONS
# ============================================================

CURRENCY_OPTIONS = {
    "Chinese Yuan (CNY)": {"code": "CNY", "symbol": "¥"},
    "Malaysian Ringgit (MYR)": {"code": "MYR", "symbol": "RM"},
    "US Dollar (USD)": {"code": "USD", "symbol": "$"},
    "Singapore Dollar (SGD)": {"code": "SGD", "symbol": "S$"},
    "Euro (EUR)": {"code": "EUR", "symbol": "€"},
    "British Pound (GBP)": {"code": "GBP", "symbol": "£"},
}


# ============================================================
# 7. MODEL-ARTIFACT LOADING
# ============================================================

@st.cache_resource
def load_model_artifact(model_path: str) -> dict[str, Any]:
    """Load and validate the latest complete model artifact."""
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(
            "The trained-model file was not found.\n\n"
            f"Expected location:\n{path}\n\n"
            "Place the latest PKL file in the same folder as streamlit_app.py."
        )

    raw = joblib.load(path)

    if not isinstance(raw, dict):
        raise TypeError(
            f"The PKL file must contain a dictionary artifact. Loaded type: {type(raw).__name__}"
        )

    lgb_model = raw.get("lightgbm_model") or raw.get("lgb_model")
    xgb_model = raw.get("xgboost_model") or raw.get("xgb_model")
    lgb_weight = raw.get("lightgbm_weight") or raw.get("blend_weight")

    xgb_weight = raw.get("xgboost_weight")
    if xgb_weight is None and lgb_weight is not None:
        xgb_weight = 1.0 - float(lgb_weight)

    original_feature_names = raw.get("original_feature_names") or raw.get("feature_names")
    final_feature_names = raw.get("final_feature_names") or raw.get("feature_names")
    interaction_source_features = raw.get("interaction_source_features", [])
    preprocessor_spec = raw.get("preprocessor")

    missing = []
    if lgb_model is None:
        missing.append("lightgbm_model / lgb_model")
    if xgb_model is None:
        missing.append("xgboost_model / xgb_model")
    if lgb_weight is None:
        missing.append("lightgbm_weight / blend_weight")
    if original_feature_names is None:
        missing.append("original_feature_names")
    if final_feature_names is None:
        missing.append("final_feature_names / feature_names")

    if missing:
        raise KeyError(
            "The PKL file is missing required components:\n"
            + "\n".join(f"- {item}" for item in missing)
        )

    original_feature_names = [str(f).strip() for f in list(original_feature_names)]
    final_feature_names = [str(f).strip() for f in list(final_feature_names)]
    interaction_source_features = [str(f).strip() for f in list(interaction_source_features)]

    lgb_weight = float(lgb_weight)
    xgb_weight = float(xgb_weight)

    if not (0.0 <= lgb_weight <= 1.0 and 0.0 <= xgb_weight <= 1.0):
        raise ValueError("Invalid blend weights (must be between 0.0 and 1.0).")

    if not np.isclose(lgb_weight + xgb_weight, 1.0, atol=1e-6):
        raise ValueError("The LightGBM and XGBoost weights do not add up to 1.")

    expected_count = len(final_feature_names)
    lgb_count = getattr(lgb_model, "n_features_in_", None)
    xgb_count = getattr(xgb_model, "n_features_in_", None)

    if lgb_count is not None and int(lgb_count) != expected_count:
        raise ValueError(
            f"LightGBM feature count mismatch. Model expects: {int(lgb_count)}, Final: {expected_count}"
        )
    if xgb_count is not None and int(xgb_count) != expected_count:
        raise ValueError(
            f"XGBoost feature count mismatch. Model expects: {int(xgb_count)}, Final: {expected_count}"
        )

    return {
        "raw_artifact": raw,
        "lgb_model": lgb_model,
        "xgb_model": xgb_model,
        "lgb_weight": lgb_weight,
        "xgb_weight": xgb_weight,
        "original_feature_names": original_feature_names,
        "interaction_source_features": interaction_source_features,
        "final_feature_names": final_feature_names,
        "feature_names": final_feature_names,
        "preprocessor_spec": preprocessor_spec,
        "target_name": raw.get("target_name", TARGET_NAME),
        "target_transformation": raw.get("target_transformation", "log1p"),
        "model_name": raw.get("model_type", MODEL_VERSION),
        "artifact_version": raw.get("artifact_version", "unknown"),
    }


# ============================================================
# 8. FEATURE-ENGINEERING HELPERS
# ============================================================

def safe_log1p(value: float) -> float:
    """Apply log1p to a non-negative monetary value."""
    value = float(value)
    if value < 0:
        raise ValueError("Cost values cannot be negative.")
    return float(np.log1p(value))


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
    """Create all currently supported ORIGINAL predictor values."""
    if height_cm <= 0 or weight_kg <= 0:
        raise ValueError("Height and Weight must be greater than zero.")

    bmi = float(weight_kg / ((height_cm / 100.0) ** 2))

    values = {
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
    return values


def create_original_model_input(
    *,
    required_original_features: list[str],
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
) -> pd.DataFrame:
    """Create one row containing the exact ORIGINAL predictors stored in the PKL."""
    candidates = create_feature_candidates(
        age=age,
        gender_code=gender_code,
        height_cm=height_cm,
        weight_kg=weight_kg,
        chronic_code=chronic_code,
        smoking_code=smoking_code,
        previous_inpatient_cost=previous_inpatient_cost,
        hospitalized_code=hospitalized_code,
        outpatient_cost=outpatient_cost,
        health_code=health_code,
        employed_code=employed_code,
    )

    missing = [f for f in required_original_features if f not in candidates]
    if missing:
        raise ValueError(
            "The application cannot create all original predictors required by the PKL:\n"
            + "\n".join(f"- {f}" for f in missing)
        )

    model_input = pd.DataFrame(
        [{f: candidates[f] for f in required_original_features}]
    )
    model_input = model_input.loc[:, required_original_features].apply(
        pd.to_numeric, errors="coerce"
    )

    if model_input.isna().any().any():
        invalid = model_input.columns[model_input.isna().any()].tolist()
        raise ValueError(f"Generated original input contains invalid values: {invalid}")

    if np.isinf(model_input.to_numpy(dtype=float)).any():
        raise ValueError("Generated original input contains infinite values.")

    return model_input


def create_engineered_model_input(
    *,
    original_input: pd.DataFrame,
    interaction_source_features: list[str],
    final_feature_names: list[str],
) -> pd.DataFrame:
    """Recreate pairwise interaction features and enforce final feature order."""
    engineered = original_input.copy()

    missing_sources = [
        f for f in interaction_source_features if f not in engineered.columns
    ]
    if missing_sources:
        raise KeyError(
            f"Interaction source variables missing from input: {missing_sources}"
        )

    for left_idx in range(len(interaction_source_features)):
        for right_idx in range(left_idx + 1, len(interaction_source_features)):
            left = interaction_source_features[left_idx]
            right = interaction_source_features[right_idx]
            engineered[f"{left} {right}"] = engineered[left] * engineered[right]

    missing_final = [f for f in final_feature_names if f not in engineered.columns]
    if missing_final:
        raise KeyError(
            f"Failed to recreate all final model features required by PKL: {missing_final}"
        )

    engineered = engineered.loc[:, final_feature_names].apply(
        pd.to_numeric, errors="coerce"
    )

    if engineered.isna().any().any():
        invalid = engineered.columns[engineered.isna().any()].tolist()
        raise ValueError(f"Engineered input contains invalid values: {invalid}")

    if np.isinf(engineered.to_numpy(dtype=float)).any():
        raise ValueError("Engineered input contains infinite values.")

    return engineered


# ============================================================
# 8A. NEARBY HEALTHCARE FACILITY LOCATOR (ROBUST OVERPASS)
# ============================================================

OVERPASS_API_URLS = [
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]

HEALTHCARE_SEARCH_RADIUS_M = 5000
HEALTHCARE_MAX_RESULTS = 10


def haversine_distance_km(
    latitude_1: float,
    longitude_1: float,
    latitude_2: float,
    longitude_2: float,
) -> float:
    """Calculate straight-line distance between two GPS coordinates."""
    earth_radius_km = 6371.0088

    lat1, lon1 = np.radians(float(latitude_1)), np.radians(float(longitude_1))
    lat2, lon2 = np.radians(float(latitude_2)), np.radians(float(longitude_2))

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    a = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2.0) ** 2
    )
    return float(2.0 * earth_radius_km * np.arcsin(np.sqrt(a)))


@st.cache_data(ttl=600, show_spinner=False)
def search_nearby_healthcare_facilities(
    latitude: float,
    longitude: float,
    radius_m: int = HEALTHCARE_SEARCH_RADIUS_M,
) -> list[dict[str, Any]]:
    """Retrieve nearby hospitals and clinics using reliable Overpass failover."""
    query = f"""
    [out:json][timeout:15];
    (
      nwr["amenity"="hospital"](around:{radius_m},{latitude},{longitude});
      nwr["amenity"="clinic"](around:{radius_m},{latitude},{longitude});
    );
    out center tags;
    """

    payload = None
    last_error = None

    for api_url in OVERPASS_API_URLS:
        try:
            response = requests.post(
                api_url,
                data=query,
                headers={"User-Agent": "MedicalCostPredictionApp/1.0 (Contact: research@local)"},
                timeout=18,
            )
            if response.status_code == 200:
                payload = response.json()
                break
        except (requests.RequestException, ValueError) as err:
            last_error = err
            continue

    if payload is None:
        return []

    elements = payload.get("elements", [])
    facilities: list[dict[str, Any]] = []

    for element in elements:
        tags = element.get("tags") or {}
        element_lat = element.get("lat") or (element.get("center") or {}).get("lat")
        element_lon = element.get("lon") or (element.get("center") or {}).get("lon")

        if element_lat is None or element_lon is None:
            continue

        amenity = str(tags.get("amenity", "")).lower()
        if amenity not in {"hospital", "clinic"}:
            continue

        name = str(
            tags.get("name")
            or tags.get("official_name")
            or tags.get("short_name")
            or "Unnamed healthcare facility"
        ).strip()

        street = str(tags.get("addr:street", "")).strip()
        house_number = str(tags.get("addr:housenumber", "")).strip()
        city = str(
            tags.get("addr:city")
            or tags.get("addr:town")
            or tags.get("addr:village")
            or ""
        ).strip()
        postcode = str(tags.get("addr:postcode", "")).strip()

        address_parts = []
        if house_number and street:
            address_parts.append(f"{house_number} {street}")
        elif street:
            address_parts.append(street)
        if city:
            address_parts.append(city)
        if postcode:
            address_parts.append(postcode)

        address = ", ".join(address_parts) or "Address not available"
        phone = str(tags.get("phone") or tags.get("contact:phone") or "").strip()
        website = str(tags.get("website") or tags.get("contact:website") or "").strip()

        distance_km = haversine_distance_km(
            latitude,
            longitude,
            float(element_lat),
            float(element_lon),
        )

        osm_id = f"{element.get('type', 'unknown')}/{element.get('id', '')}"

        facilities.append(
            {
                "name": name,
                "type": "Hospital" if amenity == "hospital" else "Clinic",
                "latitude": float(element_lat),
                "longitude": float(element_lon),
                "distance_km": distance_km,
                "address": address,
                "phone": phone,
                "website": website,
                "osm_id": osm_id,
            }
        )

    unique_facilities: dict[str, dict[str, Any]] = {}
    for facility in facilities:
        key = (
            facility["name"].strip().lower(),
            round(facility["latitude"], 5),
            round(facility["longitude"], 5),
        )
        existing = unique_facilities.get(str(key))
        if existing is None or facility["distance_km"] < existing["distance_km"]:
            unique_facilities[str(key)] = facility

    return sorted(
        unique_facilities.values(),
        key=lambda item: item["distance_km"],
    )[:HEALTHCARE_MAX_RESULTS]


def build_google_maps_search_url(
    name: str,
    latitude: float,
    longitude: float,
) -> str:
    """Build a Google Maps search URL for a facility."""
    query = quote(f"{name} {latitude},{longitude}")
    return f"https://www.google.com/maps/search/?api=1&query={query}"


def build_google_maps_directions_url(
    latitude: float,
    longitude: float,
) -> str:
    """Build a Google Maps directions URL to a facility."""
    return f"https://www.google.com/maps/dir/?api=1&destination={latitude},{longitude}"


# ============================================================
# 9. PREDICTION SERVICE
# ============================================================

def predict_medical_cost(
    *,
    artifact: dict[str, Any],
    original_input: pd.DataFrame,
) -> dict[str, Any]:
    """Generate the blended LightGBM + XGBoost prediction."""
    engineered_input = create_engineered_model_input(
        original_input=original_input,
        interaction_source_features=artifact["interaction_source_features"],
        final_feature_names=artifact["final_feature_names"],
    )

    lgb_model = artifact["lgb_model"]
    xgb_model = artifact["xgb_model"]

    lgb_log_prediction = float(
        np.asarray(lgb_model.predict(engineered_input)).reshape(-1)[0]
    )

    xgb_log_prediction = float(
        np.asarray(xgb_model.predict(engineered_input)).reshape(-1)[0]
    )

    blended_log_prediction = float(
        artifact["lgb_weight"] * lgb_log_prediction
        + artifact["xgb_weight"] * xgb_log_prediction
    )

    predicted_original_cost = float(max(0.0, np.expm1(blended_log_prediction)))

    return {
        "predicted_log_cost": blended_log_prediction,
        "predicted_original_cost": predicted_original_cost,
        "lgb_log_prediction": lgb_log_prediction,
        "xgb_log_prediction": xgb_log_prediction,
        "original_input": original_input,
        "engineered_input": engineered_input,
        "transformed_input": engineered_input,
    }


# ============================================================
# 10. SHAP HELPERS
# ============================================================

@st.cache_resource
def create_shap_explainers(_lgb_model: Any, _xgb_model: Any):
    """Lazily import SHAP and create cached tree explainers."""
    import shap
    return (
        shap.TreeExplainer(_lgb_model),
        shap.TreeExplainer(_xgb_model),
    )


def calculate_top_contributors(
    *,
    artifact: dict[str, Any],
    prediction_result: dict[str, Any],
    top_n: int = 5,
) -> pd.DataFrame:
    """Calculate SHAP contributions for models that contribute to the blend."""
    model_input = prediction_result["transformed_input"]

    if isinstance(model_input, pd.DataFrame):
        shap_input = model_input.copy()
    else:
        shap_input = np.asarray(model_input)

    lgb_model = artifact["lgb_model"]
    xgb_model = artifact["xgb_model"]
    lgb_weight = float(artifact["lgb_weight"])
    xgb_weight = float(artifact["xgb_weight"])

    final_feature_names = list(
        artifact.get("final_feature_names", artifact.get("feature_names", []))
    )

    weighted_shap_vectors = []
    import shap

    if lgb_weight > 1e-12:
        lgb_explainer = shap.TreeExplainer(lgb_model)
        lgb_result = lgb_explainer(shap_input)
        lgb_values = np.asarray(getattr(lgb_result, "values", lgb_result))
        if lgb_values.ndim == 2:
            lgb_values = lgb_values[0]
        elif lgb_values.ndim == 3:
            lgb_values = lgb_values[0, :, 0]
        weighted_shap_vectors.append(lgb_weight * lgb_values)

    if xgb_weight > 1e-12:
        xgb_explainer = shap.TreeExplainer(xgb_model)
        xgb_result = xgb_explainer(shap_input)
        xgb_values = np.asarray(getattr(xgb_result, "values", xgb_result))
        if xgb_values.ndim == 2:
            xgb_values = xgb_values[0]
        elif xgb_values.ndim == 3:
            xgb_values = xgb_values[0, :, 0]
        weighted_shap_vectors.append(xgb_weight * xgb_values)

    if not weighted_shap_vectors:
        raise ValueError("Neither model has a positive blending weight.")

    blended_values = np.sum(weighted_shap_vectors, axis=0)

    contribution_df = pd.DataFrame(
        {
            "Feature": [
                get_readable_feature_name(feature)
                for feature in final_feature_names
            ],
            "Raw feature name": final_feature_names,
            "SHAP contribution": blended_values,
        }
    )

    contribution_df["Absolute contribution"] = contribution_df["SHAP contribution"].abs()
    contribution_df["Effect"] = np.where(
        contribution_df["SHAP contribution"] >= 0,
        "Increased prediction",
        "Decreased prediction",
    )

    return (
        contribution_df.sort_values("Absolute contribution", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


# ============================================================
# 11. CURRENCY CONVERSION
# ============================================================

EXCHANGE_RATE_API_URL = (
    "https://v6.exchangerate-api.com/v6/{api_key}/latest/{base_currency}"
)


@st.cache_data(ttl=3600)
def get_exchange_rates(
    api_key: str,
    base_currency: str = "CNY",
) -> dict[str, Any]:
    """Retrieve exchange rates and cache them for one hour."""
    if not api_key:
        raise ValueError("EXCHANGE_RATE_API_KEY is missing.")

    url = EXCHANGE_RATE_API_URL.format(api_key=api_key, base_currency=base_currency)
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()

    if data.get("result") != "success":
        raise RuntimeError(
            f"ExchangeRate-API error: {data.get('error-type', 'unknown-error')}"
        )

    rates = data.get("conversion_rates", {})
    return {
        "rates": rates,
        "last_updated": data.get("time_last_update_utc"),
    }


def get_currency_rate_from_cny(
    *,
    target_currency: str,
    api_key: str | None,
) -> dict[str, Any]:
    """Return the rate expressed as: 1 CNY = rate * target_currency."""
    target_currency = str(target_currency).upper().strip()
    if target_currency == "CNY":
        return {"rate": 1.0, "last_updated": None}

    if not api_key:
        raise ValueError("EXCHANGE_RATE_API_KEY was not found.")

    rate_data = get_exchange_rates(api_key=api_key, base_currency="CNY")
    rates = rate_data["rates"]

    if target_currency not in rates:
        raise ValueError(f"Unsupported currency: {target_currency}")

    rate = float(rates[target_currency])
    return {"rate": rate, "last_updated": rate_data["last_updated"]}


def convert_selected_currency_to_cny(
    *,
    amount: float,
    source_currency: str,
    rate_from_cny: float,
) -> float:
    """Convert a user-entered amount into CNY."""
    amount = float(amount)
    rate_from_cny = float(rate_from_cny)
    if amount < 0:
        raise ValueError("Currency amounts cannot be negative.")
    if source_currency == "CNY":
        return amount
    return float(amount / rate_from_cny)


def convert_cny_to_selected_currency(
    *,
    amount_cny: float,
    target_currency: str,
    rate_from_cny: float,
) -> float:
    """Convert a CNY amount into the user's selected currency."""
    amount_cny = float(amount_cny)
    rate_from_cny = float(rate_from_cny)
    if amount_cny < 0:
        raise ValueError("Currency amounts cannot be negative.")
    if target_currency == "CNY":
        return amount_cny
    return float(amount_cny * rate_from_cny)


# ============================================================
# 12. GEMINI HELPER
# ============================================================

@st.cache_resource
def load_gemini_client(api_key: str):
    """Create a Gemini client only when a key is available."""
    from google import genai
    return genai.Client(api_key=api_key)


def detect_chat_intent(
    *,
    user_message: str,
    prediction_context: dict[str, Any],
) -> dict[str, Any]:
    """Ask Gemini to classify the user's message."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    client = load_gemini_client(GEMINI_API_KEY)
    current_inputs = prediction_context.get("raw_inputs", {})

    prompt = f"""
You are an intent parser inside a machine-learning medical-cost prediction application.
Classify the user's request into exactly one of these intents:

1. "explanation":
   - Explanations of current predictions, costs, SHAP values, and general prediction queries.
2. "what_if":
   - Use ONLY when the user explicitly asks to simulate or change one or more inputs.

IMPORTANT:
- Return ONLY a valid JSON object.
- Never wrap with markdown backticks or commentary.

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
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    response_text = getattr(response, "text", None)
    if not response_text:
        raise RuntimeError("Gemini returned an empty intent response.")

    cleaned = response_text.strip()
    cleaned = re.sub(r"^`{3}(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*`{3}$", "", cleaned)

    result = json.loads(cleaned)
    intent = str(result.get("intent", "explanation")).strip()
    if intent not in {"explanation", "what_if"}:
        intent = "explanation"

    changes = result.get("changes", {})
    return {"intent": intent, "changes": changes if isinstance(changes, dict) else {}}


def convert_what_if_currency_changes(
    *,
    changes: dict[str, Any],
    prediction_context: dict[str, Any],
) -> dict[str, Any]:
    """Convert hypothetical monetary values into CNY."""
    changes = dict(changes)
    currency_code = prediction_context.get("selected_currency_code", "CNY")
    exchange_rate = float(prediction_context.get("exchange_rate", 1.0) or 1.0)

    selected_outpatient = changes.pop("outpatient_cost_selected_currency", None)
    if selected_outpatient is not None:
        changes["outpatient_cost_cny"] = convert_selected_currency_to_cny(
            amount=float(selected_outpatient),
            source_currency=currency_code,
            rate_from_cny=exchange_rate,
        )

    selected_previous = changes.pop("previous_inpatient_cost_selected_currency", None)
    if selected_previous is not None:
        changes["previous_inpatient_cost_cny"] = convert_selected_currency_to_cny(
            amount=float(selected_previous),
            source_currency=currency_code,
            rate_from_cny=exchange_rate,
        )

    return changes


def validate_what_if_changes(changes: dict[str, Any]) -> dict[str, Any]:
    """Validate Gemini-extracted hypothetical feature changes."""
    allowed_features = {
        "age",
        "gender_label",
        "height_cm",
        "weight_kg",
        "chronic_illness_label",
        "smoking_label",
        "hospitalized_label",
        "health_label",
        "employed_label",
        "outpatient_cost_cny",
        "previous_inpatient_cost_cny",
    }
    cleaned = {}

    for feature, value in changes.items():
        if feature not in allowed_features:
            continue
        if feature == "age":
            value = int(value)
            if not (1 <= value <= 119):
                raise ValueError("Age must be between 1 and 119.")
        elif feature == "height_cm":
            value = float(value)
            if not (50 <= value <= 250):
                raise ValueError("Height must be between 50 and 250 cm.")
        elif feature == "weight_kg":
            value = float(value)
            if not (10 <= value <= 300):
                raise ValueError("Weight must be between 10 and 300 kg.")
        elif feature in {"outpatient_cost_cny", "previous_inpatient_cost_cny"}:
            value = float(value)
            if value < 0:
                raise ValueError("Cost values cannot be negative.")
        cleaned[feature] = value

    if not cleaned:
        raise ValueError("No supported hypothetical feature change was detected.")
    return cleaned


def run_what_if_prediction(
    *,
    artifact: dict[str, Any],
    prediction_context: dict[str, Any],
    changes: dict[str, Any],
) -> dict[str, Any]:
    """Rerun the model using hypothetical changes."""
    original_inputs = dict(prediction_context["raw_inputs"])
    modified_inputs = dict(original_inputs)
    modified_inputs.update(changes)

    new_bmi = validate_raw_inputs(
        age=int(modified_inputs["age"]),
        height_cm=float(modified_inputs["height_cm"]),
        weight_kg=float(modified_inputs["weight_kg"]),
        outpatient_cost=float(modified_inputs["outpatient_cost_cny"]),
        previous_inpatient_cost=float(modified_inputs["previous_inpatient_cost_cny"]),
    )

    new_model_input = create_original_model_input(
        required_original_features=artifact["original_feature_names"],
        age=int(modified_inputs["age"]),
        gender_code=GENDER_MAPPING[modified_inputs["gender_label"]],
        height_cm=float(modified_inputs["height_cm"]),
        weight_kg=float(modified_inputs["weight_kg"]),
        chronic_code=YES_NO_MAPPING[modified_inputs["chronic_illness_label"]],
        smoking_code=YES_NO_MAPPING[modified_inputs["smoking_label"]],
        previous_inpatient_cost=float(modified_inputs["previous_inpatient_cost_cny"]),
        hospitalized_code=YES_NO_MAPPING[modified_inputs["hospitalized_label"]],
        outpatient_cost=float(modified_inputs["outpatient_cost_cny"]),
        health_code=HEALTH_MAPPING[modified_inputs["health_label"]],
        employed_code=EMPLOYMENT_MAPPING[modified_inputs["employed_label"]],
    )

    result = predict_medical_cost(artifact=artifact, original_input=new_model_input)
    new_cost_cny = float(result["predicted_original_cost"])
    currency_code = prediction_context.get("selected_currency_code", "CNY")
    exchange_rate = float(prediction_context.get("exchange_rate", 1.0) or 1.0)

    new_cost_selected = convert_cny_to_selected_currency(
        amount_cny=new_cost_cny,
        target_currency=currency_code,
        rate_from_cny=exchange_rate,
    )

    hypothetical_top_factors = []
    try:
        top_contributors = calculate_top_contributors(
            artifact=artifact,
            prediction_result=result,
            top_n=5,
        )
        hypothetical_top_factors = [
            {
                "feature": str(row["Feature"]),
                "effect": str(row["Effect"]),
                "contribution": float(row["SHAP contribution"]),
            }
            for _, row in top_contributors.iterrows()
        ]
    except Exception:
        pass

    return {
        "changes": dict(changes),
        "original_inputs": original_inputs,
        "modified_inputs": modified_inputs,
        "bmi": new_bmi,
        "prediction_result": result,
        "predicted_log_cost": float(result["predicted_log_cost"]),
        "predicted_cost_cny": new_cost_cny,
        "predicted_cost_selected": new_cost_selected,
        "top_factors": hypothetical_top_factors,
    }


def explain_what_if_prediction(
    *,
    original_context: dict[str, Any],
    what_if_result: dict[str, Any],
    user_message: str,
) -> str:
    """Ask Gemini to explain a recalculated what-if prediction."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    client = load_gemini_client(GEMINI_API_KEY)
    old_cost_cny = float(original_context["predicted_cost_cny"])
    new_cost_cny = float(what_if_result["predicted_cost_cny"])
    difference_cny = new_cost_cny - old_cost_cny
    percentage_change = (
        (difference_cny / old_cost_cny) * 100.0 if old_cost_cny > 0 else 0.0
    )

    currency_code = original_context.get("selected_currency_code", "CNY")
    currency_symbol = original_context.get("selected_currency_symbol", "¥")
    old_cost_selected = float(
        original_context.get("predicted_cost_selected_currency", old_cost_cny)
    )
    new_cost_selected = float(what_if_result["predicted_cost_selected"])

    prompt = f"""
Explain this hypothetical prediction comparison in concise, educational terms:
- Original prediction: {currency_symbol}{old_cost_selected:,.2f} {currency_code} (¥{old_cost_cny:,.2f} CNY)
- Hypothetical prediction: {currency_symbol}{new_cost_selected:,.2f} {currency_code} (¥{new_cost_cny:,.2f} CNY)
- Difference: {difference_cny:+,.2f} CNY ({percentage_change:+.2f}%)
- Changes made: {json.dumps(what_if_result["changes"])}

Explain:
1. What was changed.
2. How the prediction changed.
3. Remind that this is an empirical statistical model association, not real-world guaranteed causation.

User question: {user_message}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return getattr(response, "text", "").strip()


def generate_gemini_explanation(
    *,
    prediction_context: dict[str, Any],
    user_message: str,
) -> str:
    """Generate a concise educational explanation using the latest prediction."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    client = load_gemini_client(GEMINI_API_KEY)
    top_factors = prediction_context.get("top_factors", [])
    factor_text = "\n".join(
        f"- {item['feature']}: {item['effect']} (SHAP: {item['contribution']:.4f})"
        for item in top_factors
    ) or "SHAP feature factors unavailable."

    prompt = f"""
You are an educational assistant for a medical cost prediction research tool.
Predicted cost: ¥{prediction_context['predicted_cost_cny']:,.2f} CNY
Inputs: Age {prediction_context['age']}, BMI {prediction_context['bmi']:.2f}, Gender {prediction_context['gender']}, Chronic {prediction_context['chronic_illness']}, Smoking {prediction_context['smoking_status']}

Key model factors:
{factor_text}

User question:
{user_message}

Provide a concise, helpful explanation. Emphasize that estimates do not represent actual clinical diagnosis or fixed financial charges.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return getattr(response, "text", "").strip()


# ============================================================
# 13. INPUT VALIDATION
# ============================================================

def validate_raw_inputs(
    *,
    age: int,
    height_cm: float,
    weight_kg: float,
    outpatient_cost: float,
    previous_inpatient_cost: float,
) -> float:
    """Validate the raw form values and return calculated BMI."""
    errors = []
    if not (1 <= age <= 119):
        errors.append("Age must be between 1 and 119.")
    if height_cm <= 0:
        errors.append("Height must be greater than zero.")
    if weight_kg <= 0:
        errors.append("Weight must be greater than zero.")
    if outpatient_cost < 0:
        errors.append("Outpatient cost cannot be negative.")
    if previous_inpatient_cost < 0:
        errors.append("Previous inpatient cost cannot be negative.")

    bmi = float(weight_kg / ((height_cm / 100.0) ** 2))
    if not (10 <= bmi <= 80):
        errors.append("Calculated BMI is outside the 10–80 range. Verify height/weight.")

    if errors:
        raise ValueError(" ".join(errors))
    return bmi


# ============================================================
# 14. SESSION STATE
# ============================================================

if "latest_prediction_context" not in st.session_state:
    st.session_state.latest_prediction_context = None

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {
            "role": "assistant",
            "content": "Hello! Generate a prediction and I can explain the estimated cost and its model factors.",
        }
    ]


# ============================================================
# 15. LOAD THE MODEL
# ============================================================

try:
    artifact = load_model_artifact(str(MODEL_PATH))
except Exception as error:
    st.error("Unable to load the trained model.")
    st.exception(error)
    st.stop()


# ============================================================
# 15A. MODEL SHAPE DIAGNOSTICS
# ============================================================

final_feature_names_for_diagnostics = artifact.get(
    "final_feature_names", artifact.get("feature_names", [])
)
lgb_expected_features = getattr(artifact["lgb_model"], "n_features_in_", "Unknown")
xgb_expected_features = getattr(artifact["xgb_model"], "n_features_in_", "Unknown")


# ============================================================
# 16. APPLICATION HEADER
# ============================================================

st.markdown(
    """
    <div class="app-hero">
        <h1>🏥 Inpatient Medical Cost Predictor</h1>
        <p>
            Enter the individual's information to estimate inpatient
            medical cost using the latest blended Gradient Boosting model.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

overview_col1, overview_col2, overview_col3 = st.columns(3)
with overview_col1:
    st.markdown(
        """
        <div class="info-card">
            <div class="info-card-title">Model</div>
            <div class="info-card-value">LightGBM + XGBoost</div>
            <div class="info-card-text">Weighted ensemble trained on CFPS survey data.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with overview_col2:
    st.markdown(
        """
        <div class="info-card">
            <div class="info-card-title">Output</div>
            <div class="info-card-value">Estimated Cost</div>
            <div class="info-card-text">Predicts log cost and re-transforms to original scale.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with overview_col3:
    st.markdown(
        """
        <div class="info-card">
            <div class="info-card-title">Notice</div>
            <div class="info-card-value">Research Only</div>
            <div class="info-card-text">Result is not a guaranteed bill or clinical diagnosis.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.caption(f"Model version: {MODEL_VERSION} · Saved model: {artifact['model_name']}")


# ============================================================
# 17. MODEL INFORMATION SIDEBAR
# ============================================================

with st.sidebar:
    st.header("About this application")
    st.write(
        "This research prototype estimates inpatient medical cost using "
        "stored interaction-feature rules and a blended LightGBM–XGBoost model."
    )
    st.divider()
    st.subheader("Model details")
    st.write("**Model:**", artifact["model_name"])
    st.write("**Original user inputs:**", len(artifact["original_feature_names"]))
    st.write("**LightGBM weight:**", f"{artifact['lgb_weight']:.4f}")
    st.write("**XGBoost weight:**", f"{artifact['xgb_weight']:.4f}")

    st.divider()
    st.subheader("Model shape diagnostics")
    st.write("**Engineered features:**", len(final_feature_names_for_diagnostics))
    st.write("**LightGBM expects:**", lgb_expected_features)
    st.write("**XGBoost expects:**", xgb_expected_features)


# ============================================================
# 18. USER INPUT FORM
# ============================================================

st.subheader("User feature inputs")

selected_currency_label = st.selectbox(
    "Preferred currency",
    options=list(CURRENCY_OPTIONS.keys()),
    index=None,
    placeholder="Select your preferred currency",
)

if selected_currency_label is None:
    st.info("Please select your preferred currency before entering medical-cost values.")
    submitted = False
else:
    selected_currency = CURRENCY_OPTIONS[selected_currency_label]
    selected_currency_code = selected_currency["code"]
    selected_currency_symbol = selected_currency["symbol"]

    with st.form("medical_cost_form"):
        st.markdown("#### Personal information")
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            age = st.number_input("Age", value=40, step=1)
            gender_label = st.selectbox("Gender", options=list(GENDER_MAPPING.keys()))
        with p_col2:
            height_cm = st.number_input("Height (cm)", value=165.0, step=0.1)
            weight_kg = st.number_input("Weight (kg)", value=60.0, step=0.1)

        calculated_bmi = float(weight_kg / ((height_cm / 100.0) ** 2))
        st.info(f"Calculated BMI: **{calculated_bmi:.2f}**")

        st.divider()
        st.markdown("#### Health and lifestyle information")
        h_col1, h_col2 = st.columns(2)
        with h_col1:
            chronic_illness_label = st.selectbox(
                "Chronic illness diagnosis", options=list(YES_NO_MAPPING.keys())
            )
            smoking_label = st.selectbox(
                "Smoking status", options=list(YES_NO_MAPPING.keys())
            )
        with h_col2:
            hospitalized_label = st.selectbox(
                "Hospitalized during past 6 months", options=list(YES_NO_MAPPING.keys())
            )
            health_label = st.selectbox(
                "Self-rated health", options=list(HEALTH_MAPPING.keys()), index=2
            )

        employed_label = st.selectbox(
            "Employment status", options=list(EMPLOYMENT_MAPPING.keys())
        )

        st.divider()
        st.markdown("#### Medical-cost information")
        c_col1, c_col2 = st.columns(2)
        with c_col1:
            outpatient_cost_selected = st.number_input(
                f"Current outpatient cost ({selected_currency_code})",
                value=0.0,
                step=100.0,
            )
        with c_col2:
            previous_inpatient_cost_selected = st.number_input(
                f"Previous inpatient cost ({selected_currency_code})",
                value=0.0,
                step=100.0,
            )

        form_validation_error = None
        try:
            validate_raw_inputs(
                age=int(age),
                height_cm=float(height_cm),
                weight_kg=float(weight_kg),
                outpatient_cost=float(outpatient_cost_selected),
                previous_inpatient_cost=float(previous_inpatient_cost_selected),
            )
        except Exception as err:
            form_validation_error = str(err)

        if form_validation_error:
            st.warning(form_validation_error)

        submitted = st.form_submit_button(
            "✨ Predict inpatient medical cost",
            use_container_width=True,
            type="primary",
            disabled=form_validation_error is not None,
        )


# ============================================================
# 19. PROCESS THE PREDICTION
# ============================================================

if submitted:
    try:
        rate_info = get_currency_rate_from_cny(
            target_currency=selected_currency_code,
            api_key=EXCHANGE_RATE_API_KEY,
        )
        exchange_rate = float(rate_info["rate"])

        outpatient_cost_cny = convert_selected_currency_to_cny(
            amount=float(outpatient_cost_selected),
            source_currency=selected_currency_code,
            rate_from_cny=exchange_rate,
        )
        previous_inpatient_cost_cny = convert_selected_currency_to_cny(
            amount=float(previous_inpatient_cost_selected),
            source_currency=selected_currency_code,
            rate_from_cny=exchange_rate,
        )

        validated_bmi = validate_raw_inputs(
            age=int(age),
            height_cm=float(height_cm),
            weight_kg=float(weight_kg),
            outpatient_cost=float(outpatient_cost_cny),
            previous_inpatient_cost=float(previous_inpatient_cost_cny),
        )

        model_input = create_original_model_input(
            required_original_features=artifact["original_feature_names"],
            age=int(age),
            gender_code=GENDER_MAPPING[gender_label],
            height_cm=float(height_cm),
            weight_kg=float(weight_kg),
            chronic_code=YES_NO_MAPPING[chronic_illness_label],
            smoking_code=YES_NO_MAPPING[smoking_label],
            previous_inpatient_cost=float(previous_inpatient_cost_cny),
            hospitalized_code=YES_NO_MAPPING[hospitalized_label],
            outpatient_cost=float(outpatient_cost_cny),
            health_code=HEALTH_MAPPING[health_label],
            employed_code=EMPLOYMENT_MAPPING[employed_label],
        )

        prediction_result = predict_medical_cost(
            artifact=artifact, original_input=model_input
        )
        predicted_log_cost = float(prediction_result["predicted_log_cost"])
        predicted_cost_cny = float(prediction_result["predicted_original_cost"])
        predicted_cost_selected = convert_cny_to_selected_currency(
            amount_cny=predicted_cost_cny,
            target_currency=selected_currency_code,
            rate_from_cny=exchange_rate,
        )

        st.success("Prediction completed successfully.")
        st.markdown("#### Predicted inpatient medical cost")

        if selected_currency_code == "CNY":
            st.metric("Prediction (CNY)", f"¥{predicted_cost_cny:,.2f} CNY")
        else:
            r1, r2 = st.columns(2)
            with r1:
                st.metric(
                    f"Selected Currency ({selected_currency_code})",
                    f"{selected_currency_symbol}{predicted_cost_selected:,.2f}",
                )
            with r2:
                st.metric("Model Base Currency (CNY)", f"¥{predicted_cost_cny:,.2f} CNY")

        # Save session context
        st.session_state.latest_prediction_context = {
            "predicted_cost_cny": predicted_cost_cny,
            "predicted_log_cost": predicted_log_cost,
            "selected_currency_label": selected_currency_label,
            "selected_currency_code": selected_currency_code,
            "selected_currency_symbol": selected_currency_symbol,
            "predicted_cost_selected_currency": predicted_cost_selected,
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
            "bmi": validated_bmi,
            "gender": gender_label,
            "chronic_illness": chronic_illness_label,
            "smoking_status": smoking_label,
            "hospitalized": hospitalized_label,
            "health_status": health_label,
            "employment_status": employed_label,
            "top_factors": [],
        }

        try:
            top_contributors = calculate_top_contributors(
                artifact=artifact, prediction_result=prediction_result, top_n=5
            )
            st.divider()
            st.subheader("Top model factors")
            st.bar_chart(
                top_contributors.set_index("Feature")[["Absolute contribution"]],
                use_container_width=True,
            )
            top_factor_context = [
                {
                    "feature": str(r["Feature"]),
                    "effect": str(r["Effect"]),
                    "contribution": float(r["SHAP contribution"]),
                }
                for _, r in top_contributors.iterrows()
            ]
            st.session_state.latest_prediction_context["top_factors"] = top_factor_context
        except Exception as s_err:
            st.warning(f"SHAP explanation unavailable: {s_err}")

    except Exception as error:
        st.error("Prediction failed.")
        st.exception(error)


# ============================================================
# 20. NEARBY HEALTHCARE FACILITIES
# ============================================================

st.divider()
st.subheader("🏥 Nearby healthcare facilities")
st.write(
    "If you would like to seek further professional consultation, "
    "use the button below to find nearby hospitals and clinics."
)

location = streamlit_geolocation()

if location:
    lat = location.get("latitude")
    lon = location.get("longitude")

    if lat is not None and lon is not None:
        with st.spinner("Searching nearby facilities..."):
            facilities = search_nearby_healthcare_facilities(latitude=lat, longitude=lon)

        if not facilities:
            st.info("No nearby facilities found or map server is temporarily busy.")
        else:
            st.success(f"Found {len(facilities)} nearby healthcare facilities.")
            for i, fac in enumerate(facilities, start=1):
                with st.container(border=True):
                    c1, c2 = st.columns([4, 1])
                    with c1:
                        st.markdown(f"**{i}. {fac['name']}**")
                        st.caption(f"{fac['type']} · 📍 {fac['address']}")
                    with c2:
                        st.metric("Distance", f"{fac['distance_km']:.2f} km")

                    a1, a2 = st.columns(2)
                    with a1:
                        st.link_button(
                            "🗺️ Directions",
                            build_google_maps_directions_url(fac["latitude"], fac["longitude"]),
                            use_container_width=True,
                        )
                    with a2:
                        st.link_button(
                            "🔎 Maps",
                            build_google_maps_search_url(fac["name"], fac["latitude"], fac["longitude"]),
                            use_container_width=True,
                        )


# ============================================================
# 21. FLOATING GEMINI CHATBOT
# ============================================================

with st.container(key="floating_chat_launcher"):
    with st.popover("🏥 Ask Me"):
        st.markdown("### Medical Cost Assistant")
        chat_container = st.container(height=280, border=True)
        with chat_container:
            for msg in st.session_state.chat_messages:
                css = "mini-chat-user" if msg["role"] == "user" else "mini-chat-assistant"
                label = "You" if msg["role"] == "user" else "Assistant"
                st.markdown(
                    f'<div class="{css}"><strong>{label}</strong><br>{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )

        with st.form("chat_form", clear_on_submit=True):
            chat_q = st.text_input("Message", placeholder="Ask a question...", label_visibility="collapsed")
            send_btn = st.form_submit_button("Send", use_container_width=True, type="primary")

        if send_btn and chat_q.strip():
            clean_q = chat_q.strip()
            st.session_state.chat_messages.append({"role": "user", "content": clean_q})

            if st.session_state.latest_prediction_context is None:
                resp = "Please generate a prediction first so I have data to explain."
            elif not GEMINI_API_KEY:
                resp = "Gemini API key is not configured."
            else:
                try:
                    ctx = st.session_state.latest_prediction_context
                    intent_res = detect_chat_intent(user_message=clean_q, prediction_context=ctx)
                    if intent_res.get("intent") == "what_if":
                        c_changes = convert_what_if_currency_changes(
                            changes=intent_res.get("changes", {}), prediction_context=ctx
                        )
                        v_changes = validate_what_if_changes(c_changes)
                        res = run_what_if_prediction(
                            artifact=artifact, prediction_context=ctx, changes=v_changes
                        )
                        resp = explain_what_if_prediction(
                            original_context=ctx, what_if_result=res, user_message=clean_q
                        )
                    else:
                        resp = generate_gemini_explanation(prediction_context=ctx, user_message=clean_q)
                except Exception as e:
                    resp = f"Assistant error: {e}"

            st.session_state.chat_messages.append({"role": "assistant", "content": resp})
            st.rerun()

st.divider()
st.caption("Research prototype · Predictions are statistical estimates derived from historical data.")
