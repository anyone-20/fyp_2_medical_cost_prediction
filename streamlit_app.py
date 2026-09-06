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

    /* ===============================
    Input fields
    =============================== */

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

# Place the latest PKL file in the same folder as streamlit_app.py.
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
    "log_qc7b log_past_qc701": (
        "Outpatient cost × Previous-year inpatient cost"
    ),
    "bmi age": "BMI × Age",
    "bmi log_past_qc701": (
        "BMI × Previous-year inpatient cost"
    ),
    "age log_past_qc701": (
        "Age × Previous-year inpatient cost"
    ),
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

    # Fallback for unseen interaction names.
    if " " in feature_name:
        parts = feature_name.split()
        readable_parts = [
            FEATURE_LABELS.get(
                part,
                part.replace("_", " ").title(),
            )
            for part in parts
        ]
        return " × ".join(readable_parts)

    return feature_name.replace("_", " ").title()


# ============================================================
# 6. CURRENCY OPTIONS
# ============================================================

CURRENCY_OPTIONS = {
    "Chinese Yuan (CNY)": {
        "code": "CNY",
        "symbol": "¥",
    },
    "Malaysian Ringgit (MYR)": {
        "code": "MYR",
        "symbol": "RM",
    },
    "US Dollar (USD)": {
        "code": "USD",
        "symbol": "$",
    },
    "Singapore Dollar (SGD)": {
        "code": "SGD",
        "symbol": "S$",
    },
    "Euro (EUR)": {
        "code": "EUR",
        "symbol": "€",
    },
    "British Pound (GBP)": {
        "code": "GBP",
        "symbol": "£",
    },
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
            "Place the latest PKL file in the same folder as "
            "streamlit_app.py, or update MODEL_PATH."
        )

    raw = joblib.load(path)

    if not isinstance(raw, dict):
        raise TypeError(
            "The PKL file must contain a dictionary artifact. "
            f"Loaded type: {type(raw).__name__}"
        )

    lgb_model = raw.get("lightgbm_model")
    if lgb_model is None:
        lgb_model = raw.get("lgb_model")

    xgb_model = raw.get("xgboost_model")
    if xgb_model is None:
        xgb_model = raw.get("xgb_model")

    lgb_weight = raw.get("lightgbm_weight")
    if lgb_weight is None:
        lgb_weight = raw.get("blend_weight")

    xgb_weight = raw.get("xgboost_weight")
    if xgb_weight is None and lgb_weight is not None:
        xgb_weight = 1.0 - float(lgb_weight)

    original_feature_names = raw.get("original_feature_names")
    if original_feature_names is None:
        original_feature_names = raw.get("feature_names")

    final_feature_names = raw.get("final_feature_names")
    if final_feature_names is None:
        final_feature_names = raw.get("feature_names")

    interaction_source_features = raw.get(
        "interaction_source_features",
        [],
    )
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
            + "\n\nAvailable keys:\n"
            + "\n".join(f"- {key}" for key in raw.keys())
        )

    original_feature_names = [
        str(feature).strip()
        for feature in list(original_feature_names)
    ]

    final_feature_names = [
        str(feature).strip()
        for feature in list(final_feature_names)
    ]

    interaction_source_features = [
        str(feature).strip()
        for feature in list(interaction_source_features)
    ]

    lgb_weight = float(lgb_weight)
    xgb_weight = float(xgb_weight)

    if not 0.0 <= lgb_weight <= 1.0:
        raise ValueError(f"Invalid LightGBM blend weight: {lgb_weight}")

    if not 0.0 <= xgb_weight <= 1.0:
        raise ValueError(f"Invalid XGBoost blend weight: {xgb_weight}")

    if not np.isclose(lgb_weight + xgb_weight, 1.0, atol=1e-6):
        raise ValueError("The LightGBM and XGBoost weights do not add up to 1.")

    expected_count = len(final_feature_names)
    lgb_count = getattr(lgb_model, "n_features_in_", None)
    xgb_count = getattr(xgb_model, "n_features_in_", None)

    if lgb_count is not None and int(lgb_count) != expected_count:
        raise ValueError(
            "LightGBM feature-count mismatch in the saved artifact.\n"
            f"Model expects: {int(lgb_count)}\n"
            f"Stored final features: {expected_count}"
        )

    if xgb_count is not None and int(xgb_count) != expected_count:
        raise ValueError(
            "XGBoost feature-count mismatch in the saved artifact.\n"
            f"Model expects: {int(xgb_count)}\n"
            f"Stored final features: {expected_count}"
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
    if height_cm <= 0:
        raise ValueError("Height must be greater than zero.")

    if weight_kg <= 0:
        raise ValueError("Weight must be greater than zero.")

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
    }

    values["health_very_good"] = float(int(health_code == 2))
    values["health_good"] = float(int(health_code == 3))
    values["health_fair"] = float(int(health_code == 4))
    values["health_poor"] = float(int(health_code == 5))

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

    missing = [
        feature
        for feature in required_original_features
        if feature not in candidates
    ]

    if missing:
        raise ValueError(
            "The application cannot create all original predictors required by the latest PKL.\n\n"
            "Unsupported original predictors:\n"
            + "\n".join(f"- {feature}" for feature in missing)
        )

    model_input = pd.DataFrame(
        [
            {
                feature: candidates[feature]
                for feature in required_original_features
            }
        ]
    )

    model_input = model_input.loc[
        :,
        required_original_features,
    ].apply(pd.to_numeric, errors="coerce")

    if model_input.isna().any().any():
        invalid = model_input.columns[model_input.isna().any()].tolist()
        raise ValueError(
            "The generated original input contains invalid values:\n"
            + "\n".join(f"- {feature}" for feature in invalid)
        )

    if np.isinf(model_input.to_numpy(dtype=float)).any():
        raise ValueError("The generated original input contains infinite values.")

    return model_input


def create_engineered_model_input(
    *,
    original_input: pd.DataFrame,
    interaction_source_features: list[str],
    final_feature_names: list[str],
) -> pd.DataFrame:
    """Recreate pairwise interaction features and enforce the final feature order."""
    engineered = original_input.copy()

    missing_sources = [
        feature
        for feature in interaction_source_features
        if feature not in engineered.columns
    ]

    if missing_sources:
        raise KeyError(
            "Interaction source variables are missing from the original input:\n"
            + "\n".join(f"- {feature}" for feature in missing_sources)
        )

    for left_index in range(len(interaction_source_features)):
        for right_index in range(left_index + 1, len(interaction_source_features)):
            left = interaction_source_features[left_index]
            right = interaction_source_features[right_index]
            engineered[f"{left} {right}"] = engineered[left] * engineered[right]

    missing_final = [
        feature
        for feature in final_feature_names
        if feature not in engineered.columns
    ]

    if missing_final:
        raise KeyError(
            "The application could not recreate all final model features required by the PKL:\n"
            + "\n".join(f"- {feature}" for feature in missing_final)
        )

    engineered = engineered.loc[:, final_feature_names].apply(
        pd.to_numeric, errors="coerce"
    )

    if engineered.isna().any().any():
        invalid = engineered.columns[engineered.isna().any()].tolist()
        raise ValueError(
            "The engineered model input contains invalid values:\n"
            + "\n".join(f"- {feature}" for feature in invalid)
        )

    if np.isinf(engineered.to_numpy(dtype=float)).any():
        raise ValueError("The engineered model input contains infinite values.")

    return engineered


# ============================================================
# 8A. NEARBY HEALTHCARE FACILITY LOCATOR
# ============================================================

OVERPASS_API_URLS = [
    "https://overpass.private.coffee/api/interpreter",
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

    lat1 = np.radians(float(latitude_1))
    lon1 = np.radians(float(longitude_1))
    lat2 = np.radians(float(latitude_2))
    lon2 = np.radians(float(longitude_2))

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
    """Retrieve nearby hospitals and clinics from OpenStreetMap."""
    query = f"""
    [out:json][timeout:25];

    (
        node["amenity"="hospital"](around:{radius_m},{latitude},{longitude});
        way["amenity"="hospital"](around:{radius_m},{latitude},{longitude});
        relation["amenity"="hospital"](around:{radius_m},{latitude},{longitude});

        node["amenity"="clinic"](around:{radius_m},{latitude},{longitude});
        way["amenity"="clinic"](around:{radius_m},{latitude},{longitude});
        relation["amenity"="clinic"](around:{radius_m},{latitude},{longitude});
    );

    out center tags;
    """

    last_error = None
    payload = None

    for api_url in OVERPASS_API_URLS:
        try:
            response = requests.post(
                api_url,
                data=query,
                headers={"User-Agent": "MedicalCostPredictionResearchApp/1.0"},
                timeout=35,
            )
            response.raise_for_status()
            payload = response.json()
            break
        except requests.RequestException as e:
            last_error = e
            continue
    else:
        raise RuntimeError(
            "All healthcare map services are temporarily unavailable. "
            f"Last error: {last_error}"
        )

    elements = payload.get("elements", [])
    facilities: list[dict[str, Any]] = []

    for element in elements:
        tags = element.get("tags") or {}
        element_lat = element.get("lat")
        element_lon = element.get("lon")

        center = element.get("center") or {}
        if element_lat is None:
            element_lat = center.get("lat")
        if element_lon is None:
            element_lon = center.get("lon")

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

    expected_lgb_features = getattr(lgb_model, "n_features_in_", None)
    expected_xgb_features = getattr(xgb_model, "n_features_in_", None)
    produced_count = int(engineered_input.shape[1])

    if expected_lgb_features is not None and produced_count != int(expected_lgb_features):
        raise ValueError(
            "Engineered input does not match the LightGBM model input size.\n"
            f"Produced: {produced_count}\n"
            f"Expected: {expected_lgb_features}"
        )

    if expected_xgb_features is not None and produced_count != int(expected_xgb_features):
        raise ValueError(
            "Engineered input does not match the XGBoost model input size.\n"
            f"Produced: {produced_count}\n"
            f"Expected: {expected_xgb_features}"
        )

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

    predicted_original_cost = float(
        max(0.0, np.expm1(blended_log_prediction))
    )

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


def extract_shap_vector(explainer: Any, transformed_input: Any) -> np.ndarray:
    """Convert SHAP output into one vector."""
    result = explainer(transformed_input)
    values = getattr(result, "values", result)
    values = np.asarray(values)

    if values.ndim == 1:
        return values
    if values.ndim == 2:
        return values[0]
    if values.ndim == 3:
        return values[0, :, 0]

    raise ValueError(f"Unexpected SHAP output shape: {values.shape}")


def get_transformed_feature_names(
    artifact: dict[str, Any],
    transformed_count: int,
) -> list[str]:
    """Return the final engineered feature names stored in the PKL."""
    names = list(artifact["final_feature_names"])
    if len(names) != transformed_count:
        raise ValueError(
            "Stored final feature names do not match the SHAP vector length.\n"
            f"Names: {len(names)}\n"
            f"SHAP values: {transformed_count}"
        )
    return names


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

    produced_feature_count = int(shap_input.shape[1])
    lgb_model = artifact["lgb_model"]
    xgb_model = artifact["xgb_model"]
    lgb_weight = float(artifact["lgb_weight"])
    xgb_weight = float(artifact["xgb_weight"])

    final_feature_names = list(
        artifact.get("final_feature_names", artifact.get("feature_names", []))
    )

    weighted_shap_vectors = []
    explained_models = []

    import shap

    if lgb_weight > 1e-12:
        expected_lgb_features = getattr(lgb_model, "n_features_in_", None)
        if (
            expected_lgb_features is not None
            and produced_feature_count != int(expected_lgb_features)
        ):
            raise ValueError(
                "LightGBM input feature count mismatch in SHAP calculation.\n"
                f"Current: {produced_feature_count}\n"
                f"Expected: {expected_lgb_features}"
            )

        lgb_explainer = shap.TreeExplainer(lgb_model)
        lgb_result = lgb_explainer(shap_input)
        lgb_values = np.asarray(getattr(lgb_result, "values", lgb_result))

        if lgb_values.ndim == 2:
            lgb_values = lgb_values[0]
        elif lgb_values.ndim == 3:
            lgb_values = lgb_values[0, :, 0]
        elif lgb_values.ndim != 1:
            raise ValueError(f"Unexpected LightGBM SHAP shape: {lgb_values.shape}")

        weighted_shap_vectors.append(lgb_weight * lgb_values)
        explained_models.append("LightGBM")

    if xgb_weight > 1e-12:
        expected_xgb_features = getattr(xgb_model, "n_features_in_", None)
        if (
            expected_xgb_features is not None
            and produced_feature_count != int(expected_xgb_features)
        ):
            raise ValueError(
                "XGBoost input feature count mismatch in SHAP calculation.\n"
                f"Current: {produced_feature_count}\n"
                f"Expected: {expected_xgb_features}"
            )

        xgb_explainer = shap.TreeExplainer(xgb_model)
        xgb_result = xgb_explainer(shap_input)
        xgb_values = np.asarray(getattr(xgb_result, "values", xgb_result))

        if xgb_values.ndim == 2:
            xgb_values = xgb_values[0]
        elif xgb_values.ndim == 3:
            xgb_values = xgb_values[0, :, 0]
        elif xgb_values.ndim != 1:
            raise ValueError(f"Unexpected XGBoost SHAP shape: {xgb_values.shape}")

        weighted_shap_vectors.append(xgb_weight * xgb_values)
        explained_models.append("XGBoost")

    if not weighted_shap_vectors:
        raise ValueError("Neither model has a positive blending weight.")

    blended_values = np.sum(weighted_shap_vectors, axis=0)

    if len(final_feature_names) != len(blended_values):
        raise ValueError(
            "Stored feature-name count does not match the SHAP output length.\n"
            f"Stored: {len(final_feature_names)}\n"
            f"Values: {len(blended_values)}"
        )

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

    url = EXCHANGE_RATE_API_URL.format(
        api_key=api_key,
        base_currency=base_currency,
    )

    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()

    if data.get("result") != "success":
        raise RuntimeError(
            f"ExchangeRate-API error: {data.get('error-type', 'unknown-error')}"
        )

    rates = data.get("conversion_rates", {})
    if not rates:
        raise RuntimeError("No exchange rates were returned.")

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
        return {
            "rate": 1.0,
            "last_updated": None,
        }

    if not api_key:
        raise ValueError(
            "EXCHANGE_RATE_API_KEY was not found. "
            "Configure it in the Streamlit environment or Secrets."
        )

    rate_data = get_exchange_rates(api_key=api_key, base_currency="CNY")
    rates = rate_data["rates"]

    if target_currency not in rates:
        raise ValueError(f"Unsupported currency: {target_currency}")

    rate = float(rates[target_currency])
    if rate <= 0:
        raise ValueError(f"Invalid exchange rate returned for {target_currency}.")

    return {
        "rate": rate,
        "last_updated": rate_data["last_updated"],
    }


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

    if rate_from_cny <= 0:
        raise ValueError("The exchange rate must be greater than zero.")

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

    if rate_from_cny <= 0:
        raise ValueError("The exchange rate must be greater than zero.")

    return float(amount_cny * rate_from_cny)


def convert_cny_amount(
    *,
    amount_cny: float,
    target_currency: str,
    api_key: str | None,
) -> dict[str, Any]:
    """Convert a CNY amount to a selected display currency."""
    rate_info = get_currency_rate_from_cny(
        target_currency=target_currency,
        api_key=api_key,
    )

    converted_amount = convert_cny_to_selected_currency(
        amount_cny=amount_cny,
        target_currency=target_currency,
        rate_from_cny=rate_info["rate"],
    )

    return {
        "rate": rate_info["rate"],
        "converted_amount": converted_amount,
        "last_updated": rate_info["last_updated"],
    }


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
You are an intent parser inside a machine-learning medical-cost
prediction application.

Classify the user's request into exactly one of these intents:

1. "explanation"
   Use this for:
   - Why is my predicted cost high or low?
   - Which SHAP factors increased or decreased my prediction?
   - How can the model prediction potentially become lower?
   - What does a SHAP value mean?
   - General questions about the current prediction.

2. "what_if"
   Use this ONLY when the user explicitly asks to change one or
   more model inputs and asks what the prediction would become.

Examples:
- "If my weight is 45 kg what will my cost be?"
- "What if I become employed?"
- "If I stop smoking and weigh 55 kg, what is the new result?"
- "Change my outpatient cost to 300 MYR and predict again."

IMPORTANT RULES:
- DO NOT calculate any medical-cost prediction.
- DO NOT invent a feature value.
- Extract only values explicitly requested by the user.
- Return ONLY valid JSON.
- Never include markdown fences.
- Do not add explanatory prose.

Current raw user/model values:
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

User-selected currency:
- code: {prediction_context.get("selected_currency_code", "CNY")}
- symbol: {prediction_context.get("selected_currency_symbol", "¥")}

Supported change names are ONLY:
age
gender_label
height_cm
weight_kg
chronic_illness_label
smoking_label
hospitalized_label
health_label
employed_label
outpatient_cost_cny
previous_inpatient_cost_cny

For categorical values, use ONLY these exact strings:

gender_label:
- Female
- Male

chronic_illness_label:
- No
- Yes

smoking_label:
- No
- Yes

hospitalized_label:
- No
- Yes

health_label:
- Excellent
- Very good
- Good
- Fair
- Poor

employed_label:
- Not employed
- Employed

If the user gives a medical-cost value in their selected currency
rather than CNY, return it using these special temporary keys:

outpatient_cost_selected_currency
previous_inpatient_cost_selected_currency

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
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*
