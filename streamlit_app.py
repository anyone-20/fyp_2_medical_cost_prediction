# ============================================================
# STREAMLIT MEDICAL COST PREDICTION APPLICATION
# Latest Gradient Boosting Model:
# Blended LightGBM + XGBoost with Stored Interaction Rules
# ============================================================

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
import os



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
# 3. OPTIONAL STREAMLIT SECRETS
# ============================================================

def get_secret(key: str) -> str | None:
    """Safely read a Streamlit secret."""

    try:
        return str(st.secrets[key])

    except Exception:
        return None


import os
import streamlit as st

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EXCHANGE_RATE_API_KEY = os.getenv("EXCHANGE_RATE_API_KEY")

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
    """
    Convert saved model feature names into user-friendly labels.
    """

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
    """
    Load and validate the latest complete model artifact.

    Latest PKL format:
        lightgbm_model
        xgboost_model
        lightgbm_weight
        xgboost_weight
        original_feature_names
        interaction_source_features
        final_feature_names

    Older aliases are accepted where possible.
    """

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

    # --------------------------------------------------------
    # Resolve model objects and blending weights
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Resolve original and final feature definitions
    # --------------------------------------------------------

    original_feature_names = raw.get("original_feature_names")

    # Compatibility with artifacts that stored only final names.
    if original_feature_names is None:
        original_feature_names = raw.get("feature_names")

    final_feature_names = raw.get("final_feature_names")
    if final_feature_names is None:
        final_feature_names = raw.get("feature_names")

    interaction_source_features = raw.get(
        "interaction_source_features",
        [],
    )

    # The latest artifact stores a specification dictionary rather
    # than a fitted sklearn transformer. It is not called directly.
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
        raise ValueError(
            f"Invalid LightGBM blend weight: {lgb_weight}"
        )

    if not 0.0 <= xgb_weight <= 1.0:
        raise ValueError(
            f"Invalid XGBoost blend weight: {xgb_weight}"
        )

    if not np.isclose(
        lgb_weight + xgb_weight,
        1.0,
        atol=1e-6,
    ):
        raise ValueError(
            "The LightGBM and XGBoost weights do not add up to 1."
        )

    # Validate the model input sizes against the stored final matrix.
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
        # Compatibility alias used by some UI sections.
        "feature_names": final_feature_names,
        "preprocessor_spec": preprocessor_spec,
        "target_name": raw.get("target_name", TARGET_NAME),
        "target_transformation": raw.get(
            "target_transformation",
            "log1p",
        ),
        "model_name": raw.get(
            "model_type",
            MODEL_VERSION,
        ),
        "artifact_version": raw.get(
            "artifact_version",
            "unknown",
        ),
    }

# ============================================================
# 8. FEATURE-ENGINEERING HELPERS
# ============================================================

def safe_log1p(value: float) -> float:
    """Apply log1p to a non-negative monetary value."""

    value = float(value)

    if value < 0:
        raise ValueError(
            "Cost values cannot be negative."
        )

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
    """
    Create all currently supported ORIGINAL predictor values.

    Interaction variables are not manually requested from the user.
    They are generated later from the PKL's stored interaction rules.
    """

    if height_cm <= 0:
        raise ValueError(
            "Height must be greater than zero."
        )

    if weight_kg <= 0:
        raise ValueError(
            "Weight must be greater than zero."
        )

    bmi = float(
        weight_kg
        / ((height_cm / 100.0) ** 2)
    )

    values = {
        "age": float(age),
        "gender": float(gender_code),
        "bmi": bmi,
        "qp401": float(chronic_code),
        "qq201": float(smoking_code),
        "log_past_qc701": safe_log1p(
            previous_inpatient_cost
        ),
        "qc401": float(hospitalized_code),
        "log_qc7b": safe_log1p(
            outpatient_cost
        ),
        "qp201": float(health_code),
        "qgb1": float(employed_code),

        # Compatibility candidates for older model versions.
        # CFPS weight qp102 is measured in jin, where 1 kg = 2 jin.
        "qp102": float(weight_kg * 2.0),
    }

    # One-hot encoded self-rated-health features.
    # HEALTH_MAPPING uses:
    # 1 = Excellent (reference category)
    # 2 = Very good
    # 3 = Good
    # 4 = Fair
    # 5 = Poor
    #
    # Excellent is represented by all four dummy columns being 0.
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
    """
    Create one row containing the exact ORIGINAL predictors stored
    in the latest PKL artifact.
    """

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
            "The application cannot create all original predictors "
            "required by the latest PKL.\n\n"
            "Unsupported original predictors:\n"
            + "\n".join(f"- {feature}" for feature in missing)
            + "\n\nAdd matching user inputs and coding rules to "
            "create_feature_candidates()."
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
        invalid = model_input.columns[
            model_input.isna().any()
        ].tolist()

        raise ValueError(
            "The generated original input contains invalid values:\n"
            + "\n".join(f"- {feature}" for feature in invalid)
        )

    if np.isinf(
        model_input.to_numpy(dtype=float)
    ).any():
        raise ValueError(
            "The generated original input contains infinite values."
        )

    return model_input


def create_engineered_model_input(
    *,
    original_input: pd.DataFrame,
    interaction_source_features: list[str],
    final_feature_names: list[str],
) -> pd.DataFrame:
    """
    Recreate pairwise interaction features exactly as specified by
    the latest saved artifact, then enforce the final feature order.
    """

    engineered = original_input.copy()

    missing_sources = [
        feature
        for feature in interaction_source_features
        if feature not in engineered.columns
    ]

    if missing_sources:
        raise KeyError(
            "Interaction source variables are missing from the "
            "original input:\n"
            + "\n".join(f"- {feature}" for feature in missing_sources)
        )

    for left_index in range(
        len(interaction_source_features)
    ):
        for right_index in range(
            left_index + 1,
            len(interaction_source_features)
        ):
            left = interaction_source_features[left_index]
            right = interaction_source_features[right_index]
            interaction_name = f"{left} {right}"

            engineered[interaction_name] = (
                engineered[left]
                * engineered[right]
            )

    missing_final = [
        feature
        for feature in final_feature_names
        if feature not in engineered.columns
    ]

    if missing_final:
        raise KeyError(
            "The application could not recreate all final model "
            "features required by the PKL:\n"
            + "\n".join(f"- {feature}" for feature in missing_final)
        )

    engineered = engineered.loc[
        :,
        final_feature_names,
    ].apply(pd.to_numeric, errors="coerce")

    if engineered.isna().any().any():
        invalid = engineered.columns[
            engineered.isna().any()
        ].tolist()

        raise ValueError(
            "The engineered model input contains invalid values:\n"
            + "\n".join(f"- {feature}" for feature in invalid)
        )

    if np.isinf(
        engineered.to_numpy(dtype=float)
    ).any():
        raise ValueError(
            "The engineered model input contains infinite values."
        )

    return engineered

# ============================================================
# 9. PREDICTION SERVICE
# ============================================================

def predict_medical_cost(
    *,
    artifact: dict[str, Any],
    original_input: pd.DataFrame,
) -> dict[str, Any]:
    """
    Recreate the stored interaction features and generate the
    blended LightGBM + XGBoost prediction.

    No fit(), fit_transform(), or training operation is performed.
    """

    engineered_input = create_engineered_model_input(
        original_input=original_input,
        interaction_source_features=artifact[
            "interaction_source_features"
        ],
        final_feature_names=artifact[
            "final_feature_names"
        ],
    )

    lgb_model = artifact["lgb_model"]
    xgb_model = artifact["xgb_model"]

    expected_lgb_features = getattr(
        lgb_model,
        "n_features_in_",
        None,
    )
    expected_xgb_features = getattr(
        xgb_model,
        "n_features_in_",
        None,
    )

    produced_count = int(
        engineered_input.shape[1]
    )

    if (
        expected_lgb_features is not None
        and produced_count != int(expected_lgb_features)
    ):
        raise ValueError(
            "Engineered input does not match the LightGBM "
            "model input size.\n"
            f"Produced: {produced_count}\n"
            f"Expected: {expected_lgb_features}"
        )

    if (
        expected_xgb_features is not None
        and produced_count != int(expected_xgb_features)
    ):
        raise ValueError(
            "Engineered input does not match the XGBoost "
            "model input size.\n"
            f"Produced: {produced_count}\n"
            f"Expected: {expected_xgb_features}"
        )

    lgb_log_prediction = float(
        np.asarray(
            lgb_model.predict(
                engineered_input
            )
        ).reshape(-1)[0]
    )

    xgb_log_prediction = float(
        np.asarray(
            xgb_model.predict(
                engineered_input
            )
        ).reshape(-1)[0]
    )

    blended_log_prediction = float(
        artifact["lgb_weight"]
        * lgb_log_prediction
        + artifact["xgb_weight"]
        * xgb_log_prediction
    )

    predicted_original_cost = float(
        max(
            0.0,
            np.expm1(
                blended_log_prediction
            ),
        )
    )

    return {
        "predicted_log_cost": blended_log_prediction,
        "predicted_original_cost": predicted_original_cost,
        "lgb_log_prediction": lgb_log_prediction,
        "xgb_log_prediction": xgb_log_prediction,
        "original_input": original_input,
        "engineered_input": engineered_input,
        # Compatibility alias used by the SHAP section.
        "transformed_input": engineered_input,
    }

# ============================================================
# 10. SHAP HELPERS
# ============================================================

@st.cache_resource
def create_shap_explainers(
    _lgb_model: Any,
    _xgb_model: Any,
):
    """
    Lazily import SHAP and create cached tree explainers.
    """

    import shap

    return (
        shap.TreeExplainer(
            _lgb_model
        ),
        shap.TreeExplainer(
            _xgb_model
        ),
    )


def extract_shap_vector(
    explainer: Any,
    transformed_input: Any,
) -> np.ndarray:
    """Convert SHAP output into one vector."""

    result = explainer(
        transformed_input
    )

    values = getattr(
        result,
        "values",
        result,
    )

    values = np.asarray(values)

    if values.ndim == 1:
        return values

    if values.ndim == 2:
        return values[0]

    if values.ndim == 3:
        return values[0, :, 0]

    raise ValueError(
        "Unexpected SHAP output shape: "
        f"{values.shape}"
    )


def get_transformed_feature_names(
    artifact: dict[str, Any],
    transformed_count: int,
) -> list[str]:
    """
    Return the final engineered feature names stored in the PKL.
    """

    names = list(
        artifact["final_feature_names"]
    )

    if len(names) != transformed_count:
        raise ValueError(
            "Stored final feature names do not match the SHAP "
            "vector length.\n"
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
    """
    Calculate SHAP contributions only for models that actually
    contribute to the blended prediction.

    This prevents a zero-weight model with a different training
    feature count from causing SHAP shape errors.
    """

    model_input = prediction_result["transformed_input"]

    if isinstance(model_input, pd.DataFrame):
        shap_input = model_input.copy()
    else:
        shap_input = np.asarray(model_input)

    produced_feature_count = int(
        shap_input.shape[1]
    )

    lgb_model = artifact["lgb_model"]
    xgb_model = artifact["xgb_model"]

    lgb_weight = float(
        artifact["lgb_weight"]
    )
    xgb_weight = float(
        artifact["xgb_weight"]
    )

    final_feature_names = list(
        artifact.get(
            "final_feature_names",
            artifact.get(
                "feature_names",
                [],
            ),
        )
    )

    weighted_shap_vectors = []
    explained_models = []

    import shap

    # --------------------------------------------------------
    # Explain LightGBM only when it contributes to the blend
    # --------------------------------------------------------

    if lgb_weight > 1e-12:

        expected_lgb_features = getattr(
            lgb_model,
            "n_features_in_",
            None,
        )

        if (
            expected_lgb_features is not None
            and produced_feature_count
            != int(expected_lgb_features)
        ):
            raise ValueError(
                "LightGBM contributes to the prediction, but its "
                "SHAP input feature count does not match training.\n"
                f"Current input: {produced_feature_count}\n"
                f"LightGBM expects: {expected_lgb_features}"
            )

        lgb_explainer = shap.TreeExplainer(
            lgb_model
        )

        lgb_result = lgb_explainer(
            shap_input
        )

        lgb_values = np.asarray(
            getattr(
                lgb_result,
                "values",
                lgb_result,
            )
        )

        if lgb_values.ndim == 2:
            lgb_values = lgb_values[0]

        elif lgb_values.ndim == 3:
            lgb_values = lgb_values[0, :, 0]

        elif lgb_values.ndim != 1:
            raise ValueError(
                "Unexpected LightGBM SHAP output shape: "
                f"{lgb_values.shape}"
            )

        weighted_shap_vectors.append(
            lgb_weight * lgb_values
        )

        explained_models.append(
            "LightGBM"
        )

    # --------------------------------------------------------
    # Explain XGBoost only when it contributes to the blend
    # --------------------------------------------------------

    if xgb_weight > 1e-12:

        expected_xgb_features = getattr(
            xgb_model,
            "n_features_in_",
            None,
        )

        if (
            expected_xgb_features is not None
            and produced_feature_count
            != int(expected_xgb_features)
        ):
            raise ValueError(
                "XGBoost contributes to the prediction, but its "
                "SHAP input feature count does not match training.\n"
                f"Current input: {produced_feature_count}\n"
                f"XGBoost expects: {expected_xgb_features}"
            )

        xgb_explainer = shap.TreeExplainer(
            xgb_model
        )

        xgb_result = xgb_explainer(
            shap_input
        )

        xgb_values = np.asarray(
            getattr(
                xgb_result,
                "values",
                xgb_result,
            )
        )

        if xgb_values.ndim == 2:
            xgb_values = xgb_values[0]

        elif xgb_values.ndim == 3:
            xgb_values = xgb_values[0, :, 0]

        elif xgb_values.ndim != 1:
            raise ValueError(
                "Unexpected XGBoost SHAP output shape: "
                f"{xgb_values.shape}"
            )

        weighted_shap_vectors.append(
            xgb_weight * xgb_values
        )

        explained_models.append(
            "XGBoost"
        )

    if not weighted_shap_vectors:
        raise ValueError(
            "Neither model has a positive blending weight."
        )

    blended_values = np.sum(
        weighted_shap_vectors,
        axis=0,
    )

    if len(final_feature_names) != len(
        blended_values
    ):
        raise ValueError(
            "Stored feature-name count does not match the SHAP "
            "output length.\n"
            f"Stored names: {len(final_feature_names)}\n"
            f"SHAP values: {len(blended_values)}\n"
            f"Explained models: {explained_models}"
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

    contribution_df[
        "Absolute contribution"
    ] = contribution_df[
        "SHAP contribution"
    ].abs()

    contribution_df[
        "Effect"
    ] = np.where(
        contribution_df[
            "SHAP contribution"
        ] >= 0,
        "Increased prediction",
        "Decreased prediction",
    )

    contribution_df = (
        contribution_df
        .sort_values(
            "Absolute contribution",
            ascending=False,
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    return contribution_df


# ============================================================
# 11. CURRENCY CONVERSION
# ============================================================

EXCHANGE_RATE_API_URL = (
    "https://v6.exchangerate-api.com/v6/"
    "{api_key}/latest/{base_currency}"
)


@st.cache_data(ttl=3600)
def get_exchange_rates(
    api_key: str,
    base_currency: str = "CNY",
) -> dict[str, Any]:
    """
    Retrieve exchange rates and cache them for one hour.

    The application intentionally requests rates with CNY as the
    base currency so the SAME rate can be used for:
      1. User-selected currency -> CNY model input
      2. CNY model prediction -> user-selected currency output
    """

    if not api_key:
        raise ValueError(
            "EXCHANGE_RATE_API_KEY is missing."
        )

    url = EXCHANGE_RATE_API_URL.format(
        api_key=api_key,
        base_currency=base_currency,
    )

    response = requests.get(
        url,
        timeout=10,
    )
    response.raise_for_status()

    data = response.json()

    if data.get("result") != "success":
        raise RuntimeError(
            "ExchangeRate-API error: "
            f"{data.get('error-type', 'unknown-error')}"
        )

    rates = data.get(
        "conversion_rates",
        {},
    )

    if not rates:
        raise RuntimeError(
            "No exchange rates were returned."
        )

    return {
        "rates": rates,
        "last_updated": data.get(
            "time_last_update_utc"
        ),
    }


def get_currency_rate_from_cny(
    *,
    target_currency: str,
    api_key: str | None,
) -> dict[str, Any]:
    """
    Return the rate expressed as:

        1 CNY = rate × target_currency

    Example:
        1 CNY = 0.65 MYR

    This single rate is used in both directions so the user's input
    conversion and the displayed prediction remain consistent.
    """

    target_currency = str(
        target_currency
    ).upper().strip()

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

    rate_data = get_exchange_rates(
        api_key=api_key,
        base_currency="CNY",
    )

    rates = rate_data["rates"]

    if target_currency not in rates:
        raise ValueError(
            f"Unsupported currency: {target_currency}"
        )

    rate = float(
        rates[target_currency]
    )

    if rate <= 0:
        raise ValueError(
            f"Invalid exchange rate returned for {target_currency}."
        )

    return {
        "rate": rate,
        "last_updated": rate_data[
            "last_updated"
        ],
    }


def convert_selected_currency_to_cny(
    *,
    amount: float,
    source_currency: str,
    rate_from_cny: float,
) -> float:
    """
    Convert a user-entered amount into CNY.

    If:
        1 CNY = rate_from_cny × source_currency

    then:
        CNY = source_currency_amount / rate_from_cny
    """

    amount = float(amount)
    rate_from_cny = float(rate_from_cny)

    if amount < 0:
        raise ValueError(
            "Currency amounts cannot be negative."
        )

    if source_currency == "CNY":
        return amount

    if rate_from_cny <= 0:
        raise ValueError(
            "The exchange rate must be greater than zero."
        )

    return float(
        amount / rate_from_cny
    )


def convert_cny_to_selected_currency(
    *,
    amount_cny: float,
    target_currency: str,
    rate_from_cny: float,
) -> float:
    """
    Convert a CNY amount into the user's selected currency.
    """

    amount_cny = float(amount_cny)
    rate_from_cny = float(rate_from_cny)

    if amount_cny < 0:
        raise ValueError(
            "Currency amounts cannot be negative."
        )

    if target_currency == "CNY":
        return amount_cny

    if rate_from_cny <= 0:
        raise ValueError(
            "The exchange rate must be greater than zero."
        )

    return float(
        amount_cny * rate_from_cny
    )


# Backward-compatible helper used by any older section of the app.
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

    converted_amount = (
        convert_cny_to_selected_currency(
            amount_cny=amount_cny,
            target_currency=target_currency,
            rate_from_cny=rate_info["rate"],
        )
    )

    return {
        "rate": rate_info["rate"],
        "converted_amount": converted_amount,
        "last_updated": rate_info[
            "last_updated"
        ],
    }


# ============================================================
# 12. GEMINI HELPER
# ============================================================

@st.cache_resource
def load_gemini_client(
    api_key: str,
):
    """Create a Gemini client only when a key is available."""

    from google import genai

    return genai.Client(
        api_key=api_key
    )


def generate_gemini_explanation(
    *,
    prediction_context: dict[str, Any],
    user_message: str,
) -> str:
    """
    Generate a concise educational explanation using the latest
    prediction, SHAP factors, and currency-conversion result.
    """

    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured."
        )

    client = load_gemini_client(
        GEMINI_API_KEY
    )

    # ========================================================
    # 1. Prepare SHAP/model-factor context
    # ========================================================

    top_factors = prediction_context.get(
        "top_factors",
        [],
    )

    if top_factors:
        factor_text = "\n".join(
            (
                f"- {item['feature']}: "
                f"{item['effect']} "
                f"(SHAP contribution "
                f"{item['contribution']:.4f})"
            )
            for item in top_factors
        )
    else:
        factor_text = (
            "Model-factor information is unavailable."
        )

    # ========================================================
    # 2. Prepare currency-conversion context
    # ========================================================

    converted_cost = prediction_context.get(
        "converted_cost"
    )

    currency_code = prediction_context.get(
        "selected_currency_code",
        "CNY",
    )

    currency_symbol = prediction_context.get(
        "selected_currency_symbol",
        "¥",
    )

    exchange_rate = prediction_context.get(
        "exchange_rate"
    )

    exchange_rate_updated = prediction_context.get(
        "exchange_rate_updated"
    )

    currency_conversion_error = prediction_context.get(
        "currency_conversion_error"
    )

    outpatient_input_selected = prediction_context.get(
        "outpatient_cost_selected_currency"
    )
    previous_input_selected = prediction_context.get(
        "previous_inpatient_cost_selected_currency"
    )
    outpatient_input_cny = prediction_context.get(
        "outpatient_cost_cny"
    )
    previous_input_cny = prediction_context.get(
        "previous_inpatient_cost_cny"
    )

    input_currency_context = (
        f"- Current outpatient input: "
        f"{currency_symbol}{outpatient_input_selected:,.2f} "
        f"{currency_code} -> "
        f"¥{outpatient_input_cny:,.2f} CNY\n"
        f"- Previous inpatient input: "
        f"{currency_symbol}{previous_input_selected:,.2f} "
        f"{currency_code} -> "
        f"¥{previous_input_cny:,.2f} CNY"
        if (
            outpatient_input_selected is not None
            and previous_input_selected is not None
            and outpatient_input_cny is not None
            and previous_input_cny is not None
        )
        else "- Input currency-conversion details are unavailable."
    )

    if (
        converted_cost is not None
        and currency_code != "CNY"
        and exchange_rate is not None
    ):
        currency_context = (
            f"{input_currency_context}\n"
            f"- Predicted cost in selected currency: "
            f"{currency_symbol}{converted_cost:,.2f} "
            f"{currency_code}\n"
            f"- Prediction in model base currency: "
            f"¥{prediction_context['predicted_cost_cny']:,.2f} CNY\n"
            f"- Exchange rate used: "
            f"1 CNY = {exchange_rate:.6f} "
            f"{currency_code}\n"
            f"- Exchange-rate update time: "
            f"{exchange_rate_updated or 'Unavailable'}"
        )

    elif currency_code == "CNY":
        currency_context = (
            f"{input_currency_context}\n"
            "- No exchange-rate conversion was required because "
            "the selected currency is CNY."
        )

    elif currency_conversion_error:
        currency_context = (
            "- Currency conversion was unavailable.\n"
            f"- Conversion error: "
            f"{currency_conversion_error}"
        )

    else:
        currency_context = (
            "- Currency conversion was not available for "
            "this prediction."
        )

    # ========================================================
    # 3. Build Gemini prompt
    # ========================================================

    prompt = f"""
You are an educational assistant inside a machine-learning
application that estimates inpatient medical costs.

The prediction is generated by a blended LightGBM and XGBoost
regression model trained using historical CFPS survey data.

Important rules:
1. Explain the result using simple and understandable language.
2. State that the prediction is an estimate and not a guaranteed bill.
3. Explain that SHAP describes model behaviour and does not prove
   medical causation.
4. Do not diagnose illness or recommend treatment.
5. Do not invent patient details, medical facts, exchange rates,
   or converted amounts.
6. Use only the currency values supplied in the context below.
7. Explain that exchange rates can change over time.
8. Keep the response concise and directly answer the user's question.

Prediction context:
- Predicted inpatient cost:
  ¥{prediction_context['predicted_cost_cny']:,.2f} CNY
- Log-scale prediction:
  {prediction_context['predicted_log_cost']:.4f}

Currency information:
{currency_context}

Patient/model inputs:
- Age: {prediction_context['age']}
- BMI: {prediction_context['bmi']:.2f}
- Gender: {prediction_context['gender']}
- Chronic illness:
  {prediction_context['chronic_illness']}
- Smoking status:
  {prediction_context['smoking_status']}
- Hospitalized during the relevant period:
  {prediction_context['hospitalized']}
- Self-rated health:
  {prediction_context['health_status']}
- Employment status:
  {prediction_context['employment_status']}

Top model factors:
{factor_text}

User question:
{user_message}
"""

    # ========================================================
    # 4. Call Gemini
    # ========================================================

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    response_text = getattr(
        response,
        "text",
        None,
    )

    if not response_text:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    return response_text.strip()


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

    if age < 1 or age > 119:
        errors.append(
            "Age must be between 1 and 119."
        )

    if height_cm <= 0:
        errors.append(
            "Height must be greater than zero."
        )

    if weight_kg <= 0:
        errors.append(
            "Weight must be greater than zero."
        )

    if outpatient_cost < 0:
        errors.append(
            "Outpatient medical cost cannot be negative."
        )

    if previous_inpatient_cost < 0:
        errors.append(
            "Previous inpatient cost cannot be negative."
        )

    bmi = float(
        weight_kg
        / ((height_cm / 100.0) ** 2)
    )

    if bmi < 10 or bmi > 80:
        errors.append(
            "The calculated BMI is outside the expected range "
            "of 10 to 80. Verify the height and weight."
        )

    if errors:
        raise ValueError(
            " ".join(errors)
        )

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
            "content": (
                "Hello! Generate a prediction and I can explain "
                "the estimated cost and its model factors."
            ),
        }
    ]


# ============================================================
# 15. LOAD THE MODEL
# ============================================================

try:
    artifact = load_model_artifact(
        str(MODEL_PATH)
    )

except Exception as error:
    st.error(
        "Unable to load the trained model."
    )

    st.exception(error)
    st.stop()


# ============================================================
# 15A. MODEL SHAPE DIAGNOSTICS
# This must appear only after artifact has been loaded.
# ============================================================

final_feature_names_for_diagnostics = artifact.get(
    "final_feature_names",
    artifact.get(
        "feature_names",
        [],
    ),
)

lgb_expected_features = getattr(
    artifact["lgb_model"],
    "n_features_in_",
    "Unknown",
)

xgb_expected_features = getattr(
    artifact["xgb_model"],
    "n_features_in_",
    "Unknown",
)


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


overview_col1, overview_col2, overview_col3 = st.columns(
    3
)

with overview_col1:
    st.markdown(
        """
        <div class="info-card">
            <div class="info-card-title">Model</div>
            <div class="info-card-value">LightGBM + XGBoost</div>
            <div class="info-card-text">
                A weighted Gradient Boosting ensemble trained on
                historical CFPS survey data.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with overview_col2:
    st.markdown(
        """
        <div class="info-card">
            <div class="info-card-title">Output</div>
            <div class="info-card-value">Estimated inpatient cost</div>
            <div class="info-card-text">
                The model predicts log cost and converts it back to
                the original CNY cost scale.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with overview_col3:
    st.markdown(
        """
        <div class="info-card">
            <div class="info-card-title">Important</div>
            <div class="info-card-value">Research estimate only</div>
            <div class="info-card-text">
                The result is not a guaranteed medical bill, diagnosis,
                or financial recommendation.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.caption(
    f"Model version: {MODEL_VERSION} · "
    f"Saved model: {artifact['model_name']}"
)


# ============================================================
# 17. MODEL INFORMATION SIDEBAR
# ============================================================

with st.sidebar:
    st.header(
        "About this application"
    )

    st.write(
        "This research prototype estimates inpatient medical "
        "cost using stored interaction-feature rules and a blended "
        "LightGBM–XGBoost model."
    )

    st.divider()

    st.subheader(
        "Model details"
    )

    st.write(
        "**Model:**",
        artifact["model_name"],
    )

    st.write(
        "**Original user inputs:**",
        len(
            artifact["original_feature_names"]
        ),
    )

    st.write(
        "**LightGBM weight:**",
        f"{artifact['lgb_weight']:.4f}",
    )

    st.write(
        "**XGBoost weight:**",
        f"{artifact['xgb_weight']:.4f}",
    )


    st.divider()

    st.subheader(
        "Model shape diagnostics"
    )

    st.write(
        "**Current engineered features:**",
        len(final_feature_names_for_diagnostics),
    )

    st.write(
        "**LightGBM expects:**",
        lgb_expected_features,
    )

    st.write(
        "**XGBoost expects:**",
        xgb_expected_features,
    )

    with st.expander(
        "Original user-input features"
    ):
        for number, feature in enumerate(
            artifact["original_feature_names"],
            start=1,
        ):
            st.write(
                f"{number}. {feature}"
            )

    with st.expander(
        "Final engineered model features"
    ):
        for number, feature in enumerate(
            artifact["final_feature_names"],
            start=1,
        ):
            st.write(
                f"{number}. {feature}"
            )

    st.divider()

    st.caption(
        "The floating chat button at the bottom-right can explain "
        "the latest prediction and its model factors."
    )


# ============================================================
# 18. USER INPUT FORM
# ============================================================

st.subheader(
    "User feature inputs"
)

st.write(
    "Choose your preferred currency first. Then enter all required "
    "personal, health, employment, and medical-cost information."
)

# ------------------------------------------------------------
# 18A. CURRENCY MUST BE SELECTED BEFORE THE FORM IS DISPLAYED
# ------------------------------------------------------------
#
# This selectbox is intentionally OUTSIDE st.form().
# Streamlit widgets inside a form do not trigger an immediate rerun,
# so putting the currency selector inside the form would prevent the
# cost-field labels from updating as soon as the user changes currency.
#
# A placeholder is used so the user must make an explicit selection.
# ------------------------------------------------------------

st.markdown(
    """
    <div style="
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 8px;
    ">
        Preferred currency
    </div>
    """,
    unsafe_allow_html=True,
)

selected_currency_label = st.selectbox(
    "Preferred currency",
    options=list(
        CURRENCY_OPTIONS.keys()
    ),
    index=None,
    placeholder="Select your preferred currency",
    label_visibility="collapsed",
    help=(
        "Select the currency you want to use for both medical-cost "
        "inputs and the main prediction display."
    ),
)

if selected_currency_label is None:
    st.info(
        "Please select your preferred currency before entering "
        "medical-cost values."
    )
    submitted = False

else:
    selected_currency = CURRENCY_OPTIONS[
        selected_currency_label
    ]

    selected_currency_code = selected_currency[
        "code"
    ]
    selected_currency_symbol = selected_currency[
        "symbol"
    ]

    st.caption(
        "Selected currency: "
        f"{selected_currency_label}. "
        "Medical-cost inputs will be converted to CNY before "
        "feature engineering and model prediction."
    )

    with st.form(
        "medical_cost_form"
    ):
        st.markdown(
            "#### Personal information"
        )

        personal_col1, personal_col2 = st.columns(
            2
        )

        with personal_col1:
            age = st.number_input(
                "Age",
                min_value=1,
                max_value=119,
                value=40,
                step=1,
                help=(
                    "Enter the individual's age in completed years."
                ),
            )

            gender_label = st.selectbox(
                "Gender",
                options=list(
                    GENDER_MAPPING.keys()
                ),
                help=(
                    "Select the gender category used by the model."
                ),
            )

        with personal_col2:
            height_cm = st.number_input(
                "Height (cm)",
                min_value=50.0,
                max_value=250.0,
                value=165.0,
                step=0.1,
                help=(
                    "Height is used together with weight to calculate BMI."
                ),
            )

            weight_kg = st.number_input(
                "Weight (kg)",
                min_value=10.0,
                max_value=300.0,
                value=60.0,
                step=0.1,
                help=(
                    "Weight is used together with height to calculate BMI."
                ),
            )

        calculated_bmi = float(
            weight_kg
            / ((height_cm / 100.0) ** 2)
        )

        bmi_status = (
            "Underweight"
            if calculated_bmi < 18.5
            else "Normal range"
            if calculated_bmi < 25
            else "Overweight"
            if calculated_bmi < 30
            else "High BMI"
        )

        st.info(
            f"Calculated BMI: **{calculated_bmi:.2f}** "
            f"({bmi_status})"
        )

        st.divider()

        st.markdown(
            "#### Health and lifestyle information"
        )

        health_col1, health_col2 = st.columns(
            2
        )

        with health_col1:
            chronic_illness_label = st.selectbox(
                "Chronic illness diagnosis",
                options=list(
                    YES_NO_MAPPING.keys()
                ),
                help=(
                    "Whether the individual has been diagnosed "
                    "with a chronic illness."
                ),
            )

            smoking_label = st.selectbox(
                "Smoking status",
                options=list(
                    YES_NO_MAPPING.keys()
                ),
                help=(
                    "Whether the individual currently smokes."
                ),
            )

        with health_col2:
            hospitalized_label = st.selectbox(
                "Hospitalized in the past 6 months",
                options=list(
                    YES_NO_MAPPING.keys()
                ),
                help=(
                    "Whether the individual was hospitalized "
                    "during the relevant survey period."
                ),
            )

            health_label = st.selectbox(
                "Self-rated health",
                options=list(
                    HEALTH_MAPPING.keys()
                ),
                index=2,
                help=(
                    "Select the individual's own assessment "
                    "of their current health."
                ),
            )

        employed_label = st.selectbox(
            "Employment status",
            options=list(
                EMPLOYMENT_MAPPING.keys()
            ),
            help=(
                "Select whether the individual is currently employed."
            ),
        )

        st.divider()

        st.markdown(
            "#### Medical-cost information"
        )

        st.caption(
            "Enter both amounts in "
            f"{selected_currency_label}. "
            "They will be converted to CNY automatically before "
            "the model applies log1p and interaction-feature rules."
        )

        cost_col1, cost_col2 = st.columns(
            2
        )

        with cost_col1:
            outpatient_cost_selected = st.number_input(
                (
                    "Current outpatient medical cost "
                    f"({selected_currency_code})"
                ),
                min_value=0.0,
                value=0.0,
                step=100.0,
                help=(
                    "Enter current outpatient medical spending in "
                    f"{selected_currency_label}."
                ),
            )

        with cost_col2:
            previous_inpatient_cost_selected = st.number_input(
                (
                    "Previous inpatient medical cost "
                    f"({selected_currency_code})"
                ),
                min_value=0.0,
                value=0.0,
                step=100.0,
                help=(
                    "Enter previous inpatient medical spending in "
                    f"{selected_currency_label}."
                ),
            )

        st.warning(
            "Review all entered values before submitting. "
            "The prediction is an estimate derived from historical data."
        )

        submitted = st.form_submit_button(
            "✨ Predict inpatient medical cost",
            use_container_width=True,
            type="primary",
        )


# ============================================================
# 19. PROCESS THE PREDICTION
# ============================================================

if submitted:
    try:
        # ----------------------------------------------------
        # 19A. GET ONE EXCHANGE RATE FOR THE WHOLE PREDICTION
        # ----------------------------------------------------
        #
        # The same rate is used for:
        #   selected currency -> CNY inputs
        #   CNY prediction -> selected currency output
        #
        # This avoids using two slightly different rates within
        # one prediction.
        # ----------------------------------------------------

        rate_info = get_currency_rate_from_cny(
            target_currency=selected_currency_code,
            api_key=EXCHANGE_RATE_API_KEY,
        )

        exchange_rate = float(
            rate_info["rate"]
        )
        exchange_rate_updated = rate_info[
            "last_updated"
        ]

        # ----------------------------------------------------
        # 19B. CONVERT BOTH USER COST INPUTS TO CNY FIRST
        # ----------------------------------------------------

        outpatient_cost_cny = (
            convert_selected_currency_to_cny(
                amount=float(
                    outpatient_cost_selected
                ),
                source_currency=(
                    selected_currency_code
                ),
                rate_from_cny=exchange_rate,
            )
        )

        previous_inpatient_cost_cny = (
            convert_selected_currency_to_cny(
                amount=float(
                    previous_inpatient_cost_selected
                ),
                source_currency=(
                    selected_currency_code
                ),
                rate_from_cny=exchange_rate,
            )
        )

        # ----------------------------------------------------
        # 19C. VALIDATE THE VALUES THAT WILL ACTUALLY ENTER
        #      THE MODEL (CNY)
        # ----------------------------------------------------

        validated_bmi = validate_raw_inputs(
            age=int(age),
            height_cm=float(height_cm),
            weight_kg=float(weight_kg),
            outpatient_cost=float(
                outpatient_cost_cny
            ),
            previous_inpatient_cost=float(
                previous_inpatient_cost_cny
            ),
        )

        # ----------------------------------------------------
        # 19D. BUILD MODEL INPUT USING CNY VALUES
        # ----------------------------------------------------
        #
        # create_feature_candidates() later applies np.log1p().
        # Therefore the correct sequence is:
        #
        # selected currency
        # -> CNY
        # -> log1p(CNY)
        # -> interaction features
        # -> LightGBM/XGBoost
        #
        # ----------------------------------------------------

        model_input = create_original_model_input(
            required_original_features=artifact[
                "original_feature_names"
            ],
            age=int(age),
            gender_code=GENDER_MAPPING[
                gender_label
            ],
            height_cm=float(height_cm),
            weight_kg=float(weight_kg),
            chronic_code=YES_NO_MAPPING[
                chronic_illness_label
            ],
            smoking_code=YES_NO_MAPPING[
                smoking_label
            ],
            previous_inpatient_cost=float(
                previous_inpatient_cost_cny
            ),
            hospitalized_code=YES_NO_MAPPING[
                hospitalized_label
            ],
            outpatient_cost=float(
                outpatient_cost_cny
            ),
            health_code=HEALTH_MAPPING[
                health_label
            ],
            employed_code=EMPLOYMENT_MAPPING[
                employed_label
            ],
        )

        prediction_result = predict_medical_cost(
            artifact=artifact,
            original_input=model_input,
        )

        predicted_log_cost = float(
            prediction_result[
                "predicted_log_cost"
            ]
        )

        predicted_cost_cny = float(
            prediction_result[
                "predicted_original_cost"
            ]
        )

        # ----------------------------------------------------
        # 19E. CONVERT CNY PREDICTION BACK TO USER CURRENCY
        # ----------------------------------------------------

        predicted_cost_selected = (
            convert_cny_to_selected_currency(
                amount_cny=predicted_cost_cny,
                target_currency=(
                    selected_currency_code
                ),
                rate_from_cny=exchange_rate,
            )
        )

        st.success(
            "Prediction completed successfully."
        )

       

        # ----------------------------------------------------
        # 19F. SHOW PREDICTION IN BOTH CURRENCIES
        # ----------------------------------------------------

        st.markdown(
            "#### Predicted inpatient medical cost"
        )

        if selected_currency_code == "CNY":
            st.metric(
                "Prediction (CNY)",
                f"¥{predicted_cost_cny:,.2f} CNY",
            )

        else:
            result_col1, result_col2 = st.columns(
                2
            )

            with result_col1:
                st.metric(
                    (
                        "Prediction in your selected "
                        f"currency ({selected_currency_code})"
                    ),
                    (
                        f"{selected_currency_symbol}"
                        f"{predicted_cost_selected:,.2f} "
                        f"{selected_currency_code}"
                    ),
                )

            with result_col2:
                st.metric(
                    "Prediction in model base currency (CNY)",
                    f"¥{predicted_cost_cny:,.2f} CNY",
                )

            st.caption(
                "The machine-learning model predicts on the CNY scale. "
                "The selected-currency value is a conversion of the "
                "same prediction."
            )

        # A compatibility dictionary keeps the older Gemini logic and
        # any downstream code that expects currency_result working.
        currency_result = {
            "rate": exchange_rate,
            "converted_amount": predicted_cost_selected,
            "last_updated": exchange_rate_updated,
        }
        currency_error = None

        # ----------------------------------------------------
        # SHAP explanations
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "Top model factors"
        )

        top_factor_context = []

        try:
            top_contributors = (
                calculate_top_contributors(
                    artifact=artifact,
                    prediction_result=(
                        prediction_result
                    ),
                    top_n=5,
                )
            )

            st.bar_chart(
                top_contributors.set_index(
                    "Feature"
                )[
                    [
                        "Absolute contribution"
                    ]
                ],
                use_container_width=True,
            )

            for index, row in (
                top_contributors.iterrows()
            ):
                contribution = float(
                    row[
                        "SHAP contribution"
                    ]
                )

                direction = (
                    "increased"
                    if contribution >= 0
                    else "reduced"
                )

                st.write(
                    f"{index + 1}. "
                    f"**{row['Feature']}** "
                    f"{direction} the model prediction."
                )

                top_factor_context.append(
                    {
                        "feature": str(
                            row["Feature"]
                        ),
                        "effect": str(
                            row["Effect"]
                        ),
                        "contribution": contribution,
                    }
                )

            with st.expander(
                "View detailed model-factor values"
            ):
                st.dataframe(
                    top_contributors,
                    use_container_width=True,
                    hide_index=True,
                )

                st.caption(
                    "SHAP values describe model behaviour on "
                    "the log-cost scale. They do not prove "
                    "medical causation."
                )

        except Exception as shap_error:
            st.warning(
                "The prediction worked, but SHAP explanations "
                "could not be generated: "
                f"{shap_error}"
            )

        # ----------------------------------------------------
        # Save prediction-aware session context
        # ----------------------------------------------------

        st.session_state.latest_prediction_context = {
            "predicted_cost_cny": predicted_cost_cny,
            "predicted_log_cost": predicted_log_cost,

            "selected_currency_label": selected_currency_label,
            "selected_currency_code": selected_currency_code,
            "selected_currency_symbol": selected_currency_symbol,

            "converted_cost": predicted_cost_selected,
            "predicted_cost_selected_currency": (
                predicted_cost_selected
            ),

            "outpatient_cost_selected_currency": float(
                outpatient_cost_selected
            ),
            "previous_inpatient_cost_selected_currency": float(
                previous_inpatient_cost_selected
            ),
            "outpatient_cost_cny": outpatient_cost_cny,
            "previous_inpatient_cost_cny": (
                previous_inpatient_cost_cny
            ),

            "exchange_rate": exchange_rate,
            "exchange_rate_updated": exchange_rate_updated,
            "currency_conversion_error": None,

            "age": int(age),
            "bmi": validated_bmi,
            "gender": gender_label,
            "chronic_illness": chronic_illness_label,
            "smoking_status": smoking_label,
            "hospitalized": hospitalized_label,
            "health_status": health_label,
            "employment_status": employed_label,
            "top_factors": top_factor_context,
        }

        # ----------------------------------------------------
        # Prediction verification
        # ----------------------------------------------------

        st.divider()

        with st.expander(
            "Prediction verification",
            expanded=False,
        ):

            manual_blend = float(
                artifact["lgb_weight"]
                * prediction_result[
                    "lgb_log_prediction"
                ]
                + artifact["xgb_weight"]
                * prediction_result[
                    "xgb_log_prediction"
                ]
            )

            retransformed_cost = float(
                max(
                    0.0,
                    np.expm1(
                        predicted_log_cost
                    ),
                )
            )

            verification_df = pd.DataFrame(
                {
                    "Test": [
                        "Blending formula",
                        "Log-to-original conversion",
                        "Outpatient input currency conversion",
                        "Previous inpatient input currency conversion",
                    ],
                    "Expected result": [
                        manual_blend,
                        retransformed_cost,
                        (
                            float(outpatient_cost_selected)
                            / exchange_rate
                            if selected_currency_code != "CNY"
                            else float(outpatient_cost_selected)
                        ),
                        (
                            float(previous_inpatient_cost_selected)
                            / exchange_rate
                            if selected_currency_code != "CNY"
                            else float(previous_inpatient_cost_selected)
                        ),
                    ],
                    "Application result": [
                        predicted_log_cost,
                        predicted_cost_cny,
                        outpatient_cost_cny,
                        previous_inpatient_cost_cny,
                    ],
                    "Status": [
                        (
                            "Pass"
                            if np.isclose(
                                manual_blend,
                                predicted_log_cost,
                                rtol=1e-12,
                                atol=1e-12,
                            )
                            else "Fail"
                        ),
                        (
                            "Pass"
                            if np.isclose(
                                retransformed_cost,
                                predicted_cost_cny,
                                rtol=1e-12,
                                atol=1e-12,
                            )
                            else "Fail"
                        ),
                        (
                            "Pass"
                            if np.isclose(
                                (
                                    float(outpatient_cost_selected)
                                    / exchange_rate
                                    if selected_currency_code != "CNY"
                                    else float(outpatient_cost_selected)
                                ),
                                outpatient_cost_cny,
                                rtol=1e-12,
                                atol=1e-12,
                            )
                            else "Fail"
                        ),
                        (
                            "Pass"
                            if np.isclose(
                                (
                                    float(previous_inpatient_cost_selected)
                                    / exchange_rate
                                    if selected_currency_code != "CNY"
                                    else float(previous_inpatient_cost_selected)
                                ),
                                previous_inpatient_cost_cny,
                                rtol=1e-12,
                                atol=1e-12,
                            )
                            else "Fail"
                        ),
                    ],
                }
            )

            st.dataframe(
                verification_df,
                use_container_width=True,
                hide_index=True,
            )

        # ----------------------------------------------------
        # Technical details
        # ----------------------------------------------------

        with st.expander(
            "View technical prediction details"
        ):
            st.write(
                "Model path:",
                str(MODEL_PATH),
            )

            st.write(
                "Required original features:",
                artifact["original_feature_names"],
            )

            st.write(
                "Selected input/output currency:",
                selected_currency_code,
            )

            st.write(
                "Exchange rate used:",
                (
                    f"1 CNY = {exchange_rate:.6f} "
                    f"{selected_currency_code}"
                ),
            )

            st.write(
                "Current outpatient cost entered by user:",
                (
                    f"{selected_currency_symbol}"
                    f"{float(outpatient_cost_selected):,.2f} "
                    f"{selected_currency_code}"
                ),
            )

            st.write(
                "Current outpatient cost sent to model:",
                f"¥{outpatient_cost_cny:,.2f} CNY",
            )

            st.write(
                "Previous inpatient cost entered by user:",
                (
                    f"{selected_currency_symbol}"
                    f"{float(previous_inpatient_cost_selected):,.2f} "
                    f"{selected_currency_code}"
                ),
            )

            st.write(
                "Previous inpatient cost sent to model:",
                f"¥{previous_inpatient_cost_cny:,.2f} CNY",
            )

            st.write(
                "LightGBM weight:",
                artifact["lgb_weight"],
            )

            st.write(
                "XGBoost weight:",
                artifact["xgb_weight"],
            )

            st.write(
                "LightGBM log prediction:",
                prediction_result[
                    "lgb_log_prediction"
                ],
            )

            st.write(
                "XGBoost log prediction:",
                prediction_result[
                    "xgb_log_prediction"
                ],
            )

            st.write(
                "Blended log prediction:",
                predicted_log_cost,
            )

            st.write(
                "Original-scale prediction (CNY):",
                predicted_cost_cny,
            )

            st.write(
                (
                    "Converted prediction "
                    f"({selected_currency_code}):"
                ),
                predicted_cost_selected,
            )

            st.write(
                "Generated original model input:"
            )

            st.dataframe(
                model_input,
                use_container_width=True,
                hide_index=True,
            )

            st.write(
                "Generated engineered model input:"
            )

            st.dataframe(
                prediction_result["engineered_input"],
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download generated model input",
                data=model_input.to_csv(
                    index=False
                ),
                file_name=(
                    "generated_original_model_input.csv"
                ),
                mime="text/csv",
                use_container_width=True,
            )

    except Exception as error:
        st.error(
            "Prediction failed."
        )

        st.exception(error)


# ============================================================
# 20. FLOATING GEMINI CHATBOT
# ============================================================
#
# Streamlit does not provide a native floating chat widget.
# The keyed container below is fixed to the bottom-right using
# CSS, while st.popover provides the expandable chat panel.
# ============================================================

with st.container(
    key="floating_chat_launcher"
):
    with st.popover(
        "🏥 Ask Me",
        help=(
            "Open the Medical Cost Prediction Assistant"
        ),
    ):
        st.markdown(
            "### Medical Cost Assistant"
        )

        st.caption(
            "Ask about the latest prediction or its model factors. "
            "The assistant does not provide diagnosis or treatment advice."
        )

        if (
            st.session_state.latest_prediction_context
            is None
        ):
            st.info(
                "Generate a prediction first for a personalised explanation."
            )

        else:
            latest_context = (
                st.session_state.latest_prediction_context
            )

            latest_cost_cny = float(
                latest_context[
                    "predicted_cost_cny"
                ]
            )

            latest_currency_code = (
                latest_context.get(
                    "selected_currency_code",
                    "CNY",
                )
            )

            latest_currency_symbol = (
                latest_context.get(
                    "selected_currency_symbol",
                    "¥",
                )
            )

            latest_selected_cost = (
                latest_context.get(
                    "predicted_cost_selected_currency",
                    latest_cost_cny,
                )
            )

            if latest_currency_code == "CNY":
                st.success(
                    f"Latest prediction: "
                    f"¥{latest_cost_cny:,.2f} CNY"
                )
            else:
                st.success(
                    "Latest prediction: "
                    f"{latest_currency_symbol}"
                    f"{latest_selected_cost:,.2f} "
                    f"{latest_currency_code} "
                    f"(¥{latest_cost_cny:,.2f} CNY)"
                )

        chat_history_container = st.container(
            height=290,
            border=True,
        )

        with chat_history_container:
            for message in (
                st.session_state.chat_messages
            ):
                css_class = (
                    "mini-chat-user"
                    if message["role"] == "user"
                    else "mini-chat-assistant"
                )

                role_label = (
                    "You"
                    if message["role"] == "user"
                    else "Assistant"
                )

                st.markdown(
                    (
                        f'<div class="{css_class}">'
                        f"<strong>{role_label}</strong><br>"
                        f"{message['content']}"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )

        with st.form(
            "floating_chat_form",
            clear_on_submit=True,
        ):
            chat_question = st.text_input(
                "Message",
                placeholder=(
                    "Ask why the predicted cost is high or low..."
                ),
                label_visibility="collapsed",
            )

            send_chat = st.form_submit_button(
                "Send",
                use_container_width=True,
                type="primary",
            )

        clear_col, status_col = st.columns(
            [1, 2]
        )

        with clear_col:
            clear_chat = st.button(
                "Clear",
                key="clear_floating_chat",
                use_container_width=True,
            )

        with status_col:
            if GEMINI_API_KEY:
                st.caption(
                    "🟢 AI assistant ready"
                )
            else:
                st.caption(
                    "🟠 Gemini API key missing"
                )

        if clear_chat:
            st.session_state.chat_messages = [
                {
                    "role": "assistant",
                    "content": (
                        "Chat history cleared. Generate a prediction "
                        "and ask me to explain it."
                    ),
                }
            ]

            st.rerun()

        if (
            send_chat
            and chat_question.strip()
        ):
            clean_question = (
                chat_question.strip()
            )

            st.session_state.chat_messages.append(
                {
                    "role": "user",
                    "content": clean_question,
                }
            )

            if (
                st.session_state.latest_prediction_context
                is None
            ):
                assistant_response = (
                    "Please generate a prediction first so I can "
                    "explain the result and its model factors."
                )

            elif not GEMINI_API_KEY:
                assistant_response = (
                    "Gemini is unavailable because GEMINI_API_KEY "
                    "is not configured in Streamlit Secrets."
                )

            else:
                try:
                    assistant_response = (
                        generate_gemini_explanation(
                            prediction_context=(
                                st.session_state
                                .latest_prediction_context
                            ),
                            user_message=clean_question,
                        )
                    )

                except Exception as error:
                    assistant_response = (
                        "The assistant could not generate a response: "
                        f"{error}"
                    )

            st.session_state.chat_messages.append(
                {
                    "role": "assistant",
                    "content": assistant_response,
                }
            )

            st.rerun()


# ============================================================
# 21. FOOTER
# ============================================================

st.divider()

st.caption(
    "Research prototype · Predictions are estimates derived from "
    "historical CFPS survey data and may differ from actual medical expenses."
)
