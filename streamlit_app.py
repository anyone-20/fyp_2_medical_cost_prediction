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
# ============================================================

st.markdown(
    """
    <style>
    /* Main page spacing */
    .block-container {
        max-width: 1120px;
        padding-top: 1.8rem;
        padding-bottom: 5.5rem;
    }

    /* Hide Streamlit's default footer */
    footer {
        visibility: hidden;
    }

    /* Hero section */
    .app-hero {
        padding: 1.35rem 1.45rem;
        border: 1px solid rgba(49, 51, 63, 0.15);
        border-radius: 18px;
        margin-bottom: 1rem;
        background: linear-gradient(
            135deg,
            rgba(240, 248, 255, 0.95),
            rgba(248, 250, 252, 0.95)
        );
    }

    .app-hero h1 {
        margin: 0;
        font-size: 2rem;
        line-height: 1.2;
    }

    .app-hero p {
        margin: 0.55rem 0 0 0;
        opacity: 0.8;
        font-size: 1rem;
    }

    /* Information cards */
    .info-card {
        padding: 1rem 1.1rem;
        border: 1px solid rgba(49, 51, 63, 0.12);
        border-radius: 15px;
        background: rgba(255, 255, 255, 0.75);
        min-height: 110px;
    }

    .info-card-title {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.65;
        margin-bottom: 0.35rem;
    }

    .info-card-value {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .info-card-text {
        font-size: 0.88rem;
        opacity: 0.75;
        line-height: 1.4;
    }

    /* Form container */
    div[data-testid="stForm"] {
        border: 1px solid rgba(49, 51, 63, 0.13);
        border-radius: 18px;
        padding: 1.1rem 1.2rem 1.25rem 1.2rem;
        background: rgba(255, 255, 255, 0.72);
    }

    /* Tab spacing */
    button[data-baseweb="tab"] {
        font-weight: 650;
    }

    /* Primary button */
    div[data-testid="stFormSubmitButton"] button {
        border-radius: 12px;
        min-height: 3rem;
        font-weight: 700;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        border: 1px solid rgba(49, 51, 63, 0.12);
        border-radius: 14px;
        padding: 0.85rem 1rem;
        background: rgba(255, 255, 255, 0.75);
    }

    /* Floating chat launcher container */
    .st-key-floating_chat_launcher {
        position: fixed;
        right: 24px;
        bottom: 24px;
        z-index: 999999;
        width: auto;
    }

    @keyframes floatingChatButton {

    0% {
        transform: translateY(0);
    }

    50% {
        transform: translateY(-8px);
    }

    100% {
        transform: translateY(0);
    }
}

.st-key-floating_chat_launcher {
    animation: floatingChatButton 3s ease-in-out infinite;
}

    /* Circular chat button */
    .st-key-floating_chat_launcher
    button[data-testid="stPopoverButton"] {
        width: 58px;
        height: 58px;
        min-width: 58px;
        border-radius: 50%;
        padding: 0;
        font-size: 1.55rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.22);
        border: 1px solid rgba(255, 255, 255, 0.35);
    }

    /* Popover width */
    div[data-baseweb="popover"] {
        max-width: min(410px, calc(100vw - 32px));
    }

    /* Chat message cards */
    .mini-chat-user {
        padding: 0.65rem 0.75rem;
        border-radius: 12px 12px 4px 12px;
        margin: 0.4rem 0 0.4rem 2rem;
        background: rgba(37, 99, 235, 0.12);
    }

    .mini-chat-assistant {
        padding: 0.65rem 0.75rem;
        border-radius: 12px 12px 12px 4px;
        margin: 0.4rem 2rem 0.4rem 0;
        background: rgba(100, 116, 139, 0.12);
    }

    /* Mobile spacing */
    @media (max-width: 700px) {
        .block-container {
            padding-left: 0.85rem;
            padding-right: 0.85rem;
        }

        .app-hero h1 {
            font-size: 1.55rem;
        }

        .st-key-floating_chat_launcher {
            right: 16px;
            bottom: 16px;
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


GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
EXCHANGE_RATE_API_KEY = get_secret("EXCHANGE_RATE_API_KEY")


# ============================================================
# 4. HUMAN-READABLE FEATURE LABELS
# ============================================================

FEATURE_LABELS = {
    "age": "Age",
    "gender": "Gender",
    "bmi": "Body mass index",
    "qp401": "Chronic illness status",
    "qq201": "Smoking status",
    "log_past_qc701": "Previous inpatient medical cost",
    "qc401": "Hospitalization status",
    "log_qc7b": "Outpatient medical cost",
    "qp201": "Self-rated health",
    "qgb1": "Employment status",
    "qp102": "Body weight",
    "qp605_s_1": "Medical insurance category",
    "log_qi202": "Retired allowance",
    "log_qc7b bmi": "Outpatient cost × BMI",
    "log_qc7b age": "Outpatient cost × age",
    "qc401 age": "Hospitalization × age",
    "qc401 bmi": "Hospitalization × BMI",
    "bmi age": "BMI × age",
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
    Calculate blended SHAP contributions in transformed feature
    space.
    """

    transformed_input = prediction_result[
        "transformed_input"
    ]

    (
        lgb_explainer,
        xgb_explainer,
    ) = create_shap_explainers(
        artifact["lgb_model"],
        artifact["xgb_model"],
    )

    lgb_values = extract_shap_vector(
        lgb_explainer,
        transformed_input,
    )

    xgb_values = extract_shap_vector(
        xgb_explainer,
        transformed_input,
    )

    if len(lgb_values) != len(xgb_values):
        raise ValueError(
            "LightGBM and XGBoost returned different numbers "
            "of SHAP values."
        )

    blended_values = (
        artifact["lgb_weight"]
        * lgb_values
        + artifact["xgb_weight"]
        * xgb_values
    )

    transformed_names = get_transformed_feature_names(
        artifact,
        len(blended_values),
    )

    contribution_df = pd.DataFrame(
        {
            "Feature": transformed_names,
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
    """Retrieve exchange rates and cache them for one hour."""

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


def convert_cny_amount(
    *,
    amount_cny: float,
    target_currency: str,
    api_key: str | None,
) -> dict[str, Any]:
    """Convert a CNY amount to the selected display currency."""

    if target_currency == "CNY":
        return {
            "rate": 1.0,
            "converted_amount": float(
                amount_cny
            ),
            "last_updated": None,
        }

    if not api_key:
        raise ValueError(
            "EXCHANGE_RATE_API_KEY was not found in "
            "Streamlit Secrets."
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

    return {
        "rate": rate,
        "converted_amount": float(
            amount_cny
        ) * rate,
        "last_updated": rate_data[
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
    """Generate a concise educational explanation."""

    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured."
        )

    client = load_gemini_client(
        GEMINI_API_KEY
    )

    top_factors = prediction_context.get(
        "top_factors",
        [],
    )

    if top_factors:
        factor_text = "\n".join(
            (
                f"- {item['feature']}: "
                f"{item['effect']} "
                f"(SHAP {item['contribution']:.4f})"
            )
            for item in top_factors
        )

    else:
        factor_text = (
            "SHAP contributors are unavailable."
        )

    prompt = f"""
You are an educational assistant inside a machine-learning
application that estimates inpatient medical costs.

The prediction is produced by a blended LightGBM and XGBoost
regression model trained on historical survey data.

Rules:
1. Explain the result in simple language.
2. State that the prediction is an estimate, not a guaranteed bill.
3. Explain that SHAP describes model behaviour, not medical causation.
4. Do not diagnose illness or recommend treatment.
5. Do not invent values or patient information.
6. Keep the answer concise.

Prediction context:
- Predicted cost: ¥{prediction_context['predicted_cost_cny']:,.2f} CNY
- Log prediction: {prediction_context['predicted_log_cost']:.4f}
- Age: {prediction_context['age']}
- BMI: {prediction_context['bmi']:.2f}
- Gender: {prediction_context['gender']}
- Chronic illness: {prediction_context['chronic_illness']}
- Smoking status: {prediction_context['smoking_status']}
- Hospitalization status: {prediction_context['hospitalized']}
- Self-rated health: {prediction_context['health_status']}
- Employment status: {prediction_context['employment_status']}

Top model contributors:
{factor_text}

User question:
{user_message}
"""

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
    "Enter all required personal, health, employment, and "
    "medical-cost information in the section below."
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
            "Hospitalized during the survey period",
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

    cost_col1, cost_col2 = st.columns(
        2
    )

    with cost_col1:
        outpatient_cost = st.number_input(
            "Current outpatient medical cost (CNY)",
            min_value=0.0,
            value=0.0,
            step=100.0,
            help=(
                "Enter outpatient medical spending for the "
                "current survey period."
            ),
        )

    with cost_col2:
        previous_inpatient_cost = st.number_input(
            "Previous inpatient medical cost (CNY)",
            min_value=0.0,
            value=0.0,
            step=100.0,
            help=(
                "Enter inpatient medical spending from the "
                "previous survey period."
            ),
        )

    selected_currency_label = st.selectbox(
        "Display the prediction in",
        options=list(
            CURRENCY_OPTIONS.keys()
        ),
        index=1,
        help=(
            "The model always predicts in CNY. Other currencies "
            "are approximate display conversions."
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
        validated_bmi = validate_raw_inputs(
            age=int(age),
            height_cm=float(height_cm),
            weight_kg=float(weight_kg),
            outpatient_cost=float(
                outpatient_cost
            ),
            previous_inpatient_cost=float(
                previous_inpatient_cost
            ),
        )

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
                previous_inpatient_cost
            ),
            hospitalized_code=YES_NO_MAPPING[
                hospitalized_label
            ],
            outpatient_cost=float(
                outpatient_cost
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

        selected_currency = (
            CURRENCY_OPTIONS[
                selected_currency_label
            ]
        )

        currency_result = None
        currency_error = None

        try:
            currency_result = convert_cny_amount(
                amount_cny=predicted_cost_cny,
                target_currency=selected_currency[
                    "code"
                ],
                api_key=EXCHANGE_RATE_API_KEY,
            )

        except Exception as error:
            currency_error = str(error)

        st.success(
            "Prediction completed successfully."
        )

        st.metric(
            "Estimated inpatient medical cost",
            f"¥{predicted_cost_cny:,.2f} CNY",
        )

        if (
            currency_result is not None
            and selected_currency["code"]
            != "CNY"
        ):
            st.metric(
                "Approximate converted cost",
                (
                    f"{selected_currency['symbol']}"
                    f"{currency_result['converted_amount']:,.2f} "
                    f"{selected_currency['code']}"
                ),
            )

            st.caption(
                "Exchange rate used: "
                f"1 CNY = {currency_result['rate']:.6f} "
                f"{selected_currency['code']}"
            )

            if currency_result[
                "last_updated"
            ]:
                st.caption(
                    "Exchange-rate update time: "
                    f"{currency_result['last_updated']}"
                )

        elif currency_error:
            st.warning(
                "The model prediction succeeded, but currency "
                "conversion was unavailable: "
                f"{currency_error}"
            )

        summary_col1, summary_col2 = st.columns(
            2
        )

        with summary_col1:
            st.metric(
                "Calculated BMI",
                f"{validated_bmi:.2f}",
            )

        with summary_col2:
            st.metric(
                "Log-scale prediction",
                f"{predicted_log_cost:.4f}",
            )

        # ----------------------------------------------------
        # Prediction verification
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "Prediction verification"
        )

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
                ],
                "Expected result": [
                    manual_blend,
                    retransformed_cost,
                ],
                "Application result": [
                    predicted_log_cost,
                    predicted_cost_cny,
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
                ],
            }
        )

        st.dataframe(
            verification_df,
            use_container_width=True,
            hide_index=True,
        )

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
                "View detailed SHAP values"
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
            "age": int(age),
            "bmi": validated_bmi,
            "gender": gender_label,
            "chronic_illness": (
                chronic_illness_label
            ),
            "smoking_status": smoking_label,
            "hospitalized": hospitalized_label,
            "health_status": health_label,
            "employment_status": employed_label,
            "top_factors": top_factor_context,
        }

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
                "Original-scale prediction:",
                predicted_cost_cny,
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
        "🏥 Ask AI",
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
            latest_cost = (
                st.session_state.latest_prediction_context[
                    "predicted_cost_cny"
                ]
            )

            st.success(
                f"Latest prediction: ¥{latest_cost:,.2f} CNY"
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
