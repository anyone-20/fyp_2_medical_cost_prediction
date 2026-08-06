 ============================================================
# STREAMLIT MEDICAL COST PREDICTION APPLICATION
# Latest Gradient Boosting model:
# Blended LightGBM + XGBoost with saved preprocessor
# ============================================================

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests
import shap
import streamlit as st
from google import genai


# ============================================================
# 1. PAGE CONFIGURATION
# Must be the first Streamlit command.
# ============================================================

st.set_page_config(
    page_title="Medical Cost Prediction",
    page_icon="🏥",
    layout="centered",
)


# ============================================================
# 2. PATHS AND MODEL INFORMATION
# ============================================================

BASE_DIR = Path(C:\Users\leeji\FYP2\CFPS\ML\2020_2.0\GB\saved_models).resolve().parent

# Place the latest PKL file in the same folder as this script.
MODEL_PATH = BASE_DIR / "cleaned_2020_GB_2.0.pkl"

MODEL_VERSION = "Blended LightGBM + XGBoost 2.0"
TARGET_NAME = "log_qc701"

# The latest model is expected to use these ten original inputs.
EXPECTED_RAW_FEATURES = [
    "age",
    "gender",
    "bmi",
    "qp401",
    "qq201",
    "log_past_qc701",
    "qc401",
    "log_qc7b",
    "qp201",
    "qgb1",
]


# ============================================================
# 3. STREAMLIT SECRETS
# ============================================================

def get_secret(key: str) -> str | None:
    """Read a Streamlit secret without crashing the app."""

    try:
        value = st.secrets[key]
    except (KeyError, FileNotFoundError):
        return None

    if value is None:
        return None

    return str(value)


GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
EXCHANGE_RATE_API_KEY = get_secret("EXCHANGE_RATE_API_KEY")


# ============================================================
# 4. GEMINI CLIENT
# ============================================================

@st.cache_resource
def load_gemini_client(api_key: str):
    """Create and cache the Gemini client."""

    return genai.Client(api_key=api_key)


if GEMINI_API_KEY:
    try:
        gemini_client = load_gemini_client(GEMINI_API_KEY)
        gemini_available = True
    except Exception as error:
        gemini_client = None
        gemini_available = False
        st.warning(f"Gemini could not be initialized: {error}")
else:
    gemini_client = None
    gemini_available = False


# ============================================================
# 5. SESSION STATE
# ============================================================

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! I can explain the predicted inpatient medical "
                "cost, the important model factors, and the meaning of "
                "the result. I cannot provide a medical diagnosis or "
                "treatment advice."
            ),
        }
    ]

if "latest_prediction_context" not in st.session_state:
    st.session_state.latest_prediction_context = None


# ============================================================
# 6. MODEL ARTIFACT HELPERS
# ============================================================

def first_available_key(
    artifact: dict[str, Any],
    possible_keys: list[str],
) -> str | None:
    """Return the first existing key from a list of alternatives."""

    for key in possible_keys:
        if key in artifact:
            return key
    return None


@st.cache_resource
def load_model_artifact(model_path: Path) -> dict[str, Any]:
    """
    Load and normalize the latest Gradient Boosting artifact.

    Supported key aliases are included so the application can load
    either the latest names or an earlier equivalent naming scheme.
    """

    if not model_path.exists():
        raise FileNotFoundError(
            "The model file was not found. Expected location:\n"
            f"{model_path}"
        )

    loaded = joblib.load(model_path)

    if not isinstance(loaded, dict):
        raise TypeError(
            "The saved model must be a dictionary artifact. "
            f"Loaded type: {type(loaded).__name__}"
        )

    lgb_key = first_available_key(
        loaded,
        ["lightgbm_model", "lgb_model"],
    )
    xgb_key = first_available_key(
        loaded,
        ["xgboost_model", "xgb_model"],
    )
    preprocessor_key = first_available_key(
        loaded,
        ["preprocessor", "poly_transformer", "transformer"],
    )
    lgb_weight_key = first_available_key(
        loaded,
        ["lightgbm_weight", "blend_weight", "weight"],
    )

    missing_components = []

    if lgb_key is None:
        missing_components.append("LightGBM model")
    if xgb_key is None:
        missing_components.append("XGBoost model")
    if preprocessor_key is None:
        missing_components.append("saved preprocessor")
    if lgb_weight_key is None:
        missing_components.append("LightGBM blend weight")
    if "feature_names" not in loaded:
        missing_components.append("feature_names")

    if missing_components:
        raise KeyError(
            "The PKL artifact is missing required components:\n- "
            + "\n- ".join(missing_components)
            + "\n\nAvailable keys:\n- "
            + "\n- ".join(map(str, loaded.keys()))
        )

    lightgbm_weight = float(loaded[lgb_weight_key])
    xgboost_weight = float(
        loaded.get("xgboost_weight", 1.0 - lightgbm_weight)
    )

    if not 0.0 <= lightgbm_weight <= 1.0:
        raise ValueError(
            f"Invalid LightGBM weight: {lightgbm_weight}"
        )

    if not 0.0 <= xgboost_weight <= 1.0:
        raise ValueError(
            f"Invalid XGBoost weight: {xgboost_weight}"
        )

    if not np.isclose(
        lightgbm_weight + xgboost_weight,
        1.0,
        atol=1e-6,
    ):
        raise ValueError(
            "The LightGBM and XGBoost weights do not add up to 1."
        )

    feature_names = [
        str(feature).strip()
        for feature in loaded["feature_names"]
    ]

    if not feature_names:
        raise ValueError("The saved feature_names list is empty.")

    return {
        "lightgbm_model": loaded[lgb_key],
        "xgboost_model": loaded[xgb_key],
        "preprocessor": loaded[preprocessor_key],
        "lightgbm_weight": lightgbm_weight,
        "xgboost_weight": xgboost_weight,
        "feature_names": feature_names,
        "target_name": loaded.get("target_name", TARGET_NAME),
        "zero_targets_removed": bool(
            loaded.get("zero_targets_removed", False)
        ),
        "raw_artifact": loaded,
        "source_keys": {
            "lightgbm_model": lgb_key,
            "xgboost_model": xgb_key,
            "preprocessor": preprocessor_key,
            "lightgbm_weight": lgb_weight_key,
        },
    }


try:
    artifact = load_model_artifact(MODEL_PATH)
except Exception as error:
    st.error(f"Unable to load the trained model: {error}")
    st.stop()


# ============================================================
# 7. INPUT MAPPINGS
# ============================================================

GENDER_MAPPING = {
    "Female": 0,
    "Male": 1,
}

YES_NO_MAPPING = {
    "No": 0,
    "Yes": 1,
}

HEALTH_MAPPING = {
    "Excellent": 1,
    "Very good": 2,
    "Good": 3,
    "Fair": 4,
    "Poor": 5,
}

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
}


# ============================================================
# 8. CURRENCY CONVERSION
# ============================================================

CURRENCY_OPTIONS = {
    "Chinese Yuan (CNY)": {"code": "CNY", "symbol": "¥"},
    "Malaysian Ringgit (MYR)": {"code": "MYR", "symbol": "RM"},
    "US Dollar (USD)": {"code": "USD", "symbol": "$"},
    "Singapore Dollar (SGD)": {"code": "SGD", "symbol": "S$"},
    "Euro (EUR)": {"code": "EUR", "symbol": "€"},
    "British Pound (GBP)": {"code": "GBP", "symbol": "£"},
}

EXCHANGE_RATE_API_URL = (
    "https://v6.exchangerate-api.com/v6/"
    "{api_key}/latest/{base_currency}"
)


@st.cache_data(ttl=3600)
def get_exchange_rates(
    *,
    api_key: str,
    base_currency: str = "CNY",
) -> dict[str, Any]:
    """Retrieve and cache exchange rates for one hour."""

    if not api_key:
        raise ValueError("EXCHANGE_RATE_API_KEY is missing.")

    base_currency = base_currency.strip().upper()
    url = EXCHANGE_RATE_API_URL.format(
        api_key=api_key,
        base_currency=base_currency,
    )

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.Timeout as error:
        raise RuntimeError(
            "The exchange-rate service timed out."
        ) from error
    except requests.RequestException as error:
        raise RuntimeError(
            "Unable to connect to ExchangeRate-API."
        ) from error

    try:
        data = response.json()
    except ValueError as error:
        raise RuntimeError(
            "ExchangeRate-API returned invalid JSON."
        ) from error

    if data.get("result") != "success":
        raise RuntimeError(
            "ExchangeRate-API error: "
            f"{data.get('error-type', 'unknown-error')}"
        )

    rates = data.get("conversion_rates", {})
    if not rates:
        raise RuntimeError("No exchange rates were returned.")

    return {
        "rates": rates,
        "last_updated": data.get("time_last_update_utc"),
        "next_update": data.get("time_next_update_utc"),
    }


def convert_cny_amount(
    *,
    amount_cny: float,
    target_currency: str,
    api_key: str | None,
) -> dict[str, Any]:
    """Convert a non-negative CNY amount into another currency."""

    if amount_cny < 0:
        raise ValueError("The CNY amount cannot be negative.")

    target_currency = target_currency.strip().upper()

    if target_currency == "CNY":
        return {
            "rate": 1.0,
            "converted_amount": float(amount_cny),
            "last_updated": None,
            "next_update": None,
        }

    if not api_key:
        raise ValueError(
            "EXCHANGE_RATE_API_KEY was not found in Streamlit Secrets."
        )

    rate_data = get_exchange_rates(
        api_key=api_key,
        base_currency="CNY",
    )
    rates = rate_data["rates"]

    if target_currency not in rates:
        raise ValueError(
            f"Currency '{target_currency}' is unsupported."
        )

    rate = float(rates[target_currency])

    return {
        "rate": rate,
        "converted_amount": float(amount_cny) * rate,
        "last_updated": rate_data["last_updated"],
        "next_update": rate_data["next_update"],
    }


# ============================================================
# 9. MODEL INPUT AND PREDICTION FUNCTIONS
# ============================================================

def validate_raw_inputs(
    *,
    age: int,
    height_cm: float,
    weight_kg: float,
    outpatient_cost: float,
    previous_inpatient_cost: float,
) -> float:
    """Validate user inputs and return calculated BMI."""

    errors = []

    if age < 1 or age > 119:
        errors.append("Age must be between 1 and 119.")
    if height_cm <= 0:
        errors.append("Height must be greater than zero.")
    if weight_kg <= 0:
        errors.append("Weight must be greater than zero.")
    if outpatient_cost < 0:
        errors.append("Outpatient medical cost cannot be negative.")
    if previous_inpatient_cost < 0:
        errors.append("Previous inpatient cost cannot be negative.")

    bmi = weight_kg / (height_cm / 100.0) ** 2

    if bmi < 10 or bmi > 80:
        errors.append(
            "The calculated BMI is outside the expected range of "
            "10 to 80. Please verify the height and weight."
        )

    if errors:
        raise ValueError(" ".join(errors))

    return float(bmi)


def create_model_input(
    *,
    required_features: list[str],
    age: int,
    gender_code: int,
    bmi: float,
    chronic_illness_code: int,
    smoker_code: int,
    previous_inpatient_cost: float,
    hospitalized_code: int,
    outpatient_cost: float,
    health_code: int,
    employed_code: int,
) -> pd.DataFrame:
    """
    Create the ten original inputs expected by the saved preprocessor.

    Monetary variables are transformed with np.log1p, matching the
    cleaned modelling dataset.
    """

    raw_values = {
        "age": float(age),
        "gender": float(gender_code),
        "bmi": float(bmi),
        "qp401": float(chronic_illness_code),
        "qq201": float(smoker_code),
        "log_past_qc701": float(np.log1p(previous_inpatient_cost)),
        "qc401": float(hospitalized_code),
        "log_qc7b": float(np.log1p(outpatient_cost)),
        "qp201": float(health_code),
        "qgb1": float(employed_code),
    }

    missing_features = [
        feature
        for feature in required_features
        if feature not in raw_values
    ]

    if missing_features:
        raise KeyError(
            "The latest PKL requires input features that this UI does "
            "not currently create:\n- "
            + "\n- ".join(missing_features)
            + "\n\nSaved feature list:\n- "
            + "\n- ".join(required_features)
        )

    return pd.DataFrame(
        [[raw_values[feature] for feature in required_features]],
        columns=required_features,
        dtype=float,
    )


def transform_model_input(
    artifact: dict[str, Any],
    model_input: pd.DataFrame,
):
    """Apply the preprocessor fitted only on the 2020 training data."""

    required_features = artifact["feature_names"]

    missing_features = [
        feature
        for feature in required_features
        if feature not in model_input.columns
    ]
    if missing_features:
        raise KeyError(
            "Model input is missing required features:\n- "
            + "\n- ".join(missing_features)
        )

    ordered_input = model_input.loc[:, required_features].copy()

    for column in required_features:
        ordered_input[column] = pd.to_numeric(
            ordered_input[column],
            errors="coerce",
        )

    if ordered_input.isna().any().any():
        bad_columns = ordered_input.columns[
            ordered_input.isna().any()
        ].tolist()
        raise ValueError(
            "Model input contains invalid numeric values in:\n- "
            + "\n- ".join(bad_columns)
        )

    transformed_input = artifact["preprocessor"].transform(
        ordered_input
    )

    transformed_array = (
        transformed_input.toarray()
        if hasattr(transformed_input, "toarray")
        else np.asarray(transformed_input)
    )

    if transformed_array.ndim != 2 or transformed_array.shape[0] != 1:
        raise ValueError(
            "The saved preprocessor produced an unexpected shape: "
            f"{transformed_array.shape}"
        )

    return ordered_input, transformed_input, transformed_array


def predict_cost(
    *,
    artifact: dict[str, Any],
    model_input: pd.DataFrame,
) -> dict[str, Any]:
    """Predict log cost and original cost with the saved blend."""

    ordered_input, transformed_input, transformed_array = (
        transform_model_input(artifact, model_input)
    )

    lgb_model = artifact["lightgbm_model"]
    xgb_model = artifact["xgboost_model"]

    expected_lgb = getattr(lgb_model, "n_features_in_", None)
    expected_xgb = getattr(xgb_model, "n_features_in_", None)

    if (
        expected_lgb is not None
        and transformed_array.shape[1] != int(expected_lgb)
    ):
        raise ValueError(
            "The transformed input feature count does not match "
            "LightGBM. "
            f"Generated: {transformed_array.shape[1]}, "
            f"expected: {expected_lgb}."
        )

    if (
        expected_xgb is not None
        and transformed_array.shape[1] != int(expected_xgb)
    ):
        raise ValueError(
            "The transformed input feature count does not match "
            "XGBoost. "
            f"Generated: {transformed_array.shape[1]}, "
            f"expected: {expected_xgb}."
        )

    lgb_log_prediction = float(
        np.asarray(lgb_model.predict(transformed_input)).reshape(-1)[0]
    )
    xgb_log_prediction = float(
        np.asarray(xgb_model.predict(transformed_input)).reshape(-1)[0]
    )

    lightgbm_weight = artifact["lightgbm_weight"]
    xgboost_weight = artifact["xgboost_weight"]

    predicted_log_cost = float(
        lightgbm_weight * lgb_log_prediction
        + xgboost_weight * xgb_log_prediction
    )

    predicted_original_cost = max(
        0.0,
        float(np.expm1(predicted_log_cost)),
    )

    return {
        "predicted_log_cost": predicted_log_cost,
        "predicted_original_cost": predicted_original_cost,
        "lightgbm_log_prediction": lgb_log_prediction,
        "xgboost_log_prediction": xgb_log_prediction,
        "model_input": ordered_input,
        "transformed_input": transformed_input,
        "transformed_array": transformed_array,
    }


# ============================================================
# 10. SHAP EXPLANATION FUNCTIONS
# ============================================================

@st.cache_resource
def create_shap_explainers(_lgb_model, _xgb_model):
    """Create and cache tree explainers for both fitted models."""

    return (
        shap.TreeExplainer(_lgb_model),
        shap.TreeExplainer(_xgb_model),
    )


def extract_single_row_shap_values(explainer, transformed_array):
    """Normalize SHAP output to a one-dimensional array."""

    try:
        result = explainer.shap_values(transformed_array)
    except Exception:
        result = explainer(transformed_array)

    if hasattr(result, "values"):
        values = result.values
    else:
        values = result

    if isinstance(values, list):
        values = values[0]

    values = np.asarray(values)

    if values.ndim == 1:
        return values
    if values.ndim == 2:
        return values[0]
    if values.ndim == 3:
        return values[0, :, 0]

    raise ValueError(
        f"Unexpected SHAP output shape: {values.shape}"
    )


def get_transformed_feature_names(
    artifact: dict[str, Any],
    transformed_feature_count: int,
) -> list[str]:
    """Recover feature names produced by the saved preprocessor."""

    raw_artifact = artifact["raw_artifact"]

    for key in [
        "transformed_feature_names",
        "polynomial_feature_names",
        "processed_feature_names",
    ]:
        if key in raw_artifact:
            names = [str(name) for name in raw_artifact[key]]
            if len(names) == transformed_feature_count:
                return names

    preprocessor = artifact["preprocessor"]

    if hasattr(preprocessor, "get_feature_names_out"):
        try:
            names = list(
                preprocessor.get_feature_names_out(
                    artifact["feature_names"]
                )
            )
            if len(names) == transformed_feature_count:
                return [str(name) for name in names]
        except Exception:
            try:
                names = list(preprocessor.get_feature_names_out())
                if len(names) == transformed_feature_count:
                    return [str(name) for name in names]
            except Exception:
                pass

    return [
        f"Transformed feature {index}"
        for index in range(1, transformed_feature_count + 1)
    ]


def prettify_transformed_feature_name(name: str) -> str:
    """Convert technical transformed feature names into labels."""

    clean_name = str(name)

    # Remove common sklearn prefixes.
    if "__" in clean_name:
        clean_name = clean_name.split("__", 1)[1]

    for raw_name, display_name in FEATURE_LABELS.items():
        clean_name = clean_name.replace(raw_name, display_name)

    clean_name = clean_name.replace("^2", " squared")
    return clean_name


def calculate_top_contributors(
    *,
    artifact: dict[str, Any],
    transformed_array: np.ndarray,
    top_n: int = 5,
) -> pd.DataFrame:
    """Calculate blended SHAP contributions after preprocessing."""

    lgb_explainer, xgb_explainer = create_shap_explainers(
        artifact["lightgbm_model"],
        artifact["xgboost_model"],
    )

    lgb_values = extract_single_row_shap_values(
        lgb_explainer,
        transformed_array,
    )
    xgb_values = extract_single_row_shap_values(
        xgb_explainer,
        transformed_array,
    )

    if len(lgb_values) != transformed_array.shape[1]:
        raise ValueError(
            "The number of LightGBM SHAP values does not match "
            "the transformed feature count."
        )
    if len(xgb_values) != transformed_array.shape[1]:
        raise ValueError(
            "The number of XGBoost SHAP values does not match "
            "the transformed feature count."
        )

    blended_values = (
        artifact["lightgbm_weight"] * lgb_values
        + artifact["xgboost_weight"] * xgb_values
    )

    transformed_names = get_transformed_feature_names(
        artifact,
        transformed_array.shape[1],
    )

    contribution_df = pd.DataFrame(
        {
            "Feature": transformed_names,
            "SHAP contribution": blended_values,
        }
    )
    contribution_df["Absolute contribution"] = contribution_df[
        "SHAP contribution"
    ].abs()
    contribution_df["Effect"] = np.where(
        contribution_df["SHAP contribution"] >= 0,
        "Increased prediction",
        "Decreased prediction",
    )
    contribution_df["Display feature"] = contribution_df[
        "Feature"
    ].map(prettify_transformed_feature_name)

    return (
        contribution_df.sort_values(
            "Absolute contribution",
            ascending=False,
        )
        .head(top_n)
        .reset_index(drop=True)
    )


# ============================================================
# 11. NEXT-YEAR SCENARIO
# ============================================================

def predict_next_year_cost(
    *,
    artifact: dict[str, Any],
    current_age: int,
    gender_code: int,
    height_cm: float,
    weight_kg: float,
    chronic_illness_code: int,
    smoker_code: int,
    hospitalized_code: int,
    outpatient_cost: float,
    health_code: int,
    employed_code: int,
    current_prediction: float,
) -> dict[str, Any]:
    """
    Produce a one-year scenario projection.

    The current prediction becomes the previous inpatient cost,
    age increases by one year, and all other inputs remain unchanged.
    """

    next_year_age = current_age + 1
    bmi = weight_kg / (height_cm / 100.0) ** 2

    next_year_input = create_model_input(
        required_features=artifact["feature_names"],
        age=next_year_age,
        gender_code=gender_code,
        bmi=bmi,
        chronic_illness_code=chronic_illness_code,
        smoker_code=smoker_code,
        previous_inpatient_cost=current_prediction,
        hospitalized_code=hospitalized_code,
        outpatient_cost=outpatient_cost,
        health_code=health_code,
        employed_code=employed_code,
    )

    prediction = predict_cost(
        artifact=artifact,
        model_input=next_year_input,
    )

    return {
        "age": next_year_age,
        "predicted_log_cost": prediction["predicted_log_cost"],
        "predicted_original_cost": prediction[
            "predicted_original_cost"
        ],
        "model_input": prediction["model_input"],
    }


# ============================================================
# 12. APPLICATION HEADER
# ============================================================

st.title("🏥 Medical Cost Prediction")

st.write(
    "Enter the individual's demographic, health, healthcare-use, "
    "and employment information below."
)

st.info(
    "This application estimates inpatient medical cost using a "
    "blended LightGBM and XGBoost regression model. The result is "
    "an estimate and should not be treated as medical or financial "
    "advice."
)

st.caption(f"Model version: {MODEL_VERSION}")
st.caption(f"Model file: {MODEL_PATH.name}")


# ============================================================
# 13. MODEL COMPATIBILITY STATUS
# ============================================================

saved_feature_names = artifact["feature_names"]

with st.expander("Model compatibility status"):
    st.write("Saved original features:", saved_feature_names)
    st.write("Number of saved original features:", len(saved_feature_names))
    st.write("LightGBM weight:", artifact["lightgbm_weight"])
    st.write("XGBoost weight:", artifact["xgboost_weight"])
    st.write("Target:", artifact["target_name"])
    st.write(
        "Zero targets removed during training:",
        artifact["zero_targets_removed"],
    )

    if saved_feature_names == EXPECTED_RAW_FEATURES:
        st.success(
            "The saved feature list exactly matches the latest "
            "10-feature Streamlit interface."
        )
    elif set(saved_feature_names) == set(EXPECTED_RAW_FEATURES):
        st.success(
            "The saved feature set matches the latest interface. "
            "The application will use the saved feature order."
        )
    else:
        missing_from_ui = [
            feature
            for feature in saved_feature_names
            if feature not in EXPECTED_RAW_FEATURES
        ]
        unused_ui_features = [
            feature
            for feature in EXPECTED_RAW_FEATURES
            if feature not in saved_feature_names
        ]

        if missing_from_ui:
            st.error(
                "Features required by the PKL but not created by this UI: "
                f"{missing_from_ui}"
            )
        if unused_ui_features:
            st.warning(
                "UI features not required by the PKL: "
                f"{unused_ui_features}"
            )


# ============================================================
# 14. USER INPUT FORM
# ============================================================

with st.form("medical_cost_form"):
    st.subheader("Personal information")

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=119,
        value=40,
        step=1,
        help=(
            "The maximum age is 119 because the next-year scenario "
            "increases age by one year."
        ),
    )

    gender_label = st.selectbox(
        "Gender",
        options=list(GENDER_MAPPING.keys()),
    )

    height_cm = st.number_input(
        "Height (cm)",
        min_value=50.0,
        max_value=250.0,
        value=165.0,
        step=0.1,
    )

    weight_kg = st.number_input(
        "Weight (kg)",
        min_value=10.0,
        max_value=300.0,
        value=60.0,
        step=0.1,
    )

    calculated_bmi = weight_kg / (height_cm / 100.0) ** 2
    st.caption(f"Calculated BMI: {calculated_bmi:.2f}")

    st.subheader("Health and healthcare information")

    chronic_illness_label = st.selectbox(
        "Has the individual been diagnosed with a chronic illness?",
        options=list(YES_NO_MAPPING.keys()),
    )

    smoker_label = st.selectbox(
        "Does the individual smoke?",
        options=list(YES_NO_MAPPING.keys()),
    )

    hospitalized_label = st.selectbox(
        "Was the individual hospitalized during the survey period?",
        options=list(YES_NO_MAPPING.keys()),
    )

    health_label = st.selectbox(
        "How would the individual rate their health?",
        options=list(HEALTH_MAPPING.keys()),
        index=2,
    )

    outpatient_cost = st.number_input(
        "Outpatient medical cost (CNY)",
        min_value=0.0,
        value=0.0,
        step=100.0,
        help=(
            "The app applies log1p to create log_qc7b before "
            "sending the value to the saved preprocessor."
        ),
    )

    previous_inpatient_cost = st.number_input(
        "Previous inpatient medical cost (CNY)",
        min_value=0.0,
        value=0.0,
        step=100.0,
        help=(
            "The app applies log1p to create log_past_qc701 before "
            "prediction."
        ),
    )

    st.subheader("Employment information")

    employed_label = st.selectbox(
        "Is the individual currently employed?",
        options=list(YES_NO_MAPPING.keys()),
    )

    st.subheader("Display preference")

    selected_currency_label = st.selectbox(
        "Display predicted cost in",
        options=list(CURRENCY_OPTIONS.keys()),
        index=1,
        help=(
            "The model prediction remains in Chinese yuan. The "
            "selected currency is used only for an approximate display."
        ),
    )

    submitted = st.form_submit_button(
        "Predict medical cost",
        use_container_width=True,
        type="primary",
    )


# ============================================================
# 15. PROCESS USER INPUT
# ============================================================

if submitted:
    try:
        validated_bmi = validate_raw_inputs(
            age=int(age),
            height_cm=float(height_cm),
            weight_kg=float(weight_kg),
            outpatient_cost=float(outpatient_cost),
            previous_inpatient_cost=float(previous_inpatient_cost),
        )

        gender_code = GENDER_MAPPING[gender_label]
        chronic_illness_code = YES_NO_MAPPING[chronic_illness_label]
        smoker_code = YES_NO_MAPPING[smoker_label]
        hospitalized_code = YES_NO_MAPPING[hospitalized_label]
        health_code = HEALTH_MAPPING[health_label]
        employed_code = YES_NO_MAPPING[employed_label]

        model_input = create_model_input(
            required_features=artifact["feature_names"],
            age=int(age),
            gender_code=gender_code,
            bmi=validated_bmi,
            chronic_illness_code=chronic_illness_code,
            smoker_code=smoker_code,
            previous_inpatient_cost=float(previous_inpatient_cost),
            hospitalized_code=hospitalized_code,
            outpatient_cost=float(outpatient_cost),
            health_code=health_code,
            employed_code=employed_code,
        )

        prediction_result = predict_cost(
            artifact=artifact,
            model_input=model_input,
        )

        predicted_log_cost = prediction_result["predicted_log_cost"]
        predicted_medical_cost = prediction_result[
            "predicted_original_cost"
        ]
        lgb_log_prediction = prediction_result[
            "lightgbm_log_prediction"
        ]
        xgb_log_prediction = prediction_result[
            "xgboost_log_prediction"
        ]
        ordered_model_input = prediction_result["model_input"]
        transformed_array = prediction_result["transformed_array"]

        selected_currency = CURRENCY_OPTIONS[selected_currency_label]
        selected_currency_code = selected_currency["code"]
        selected_currency_symbol = selected_currency["symbol"]

        currency_result = None
        currency_error_message = None

        try:
            currency_result = convert_cny_amount(
                amount_cny=predicted_medical_cost,
                target_currency=selected_currency_code,
                api_key=EXCHANGE_RATE_API_KEY,
            )
        except Exception as currency_error:
            currency_error_message = str(currency_error)

        converted_cost_for_chat = predicted_medical_cost
        exchange_rate_for_chat = None

        if currency_result is not None:
            converted_cost_for_chat = float(
                currency_result["converted_amount"]
            )
            exchange_rate_for_chat = float(currency_result["rate"])

        st.session_state.latest_prediction_context = {
            "predicted_cost_cny": float(predicted_medical_cost),
            "predicted_log_cost": float(predicted_log_cost),
            "display_currency": selected_currency_code,
            "display_currency_symbol": selected_currency_symbol,
            "converted_cost": float(converted_cost_for_chat),
            "exchange_rate": exchange_rate_for_chat,
            "bmi": float(validated_bmi),
            "age": int(age),
            "gender": gender_label,
            "hospitalized": hospitalized_label,
            "chronic_illness": chronic_illness_label,
            "smoking_status": smoker_label,
            "health_status": health_label,
            "employment_status": employed_label,
            "top_factors": [],
        }

        st.success("Prediction completed successfully.")

        st.metric(
            label="Estimated inpatient medical cost in Chinese yuan",
            value=f"¥{predicted_medical_cost:,.2f} CNY",
        )

        if (
            currency_result is not None
            and selected_currency_code != "CNY"
        ):
            st.metric(
                label="Approximate converted cost",
                value=(
                    f"{selected_currency_symbol}"
                    f"{currency_result['converted_amount']:,.2f} "
                    f"{selected_currency_code}"
                ),
            )
            st.caption(
                "Exchange rate used: "
                f"1 CNY = {currency_result['rate']:.6f} "
                f"{selected_currency_code}"
            )
            if currency_result["last_updated"]:
                st.caption(
                    "Exchange-rate update time: "
                    f"{currency_result['last_updated']}"
                )
            st.warning(
                "The converted value is approximate. Banks and "
                "payment providers may use different rates or fees."
            )
        elif currency_error_message:
            st.warning(
                "The prediction succeeded, but currency conversion "
                f"was unavailable: {currency_error_message}"
            )

        summary_col1, summary_col2 = st.columns(2)
        with summary_col1:
            st.metric("Calculated BMI", f"{validated_bmi:.2f}")
        with summary_col2:
            st.metric("Log-scale prediction", f"{predicted_log_cost:.4f}")

        # ----------------------------------------------------
        # Next-year scenario
        # ----------------------------------------------------

        st.divider()
        st.subheader("Next-year medical-cost projection")

        next_year_result = predict_next_year_cost(
            artifact=artifact,
            current_age=int(age),
            gender_code=gender_code,
            height_cm=float(height_cm),
            weight_kg=float(weight_kg),
            chronic_illness_code=chronic_illness_code,
            smoker_code=smoker_code,
            hospitalized_code=hospitalized_code,
            outpatient_cost=float(outpatient_cost),
            health_code=health_code,
            employed_code=employed_code,
            current_prediction=predicted_medical_cost,
        )

        next_year_cost = float(
            next_year_result["predicted_original_cost"]
        )
        cost_change = next_year_cost - predicted_medical_cost
        percentage_change = (
            cost_change / predicted_medical_cost * 100.0
            if predicted_medical_cost > 0
            else 0.0
        )

        st.metric(
            label="Estimated inpatient medical cost next year",
            value=f"¥{next_year_cost:,.2f} CNY",
            delta=f"¥{cost_change:,.2f} CNY",
        )

        comparison_df = pd.DataFrame(
            {
                "Period": [
                    "Current prediction",
                    "Next-year projection",
                ],
                "Medical cost": [
                    predicted_medical_cost,
                    next_year_cost,
                ],
            }
        )
        st.bar_chart(
            comparison_df.set_index("Period"),
            use_container_width=True,
        )

        direction = "higher" if percentage_change >= 0 else "lower"
        st.write(
            "The projected cost is "
            f"**{abs(percentage_change):.2f}% {direction}** than "
            "the current prediction."
        )
        st.warning(
            "This is a scenario projection, not a validated time-series "
            "forecast. It increases age by one year and uses the current "
            "prediction as the previous inpatient cost while keeping "
            "other entered conditions unchanged."
        )

        # ----------------------------------------------------
        # SHAP explanation
        # ----------------------------------------------------

        st.divider()
        st.subheader("Top factors influencing the prediction")

        try:
            top_contributors = calculate_top_contributors(
                artifact=artifact,
                transformed_array=transformed_array,
                top_n=5,
            )

            top_factor_context = []
            for _, row in top_contributors.iterrows():
                top_factor_context.append(
                    {
                        "feature": row["Display feature"],
                        "effect": row["Effect"],
                        "contribution": float(
                            row["SHAP contribution"]
                        ),
                    }
                )

            if st.session_state.latest_prediction_context is not None:
                st.session_state.latest_prediction_context[
                    "top_factors"
                ] = top_factor_context

            st.bar_chart(
                top_contributors.set_index("Display feature")[[
                    "Absolute contribution"
                ]],
                use_container_width=True,
            )

            for rank, row in top_contributors.iterrows():
                contribution = float(row["SHAP contribution"])
                icon = "⬆️" if contribution >= 0 else "⬇️"
                direction_text = (
                    "increased the predicted cost"
                    if contribution >= 0
                    else "reduced the predicted cost"
                )
                st.write(
                    f"{rank + 1}. {icon} **{row['Display feature']}** "
                    f"{direction_text}."
                )

            with st.expander("View detailed feature contributions"):
                st.dataframe(
                    top_contributors[[
                        "Display feature",
                        "SHAP contribution",
                        "Absolute contribution",
                        "Effect",
                    ]].rename(
                        columns={"Display feature": "Feature"}
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption(
                    "SHAP values are calculated after the saved "
                    "preprocessor and are measured on the model's "
                    "log-cost scale. They explain model behaviour, not "
                    "medical causation."
                )
        except Exception as shap_error:
            st.warning(
                "The prediction worked, but SHAP explanation could "
                f"not be generated: {shap_error}"
            )

        # ----------------------------------------------------
        # Prediction verification
        # ----------------------------------------------------

        st.divider()
        st.subheader("Prediction verification")

        manual_blend = (
            artifact["lightgbm_weight"] * lgb_log_prediction
            + artifact["xgboost_weight"] * xgb_log_prediction
        )
        retransformed_cost = max(
            0.0,
            float(np.expm1(predicted_log_cost)),
        )

        blend_passed = np.isclose(
            predicted_log_cost,
            manual_blend,
            rtol=1e-12,
            atol=1e-12,
        )
        retransformation_passed = np.isclose(
            predicted_medical_cost,
            retransformed_cost,
            rtol=1e-12,
            atol=1e-12,
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
                    predicted_medical_cost,
                ],
                "Absolute difference": [
                    abs(predicted_log_cost - manual_blend),
                    abs(predicted_medical_cost - retransformed_cost),
                ],
                "Status": [
                    "Pass" if blend_passed else "Fail",
                    "Pass" if retransformation_passed else "Fail",
                ],
            }
        )
        st.dataframe(
            verification_df,
            use_container_width=True,
            hide_index=True,
        )

        if blend_passed and retransformation_passed:
            st.success(
                "The prediction passed the built-in consistency checks."
            )
        else:
            st.error(
                "The prediction failed at least one consistency check."
            )

        # ----------------------------------------------------
        # Technical details
        # ----------------------------------------------------

        with st.expander("View technical prediction details"):
            st.write("Model version:", MODEL_VERSION)
            st.write("Model path:", str(MODEL_PATH))
            st.write("Required original features:", saved_feature_names)
            st.write(
                "Transformed feature count:",
                int(transformed_array.shape[1]),
            )
            st.write(
                "LightGBM weight:", artifact["lightgbm_weight"]
            )
            st.write(
                "XGBoost weight:", artifact["xgboost_weight"]
            )
            st.write("LightGBM log prediction:", lgb_log_prediction)
            st.write("XGBoost log prediction:", xgb_log_prediction)
            st.write("Final blended log prediction:", predicted_log_cost)
            st.write(
                "Final original-scale prediction:",
                predicted_medical_cost,
            )
            st.write("Generated original model input:")
            st.dataframe(
                ordered_model_input,
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                label="Download generated model input",
                data=ordered_model_input.to_csv(index=False),
                file_name="generated_model_input.csv",
                mime="text/csv",
                use_container_width=True,
            )

    except Exception as error:
        st.error(f"Prediction failed: {error}")


# ============================================================
# 16. GEMINI PREDICTION-AWARE CHATBOT
# ============================================================

st.divider()
st.subheader("💬 Medical Cost Prediction Assistant")

st.caption(
    "Ask about the estimated cost, SHAP factors, or how to "
    "interpret the model output. This assistant does not provide "
    "medical diagnosis or treatment advice."
)

if st.session_state.latest_prediction_context is None:
    st.info(
        "Generate a medical-cost prediction first for a personalized "
        "explanation. General model questions can still be asked."
    )
else:
    latest_cost = st.session_state.latest_prediction_context[
        "predicted_cost_cny"
    ]
    st.success(
        "Latest prediction available to the assistant: "
        f"¥{latest_cost:,.2f} CNY."
    )

if st.button("Clear chat", key="clear_gemini_chat"):
    st.session_state.chat_messages = [
        {
            "role": "assistant",
            "content": (
                "Chat history cleared. I can explain the medical-cost "
                "prediction and model factors."
            ),
        }
    ]
    st.rerun()

for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_message = st.chat_input(
    "Ask why the cost is high or what the factors mean",
    key="medical_cost_chat_input",
)

if user_message:
    st.session_state.chat_messages.append(
        {"role": "user", "content": user_message}
    )

    with st.chat_message("user"):
        st.markdown(user_message)

    with st.chat_message("assistant"):
        if not gemini_available or gemini_client is None:
            assistant_response = (
                "The Gemini assistant is unavailable. Check the "
                "GEMINI_API_KEY secret and the google-genai dependency."
            )
        else:
            try:
                prediction_context = (
                    st.session_state.latest_prediction_context
                )

                if prediction_context is None:
                    prediction_text = (
                        "No prediction has been generated in this "
                        "session. Explain only the model in general "
                        "terms and do not invent patient values."
                    )
                else:
                    top_factors = prediction_context.get(
                        "top_factors", []
                    )

                    if top_factors:
                        factor_text = "\n".join(
                            "- "
                            f"{factor['feature']}: {factor['effect']} "
                            f"(SHAP contribution "
                            f"{factor['contribution']:.4f})"
                            for factor in top_factors
                        )
                    else:
                        factor_text = (
                            "SHAP contributors are unavailable for "
                            "this prediction."
                        )

                    prediction_text = f"""
Latest model context:

Predicted inpatient medical cost:
¥{prediction_context['predicted_cost_cny']:,.2f} CNY

Log-scale prediction:
{prediction_context['predicted_log_cost']:.4f}

Selected display currency:
{prediction_context['display_currency']}

Approximate converted cost:
{prediction_context['display_currency_symbol']}{prediction_context['converted_cost']:,.2f} {prediction_context['display_currency']}

Exchange rate from CNY:
{prediction_context.get('exchange_rate')}

Age: {prediction_context['age']}
Gender: {prediction_context['gender']}
Calculated BMI: {prediction_context['bmi']:.2f}
Hospitalized: {prediction_context['hospitalized']}
Chronic illness: {prediction_context['chronic_illness']}
Smoking status: {prediction_context['smoking_status']}
Self-rated health: {prediction_context['health_status']}
Employment status: {prediction_context['employment_status']}

Top model contributors:
{factor_text}
"""

                recent_messages = st.session_state.chat_messages[-8:]
                conversation_text = "\n".join(
                    f"{message['role']}: {message['content']}"
                    for message in recent_messages
                )

                gemini_prompt = f"""
You are an educational assistant inside a machine-learning
application that estimates inpatient medical costs.

The prediction is produced by a blended LightGBM and XGBoost
regression model. The application sends the ten original input
features through the preprocessor fitted on the 2020 training data
and then combines both model predictions using saved weights.

Rules:
1. Explain the predicted cost in simple language.
2. Explain SHAP factors and their direction.
3. State that SHAP describes model behaviour and not causation.
4. State that the result is an estimate, not a guaranteed bill.
5. Do not diagnose diseases or recommend treatment.
6. Do not recommend insurance or financial products.
7. Do not invent values, metrics, or patient details.
8. Explain that currency conversion is approximate.
9. Keep the response concise and directly relevant.

{prediction_text}

Recent conversation:
{conversation_text}

Latest user message:
{user_message}
"""

                with st.spinner("Generating explanation..."):
                    gemini_response = (
                        gemini_client.models.generate_content(
                            model="gemini-2.5-flash",
                            contents=gemini_prompt,
                        )
                    )

                response_text = getattr(gemini_response, "text", None)
                assistant_response = (
                    response_text.strip()
                    if response_text
                    else "Gemini returned an empty response."
                )

            except Exception as error:
                assistant_response = (
                    "The assistant could not generate a response: "
                    f"{error}"
                )

        st.markdown(assistant_response)

    st.session_state.chat_messages.append(
        {"role": "assistant", "content": assistant_response}
    )
