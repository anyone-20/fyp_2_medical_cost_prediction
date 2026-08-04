# ============================================================
# STREAMLIT MEDICAL COST PREDICTION APPLICATION
# Blended LightGBM + XGBoost Model
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import shap
import streamlit as st
import requests

from google import genai
from feature_engineering import create_model_features

from model_service import (
    load_model_artifact,
    predict_cost,
)

# ============================================================
# PAGE CONFIGURATION
# Must be the first Streamlit command in the script.
# ============================================================

st.set_page_config(
    page_title="Medical Cost Prediction",
    page_icon="🏥",
    layout="centered",
)


# ============================================================
# READ STREAMLIT SECRETS
# ============================================================

def get_secret(
    key: str,
) -> str | None:
    """
    Read a Streamlit secret without crashing the app.
    """

    try:
        return str(
            st.secrets[key]
        )

    except KeyError:
        return None


GEMINI_API_KEY = get_secret(
    "GEMINI_API_KEY"
)

EXCHANGE_RATE_API_KEY = get_secret(
    "EXCHANGE_RATE_API_KEY"
)

# ============================================================
# GEMINI CLIENT
# ============================================================

@st.cache_resource
def load_gemini_client(
    api_key: str,
):
    """
    Create and cache the Gemini API client.
    """

    return genai.Client(
        api_key=api_key
    )


if GEMINI_API_KEY:

    try:
        gemini_client = load_gemini_client(
            GEMINI_API_KEY
        )

        gemini_available = True

    except Exception as error:
        gemini_client = None
        gemini_available = False

        st.warning(
            "Gemini could not be initialized: "
            f"{error}"
        )

else:
    gemini_client = None
    gemini_available = False

# ============================================================
# GEMINI CONNECTION TEST
# ============================================================

def test_gemini_connection() -> str:
    """
    Send a minimal request to confirm that the configured
    Gemini API key and client are working.
    """

    if not gemini_available or gemini_client is None:
        raise RuntimeError(
            "Gemini is unavailable. Check GEMINI_API_KEY "
            "in Streamlit Community Cloud Secrets."
        )

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=(
            "Reply with exactly: Gemini connection successful."
        ),
    )

    response_text = getattr(response, "text", None)

    if not response_text:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    return response_text.strip()


# ============================================================
# INITIALIZE CHAT AND PREDICTION STATE
# ============================================================

if "chat_messages" not in st.session_state:

    st.session_state.chat_messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! I can explain your predicted "
                "inpatient medical cost, the important "
                "model factors, and the meaning of the "
                "result. I cannot provide a medical "
                "diagnosis or treatment advice."
            ),
        }
    ]


if "latest_prediction_context" not in st.session_state:

    st.session_state.latest_prediction_context = None
    
# ============================================================
# 2. PATH AND MODEL INFORMATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "medical_cost_model.pkl"
)

MODEL_VERSION = (
    "LightGBM-XGBoost Blend 1.0"
)


# ============================================================
# CURRENCY DISPLAY OPTIONS
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
# 3. HUMAN-READABLE FEATURE LABELS
# ============================================================

FEATURE_LABELS = {
    "log_qc7b": (
        "Outpatient medical cost"
    ),

    "qc401 age": (
        "Hospitalization and age"
    ),

    "qc401 bmi": (
        "Hospitalization and BMI"
    ),

    "qc401": (
        "Hospitalization status"
    ),

    "log_qc7b bmi": (
        "Outpatient cost and BMI"
    ),

    "log_qc7b age": (
        "Outpatient cost and age"
    ),

    "qgb1": (
        "Employment status"
    ),

    "log_past_qc701": (
        "Previous inpatient medical cost"
    ),

    "qp201": (
        "Self-rated health"
    ),

    "qp401": (
        "Chronic illness status"
    ),

    "qp605_s_1": (
        "Medical insurance category"
    ),

    "age": (
        "Age"
    ),

    "bmi": (
        "BMI"
    ),

    "log_qc7b qc401": (
        "Outpatient cost and hospitalization"
    ),

    "bmi age": (
        "BMI and age"
    ),

    "qp102": (
        "Body weight"
    ),
    "log_qi202": (
        "Retired Allowance"
    ),
}


# ============================================================
# 4. INPUT MAPPINGS
# ============================================================

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


INSURANCE_MAPPING = {
    "Insurance category 1": 1,
    "Insurance category 2": 2,
    "Insurance category 3": 3,
    "Insurance category 4": 4,
    "Insurance category 5": 5,
    "Insurance category 6": 6,
    "No insurance": 78,
}


# ============================================================
# CURRENCY CONVERSION SERVICE
# ============================================================

EXCHANGE_RATE_API_URL = (
    "https://v6.exchangerate-api.com/v6/"
    "{api_key}/latest/{base_currency}"
)


@st.cache_data(ttl=3600)
def get_exchange_rates(
    *,
    api_key: str,
    base_currency: str = "CNY",
):
    """
    Retrieve and cache exchange rates for one hour.

    ExchangeRate-API returns all supported target-currency
    rates relative to the requested base currency.
    """

    if not api_key:
        raise ValueError(
            "EXCHANGE_RATE_API_KEY is missing."
        )

    base_currency = base_currency.strip().upper()

    url = EXCHANGE_RATE_API_URL.format(
        api_key=api_key,
        base_currency=base_currency,
    )

    try:
        response = requests.get(
            url,
            timeout=10,
        )
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
        error_type = data.get(
            "error-type",
            "unknown-error",
        )

        raise RuntimeError(
            f"ExchangeRate-API error: {error_type}"
        )

    rates = data.get(
        "conversion_rates",
        {}
    )

    if not rates:
        raise RuntimeError(
            "No exchange rates were returned."
        )

    return {
        "base_currency": (
            data.get("base_code")
            or base_currency
        ),
        "rates": rates,
        "last_updated": data.get(
            "time_last_update_utc"
        ),
        "next_update": data.get(
            "time_next_update_utc"
        ),
    }


def convert_cny_amount(
    *,
    amount_cny: float,
    target_currency: str,
    api_key: str | None,
):
    """
    Convert a non-negative CNY amount into a display currency.
    """

    if amount_cny < 0:
        raise ValueError(
            "The CNY amount cannot be negative."
        )

    target_currency = (
        target_currency
        .strip()
        .upper()
    )

    if target_currency == "CNY":
        return {
            "rate": 1.0,
            "converted_amount": float(
                amount_cny
            ),
            "last_updated": None,
            "next_update": None,
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
            f"Currency '{target_currency}' is unsupported."
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
        "next_update": rate_data[
            "next_update"
        ],
    }


# ============================================================
# 5. LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    """
    Load and validate the trained model artifact.
    """

    return load_model_artifact(
        model_path=MODEL_PATH
    )


# ============================================================
# 6. NEXT-YEAR COST PROJECTION
# ============================================================

def predict_next_year_cost(
    *,
    artifact,
    current_age,
    height_cm,
    weight_kg,
    hospitalized_code,
    outpatient_cost,
    current_prediction,
    retired_allowance,
    employed_code,
    health_code,
    chronic_illness_code,
    insurance_code,
):
    """
    Produce a one-year scenario projection.

    Assumptions
    -----------
    1. Age increases by one year.
    2. Height and weight remain unchanged.
    3. All other entered conditions remain unchanged.
    4. The current prediction becomes the previous inpatient
       cost for the next-year scenario.
    """

    next_year_age = (
        current_age + 1
    )

    next_year_input = create_model_features(
        age=next_year_age,
        height_cm=height_cm,
        weight_kg=weight_kg,
        hospitalized_code=(
            hospitalized_code
        ),
        outpatient_cost=(
            outpatient_cost
        ),
        previous_inpatient_cost=(
            current_prediction
        ),
        qi202_value=(
            retired_allowance
        ),
        employed_code=(
            employed_code
        ),
        health_code=(
            health_code
        ),
        chronic_illness_code=(
            chronic_illness_code
        ),
        insurance_code=(
            insurance_code
        ),
        required_features=artifact[
            "feature_names"
        ],
    )

    prediction_result = predict_cost(
        artifact=artifact,
        model_input=next_year_input,
    )

    return {
        "Age": next_year_age,

        "Log cost": prediction_result[
            "predicted_log_cost"
        ],

        "Predicted cost": prediction_result[
            "predicted_original_cost"
        ],

        "Model input": prediction_result[
            "model_input"
        ],
    }


# ============================================================
# 7. SHAP HELPER FUNCTIONS
# ============================================================

def extract_shap_values(
    explainer,
    model_input,
):
    """
    Convert SHAP output into a one-dimensional array.
    """

    shap_result = explainer(
        model_input
    )

    if hasattr(
        shap_result,
        "values"
    ):
        values = shap_result.values

    else:
        values = shap_result

    values = np.asarray(
        values
    )

    if values.ndim == 1:
        return values

    if values.ndim == 2:
        return values[0]

    if values.ndim == 3:
        return values[
            0,
            :,
            0
        ]

    raise ValueError(
        "Unexpected SHAP output shape: "
        f"{values.shape}"
    )


@st.cache_resource
def create_shap_explainers(
    _lgb_model,
    _xgb_model,
):
    """
    Create and cache SHAP explainers.

    Leading underscores prevent Streamlit from attempting
    to hash the fitted model objects.
    """

    lgb_explainer = (
        shap.TreeExplainer(
            _lgb_model
        )
    )

    xgb_explainer = (
        shap.TreeExplainer(
            _xgb_model
        )
    )

    return (
        lgb_explainer,
        xgb_explainer,
    )


def calculate_top_contributors(
    *,
    artifact,
    model_input,
    top_n=3,
):
    """
    Calculate blended SHAP contributions for the
    LightGBM-XGBoost ensemble.
    """

    required_features = list(
        artifact[
            "feature_names"
        ]
    )

    ordered_input = model_input[
        required_features
    ]

    blend_weight = float(
        artifact[
            "blend_weight"
        ]
    )

    (
        lgb_explainer,
        xgb_explainer,
    ) = create_shap_explainers(
        artifact[
            "lgb_model"
        ],
        artifact[
            "xgb_model"
        ],
    )

    lgb_values = extract_shap_values(
        lgb_explainer,
        ordered_input,
    )

    xgb_values = extract_shap_values(
        xgb_explainer,
        ordered_input,
    )

    if len(
        lgb_values
    ) != len(
        required_features
    ):
        raise ValueError(
            "The number of LightGBM SHAP values does not "
            "match the number of model features."
        )

    if len(
        xgb_values
    ) != len(
        required_features
    ):
        raise ValueError(
            "The number of XGBoost SHAP values does not "
            "match the number of model features."
        )

    blended_values = (
        blend_weight
        * lgb_values
        + (
            1 - blend_weight
        )
        * xgb_values
    )

    contribution_df = pd.DataFrame(
        {
            "Feature": (
                required_features
            ),

            "SHAP contribution": (
                blended_values
            ),
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

    contribution_df[
        "Display feature"
    ] = contribution_df[
        "Feature"
    ].map(
        FEATURE_LABELS
    )

    contribution_df[
        "Display feature"
    ] = contribution_df[
        "Display feature"
    ].fillna(
        contribution_df[
            "Feature"
        ]
    )

    contribution_df = (
        contribution_df
        .sort_values(
            by=(
                "Absolute contribution"
            ),
            ascending=False,
        )
        .head(
            top_n
        )
        .reset_index(
            drop=True
        )
    )

    return contribution_df


# ============================================================
# 8. VALIDATE RAW USER INPUT
# ============================================================

def validate_raw_inputs(
    *,
    age,
    height_cm,
    weight_kg,
    outpatient_cost,
    previous_inpatient_cost,
    retired_allowance,
):
    """
    Validate raw values entered through the UI.
    """

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

    if retired_allowance < 0:
        errors.append(
            "Retired allowance cannot be negative."
        )

    bmi = (
        weight_kg
        / (
            height_cm / 100
        ) ** 2
    )

    if bmi < 10 or bmi > 80:
        errors.append(
            "The calculated BMI is outside the expected "
            "range of 10 to 80. Please verify the height "
            "and weight values."
        )

    if errors:
        raise ValueError(
            " ".join(
                errors
            )
        )

    return bmi


# ============================================================
# 9. LOAD TRAINED MODEL
# ============================================================

try:
    artifact = load_model()

except Exception as error:
    st.error(
        "Unable to load the trained model: "
        f"{error}"
    )

    st.stop()


# ============================================================
# 10. APPLICATION HEADER
# ============================================================

st.title(
    "🏥 Medical Cost Prediction"
)

st.write(
    "Enter the individual's personal, healthcare, "
    "employment, and insurance information below."
)

st.info(
    "This application estimates inpatient medical cost "
    "using a blended LightGBM and XGBoost model. The result "
    "should not be treated as medical or financial advice."
)

st.caption(
    f"Model version: {MODEL_VERSION}"
)




# ============================================================
# 11. USER INPUT FORM
# ============================================================

with st.form(
    "medical_cost_form"
):

    st.subheader(
        "Personal information"
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=119,
        value=40,
        step=1,
        key="age_input",
        help=(
            "The maximum age is 119 because the next-year "
            "projection increases age by one year."
        ),
    )

    height_cm = st.number_input(
        "Height (cm)",
        min_value=50.0,
        max_value=250.0,
        value=165.0,
        step=0.1,
        key="height_input",
        help=(
            "Enter height in centimetres."
        ),
    )

    weight_kg = st.number_input(
        "Weight (kg)",
        min_value=10.0,
        max_value=300.0,
        value=60.0,
        step=0.1,
        key="weight_input",
        help=(
            "Enter weight in kilograms."
        ),
    )

    calculated_bmi = (
        weight_kg
        / (
            height_cm / 100
        ) ** 2
    )

    st.caption(
        "Calculated BMI: "
        f"{calculated_bmi:.2f}"
    )

    st.subheader(
        "Healthcare information"
    )

    hospitalized_label = st.selectbox(
        "Were you hospitalized during the survey period?",
        options=[
            "No",
            "Yes",
        ],
        key="hospitalized_input",
    )

    outpatient_cost = st.number_input(
        "Outpatient medical cost",
        min_value=0.0,
        value=0.0,
        step=100.0,
        key="outpatient_cost_input",
        help=(
            "Enter the individual's outpatient medical "
            "cost during the survey period."
        ),
    )

    previous_inpatient_cost = st.number_input(
        "Previous inpatient medical cost",
        min_value=0.0,
        value=0.0,
        step=100.0,
        key="previous_cost_input",
        help=(
            "Enter the inpatient medical cost from the "
            "previous survey period."
        ),
    )

    chronic_illness_label = st.selectbox(
        "Has the individual been diagnosed with a chronic illness?",
        options=[
            "No",
            "Yes",
        ],
        key="chronic_input",
    )

    health_label = st.selectbox(
        "How would the individual rate their health?",
        options=[
            "Excellent",
            "Very good",
            "Good",
            "Fair",
            "Poor",
        ],
        key="health_input",
    )

    st.subheader(
        "Employment and insurance"
    )

    employed_label = st.selectbox(
        "Is the individual currently employed?",
        options=[
            "No",
            "Yes",
        ],
        key="employment_input",
    )

    retired_allowance = st.number_input(
        "Retired allowance (QI202)",
        min_value=0.0,
        value=0.0,
        step=100.0,
        key="retired_allowance_input",
        help=(
            "Enter the raw QI202 retired-allowance value "
            "using the same currency unit and survey period "
            "as the training dataset."
        ),
    )

    insurance_label = st.selectbox(
        "Medical insurance category",
        options=[
            "Insurance category 1",
            "Insurance category 2",
            "Insurance category 3",
            "Insurance category 4",
            "Insurance category 5",
            "Insurance category 6",
            "No insurance",
        ],
        key="insurance_input",
        help=(
            "Replace the generic labels with the official "
            "insurance descriptions used in the survey."
        ),
    )

    st.subheader(
        "Display preference"
    )

    selected_currency_label = st.selectbox(
        "Display predicted cost in",
        options=list(
            CURRENCY_OPTIONS.keys()
        ),
        index=1,
        key="display_currency_input",
        help=(
            "The model prediction remains in Chinese yuan. "
            "The selected currency is used only to display "
            "an approximate converted value."
        ),
    )

    submitted = st.form_submit_button(
        "Predict medical cost",
        use_container_width=True,
        type="primary",
    )


# ============================================================
# 12. PROCESS USER INPUT
# ============================================================

if submitted:

    try:
        # ----------------------------------------------------
        # Validate raw input
        # ----------------------------------------------------

        validated_bmi = validate_raw_inputs(
            age=age,
            height_cm=height_cm,
            weight_kg=weight_kg,
            outpatient_cost=outpatient_cost,
            previous_inpatient_cost=(
                previous_inpatient_cost
            ),
            retired_allowance=(
                retired_allowance
            ),
        )

        # ----------------------------------------------------
        # Convert user labels to model codes
        # ----------------------------------------------------

        hospitalized_code = (
            YES_NO_MAPPING[
                hospitalized_label
            ]
        )

        employed_code = (
            YES_NO_MAPPING[
                employed_label
            ]
        )

        chronic_illness_code = (
            YES_NO_MAPPING[
                chronic_illness_label
            ]
        )

        health_code = (
            HEALTH_MAPPING[
                health_label
            ]
        )

        insurance_code = (
            INSURANCE_MAPPING[
                insurance_label
            ]
        )

        # ----------------------------------------------------
        # Generate trained-model features
        # ----------------------------------------------------

        model_input = create_model_features(
            age=age,
            height_cm=height_cm,
            weight_kg=weight_kg,
            hospitalized_code=(
                hospitalized_code
            ),
            outpatient_cost=(
                outpatient_cost
            ),
            previous_inpatient_cost=(
                previous_inpatient_cost
            ),
            qi202_value=(
                retired_allowance
            ),
            employed_code=(
                employed_code
            ),
            health_code=(
                health_code
            ),
            chronic_illness_code=(
                chronic_illness_code
            ),
            insurance_code=(
                insurance_code
            ),
            required_features=artifact[
                "feature_names"
            ],
        )

        # ----------------------------------------------------
        # Current medical-cost prediction
        # ----------------------------------------------------

        prediction_result = predict_cost(
            artifact=artifact,
            model_input=model_input,
        )

        predicted_log_cost = (
            prediction_result[
                "predicted_log_cost"
            ]
        )

        predicted_medical_cost = (
            prediction_result[
                "predicted_original_cost"
            ]
        )

        lgb_log_prediction = (
            prediction_result[
                "lgb_log_prediction"
            ]
        )

        xgb_log_prediction = (
            prediction_result[
                "xgb_log_prediction"
            ]
        )

        ordered_model_input = (
            prediction_result[
                "model_input"
            ]
        )

        selected_currency = (
            CURRENCY_OPTIONS[
                selected_currency_label
            ]
        )

        selected_currency_code = (
            selected_currency["code"]
        )

        selected_currency_symbol = (
            selected_currency["symbol"]
        )

        currency_result = None
        currency_error_message = None

        try:
            currency_result = convert_cny_amount(
                amount_cny=(
                    predicted_medical_cost
                ),
                target_currency=(
                    selected_currency_code
                ),
                api_key=(
                    EXCHANGE_RATE_API_KEY
                ),
            )

        except Exception as currency_error:
            currency_error_message = str(
                currency_error
            )

        converted_cost_for_chat = float(
            predicted_medical_cost
        )

        exchange_rate_for_chat = None

        if currency_result is not None:
            converted_cost_for_chat = float(
                currency_result[
                    "converted_amount"
                ]
            )
            exchange_rate_for_chat = float(
                currency_result["rate"]
            )

        # Store limited, prediction-aware context for Gemini.
        # Raw monetary input fields are intentionally excluded.
        st.session_state.latest_prediction_context = {
            "predicted_cost_cny": float(
                predicted_medical_cost
            ),
            "predicted_log_cost": float(
                predicted_log_cost
            ),
            "display_currency": (
                selected_currency_code
            ),
            "display_currency_symbol": (
                selected_currency_symbol
            ),
            "converted_cost": (
                converted_cost_for_chat
            ),
            "exchange_rate": (
                exchange_rate_for_chat
            ),
            "bmi": float(
                validated_bmi
            ),
            "age": int(
                age
            ),
            "hospitalized": hospitalized_label,
            "chronic_illness": chronic_illness_label,
            "health_status": health_label,
            "employment_status": employed_label,
            "insurance_category": insurance_label,
            "top_factors": [],
        }

        st.success(
            "Prediction completed successfully."
        )

        st.metric(
            label=(
                "Estimated inpatient medical cost "
                "in Chinese yuan"
            ),
            value=(
                f"¥{predicted_medical_cost:,.2f} CNY"
            ),
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
                f"1 CNY = "
                f"{currency_result['rate']:.6f} "
                f"{selected_currency_code}"
            )

            if currency_result["last_updated"]:
                st.caption(
                    "Exchange-rate update time: "
                    f"{currency_result['last_updated']}"
                )

            st.warning(
                "The converted value is approximate. Banks, "
                "cards, and payment providers may use "
                "different rates or fees."
            )

        elif currency_error_message:
            st.warning(
                "The model prediction succeeded, but "
                "currency conversion was unavailable: "
                f"{currency_error_message}"
            )

        st.caption(
            "The model's original output remains in Chinese "
            "yuan because the target variable was trained in "
            "that currency."
        )

        # ----------------------------------------------------
        # Prediction summary
        # ----------------------------------------------------

        summary_col1, summary_col2 = (
            st.columns(
                2
            )
        )

        with summary_col1:
            st.metric(
                label="Calculated BMI",
                value=f"{validated_bmi:.2f}",
            )

        with summary_col2:
            st.metric(
                label="Log-scale prediction",
                value=f"{predicted_log_cost:.4f}",
            )

        # ----------------------------------------------------
        # Next-year scenario projection
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "Next-year medical-cost projection"
        )

        next_year_result = (
            predict_next_year_cost(
                artifact=artifact,
                current_age=age,
                height_cm=height_cm,
                weight_kg=weight_kg,
                hospitalized_code=(
                    hospitalized_code
                ),
                outpatient_cost=(
                    outpatient_cost
                ),
                current_prediction=(
                    predicted_medical_cost
                ),
                retired_allowance=(
                    retired_allowance
                ),
                employed_code=(
                    employed_code
                ),
                health_code=(
                    health_code
                ),
                chronic_illness_code=(
                    chronic_illness_code
                ),
                insurance_code=(
                    insurance_code
                ),
            )
        )

        next_year_cost = float(
            next_year_result[
                "Predicted cost"
            ]
        )

        cost_change = (
            next_year_cost
            - predicted_medical_cost
        )

        if predicted_medical_cost > 0:
            percentage_change = (
                cost_change
                / predicted_medical_cost
                * 100
            )

        else:
            percentage_change = 0.0

        st.metric(
            label=(
                "Estimated inpatient medical cost next year"
            ),
            value=(
                f"¥{next_year_cost:,.2f} CNY"
            ),
            delta=(
                f"¥{cost_change:,.2f} CNY"
            ),
        )

        if (
            currency_result is not None
            and selected_currency_code != "CNY"
        ):
            next_year_converted = (
                next_year_cost
                * currency_result["rate"]
            )

            converted_change = (
                next_year_converted
                - currency_result[
                    "converted_amount"
                ]
            )

            st.metric(
                label=(
                    "Approximate next-year converted cost"
                ),
                value=(
                    f"{selected_currency_symbol}"
                    f"{next_year_converted:,.2f} "
                    f"{selected_currency_code}"
                ),
                delta=(
                    f"{selected_currency_symbol}"
                    f"{converted_change:,.2f}"
                ),
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
            comparison_df.set_index(
                "Period"
            ),
            use_container_width=True,
        )

        if percentage_change >= 0:
            st.write(
                "The projected cost is "
                f"**{percentage_change:.2f}% higher** "
                "than the current prediction."
            )

        else:
            st.write(
                "The projected cost is "
                f"**{abs(percentage_change):.2f}% lower** "
                "than the current prediction."
            )

        st.warning(
            "The next-year value is a scenario projection, "
            "not a validated time-series forecast. It assumes "
            "that height, weight, outpatient cost, "
            "hospitalization status, employment, health "
            "condition, chronic illness status, and insurance "
            "category remain unchanged."
        )

        # ----------------------------------------------------
        # SHAP contributors
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "Top factors influencing the current prediction"
        )

        try:
            top_contributors = (
                calculate_top_contributors(
                    artifact=artifact,
                    model_input=(
                        ordered_model_input
                    ),
                    top_n=3,
                )
            )

            top_factor_context = []

            for _, contributor_row in (
                top_contributors.iterrows()
            ):
                top_factor_context.append(
                    {
                        "feature": contributor_row[
                            "Display feature"
                        ],
                        "effect": contributor_row[
                            "Effect"
                        ],
                        "contribution": float(
                            contributor_row[
                                "SHAP contribution"
                            ]
                        ),
                    }
                )

            if (
                st.session_state.latest_prediction_context
                is not None
            ):
                st.session_state.latest_prediction_context[
                    "top_factors"
                ] = top_factor_context

            contribution_chart = (
                top_contributors
                .set_index(
                    "Display feature"
                )[
                    [
                        "Absolute contribution"
                    ]
                ]
            )

            st.bar_chart(
                contribution_chart,
                use_container_width=True,
            )

            for rank, row in (
                top_contributors
                .iterrows()
            ):
                display_name = row[
                    "Display feature"
                ]

                contribution = float(
                    row[
                        "SHAP contribution"
                    ]
                )

                displayed_rank = (
                    rank + 1
                )

                if contribution >= 0:
                    icon = "⬆️"

                    direction = (
                        "increased the predicted cost"
                    )

                else:
                    icon = "⬇️"

                    direction = (
                        "reduced the predicted cost"
                    )

                st.write(
                    f"{displayed_rank}. "
                    f"{icon} "
                    f"**{display_name}** "
                    f"{direction}."
                )

            with st.expander(
                "View detailed feature contributions"
            ):
                display_contributions = (
                    top_contributors[
                        [
                            "Display feature",
                            "SHAP contribution",
                            "Absolute contribution",
                            "Effect",
                        ]
                    ]
                    .rename(
                        columns={
                            "Display feature": (
                                "Feature"
                            ),
                        }
                    )
                )

                st.dataframe(
                    display_contributions,
                    use_container_width=True,
                    hide_index=True,
                )

                st.caption(
                    "SHAP values are measured on the model's "
                    "log-cost scale. The absolute value "
                    "represents the strength of influence, "
                    "while the sign indicates whether the "
                    "feature increased or decreased the "
                    "prediction."
                )

        except Exception as shap_error:
            st.warning(
                "The medical-cost prediction worked, but "
                "the feature explanation could not be "
                "generated: "
                f"{shap_error}"
            )

        # ----------------------------------------------------
        # UI/model consistency information
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "Prediction verification"
        )

        blend_weight = float(
            artifact[
                "blend_weight"
            ]
        )

        manually_calculated_blend = (
            blend_weight
            * lgb_log_prediction
            + (
                1 - blend_weight
            )
            * xgb_log_prediction
        )

        blend_difference = abs(
            predicted_log_cost
            - manually_calculated_blend
        )

        retransformed_cost = max(
            0.0,
            float(
                np.expm1(
                    predicted_log_cost
                )
            ),
        )

        retransformation_difference = abs(
            predicted_medical_cost
            - retransformed_cost
        )

        blend_passed = np.isclose(
            predicted_log_cost,
            manually_calculated_blend,
            rtol=1e-12,
            atol=1e-12,
        )

        retransformation_passed = (
            np.isclose(
                predicted_medical_cost,
                retransformed_cost,
                rtol=1e-12,
                atol=1e-12,
            )
        )

        verification_df = pd.DataFrame(
            {
                "Test": [
                    "Blending formula",
                    "Log-to-original conversion",
                ],

                "Expected result": [
                    manually_calculated_blend,
                    retransformed_cost,
                ],

                "Application result": [
                    predicted_log_cost,
                    predicted_medical_cost,
                ],

                "Absolute difference": [
                    blend_difference,
                    retransformation_difference,
                ],

                "Status": [
                    (
                        "Pass"
                        if blend_passed
                        else "Fail"
                    ),

                    (
                        "Pass"
                        if retransformation_passed
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

        if (
            blend_passed
            and retransformation_passed
        ):
            st.success(
                "The prediction calculation passed the "
                "built-in consistency checks."
            )

        else:
            st.error(
                "The prediction did not pass all consistency "
                "checks. Verify the blending or target "
                "retransformation logic."
            )

        # ----------------------------------------------------
        # Technical details
        # ----------------------------------------------------

        with st.expander(
            "View technical prediction details"
        ):

            st.write(
                "Model version:",
                MODEL_VERSION,
            )

            st.write(
                "Model path:",
                str(
                    MODEL_PATH
                ),
            )

            st.write(
                "Required model features:",
                list(
                    artifact[
                        "feature_names"
                    ]
                ),
            )

            st.write(
                "Number of model features:",
                len(
                    artifact[
                        "feature_names"
                    ]
                ),
            )

            st.write(
                "Blending weight:",
                blend_weight,
            )

            st.write(
                "LightGBM log prediction:",
                lgb_log_prediction,
            )

            st.write(
                "XGBoost log prediction:",
                xgb_log_prediction,
            )

            st.write(
                "Manually calculated blended prediction:",
                manually_calculated_blend,
            )

            st.write(
                "Final blended log prediction:",
                predicted_log_cost,
            )

            st.write(
                "Final original-scale prediction:",
                predicted_medical_cost,
            )

            st.write(
                "Generated model input:"
            )

            st.dataframe(
                ordered_model_input,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                label=(
                    "Download generated model input"
                ),
                data=(
                    ordered_model_input
                    .to_csv(
                        index=False
                    )
                ),
                file_name=(
                    "generated_model_input.csv"
                ),
                mime="text/csv",
                use_container_width=True,
            )

    except Exception as error:
        st.error(
            "Prediction failed: "
            f"{error}"
        )

# ============================================================
# GEMINI PREDICTION-AWARE CHATBOT
# This section is intentionally outside the prediction form
# and outside `if submitted`, so it is always visible.
# ============================================================

st.divider()

st.subheader(
    "💬 Medical Cost Prediction Assistant"
)

st.caption(
    "Ask about the estimated cost, SHAP factors, or how to "
    "interpret the model output. This assistant does not "
    "provide medical diagnosis or treatment advice."
)

if st.session_state.latest_prediction_context is None:
    st.info(
        "Generate a medical-cost prediction first for a "
        "personalised explanation. General model questions "
        "can still be asked."
    )
else:
    latest_cost = (
        st.session_state.latest_prediction_context[
            "predicted_cost_cny"
        ]
    )

    st.success(
        "Latest prediction available to the assistant: "
        f"¥{latest_cost:,.2f} CNY."
    )

if st.button(
    "Clear chat",
    key="clear_gemini_chat",
):
    st.session_state.chat_messages = [
        {
            "role": "assistant",
            "content": (
                "Chat history cleared. I can explain the "
                "medical-cost prediction and model factors."
            ),
        }
    ]
    st.rerun()

for message in st.session_state.chat_messages:
    with st.chat_message(
        message["role"]
    ):
        st.markdown(
            message["content"]
        )

user_message = st.chat_input(
    "Ask why the cost is high or what the factors mean",
    key="medical_cost_chat_input",
)

if user_message:
    st.session_state.chat_messages.append(
        {
            "role": "user",
            "content": user_message,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_message)

    with st.chat_message("assistant"):
        if not gemini_available or gemini_client is None:
            assistant_response = (
                "The Gemini assistant is currently "
                "unavailable. Check the API key and the "
                "google-genai dependency."
            )
        else:
            try:
                prediction_context = (
                    st.session_state.latest_prediction_context
                )

                if prediction_context is None:
                    prediction_text = """
No prediction has been generated in this session.
Explain the model only in general terms. Do not claim to
know the user's cost or personal contributing factors.
"""
                else:
                    top_factors = prediction_context.get(
                        "top_factors",
                        [],
                    )

                    if top_factors:
                        factor_lines = []

                        for factor in top_factors:
                            factor_lines.append(
                                "- "
                                f"{factor['feature']}: "
                                f"{factor['effect']} "
                                f"(SHAP contribution "
                                f"{factor['contribution']:.4f})"
                            )

                        factor_text = "\n".join(
                            factor_lines
                        )
                    else:
                        factor_text = (
                            "SHAP contributors are unavailable "
                            "for this prediction."
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
{prediction_context['display_currency_symbol']}
{prediction_context['converted_cost']:,.2f}
{prediction_context['display_currency']}

Exchange rate from CNY:
{prediction_context.get('exchange_rate')}

Age:
{prediction_context['age']}

Calculated BMI:
{prediction_context['bmi']:.2f}

Hospitalized during the survey period:
{prediction_context['hospitalized']}

Chronic illness:
{prediction_context['chronic_illness']}

Self-rated health:
{prediction_context['health_status']}

Employment status:
{prediction_context['employment_status']}

Insurance category:
{prediction_context['insurance_category']}

Top model contributors:
{factor_text}
"""

                recent_messages = (
                    st.session_state.chat_messages[-8:]
                )

                conversation_text = "\n".join(
                    (
                        f"{message['role']}: "
                        f"{message['content']}"
                    )
                    for message in recent_messages
                )

                gemini_prompt = f"""
You are an educational assistant inside a machine-learning
application that estimates inpatient medical costs.

The prediction is produced by a blended LightGBM and XGBoost
regression model trained using historical CFPS survey data.

Rules:
1. Explain the predicted cost in simple language.
2. Explain SHAP factors and whether each factor increased or
   decreased the model prediction.
3. State that SHAP describes model behaviour and does not
   prove medical causation.
4. State that the prediction is an estimate, not a guaranteed
   medical bill.
5. Do not diagnose diseases.
6. Do not recommend medication, treatment, insurance plans,
   or financial products.
7. Do not invent values, model metrics, or patient details.
8. Explain that converted currency values are approximate
   and may differ from bank or payment-provider rates.
9. Encourage consultation with qualified healthcare or
   financial professionals for personal decisions.
10. Keep the response concise and directly relevant.

{prediction_text}

Recent conversation:
{conversation_text}

Latest user message:
{user_message}
"""

                with st.spinner(
                    "Generating explanation..."
                ):
                    gemini_response = (
                        gemini_client.models.generate_content(
                            model="gemini-2.5-flash",
                            contents=gemini_prompt,
                        )
                    )

                response_text = getattr(
                    gemini_response,
                    "text",
                    None,
                )

                if not response_text:
                    assistant_response = (
                        "Gemini returned an empty response. "
                        "Please try again."
                    )
                else:
                    assistant_response = (
                        response_text.strip()
                    )

            except Exception as error:
                assistant_response = (
                    "The assistant could not generate a "
                    f"response: {error}"
                )

        st.markdown(assistant_response)

    st.session_state.chat_messages.append(
        {
            "role": "assistant",
            "content": assistant_response,
        }
    )
