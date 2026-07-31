from pathlib import Path

import joblib
import numpy as np
import streamlit as st

from feature_engineering import create_model_features


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Medical Cost Prediction",
    page_icon="🏥",
    layout="centered",
)


# =========================================================
# 2. MODEL FILE PATH
# =========================================================

MODEL_PATH = "C:\Users\leeji\FYP2\CFPS\ML\2020\Gradient Boosting\medical_cost_model.pkl"


# =========================================================
# 3. LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    """
    Load the saved machine-learning model artifact.

    The PKL file should contain:
    - lgb_model
    - xgb_model
    - blend_weight
    - feature_names
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at: {MODEL_PATH}"
        )

    artifact = joblib.load(MODEL_PATH)

    required_keys = [
        "lgb_model",
        "xgb_model",
        "blend_weight",
        "feature_names",
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in artifact
    ]

    if missing_keys:
        raise KeyError(
            f"The PKL file is missing these items: {missing_keys}"
        )

    return artifact


try:
    artifact = load_model()

except Exception as error:
    st.error(
        f"Unable to load the trained model: {error}"
    )
    st.stop()


# =========================================================
# 4. APPLICATION TITLE
# =========================================================

st.title("🏥 Medical Cost Prediction")

st.write(
    "Enter the individual's personal, healthcare, "
    "employment, and insurance information below."
)

st.info(
    "The model estimates inpatient medical cost based on "
    "the information entered. The result is a machine-learning "
    "estimate and should not be treated as medical advice."
)


# =========================================================
# 5. STREAMLIT INPUT FORM
# =========================================================

with st.form("medical_cost_form"):

    st.subheader("Personal information")

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=40,
        step=1,
        help="Enter the individual's age in years.",
    )

    height_cm = st.number_input(
        "Height (cm)",
        min_value=50.0,
        max_value=250.0,
        value=165.0,
        step=0.1,
        help="Enter height in centimetres.",
    )

    weight_kg = st.number_input(
        "Weight (kg)",
        min_value=10.0,
        max_value=300.0,
        value=60.0,
        step=0.1,
        help="Enter weight in kilograms.",
    )

    st.subheader("Healthcare information")

    hospitalized_label = st.selectbox(
        "Were you hospitalized during the survey period?",
        options=[
            "No",
            "Yes",
        ],
    )

    outpatient_cost = st.number_input(
        "Outpatient medical cost",
        min_value=0.0,
        value=0.0,
        step=100.0,
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
        help=(
            "Enter the inpatient medical cost from the "
            "previous survey year."
        ),
    )

    chronic_illness_label = st.selectbox(
        "Has the individual been diagnosed with a chronic illness?",
        options=[
            "No",
            "Yes",
        ],
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
    )

    st.subheader("Employment and insurance")

    employed_label = st.selectbox(
        "Is the individual currently employed?",
        options=[
            "No",
            "Yes",
        ],
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
        help=(
            "Replace the generic insurance labels with the "
            "actual descriptions from your questionnaire."
        ),
    )

    submitted = st.form_submit_button(
        "Predict medical cost",
        use_container_width=True,
    )


# =========================================================
# 6. PROCESS INPUT AFTER BUTTON CLICK
# =========================================================

if submitted:

    try:
        # -------------------------------------------------
        # Convert human-readable labels into model codes
        # -------------------------------------------------

        yes_no_mapping = {
            "No": 0,
            "Yes": 1,
        }

        health_mapping = {
            "Excellent": 1,
            "Very good": 2,
            "Good": 3,
            "Fair": 4,
            "Poor": 5,
        }

        insurance_mapping = {
            "Insurance category 1": 1,
            "Insurance category 2": 2,
            "Insurance category 3": 3,
            "Insurance category 4": 4,
            "Insurance category 5": 5,
            "Insurance category 6": 6,
            "No insurance": 78,
        }

        hospitalized_code = yes_no_mapping[
            hospitalized_label
        ]

        employed_code = yes_no_mapping[
            employed_label
        ]

        chronic_illness_code = yes_no_mapping[
            chronic_illness_label
        ]

        health_code = health_mapping[
            health_label
        ]

        insurance_code = insurance_mapping[
            insurance_label
        ]

        # -------------------------------------------------
        # Step 4: Create exact model input features
        # -------------------------------------------------

        model_input = create_model_features(
            age=age,
            height_cm=height_cm,
            weight_kg=weight_kg,
            hospitalized_code=hospitalized_code,
            outpatient_cost=outpatient_cost,
            previous_inpatient_cost=previous_inpatient_cost,
            employed_code=employed_code,
            health_code=health_code,
            chronic_illness_code=chronic_illness_code,
            insurance_code=insurance_code,
            required_features=artifact["feature_names"],
        )

        # -------------------------------------------------
        # Verify feature order
        # -------------------------------------------------

        required_features = artifact[
            "feature_names"
        ]

        model_input = model_input[
            required_features
        ]

        # -------------------------------------------------
        # Step 5: Make predictions
        # -------------------------------------------------

        lgb_prediction = artifact[
            "lgb_model"
        ].predict(model_input)

        xgb_prediction = artifact[
            "xgb_model"
        ].predict(model_input)

        blend_weight = float(
            artifact["blend_weight"]
        )

        predicted_log_cost = (
            blend_weight * lgb_prediction
            + (1 - blend_weight) * xgb_prediction
        )[0]

        # Convert log1p prediction back to original scale
        predicted_medical_cost = np.expm1(
            predicted_log_cost
        )

        # Prevent negative values caused by numerical precision
        predicted_medical_cost = max(
            0.0,
            float(predicted_medical_cost),
        )

        # -------------------------------------------------
        # Display prediction result
        # -------------------------------------------------

        st.success("Prediction completed successfully.")

        st.metric(
            label="Estimated inpatient medical cost",
            value=f"{predicted_medical_cost:,.2f}",
        )

        st.caption(
            "Use the same currency unit as the original "
            "medical-cost variable in your dataset."
        )

        # -------------------------------------------------
        # Technical details for testing
        # -------------------------------------------------

        with st.expander(
            "View technical prediction details"
        ):

            st.write(
                "Required model features:",
                required_features,
            )

            st.write(
                "Blending weight:",
                blend_weight,
            )

            st.write(
                "LightGBM log prediction:",
                float(lgb_prediction[0]),
            )

            st.write(
                "XGBoost log prediction:",
                float(xgb_prediction[0]),
            )

            st.write(
                "Final blended log prediction:",
                float(predicted_log_cost),
            )

            st.write(
                "Final original-scale prediction:",
                predicted_medical_cost,
            )

            st.write(
                "Generated model input:"
            )

            st.dataframe(
                model_input,
                use_container_width=True,
            )

    except Exception as error:
        st.error(
            f"Prediction failed: {error}"
        )
