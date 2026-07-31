from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
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

MODEL_PATH = Path(__file__).parent / "medical_cost_model.pkl"


# =========================================================
# 3. HUMAN-READABLE FEATURE LABELS
# =========================================================

FEATURE_LABELS = {
    "log_qc7b": "Outpatient medical cost",
    "qc401 age": "Hospitalization and age",
    "qc401 bmi": "Hospitalization and BMI",
    "qc401": "Hospitalization status",
    "log_qc7b bmi": "Outpatient cost and BMI",
    "log_qc7b age": "Outpatient cost and age",
    "qgb1": "Employment status",
    "log_past_qc701": "Previous inpatient medical cost",
    "qp201": "Self-rated health",
    "qp401": "Chronic illness status",
    "qp605_s_1": "Medical insurance category",
    "age": "Age",
    "bmi": "BMI",
    "log_qc7b qc401": (
        "Outpatient cost and hospitalization"
    ),
    "bmi age": "BMI and age",
    "qp102": "Body weight",
}


# =========================================================
# 4. LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    """
    Load and validate the saved machine-learning artifact.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at: {MODEL_PATH}"
        )

    loaded_artifact = joblib.load(MODEL_PATH)

    required_keys = [
        "lgb_model",
        "xgb_model",
        "blend_weight",
        "feature_names",
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in loaded_artifact
    ]

    if missing_keys:
        raise KeyError(
            "The model file is missing these required items: "
            f"{missing_keys}"
        )

    return loaded_artifact


# =========================================================
# 5. COST PREDICTION FUNCTION
# =========================================================

def predict_cost(
    artifact,
    model_input,
):
    """
    Generate the blended LightGBM and XGBoost prediction.

    Returns:
        predicted_log_cost
        predicted_original_cost
        LightGBM log prediction
        XGBoost log prediction
    """

    required_features = list(
        artifact["feature_names"]
    )

    model_input = model_input[
        required_features
    ]

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

    predicted_original_cost = np.expm1(
        predicted_log_cost
    )

    predicted_original_cost = max(
        0.0,
        float(predicted_original_cost),
    )

    return (
        float(predicted_log_cost),
        predicted_original_cost,
        float(lgb_prediction[0]),
        float(xgb_prediction[0]),
    )


# =========================================================
# 6. NEXT-YEAR COST PROJECTION
# =========================================================

def predict_next_year_cost(
    *,
    artifact,
    current_age,
    height_cm,
    weight_kg,
    hospitalized_code,
    outpatient_cost,
    current_prediction,
    employed_code,
    health_code,
    chronic_illness_code,
    insurance_code,
):
    """
    Estimate next year's inpatient medical cost.

    Assumptions:
    - Age increases by one year.
    - Height and weight remain unchanged.
    - Other entered conditions remain unchanged.
    - The current predicted cost becomes next year's
      previous inpatient cost.
    """

    next_year_age = current_age + 1

    next_year_input = create_model_features(
        age=next_year_age,
        height_cm=height_cm,
        weight_kg=weight_kg,
        hospitalized_code=hospitalized_code,
        outpatient_cost=outpatient_cost,
        previous_inpatient_cost=current_prediction,
        employed_code=employed_code,
        health_code=health_code,
        chronic_illness_code=chronic_illness_code,
        insurance_code=insurance_code,
        required_features=artifact[
            "feature_names"
        ],
    )

    (
        next_year_log_cost,
        next_year_cost,
        _,
        _,
    ) = predict_cost(
        artifact,
        next_year_input,
    )

    return {
        "Age": next_year_age,
        "Log cost": next_year_log_cost,
        "Predicted cost": next_year_cost,
    }


# =========================================================
# 7. SHAP HELPER FUNCTIONS
# =========================================================

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

    if hasattr(shap_result, "values"):
        values = shap_result.values
    else:
        values = shap_result

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


@st.cache_resource
def create_shap_explainers(
    _lgb_model,
    _xgb_model,
):
    """
    Create and cache SHAP explainers.

    The leading underscores tell Streamlit not to hash
    the model objects.
    """

    lgb_explainer = shap.TreeExplainer(
        _lgb_model
    )

    xgb_explainer = shap.TreeExplainer(
        _xgb_model
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
    Calculate weighted SHAP contributions for the
    blended LightGBM and XGBoost prediction.
    """

    required_features = list(
        artifact["feature_names"]
    )

    model_input = model_input[
        required_features
    ]

    blend_weight = float(
        artifact["blend_weight"]
    )

    (
        lgb_explainer,
        xgb_explainer,
    ) = create_shap_explainers(
        artifact["lgb_model"],
        artifact["xgb_model"],
    )

    lgb_values = extract_shap_values(
        lgb_explainer,
        model_input,
    )

    xgb_values = extract_shap_values(
        xgb_explainer,
        model_input,
    )

    if len(lgb_values) != len(required_features):
        raise ValueError(
            "The number of LightGBM SHAP values does not "
            "match the number of model features."
        )

    if len(xgb_values) != len(required_features):
        raise ValueError(
            "The number of XGBoost SHAP values does not "
            "match the number of model features."
        )

    blended_values = (
        blend_weight * lgb_values
        + (1 - blend_weight) * xgb_values
    )

    contribution_df = pd.DataFrame(
        {
            "Feature": required_features,
            "SHAP contribution": blended_values,
        }
    )

    contribution_df[
        "Absolute contribution"
    ] = contribution_df[
        "SHAP contribution"
    ].abs()

    contribution_df["Effect"] = np.where(
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
    ].map(FEATURE_LABELS)

    contribution_df[
        "Display feature"
    ] = contribution_df[
        "Display feature"
    ].fillna(
        contribution_df["Feature"]
    )

    contribution_df = contribution_df.sort_values(
        by="Absolute contribution",
        ascending=False,
    ).head(top_n)

    return contribution_df


# =========================================================
# 8. LOAD THE TRAINED MODEL
# =========================================================

try:
    artifact = load_model()

except Exception as error:
    st.error(
        f"Unable to load the trained model: {error}"
    )
    st.stop()


# =========================================================
# 9. APPLICATION HEADER
# =========================================================

st.title("🏥 Medical Cost Prediction")

st.write(
    "Enter the individual's personal, healthcare, "
    "employment, and insurance information below."
)

st.info(
    "This application estimates inpatient medical cost "
    "using a trained machine-learning model. The result "
    "should not be treated as medical or financial advice."
)


# =========================================================
# 10. USER INPUT FORM
# =========================================================

with st.form("medical_cost_form"):

    st.subheader("Personal information")

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=119,
        value=40,
        step=1,
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
            "Replace these generic labels with the official "
            "insurance descriptions from your questionnaire."
        ),
    )

    submitted = st.form_submit_button(
        "Predict medical cost",
        use_container_width=True,
    )


# =========================================================
# 11. PROCESS USER INPUT
# =========================================================

if submitted:

    try:
        # -------------------------------------------------
        # Convert labels into numeric model codes
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
        # Generate features expected by the trained model
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
            required_features=artifact[
                "feature_names"
            ],
        )

        # -------------------------------------------------
        # Current medical-cost prediction
        # -------------------------------------------------

        (
            predicted_log_cost,
            predicted_medical_cost,
            lgb_log_prediction,
            xgb_log_prediction,
        ) = predict_cost(
            artifact,
            model_input,
        )

        st.success(
            "Prediction completed successfully."
        )

        st.metric(
            label="Estimated current inpatient medical cost",
            value=f"{predicted_medical_cost:,.2f}",
        )

        st.caption(
            "The amount uses the same currency unit as the "
            "target medical-cost variable in the dataset."
        )

        # -------------------------------------------------
        # Next-year cost prediction
        # -------------------------------------------------

        st.divider()

        st.subheader(
            "Next-year medical-cost projection"
        )

        next_year_result = predict_next_year_cost(
            artifact=artifact,
            current_age=age,
            height_cm=height_cm,
            weight_kg=weight_kg,
            hospitalized_code=hospitalized_code,
            outpatient_cost=outpatient_cost,
            current_prediction=predicted_medical_cost,
            employed_code=employed_code,
            health_code=health_code,
            chronic_illness_code=chronic_illness_code,
            insurance_code=insurance_code,
        )

        next_year_cost = float(
            next_year_result["Predicted cost"]
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
            label="Estimated inpatient medical cost next year",
            value=f"{next_year_cost:,.2f}",
            delta=f"{cost_change:,.2f}",
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
            "that height, weight, outpatient cost, hospitalization "
            "status, employment, health condition, chronic illness "
            "status, and insurance category remain unchanged."
        )

        # -------------------------------------------------
        # Top three SHAP contributors
        # -------------------------------------------------

        st.divider()

        st.subheader(
            "Top factors influencing the current prediction"
        )

        try:
            top_contributors = (
                calculate_top_contributors(
                    artifact=artifact,
                    model_input=model_input,
                    top_n=3,
                )
            )

            contribution_chart = (
                top_contributors
                .set_index(
                    "Display feature"
                )[
                    ["Absolute contribution"]
                ]
            )

            st.bar_chart(
                contribution_chart,
                use_container_width=True,
            )

            for rank, (_, row) in enumerate(
                top_contributors.iterrows(),
                start=1,
            ):
                display_name = row[
                    "Display feature"
                ]

                contribution = float(
                    row["SHAP contribution"]
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
                    f"{rank}. {icon} **{display_name}** "
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
                            "Display feature": "Feature",
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
                    "log-cost scale. The absolute value represents "
                    "the strength of influence, while the sign "
                    "indicates whether the feature increased or "
                    "decreased the prediction."
                )

        except Exception as shap_error:
            st.warning(
                "The medical-cost prediction worked, but the "
                "feature explanation could not be generated: "
                f"{shap_error}"
            )

        # -------------------------------------------------
        # Technical details
        # -------------------------------------------------

        with st.expander(
            "View technical prediction details"
        ):

            st.write(
                "Required model features:",
                list(
                    artifact["feature_names"]
                ),
            )

            st.write(
                "Blending weight:",
                float(
                    artifact["blend_weight"]
                ),
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
                model_input,
                use_container_width=True,
                hide_index=True,
            )

    except Exception as error:
        st.error(
            f"Prediction failed: {error}"
        )
