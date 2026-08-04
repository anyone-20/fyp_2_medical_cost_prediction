# ============================================================
# STREAMLIT MEDICAL COST PREDICTION APPLICATION
# Blended LightGBM + XGBoost Model
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import shap
import streamlit as st

from feature_engineering import create_model_features

from external_services import (
    convert_currency,
    create_gemini_client,
    generate_chatbot_response,
)

from model_service import (
    load_model_artifact,
    predict_cost,
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
    return create_gemini_client(
        api_key
    )


if GEMINI_API_KEY:

    try:
        gemini_client = load_gemini_client(
            GEMINI_API_KEY
        )

    except Exception:
        gemini_client = None

else:
    gemini_client = None
    
# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Medical Cost Prediction",
    page_icon="🏥",
    layout="centered",
)


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

        st.success(
            "Prediction completed successfully."
        )

        st.metric(
            label=(
                "Estimated current inpatient medical cost"
            ),
            value=(
                f"{predicted_medical_cost:,.2f}"
            ),
        )

        st.caption(
            "The amount uses the same currency unit as the "
            "medical-cost target variable in the dataset."
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
                f"{next_year_cost:,.2f}"
            ),
            delta=(
                f"{cost_change:,.2f}"
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
