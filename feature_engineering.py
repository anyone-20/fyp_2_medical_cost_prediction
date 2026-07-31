import numpy as np
import pandas as pd


def create_model_features(
    *,
    age,
    height_cm,
    weight_kg,
    hospitalized_code,
    outpatient_cost,
    previous_inpatient_cost,
    employed_code,
    health_code,
    chronic_illness_code,
    insurance_code,
    required_features,
):
    """
    Convert Streamlit user inputs into the exact processed
    columns required by the saved machine-learning model.
    """

    if not 0 < age <= 120:
        raise ValueError("Age must be between 1 and 120.")

    if not 50 <= height_cm <= 250:
        raise ValueError(
            "Height must be between 50 and 250 cm."
        )

    if not 10 <= weight_kg <= 300:
        raise ValueError(
            "Weight must be between 10 and 300 kg."
        )

    if outpatient_cost < 0:
        raise ValueError(
            "Outpatient cost cannot be negative."
        )

    if previous_inpatient_cost < 0:
        raise ValueError(
            "Previous inpatient cost cannot be negative."
        )

    # Calculate BMI
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)

    # Apply the same log1p transformations used during training
    log_qc7b = np.log1p(outpatient_cost)
    log_past_qc701 = np.log1p(
        previous_inpatient_cost
    )

    # Create all possible model features
    all_features = pd.DataFrame(
        {
            "log_qc7b": [log_qc7b],
            "qc401 age": [
                hospitalized_code * age
            ],
            "qc401 bmi": [
                hospitalized_code * bmi
            ],
            "qc401": [
                hospitalized_code
            ],
            "log_qc7b bmi": [
                log_qc7b * bmi
            ],
            "log_qc7b age": [
                log_qc7b * age
            ],
            "qgb1": [
                employed_code
            ],
            "log_past_qc701": [
                log_past_qc701
            ],
            "qp201": [
                health_code
            ],
            "qp401": [
                chronic_illness_code
            ],
            "qp605_s_1": [
                insurance_code
            ],
            "age": [
                age
            ],
            "bmi": [
                bmi
            ],
            "log_qc7b qc401": [
                log_qc7b * hospitalized_code
            ],
            "bmi age": [
                bmi * age
            ],
            "qp102": [
                weight_kg / 0.5
            ],
        }
    )

    missing_features = [
        feature
        for feature in required_features
        if feature not in all_features.columns
    ]

    if missing_features:
        raise ValueError(
            "The application cannot create these required "
            f"features: {missing_features}"
        )

    return all_features[required_features]
