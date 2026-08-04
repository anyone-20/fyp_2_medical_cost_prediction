from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from feature_engineering import create_model_features


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "medical_cost_model.pkl"


def load_model_artifact(
    model_path: Path = MODEL_PATH,
):
    """
    Load and validate the saved LightGBM-XGBoost artifact.
    """

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at: {model_path}"
        )

    artifact = joblib.load(model_path)

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
            "The model artifact is missing: "
            f"{missing_keys}"
        )

    return artifact


def validate_model_input(
    artifact,
    model_input: pd.DataFrame,
):
    """
    Validate the generated model input before prediction.
    """

    if not isinstance(model_input, pd.DataFrame):
        raise TypeError(
            "model_input must be a pandas DataFrame."
        )

    required_features = list(
        artifact["feature_names"]
    )

    missing_features = [
        feature
        for feature in required_features
        if feature not in model_input.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing required model features: "
            f"{missing_features}"
        )

    ordered_input = model_input[
        required_features
    ].copy()

    if len(ordered_input) != 1:
        raise ValueError(
            "The prediction function expects exactly "
            "one patient record."
        )

    non_numeric_columns = ordered_input.select_dtypes(
        exclude=[np.number]
    ).columns.tolist()

    if non_numeric_columns:
        raise ValueError(
            "Non-numeric model features detected: "
            f"{non_numeric_columns}"
        )

    if ordered_input.isnull().any().any():
        null_columns = ordered_input.columns[
            ordered_input.isnull().any()
        ].tolist()

        raise ValueError(
            "Missing values detected in: "
            f"{null_columns}"
        )

    if not np.isfinite(
        ordered_input.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "The model input contains infinite "
            "or invalid numeric values."
        )

    return ordered_input


def predict_cost(
    artifact,
    model_input: pd.DataFrame,
):
    """
    Generate blended LightGBM and XGBoost predictions.
    """

    ordered_input = validate_model_input(
        artifact=artifact,
        model_input=model_input,
    )

    lgb_prediction = np.asarray(
        artifact["lgb_model"].predict(
            ordered_input
        )
    ).reshape(-1)

    xgb_prediction = np.asarray(
        artifact["xgb_model"].predict(
            ordered_input
        )
    ).reshape(-1)

    if len(lgb_prediction) != 1:
        raise ValueError(
            "Unexpected LightGBM prediction shape."
        )

    if len(xgb_prediction) != 1:
        raise ValueError(
            "Unexpected XGBoost prediction shape."
        )

    blend_weight = float(
        artifact["blend_weight"]
    )

    if not 0 <= blend_weight <= 1:
        raise ValueError(
            "blend_weight must be between 0 and 1."
        )

    lgb_log_prediction = float(
        lgb_prediction[0]
    )

    xgb_log_prediction = float(
        xgb_prediction[0]
    )

    predicted_log_cost = (
        blend_weight * lgb_log_prediction
        + (1 - blend_weight)
        * xgb_log_prediction
    )

    predicted_original_cost = max(
        0.0,
        float(
            np.expm1(
                predicted_log_cost
            )
        ),
    )

    return {
        "predicted_log_cost": (
            predicted_log_cost
        ),
        "predicted_original_cost": (
            predicted_original_cost
        ),
        "lgb_log_prediction": (
            lgb_log_prediction
        ),
        "xgb_log_prediction": (
            xgb_log_prediction
        ),
        "model_input": ordered_input,
    }


def build_patient_features(
    *,
    artifact,
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
):
    """
    Generate the exact feature structure expected by
    the trained models.
    """

    return create_model_features(
        age=age,
        height_cm=height_cm,
        weight_kg=weight_kg,
        hospitalized_code=hospitalized_code,
        outpatient_cost=outpatient_cost,
        previous_inpatient_cost=(
            previous_inpatient_cost
        ),
        employed_code=employed_code,
        health_code=health_code,
        chronic_illness_code=(
            chronic_illness_code
        ),
        insurance_code=insurance_code,
        required_features=artifact[
            "feature_names"
        ],
    )
