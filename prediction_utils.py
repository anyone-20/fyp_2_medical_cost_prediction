# prediction_utils.py

from typing import Any

import numpy as np
import pandas as pd


def validate_input_data(
    input_data: dict,
    feature_names: list[str]
) -> None:
    """
    Check whether the user input contains exactly the features
    expected by the trained model.
    """

    missing_features = [
        feature
        for feature in feature_names
        if feature not in input_data
    ]

    unexpected_features = [
        feature
        for feature in input_data
        if feature not in feature_names
    ]

    if missing_features:
        raise ValueError(
            f"Missing features: {missing_features}"
        )

    if unexpected_features:
        raise ValueError(
            f"Unexpected features: {unexpected_features}"
        )


def prepare_gradient_boosting_input(
    input_data: dict,
    feature_names: list[str]
) -> pd.DataFrame:
    """
    Convert one patient's dictionary into a DataFrame using
    the exact feature order used during model training.
    """

    validate_input_data(
        input_data=input_data,
        feature_names=feature_names
    )

    input_df = pd.DataFrame(
        [input_data],
        columns=feature_names
    )

    # Force all model inputs to numeric values.
    for column in input_df.columns:
        input_df[column] = pd.to_numeric(
            input_df[column],
            errors="raise"
        )

    if input_df.isnull().any().any():
        missing_columns = input_df.columns[
            input_df.isnull().any()
        ].tolist()

        raise ValueError(
            f"Input contains missing values: "
            f"{missing_columns}"
        )

    if not np.isfinite(
        input_df.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Input contains infinite or invalid values."
        )

    return input_df


def predict_gradient_boosting_cost(
    model: Any,
    input_data: dict,
    feature_names: list[str],
    target_is_log1p: bool = True
) -> dict:
    """
    Generate Gradient Boosting predictions.

    Parameters
    ----------
    model:
        Fitted Gradient Boosting model or complete pipeline.

    input_data:
        Dictionary containing one patient's model predictors.

    feature_names:
        Exact original feature names and order used during
        training.

    target_is_log1p:
        True when the model was trained on log1p(medical cost).

    Returns
    -------
    dict:
        Raw model prediction and original medical-cost
        prediction.
    """

    input_df = prepare_gradient_boosting_input(
        input_data=input_data,
        feature_names=feature_names
    )

    raw_prediction = float(
        model.predict(input_df)[0]
    )

    if target_is_log1p:
        original_prediction = float(
            np.expm1(raw_prediction)
        )
    else:
        original_prediction = raw_prediction

    original_prediction = max(
        original_prediction,
        0.0
    )

    return {
        "raw_prediction": raw_prediction,
        "original_prediction": original_prediction,
        "input_dataframe": input_df
    }
