import streamlit as st
import joblib
import pandas as pd

# Load the dictionary artifacts
artifacts = joblib.load("gradient_boosting_pipeline.pkl")

preprocessor = artifacts['preprocessor']
lgb_model = artifacts['lgb_model']
xgb_model = artifacts['xgb_model']
weight = artifacts['weight']
feature_names = artifacts['feature_names']

# 1. Create raw input dataframe from Streamlit form inputs
raw_input = pd.DataFrame([{
    'age': age_input,
    'bmi': bmi_input,
    # ... add all original user input features
}])

# 2. Transform using preprocessor
X_trans = preprocessor.transform(raw_input)

# If preprocessor outputs a DataFrame or Array, align features:
if feature_names:
    if hasattr(X_trans, 'loc'):
        X_trans = X_trans[feature_names]
    else:
        # If output is numpy array
        X_trans = pd.DataFrame(X_trans, columns=preprocessor.get_feature_names_out())[feature_names]

# 3. Make prediction
pred_lgb = lgb_model.predict(X_trans)
pred_xgb = xgb_model.predict(X_trans)
final_pred = (weight * pred_lgb) + ((1 - weight) * pred_xgb)

st.success(f"Predicted Cost: ${final_pred[0]:,.2f}")
