import streamlit as st
import pandas as pd
import joblib

# 1. Load your saved artifacts dictionary
artifacts = joblib.load("gradient_boosting_pipeline.pkl")

preprocessor = artifacts['preprocessor']
lgb_model = artifacts['lgb_model']
xgb_model = artifacts['xgb_model']
weight = artifacts['weight']
feature_names = artifacts['feature_names']

st.title("Medical Cost Prediction App")

# 2. Create Streamlit input widgets for your features
# (Make sure to match the exact feature names your model was trained on)
age = st.number_input("Age", min_value=0, max_value=120, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
length_of_stay = st.number_input("Length of Stay (days)", min_value=1, max_value=365, value=3)

# Add any other categorical or numerical inputs your model requires
# e.g., sex = st.selectbox("Sex", ["male", "female"])
# e.g., smoker = st.selectbox("Smoker", ["yes", "no"])

# 3. Predict button
if st.button("Calculate Predicted Cost"):
    # Pack the user inputs into a DataFrame matching your raw columns
    raw_input = pd.DataFrame([{
        'age': age,
        'bmi': bmi,
        'length_of_stay': length_of_stay,
        # 'sex': sex,
        # 'smoker': smoker,
        # (Include every original raw feature your preprocessor expects!)
    }])

    # Transform through your preprocessor
    X_trans = preprocessor.transform(raw_input)

    # Align to pruned features if necessary
    if feature_names:
        if hasattr(X_trans, 'loc'):
            X_trans = X_trans[feature_names]
        else:
            X_trans = pd.DataFrame(X_trans, columns=preprocessor.get_feature_names_out())[feature_names]

    # Run predictions and blend
    pred_lgb = lgb_model.predict(X_trans)
    pred_xgb = xgb_model.predict(X_trans)
    final_pred = (weight * pred_lgb) + ((1 - weight) * pred_xgb)

    # Display result
    st.success(f"Predicted Medical Cost: ${final_pred[0]:,.2f}")
