import joblib
import pandas as pd
import streamlit as st

# Load trained model
model = joblib.load("gradient_boosting.pkl")

st.title("🏥 Inpatient Cost Prediction AI")
st.write("Machine Learning model to predict inpatient healthcare cost")

age = st.number_input("Age", min_value=0, max_value=100)
bmi = st.number_input("BMI", min_value=0.0)
gender = st.selectbox("Gender", ["Male", "Female"])
insurance = st.selectbox("Insurance", ["Yes", "No"])

if st.button("Predict Cost"):
    # 1. Collect inputs into a raw DataFrame
    input_data = pd.DataFrame(
        {
            "age": [age],
            "bmi": [bmi],
            "gender": [gender],
            "has_insurance": [insurance],
        }
    )

    # 2. Extract the exact feature names your model was trained on
    expected_features = model.feature_names_in_

    # 3. Align columns to match training features (fills missing columns with 0)
    input_data_aligned = input_data.reindex(
        columns=expected_features, fill_value=0
    )

    # 4. Predict using the aligned DataFrame
    prediction = model.predict(input_data_aligned)

    st.success(f"Predicted inpatient cost: RM {prediction[0]:,.2f}")
