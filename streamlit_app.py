import streamlit as st
import joblib
import pandas as pd


# Load trained model
model = joblib.load("gradient_boosting.pkl")


st.title("🏥 Inpatient Cost Prediction AI")

st.write(
    "Machine Learning model to predict inpatient healthcare cost"
)


age = st.number_input(
    "Age",
    min_value=0,
    max_value=100
)


bmi = st.number_input(
    "BMI",
    min_value=0.0
)


gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)


insurance = st.selectbox(
    "Insurance",
    ["Yes", "No"]
)



if st.button("Predict Cost"):

    input_data = pd.DataFrame({

        "age":[age],
        "bmi":[bmi],
        "gender":[gender],
        "has_insurance":[insurance]

    })


    prediction = model.predict(
        input_data
    )


    st.success(
        f"Predicted inpatient cost: RM {prediction[0]:,.2f}"
    )
