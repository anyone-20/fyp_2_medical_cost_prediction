import joblib
import pandas as pd
import streamlit as st


# Load custom class definition in streamlit if using class object
class BlendedCostModel:

    def __init__(self, preprocessor, lgb_model, xgb_model, weight):
        self.preprocessor = preprocessor
        self.lgb_model = lgb_model
        self.xgb_model = xgb_model
        self.weight = weight

    def predict(self, X):
        if self.preprocessor is not None:
            X_trans = self.preprocessor.transform(X)
        else:
            X_trans = X
        lgb_preds = self.lgb_model.predict(X_trans)
        xgb_preds = self.xgb_model.predict(X_trans)
        return (self.weight * lgb_preds) + ((1 - self.weight) * xgb_preds)


# Load pipeline
pipeline = joblib.load("gradient_boosting_pipeline.pkl")

st.title("🏥 Inpatient Cost Prediction AI")

age = st.number_input("Age", min_value=0, max_value=100)
bmi = st.number_input("BMI", min_value=0.0)
gender = st.selectbox("Gender", ["Male", "Female"])
insurance = st.selectbox("Insurance", ["Yes", "No"])

if st.button("Predict Cost"):
    input_data = pd.DataFrame(
        {
            "age": [age],
            "bmi": [bmi],
            "gender": [gender],
            "has_insurance": [insurance],
        }
    )

    # Pipeline handles transformations and ensemble prediction automatically
    prediction = pipeline.predict(input_data)
    st.success(f"Predicted inpatient cost: RM {prediction[0]:,.2f}")
