import streamlit as st


with st.form("medical_cost_form"):

    st.subheader("Personal information")

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=40,
        step=1,
    )

    height_cm = st.number_input(
        "Height (cm)",
        min_value=50.0,
        max_value=250.0,
        value=165.0,
        step=0.1,
    )

    weight_kg = st.number_input(
        "Weight (kg)",
        min_value=10.0,
        max_value=300.0,
        value=60.0,
        step=0.1,
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
    )

    previous_inpatient_cost = st.number_input(
        "Previous inpatient medical cost",
        min_value=0.0,
        value=0.0,
        step=100.0,
    )

    chronic_illness_label = st.selectbox(
        "Have you been diagnosed with a chronic illness?",
        options=[
            "No",
            "Yes",
        ],
    )

    health_label = st.selectbox(
        "How would you rate your health?",
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
        "Are you currently employed?",
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
            "No insurance",
        ],
    )

    submitted = st.form_submit_button(
        "Predict medical cost",
        use_container_width=True,
    )
    if submitted:

    health_mapping = {
        "Excellent": 1,
        "Very good": 2,
        "Good": 3,
        "Fair": 4,
        "Poor": 5,
    }

    health_code = health_mapping[health_label]


chronic_mapping = {
    "No": 0,
    "Yes": 1,
}

chronic_code = chronic_mapping[chronic_illness_label]

employed_mapping = {
    "No": 0,
    "Yes": 1,
}

employed_code = employed_mapping[employed_label]

hospitalized_mapping = {
    "No": 0,
    "Yes": 1,
}

hospitalized_code = hospitalized_mapping[hospitalized_label]

