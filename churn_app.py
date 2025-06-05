import streamlit as st
import pandas as pd
import joblib
# import numpy as np

# load model & encoder
model = joblib.load("churn_model.pkl")
# encoder = joblib.load("encoder.pkl")

st.title("churn prediction app")

# input form
gender = st.selectbox("Gender", ['Female', 'Male'])  
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ['Yes', 'No'])
dependents = st.selectbox("Dependents", ['Yes', 'No'])
tenure = st.slider("Tenture (months)", 0, 72, 12)
internet = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
paperless_billing = st.selectbox('PaperlessBilling', ['Yes', 'No'])  
phone_service = st.selectbox("Phone Service", ['Yes', 'No'])
total_charges = st.number_input("Total Charges", 0.0, 200.0, 50.0)


if st.button("Test With High-Risk Case"):
    input_data = pd.DataFrame({
        'gender': [1],  # Male
        'SeniorCitizen': [1],
        'Partner': [0],
        'Dependents': [0],
        'tenure': [1],
        'PhoneService': [0],
        'PaperlessBilling': [1],
        'MonthlyCharges': [95.0],
        'TotalCharges': [100.0]
    })
    st.write(input_data)
    result = model.predict(input_data)
    st.write("Raw Prediction:", result)
    if result[0] == 1:
        st.error(" This customer is likely to **churn**.")
    else:
        st.success(" This customer is **not likely** to churn.")
