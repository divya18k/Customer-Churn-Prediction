import streamlit as st
import pandas as pd
import joblib

# Load model and feature columns
model = joblib.load("churn_model.pkl")
model_features = joblib.load("model_features.pkl")

# App layout
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("📉 Customer Churn Prediction Dashboard")

# Tabs for input types
tab1, tab2 = st.tabs(["🧍 Manual Input", "📁 File Upload"])

# --------------------------
# TAB 1: Manual Input
# --------------------------
with tab1:
    st.header("🔍 Predict Churn for a Single Customer")
    st.markdown("Fill in the customer details to predict churn probability.")

    def get_user_input():
        gender = st.selectbox("Gender", ['Male', 'Female'])
        SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
        Partner = st.selectbox("Has Partner?", ['Yes', 'No'])
        Dependents = st.selectbox("Has Dependents?", ['Yes', 'No'])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        PhoneService = st.selectbox("Phone Service", ['Yes', 'No'])
        MultipleLines = st.selectbox("Multiple Lines", ['Yes', 'No', 'No phone service'])
        InternetService = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
        OnlineSecurity = st.selectbox("Online Security", ['Yes', 'No', 'No internet service'])
        OnlineBackup = st.selectbox("Online Backup", ['Yes', 'No', 'No internet service'])
        DeviceProtection = st.selectbox("Device Protection", ['Yes', 'No', 'No internet service'])
        TechSupport = st.selectbox("Tech Support", ['Yes', 'No', 'No internet service'])
        StreamingTV = st.selectbox("Streaming TV", ['Yes', 'No', 'No internet service'])
        StreamingMovies = st.selectbox("Streaming Movies", ['Yes', 'No', 'No internet service'])
        Contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
        PaperlessBilling = st.selectbox("Paperless Billing", ['Yes', 'No'])
        PaymentMethod = st.selectbox("Payment Method", [
            'Electronic check', 'Mailed check', 
            'Bank transfer (automatic)', 'Credit card (automatic)'
        ])
        MonthlyCharges = st.slider("Monthly Charges", 0.0, 150.0, 50.0)
        TotalCharges = st.slider("Total Charges", 0.0, 10000.0, 500.0)

        data = {
            'gender': gender,
            'SeniorCitizen': SeniorCitizen,
            'Partner': Partner,
            'Dependents': Dependents,
            'tenure': tenure,
            'PhoneService': PhoneService,
            'MultipleLines': MultipleLines,
            'InternetService': InternetService,
            'OnlineSecurity': OnlineSecurity,
            'OnlineBackup': OnlineBackup,
            'DeviceProtection': DeviceProtection,
            'TechSupport': TechSupport,
            'StreamingTV': StreamingTV,
            'StreamingMovies': StreamingMovies,
            'Contract': Contract,
            'PaperlessBilling': PaperlessBilling,
            'PaymentMethod': PaymentMethod,
            'MonthlyCharges': MonthlyCharges,
            'TotalCharges': TotalCharges
        }

        return pd.DataFrame([data])

    user_data = get_user_input()
    encoded = pd.get_dummies(user_data)
    encoded = encoded.reindex(columns=model_features, fill_value=0)

    if st.button("📊 Predict Churn"):
        prob = model.predict_proba(encoded)[0][1]
        st.subheader(f"🧠 Churn Probability: `{prob:.2f}`")
        if prob > 0.4:
            st.error("⚠️ High Risk: Customer is likely to churn.")
        else:
            st.success("✅ Low Risk: Customer is not likely to churn.")

# --------------------------
# TAB 2: File Upload
# --------------------------
with tab2:
    st.header("📂 Predict Churn from File")
    st.markdown("Upload a CSV or Excel file with customer details.")

    uploaded_file = st.file_uploader("Upload Customer File", type=["csv", "xlsx"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)

        # Drop customerID if exists
        df.drop(columns=["customerID"], errors="ignore", inplace=True)

        st.success(f"✅ Loaded {len(df)} rows successfully.")
        st.dataframe(df.head())

        index = st.number_input("Select Row Index (0-based)", 0, len(df) - 1)
        selected_row = df.iloc[[index]]

        st.markdown("### 🔍 Selected Row Data")
        st.write(selected_row)

        # Prepare input
        input_encoded = pd.get_dummies(selected_row)
        input_encoded = input_encoded.reindex(columns=model_features, fill_value=0)

        if st.button("🔎 Predict Selected Row"):
            prob = model.predict_proba(input_encoded)[0][1]
            st.subheader(f"🧠 Churn Probability: `{prob:.2f}`")
            if prob > 0.4:
                st.error("⚠️ High Risk: Customer is likely to churn.")
            else:
                st.success("✅ Low Risk: Customer is not likely to churn.")
