import streamlit as st
import joblib
import numpy as np
import pandas as pd

# -------------------------
# Load model + features
# -------------------------
model = joblib.load("models/logistic_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")
scaler = joblib.load("models/scaler.pkl")

# -------------------------
# Page setup
# -------------------------
st.set_page_config(page_title="Churn Predictor", page_icon="📊", layout="centered")

st.title("📊 Customer Churn Prediction System")
st.write("AI model predicts whether a customer will churn or not.")

st.divider()

# -------------------------
# USER INPUT
# -------------------------
tenure = st.slider("📆 Tenure (Months)", 0, 72, 12)
monthly = st.number_input("💰 Monthly Charges", 0.0, 200.0, 50.0)

contract = st.selectbox("📄 Contract Type", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("🌐 Internet Service", ["DSL", "Fiber optic", "No"])

st.divider()

# -------------------------
# PREDICTION
# -------------------------
if st.button("🚀 Predict Churn", use_container_width=True):

    # Create full feature set (IMPORTANT FIX)
    input_dict = {col: 0 for col in feature_columns}

    # numeric features
    if "tenure" in input_dict:
        input_dict["tenure"] = tenure

    if "MonthlyCharges" in input_dict:
        input_dict["MonthlyCharges"] = monthly

    # contract mapping (SAFE)
    if contract == "Month-to-month" and "Contract_Month-to-month" in input_dict:
        input_dict["Contract_Month-to-month"] = 1

    if contract == "One year" and "Contract_One year" in input_dict:
        input_dict["Contract_One year"] = 1

    if contract == "Two year" and "Contract_Two year" in input_dict:
        input_dict["Contract_Two year"] = 1

    # internet mapping (SAFE)
    if internet == "Fiber optic" and "InternetService_Fiber optic" in input_dict:
        input_dict["InternetService_Fiber optic"] = 1

    if internet == "No" and "InternetService_No" in input_dict:
        input_dict["InternetService_No"] = 1

    if internet == "DSL" and "InternetService_DSL" in input_dict:
        input_dict["InternetService_DSL"] = 1

    # convert to dataframe
    import pandas as pd

    # create dataframe from input_dict to include all mapped features
    input_df = pd.DataFrame([input_dict], columns=feature_columns)

    # -------------------------
    # MODEL PREDICTION
    # -------------------------
    input_df_scaled = scaler.transform(input_df)
    prediction = model.predict(input_df_scaled)

    # probability (IMPORTANT FIX)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df_scaled)[0][1]
    else:
        proba = 0.0

    st.divider()

    st.subheader("📌 Prediction Result")

    # -------------------------
    # RISK LOGIC
    # -------------------------
    if proba >= 0.70:
        st.error(f"⚠ HIGH RISK: Customer WILL CHURN")
        st.write(f"Churn Probability: {proba:.2f}")

    elif proba >= 0.40:
        st.warning(f"⚠ MEDIUM RISK: Customer may churn")
        st.write(f"Churn Probability: {proba:.2f}")

    else:
        st.success(f"✅ LOW RISK: Customer will NOT churn")
        st.write(f"Churn Probability: {proba:.2f}")

    # probability bar
    st.progress(float(proba))