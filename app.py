import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load files
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

st.title("HR Attrition Prediction System")

age = st.slider("Age", 18, 60)
monthly_income = st.number_input("Monthly Income", min_value=1000)

if st.button("Predict"):

    input_data = pd.DataFrame([[0]*len(feature_names)], columns=feature_names)

    if "Age" in input_data.columns:
        input_data["Age"] = age

    if "MonthlyIncome" in input_data.columns:
        input_data["MonthlyIncome"] = monthly_income

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)

    if prediction[0] == 1:
        st.error("HIGH ATTRITION RISK")
    else:
        st.success("LOW ATTRITION RISK")