import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# PAGE CONFIG
st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")

# LOAD FILES
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# CUSTOM CSS
st.markdown("""
<style>

.stApp {
    background-color: #061b11;
    color: white;
}

h1 {
    text-align: center;
    color: #00ff88;
    font-size: 60px !important;
    text-shadow: 0px 0px 20px #00ff88;
}

.big-font {
    text-align: center;
    font-size:20px;
    color:white;
}

div.stButton > button {
    background-color: #00ff88;
    color: black;
    font-size: 20px;
    border-radius: 10px;
    width: 100%;
    height: 60px;
}

.result-box {
    padding: 30px;
    border-radius: 15px;
    text-align:center;
    font-size:30px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# TITLE
st.markdown("<h1>EMPLOYEE ATTRITION PREDICTION SYSTEM</h1>", unsafe_allow_html=True)
st.markdown("<p class='big-font'>Predict employee attrition using machine learning</p>", unsafe_allow_html=True)

st.write("")

# INPUTS
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 60, 30)
    income = st.number_input("Monthly Income", 1000, 100000, 10000)
    distance = st.slider("Distance From Home", 1, 30, 5)

with col2:
    travel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
    overtime = st.selectbox("Overtime", ["Yes", "No"])
    gender = st.selectbox("Gender", ["Male", "Female"])

st.write("")

# PREDICT BUTTON
if st.button("⚡ Predict Attrition"):

    # EMPTY INPUT DATAFRAME
    input_data = pd.DataFrame([[0]*len(feature_names)], columns=feature_names)

    # FILL VALUES
    if "Age" in input_data.columns:
        input_data["Age"] = age

    if "MonthlyIncome" in input_data.columns:
        input_data["MonthlyIncome"] = income

    if "DistanceFromHome" in input_data.columns:
        input_data["DistanceFromHome"] = distance

    # SCALE
    scaled_data = scaler.transform(input_data)

    # PREDICTION
    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0][1] * 100

    # RESULT
    if prediction == 1:
        st.markdown(f"""
        <div class="result-box" style="background-color:#2b0d0d;color:#ff4d6d;border:2px solid red;">
        ⚠ HIGH ATTRITION RISK<br><br>
        Employee may leave the company
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class="result-box" style="background-color:#0d2b16;color:#00ff88;border:2px solid #00ff88;">
        ✅ LOW ATTRITION RISK<br><br>
        Employee likely to stay
        </div>
        """, unsafe_allow_html=True)

    # GAUGE CHART
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability,
        number = {'suffix': "%"},
        title = {'text': "Attrition Probability"},
        gauge = {
            'axis': {'range': [0,100]},
            'bar': {'color': "#00ff88"},
            'steps': [
                {'range': [0,50], 'color': "#1b5e20"},
                {'range': [50,100], 'color': "#7f0000"}
            ]
        }
    ))

    fig.update_layout(
        paper_bgcolor="#061b11",
        font={'color': "white"}
    )

    st.plotly_chart(fig, use_container_width=True)