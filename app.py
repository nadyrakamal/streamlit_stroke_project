import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load model and scaler
with open("final_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ---------- Page Config ----------
st.set_page_config(
    page_title="Stroke Risk Prediction",
    page_icon="🧠",
    layout="wide"
)

# ---------- Background + Styling ----------
def set_bg(url: str):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(0,0,0,0.65), rgba(0,0,0,0.65)),
                        url("{url}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
        }}
        .card {{
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 16px;
            padding: 18px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Pick ONE background (you can change later)
set_bg("https://images.unsplash.com/photo-1559757175-5700dde675bc?auto=format&fit=crop&w=1600&q=80")

# ---------- Header ----------
st.markdown(
    """
    <div class="card">
        <h1 style="margin-bottom: 6px;">🧠 Stroke Risk Prediction System</h1>
        <p style="margin-top: 0;">
            Predict stroke risk using patient health indicators to support early screening and preventive healthcare.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

# ---------- Feature columns (must match training) ----------
FEATURE_COLUMNS = [
    'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
    'gender_Male', 'gender_Other',
    'ever_married_Yes',
    'work_type_Never_worked', 'work_type_Private',
    'work_type_Self-employed', 'work_type_children',
    'Residence_type_Urban',
    'smoking_status_formerly smoked',
    'smoking_status_never smoked',
    'smoking_status_smokes'
]

# ---------- Sidebar Inputs ----------
st.sidebar.header("Patient Inputs")

age = st.sidebar.slider("Age", 0, 100, 50)
avg_glucose = st.sidebar.number_input("Average Glucose Level", 50.0, 300.0, 100.0, step=1.0)
bmi = st.sidebar.number_input("BMI", 10.0, 60.0, 25.0, step=0.1)

gender = st.sidebar.selectbox("Gender", ["Female", "Male", "Other"])
hypertension = st.sidebar.selectbox("Hypertension", ["No", "Yes"])
heart_disease = st.sidebar.selectbox("Heart Disease", ["No", "Yes"])
ever_married = st.sidebar.selectbox("Ever Married", ["No", "Yes"])
work_type = st.sidebar.selectbox("Work Type", ["Private", "Self-employed", "Never worked", "children"])
residence = st.sidebar.selectbox("Residence Type", ["Urban", "Rural"])
smoking = st.sidebar.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes"])

predict_btn = st.sidebar.button("Predict Stroke Risk ✅", use_container_width=True)

# ---------- Build input row (OHE must match training) ----------
input_dict = {
    'age': age,
    'hypertension': 1 if hypertension == "Yes" else 0,
    'heart_disease': 1 if heart_disease == "Yes" else 0,
    'avg_glucose_level': avg_glucose,
    'bmi': bmi,

    'gender_Male': 1 if gender == "Male" else 0,
    'gender_Other': 1 if gender == "Other" else 0,

    'ever_married_Yes': 1 if ever_married == "Yes" else 0,

    'work_type_Never_worked': 1 if work_type == "Never worked" else 0,
    'work_type_Private': 1 if work_type == "Private" else 0,
    'work_type_Self-employed': 1 if work_type == "Self-employed" else 0,
    'work_type_children': 1 if work_type == "children" else 0,

    'Residence_type_Urban': 1 if residence == "Urban" else 0,

    'smoking_status_formerly smoked': 1 if smoking == "formerly smoked" else 0,
    'smoking_status_never smoked': 1 if smoking == "never smoked" else 0,
    'smoking_status_smokes': 1 if smoking == "smokes" else 0
}

input_df = pd.DataFrame([input_dict])[FEATURE_COLUMNS]
input_scaled = scaler.transform(input_df)

# ---------- Main Layout ----------
col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Current Input Summary")
    st.dataframe(input_df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Prediction Result")

    if predict_btn:
        pred = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0][1]
        pct = int(round(proba * 100))

        st.write(f"**Predicted probability of stroke:** **{proba:.2f}** ({pct}%)")
        st.progress(pct)

        if pred == 1:
            st.error("⚠️ High Risk of Stroke")
            st.write("**Suggested action:** Encourage medical screening and follow-up.")
        else:
            st.success("✅ Low Risk of Stroke")
            st.write("**Suggested action:** Maintain healthy lifestyle and routine check-ups.")

        st.caption("Note: This tool supports screening, not medical diagnosis.")
    else:
        st.info("Use the sidebar to enter values, then click **Predict Stroke Risk ✅**.")

    st.markdown("</div>", unsafe_allow_html=True)