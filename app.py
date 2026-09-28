import pandas as pd
import streamlit as st

from train_model import build_model

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")


@st.cache_resource
def load_model():
    # Trains once when the app starts (takes under a second), then is reused
    return build_model("heart.csv")


model, test_acc, test_f1 = load_model()

st.title("❤️ Heart Disease Risk Predictor")
st.caption(f"Logistic Regression trained on 918 patients · "
           f"test accuracy {test_acc:.1%} · F1 {test_f1:.2f}")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100, 50)
    sex = st.selectbox("Sex", ["M", "F"], format_func=lambda s: "Male" if s == "M" else "Female")
    chest_pain = st.selectbox(
        "Chest pain type",
        ["ASY", "NAP", "ATA", "TA"],
        format_func=lambda c: {
            "ASY": "ASY – Asymptomatic",
            "NAP": "NAP – Non-anginal pain",
            "ATA": "ATA – Atypical angina",
            "TA": "TA – Typical angina",
        }[c],
    )
    resting_bp = st.number_input("Resting blood pressure (mm Hg)", 80, 220, 130)
    cholesterol = st.number_input("Serum cholesterol (mg/dl)", 0, 650, 220,
                                  help="Enter 0 if unknown – the model fills in the average.")
    fasting_bs = st.selectbox("Fasting blood sugar > 120 mg/dl", [0, 1],
                              format_func=lambda v: "Yes" if v else "No")

with col2:
    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "ST", "LVH"],
        format_func=lambda r: {
            "Normal": "Normal",
            "ST": "ST – ST-T wave abnormality",
            "LVH": "LVH – Left ventricular hypertrophy",
        }[r],
    )
    max_hr = st.number_input("Max heart rate achieved", 60, 220, 140)
    exercise_angina = st.selectbox("Exercise-induced angina", ["N", "Y"],
                                   format_func=lambda v: "Yes" if v == "Y" else "No")
    oldpeak = st.number_input("Oldpeak (ST depression)", -3.0, 7.0, 1.0, step=0.1)
    st_slope = st.selectbox("ST slope", ["Up", "Flat", "Down"])

if st.button("Predict", type="primary", use_container_width=True):
    patient = pd.DataFrame([{
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG": resting_ecg,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope,
    }])

    prob = model.predict_proba(patient)[0, 1]

    st.metric("Estimated probability of heart disease", f"{prob:.0%}")
    st.progress(float(prob))
    if prob >= 0.5:
        st.error("⚠️ High risk of heart disease")
    else:
        st.success("✅ Low risk of heart disease")

st.divider()
st.caption("This is a machine-learning demo for educational purposes only. "
           "It is not a medical diagnosis — consult a doctor for any health concerns.")
