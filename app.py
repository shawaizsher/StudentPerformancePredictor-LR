import streamlit as st
import numpy as np
import pickle

# Load the trained model and scaler
with open('logistic_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.set_page_config(page_title="Test Prep Predictor", layout="centered")
st.title("Student Test Preparation Predictor")

st.markdown("Predict whether a student has completed the test preparation course based on their background and scores.")

# Input form
with st.form("prediction_form"):
    gender = st.selectbox("Gender", ["Female", "Male"])
    lunch = st.selectbox("Lunch Type", ["Standard", "Free/Reduced"])
    education = st.selectbox("Parental Education Level", [
        "Some High School", "High School", "Some College", "Associate's Degree", "Bachelor's Degree", "Master's Degree"
    ])
    
    math_score = st.slider("Math Score", 0, 100, 70)
    reading_score = st.slider("Reading Score", 0, 100, 70)
    writing_score = st.slider("Writing Score", 0, 100, 70)

    race = st.selectbox("Race/Ethnicity", ["Group A", "Group B", "Group C", "Group D", "Group E"])
    
    submit = st.form_submit_button("Predict")

# Encode input
if submit:
    gender = 0 if gender == "Female" else 1
    lunch = 0 if lunch == "Standard" else 1
    edu_map = {
        "Some High School": 0,
        "High School": 1,
        "Some College": 2,
        "Associate's Degree": 3,
        "Bachelor's Degree": 4,
        "Master's Degree": 5
    }
    education = edu_map[education]
    
    race_encoded = [0]*5
    race_index = {"Group A": 0, "Group B": 1, "Group C": 2, "Group D": 3, "Group E": 4}[race]
    race_encoded[race_index] = 1

    input_data = [gender, education, lunch, math_score, reading_score, writing_score] + race_encoded
    input_array = np.array([input_data])
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0][prediction]

    if prediction == 1:
        st.success(f"✅ The student is **likely to have completed** the test preparation course. (Confidence: {prediction_proba:.2f})")
    else:
        st.error(f"❌ The student is **unlikely to have completed** the test preparation course. (Confidence: {prediction_proba:.2f})")
