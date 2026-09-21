
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="AI Employee Performance Analytics",
    page_icon="📊",
    layout="centered"
)

st.title("📊 AI Employee Performance Analytics")
st.write("Predict employee performance using Machine Learning.")

st.caption(
    "Academic project using synthetic data. "
    "This demo is not intended for real employee evaluation."
)

model_path = Path("employee_performance_model.pkl")

if not model_path.exists():
    st.error("Model file not found in the repository.")
    st.stop()

model = joblib.load(model_path)

st.sidebar.header("Employee Details")

experience = st.sidebar.number_input(
    "Experience (Years)", 0.0, 50.0, 5.0, 1.0
)

training = st.sidebar.number_input(
    "Training Hours", 0.0, 500.0, 60.0, 5.0
)

attendance = st.sidebar.number_input(
    "Attendance Percentage", 0.0, 100.0, 90.0, 1.0
)

projects = st.sidebar.number_input(
    "Projects Completed", 0, 100, 8, 1
)

if st.button("Predict Performance"):

    input_data = pd.DataFrame({
        "Experience_Years": [experience],
        "Training_Hours": [training],
        "Attendance_Percentage": [attendance],
        "Projects_Completed": [projects]
    })

    prediction = float(model.predict(input_data)[0])

    prediction = max(0, min(100, prediction))

    st.success(
        f"Predicted Performance Score: {prediction:.2f}"
    )

    st.subheader("Input Details")
    st.dataframe(input_data)

          
           
