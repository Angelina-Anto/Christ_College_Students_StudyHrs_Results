import streamlit as st
import joblib

model = joblib.load("logistic_regression_study_hours_model.pkl")
st.title("👒Student Pass/Fail based on Study Hours and Attendence")
hours = st.number_input("Enter Study Hours:", min_value = 0.0, max_value = 15.0, value = 5.0)
attendance = st.number_input("Enter Attendance:", min_value = 0.0, max_value = 100.0, value = 50.0)
if st.button("Predict"):
  prediction = model.predict([[hours,attendance]])
  probability = model.predict_proba([[hours,attendance]])
  fail_probability = probability[0][0]*100
  pass_probability = probability[0][1]*100
  if prediction[0] == 1:
    st.success("😎Pass")
    st.write("Pass probability: ", round(pass_probability, 2), "%")

  else:
    st.error("🥺Fail")
    st.write("Fail probability: ", round(fail_probability, 2), "%")
