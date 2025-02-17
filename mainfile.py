import streamlit as st
import pandas as pd
from scripts import model_utils, chatbot_utils
import os
from streamlit_option_menu import option_menu
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
from streamlit_lottie import st_lottie
import requests
from fuzzywuzzy import process
from docx import Document
# Constants
DIABETES_FEATURE_NAMES = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']
CARDIO_FEATURE_NAMES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# File Paths
MODEL_DIR = "models"
KNOWLEDGE_BASE_FILE = os.path.join("knowledge_base", "diabetesKB.docx")

# Load Lottie animations
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load animation
lottie_animation = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_jk9zjiv4.json")

# Set page configuration
st.set_page_config(page_title="Multi-Disease Detection System with AI chatbot", layout="wide", page_icon="🧑‍⚕️")

# Sidebar for navigation
with st.sidebar:
    selected = option_menu(
        "Multi-Disease Prediction System",
        ["Diabetes Prediction", "Cardiovascular Disease Prediction", 'AI Chatbot', 'About'],
        icons=['activity', 'heart', 'chat', 'exclamation-circle'],
        menu_icon="hospital",
        default_index=0,
        styles={
            "container": {"padding": "5!important", "background-color": "#f0f2f6"},
            "icon": {"color": "black", "font-size": "25px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "green"},
        }
    )

# Function for Diabetes Prediction Page
if selected == "Diabetes Prediction":
    st.title("Diabetes Prediction")

    # Input fields for diabetes prediction
    gender = st.selectbox('Gender', ['Female', 'Male'])
    age = st.number_input('Age', min_value=0.0, max_value=120.0, value=50.0)
    hypertension = st.selectbox('Hypertension', [0, 1])
    heart_disease = st.selectbox('Heart Disease', [0, 1])
    smoking_history = st.selectbox('Smoking History', ['never', 'former', 'current', 'not current', 'ever', 'No Info'])
    bmi = st.number_input('BMI', min_value=0.0, max_value=100.0, value=25.0)
    hba1c_level = st.number_input('HbA1c Level', min_value=0.0, max_value=20.0, value=6.0)
    blood_glucose_level = st.number_input('Blood Glucose Level', min_value=0, max_value=500, value=150)

    # Prediction button
    if st.button('Predict Diabetes'):
        try:
            # Load the model and scaler
            diabetes_model, diabetes_scaler = model_utils.load_model_and_scaler(
                os.path.join(MODEL_DIR, "diabetes_model.pkl"),
                os.path.join(MODEL_DIR, "diabetes_scaler.pkl")
            )

            # Prepare input data
            input_data = [gender, age, hypertension, heart_disease, smoking_history, bmi, hba1c_level, blood_glucose_level]

            # Make prediction
            prediction = model_utils.predict_diabetes(input_data, diabetes_model, diabetes_scaler, DIABETES_FEATURE_NAMES)

            # Display result
            if prediction == 0:
                st.success('The person is not diabetic')
            else:
                st.error('The person is diabetic')

        except Exception as e:
            st.error(f"Error: {e}")

# Function for Cardiovascular Disease Prediction Page
elif selected == "Cardiovascular Disease Prediction":
    st.title("Cardiovascular Disease Prediction")

    # Input fields for cardiovascular disease prediction
    age = st.number_input('Age (Years)', min_value=1, max_value=120, value=50)
    sex = st.selectbox('Sex', [0, 1])  # 0 for female, 1 for male
    cp = st.selectbox('Chest Pain Type', [0, 1, 2, 3])
    trestbps = st.number_input('Resting Blood Pressure', min_value=50, max_value=300, value=120)
    chol = st.number_input('Cholesterol', min_value=50, max_value=600, value=200)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1])
    restecg = st.selectbox('Resting ECG', [0, 1, 2])
    thalach = st.number_input('Max Heart Rate Achieved', min_value=50, max_value=250, value=150)
    exang = st.selectbox('Exercise Induced Angina', [0, 1])
    oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=10.0, value=1.0)
    slope = st.selectbox('Slope of Peak Exercise ST Segment', [0, 1, 2])
    ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy', [0, 1, 2, 3, 4])
    thal = st.selectbox('Thal', [0, 1, 2, 3])

    # Prediction button
    if st.button('Predict Cardiovascular Disease'):
        try:
            # Load the model and scaler
            cardio_model, cardio_scaler = model_utils.load_model_and_scaler(
                os.path.join(MODEL_DIR, "cardio_model.pkl"),
                os.path.join(MODEL_DIR, "cardio_scaler.pkl")
            )

            # Prepare input data
            input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

            # Make prediction
            prediction = model_utils.predict_cardio(input_data, cardio_model, cardio_scaler, CARDIO_FEATURE_NAMES)

            # Display result
            if prediction == 0:
                st.success('The person is not at risk of cardiovascular disease')
            else:
                st.error('The person is at risk of cardiovascular disease')

        except Exception as e:
            st.error(f"Error: {e}")

# Function for AI Chatbot Page
elif selected == "AI Chatbot":
    st.title("AI Chatbot")
    st.write("Ask me anything about diabetes!")

    # Load knowledge base
    try:
        knowledge_base = chatbot_utils.load_knowledge_base(KNOWLEDGE_BASE_FILE)
    except Exception as e:
        st.error(f"Error loading knowledge base: {e}")
        knowledge_base = None

    # Chatbot interface
    user_query = st.text_input("You:", "")
    if user_query:
        try:
            response = chatbot_utils.get_chatbot_response(user_query, knowledge_base)
            st.text_area("Chatbot:", value=response, height=200)
        except Exception as e:
            st.error(f"Chatbot error: {e}")

# Function for About Page
elif selected == "About":
    st.title("About Multi-Disease Prediction System")
    st.write("This system is designed to predict diabetes and cardiovascular disease based on user inputs...")

