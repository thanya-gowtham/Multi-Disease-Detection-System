import streamlit as st
import pandas as pd
import numpy as np
from scripts import model_utils, chatbot_utils
import os
from streamlit_option_menu import option_menu
import io
from streamlit_lottie import st_lottie
import requests
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# Constants
DIABETES_FEATURES = ['gender', 'age', 'hypertension', 'heart_disease',
                    'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']
CARDIO_FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                  'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# Path Configuration
MODEL_DIR = "models"
DATA_DIR = "data"
KNOWLEDGE_BASE = os.path.join("knowledge_base", "medical_knowledge.docx")

# Lottie Animation Loader
def load_lottie(url):
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None

lottie_health = load_lottie("https://assets8.lottiefiles.com/packages/lf20_5njp3vgg.json")

# Page Configuration
st.set_page_config(
    page_title="HealthGuard AI - Multi-Disease Prediction System",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS Styling
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');

    body {{
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: #ffffff;
    }}

    .stApp {{
        background-color: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 2rem;
    }}

    .stButton>button {{
        background: linear-gradient(45deg, #00b4d8, #0077b6);
        border-radius: 25px;
        border: none;
        color: white;
        padding: 0.8rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }}

    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 180, 216, 0.3);
    }}

    .stTextInput>div>div>input, .stNumberInput>div>div>input {{
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid #00b4d8;
        color: Black;
        border-radius: 10px;
        padding: 0.5rem 1rem;
    }}

    .stSelectbox>div>div>select {{
        background: rgba(255, 255, 255, 0.1);
        color: white;
        border-radius: 10px;
        padding: 0.5rem 1rem;
    }}

    h1, h2, h3 {{
        color: #00b4d8 !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }}
</style>
""", unsafe_allow_html=True)

# PDF Report Generator
def generate_medical_report(title, patient_name, inputs, prediction):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title Section
    story.append(Paragraph(f"<font color='#00b4d8'>{title}</font>", styles['Title']))
    story.append(Spacer(1, 20))

    # Patient Information
    story.append(Paragraph(f"<b>Patient Name:</b> {patient_name}", styles['Normal']))
    story.append(Spacer(1, 15))

    # Input Parameters
    story.append(Paragraph("<b>Health Parameters:</b>", styles['Heading3']))
    if isinstance(inputs, dict):
        for param, value in inputs.items():
            story.append(Paragraph(f"{param}: {value}", styles['Normal']))
    else:
        story.append(Paragraph(f"Error: Invalid input data format", styles['Normal']))
    story.append(Spacer(1, 20))

    # Prediction Result
    story.append(Paragraph("<b>Diagnostic Prediction:</b>", styles['Heading3']))
    story.append(Paragraph(prediction, styles['Normal']))

    doc.build(story)
    buffer.seek(0)
    return buffer

# Navigation Sidebar
with st.sidebar:
    selected = option_menu(
        menu_title="Multi-Disease Detection",
        options=["Diabetes Risk", "Heart Health", "Medical Chatbot", "About"],
        icons=["activity", "heart-pulse", "robot", "info-circle"],
        menu_icon="heart-pulse",
        default_index=0,
        styles={
            "container": {"padding": "10px", "background-color": "rgba(255,255,255,0.05)"},
            "icon": {"color": "#00b4d8", "font-size": "20px"},
            "nav-link": {"color": "black", "font-size": "16px", "margin": "8px 0"},
            "nav-link-selected": {"background-color": "#00b4d8", "font-weight": "bold"}
        }
    )

# Diabetes Prediction Section
if selected == "Diabetes Risk":
    st.title("🔍 Diabetes Risk Assessment")
    st_lottie(lottie_health, height=200, key="diabetes")

    with st.form("diabetes_form"):
        cols = st.columns(2)

        with cols[0]:
            name = st.text_input("Patient Name")
            age = st.number_input("Age", 1, 120, 25)
            gender = st.selectbox("Gender", ["Female", "Male"])
            hypertension = st.selectbox("Hypertension History", ["No", "Yes"])
            heart_disease = st.selectbox("Heart Disease History", ["No", "Yes"])

        with cols[1]:
            smoking = st.selectbox("Smoking Status",
                                ["Never", "Former", "Current", "Not Current", "Ever", "Unknown"])
            bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
            hba1c = st.number_input("HbA1c Level", 3.0, 15.0, 5.7)
            glucose = st.number_input("Blood Glucose Level", 50, 300, 100)

        if st.form_submit_button("Analyze Diabetes Risk"):
            try:
                # Convert categorical inputs
                gender_num = 1 if gender == "Male" else 0
                hypertension_num = 1 if hypertension == "Yes" else 0
                heart_disease_num = 1 if heart_disease == "Yes" else 0
                smoke_mapping = {"Never":0, "Former":1, "Current":2,
                                "Not Current":3, "Ever":4, "Unknown":5}
                smoking_num = smoke_mapping[smoking]

                # Prepare input array
                input_data = [gender_num, age, hypertension_num, heart_disease_num,
                            smoking_num, bmi, hba1c, glucose]

                # Load model and predict
                model, scaler = model_utils.load_model_and_scaler(
                    os.path.join(MODEL_DIR, "diabetes_model.pkl"),
                    os.path.join(MODEL_DIR, "diabetes_scaler.pkl")
                )
                prediction = model_utils.predict_diabetes(input_data, model, scaler, DIABETES_FEATURES)

                # Display results
                if prediction == 1:
                    st.error("🚨 High Diabetes Risk Detected")
                    st.markdown("""
                    **Recommendations:**
                    - Consult an endocrinologist immediately
                    - Monitor blood sugar levels regularly
                    - Adopt low-glycemic diet
                    """)
                else:
                    st.success("✅ No Diabetes Risk Detected")

                # Generate report
                report_data = {
                    "Gender": gender,
                    "Age": age,
                    "Hypertension": hypertension,
                    "Heart Disease": heart_disease,
                    "Smoking Status": smoking,
                    "BMI": bmi,
                    "HbA1c Level": hba1c,
                    "Blood Glucose": glucose
                }
                pdf = generate_medical_report("Diabetes Risk Report", name, report_data,
                                            "High Risk" if prediction == 1 else "Low Risk")
                st.session_state["diabetes_report_pdf"] = pdf
            except Exception as e:
                st.error(f"Error in prediction: {str(e)}")
                error_report_data = {"Error": str(e)}  # Create a dictionary for the error
                pdf = generate_medical_report("Diabetes Risk Report", name, error_report_data, "Error")
                st.session_state["diabetes_report_pdf"] = pdf

    if "diabetes_report_pdf" in st.session_state:
        pdf = st.session_state["diabetes_report_pdf"]
        st.download_button(
            label="📄 Download Full Report",
            data=pdf,
            file_name="diabetes_report.pdf",
            mime="application/pdf",
            key="diabetes_download_button"
        )

# Cardiovascular Prediction Section
elif selected == "Heart Health":
    st.title("❤️ Cardiovascular Health Analysis")
    st_lottie(lottie_health, height=200, key="cardio")

    with st.form("cardio_form"):
        cols = st.columns(2)

        with cols[0]:
            name = st.text_input("Patient Name")
            age = st.number_input("Age", 1, 120, 45)
            gender = st.selectbox("Gender", ["Female", "Male"])
            bp = st.number_input("Resting BP (mmHg)", 50, 250, 120)
            cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
            fbs = st.selectbox("Fasting Sugar >120mg/dL", ["No", "Yes"])

        with cols[1]:
            ecg_options = ["Normal", "ST Abnormality", "LV Hypertrophy"]
            ecg = st.selectbox("Resting ECG", ecg_options)
            angina = st.selectbox("Exercise Angina", ["No", "Mild", "Severe"])
            depression = st.number_input("ST Depression", 0.0, 5.0, 1.0)
            slope_options = ["Upsloping", "Flat", "Downsloping"]
            slope = st.selectbox("ST Slope", slope_options)
            vessels = st.selectbox("Fluoroscopy Vessels", [0,1,2,3,4])
            thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])

        if st.form_submit_button("Analyze Heart Health"):
            try:
                # Convert medical parameters
                gender_num = 1 if gender == "Male" else 0
                fbs_num = 1 if fbs == "Yes" else 0
                ecg_num = ecg_options.index(ecg)
                angina_num = ["No", "Mild", "Severe"].index(angina)
                slope_num = slope_options.index(slope)
                thal_num = ["Normal", "Fixed Defect", "Reversible Defect"].index(thal)

                # Prepare input array
                input_data = [age, gender_num, 0, bp, cholesterol, fbs_num,
                             ecg_num, 150, angina_num, depression, slope_num, vessels, thal_num]

                # Load model and predict
                model, scaler = model_utils.load_model_and_scaler(
                    os.path.join(MODEL_DIR, "cardio_model.pkl"),
                    os.path.join(MODEL_DIR, "cardio_scaler.pkl")
                )
                prediction = model_utils.predict_cardio(input_data, model, scaler, CARDIO_FEATURES)

                # Display results
                if prediction == 1:
                    st.error("🚨 High Cardiovascular Risk Detected")
                    st.markdown("""
                    **Recommendations:**
                    - Consult a cardiologist immediately
                    - Start heart-healthy diet
                    - Regular cardiac monitoring
                    """)
                else:
                    st.success("✅ Healthy Cardiovascular Profile")

                # Generate report
                report_data = {
                    "Age": age,
                    "Gender": gender,
                    "Blood Pressure": f"{bp} mmHg",
                    "Cholesterol": f"{cholesterol} mg/dL",
                    "Fasting Sugar": fbs,
                    "ECG Findings": ecg,
                    "Exercise Angina": angina,
                    "ST Depression": depression,
                    "ST Slope": slope,
                    "Thalassemia": thal
                }
                pdf = generate_medical_report("Cardiac Health Report", name, report_data,
                                            "High Risk" if prediction==1 else "Low Risk")
                st.session_state["cardio_report_pdf"] = pdf

            except Exception as e:
                st.error(f"Error in prediction: {str(e)}")
                error_report_data = {"Error": str(e)}  # Create a dictionary for the error
                pdf = generate_medical_report("Cardiac Health Report", name, error_report_data, "Error")
                st.session_state["cardio_report_pdf"] = pdf

    if "cardio_report_pdf" in st.session_state:
        pdf = st.session_state["cardio_report_pdf"]
        st.download_button(
            label="📄 Download Full Report",
            data=pdf,
            file_name="cardiac_report.pdf",
            mime="application/pdf",
            key="cardio_download_button"
        )

# Chatbot Section
# Chatbot Section
elif selected == "Medical Chatbot":
    st.title("🤖 HealthGuard Assistant")
    st.write("Ask me anything about health concerns or disease prevention!")

    # Load knowledge base (load it once)
    if 'knowledge_base' not in st.session_state:
        st.session_state['knowledge_base'] = chatbot_utils.load_knowledge_base(KNOWLEDGE_BASE)

    knowledge_base = st.session_state['knowledge_base']

    # Chatbot implementation
    user_query = st.chat_input("Type your health query...")
    if user_query:
        try:
            if isinstance(knowledge_base, dict):
                response = chatbot_utils.get_chatbot_response(user_query, knowledge_base) # Corrected function name
                with st.chat_message("user"):
                    st.write(user_query)
                with st.chat_message("assistant"):
                    st.write(response)
            else:
                st.error("Failed to load the knowledge base. Please check the file path and format.")
        except Exception as e:
            st.error(f"Chatbot error: {str(e)}")


# About Section
elif selected == "About":
    st.title("About Multi-Disease Detection")
    st.markdown("""
    ## Comprehensive Health Prediction System

    **Multi-Disease Detection** is an advanced diagnostic platform combining machine learning
    with medical expertise to provide:

    - Early disease risk detection
    - Multi-disease prediction capabilities
    - AI-powered health recommendations
    - Comprehensive medical reports

    ### Key Features:
    🔹 Diabetes & Cardiovascular Risk Assessment
    🔹 Symptom Analysis & Prevention Guidelines
    🔹 Interactive Health Monitoring
    🔹 Medical Knowledge Base Integration

    *DEVELOPED FOR SET PROJECT 2025 , VIT VELLORE*
    """)

    st_lottie(lottie_health, height=300, key="about")
